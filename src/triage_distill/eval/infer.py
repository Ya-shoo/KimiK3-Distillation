"""Inference runner: trained adapter -> preds.jsonl for the scorer. Pure plumbing.

Runs every ticket in an eval file (default `data/train/val_eval.jsonl`) through the
student with SCHEMA-GUARANTEED OUTPUT (HANDOFF Section 7: the scorer counts invalid output
as wrong, so this isn't optional). classify mode decodes fully constrained (the enum
constraint is cheap); reason mode generates free-form first and constrained-re-decodes
only the rows that fail schema validation - LMFE's per-token constraint is ~100x
slower inside free-text rationale fields, and the trained model virtually never
needs the net. Either way `category` ends up in the frozen 27-label enum or null.

The system prompt is imported from `train.prepare` (SYS_CLASSIFY / SYS_REASON), so
training and inference can never drift apart.

    # ablation / recipe A (label-only at inference - the fast path):
    uv run --no-sync python -m triage_distill.eval.infer --adapter runs/ablation/adapter --mode classify --out runs/ablation/preds.jsonl
    # recipe B (reason-then-label):
    uv run --no-sync python -m triage_distill.eval.infer --adapter runs/recipe_b/adapter --mode reason --out runs/recipe_b/preds.jsonl
    # zero-shot base-model baseline (no adapter):
    uv run --no-sync python -m triage_distill.eval.infer --mode classify --out runs/base_zeroshot/preds.jsonl

Then: uv run --no-sync python -m triage_distill.eval.score --preds <out> --gold data/train/val_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from triage_distill.datasets import cfg
from triage_distill.schema import load_label_space
from triage_distill.train.prepare import SYS_CLASSIFY, SYS_REASON

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = "Qwen/Qwen3-4B"

# mode -> (system prompt, max_new_tokens headroom)
MODES = {
    "classify": (SYS_CLASSIFY, 32),
    "reason": (SYS_REASON, 512),
}


def _schema(mode: str, labels: list[str]) -> dict:
    """Per-mode output schema; the category enum comes from the dataset's frozen label space."""
    enum = {"type": "string", "enum": labels}
    if mode == "classify":
        return {"type": "object", "properties": {"category": enum},
                "required": ["category"], "additionalProperties": False}
    return {  # reason: same field order the student was trained on - reasoning first
        "type": "object",
        "properties": {
            "evidence_to_intent": {"type": "string"},
            "why_not_alternatives": {"type": "string"},
            "category": enum,
        },
        "required": ["evidence_to_intent", "why_not_alternatives", "category"],
        "additionalProperties": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="runs/<name>/adapter (omit = zero-shot base model)")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--mode", choices=MODES, required=True,
                    help="classify = SYS_CLASSIFY (ablation + recipe A); reason = SYS_REASON (recipe B)")
    ap.add_argument("--dataset", default="bitext", help="picks the frozen label space + default eval data")
    ap.add_argument("--data", default=None, help="eval JSONL (default: the dataset's val_eval.jsonl)")
    ap.add_argument("--out", required=True, help="preds.jsonl path ({id, pred, raw})")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke tests)")
    args = ap.parse_args()

    dcfg = cfg(args.dataset)
    labels = list(load_label_space(dcfg.label_space))
    if args.data is None:
        args.data = str(dcfg.train_dir / "val_eval.jsonl")
    data_path = (REPO_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    system_prompt, max_new = MODES[args.mode]

    from unsloth import FastLanguageModel

    # lm-format-enforcer 0.11.x imports PreTrainedTokenizerBase from a module path
    # that transformers 5.x removed - restore the alias before importing the integration.
    import transformers
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "PreTrainedTokenizerBase"):
        _tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    model_name = args.adapter or args.base
    print(f"loading {model_name} ({'adapter' if args.adapter else 'base, zero-shot'})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # silence per-batch generation-config warnings

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(_schema(args.mode, labels)))
    label_set = set(labels)

    def render(text: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Ticket: "{text}"'},  # must match prepare._msg
        ]
        try:  # Qwen3: suppress native thinking - the JSON is the whole contract
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # tokenizer without the kwarg (other student families)
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out_path = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(batch: list[dict], constrained: bool) -> list[str]:
        enc = tokenizer([render(r["text"]) for r in batch], return_tensors="pt",
                        padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_fn if constrained else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        return [tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in gen[:, enc.input_ids.shape[1]:]]

    def parse(raw: str) -> str | None:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        pred = obj.get("category") if isinstance(obj, dict) else None
        return pred if pred in label_set else None

    # Speed vs guarantee: LMFE's per-token constraint is cheap for the classify enum
    # but ~100x slower inside free-text rationale fields (nearly the whole vocab is
    # legal each step). So reason mode generates UNCONSTRAINED first - the format is
    # burned in by training - and only rows failing schema validation are re-decoded
    # under the full constraint. Net guarantee is unchanged: every pred is in-enum,
    # or null and counted wrong by the scorer.
    constrain_first_pass = args.mode == "classify"

    results: dict[int, dict] = {}
    retry: list[dict] = []
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        for r, raw in zip(batch, generate(batch, constrained=constrain_first_pass)):
            pred = parse(raw)
            if pred is None and not constrain_first_pass:
                retry.append(r)
            else:
                results[r["id"]] = {"id": r["id"], "pred": pred, "raw": raw}
        done = min(start + args.batch_size, len(rows))
        print(f"\r{done}/{len(rows)}  ({done / (time.time() - t0):.1f} tickets/s)",
              end="", flush=True)

    if retry:
        print(f"\n{len(retry)} rows failed schema validation - re-decoding constrained")
        for start in range(0, len(retry), 4):  # small batches: constrained is slow
            batch = retry[start : start + 4]
            for r, raw in zip(batch, generate(batch, constrained=True)):
                results[r["id"]] = {"id": r["id"], "pred": parse(raw), "raw": raw,
                                    "constrained_retry": True}
            print(f"\rretry {min(start + 4, len(retry))}/{len(retry)}", end="", flush=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(results[r["id"]], ensure_ascii=False) + "\n")

    secs = time.time() - t0
    n_ok = sum(1 for v in results.values() if v["pred"] is not None)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}  |  valid {n_ok}/{len(rows)}  |  "
          f"retries {len(retry)}  |  {secs / 60:.1f} min ({len(rows) / secs:.1f} tickets/s)")
    print(f"score it: uv run --no-sync python -m triage_distill.eval.score "
          f"--preds {out_path.relative_to(REPO_ROOT)} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
