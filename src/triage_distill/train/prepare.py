"""Render gold-gated targets into student training data (model-agnostic `messages`).

Reads `data/label/targets/{recipe_a,recipe_b,ablation}.jsonl` and emits, per recipe,
a chat `messages` JSONL that the 4090 training script feeds through the student
tokenizer's chat template. Also builds `val_eval.jsonl` from the val split (ticket +
gold, no teacher) for the eval harness to score each trained student on.

Deliberately model-agnostic: a `messages` list ([{role, content}]) carries no
tokenizer assumptions, so switching students (Qwen / Llama / Phi) is just applying a
different chat template on the training box — no re-prep, no re-labeling.

    uv run python -m triage_distill.train.prepare

--- YOUR KNOB -------------------------------------------------------------------
The three system prompts below are the *student's* inference contract (kept lean —
the task is baked into the weights, unlike the teacher's 27-gloss prompt). Tune them,
or say the word and I'll promote them to a `prompts/student.md` design surface like
`prompts/teacher.md`. Whatever you set here MUST match what the eval harness sends at
inference — the eval scorer imports these same constants so they stay in lockstep.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETS = REPO_ROOT / "data" / "label" / "targets"
VAL = REPO_ROOT / "data" / "splits" / "val.parquet"
OUT_DIR = REPO_ROOT / "data" / "train"

# Student inference contract (your knob — keep eval + training in sync via these).
SYS_CLASSIFY = (
    "You are a support-ticket triage classifier. Read the ticket and respond with a "
    'JSON object: {"category": "<intent label>"}.'
)
SYS_EXPLAIN = (
    "You are a support-ticket triage analyst. Read the ticket and respond with a JSON "
    'object: {"evidence_to_intent": "<...>", "why_not_alternatives": "<...>"}.'
)
SYS_REASON = (
    "You are a support-ticket triage classifier. Reason briefly, then commit to one "
    'intent label. Respond with a JSON object, reasoning fields first: '
    '{"evidence_to_intent": "<...>", "why_not_alternatives": "<...>", "category": "<intent label>"}.'
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python -m triage_distill.label.targets` first.")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _msg(system: str, user: str, target: dict) -> dict:
    """One chat example: system + user ticket + assistant JSON answer."""
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f'Ticket: "{user}"'},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}

    # Recipe A — multi-task: the task's system prompt selects classify vs explain.
    a_rows = []
    for r in _load_jsonl(TARGETS / "recipe_a.jsonl"):
        sys = SYS_CLASSIFY if r["task"] == "classify" else SYS_EXPLAIN
        a_rows.append(_msg(sys, r["input"], r["target"]))
    _write(OUT_DIR / "recipe_a.messages.jsonl", a_rows)
    counts["recipe_a"] = len(a_rows)

    # Recipe B — single sequence: reason then label.
    b_rows = [_msg(SYS_REASON, r["input"], r["target"]) for r in _load_jsonl(TARGETS / "recipe_b.jsonl")]
    _write(OUT_DIR / "recipe_b.messages.jsonl", b_rows)
    counts["recipe_b"] = len(b_rows)

    # Ablation — label only (control).
    ab_rows = [_msg(SYS_CLASSIFY, r["input"], r["target"]) for r in _load_jsonl(TARGETS / "ablation.jsonl")]
    _write(OUT_DIR / "ablation.messages.jsonl", ab_rows)
    counts["ablation"] = len(ab_rows)

    # Val eval set — ticket + gold only (the student must produce the category itself).
    val = pd.read_parquet(VAL).reset_index(drop=True)
    val_rows = [{"id": i, "text": t, "gold": g} for i, (t, g) in enumerate(zip(val["text"], val["label"]))]
    _write(OUT_DIR / "val_eval.jsonl", val_rows)
    counts["val_eval"] = len(val_rows)

    print(json.dumps(counts, indent=2))
    print(f"-> {OUT_DIR.relative_to(REPO_ROOT)}/ (*.messages.jsonl for training + val_eval.jsonl for scoring)")


if __name__ == "__main__":
    main()
