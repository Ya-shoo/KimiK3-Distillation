"""QLoRA SFT for the M2 student runs - OWNER'S SURFACE.

Trains Qwen3-4B (4-bit + LoRA, via Unsloth) on ONE of the three `messages` files, so
the only variable across runs is the rationale signal:

    uv run --no-sync python -m triage_distill.train.train --data data/train/ablation.messages.jsonl --name ablation
    uv run --no-sync python -m triage_distill.train.train --data data/train/recipe_b.messages.jsonl --name recipe_b
    uv run --no-sync python -m triage_distill.train.train --data data/train/recipe_a.messages.jsonl --name recipe_a

The hyperparameters in `KNOBS` below are deliberately UNSET - they're the owner's
decisions (HANDOFF-M2-4090 Section 3 has the coaching ranges). The script fails fast and
lists what's missing. Everything around the knobs - data loading, chat-template
rendering, loss-mask verification peek, checkpointing, run metadata, loss curve -
is plumbing and already wired.

Start with `--dry-run` (no knobs needed): renders the data through the tokenizer's
chat template and prints token-length percentiles - that's the input to your
`max_seq_len` decision - plus one fully-rendered example so you can see exactly
what the model trains on (Qwen3 chat template quirks included).

Outputs per run under `runs/<name>/`:
    adapter/            LoRA adapter + tokenizer (what infer.py loads)
    run_config.json     knobs + data file + git commit (reproducibility)
    trainer_log.jsonl   per-logging-step loss/lr history
    loss_curve.png      the same, as a chart
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = "Qwen/Qwen3-4B"


# ── OWNER KNOBS ──────────────────────────────────────────────────────────────
# Coaching ranges (HANDOFF Section 3), not prescriptions. Unset = the script refuses to
# train and tells you what's missing. Tune only against VAL macro-F1 (never test).
@dataclass
class Knobs:
    lora_r: int | None = None              # 16–32
    lora_alpha: int | None = None          # ~2× r
    target_modules: tuple[str, ...] | None = None
    #   attention only:  ("q_proj", "k_proj", "v_proj", "o_proj")
    #   + MLP:           (..., "gate_proj", "up_proj", "down_proj")
    learning_rate: float | None = None     # 1e-4 – 2e-4
    epochs: float | None = None            # 1–3 (small data - watch val overfit)
    per_device_batch: int | None = None    # bs × grad_accum = effective batch; size to fill 24 GB
    grad_accum: int | None = None
    max_seq_len: int | None = None         # size to recipe B's ~p95 (see --dry-run stats)

    # Plan of record (HANDOFF Section 3): completion-only loss - learn from the assistant
    # JSON, not from re-predicting the prompt. Flip to False only to see why it matters.
    completion_only_loss: bool = True

    # Plumbing defaults - override if you have an opinion, ignore otherwise.
    warmup_ratio: float = 0.03
    lr_scheduler: str = "linear"
    weight_decay: float = 0.01
    lora_dropout: float = 0.0              # 0 keeps Unsloth's fast path
    seed: int = 42


KNOBS = Knobs(
    # Owner-set 2026-07-28. One config for all three runs - the recipe is the only
    # variable. epochs=3 + per-epoch checkpoints: best epoch is selected on VAL via
    # eval.epoch_sweep, so the epoch count is a measurement, not a guess.
    lora_r=16,
    lora_alpha=32,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"),
    learning_rate=2e-4,
    epochs=3,
    per_device_batch=32,
    grad_accum=1,
    max_seq_len=256,
)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = ("lora_r", "lora_alpha", "target_modules", "learning_rate",
            "epochs", "per_device_batch", "grad_accum", "max_seq_len")


def _load_messages(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows or "messages" not in rows[0]:
        raise ValueError(f"{path} doesn't look like a *.messages.jsonl file")
    return rows


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _length_stats(tokenizer, texts: list[str]) -> dict:
    import numpy as np
    lens = np.array([len(tokenizer(t).input_ids) for t in texts])
    pct = {f"p{p}": int(np.percentile(lens, p)) for p in (50, 90, 95, 99)}
    return {"n": len(lens), **pct, "max": int(lens.max())}


def _render(tokenizer, rows: list[dict]) -> list[str]:
    """Apply the student tokenizer's chat template to each messages row."""
    return [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows
    ]


def _peek_mask(trainer, tokenizer) -> None:
    """Show what the loss actually sees for example 0 - verify the mask boundary."""
    try:
        ex = trainer.train_dataset[0]
        ids, labels = ex["input_ids"], ex.get("labels")
        if labels is None:
            print("[peek] no `labels` on the dataset yet (collator-time masking) - skipping")
            return
        kept = [i for i, l in zip(ids, labels) if l != -100]
        print("\n[peek] tokens contributing to the loss (example 0):")
        print(repr(tokenizer.decode(kept)))
        print(f"[peek] {len(kept)}/{len(ids)} tokens unmasked\n")
    except Exception as e:  # peek is diagnostics only - never kill a run over it
        print(f"[peek] skipped ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="one of data/train/*.messages.jsonl")
    ap.add_argument("--name", required=True, help="run name -> runs/<name>/")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true",
                    help="render + token-length stats + one full example, then exit (no knobs needed)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override KNOBS.seed for multi-seed variance runs (KNOBS stays the owner's file)")
    ap.add_argument("--epochs", type=float, default=None,
                    help="override KNOBS.epochs (e.g. the step-matched 6-epoch ablation control)")
    ap.add_argument("--per-device-batch", type=int, default=None,
                    help="memory-layout override; pair with --grad-accum to keep the effective batch")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="e.g. 16x2 instead of 32x1: same effective batch/steps/schedule, ~half the activation VRAM")
    ap.add_argument("--stage", type=int, default=None,
                    help="train up to epoch N this process, resuming from the last checkpoint. "
                         "Workaround for a per-step VRAM creep in the Windows stack that "
                         "livelocks runs longer than ~350 steps - one process per epoch stays "
                         "under the wall. The LR schedule is built for the FULL run and a "
                         "callback halts at the stage boundary, so staged == unstaged training "
                         "(the Bitext recipe_a run predates this and sawtoothed - see findings).")
    args = ap.parse_args()

    data_path = (REPO_ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    rows = _load_messages(data_path)
    print(f"{data_path.name}: {len(rows):,} examples")

    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        texts = _render(tokenizer, rows)
        print("token lengths (chat-templated):", json.dumps(_length_stats(tokenizer, texts)))
        print("\n-- example 0, exactly as the model sees it --\n")  # ASCII: Windows console is cp1252
        print(texts[0])
        return

    if args.seed is not None:
        KNOBS.seed = args.seed
    if args.epochs is not None:
        KNOBS.epochs = args.epochs
    if args.per_device_batch is not None:
        KNOBS.per_device_batch = args.per_device_batch
    if args.grad_accum is not None:
        KNOBS.grad_accum = args.grad_accum

    missing = [k for k in REQUIRED if getattr(KNOBS, k) is None]
    if missing:
        raise SystemExit(
            "KNOBS not set: " + ", ".join(missing)
            + f"\nEdit the KNOBS block in {Path(__file__).relative_to(REPO_ROOT)} "
              "(coaching ranges are in the comments / HANDOFF-M2-4090 Section 3). "
              "Run with --dry-run first for the max_seq_len token-length stats."
        )

    # Unsloth must be imported before transformers/trl to apply its patches.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    out_dir = REPO_ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=KNOBS.max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto (bf16 on the 4090)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=KNOBS.lora_r,
        lora_alpha=KNOBS.lora_alpha,
        target_modules=list(KNOBS.target_modules),
        lora_dropout=KNOBS.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=KNOBS.seed,
    )

    ds = Dataset.from_list([{"text": t} for t in _render(tokenizer, rows)])

    stage = args.stage or KNOBS.epochs
    final_stage = stage >= KNOBS.epochs

    config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_length=KNOBS.max_seq_len,
        per_device_train_batch_size=KNOBS.per_device_batch,
        gradient_accumulation_steps=KNOBS.grad_accum,
        # ALWAYS the full run's epochs: the scheduler derives its decay total from this,
        # and passing the stage here made stage-1 anneal to zero LR in one epoch (the
        # Bitext recipe_a sawtooth). _StopAfterStage halts at the stage boundary instead.
        num_train_epochs=KNOBS.epochs,
        learning_rate=KNOBS.learning_rate,
        lr_scheduler_type=KNOBS.lr_scheduler,
        warmup_ratio=KNOBS.warmup_ratio,
        weight_decay=KNOBS.weight_decay,
        seed=KNOBS.seed,
        logging_steps=5,
        save_strategy="epoch",      # one adapter checkpoint per epoch -> eval.epoch_sweep
        bf16=True,
        packing=False,
        # dataset_num_proc: leave UNSET on Windows. Any int (even 1) makes datasets
        # spawn a worker pool, and spawned workers can't import unsloth's compiled
        # cache module (ModuleNotFoundError: UnslothSFTTrainer). None = main process.
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=config)

    if KNOBS.completion_only_loss:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    _peek_mask(trainer, tokenizer)

    # Belt-and-braces against the VRAM creep: flush the CUDA allocator cache
    # periodically (costs ~ms; a fragmentation-driven creep dies here).
    from transformers import TrainerCallback

    class _FlushCache(TrainerCallback):
        def on_step_end(self, targs, state, control, **kw):
            if state.global_step % 50 == 0:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()

    class _StopAfterStage(TrainerCallback):
        """End this process at the stage boundary; the epoch checkpoint is already saved."""
        def on_epoch_end(self, targs, state, control, **kw):
            if state.epoch is not None and state.epoch >= stage - 1e-6:
                control.should_training_stop = True

    trainer.add_callback(_FlushCache())
    if not final_stage:
        trainer.add_callback(_StopAfterStage())

    resume = args.stage is not None and args.stage > 1
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    mins = (time.time() - t0) / 60

    if not final_stage:
        print(f"\nStage {stage}/{KNOBS.epochs} done in {mins:.1f} min - "
              f"checkpoint saved; run --stage {stage + 1} to continue")
        return

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (out_dir / "run_config.json").write_text(json.dumps({
        "name": args.name,
        "base": args.base,
        "data": str(data_path.relative_to(REPO_ROOT)),
        "n_examples": len(rows),
        "knobs": asdict(KNOBS),
        "git_commit": _git_commit(),
        "train_minutes": round(mins, 1),
        "final_loss": result.training_loss,
    }, indent=2, default=list))
    with (out_dir / "trainer_log.jsonl").open("w") as fh:
        for entry in trainer.state.log_history:
            fh.write(json.dumps(entry) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [(e["step"], e["loss"]) for e in trainer.state.log_history if "loss" in e]
        if pts:
            xs, ys = zip(*pts)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, ys)
            ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_title(args.name)
            fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=120)
    except Exception as e:
        print(f"[chart] skipped ({e})")

    print(f"\nDone in {mins:.1f} min - adapter at {adapter_dir.relative_to(REPO_ROOT)}")
    print("Next: uv run --no-sync python -m triage_distill.eval.infer "
          f"--adapter {adapter_dir.relative_to(REPO_ROOT)} --mode <classify|reason> "
          f"--out runs/{args.name}/preds.jsonl")


if __name__ == "__main__":
    main()
