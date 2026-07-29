"""Score every epoch checkpoint of a run on val — plumbing for the epochs decision.

Training with save_strategy="epoch" leaves runs/<name>/checkpoints/checkpoint-<step>/
adapters. This sweep runs each one through the inference runner (a fresh subprocess
per checkpoint, so VRAM is fully released between loads), scores it, and writes:

    runs/<name>/preds_epoch<k>.jsonl     raw predictions per epoch
    runs/<name>/epoch_scores.json        the per-epoch metrics table (report material)
    runs/<name>/val_f1_by_epoch.png      val macro-F1 vs epoch

    uv run --no-sync python -m triage_distill.eval.epoch_sweep --run runs/ablation --mode classify
    uv run --no-sync python -m triage_distill.eval.epoch_sweep --run runs/recipe_b --mode reason
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from triage_distill.datasets import cfg
from triage_distill.eval.score import _read_jsonl, score
from triage_distill.schema import load_label_space

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja")


def _checkpoints(run_dir: Path) -> list[Path]:
    ckpts = [(int(m.group(1)), p) for p in (run_dir / "checkpoints").glob("checkpoint-*")
             if (m := re.match(r"checkpoint-(\d+)$", p.name))]
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir / 'checkpoints'} — train first")
    return [p for _, p in sorted(ckpts)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="runs/<name>")
    ap.add_argument("--mode", choices=("classify", "reason"), required=True)
    ap.add_argument("--dataset", default="bitext", help="picks the frozen label space + default gold")
    ap.add_argument("--gold", default=None, help="default: the dataset's val_eval.jsonl")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    dcfg = cfg(args.dataset)
    labels = list(load_label_space(dcfg.label_space))
    if args.gold is None:
        args.gold = str((dcfg.train_dir / "val_eval.jsonl").relative_to(REPO_ROOT))
    run_dir = (REPO_ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    gold_path = REPO_ROOT / args.gold
    gold = {r["id"]: r["gold"] for r in _read_jsonl(gold_path)}
    ckpts = _checkpoints(run_dir)
    print(f"{run_dir.name}: {len(ckpts)} epoch checkpoints -> {[p.name for p in ckpts]}")

    results = []
    for epoch, ckpt in enumerate(ckpts, start=1):
        # HF Trainer usually saves the tokenizer with each checkpoint; backfill from
        # the final adapter dir if this one is missing it.
        for f in TOKENIZER_FILES:
            src = run_dir / "adapter" / f
            if not (ckpt / f).exists() and src.exists():
                shutil.copy2(src, ckpt / f)

        preds_path = run_dir / f"preds_epoch{epoch}.jsonl"
        cmd = [sys.executable, "-m", "triage_distill.eval.infer",
               "--adapter", str(ckpt), "--mode", args.mode, "--dataset", args.dataset,
               "--data", args.gold, "--out", str(preds_path),
               "--batch-size", str(args.batch_size)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n── epoch {epoch} ({ckpt.name}) ──")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        preds = {r["id"]: r.get("pred") for r in _read_jsonl(preds_path)}
        gold_used = {i: g for i, g in gold.items() if i in preds} if args.limit else gold
        rep = score(preds, gold_used, labels=labels)
        row = {"epoch": epoch, "checkpoint": ckpt.name,
               "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
               "invalid": rep["invalid"], "worst_classes": rep["worst_classes"]}
        if "oos" in rep:  # CLINC: the escalate signal, tracked per epoch
            row["oos"] = rep["oos"]
        results.append(row)
        print(f"epoch {epoch}: macro-F1={rep['macro_f1']:.4f}  acc={rep['accuracy']:.4f}  "
              f"invalid={rep['invalid']}"
              + (f"  oos-recall={rep['oos']['recall']}" if "oos" in rep else ""))

    best = max(results, key=lambda r: r["macro_f1"])
    out = {"run": run_dir.name, "mode": args.mode, "gold": args.gold,
           "limit": args.limit, "best_epoch": best["epoch"], "epochs": results}
    (run_dir / "epoch_scores.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'epoch':>5}  {'macro-F1':>9}  {'acc':>7}  {'invalid':>7}")
    for r in results:
        mark = "  <- best" if r["epoch"] == best["epoch"] else ""
        print(f"{r['epoch']:>5}  {r['macro_f1']:>9.4f}  {r['accuracy']:>7.4f}  {r['invalid']:>7}{mark}")
    print(f"-> {run_dir / 'epoch_scores.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["epoch"] for r in results]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [r["macro_f1"] for r in results], marker="o", label="macro-F1")
        ax.plot(xs, [r["accuracy"] for r in results], marker="s", label="accuracy")
        ax.set_xticks(xs); ax.set_xlabel("epoch"); ax.set_title(f"{run_dir.name} — val by epoch")
        ax.legend(); fig.tight_layout()
        fig.savefig(run_dir / "val_f1_by_epoch.png", dpi=120)
        print(f"-> {run_dir / 'val_f1_by_epoch.png'}")
    except Exception as e:
        print(f"[chart] skipped ({e})")


if __name__ == "__main__":
    main()
