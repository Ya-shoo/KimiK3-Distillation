"""Deterministic class-balanced subsample of TRAIN for teacher labeling.

Labeling every train row with K3 would cost more than it's worth; distillation needs a
few thousand well-spread examples, not all of them. This picks an equal number per
class so the student sees a balanced signal, and freezes the choice (seeded + hashed,
like the splits) so the labeled set is reproducible. Works for any registered dataset:

    uv run python -m triage_distill.data.subsample                      # bitext, ~3k
    uv run python -m triage_distill.data.subsample --dataset clinc      # clinc, ~3k (~20/class)
    uv run python -m triage_distill.data.subsample --dataset clinc --per-class 30

Size is a methodology knob (`--per-class` / `--total`). The sacred TEST split is never
touched - we only ever subsample TRAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from triage_distill.datasets import cfg

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 42
DEFAULT_TOTAL = 3000  # default subsample size; per-class = round(total / n_classes)


def _hash(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{t}\t{lbl}" for t, lbl in zip(df["text"], df["label"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build(per_class: int, train_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(train_path)
    parts = []
    for lbl, g in df.groupby("label", sort=True):
        k = min(per_class, len(g))
        parts.append(g.sample(n=k, random_state=SEED))
    # Sort for a stable id assignment; training shuffles anyway.
    sub = pd.concat(parts).sort_values(["label", "text"]).reset_index(drop=True)
    sub.insert(0, "id", sub.index)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bitext")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-class", type=int, help="rows to sample per class (clamped to class size)")
    g.add_argument("--total", type=int, help="approx total rows; per-class = round(total / n_classes)")
    args = ap.parse_args()

    c = cfg(args.dataset)
    train_path = c.splits_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing. Build the '{args.dataset}' splits first.")
    n_classes = pd.read_parquet(train_path, columns=["label"])["label"].nunique()
    if args.per_class:
        per_class = args.per_class
    else:
        per_class = max(1, round((args.total or DEFAULT_TOTAL) / n_classes))

    sub = build(per_class, train_path)
    c.label_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = c.label_dir / "subsample.parquet"
    sub.to_parquet(out_parquet, index=False)

    dist = sub["label"].value_counts().sort_index().to_dict()
    manifest = {
        "dataset": args.dataset,
        "seed": SEED,
        "per_class_target": per_class,
        "rows": len(sub),
        "n_classes": int(sub["label"].nunique()),
        "source_train_hash": _hash(pd.read_parquet(train_path)),
        "subsample_hash": _hash(sub),
        "min_per_class": int(min(dist.values())),
        "max_per_class": int(max(dist.values())),
        "class_distribution": {k: int(v) for k, v in dist.items()},
    }
    c.subsample_manifest.parent.mkdir(parents=True, exist_ok=True)
    c.subsample_manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "class_distribution"}, indent=2))
    print(f"\nwrote {len(sub)} rows -> {out_parquet.relative_to(REPO_ROOT)}")
    print(f"per-class: min={manifest['min_per_class']} max={manifest['max_per_class']} "
          f"(balanced unless a class had < {per_class} rows)")


if __name__ == "__main__":
    main()
