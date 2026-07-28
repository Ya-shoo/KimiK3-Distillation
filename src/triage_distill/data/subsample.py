"""Deterministic class-balanced subsample of TRAIN for teacher labeling.

Labeling every one of the 16,660 train rows with K3 would cost more than it's
worth; distillation needs a few thousand well-spread examples, not all of them
(see SPEC "Rationale distillation"). This picks an equal number per class so the
student sees a balanced signal, and freezes the choice (seeded + hashed, like the
splits) so the labeled set is reproducible. Size is a methodology knob:

    uv run python -m triage_distill.data.subsample --per-class 111   # ~3k rows
    uv run python -m triage_distill.data.subsample --total 4000      # ~148/class

The sacred TEST split is never touched here — we only ever subsample TRAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN = REPO_ROOT / "data" / "splits" / "train.parquet"
OUT_DIR = REPO_ROOT / "data" / "label"
OUT_PARQUET = OUT_DIR / "subsample.parquet"
ARTIFACTS = REPO_ROOT / "artifacts"
MANIFEST = ARTIFACTS / "subsample_manifest.json"

SEED = 42


def _hash(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{t}\t{lbl}" for t, lbl in zip(df["text"], df["label"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build(per_class: int) -> pd.DataFrame:
    df = pd.read_parquet(TRAIN)
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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-class", type=int, help="rows to sample per class (clamped to class size)")
    g.add_argument("--total", type=int, help="approx total rows; per-class = round(total / 27)")
    args = ap.parse_args()

    n_classes = pd.read_parquet(TRAIN, columns=["label"])["label"].nunique()
    if args.total:
        per_class = max(1, round(args.total / n_classes))
    elif args.per_class:
        per_class = args.per_class
    else:
        per_class = 111  # ~3k rows across 27 classes — the default starting point

    sub = build(per_class)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(OUT_PARQUET, index=False)

    dist = sub["label"].value_counts().sort_index().to_dict()
    manifest = {
        "seed": SEED,
        "per_class_target": per_class,
        "rows": len(sub),
        "n_classes": int(sub["label"].nunique()),
        "source_train_hash": _hash(pd.read_parquet(TRAIN)),
        "subsample_hash": _hash(sub),
        "min_per_class": int(min(dist.values())),
        "max_per_class": int(max(dist.values())),
        "class_distribution": {k: int(v) for k, v in dist.items()},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "class_distribution"}, indent=2))
    print(f"\nwrote {len(sub)} rows -> {OUT_PARQUET.relative_to(REPO_ROOT)}")
    print(f"per-class: min={manifest['min_per_class']} max={manifest['max_per_class']} "
          f"(balanced unless a class had < {per_class} rows)")


if __name__ == "__main__":
    main()
