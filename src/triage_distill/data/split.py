"""Deterministic stratified train/val/test splits.

The TEST split is sacred: it is scored by the teacher, every panel model, the
student, and the ablation - never used for training or prompt tuning. Splits are
seeded and hashed so the exact partition is reproducible and comparable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw" / "bitext.parquet"
SPLITS = REPO_ROOT / "data" / "splits"
ARTIFACTS = REPO_ROOT / "artifacts"

SEED = 42
TEST_FRAC = 0.20   # of the whole
VAL_FRAC = 0.10    # of the whole


def _hash(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{t}\t{lbl}" for t, lbl in zip(df["text"], df["label"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. Run `python -m triage_distill.data.download` first.")

    SPLITS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW)

    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, random_state=SEED, stratify=df["label"]
    )
    val_rel = VAL_FRAC / (1 - TEST_FRAC)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=SEED, stratify=train_val["label"]
    )

    manifest = {"seed": SEED, "source_rows": len(df), "fractions": {"train": round(1 - TEST_FRAC - VAL_FRAC, 3), "val": VAL_FRAC, "test": TEST_FRAC}, "splits": {}}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.reset_index(drop=True)
        part.to_parquet(SPLITS / f"{name}.parquet", index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "hash": _hash(part),
            "n_classes": int(part["label"].nunique()),
        }

    (ARTIFACTS / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
