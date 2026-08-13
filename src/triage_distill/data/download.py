"""Download the Bitext customer-support dataset and freeze the label space.

Mapping from Bitext columns to our task:
- text     = Bitext `instruction`
- label    = Bitext `intent`   (fine-grained, ~27 classes; e.g. "get_refund")
- coarse   = Bitext `category` (11 coarse groups; kept for analysis only)

NB: the spec's output field `category` is the FINE-GRAINED `intent` here - Bitext's
own `category` column is the coarse grouping, which we keep under `coarse`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(HF_DATASET, split="train")
    df = (
        ds.to_pandas()[["instruction", "intent", "category"]]
        .rename(columns={"instruction": "text", "intent": "label", "category": "coarse"})
    )
    df["text"] = df["text"].astype(str).str.strip()
    df = (
        df.dropna(subset=["text", "label"])
        .drop_duplicates(subset=["text", "label"])
        .reset_index(drop=True)
    )

    # Near-duplicate guard: Bitext is templated, so many rows are the same skeleton
    # with different placeholder values. Collapse them on a normalized key BEFORE
    # splitting, so identical phrasings cannot leak across train/val/test.
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\{\{.*?\}\}", " ", s)      # strip placeholders e.g. {{Order Number}}
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    df["_norm"] = df["text"].map(_norm)
    conflicts = int((df.groupby("_norm")["label"].nunique() > 1).sum())
    before = len(df)
    df = df.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm").reset_index(drop=True)
    print(
        f"Near-dup collapse: {before:,} -> {len(df):,} rows "
        f"({before - len(df):,} removed; {conflicts} normalized keys had label conflicts)"
    )

    raw_path = RAW_DIR / "bitext.parquet"
    df.to_parquet(raw_path, index=False)

    labels = sorted(df["label"].unique().tolist())
    coarse = sorted(df["coarse"].dropna().unique().tolist())
    (ARTIFACTS_DIR / "label_space.json").write_text(
        json.dumps(
            {"labels": labels, "n_classes": len(labels), "coarse_groups": coarse, "source": HF_DATASET},
            indent=2,
        )
    )

    print(f"Rows: {len(df):,}  |  classes: {len(labels)}  |  coarse groups: {len(coarse)}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {ARTIFACTS_DIR / 'label_space.json'}")


if __name__ == "__main__":
    main()
