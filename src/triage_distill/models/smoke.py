"""Smoke-test the K3 teacher on a few real train tickets.

Needs TEACHER_API_KEY + TEACHER_MODEL (+ TEACHER_BASE_URL). Run:
    uv run python -m triage_distill.models.smoke [N]
Shows each ticket, K3's rationale, predicted vs gold, and the teacher-vs-gold
agreement (a preview of gold-gating) plus estimated cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from triage_distill.models.teacher import Teacher

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN = REPO_ROOT / "data" / "splits" / "train.parquet"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    df = pd.read_parquet(TRAIN).sample(n, random_state=0).reset_index(drop=True)
    teacher = Teacher()
    print(f"model: {teacher.model}\n")

    agree, tot_cost = 0, 0.0
    for _, row in df.iterrows():
        out = teacher.label(row.text)
        pred, gold = out["category"], row.label
        ok = pred == gold
        agree += ok
        tot_cost += out["cost_usd"]
        print(f"{'✓' if ok else '✗'} gold={gold:<24} pred={pred}")
        print(f"   ticket: {row.text[:100]}")
        print(f"   → {out['result'].get('evidence_to_intent', '')}")
        print(f"   → {out['result'].get('why_not_alternatives', '')}\n")

    print(f"teacher-vs-gold agreement: {agree}/{len(df)}   est. cost: ${tot_cost:.4f}")


if __name__ == "__main__":
    main()
