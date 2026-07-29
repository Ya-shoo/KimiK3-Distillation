"""Smoke-test the K3 teacher on a few real train tickets (any dataset).

Needs TEACHER_API_KEY + TEACHER_MODEL (+ TEACHER_BASE_URL). Run:
    uv run python -m triage_distill.models.smoke [N] [--dataset bitext|clinc]
Shows each ticket, K3's rationale, predicted vs gold, and the teacher-vs-gold
agreement (a preview of gold-gating) plus estimated cost.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from triage_distill.datasets import cfg
from triage_distill.models.teacher import Teacher

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", nargs="?", type=int, default=10, help="number of tickets to sample")
    ap.add_argument("--dataset", default="bitext")
    args = ap.parse_args()

    c = cfg(args.dataset)
    df = pd.read_parquet(c.splits_dir / "train.parquet").sample(args.n, random_state=0).reset_index(drop=True)
    teacher = Teacher(prompt_path=c.teacher_prompt, label_space_path=c.label_space)
    print(f"dataset: {args.dataset} | model: {teacher.model}\n")

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
