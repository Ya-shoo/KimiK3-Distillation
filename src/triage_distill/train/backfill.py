"""Back-fill gate-evacuated classes with bare gold-label rows (agent improvement line).

The gold-gate drops rows where the teacher's label disagrees with gold - protecting
training from contaminated rationales, but also deleting entire confusable classes
(CLINC `reminder_update`: 40/40 dropped -> student F1 0.0). The gold LABEL of a
dropped row is still trustworthy (it's the dataset's own answer); only the
teacher's reasoning isn't. So: append `query -> {category: gold}` label-only rows
for every dropped row to the ablation training set. No rationale is fabricated.

    uv run --no-sync python -m triage_distill.train.backfill --dataset clinc

Writes <train_dir>/ablation_bf.messages.jsonl (= ablation rows + backfill rows).
"""
from __future__ import annotations

import argparse
import json

from triage_distill.datasets import cfg
from triage_distill.train.prepare import SYS_CLASSIFY, _msg

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="clinc")
    args = ap.parse_args()
    c = cfg(args.dataset)

    labeled = [json.loads(l) for l in
               (c.label_dir / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dropped = [r for r in labeled if r.get("gold_match") is False]

    base = (c.train_dir / "ablation.messages.jsonl").read_text(encoding="utf-8")
    out = c.train_dir / "ablation_bf.messages.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        fh.write(base)
        for r in dropped:
            fh.write(json.dumps(_msg(SYS_CLASSIFY, r["text"], {"category": r["gold"]}),
                                ensure_ascii=False) + "\n")
    n_base = len(base.splitlines())
    print(f"{out.name}: {n_base} ablation rows + {len(dropped)} gold-label backfill rows "
          f"= {n_base + len(dropped)}")

if __name__ == "__main__":
    main()
