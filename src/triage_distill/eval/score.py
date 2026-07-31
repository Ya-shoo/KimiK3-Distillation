"""Score category predictions against gold — pure metrics, no GPU.

The headline is **macro-F1** (every intent weighted equally, so rare classes can't be
ignored); we also report accuracy, per-class F1, and the invalid-output rate (a
prediction that isn't valid JSON or isn't one of the 27 labels counts as WRONG — the
honest way to score a generative classifier). The student inference that produces the
predictions runs on the 4090; this scorer runs anywhere and is reused for every
frontier-panel model in M3, so all models are graded by identical code.

    uv run python -m triage_distill.eval.score --preds preds.jsonl --gold data/train/val_eval.jsonl

preds.jsonl : one {"id": <int>, "pred": "<category or null>"} per line
gold.jsonl  : one {"id": <int>, "gold": "<category>"} per line  (val_eval.jsonl works)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from triage_distill.datasets import cfg
from triage_distill.schema import load_label_space

REPO_ROOT = Path(__file__).resolve().parents[3]
OOS = "oos"  # CLINC's out-of-scope class — doubles as the escalate signal (HANDOFF start-here #3)
INVALID = "__invalid__"  # sentinel for unparseable / out-of-space predictions → always wrong


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def score(preds: dict[int, str | None], gold: dict[int, str], labels: list[str] | None = None) -> dict:
    """Align preds↔gold by id; return macro-F1, accuracy, per-class F1, invalid rate."""
    labels = labels or list(load_label_space())
    label_set = set(labels)
    ids = sorted(gold)
    missing = [i for i in ids if i not in preds]
    y_true, y_pred, n_invalid = [], [], 0
    for i in ids:
        p = preds.get(i)
        if p not in label_set:  # None, unparseable, or hallucinated label
            p, n_invalid = INVALID, n_invalid + 1
        y_true.append(gold[i])
        y_pred.append(p)

    per_class = dict(zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)))
    report = {
        "n": len(ids),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "invalid": n_invalid,
        "invalid_rate": round(n_invalid / len(ids), 4) if ids else 0.0,
        "missing_preds": len(missing),
        "worst_classes": sorted(((round(float(v), 4), k) for k, v in per_class.items()))[:5],
        "per_class_f1": {k: round(float(v), 4) for k, v in per_class.items()},
    }
    if OOS in label_set:  # escalate signal: of the truly out-of-scope queries, how many did we catch?
        oos_gold = sum(1 for t in y_true if t == OOS)
        oos_pred = sum(1 for p in y_pred if p == OOS)
        oos_hit = sum(1 for t, p in zip(y_true, y_pred) if t == p == OOS)
        report["oos"] = {
            "recall": round(oos_hit / oos_gold, 4) if oos_gold else None,
            "precision": round(oos_hit / oos_pred, 4) if oos_pred else None,
            "n_gold": oos_gold,
            "n_pred": oos_pred,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="JSONL of {id, pred}")
    ap.add_argument("--dataset", default="bitext", help="picks the frozen label space + default gold")
    ap.add_argument("--gold", default=None, help="JSONL of {id, gold} (default: the dataset's val_eval.jsonl)")
    ap.add_argument("--out", default=None, help="optional path to write the full report JSON")
    args = ap.parse_args()

    c = cfg(args.dataset)
    labels = list(load_label_space(c.label_space))
    gold_path = Path(args.gold) if args.gold else c.train_dir / "val_eval.jsonl"
    preds = {r["id"]: r.get("pred") for r in _read_jsonl(Path(args.preds))}
    gold = {r["id"]: r["gold"] for r in _read_jsonl(gold_path)}
    report = score(preds, gold, labels=labels)

    print(f"n={report['n']}  macro-F1={report['macro_f1']:.4f}  acc={report['accuracy']:.4f}  "
          f"invalid={report['invalid']} ({report['invalid_rate']:.1%})")
    print("weakest classes (F1):", ", ".join(f"{k}={v}" for v, k in report["worst_classes"]))
    if "oos" in report:
        o = report["oos"]
        print(f"oos (escalate): recall={o['recall']}  precision={o['precision']}  "
              f"gold={o['n_gold']}  predicted={o['n_pred']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
