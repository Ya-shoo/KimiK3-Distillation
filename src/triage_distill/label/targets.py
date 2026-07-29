"""Turn gold-gated K3 labels into student training targets (Recipe A / B / ablation).

Gold-gating (the quality filter): keep a row only when K3's predicted `category`
matched the Bitext `gold`. When they disagree, K3's reasoning went somewhere the
gold label says is wrong, so its rationale is untrustworthy as a teaching signal —
we drop it. The kept rows use the **gold** label as the student's `category` target.

Three format-agnostic intermediate files (input + structured target). The training
step (M2) renders these into the model's chat template — kept separate so the
training-config design stays the user's surface, not baked in here.

- recipe_a.jsonl  (multi-task; label-only at inference — SPEED): 2 rows per ticket
    {task: classify, input, target: {category}}
    {task: explain,  input, target: {evidence_to_intent, why_not_alternatives}}
- recipe_b.jsonl  (single sequence; reason-then-label — INTERPRETABLE): 1 row per ticket
    {input, target: {evidence_to_intent, why_not_alternatives, category}}
- ablation.jsonl  (control; label only): {input, target: {category}}

    uv run python -m triage_distill.label.targets
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from triage_distill.datasets import cfg

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python -m triage_distill.label.run` first.")
    recs = []
    for line in path.read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bitext")
    args = ap.parse_args()
    c = cfg(args.dataset)
    OUT_DIR = c.label_dir / "targets"
    recs = _load(c.label_dir / "labeled.jsonl")
    ok = [r for r in recs if "error" not in r]
    errors = len(recs) - len(ok)
    kept = [r for r in ok if r["gold_match"]]
    dropped = [r for r in ok if not r["gold_match"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe_a, recipe_b, ablation = [], [], []
    for r in kept:
        text, gold, res = r["text"], r["gold"], r["result"]
        evidence = res.get("evidence_to_intent", "")
        why_not = res.get("why_not_alternatives", "")
        recipe_a.append({"task": "classify", "input": text, "target": {"category": gold}})
        recipe_a.append({"task": "explain", "input": text,
                         "target": {"evidence_to_intent": evidence, "why_not_alternatives": why_not}})
        recipe_b.append({"input": text,
                         "target": {"evidence_to_intent": evidence, "why_not_alternatives": why_not, "category": gold}})
        ablation.append({"input": text, "target": {"category": gold}})

    _write(OUT_DIR / "recipe_a.jsonl", recipe_a)
    _write(OUT_DIR / "recipe_b.jsonl", recipe_b)
    _write(OUT_DIR / "ablation.jsonl", ablation)

    # Drop diagnostics — where does K3 disagree with Bitext gold? (methodology signal)
    drop_by_gold = Counter(r["gold"] for r in dropped)
    confusions = Counter((r["gold"], r["pred"]) for r in dropped)
    report = {
        "labeled_rows": len(recs),
        "errors": errors,
        "gold_match_kept": len(kept),
        "gold_mismatch_dropped": len(dropped),
        "keep_rate": round(len(kept) / len(ok), 4) if ok else 0,
        "targets": {"recipe_a": len(recipe_a), "recipe_b": len(recipe_b), "ablation": len(ablation)},
        "top_dropped_classes": drop_by_gold.most_common(10),
        "top_confusions_gold_to_pred": [
            {"gold": g, "pred": p, "n": n} for (g, p), n in confusions.most_common(10)
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("top_dropped_classes", "top_confusions_gold_to_pred")}, indent=2))
    print(f"\nkept {len(kept)}/{len(ok)} ({report['keep_rate']:.1%}); dropped {len(dropped)} gold-mismatches, {errors} errors")
    if confusions:
        print("top gold→pred disagreements (K3 vs gold):")
        for (g, p), n in confusions.most_common(5):
            print(f"  {g:>24} → {p:<24} ×{n}")
    print(f"-> {OUT_DIR.relative_to(REPO_ROOT)}/ (recipe_a, recipe_b, ablation .jsonl + report.json)")


if __name__ == "__main__":
    main()
