"""Package a dataset's training runs into the git-tracked eval artifacts (HANDOFF Section 10/Section 11).

`runs/` holds weights and stays off git; the Mac builds M3 + the paper from NUMBERS.
This copies, per run, the eval outputs the paper needs into `artifacts/<dataset>/eval/`
(epoch scores, run config, trainer loss log, per-epoch preds) and assembles
`findings.json` - the PAPER-OUTLINE Part B log (learning curves, best epoch,
diminishing-returns knee, train-vs-val signal, seed inventory, adjustment changelog).

The changelog + caveats are prose the owner curates: pass --notes with a JSON file of
{"adjustment_changelog": [...], "caveats": [...]} to merge in, so the curated
interpretation lives next to the measured numbers.

    uv run --no-sync python -m triage_distill.eval.collect --dataset bitext --runs ablation recipe_a recipe_b
    uv run --no-sync python -m triage_distill.eval.collect --dataset clinc --runs clinc_ablation clinc_recipe_a clinc_recipe_b
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from triage_distill.datasets import cfg

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _train_loss_by_epoch(trainer_log: Path, n_epochs: int) -> list[dict]:
    """Mean + last train loss per epoch from the trainer's log history."""
    entries = [e for e in _read_jsonl(trainer_log) if "loss" in e and "epoch" in e]
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for e in entries:
        # trainer logs epoch as a float in (0, n_epochs]; bucket (k-1, k] -> epoch k
        k = min(n_epochs, int(-(-e["epoch"] // 1)))
        buckets[k].append((e["step"], e["loss"]))
    out = []
    for k in sorted(buckets):
        pts = sorted(buckets[k])
        losses = [l for _, l in pts]
        out.append({"epoch": k, "train_loss_mean": round(sum(losses) / len(losses), 4),
                    "train_loss_last": round(losses[-1], 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bitext")
    ap.add_argument("--runs", nargs="+", required=True, help="run dir names under runs/")
    ap.add_argument("--notes", default=None,
                    help="JSON file of curated prose to merge: adjustment_changelog, caveats, seeds note")
    ap.add_argument("--groups", default=None,
                    help="JSON file of seed families: {family: {anchor: run, members: [runs...]}}. "
                         "Each member's val macro-F1 is read AT THE ANCHOR'S BEST EPOCH and the "
                         "family gets mean +/- sample-sigma (the paper's error bars).")
    args = ap.parse_args()

    c = cfg(args.dataset)
    out_dir = c.eval_dir
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    findings: dict = {"dataset": args.dataset, "git_commit": _git_commit(), "runs": {}}
    seeds_seen: set[int] = set()

    for name in args.runs:
        run_dir = REPO_ROOT / "runs" / name
        scores = json.loads((run_dir / "epoch_scores.json").read_text())
        config = json.loads((run_dir / "run_config.json").read_text())

        shutil.copy2(run_dir / "epoch_scores.json", out_dir / f"epoch_scores_{name}.json")
        shutil.copy2(run_dir / "run_config.json", out_dir / f"run_config_{name}.json")
        shutil.copy2(run_dir / "trainer_log.jsonl", out_dir / f"trainer_log_{name}.jsonl")
        for p in sorted(run_dir.glob("preds_epoch*.jsonl")):
            shutil.copy2(p, preds_dir / f"{name}_{p.name}")

        curve = [{"epoch": e["epoch"], "val_macro_f1": round(e["macro_f1"], 4),
                  "val_accuracy": round(e["accuracy"], 4), "invalid": e["invalid"]}
                 for e in scores["epochs"]]
        f1s = [e["macro_f1"] for e in scores["epochs"]]
        best = scores["best_epoch"]
        # the knee: first epoch after which val F1 stops improving
        knee = next((e["epoch"] for i, e in enumerate(scores["epochs"][:-1])
                     if f1s[i + 1] <= f1s[i]), scores["epochs"][-1]["epoch"])
        seeds_seen.add(config["knobs"].get("seed"))

        findings["runs"][name] = {
            "mode": scores["mode"],
            "n_train_examples": config["n_examples"],
            "knobs": config["knobs"],
            "train_minutes": config.get("train_minutes"),
            "learning_curve": curve,
            "best_epoch": best,
            "best_val_macro_f1": round(max(f1s), 4),
            "diminishing_returns_after_epoch": knee,
            "train_loss_by_epoch": _train_loss_by_epoch(run_dir / "trainer_log.jsonl",
                                                        len(scores["epochs"])),
            "worst_classes_at_best": next(e["worst_classes"] for e in scores["epochs"]
                                          if e["epoch"] == best),
        }

    findings["seed_variance"] = {
        "seeds": sorted(seeds_seen),
        "n_seeds": len(seeds_seen),
    }
    if args.groups:
        import statistics
        families = {}
        for fam, g in json.loads(Path(args.groups).read_text(encoding="utf-8")).items():
            anchor = json.loads((REPO_ROOT / "runs" / g["anchor"] / "epoch_scores.json").read_text())
            epoch = anchor["best_epoch"]
            vals, oos_vals = {}, {}
            for m in g["members"]:
                sc = json.loads((REPO_ROOT / "runs" / m / "epoch_scores.json").read_text())
                row = next((e for e in sc["epochs"] if e["epoch"] == epoch), None)
                if row is None:
                    print(f"[groups] {fam}: {m} has no epoch {epoch} - excluded")
                    continue
                vals[m] = round(row["macro_f1"], 4)
                if "oos" in row:
                    oos_vals[m] = row["oos"]["recall"]
            fam_out = {
                "epoch": epoch,
                "n": len(vals),
                "macro_f1_mean": round(statistics.mean(vals.values()), 4) if vals else None,
                "macro_f1_sigma": round(statistics.stdev(vals.values()), 4) if len(vals) > 1 else None,
                "per_run": vals,
            }
            if oos_vals:
                fam_out["oos_recall_mean"] = round(statistics.mean(oos_vals.values()), 4)
                fam_out["oos_recall_per_run"] = oos_vals
            families[fam] = fam_out
        findings["seed_variance"]["families"] = families
    if args.notes:
        findings.update(json.loads(Path(args.notes).read_text(encoding="utf-8")))

    (out_dir / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    rel = out_dir.relative_to(REPO_ROOT)
    print(f"-> {rel}/findings.json  (+ per-run epoch_scores / run_config / trainer_log / preds)")
    for name, r in findings["runs"].items():
        print(f"   {name:>16}: best epoch {r['best_epoch']}  val macro-F1 {r['best_val_macro_f1']}")


if __name__ == "__main__":
    main()
