"""Sequential driver for the confirmed CLINC + Bitext-backfill run matrix.

The plan (owner-confirmed 2026-07-28): 3 seeds per recipe on CLINC, 3 epochs with a
full per-epoch val sweep on the primary seed (42), extra seeds scored at the primary
seed's best epoch (held fixed), a step-matched 6-epoch ablation control on CLINC
(1074 optimizer steps ~= recipe A's 1071), Bitext backfilled with 2 extra seeds, and
a Bitext recipe_a s42 re-run under the fixed stage schedule (the original sawtoothed
its LR across stages — see artifacts/eval/findings.json caveats). Student prompts are
identical across datasets.

Every training run is staged one epoch per process (the Windows VRAM-creep livelock,
docs/ENV-4090-WINDOWS.md) with per-stage timeouts so a livelock kills the run, not
the night. Completed work is skipped on re-invocation, so the driver is resumable.

    uv run --no-sync python -m triage_distill.train.matrix --dry-run   # print the plan
    uv run --no-sync python -m triage_distill.train.matrix             # run it all
    uv run --no-sync python -m triage_distill.train.matrix --only clinc_recipe_a_s42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Windows: child processes inherit a cp1252 default that chokes on non-ASCII prints
# and file IO; force UTF-8 mode for everything the driver launches.
ENV = {**os.environ, "PYTHONUTF8": "1"}

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "runs"
LOG = RUNS / "matrix.log"
BATCH = 32  # must match KNOBS.per_device_batch; used only to infer completed stages

TRAIN_TIMEOUT_MIN = 45          # a stage is ~5-10 min; way past this = livelocked
SWEEP_TIMEOUT_MIN = {"classify": 120, "reason": 180}
LAUNCH_COOLDOWN_S = 8           # observed 0xC0000005 during unsloth import when a CUDA process
RETRY_COOLDOWN_S = 30           # launched <1s after the previous one exited (WDDM teardown race)

RECIPES = (("recipe_a", "classify"), ("ablation", "classify"), ("recipe_b", "reason"))
SEEDS_EXTRA = (1337, 2024)


def build_plan() -> list[dict]:
    C, B = "data/clinc/train", "data/train"
    plan = []
    # CLINC primary seed — recipe_a first: its 357 steps/stage probes the livelock wall early.
    for rec, mode in RECIPES:
        plan.append(dict(name=f"clinc_{rec}_s42", dataset="clinc", data=f"{C}/{rec}.messages.jsonl",
                         mode=mode, seed=42, epochs=3, sweep="full", best_from=None))
    # Step-matched control: label-only data at recipe A's optimizer budget.
    plan.append(dict(name="clinc_ablation_ctrl6_s42", dataset="clinc",
                     data=f"{C}/ablation.messages.jsonl", mode="classify", seed=42,
                     epochs=6, sweep="full", best_from=None))
    # CLINC extra seeds, scored at the primary seed's best epoch.
    for seed in SEEDS_EXTRA:
        for rec, mode in RECIPES:
            plan.append(dict(name=f"clinc_{rec}_s{seed}", dataset="clinc",
                             data=f"{C}/{rec}.messages.jsonl", mode=mode, seed=seed,
                             epochs=3, sweep="best", best_from=f"clinc_{rec}_s42"))
    # Bitext recipe_a s42 under the FIXED stage schedule (legacy runs/recipe_a sawtoothed;
    # kept for the record). Runs before the backfill so its best epoch can anchor them.
    plan.append(dict(name="bitext_recipe_a_s42v2", dataset="bitext",
                     data=f"{B}/recipe_a.messages.jsonl", mode="classify", seed=42,
                     epochs=3, sweep="full", best_from=None))
    # Bitext seed backfill. ablation/recipe_b anchor on the legacy runs (their schedules
    # were clean single-process); recipe_a anchors on the fixed-schedule re-run.
    for seed in SEEDS_EXTRA:
        for rec, mode in RECIPES:
            anchor = "bitext_recipe_a_s42v2" if rec == "recipe_a" else rec
            plan.append(dict(name=f"bitext_{rec}_s{seed}", dataset="bitext",
                             data=f"{B}/{rec}.messages.jsonl", mode=mode, seed=seed,
                             epochs=3, sweep="best", best_from=anchor))
    return plan


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd: list[str], timeout_min: float, retries: int = 1) -> str:
    """Run with a pre-launch cooldown; retry crashes once. Timeouts (livelocks) don't retry —
    a second 45-min hang would cost more than the run is worth tonight."""
    res = "not run"
    for attempt in range(retries + 1):
        time.sleep(LAUNCH_COOLDOWN_S if attempt == 0 else RETRY_COOLDOWN_S)
        log(("  $ " if attempt == 0 else f"  $ [retry {attempt}] ") + " ".join(cmd))
        try:
            r = subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout_min * 60, env=ENV)
            if r.returncode == 0:
                return "ok"
            res = f"exit {r.returncode}"
        except subprocess.TimeoutExpired:
            return "TIMEOUT (livelock?)"
        log(f"  attempt {attempt + 1}: {res}")
    return res


def _stages_done(run_dir: Path, steps_per_epoch: int) -> int:
    """Highest completed epoch, inferred from existing checkpoint step numbers.

    A hard kill mid-save leaves a checkpoint whose trainer_state.json is all NULs
    (allocated, never flushed) — HF resume would crash on it, so quarantine it.
    """
    import shutil
    best = 0
    for p in (run_dir / "checkpoints").glob("checkpoint-*"):
        if m := re.match(r"checkpoint-(\d+)$", p.name):
            try:
                json.loads((p / "trainer_state.json").read_text(encoding="utf-8"))
            except Exception:
                log(f"  corrupt checkpoint {p.name} (truncated save) — deleting")
                shutil.rmtree(p, ignore_errors=True)
                continue
            best = max(best, int(m.group(1)) // steps_per_epoch)
    return best


def run_one(spec: dict) -> str:
    name, run_dir = spec["name"], RUNS / spec["name"]
    if (run_dir / "epoch_scores.json").exists():
        log(f"{name}: epoch_scores.json exists — skip")
        return "done"

    # --- train (staged one epoch per process) ---
    if not (run_dir / "adapter").exists():
        n_rows = sum(1 for l in (REPO_ROOT / spec["data"]).open(encoding="utf-8") if l.strip())
        spe = math.ceil(n_rows / BATCH)
        start = _stages_done(run_dir, spe) + 1
        # Post-reboot the VRAM floor rose ~6GB and bs32 livelocks into the WDDM ceiling
        # intermittently (recipe_b stage 1 passed, stage 2 froze). Default ALL training to
        # 16x2: same effective batch/steps/schedule, ~half the activation memory.
        # Optimizer-step numbering is unchanged, so spe still holds.
        lowmem = True
        for stage in range(start, int(spec["epochs"]) + 1):
            cmd = [sys.executable, "-m", "triage_distill.train.train",
                   "--data", spec["data"], "--name", name, "--stage", str(stage),
                   "--seed", str(spec["seed"]), "--epochs", str(spec["epochs"])]
            low = ["--per-device-batch", "16", "--grad-accum", "2"]
            res = sh(cmd + (low if lowmem else []), TRAIN_TIMEOUT_MIN)
            if res.startswith("TIMEOUT") and not lowmem:
                lowmem = True
                log(f"{name}: stage {stage} livelocked at bs32x1 — retrying as bs16xga2 "
                    "(same effective batch, half the activation VRAM)")
                res = sh(cmd + low, TRAIN_TIMEOUT_MIN)
            if res != "ok":
                log(f"{name}: stage {stage} FAILED ({res}) — run abandoned")
                return f"train stage {stage}: {res}"

    # --- sweep ---
    cmd = [sys.executable, "-m", "triage_distill.eval.epoch_sweep",
           "--run", f"runs/{name}", "--mode", spec["mode"], "--dataset", spec["dataset"]]
    if spec["sweep"] == "best":
        anchor = RUNS / spec["best_from"] / "epoch_scores.json"
        if not anchor.exists():
            log(f"{name}: anchor {spec['best_from']} has no epoch_scores.json — skip sweep")
            return "no anchor"
        cmd += ["--only-epoch", str(json.loads(anchor.read_text())["best_epoch"])]
    res = sh(cmd, SWEEP_TIMEOUT_MIN[spec["mode"]])
    if res != "ok":
        log(f"{name}: sweep FAILED ({res})")
        return f"sweep: {res}"
    log(f"{name}: complete")
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    args = ap.parse_args()

    plan = build_plan()
    if args.only:
        plan = [s for s in plan if args.only in s["name"]]
    if args.dry_run:
        for s in plan:
            extra = f" @best-of:{s['best_from']}" if s["sweep"] == "best" else ""
            print(f"{s['name']:>28}  {s['data']:<40} mode={s['mode']:<8} "
                  f"seed={s['seed']:<5} epochs={s['epochs']}{extra}")
        print(f"({len(plan)} runs)")
        return

    RUNS.mkdir(exist_ok=True)
    if sys.platform == "win32":  # hold the box awake for the whole pass (auto-reverts on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    log(f"=== matrix start: {len(plan)} runs ===")
    outcomes = {}
    for spec in plan:
        log(f"--- {spec['name']} ---")
        outcomes[spec["name"]] = run_one(spec)
    log("=== matrix done ===")
    for n, o in outcomes.items():
        log(f"  {n:>28}: {o}")
    (RUNS / "matrix_outcomes.json").write_text(json.dumps(outcomes, indent=2))
    failed = {n: o for n, o in outcomes.items() if o not in ("done",)}
    if failed:
        raise SystemExit(f"{len(failed)} run(s) incomplete: {failed}")
    # all clean: tell the watchdog to stand down
    (RUNS / "matrix.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
