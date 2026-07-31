"""Crash-resilient babysitter for the training matrix.

The box bluescreened twice on 2026-07-29 (bugcheck 0x139, both times under GPU
load), killing the driver mid-matrix. This watchdog: relaunches the driver whenever
it isn't running, holds the machine awake (SetThreadExecutionState — process-scoped,
auto-reverts), and appends GPU temp/power telemetry to runs/gpu_telemetry.csv every
tick so the next crash leaves a trace. It's registered under HKCU\\...\\Run (via
matrix bring-up) so a post-crash logon auto-resumes the work, and it deregisters
itself once runs/matrix.done appears or it has given up.

    .venv\\Scripts\\pythonw.exe -m triage_distill.train.watchdog
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "runs"
DONE = RUNS / "matrix.done"
STATE = RUNS / "watchdog_state.json"
WLOG = RUNS / "watchdog.log"
TELEMETRY = RUNS / "gpu_telemetry.csv"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "KimiK3Watchdog"
MAX_RELAUNCH = 8
TICK_S = 30

ES_KEEP_AWAKE = 0x80000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
ES_RELEASE = 0x80000000     # ES_CONTINUOUS


def wlog(msg: str) -> None:
    with WLOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def _procs_with(needle: str) -> int:
    """Other live python processes whose cmdline matches. Name-filtered so the
    `cmd /c` wrappers don't count, and parent-excluded because the uv venv
    python.exe is a shim that spawns the real interpreter with the same cmdline."""
    import os
    import psutil
    skip = {os.getpid(), os.getppid()}
    n = 0
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in skip or not (p.info["name"] or "").lower().startswith("python"):
                continue
            if needle in " ".join(p.info["cmdline"] or []):
                n += 1
        except Exception:
            continue
    return n


def _launch_matrix() -> None:
    # Real file handles for stdout/stderr: under a detached/console-less parent the
    # cmd-with-redirection form inherits invalid std handles and dies before launching.
    import os
    out = (RUNS / "matrix_console.log").open("a", encoding="utf-8", errors="replace")
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: the driver outlives this watchdog.
    subprocess.Popen(
        [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "triage_distill.train.matrix"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONUTF8": "1"},
        stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=0x00000008 | 0x00000200)


def _telemetry() -> tuple[float, float] | None:
    """Append a telemetry line; return (power_W, mem_MiB) for the livelock detector."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,"
             "utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip()
            with TELEMETRY.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            parts = line.split(", ")
            return float(parts[2].split()[0]), float(parts[4].split()[0])
    except Exception:
        pass
    return None


def _kill_wedged_gpu_procs() -> None:
    """A WDDM livelock never recovers: kill the training/inference process so the
    driver's retry ladder restarts it (at the low-memory layout) within minutes,
    instead of waiting out the 45-min stage timeout."""
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if (p.info["name"] or "").lower().startswith("python") and (
                    "train.train" in cl or "eval.infer" in cl):
                wlog(f"livelock: killing wedged {p.info['pid']} ({cl[-60:]})")
                p.kill()
        except Exception:
            continue


def _deregister() -> None:
    subprocess.run(["reg", "delete", rf"HKCU\{RUN_KEY}", "/v", RUN_NAME, "/f"],
                   capture_output=True)


def main() -> None:
    wlog("watchdog starting (pre-guard)")  # first breath BEFORE anything that can throw
    try:
        others = _procs_with("triage_distill.train.watchdog")
    except Exception as e:  # logon-storm psutil hiccup: assume alone rather than die silently
        wlog(f"guard check failed ({e!r}) — assuming no other watchdog")
        others = 0
    if others > 0:
        wlog("another watchdog is running — exiting")
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_KEEP_AWAKE)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"relaunches": 0}
    wlog(f"watchdog up (relaunches so far: {state['relaunches']})")
    wedge_ticks = 0
    try:
        while True:
            if DONE.exists():
                wlog("matrix.done present — deregistering, exiting")
                _deregister()
                return
            if _procs_with("triage_distill.train.matrix") == 0:
                if state["relaunches"] >= MAX_RELAUNCH:
                    wlog(f"driver down but {MAX_RELAUNCH} relaunches spent — giving up")
                    _deregister()
                    return
                state["relaunches"] += 1
                STATE.write_text(json.dumps(state))
                wlog(f"driver not running — relaunch #{state['relaunches']}")
                _launch_matrix()
                time.sleep(90)  # let it get past imports before re-checking
            elif state["relaunches"]:
                # driver survived past a full tick — only fast-fail loops should
                # accumulate toward the cap, not recoveries from machine crashes
                state = {"relaunches": 0}
                STATE.write_text(json.dumps(state))
            t = _telemetry()
            if t is not None:  # livelock signature: VRAM at ceiling, compute idle
                power, mem = t
                wedge_ticks = wedge_ticks + 1 if (mem > 23000 and power < 150) else 0
                if wedge_ticks >= 3:
                    wlog(f"livelock pattern ({mem:.0f} MiB / {power:.0f} W x{wedge_ticks} ticks)")
                    _kill_wedged_gpu_procs()
                    wedge_ticks = 0
            time.sleep(TICK_S)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_RELEASE)


if __name__ == "__main__" and sys.platform == "win32":
    try:
        main()
    except Exception as e:  # pythonw is silent — leave a trace no matter what
        try:
            wlog(f"FATAL: {e!r}")
        finally:
            raise
