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
    cmd = (f'cd /d "{REPO_ROOT}" && set PYTHONUTF8=1&& '
           f'"{REPO_ROOT}\\.venv\\Scripts\\python.exe" -m triage_distill.train.matrix '
           f'>> "{RUNS}\\matrix_console.log" 2>&1')
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: the driver outlives this watchdog.
    subprocess.Popen(["cmd", "/c", cmd], creationflags=0x00000008 | 0x00000200,
                     close_fds=True)


def _telemetry() -> None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,"
             "utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            with TELEMETRY.open("a", encoding="utf-8") as fh:
                fh.write(r.stdout.strip() + "\n")
    except Exception:
        pass


def _deregister() -> None:
    subprocess.run(["reg", "delete", rf"HKCU\{RUN_KEY}", "/v", RUN_NAME, "/f"],
                   capture_output=True)


def main() -> None:
    if _procs_with("triage_distill.train.watchdog") > 0:
        return  # another watchdog already holds the fort
    ctypes.windll.kernel32.SetThreadExecutionState(ES_KEEP_AWAKE)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"relaunches": 0}
    wlog(f"watchdog up (relaunches so far: {state['relaunches']})")
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
            _telemetry()
            time.sleep(TICK_S)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_RELEASE)


if __name__ == "__main__" and sys.platform == "win32":
    main()
