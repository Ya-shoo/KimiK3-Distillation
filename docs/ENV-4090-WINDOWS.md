# 4090 box (Windows) - environment notes

Set up 2026-07-28 per HANDOFF-M2-4090 Section 1. Working versions at the bottom.

## ⚠️ The one rule: always `uv run --no-sync` on this box

Unsloth lives **outside** the lockfile (per the handoff - its CUDA install matrix
doesn't belong in `pyproject.toml`), and it pins *older* versions than the lock
(transformers 5.5 vs 5.14, trl 0.24 vs 1.9, torch 2.11 vs 2.13). A bare `uv run`
or `uv sync` will "repair" the venv back to the lockfile - **uninstalling unsloth
and breaking training**. So:

```powershell
uv run --no-sync python -m ...        # every time, or: $env:UV_NO_SYNC = "1"
```

## Rebuild from scratch (or repair after an accidental sync)

```powershell
uv sync --group train                  # torch 2.13+cu130 via the pyproject cu130 index
uv pip install unsloth                 # pulls unsloth's pins - but a CPU torch 2.11 (!)
uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 --reinstall-package torch
uv pip install lm-format-enforcer      # constrained decoding for eval/infer.py
.venv\Scripts\python.exe -c "import unsloth, torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu130 True
```

Why the dance: PyPI's Windows torch wheels are CPU-only, and `--torch-backend=auto`
doesn't detect CUDA on Windows (it happily installs `+cpu`). Unsloth's resolver
downgrades torch to its pin (2.11.0) from PyPI, so the CUDA build has to be
force-reinstalled from the cu130 index afterwards. `[tool.uv.sources]` in
`pyproject.toml` handles this automatically for the *lockfile* torch (2.13), but
`uv pip` installs bypass it.

## Windows quirks encountered

- **Never launch anything via Task Scheduler (`schtasks`) + uv/venv-shim in this repo.**
  The task session cannot resolve the uv-managed base interpreter
  (`AppData\Roaming\uv\python\...`): the venv shim dies with "No Python at ...", and
  `uv run` - even with `--no-sync` - declares the venv "linked to non-existent Python
  interpreter", downloads a fresh CPython, and starts RECREATING the venv. A killed
  attempt (2026-07-29) left the venv partially gutted (annotated-types, certifi,
  charset-normalizer, einops, aiohttp, accelerate, bitsandbytes stripped; torch/unsloth
  survived). Repair = targeted `uv pip install` of the missing leaves (probe first with
  an import loop), NEVER `uv sync`. For a process that must survive the launcher's exit,
  spawn via WMI instead: `Invoke-CimMethod Win32_Process Create` with
  `cmd /c cd /d <repo> && set PYTHONUTF8=1&& .venv\Scripts\python.exe -m ...` - it runs
  outside the caller's job object with a normal user environment.
- **A CUDA process launched <1s after another CUDA process exits can die with
  0xC0000005** (access violation) during unsloth import - WDDM teardown race. Both the
  matrix driver and epoch_sweep now cool down 8s before each launch and retry a crash
  once. Timeouts (livelocks) are not retried.

- **uv installer race**: first `uv sync` after installing uv failed with "Missing
  expected target directory for Python minor version link" - just re-run, the
  second attempt succeeds.
- **never set `dataset_num_proc`** (SFTConfig / datasets.map) - any int, *including
  1*, makes `datasets` spawn a worker pool, and on Windows the spawned worker can't
  import unsloth's compiled-cache module (`ModuleNotFoundError: UnslothSFTTrainer`).
  Leave it unset (`None`) so the map runs in the main process.
- **lm-format-enforcer 0.11.x vs transformers 5.x**: LMFE imports
  `PreTrainedTokenizerBase` from a removed module path; `eval/infer.py` shims the
  alias before the import. Remove the shim if LMFE ships a fix.
- vLLM does not run on native Windows - inference here is transformers/unsloth.
- **Training runs longer than ~350 optimizer steps livelock** (not crash): VRAM
  creeps per step somewhere in the unsloth 2026.7.5 / trl 0.24 / transformers 5.5
  stack until it hits the 24 GB WDDM ceiling, where Windows pages GPU memory
  instead of raising OOM - the process sits at "100%" GPU / ~105 W forever, and it
  reproduces at the same step every run. The creep is CUDA allocator cache
  accumulation: `expandable_segments:True` alone did not fix it, but the
  every-50-steps `gc.collect()+empty_cache()` callback in `train.py` measurably
  resets it (observed −6.8 GB at a flush, steady 12.6 GB baseline after).
  Defense-in-depth workaround also built into `train.py`:
  `--stage N` runs one epoch per process with HF checkpoint resume (optimizer /
  scheduler / RNG restored - same training run, just split across processes).
  Runs ≤270 steps (ablation/recipe B at these knobs) fit in one process.
- **LMFE constrained decoding is ~100x slower on free-text JSON fields** (string
  fields allow ~the whole vocab per step; the per-token python cost scales with
  the allowed set). Enum-only schemas run at full speed. `eval/infer.py` reason
  mode therefore free-runs first and constrained-re-decodes only schema failures.

## Working version snapshot (verified: train dry-run + constrained-decode smoke test)

| package | version |
|---|---|
| python | 3.12.13 (uv-managed) |
| torch | 2.11.0+cu130 |
| unsloth / unsloth-zoo | 2026.7.5 / 2026.7.6 |
| transformers | 5.5.0 |
| trl | 0.24.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.50.0 |
| xformers | 0.0.35 |
| triton-windows | 3.7.1.post27 |
| lm-format-enforcer | 0.11.3 |
| NVIDIA driver | 596.49 (CUDA 13.0) |
