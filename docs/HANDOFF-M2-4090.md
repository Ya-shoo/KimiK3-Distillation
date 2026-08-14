# Handoff - M2 student training on the RTX 4090

## ⚡ START HERE - status & what changed since Bitext (read first)
If you just finished the **Bitext** training, the plan has expanded. Current state:

1. **Bitext M2 is done** - Recipe A won @ ~2 epochs. **Reuse that config as the CLINC starting point.**
   If you haven't yet, first **push your Bitext results** (Section 10) and **log the Section 11 findings** (the paper needs them).
2. **There are now TWO benchmarks.** Your next task is the **CLINC-150 track** - mirror your Bitext run on the
   new committed data (`data/clinc/`): `prepare --dataset clinc` → train **A / B / ablation** on the **151 labels**
   (150 intents + `oos`) → score on the CLINC test sample. Everything downstream takes `--dataset clinc`.
3. **Eval is label-only.** Recipe A (the winner) emits label-only at inference, so score everything label-only
   (constrained JSON, `category` only). `oos` doubles as the escalate signal - report its recall.
4. **The panel + paper are the M3/M6 targets.** The finalized 7-model two-tier panel is in `configs/models.yaml`;
   the end deliverable is a mini-paper + resume artifact spec'd in `docs/PAPER-OUTLINE.md`. **Emit Section 11 findings as
   JSON** (per-epoch curves, best epoch, adjustment changelog, seed variance) - the paper + charts read them.
5. **Round-trip:** push code + `artifacts/<dataset>/eval/*.json` back (Section 10); the Mac builds M3 + the paper.
   Weights stay off git (→ Hugging Face).

The rest of this doc (Section 0–11) is the original Bitext-M2 guide - still valid for **mechanics**; just apply it with
`--dataset clinc`. And keep the **collaboration model** below: the owner drives the training-loop/LoRA decisions;
you scaffold, coach, and log.

---

**Focus for this machine:** train the small **student** (Qwen3-4B, QLoRA) on the K3-distilled
data produced in M1, evaluate on the val split, and run the **A vs B vs ablation** controlled
experiment. All the teacher/labeling work is done and committed - this box does the GPU training.

Repo: `KimiK3-Distillation`. Plan of record: **`SPEC.md`** (read Section 5 Method + Section 6 Eval). Don't
duplicate it here. Locked kickoff decisions live in the project memory; the concrete M1 outcome
is summarized below.

---

## ⚠️ Collaboration model (read this first)
The **owner personally does the load-bearing ML** to learn it - here that means the **QLoRA
training loop, the LoRA hyperparameters, loss-masking, and interpreting A/B/ablation.** The
agent's job is to **scaffold and coach, not hand over a finished training config.** Wire the
plumbing (data loading, inference runner, metrics, charts, checkpoint/logging), and leave the
conceptual training choices to the owner. Offer to take pieces over only when asked.

---

## 0. Get the code + data onto this box
Already committed + pushed to a **private** repo. On this box:
```bash
git clone https://github.com/Ya-shoo/KimiK3-Distillation.git
cd KimiK3-Distillation
```
The K3-labeled set is **git-tracked on purpose** (it cost ~$16 and isn't cheaply regenerable):
`data/label/labeled.jsonl`, `data/label/subsample.parquet`, `data/label/targets/*` all travel with
the repo. Raw Bitext + splits stay git-ignored (regenerate with the seeded scripts if needed).
The `data/train/*.messages.jsonl` training files are **not** committed - they're regenerated here
by `prepare.py` (deterministic) in step 2.

*(No `.env` / API key is needed for M2 - training is fully local. The key only matters for the M3
frontier-panel eval, later, back on whichever box runs the API calls.)*

## 1. Environment (CUDA box only)
```bash
# uv + Python 3.12 (pinned; do NOT use system python)
uv sync --group train           # torch, transformers, trl, peft, accelerate, bitsandbytes, datasets
# Unsloth is installed separately per its own CUDA install matrix - see https://github.com/unslothai/unsloth
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The `train` dependency group is CUDA-only and was intentionally **not** installed on the Mac.

## 2. Regenerate the training data (already-built plumbing)
```bash
uv run python -m triage_distill.train.prepare
```
Writes to `data/train/`:
- `recipe_a.messages.jsonl` - 5,722 rows (multi-task: `classify` + `explain`, system prompt selects the task)
- `recipe_b.messages.jsonl` - 2,861 rows (single sequence: reason-then-label)
- `ablation.messages.jsonl` - 2,861 rows (label-only control)
- `val_eval.jsonl` - 2,381 rows (`{id, text, gold}`; the student must produce the category itself)

Each training line is `{"messages": [ {system}, {user: 'Ticket: "..."'}, {assistant: <JSON answer>} ]}` -
**model-agnostic**: apply the student tokenizer's chat template on this box. Switching students
(Qwen/Llama/Phi) is just a different chat template, no re-prep.

> **Student prompts are the owner's knob.** They live as `SYS_CLASSIFY / SYS_EXPLAIN / SYS_REASON`
> constants in `src/triage_distill/train/prepare.py` and are imported by the eval side, so
> **training and inference stay in lockstep** - change them in one place. (Can be promoted to a
> `prompts/student.md` design surface like `prompts/teacher.md` if you want to iterate.)

## 3. The M2 task - three QLoRA runs on Qwen3-4B
Train the **same** student + config three times, once per target file, so the only variable is the
rationale signal (that's what makes it a controlled experiment, not just "I fine-tuned a model"):

| Run | Train file | Purpose |
|---|---|---|
| **ablation** (do first) | `ablation.messages.jsonl` | baseline everything else must beat |
| **recipe B** | `recipe_b.messages.jsonl` | reason-then-label (interpretable) |
| **recipe A** | `recipe_a.messages.jsonl` | multi-task; label-only at inference (speed) |

**Decisions that are the owner's to make** (coaching ranges, not prescriptions):
- LoRA `r` (16–32) and `alpha` (~2×r); `target_modules` (attn q/k/v/o ± MLP gate/up/down)
- learning rate (1e-4–2e-4), epochs (1–3 - small data, watch val overfit), effective batch (bs × grad-accum to fill 24 GB)
- **loss-masking = completion-only** - compute loss on the assistant JSON only, not the prompt (TRL's `DataCollatorForCompletionOnlyLM` / SFTTrainer). This is *the* concept for "learns to produce JSON."
- `max_seq_len` - Recipe B is longest (rationale + label); size to its ~95th-pct token length.

**Base model:** `Qwen/Qwen3-4B` (Apache-2.0, native reasoning mode, first-class Unsloth support). The
choice is cheap to revisit - targets are model-agnostic. Optional: a 1.7B/4B/8B size-frontier.

## 4. Plumbing status - built vs to-build
- ✅ **Built + tested (on the Mac, no GPU):**
  - `triage_distill.train.prepare` - targets → `messages` training files (Section 2)
  - `triage_distill.eval.score` - macro-F1 (headline) + accuracy + per-class F1 + invalid-rate;
    invalid/hallucinated outputs count as **wrong**. Reused verbatim for the M3 frontier panel.
    Usage: `uv run python -m triage_distill.eval.score --preds preds.jsonl --gold data/train/val_eval.jsonl`
- 🚧 **To build here:**
  - `train.py` - **owner's surface.** Load 4-bit Qwen3-4B + LoRA, SFT on a chosen `*.messages.jsonl`,
    save the adapter. Agent scaffolds the runnable skeleton around the owner's hyperparameters.
  - **Inference runner** (plumbing) - load a trained adapter, run each `val_eval.jsonl` ticket with
    **constrained JSON decoding** (`triage_distill.schema.category_json_schema()` - enforce the 27-label
    enum so output is always parseable), emit `preds.jsonl` = `{id, pred}` for the scorer. For Recipe A,
    inference uses the `classify` task only (label-only → fast).

## 5. Eval loop + discipline
```
trained adapter ──► inference runner (val_eval) ──► preds.jsonl ──► score.py ──► macro-F1
```
- **Tune only on val.** The **test split is sacred** (`d7e24e9e…`, 4,761 rows) - scored exactly once
  in M3 alongside the frontier panel, never for training or tuning.
- **Pass gate (SPEC Section 1):** student macro-F1 **≥ 97.5% of K3's** macro-F1, **and** rationale students
  (A/B) **beat** the ablation. ⚠️ The 97.5% needs a **K3 baseline** scored with the *same* `score.py`
  on the same split - that's an M3 panel task and doesn't exist yet, so for now compare A/B/ablation to
  each other and track absolute val macro-F1.

## 6. Key facts (frozen)
- **Label space:** 27 intents, frozen at `artifacts/label_space.json` - nothing hardcodes the list; load via `schema.load_label_space()`.
- **Subsample:** 2,997 rows, 111/class, `subsample_hash=b8cc7b00ecb64726` (`artifacts/subsample_manifest.json`).
- **M1 labeling:** 0 errors, **95.5% gold-gate keep rate** (2,861 kept / 136 dropped), ~$15.86. Detail + the
  Bitext label-noise finding (`delivery_period→track_order`) are in the `m1-teacher-labeling` project memory.

## 7. Gotchas
- Don't `uv sync` the base group and expect training - you need `--group train` (CUDA).
- Don't train or tune on the **test** split. Val is for tuning; test is scored once (M3).
- Keep `SYS_*` prompts identical between `prepare.py` (training) and the inference runner (import them).
- Constrained decoding at inference isn't optional - the scorer marks invalid outputs wrong.
- `data/` is git-ignored **except** the tracked M1 artifacts (see Section 0); `data/train/` regenerates via `prepare.py`.

## 8. Suggested skills (invoke as needed on this box)
- **`run`** - when it's time to actually launch/smoke-test the trained student.
- **`claude-api`** - only for the later M3 panel/API pricing questions (not needed for local M2).
- **`handoff`** - to produce the next handoff (e.g. M2 → M3) when this box's work wraps.

## 9. File reference (don't duplicate - read these)
- `SPEC.md` - the plan (Section 5 Method, Section 6 Eval).
- `src/triage_distill/schema.py` - frozen label space, output schemas, `category_json_schema()` for constrained decoding.
- `src/triage_distill/train/prepare.py` - data prep + the student `SYS_*` prompts (owner's knob).
- `src/triage_distill/eval/score.py` - the scorer (students now, panel in M3).
- `data/label/targets/` - gold-gated A/B/ablation targets (committed). `data/label/labeled.jsonl` - full K3 output incl. archived reasoning traces.
- *(The project notes that informed this - project overview, working model, and M1 teacher-labeling - live on the **dev Mac**, not in the repo. Everything load-bearing from them is inlined above, so this doc stands alone.)*

## 10. Sending results back to the dev Mac (GitHub round-trip)
The Mac continues to **M3** (frontier panel + accuracy-vs-cost chart) from your **numbers**, not your
weights - it has no CUDA and can't run the student. So push back **code + eval results**, and keep
**weights out of git**:

- **Weights/adapters do NOT go through git** - `.gitignore` already blocks `outputs/`, `checkpoints/`,
  `*.safetensors`, `*.gguf`, `*.bin`. To share the weights, push them to **Hugging Face** (SPEC's plan:
  model card + weights); otherwise they can stay on the 4090. The Mac doesn't need them for M3.
- **DO commit + push:** your `train.py` + inference-runner source, and the eval outputs the Mac needs.
  Write eval outputs to the **git-tracked** `artifacts/eval/` dir - e.g. `score.py --out
  artifacts/eval/recipe_b.json`, plus each run's `preds.jsonl`. (`artifacts/` is tracked; `data/` mostly isn't.)
- From the 4090: `git add -A && git commit -m "M2: student training + val scores" && git push`
- Back on the Mac: `git pull` → the training code + val macro-F1 numbers land, and we pick up M3.

**Discipline:** always `git pull` before starting on either box so the two never diverge. Linear `main`
is fine for a solo two-box flow; use a branch + PR if you prefer a review step.

## 11. Log these findings for the mini-paper (IMPORTANT)
The end deliverable is a mini research paper + a resume artifact - see **`docs/PAPER-OUTLINE.md`**.
As you train, **emit findings as JSON to `artifacts/<dataset>/eval/`** (not just terminal scrollback),
because the paper and charts read them directly. Minimum to capture (full list in PAPER-OUTLINE Part B):

- **Per-epoch val macro-F1** per recipe (A/B/ablation) → learning curves + the "which epoch won / where
  returns diminish" claim. *(Bitext already showed **Recipe A best @ 2 epochs** - save the curve behind it.)*
- **A changelog of adjustments** (LR, LoRA r/alpha, max_seq_len, loss-masking) and *why*.
- **≥2–3 seeds/recipe** → mean ± σ (feeds the error bars).
- **Train-vs-val gap per epoch** → the overfitting analysis.
- **Final test scores, scored ONCE**, all recipes × both datasets → `artifacts/<dataset>/eval/test_<recipe>.json`.

Run CLINC as its own track mirroring Bitext: `prepare --dataset clinc` → train A/B/ablation on the 151
labels → score on the CLINC test sample. Then push results back per Section 10 for the Mac to build M3 + the paper.
