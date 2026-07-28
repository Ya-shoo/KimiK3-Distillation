# Handoff — M2 student training on the RTX 4090

**Focus for this machine:** train the small **student** (Qwen3-4B, QLoRA) on the K3-distilled
data produced in M1, evaluate on the val split, and run the **A vs B vs ablation** controlled
experiment. All the teacher/labeling work is done and committed — this box does the GPU training.

Repo: `KimiK3-Distillation`. Plan of record: **`SPEC.md`** (read §5 Method + §6 Eval). Don't
duplicate it here. Locked kickoff decisions live in the project memory; the concrete M1 outcome
is summarized below.

---

## ⚠️ Collaboration model (read this first)
The **owner personally does the load-bearing ML** to learn it — here that means the **QLoRA
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
The `data/train/*.messages.jsonl` training files are **not** committed — they're regenerated here
by `prepare.py` (deterministic) in step 2.

*(No `.env` / API key is needed for M2 — training is fully local. The key only matters for the M3
frontier-panel eval, later, back on whichever box runs the API calls.)*

## 1. Environment (CUDA box only)
```bash
# uv + Python 3.12 (pinned; do NOT use system python)
uv sync --group train           # torch, transformers, trl, peft, accelerate, bitsandbytes, datasets
# Unsloth is installed separately per its own CUDA install matrix — see https://github.com/unslothai/unsloth
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The `train` dependency group is CUDA-only and was intentionally **not** installed on the Mac.

## 2. Regenerate the training data (already-built plumbing)
```bash
uv run python -m triage_distill.train.prepare
```
Writes to `data/train/`:
- `recipe_a.messages.jsonl` — 5,722 rows (multi-task: `classify` + `explain`, system prompt selects the task)
- `recipe_b.messages.jsonl` — 2,861 rows (single sequence: reason-then-label)
- `ablation.messages.jsonl` — 2,861 rows (label-only control)
- `val_eval.jsonl` — 2,381 rows (`{id, text, gold}`; the student must produce the category itself)

Each training line is `{"messages": [ {system}, {user: 'Ticket: "..."'}, {assistant: <JSON answer>} ]}` —
**model-agnostic**: apply the student tokenizer's chat template on this box. Switching students
(Qwen/Llama/Phi) is just a different chat template, no re-prep.

> **Student prompts are the owner's knob.** They live as `SYS_CLASSIFY / SYS_EXPLAIN / SYS_REASON`
> constants in `src/triage_distill/train/prepare.py` and are imported by the eval side, so
> **training and inference stay in lockstep** — change them in one place. (Can be promoted to a
> `prompts/student.md` design surface like `prompts/teacher.md` if you want to iterate.)

## 3. The M2 task — three QLoRA runs on Qwen3-4B
Train the **same** student + config three times, once per target file, so the only variable is the
rationale signal (that's what makes it a controlled experiment, not just "I fine-tuned a model"):

| Run | Train file | Purpose |
|---|---|---|
| **ablation** (do first) | `ablation.messages.jsonl` | baseline everything else must beat |
| **recipe B** | `recipe_b.messages.jsonl` | reason-then-label (interpretable) |
| **recipe A** | `recipe_a.messages.jsonl` | multi-task; label-only at inference (speed) |

**Decisions that are the owner's to make** (coaching ranges, not prescriptions):
- LoRA `r` (16–32) and `alpha` (~2×r); `target_modules` (attn q/k/v/o ± MLP gate/up/down)
- learning rate (1e-4–2e-4), epochs (1–3 — small data, watch val overfit), effective batch (bs × grad-accum to fill 24 GB)
- **loss-masking = completion-only** — compute loss on the assistant JSON only, not the prompt (TRL's `DataCollatorForCompletionOnlyLM` / SFTTrainer). This is *the* concept for "learns to produce JSON."
- `max_seq_len` — Recipe B is longest (rationale + label); size to its ~95th-pct token length.

**Base model:** `Qwen/Qwen3-4B` (Apache-2.0, native reasoning mode, first-class Unsloth support). The
choice is cheap to revisit — targets are model-agnostic. Optional: a 1.7B/4B/8B size-frontier.

## 4. Plumbing status — built vs to-build
- ✅ **Built + tested (on the Mac, no GPU):**
  - `triage_distill.train.prepare` — targets → `messages` training files (§2)
  - `triage_distill.eval.score` — macro-F1 (headline) + accuracy + per-class F1 + invalid-rate;
    invalid/hallucinated outputs count as **wrong**. Reused verbatim for the M3 frontier panel.
    Usage: `uv run python -m triage_distill.eval.score --preds preds.jsonl --gold data/train/val_eval.jsonl`
- 🚧 **To build here:**
  - `train.py` — **owner's surface.** Load 4-bit Qwen3-4B + LoRA, SFT on a chosen `*.messages.jsonl`,
    save the adapter. Agent scaffolds the runnable skeleton around the owner's hyperparameters.
  - **Inference runner** (plumbing) — load a trained adapter, run each `val_eval.jsonl` ticket with
    **constrained JSON decoding** (`triage_distill.schema.category_json_schema()` — enforce the 27-label
    enum so output is always parseable), emit `preds.jsonl` = `{id, pred}` for the scorer. For Recipe A,
    inference uses the `classify` task only (label-only → fast).

## 5. Eval loop + discipline
```
trained adapter ──► inference runner (val_eval) ──► preds.jsonl ──► score.py ──► macro-F1
```
- **Tune only on val.** The **test split is sacred** (`d7e24e9e…`, 4,761 rows) — scored exactly once
  in M3 alongside the frontier panel, never for training or tuning.
- **Pass gate (SPEC §1):** student macro-F1 **≥ 97.5% of K3's** macro-F1, **and** rationale students
  (A/B) **beat** the ablation. ⚠️ The 97.5% needs a **K3 baseline** scored with the *same* `score.py`
  on the same split — that's an M3 panel task and doesn't exist yet, so for now compare A/B/ablation to
  each other and track absolute val macro-F1.

## 6. Key facts (frozen)
- **Label space:** 27 intents, frozen at `artifacts/label_space.json` — nothing hardcodes the list; load via `schema.load_label_space()`.
- **Subsample:** 2,997 rows, 111/class, `subsample_hash=b8cc7b00ecb64726` (`artifacts/subsample_manifest.json`).
- **M1 labeling:** 0 errors, **95.5% gold-gate keep rate** (2,861 kept / 136 dropped), ~$15.86. Detail + the
  Bitext label-noise finding (`delivery_period→track_order`) are in the `m1-teacher-labeling` project memory.

## 7. Gotchas
- Don't `uv sync` the base group and expect training — you need `--group train` (CUDA).
- Don't train or tune on the **test** split. Val is for tuning; test is scored once (M3).
- Keep `SYS_*` prompts identical between `prepare.py` (training) and the inference runner (import them).
- Constrained decoding at inference isn't optional — the scorer marks invalid outputs wrong.
- `data/` is git-ignored **except** the tracked M1 artifacts (see §0); `data/train/` regenerates via `prepare.py`.

## 8. Suggested skills (invoke as needed on this box)
- **`run`** — when it's time to actually launch/smoke-test the trained student.
- **`claude-api`** — only for the later M3 panel/API pricing questions (not needed for local M2).
- **`handoff`** — to produce the next handoff (e.g. M2 → M3) when this box's work wraps.

## 9. File reference (don't duplicate — read these)
- `SPEC.md` — the plan (§5 Method, §6 Eval).
- `src/triage_distill/schema.py` — frozen label space, output schemas, `category_json_schema()` for constrained decoding.
- `src/triage_distill/train/prepare.py` — data prep + the student `SYS_*` prompts (owner's knob).
- `src/triage_distill/eval/score.py` — the scorer (students now, panel in M3).
- `data/label/targets/` — gold-gated A/B/ablation targets (committed). `data/label/labeled.jsonl` — full K3 output incl. archived reasoning traces.
- Project memory: `project-overview`, `collaboration-model`, `m1-teacher-labeling`.
