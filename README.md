# Triage-Distill

**Distill Kimi K3 into a small, ~zero-cost support-triage classifier that holds frontier-level accuracy at a fraction of the cost.**

> Thesis: you pay frontier prices for generality you don't need on a narrow task. Buy ≥97.5% of the accuracy for ~1% of the cost — and prove it with a controlled experiment.

See [`SPEC.md`](SPEC.md) for the full plan.

## Status
Phase 1 · M0 (scaffold) — data foundation + frozen label space in place.

## Task
Input: a support message. Output (one forward pass, constrained JSON):
```json
{ "category": "get_refund", "priority": "high", "escalate": true }
```
- `category` — Bitext intent (gold labels) → **Phase 1**
- `priority`, `escalate` — no public gold → **Phase 2** on owned DailyDles feedback

## Layout
- `src/triage_distill/schema.py` — frozen label space + output schemas (single source of truth)
- `src/triage_distill/data/` — download + deterministic stratified splits (**the test split is sacred**)
- `src/triage_distill/eval/` — model-agnostic metrics + cost accounting *(next)*
- `src/triage_distill/models/` — provider clients with constrained JSON decoding *(next)*
- `configs/` — pinned model ids (`models.yaml`) + pricing snapshot (`prices.yaml`)
- `artifacts/` — frozen `label_space.json` + `split_manifest.json` (committed; raw data is git-ignored)

## Setup
```bash
uv sync
cp .env.example .env      # add TEACHER_API_KEY + TEACHER_MODEL (Kimi K3 via Together/OpenRouter)
uv run python -m triage_distill.data.download   # -> artifacts/label_space.json
uv run python -m triage_distill.data.split      # -> data/splits/ + artifacts/split_manifest.json
```

## Compute
- **Dev / orchestration / API / eval / charts:** M3 MacBook.
- **Training (QLoRA):** RTX 4090 desktop (24 GB) — primary. Optional MLX LoRA run on the M3 as a "runs-anywhere" secondary.
