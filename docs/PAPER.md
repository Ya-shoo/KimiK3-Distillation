# Distilling a Frontier Model into a 4B Specialist for Intent Triage: A Controlled Study on Two Benchmarks

> **STATUS: DRAFT v1** (2026-07-30, mid-M2). Numbers marked ⟦pending⟧ await the
> multi-seed runs, the Bitext fixed-schedule re-run, and the M3 frontier panel.
> Interpretation-bearing passages are marked **[OWNER REVIEW]** — the A/B/ablation
> reading is the owner's call per the project's collaboration model.

## 1. Abstract

We distill Kimi K3 (a 2.8T-parameter frontier reasoning model) into Qwen3-4B, a
~1000× smaller student, for support-ticket / assistant intent triage, and evaluate
the *rationale-distillation* recipe ("Distilling Step-by-Step") with a controlled
experiment on two benchmarks: Bitext (27 intents, synthetic) and CLINC150 (150
intents + out-of-scope, real). A QLoRA student trained on ~2.8–5.7k gold-gated
teacher labels reaches **0.9886** val macro-F1 on Bitext and **0.9619** on
CLINC-151 at seed 42. The controlled comparison contradicts the popular recipe on
real data: on CLINC, **label-only training beats both rationale recipes** — even
when optimizer budget is step-matched — and the reason-then-label recipe
additionally collapses out-of-scope recall (0.33 vs 0.71), the metric a triage
deployment relies on for escalation. Retention against the teacher and cost
multiples are ⟦pending: M3 panel⟧. Seed-variance error bars: ⟦pending: 3-seed runs⟧.

## 2. Introduction

Narrow NLP tasks — ticket triage, intent routing, sentiment — are routinely served
by prompted frontier models at frontier prices. The generality being paid for is
mostly unused: the task has a fixed label space, bounded inputs, and no need for
open-ended generation. Distillation converts that unused generality into a
specialist: a small model, trained on the frontier model's outputs for this one
task, that runs on commodity hardware at ~zero marginal cost.

This report asks two questions. **(1) How much of the teacher's accuracy does a
4B student retain on intent triage?** (⟦pending⟧ — requires the K3 baseline scored
by identical code, M3.) **(2) Does distilling the teacher's *reasoning* — not just
its labels — measurably help,** as reported by Hsieh et al. (2023, "Distilling
Step-by-Step"), **once optimizer budget, LR schedule, and checkpoint selection are
properly controlled?** Our answer to (2), on a real benchmark, is currently *no* —
see §7.3 and §9.

## 3. Background

**Knowledge distillation** trains a student on a teacher's outputs. **Rationale
distillation** additionally trains on the teacher's explanation of each label,
on the theory that reasoning carries extra signal per example, improving accuracy
or data-efficiency. We test three recipes (§5): label-only (**ablation**),
reason-then-label in one sequence (**recipe B**), and a multi-task split where
classification and explanation are separate training tasks and inference uses only
the fast classification task (**recipe A**).

## 4. Task and data

Both benchmarks are single-label intent classification, output as constrained JSON.

| | Bitext-27 | CLINC-151 |
|---|---|---|
| Character | synthetic/templated customer support | real crowd-sourced assistant queries |
| Classes | 27 intents | 150 intents + `oos` (out-of-scope) |
| Splits | dedup-before-split, 0% leakage (hashes in `artifacts/`) | canonical CLINC splits (comparable to literature) |
| Teacher-labeled subsample | 2,997 rows (111/class) | 6,040 rows (40/class) |
| Val / test | 2,381 / 4,761 | 3,100 / 5,500 |

`oos` doubles as the deployment escalate signal: a query fitting no intent should
be routed to a human. We report oos recall/precision alongside macro-F1.

**Gold-gating.** The teacher labels each subsample row with a short rationale; rows
where the teacher's label disagrees with the dataset's gold label are dropped (the
rationale is presumed contaminated), keeping the gold label out of training either
way. Keep rates: **95.5%** (Bitext, 2,861 kept) and **94.4%** (CLINC, 5,704 kept).

**Finding: gold-gating can evacuate whole classes.** On CLINC, K3 labeled all 40
`reminder_update` rows as `reminder`, so the gate deleted the class from training
entirely; the student consequently scores **F1 = 0.0** on it (≈0.66 pt of macro-F1)
and drags down its sibling. `distance` (30/40 dropped) and `insurance_change`
(20/40) were partially evacuated the same way. The gate is not a free audit — it
converts teacher confusions into training-set class deletions. Mitigation
(back-filling gate-dropped classes with bare gold labels) is future work (§11).

## 5. Method

1. **Teacher labeling.** Kimi K3 (reasoning mode) labels each subsample row with
   `{evidence_to_intent, why_not_alternatives, category}` under a task-specific
   prompt (`prompts/teacher*.md`). Bitext: 0 API errors, ~$15.86. CLINC: 10 errors
   in 6,050 calls.
2. **Gold-gate** (§4), then render three training sets per benchmark:
   - **ablation**: `query → {category}` (2,861 / 5,704 rows)
   - **recipe B**: `query → {evidence, why_not, category}` one sequence (same rows)
   - **recipe A**: two rows per query — a `classify` task and an `explain` task
     selected by system prompt (5,722 / 11,408 rows); inference uses `classify` only.
3. **Student.** Qwen3-4B, QLoRA (4-bit base, LoRA r=16 α=32 on attention+MLP),
   lr 2e-4 linear, 3 epochs, effective batch 32, max_seq_len 256, completion-only
   loss, per-epoch checkpoints. One config across all recipes — the recipe is the
   only experimental variable. (Owner-set; coaching ranges in HANDOFF-M2-4090 §3.)
4. **Constrained inference.** Category is decoded against the frozen label-space
   enum (lm-format-enforcer); recipe B free-runs first and constrained-re-decodes
   schema failures. Invalid/hallucinated outputs are scored **wrong**.
5. **Selection discipline.** All tuning and epoch selection on val only; test is
   scored exactly once, at M3, alongside the frontier panel.

## 6. Setup

- Student training: single RTX 4090 (24 GB), Unsloth 2026.7.5 / TRL 0.24 /
  transformers 5.5, bf16. Environment specifics and Windows-stack workarounds
  (staged epochs, WDDM livelock mitigations, batch-layout note): `docs/ENV-4090-WINDOWS.md`.
- Teacher/panel pricing: pinned in `configs/prices.yaml` ⟦snapshot date at M3⟧.
- Seeds: 42 complete; 1337 and 2024 ⟦in flight⟧. Extra seeds are scored at the
  primary seed's best epoch (selection held fixed across seeds).
- Metrics: macro-F1 (headline), accuracy, per-class F1, invalid rate, oos
  recall/precision (CLINC). Scorer shared verbatim with the M3 panel.

## 7. Results

### 7.1 Headline (val macro-F1, seed 42; ⟦mean ± σ pending 3-seed runs⟧)

| Recipe | Bitext-27 | CLINC-151 | CLINC oos recall |
|---|---|---|---|
| Ablation (label-only) | 0.9800 @2 | **0.9619 @3** | 0.71 |
| Ablation, step-matched 6-epoch control | — | **0.9624 @3** | 0.72 |
| Recipe A (multi-task) | **0.9886 @2** † | 0.9574 @3 | 0.72 |
| Recipe B (reason-then-label) | 0.9757 @2 | 0.9316 @3 | **0.33** |

† Measured under an unintended warm-restart LR schedule that plausibly flatters
recipe A (see §9); a corrected re-run is ⟦in flight⟧.

Invalid outputs: zero everywhere except CLINC recipe B (8–18 of 3,100, scored wrong).

### 7.2 Retention vs teacher — ⟦pending M3: K3 baseline scored by identical code⟧

### 7.3 The ablation result **[OWNER REVIEW]**

On Bitext, recipe A beat the ablation by +0.86 pt. On CLINC the ordering inverts:
label-only wins by 0.45 pt over A and 3.0 pt over B. Three controls make the CLINC
result hard to dismiss:
- **Step-matching:** recipe A sees 2 rows/query, so at equal epochs it gets ~2×
  the optimizer steps. The 6-epoch ablation control (1,074 steps ≈ A's 1,071)
  still beats A (0.9624 vs 0.9574) — the rationale's multi-task signal does not
  merely underperform per-step; it underperforms at matched budget.
- **Schedule symmetry:** all CLINC runs share one corrected full-run LR schedule
  (§9 documents the Bitext asymmetry).
- **Selection symmetry:** every recipe gets per-epoch checkpoints and val-based
  best-epoch selection.

Candidate readings (to be adjudicated with seed error bars ⟦pending⟧): (i) the
Bitext advantage was partly artifactual (schedule + annealed checkpoints + step
budget); (ii) rationales help on templated data but add little on short, real
queries where the label is nearly surface-readable; (iii) rationale training
competes with label training for capacity at fixed LoRA rank.

### 7.4 The oos/escalate collapse under reason-then-label **[OWNER REVIEW]**

Recipe B's oos recall on CLINC is 0.28–0.35 across epochs vs 0.70–0.79 for both
label-only recipes, with oos precision comparable. Reading (i): generating the
rationale first commits the model to evidence-for-some-intent before the label
token is emitted — the rationale format has no natural "this fits nothing" path,
so probability mass flows to the nearest in-scope intent. For deployments where
escalation is the point of the system, this is an argument against
rationale-in-the-output regardless of macro-F1.

### 7.5 Learning curves and epoch knees

Bitext: all recipes peak at epoch 2 of 3 (val F1 dips at 3 while train loss keeps
falling — mild overfit past the knee). CLINC: all recipes still improve at epoch 3;
the 6-epoch control shows the knee is exactly 3 (0.9624 @3, then flat: 0.9596 /
0.9614 / 0.9617 at 4/5/6). Curves: `artifacts/*/eval/epoch_scores_*.json`.

### 7.6 Cost / speed — ⟦pending M3 measured tokens + pinned prices⟧
Recipe A's design goal survives regardless of §7.3: label-only inference emits
~10 output tokens per query (24 t/s batched classify on one 4090 vs 3.9 t/s for
reason-mode — a 6× serving-cost difference against recipe B).

## 8. Cost analysis — ⟦pending M3⟧ (savings-at-scale chart per PAPER-OUTLINE Part C)

## 9. Threats to validity

- **Bitext is synthetic**; its near-ceiling scores overstate real-world accuracy.
  Mitigated by CLINC (real) + oos.
- **The Bitext recipe-A number carries three asymmetries**, found by post-run
  audit: (1) a per-stage LR-scheduler rebuild gave A (the only staged run) a
  warm-restart sawtooth others didn't get; (2) A's per-epoch checkpoints were
  LR-annealed while others' were mid-decay snapshots — flattering A's best-epoch
  selection; (3) A had 2× optimizer steps at equal epochs. All three are corrected
  or controlled in the CLINC track; a corrected Bitext re-run is ⟦in flight⟧.
  The audit trail (trainer-log LR curves) is in `artifacts/eval/`.
- **Label noise:** Bitext `delivery_period↔track_order` caps ablation/B per-class
  F1; CLINC `reminder`/`reminder_update` (and kin) is severe enough to evacuate
  classes through the gold-gate (§4).
- **Single seed** for every number above — deltas of ≤0.5 pt are not yet
  distinguishable from seed noise ⟦3-seed σ pending⟧.
- **Teacher prompt fairness across benchmarks:** the student prompts are identical
  across benchmarks by design (frozen recipe); the teacher prompt was adapted per
  benchmark (label glosses, oos instruction).
- **Price drift** ⟦pinned at M3⟧.

## 10. Overfitting discipline

Train/val/test with test-scored-once (M3). Val-only tuning; per-epoch checkpoints;
knees measured per benchmark (§7.5); train-loss-vs-val-F1 gap logged per epoch in
`artifacts/*/eval/findings.json`. The ablation memorizes its label-only train set
(loss ≈ 0 by epoch 2) while val F1 still improves — memorization of a small
per-class sample is not, by itself, the failure mode here.

## 11. Limitations and future work

- **K3 baseline + frontier panel (M3)** — the retention claim, the money chart.
- **Seed variance + corrected Bitext re-run** — ⟦in flight tonight⟧.
- **Back-fill gate-evacuated classes** with bare gold labels (recovers
  `reminder_update` without trusting contaminated rationales).
- **Data-efficiency curves** (10/20/30/40 per class): the strongest remaining case
  for rationales is fewer-examples-needed, untested here.
- Single student size; a 1.7B/4B/8B frontier is cheap to add (targets are
  model-agnostic).
- Phase 2: priority/escalate fields on owned data; OOD probe of Bitext-trained
  students on real phrasings.

## 12. Conclusion — ⟦drafted after M3; the honest arc so far:⟧ **[OWNER REVIEW]**
A 4B QLoRA student trained on a few thousand gold-gated frontier labels is a
strong intent-triage specialist on both a synthetic and a real benchmark. The
fashionable part of the recipe — distilling rationales — did not survive contact
with controls on the real benchmark, and actively harmed the escalation signal in
its reason-then-label form. Distillation earns its keep; the rationale garnish, at
least at this scale and data regime, has yet to.

## Appendix A. Reproducibility

Frozen label spaces, split manifests and hashes, subsample manifests:
`artifacts/`. Teacher outputs incl. archived reasoning: `data/*/label/labeled.jsonl`
(committed). Exact commands: `README`/`HANDOFF-M2-4090.md`; per-run configs incl.
git commit, seed, knobs: `artifacts/*/eval/run_config_*.json`. Environment lock:
`uv.lock` + `docs/ENV-4090-WINDOWS.md`. Adjustment changelog + caveats:
`artifacts/*/eval/findings.json`.
