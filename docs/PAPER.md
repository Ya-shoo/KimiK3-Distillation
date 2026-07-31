# Distilling a Frontier Model into a 4B Specialist for Intent Triage: A Controlled Study on Two Benchmarks

> **STATUS: DRAFT v2** (2026-07-30, M2 complete: 17/17 runs, 3 seeds/recipe both
> benchmarks; **M3 frontier panel in** — §7.2/§8: 5 models on the test split, macro-F1
> + cost). One number remains ⟦pending⟧: the **student's test macro-F1** — the 4090
> selected epochs on val (correct discipline); scoring the best-epoch adapters on the
> sacred test split, once, resolves the retention multiple (§7.2) and the money chart
> (§8). Interpretation-bearing passages are marked **[OWNER REVIEW]** — the A/B/ablation
> reading is the owner's call per the project's collaboration model.

## 1. Abstract

We distill Kimi K3 (a 2.8T-parameter frontier reasoning model) into Qwen3-4B, a
~1000× smaller student, for support-ticket / assistant intent triage, and evaluate
the *rationale-distillation* recipe ("Distilling Step-by-Step") with a controlled
experiment on two benchmarks: Bitext (27 intents, synthetic) and CLINC150 (150
intents + out-of-scope, real). A QLoRA student trained on ~2.8–5.7k gold-gated
teacher labels reaches **0.9915 ± 0.0006** val macro-F1 on Bitext and
**0.9611 ± 0.0008** on CLINC-151 (3 seeds). The controlled comparison yields a
mirror-image result: the multi-task rationale recipe **wins on the synthetic
benchmark (+1.3 pt, ~8σ) and loses on the real one (−0.5 pt, ~4σ)** — the loss
persisting when optimizer budget is step-matched — and the reason-then-label
recipe collapses out-of-scope recall (0.30 vs 0.69), the metric a triage
deployment relies on for escalation. Rationale distillation's benefit is not a
property of the method alone but of the data distribution it meets. A five-model
frontier panel scored on the test split places the teacher at 95.0/90.8 macro-F1
(Bitext/CLINC) — and a cheaper generalist (Gemini 3 Flash) *beats* the teacher on the
real benchmark (93.5 vs 90.8) at ~6× lower cost; the student's marginal inference cost
is ~$0 (owned GPU) vs $190–$2,200 per million tickets/month for the cloud panel. The one
open number is the student's own test macro-F1, scored once with the panel, which sets
the final retention multiple.

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
properly controlled?** Our answer to (2) is *it depends on the data*: a robust yes
(~8σ) on the synthetic benchmark, a robust no (~4σ) on the real one — see §7.3.

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
- Seeds: 42, 1337, 2024 (complete). Extra seeds are scored at the primary seed's
  best epoch (selection held fixed across seeds).
- Metrics: macro-F1 (headline), accuracy, per-class F1, invalid rate, oos
  recall/precision (CLINC). Scorer shared verbatim with the M3 panel.

## 7. Results

### 7.1 Headline (val macro-F1, mean ± sample σ over seeds {42, 1337, 2024})

| Recipe | Bitext-27 (best @2) | CLINC-151 (best @3) | CLINC oos recall (mean) |
|---|---|---|---|
| Ablation (label-only) | 0.9788 ± 0.0013 | **0.9611 ± 0.0008** | 0.69 |
| Ablation, step-matched 6-epoch control | — | **0.9624** (n=1) | 0.72 |
| Recipe A (multi-task) | **0.9915 ± 0.0006** | 0.9563 ± 0.0015 | 0.72 |
| Recipe B (reason-then-label) | 0.9779 ± 0.0019 | 0.9317 ± 0.0013 | **0.30** |

Invalid outputs: zero everywhere except CLINC recipe B (7–18 of 3,100 per seed,
scored wrong) — all are token-budget exhaustion: the generated rationale overruns
the 512-token cap before the `category` field is emitted, so even constrained
re-decoding cannot rescue the row (§7.4).

**Zero-shot reference:** the same base model without fine-tuning, same constrained
decoding, scores **0.3148** (Bitext) and **0.2693** (CLINC; oos recall 0.00) —
distillation contributes ~+66 macro-F1 points on both benchmarks
(`artifacts/*/eval/zeroshot_base.json`).

### 7.2 The frontier panel (test macro-F1), and retention vs teacher

We score a two-tier panel of prompted frontier/efficient models on the **held-out test
split** (Bitext 4,761 / CLINC 5,500), label-only, with the *same* label glosses the
teacher saw and the *same* `score.py` used for the student. All five produced **0%
invalid** output; reasoning is disabled where the provider allows it, so this is the
"just prompt a model for the label" regime a team actually compares against.

| Model (tier) | Bitext-27 | CLINC-151 | Cost /1k tickets† |
|---|---|---|---|
| **Kimi K3 — teacher** (flagship) | **95.0** | 90.8 | $2.20 |
| Gemini 3 Flash (flagship) | 93.6 | **93.5** | $0.39 |
| GPT-5.6 Luna (efficient) | 93.6 | 89.6 | $0.35 |
| DeepSeek 3.2 (efficient) | 93.2 | 86.6 | $0.19 |
| Haiku 4.5 (efficient) | 87.9 | 88.3 | $0.79 |

† List API prices (`configs/prices.yaml`, snapshot 2026-07-30) × measured mean tokens
(§8). Panel run cost: $22 on OpenRouter. The two priciest flagships — GPT-5.6 Sol ($5/$30)
and Fable 5 ($10/$50) — were scoped out on cost.

**Panel finding — the teacher is not the ceiling on real data.** On CLINC, **Gemini 3
Flash (93.5) beats the K3 teacher (90.8) by 2.7 pt at ~6× lower cost.** K3 leads on the
synthetic Bitext (95.0) but trails on the real, 151-class benchmark — a second mirror to
the §7.3 recipe mirror. "Retention of K3" is therefore a floor, not a ceiling, for what a
specialist could reach on the real task.

**Retention (student vs K3 on test).** K3's test macro-F1 — the denominator for the
≥97.5% retention target (SPEC §1) — is **95.0 (Bitext) / 90.8 (CLINC)**. The numerator,
the student's own **test** macro-F1, is the one number still in flight: epoch selection
was on *val* (correct discipline) and the sacred test set is scored exactly once, for
student and panel together. The panel half is done (above); the student half — best-epoch
adapters on `test_eval.jsonl` — is the final measurement, after which the retention ratio
and the money chart (§8) resolve. For scale: the strongest students already reach **0.9915
± 0.0006 (Bitext, recipe A)** and **0.9611 ± 0.0008 (CLINC, ablation; 0.9715 with
gate-backfill)** on *val* — at or above K3's *test* number on the same metric, which is
exactly why a same-split test read is what makes the retention claim rigorous rather than
suggestive.

### 7.3 The ablation result: a benchmark-dependent mirror **[OWNER REVIEW]**

Recipe A beats the ablation on Bitext by **+1.27 pt at ~8σ** (0.9915 ± 0.0006 vs
0.9788 ± 0.0013) and loses to it on CLINC by **−0.48 pt at ~4σ** (0.9563 ± 0.0015
vs 0.9611 ± 0.0008). Both effects are far outside seed noise; recipe B never beats
the ablation anywhere. Controls that make the inversion hard to dismiss:
- **Step-matching (CLINC):** recipe A sees 2 rows/query, so at equal epochs it
  gets ~2× the optimizer steps. The 6-epoch ablation control (1,074 steps ≈ A's
  1,071) still beats A (0.9624 vs 0.9563 ± 0.0015) — the multi-task signal
  underperforms at matched budget, not merely per step.
- **Schedule/selection symmetry:** all seed-statistics runs share one corrected
  full-run LR schedule, per-epoch checkpoints, and val-based selection. The
  suspected Bitext schedule artifact was tested directly and refuted (§9).

Candidate readings for the mirror: (i) Bitext's templated inputs reward the
richer training signal — rationales effectively teach the template grammar —
while CLINC's short real queries carry the label near the surface, so the
rationale adds gradient noise, not information; (ii) at fixed LoRA capacity
(r=16), rationale learning competes with label learning, and only pays where the
task is compositional enough to amortize it; (iii) synthetic benchmarks can
overstate rationale-distillation gains generally — a caution for the literature,
since method papers frequently evaluate on clean/templated data.

### 7.4 The oos/escalate collapse under reason-then-label **[OWNER REVIEW]**

Recipe B's oos recall on CLINC is **0.30 mean (0.26–0.35) across all three seeds**
vs 0.69–0.72 for both label-only-at-inference recipes, with precision comparable.
Reading: generating the rationale first commits the model to
evidence-for-some-intent before the label token is emitted — the rationale format
has no natural "this fits nothing" path, so probability mass flows to the nearest
in-scope intent. The same mechanism produces recipe B's only-in-the-project
invalid outputs: rationales that ramble past the generation budget before the
label appears. For deployments where escalation is the point of the system, these
are arguments against rationale-in-the-output regardless of macro-F1.

### 7.5 Learning curves and epoch knees

Bitext: all recipes peak at epoch 2 of 3 (val F1 dips at 3 while train loss keeps
falling — mild overfit past the knee). CLINC: all recipes still improve at epoch 3;
the 6-epoch control shows the knee is exactly 3 (0.9624 @3, then flat: 0.9596 /
0.9614 / 0.9617 at 4/5/6). Curves: `artifacts/*/eval/epoch_scores_*.json`.

### 7.6 Cost / speed — ⟦pending M3 measured tokens + pinned prices⟧
Recipe A's design goal survives regardless of §7.3: label-only inference emits
~10 output tokens per query (24 t/s batched classify on one 4090 vs 3.9 t/s for
reason-mode — a 6× serving-cost difference against recipe B).

## 8. Cost analysis

Cost per ticket is `mean_input_tokens × input_price + mean_output_tokens × output_price`
at the pinned list prices (`configs/prices.yaml`, snapshot 2026-07-30). Inputs are ~660–690
tokens (the shared label-glossary prompt); label-only outputs are ~10–15 tokens, so **input
dominates** — the cost axis is essentially "price per prompt."

**Savings at scale (1M tickets/month), list API prices:**

| Model | $/1k | $/1M tickets/mo |
|---|---|---|
| Kimi K3 (teacher) | $2.20 | **$2,200** |
| Haiku 4.5 | $0.79 | $790 |
| Gemini 3 Flash | $0.39 | $390 |
| GPT-5.6 Luna | $0.35 | $350 |
| DeepSeek 3.2 | $0.19 | $190 |
| **Student (Qwen3-4B, local)** | **≈$0** | **≈$0 (electricity)** |

The student's marginal inference cost is electricity: recipe A emits ~10 output tokens per
query and batched `classify` runs at ~24 tok/s on one owned 4090, so 1M tickets/month (~0.4
queries/s sustained) is served comfortably by hardware the project already owns. Against the
**cheapest cloud option (DeepSeek, $190/mo)** the specialist saves ~$190/mo per million
tickets; against the **teacher ($2,200/mo)**, ~$2,200/mo — before the ~6× serving-speed edge
of label-only inference over reason-mode (§7.6). Break-even on the one-time ~$16 (Bitext) /
~$35 (CLINC) teacher-labeling spend plus a few GPU-hours is days, not months, at any
realistic volume.

**The hero figure (`money_chart`: cost/1k on a log x-axis vs test macro-F1)** places every
panel model from the §7.2 table; the student's point — high-and-left, at ~$0 and its test
macro-F1 — lands once the 4090 test scores arrive. The panel points already show the shape:
flagships high-and-right (K3 $2.2, accurate), efficient tier clustered mid-left
(DeepSeek/Luna/Gemini $0.2–0.4), and a wide-open cheap-and-accurate corner for the specialist.

## 9. Threats to validity

- **Bitext is synthetic**; its near-ceiling scores overstate real-world accuracy.
  Mitigated by CLINC (real) + oos.
- **A suspected artifact was found, tested, and refuted.** A post-run audit
  flagged three asymmetries favoring the original Bitext recipe-A run (a
  per-stage LR-scheduler rebuild that produced a warm-restart sawtooth; annealed
  checkpoints vs others' mid-decay snapshots; 2× optimizer steps at equal
  epochs). The corrected re-run scored *higher* (0.9920 vs 0.9886), refuting the
  inflation hypothesis; all reported seed statistics use fixed-schedule runs
  only, and the legacy run is retained in the artifacts for the record. Audit
  trail (trainer-log LR curves): `artifacts/eval/`.
- **A mid-experiment memory-layout change** (per-device batch 32×1 → 16×2 after a
  Windows VRAM livelock; identical effective batch, step count, and schedule) is
  recorded per run in `run_config_*.json`. Gradient-reduction order is the only
  difference; observed seed σ (≤0.002) bounds its effect.
- **Label noise:** Bitext `delivery_period↔track_order` caps ablation/B per-class
  F1; CLINC `reminder`/`reminder_update` (and kin) is severe enough to evacuate
  classes through the gold-gate (§4).
- **Seed count is 3** — adequate for the large effects reported (4–8σ), thin for
  sub-0.2-pt comparisons (e.g. ablation vs its 6-epoch control, or ablation vs
  recipe B on Bitext, which are statistical ties at this n).
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
- **Back-fill gate-evacuated classes** with bare gold labels (recovers
  `reminder_update` without trusting contaminated rationales).
- **Data-efficiency curves** (10/20/30/40 per class): the strongest remaining case
  for rationales is fewer-examples-needed, untested here.
- Single student size; a 1.7B/4B/8B frontier is cheap to add (targets are
  model-agnostic).
- Phase 2: priority/escalate fields on owned data; OOD probe of Bitext-trained
  students on real phrasings.

## 12. Conclusion — ⟦finalized after M3; the honest arc so far:⟧ **[OWNER REVIEW]**
A 4B QLoRA student trained on a few thousand gold-gated frontier labels is a
strong intent-triage specialist on both a synthetic (0.99) and a real (0.96)
benchmark. The fashionable part of the recipe — distilling rationales — turned
out to be a property of the data, not the method: a large, real win on templated
data that inverts into a significant loss on real queries, with the
reason-then-label variant additionally destroying the escalation signal.
Distillation earns its keep; whether the rationale garnish does depends on
whether your data looks like a template or like people.

## Appendix A. Reproducibility

Frozen label spaces, split manifests and hashes, subsample manifests:
`artifacts/`. Teacher outputs incl. archived reasoning: `data/*/label/labeled.jsonl`
(committed). Exact commands: `README`/`HANDOFF-M2-4090.md`; per-run configs incl.
git commit, seed, knobs: `artifacts/*/eval/run_config_*.json`. Environment lock:
`uv.lock` + `docs/ENV-4090-WINDOWS.md`. Adjustment changelog + caveats:
`artifacts/*/eval/findings.json`.
