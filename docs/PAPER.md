# Distilling a Frontier Model into a 4B Specialist for Intent Triage: A Controlled Study on Two Benchmarks

> **STATUS: FINAL v3** (2026-07-31). M1–M3 complete: 17/17 training runs (3 seeds ×
> 3 recipes × 2 benchmarks + controls), five-model frontier panel scored on the
> held-out test splits, and all six student test passes scored - each exactly once.
> Every number in this document traces to a committed artifact under `artifacts/`;
> the six headline test scores were independently recomputed from the raw
> prediction files (`test_*_preds.jsonl`) and match the scorer output digit-for-digit.
> The post-run concern list (`docs/TEST-EVAL-CONCERNS.md`) was worked through:
> its two post-hoc analyses are folded into Section 7.4 and Section 7.6, and its framing
> caveats into Section 7.5 and Section 9.
> Companion reads: [`PAPER-UNDERGRAD.md`](PAPER-UNDERGRAD.md) (assumes intro-ML),
> [`PAPER-ELI5.md`](PAPER-ELI5.md) (assumes nothing).

## 1. Abstract

We distill Kimi K3 (a 2.8T-parameter frontier reasoning model) into Qwen3-4B, a
~1000× smaller student, for support-ticket / assistant intent triage, and test the
*rationale-distillation* recipe ("Distilling Step-by-Step") with a controlled
experiment on two benchmarks: Bitext (27 intents, synthetic) and CLINC150 (150
intents + out-of-scope, real). On the held-out test splits - scored exactly once -
the student does not merely retain the teacher's accuracy; it **exceeds it**:
**0.9923** macro-F1 on Bitext (**104.5%** of the teacher's 0.9498, first place among
all eight systems scored, including the teacher) and **0.9220** on CLINC-151
(**101.6%** of the teacher's 0.9078, second place behind only Gemini 3 Flash).
Five of six student passes clear the pre-registered ≥97.5% retention gate; the sole
failure is the reason-then-label recipe on real data (96.1%), which also collapses
out-of-scope recall to 0.26 and costs ~20× more to serve. The controlled ablation
yields a mirror: training on the teacher's rationales **wins on the synthetic
benchmark (+1.3 pt, ~8σ) and loses on the real one (−0.5 pt, ~4σ)** - the loss
persisting under a step-matched optimizer-budget control - and the synthetic win
localizes almost entirely to a single confusable label pair. Rationale
distillation's benefit is a property of the data distribution, not of the method.
The student serves 1M tickets for ~$2–3 of electricity on the owned GPU ($7–14
renting the same GPU; ≲$25 even via a hosted API at our panel's cheapest pinned
per-token rate) versus $190–$2,260 for the cloud panel; total one-time project
spend (teacher labeling + panel) was under $100.

## 2. Introduction

Narrow NLP tasks - ticket triage, intent routing, sentiment - are routinely served
by prompted frontier models at frontier prices. The generality being paid for is
mostly unused: the task has a fixed label space, bounded inputs, and no need for
open-ended generation. Distillation converts that unused generality into a
specialist: a small model, trained on the frontier model's outputs for this one
task, that runs on commodity hardware at ~zero marginal cost.

This report asks two questions. **(1) How much of the teacher's accuracy does a
4B student retain on intent triage?** Answer: all of it, and then some - 103–104%
on the synthetic benchmark, 100–102% on the real one (Section 7.2), with the caveat that
"retention above 100%" says as much about the baseline (a *prompted* generalist)
as about the student (Section 9). **(2) Does distilling the teacher's *reasoning* - not
just its labels - measurably help,** as reported by Hsieh et al. (2023,
"Distilling Step-by-Step"), **once optimizer budget, LR schedule, and checkpoint
selection are properly controlled?** Answer: *it depends on the data* - a robust
yes (~8σ) on the synthetic benchmark, a robust no (~4σ) on the real one, with the
ordering replicating exactly on the test splits (Section 7.3).

## 3. Background

**Knowledge distillation** trains a student on a teacher's outputs. **Rationale
distillation** additionally trains on the teacher's explanation of each label, on
the theory that reasoning carries extra signal per example, improving accuracy or
data-efficiency. We test three recipes (Section 5): label-only (**ablation**),
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
| Test oos share | - | 18.2% (1,000/5,500) vs 3.2% on val |

`oos` doubles as the deployment escalate signal: a query fitting no intent should
be routed to a human. We report oos recall/precision alongside macro-F1. Note the
val/test **composition shift**: test is 18.2% oos vs 3.2% on val - it matters in Section 7.5.

**Gold-gating.** The teacher labels each subsample row with a short rationale; rows
where the teacher's label disagrees with the dataset's gold label are dropped (the
rationale is presumed contaminated), keeping the gold label out of training either
way. Keep rates: **95.5%** (Bitext, 2,861 kept) and **94.4%** (CLINC, 5,704 kept).

**Finding: gold-gating can evacuate whole classes.** On CLINC, K3 labeled all 40
`reminder_update` rows as `reminder`, so the gate deleted the class from training
entirely; the student consequently scores **F1 = 0.0** on it (≈0.66 pt of macro-F1)
and drags down its sibling. `distance` (30/40 dropped) and `insurance_change`
(20/40) were partially evacuated the same way. The gate is not a free audit - it
converts teacher confusions into training-set class deletions. The mitigation
(back-filling gate-dropped classes with bare gold labels) was implemented and
tested: it recovers `reminder_update` to F1 0.95 and lifts val macro-F1 to 0.9715
(Section 7.6).

## 5. Method

1. **Teacher labeling.** Kimi K3 (reasoning mode) labels each subsample row with
   `{evidence_to_intent, why_not_alternatives, category}` under a task-specific
   prompt (`prompts/teacher*.md`). Bitext: 0 API errors, $15.86. CLINC: 10 errors
   in 6,050 calls, ~$35.
2. **Gold-gate** (Section 4), then render three training sets per benchmark:
   - **ablation**: `query → {category}` (2,861 / 5,704 rows)
   - **recipe B**: `query → {evidence, why_not, category}` one sequence (same rows)
   - **recipe A**: two rows per query - a `classify` task and an `explain` task
     selected by system prompt (5,722 / 11,408 rows); inference uses `classify` only.
3. **Student.** Qwen3-4B, QLoRA (4-bit base, LoRA r=16 α=32 on attention+MLP),
   lr 2e-4 linear, 3 epochs, effective batch 32, max_seq_len 256, completion-only
   loss, per-epoch checkpoints. One config across all recipes - the recipe is the
   only experimental variable.
4. **Constrained inference.** Category is decoded against the frozen label-space
   enum (lm-format-enforcer); recipe B free-runs first and constrained-re-decodes
   schema failures. Invalid/hallucinated outputs are scored **wrong**.
5. **Selection discipline.** All tuning and epoch selection on val only. The test
   split was scored exactly once per system - the s42 checkpoint at its best-*val*
   epoch, deterministic decode - alongside the frontier panel, with a shared scorer.

## 6. Setup

- Student training: single RTX 4090 (24 GB), Unsloth 2026.7.5 / TRL 0.24 /
  transformers 5.5, bf16. Environment specifics and Windows-stack workarounds
  (staged epochs, WDDM livelock mitigations, batch-layout note): `docs/ENV-4090-WINDOWS.md`.
- Teacher/panel pricing pinned in `configs/prices.yaml` (OpenRouter list,
  snapshot 2026-07-28/30); panel evaluated 2026-07-30/31.
- Seeds: 42, 1337, 2024 (complete). Extra seeds are scored at the primary seed's
  best epoch (selection held fixed across seeds).
- Metrics: macro-F1 (headline), accuracy, per-class F1, invalid rate, oos
  recall/precision (CLINC). Scorer shared verbatim between students and panel;
  headline test numbers independently recomputed from raw predictions.

## 7. Results

### 7.1 Validation (macro-F1, mean ± sample σ over seeds {42, 1337, 2024})

| Recipe | Bitext-27 (best @2) | CLINC-151 (best @3) | CLINC oos recall (mean) |
|---|---|---|---|
| Ablation (label-only) | 0.9788 ± 0.0013 | **0.9611 ± 0.0008** | 0.69 |
| Ablation, step-matched 6-epoch control | - | **0.9624** (n=1) | 0.72 |
| Ablation + gate back-fill (Section 7.6) | - | **0.9715** (n=1) | 0.72 |
| Recipe A (multi-task) | **0.9915 ± 0.0006** | 0.9563 ± 0.0015 | 0.72 |
| Recipe B (reason-then-label) | 0.9779 ± 0.0019 | 0.9317 ± 0.0013 | **0.30** |

Invalid outputs: zero everywhere except CLINC recipe B (7–18 of 3,100 per seed,
scored wrong) - all token-budget exhaustion: the rationale overruns the cap before
`category` is emitted (Section 7.4).

**Zero-shot reference (val split):** the same base model without fine-tuning, same
constrained decoding, scores **0.3148** (Bitext) and **0.2693** (CLINC; oos recall
0.00) on val - never run on test, so the test tables exclude it.
Distillation contributes ~+66 macro-F1 points on both benchmarks - the fine-tuning
is the entire product (`artifacts/*/eval/zeroshot_base.json`).

### 7.2 Test - the leaderboard and the retention verdict

Held-out test splits (Bitext 4,761 / CLINC 5,500), every system scored once by the
same scorer: the three students (s42, best-val epoch) and a two-tier panel of five
prompted cloud models, label-only prompt with identical label glosses, reasoning
disabled, 0% invalid output from every panel model.

**Combined test leaderboard (macro-F1):**

| System | Bitext-27 | CLINC-151 | $/1k tickets† |
|---|---|---|---|
| **Student - recipe A** | **0.9923** ① | 0.9103 ③ | ≤$0.03 |
| **Student - ablation** | 0.9848 ② | **0.9220** ② | ≤$0.03 |
| **Student - recipe B** | 0.9801 ③ | 0.8724 ⑦ | ≤$0.03 |
| Kimi K3 - teacher (flagship) | 0.9498 ④ | 0.9078 ④ | $2.20 |
| Gemini 3 Flash (flagship) | 0.9364 ⑤ | **0.9345** ① | $0.39 |
| GPT-5.6 Luna (efficient) | 0.9360 ⑥ | 0.8957 ⑤ | $0.35 |
| DeepSeek 3.2 (efficient) | 0.9324 ⑦ | 0.8665 ⑧ | $0.19 |
| Haiku 4.5 (efficient) | 0.8788 ⑧ | 0.8832 ⑥ | $0.79 |

† List API prices × measured mean tokens (Section 8). The student figure spans serving
scenarios - ~$0.002/1k electricity on the owned 4090 up to ~$0.03/1k as a
hosted-API ceiling (Section 8) - and is never literally zero.

**Retention vs the teacher (the pre-registered ≥97.5% gate, SPEC Section 1):**

| Recipe | Bitext test (ret.) | CLINC test (ret.) | Gate |
|---|---|---|---|
| Ablation | 0.9848 (**103.7%**) | 0.9220 (**101.6%**) | pass / pass |
| Recipe A | 0.9923 (**104.5%**) | 0.9103 (**100.3%**) | pass / pass |
| Recipe B | 0.9801 (**103.2%**) | 0.8724 (**96.1%**) | pass / **FAIL** |

Findings:

- **Retention was a floor, not a ceiling.** Five of six passes exceed the teacher
  outright. The best student per benchmark beats K3 by +4.3 pt (Bitext) and
  +1.4 pt (CLINC). Section 9 explains why this is not paradoxical - the students'
  supervision was gold-gated, and the teacher competes as a *prompted* generalist.
- **The recipe ordering replicates on test with no generalization surprise.**
  Bitext: A > ablation > B, every recipe scoring slightly *above* its val
  seed-mean (+0.1 to +0.6 pt). CLINC: ablation > A > B, exactly as on val. The
  benchmark-mirror (Section 7.3) is a test-set fact, not a val artifact.
- **The teacher is not the accuracy ceiling on real data.** Gemini 3 Flash
  (93.5) beats K3 (90.8) on CLINC at ~6× lower cost, while K3 leads the panel on
  synthetic Bitext - the panel mirrors the recipe mirror.
- **Recipe B fails everything at once**: the only retention-gate failure (96.1%),
  the only invalid outputs (71/5,500; 1.3%, counted as errors by the scorer), the
  worst oos collapse (Section 7.4), and a ~20× wall-clock premium *on our eval harness*
  (125.6 vs ~6.5 min per test pass). The harness-independent part of that premium
  is the token asymmetry (~512 vs ~32 max new tokens per query); a tuned serving
  stack would compress the wall-clock gap, not the token gap.

### 7.3 The ablation result: a benchmark-dependent mirror

Recipe A beats the ablation on Bitext by **+1.27 pt at ~8σ** on val (0.9915 ±
0.0006 vs 0.9788 ± 0.0013; +0.75 pt on test) and loses to it on CLINC by
**−0.48 pt at ~4σ** (0.9563 ± 0.0015 vs 0.9611 ± 0.0008; −1.17 pt on test). Both
effects are far outside seed noise; recipe B never beats the ablation anywhere.
Controls that make the inversion hard to dismiss:

- **Step-matching (CLINC):** recipe A sees 2 rows/query, so at equal epochs it
  gets ~2× the optimizer steps. The 6-epoch ablation control (1,074 steps ≈ A's
  1,071) still beats A (0.9624 vs 0.9563 ± 0.0015) - the multi-task signal
  underperforms at matched budget, not merely per step.
- **Schedule/selection symmetry:** all seed-statistics runs share one corrected
  full-run LR schedule, per-epoch checkpoints, and val-based selection. The
  suspected Bitext schedule artifact was tested directly and refuted (Section 9).

**Where the Bitext win actually lives.** Per-class decomposition of the test gap:
recipe A's advantage concentrates almost entirely in the benchmark's one known
confusable pair, `delivery_period` ↔ `track_order` - the pair on which the
*teacher itself* scores worst (0.70/0.77). Recipe A: **0.9975 / 0.9900**; ablation:
0.8678 / 0.8671; recipe B: 0.8883 / 0.8843. Those two classes alone contribute
~0.94 pt of the 0.75 pt total gap (the remaining classes net slightly *against*
A). The rationale signal did not diffusely improve classification; it taught the
student the boundary convention between two overlapping intents.

Candidate readings for the mirror (interpretation, not measurement): (i) Bitext's
templated inputs reward the richer training signal - rationales effectively teach
the template grammar, including the boundary between near-duplicate intents -
while CLINC's short real queries carry the label near the surface, so the
rationale adds gradient noise, not information; (ii) at fixed LoRA capacity
(r=16), rationale learning competes with label learning, and only pays where the
task is compositional enough to amortize it; (iii) synthetic benchmarks can
overstate rationale-distillation gains generally - a caution for the literature,
since method papers frequently evaluate on clean/templated data.

### 7.4 The oos/escalate collapse under reason-then-label

Recipe B's oos recall on CLINC is **0.30 mean (0.26–0.35) across all seeds on
val** and **0.264 on test** (oos F1 0.41), vs 0.59–0.69 test recall for the two
label-only-at-inference recipes, with precision comparable (~0.97 everywhere).
Reading: generating the rationale first commits the model to
evidence-for-some-intent before the label token is emitted - the rationale format
has no natural "this fits nothing" path, so probability mass flows to the nearest
in-scope intent.

**Post-hoc decomposition (from the committed test predictions): the format
failure and the oos failure are one phenomenon.** 882 of 5,500 test rows (16%)
failed schema validation on the free-decode pass and went through constrained
retry - and **752 of those 882 are gold-oos rows** (85%; 75.2% of all oos).
Clean-pass rows: accuracy 0.909, zero invalids. Retried rows: accuracy 0.30. The
71 invalid outputs are all retry-path rows whose 512-token decode budget
truncates before the label field even under constraint (an intrinsic ~7.7%
retry-path rate; an LMFE MemoryError late in the run was checked and cleared -
the budget, not the error, is the mechanism), and 66 of the 71 are gold-oos.
The model does register that these inputs fit nothing - but the signal escapes
as *malformed output* rather than as the `oos` label: "needed retry" detects oos
with recall 0.752 / precision 0.853, far better than the model's own predictions
(recall 0.264), and even 206 of its 273 explicit oos calls surface via the retry
path. Recipe B's in-scope labeling is also genuinely weaker (in-scope-conditional
macro-F1 0.931 vs the ablation's 0.958), so the gate failure is not format alone -
but the escalation collapse is largely the format channel swallowing the
escalation signal. For deployments where escalation is the point of the system,
this disqualifies rationale-in-the-output regardless of macro-F1 (or demands the
inverted trick: treat schema failure itself as the escalate flag).

**But note the panel's counterpoint (Section 7.6 table): every prompted flagship beats
every student on oos F1.** Teacher 0.878, Gemini 0.904, vs best student 0.762.
The specialist wins the in-scope leaderboard while remaining materially worse at
knowing what it doesn't know - the thinnest slice of its training data (≤40 oos
rows). Escalation quality is the real gap between this student and the cloud, and
the first thing Phase 2 should buy (oos oversampling / threshold head).

### 7.5 Val→test generalization: composition shift, not overfitting

CLINC headline scores drop 3.9–5.9 pt from val to test. Decomposition shows this
is **exam-shape change, not model degradation**: test is 18.2% oos vs 3.2% on
val. Conditioning on gold in-scope rows only (removing oos from both sides), the
ablation's macro-F1 goes **0.9676 (val) → 0.9581 (test)** - a 0.95 pt shift. The
remaining headline drop is oos composition: 376 of 1,000 oos rows misfire into
in-scope intents, hitting per-class precision across 150 classes, while macro-F1
(oos = 1 class of 151) hides the traffic-weighted damage that accuracy shows
(0.9220 macro vs 0.8985 accuracy). Deployments should read the accuracy column.
On Bitext there is no gap at all: every recipe's test score is *above* its val
seed-mean (+0.1 to +0.6 pt).

A selection-side corollary: epoch selection optimized val macro-F1, a metric in
which oos carries 1/151 of the weight on a split that is 3.2% oos - so checkpoint
choice was nearly blind to the class that dominates test error mass. No hygiene
was violated (selection never touched test), but a different epoch might trade
in-scope F1 for oos recall and this protocol would not see it.

### 7.6 Prompting cannot express what fine-tuning can learn: `reminder_update`

Every prompted model in the study - teacher included, Gemini included - scores
**F1 = 0.0 on `reminder_update`** on the CLINC test set. Under this label
glossary the class is effectively *un-promptable*: no panel model ever separates
it from `reminder`. The confusion is a clean drain, not a scatter - the teacher
and Gemini predict `reminder` for **30 of 30** gold `reminder_update` test rows
(students: 22–28 of 30) - making this CLINC's exact analogue of Bitext's
`delivery_period`↔`track_order` twin pair: every benchmark here carries one
near-duplicate label pair that only weight updates ever resolve. The gold-gate consequently evacuated it from training (Section 4),
and the un-back-filled students score 0.0 as well. The back-fill mitigation
(gate-dropped classes restored with bare gold labels, no rationale) recovers it
to **F1 0.947** and lifts CLINC val macro-F1 to **0.9715** (n=1 seed, val-only -
not test-scored, honestly labeled as such). A fine-tuned 4B learns a label
distinction that prompting a 2.8T model cannot elicit; distillation ceilings are
not prompting ceilings.

### 7.7 Learning curves and epoch knees

Bitext: all recipes peak at epoch 2 of 3 (val F1 dips at 3 while train loss keeps
falling - mild overfit past the knee). CLINC: all recipes still improve at epoch
3; the 6-epoch control shows the knee is exactly 3 (0.9624 @3, then flat:
0.9596/0.9614/0.9617 at 4/5/6). Curves: `artifacts/*/eval/epoch_scores_*.json`.

## 8. Cost analysis

Cost per ticket is `mean_input_tokens × input_price + mean_output_tokens ×
output_price` at pinned list prices. Inputs are ~660–710 tokens (the shared
label-glossary prompt); label-only outputs are ~10–15 tokens, so **input
dominates** - the cost axis is essentially "price per prompt."

**Savings at scale (1M tickets/month), list API prices:**

| Model | $/1k | $/1M tickets/mo |
|---|---|---|
| Kimi K3 (teacher) | $2.20 | **$2,200** |
| Haiku 4.5 | $0.79 | $790 |
| Gemini 3 Flash | $0.39 | $390 |
| GPT-5.6 Luna | $0.35 | $350 |
| DeepSeek 3.2 | $0.19 | $190 |
| **Student - owned 4090 (electricity)** | **~$0.002** | **~$2–3** |
| **Student - rented 4090-class GPU** | ~$0.01 | ~$7–14 |
| **Student - hosted-API ceiling** | ~$0.03 | ~$25 |

**The student's cost is small, not zero - quoted honestly across three serving
scenarios.** Label-only inference emits ~10 output tokens per query and runs at
~14 tickets/s on one 4090 (measured on the 5,500-row test pass), so 1M tickets is
~20 GPU-hours (~0.4 queries/s sustained - trivial for hardware the project
already owns). (1) *Owned GPU:* ~8 kWh at a ~400 W wall draw ≈ **$2–3/1M** at
$0.30/kWh - the true marginal cost. If the card were bought for this ($~1,800
amortized over 3 years), add ~$50/mo flat, independent of volume. (2) *Rented
GPU:* the same 20 hours at prevailing 4090-class rental rates ($0.35–0.70/hr) ≈
**$7–14/1M** with no hardware owned. (3) *Hosted API (hypothetical ceiling):*
the student's prompt is ~70 tokens - the label space lives in its weights, so it
never pays the ~690–710-token label-glossary prompt every panel model pays per
call. Even priced at the panel's *cheapest* pinned per-token rate (DeepSeek's
$0.27/$0.40 per M - a 671B-MoE price a 4B would undercut severalfold), 1M
tickets ≈ **≲$25**. Every scenario is 8–100× below the cheapest cloud option
(DeepSeek, $190/1M, which the student outscores on both benchmarks) and
~100–1,000× below the teacher ($2,200/1M). One-time costs: $15.86 + ~$35 teacher
labeling, ~$22 panel evaluation (OpenRouter; $40 list), a few GPU-hours of
training (individual runs: 1–9 minutes) - break-even at any realistic volume is
days. The recipe choice compounds the economics: recipe A/ablation serve at ~14
tickets/s vs recipe B's 0.7 on this eval harness - and by its ~16× output-token
budget, the recipe that scores worst stays the most expensive to serve under any
stack.

**The money chart** (cost/1k, log x-axis, vs test macro-F1): the student sits in
the formerly empty cheap-and-accurate corner - top-left, at $0.002–0.03/1k
depending on serving scenario - on both benchmarks; on Bitext it is also simply
the top point. Flagships sit high-right (K3 at $2.2); the efficient tier
clusters mid-left ($0.19–0.79) below the student's accuracy on Bitext and
(except Gemini) on CLINC.

## 9. Threats to validity

- **"Beats the teacher" needs its asterisk.** Two mechanisms make >100% retention
  unremarkable rather than paradoxical. (1) The student's supervision is not raw
  teacher output: gold-gating filtered out the ~5% of teacher labels that
  disagreed with gold, so surviving supervision is effectively gold-quality - the
  gate injects gold-label information into the pipeline (as a *filter*, never as
  training text). (2) The comparison is a fine-tuned in-domain specialist vs a
  *prompted zero-shot* generalist; on Bitext the student additionally learns the
  dataset's labeling *conventions* (e.g. the delivery_period/track_order
  boundary) that no prompted model can infer from a glossary. The retention
  denominator is therefore "the best score achievable by *prompting* the
  teacher," not an intrinsic teacher capability - and exceeding it is the
  expected outcome on a closed-set task, not an anomaly. The honest claim is
  not "4B > 2.8T" but "for a fixed narrow task, a fine-tuned 4B beats prompting
  anything we tested."
- **Bitext is synthetic**; its near-ceiling scores overstate real-world accuracy.
  Mitigated by CLINC (real) + oos reporting.
- **A suspected artifact was found, tested, and refuted.** A post-run audit
  flagged three asymmetries favoring the original Bitext recipe-A run (a
  per-stage LR-scheduler rebuild producing a warm-restart sawtooth; annealed
  checkpoints; 2× optimizer steps at equal epochs). The corrected re-run scored
  *higher* (0.9920 vs 0.9886), refuting the inflation hypothesis; all reported
  seed statistics use fixed-schedule runs only; the legacy run is retained in the
  artifacts. Audit trail: trainer-log LR curves, `artifacts/eval/`.
- **A mid-experiment memory-layout change** (per-device batch 32×1 → 16×2 after a
  Windows VRAM livelock; identical effective batch, step count, schedule) is
  recorded per run in `run_config_*.json`; observed seed σ (≤0.002) bounds its
  effect.
- **Label noise:** Bitext `delivery_period↔track_order` caps ablation/B per-class
  F1 and is where recipe A's entire win concentrates (Section 7.3) - if those gold
  conventions are themselves arbitrary, recipe A's Bitext advantage is learning
  an arbitrary convention (it remains an advantage on the benchmark as defined).
  CLINC `reminder`/`reminder_update` is severe enough to evacuate classes through
  the gate (Section 4) and is unresolvable by any prompted model in the panel (Section 7.6).
- **Seed count is 3**, and test is single-seed (s42) by design - adequate for the
  4–8σ effects reported; thin for sub-0.2 pt comparisons (ablation vs its
  6-epoch control; ablation vs B on Bitext val - statistical ties). Every
  recipe-vs-recipe *test* delta therefore borrows val-measured σ as its
  uncertainty proxy (the deltas are 5–15× that σ, but test carries no error bars
  of its own). The back-fill result is n=1, val-only.
- **Test scored once** means no test-side error bars; val→test deltas (+0.1 to
  +0.6 pt Bitext; explained on CLINC, Section 7.5) bound the concern.
- **Teacher prompt fairness:** student prompts identical across benchmarks by
  design; the teacher prompt was adapted per benchmark (label glosses, oos
  instruction). Panel models all received the same label-only prompt per
  benchmark.
- **Price drift:** prices pinned at snapshot 2026-07-28/30; K3's reasoning-token
  billing (thinking tokens billed as output) is included in its measured cost.

## 10. Overfitting discipline

Train/val/test with test-scored-once. Val-only tuning; per-epoch checkpoints;
knees measured per benchmark (Section 7.7); train-loss-vs-val-F1 gap logged per epoch in
`artifacts/*/eval/findings.json`. The ablation memorizes its label-only train set
(loss ≈ 0 by epoch 2) while val F1 still improves - memorization of a small
per-class sample is not, by itself, the failure mode here. The end-to-end
evidence: Bitext test scores land *above* val means, and the CLINC val→test drop
decomposes into a documented composition shift (Section 7.5), not overfitting.

## 11. Limitations and future work

- **Escalation is the remaining gap** (Section 7.4): every flagship beats every student
  on oos F1. Phase 2: oversample oos, or add a calibrated escalate threshold;
  re-measure on test *once*.
- **Back-fill to all seeds + test**: the 0.9715 val result is n=1; promote it to
  the headline recipe only after seed replication and its single test scoring.
- **Data-efficiency curves** (10/20/30/40 per class): the strongest remaining case
  for rationales is fewer-examples-needed, untested here.
- Single student size; a 1.7B/4B/8B frontier is cheap to add (targets are
  model-agnostic).
- Phase 2 proper: priority/escalate heads on owned DailyDles data; OOD probe of
  Bitext-trained students on real phrasings.

## 12. Conclusion

A 4B QLoRA student trained on a few thousand gold-gated frontier labels doesn't
just retain a 2.8T teacher's intent-triage accuracy - it beats the teacher on
both benchmarks and tops the entire eight-system leaderboard on one of them, at
$2–25 per million tickets (electricity → hosted-API ceiling) vs $190–$2,200 for
the cloud. The
fashionable part of the recipe - distilling rationales - turned out to be a
property of the data, not the method: a large, real win on templated data
(localized to one confusable label pair) that inverts into a significant loss on
real queries, with the reason-then-label variant additionally destroying the
escalation signal, failing the retention gate, and costing 20× more to serve.
The residual case for the cloud is exactly one number: prompted flagships still
know what they don't know better than the specialist does (oos F1 0.88–0.90 vs
0.76). Distill for the task you can name; keep the escalation path humble; and
distrust any rationale-distillation result demonstrated only on synthetic data.

## Appendix A. Reproducibility

Frozen label spaces, split manifests and hashes, subsample manifests:
`artifacts/`. Teacher outputs incl. archived reasoning: `data/*/label/labeled.jsonl`
(committed). Exact commands: `README`/`HANDOFF-M2-4090.md`; per-run configs incl.
git commit, seed, knobs: `artifacts/*/eval/run_config_*.json`. Environment lock:
`uv.lock` + `docs/ENV-4090-WINDOWS.md`. Adjustment changelog + caveats:
`artifacts/*/eval/findings.json` (+ `test_eval` blocks). Panel raw scores:
`artifacts/*/eval/panel/`. Test predictions for independent rescoring:
`artifacts/*/eval/test_*_preds.jsonl` against `data/*/train/test_eval.jsonl`.
Rendered figures (money chart, leaderboard, recipe bars, learning curves,
savings-at-scale; light + dark, SVG + PNG): `artifacts/charts/`, regenerable via
`uv run python -m triage_distill.eval.render_charts`.
