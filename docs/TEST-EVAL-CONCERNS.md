# Test-eval concerns, caveats, and open questions (agent, 2026-07-31)

Written after the six test passes landed (commits `9d714ea`, `b8ce2fd`). These are
the things I would push back on or want answered before the paper freezes. Numbers
cited are from `artifacts/{eval,clinc/eval}/test_*.json` and the `test_eval`
sections of both findings.json files. Items marked **[post-hoc]** need only the
already-written preds files - no new model runs, no re-scoring, no extra test
consumption. Items marked **[owner call]** are framing/decision questions.

## 1. The ">100% retention" headline invites an obvious attack - preempt it. [owner call]

Five of six passes score above the teacher. A skeptical reader's first response
will be: "then the teacher is a weak baseline or the labels leak." The honest
mechanics are defensible - a fine-tuned 4B specialist beating a *prompted*
generalist on a closed 27/151-way task is textbook distillation-as-denoising,
not an anomaly - but the paper should say so explicitly rather than let
"retention 104.5%" sit unexplained. Suggest one paragraph: the teacher runs
zero-shot with a schema prompt, the student has 3 epochs of task adaptation;
"retention" is measured against the best *achievable-by-prompting* teacher
score, and exceeding it is expected when the task is closed-set. Consider
whether the term "retention" should survive at all in the >100% rows, or
whether the money chart should relabel to "% of teacher score."

## 2. Checkpoint selection happened on a val split whose oos share is 6x smaller than test's. [owner call]

Val CLINC is 3.2% oos (100/3100); test is 18.2% (1000/5500). We selected epochs
on val macro-F1, a metric where oos is 1 class of 151 - so the selection was
nearly blind to the class that dominates test error mass (oos recall 0.264-0.624
across recipes). This is not a hygiene violation (selection never saw test), but
it *is* a distribution-shift caveat: a different epoch might trade in-scope F1
for oos recall, and we would never know. The paper should state the composition
mismatch next to the val→test deltas, or the -4 to -6pt drops will read as
mysterious generalization failure when they are mostly composition.
Related question: was the 100-oos val split a deliberate design choice or an
artifact of how CLINC's official splits were carved? Worth one sentence either way.

## 3. Test deltas ride on val-measured error bars. [owner call]

Seed variance (σ 0.0006-0.0019) was measured on val; test got exactly one pass
per recipe by design. Fine - but then every recipe-vs-recipe test delta the
paper leans on (CLINC ablation +1.17pt over recipe_a; bitext recipe_a +0.75pt
over ablation) is implicitly using val σ as its uncertainty proxy. Those deltas
are 5-15x the val σ, so the conclusions look safe, but the paper should say "test
is a single seed; σ quoted from val" once, plainly, rather than imply test error
bars exist.

## 4. Did the LMFE MemoryError contaminate more than one batch? [RESOLVED 2026-07-31]

Verified from `test_recipe_b_preds.jsonl` (no model involvement): all 71
invalids are `constrained_retry` rows (clean-pass rows: 0 invalid), and they are
spread uniformly across the retry sequence - 67 occurred *before* the
MemoryError near retry row ~868 (7.7% base rate), 4/14 after. No cache
contamination; the "at most ~4" attribution in findings.json holds.

The sharper finding this surfaced: the constrained retry path has an intrinsic
~7.7% invalid rate with LMFE fully functional - the 512-token decode budget
truncates mid-JSON even when every emitted token is schema-legal. The budget,
not the MemoryError, is the mechanism behind the 71 invalids. This feeds
directly into item 5's knowledge-vs-format decomposition.

## 5. Decompose recipe_b's CLINC failure: knowledge vs format. [post-hoc]

recipe_b fails the gate at 96.1%, and 882 rows (16%) went through the retry
path. Two different stories are entangled:
(a) reasoning-at-inference *chooses worse labels* at 151 classes, or
(b) reasoning-at-inference *fails to emit parseable output* and the retry path
degrades it further.
Split macro-F1 over the 4,618 clean-pass rows vs the 882 retried rows (both
already labeled in the preds file). If clean-pass rows alone clear the gate, the
finding is really "format fragility at scale," which points at decode budget and
the retry implementation - not at rationale-style distillation itself. That
materially changes the Section 7.4 sentence. Same check on bitext recipe_b (2 invalids,
near-zero retries) as the control.

## 6. Where does the dead class's probability mass go? [post-hoc]

`reminder_update` is F1=0.0 for recipe_b on test and ~0.0 across nearly all
CLINC runs on val - chronic, not a test artifact. One confusion-row from the
preds files answers whether it all drains into `reminder` (label-space overlap →
dataset critique, one sentence in limitations) or scatters (model failure). The
`reminder`/`reminder_update` pair smells like Bitext's `delivery_period`/
`track_order` twins; if so, the paper gets a nice cross-benchmark symmetry point
about label-noise floors instead of an unexplained zero.

## 7. The 20x inference-cost figure is an artifact of our stack - scope it. [owner call]

125.6 min vs ~6.5 min is real on this 4090 + transformers + LMFE batch-4 retry
path, but a serving stack (vLLM, grammar-compiled constraints, larger retry
batches) would compress the gap substantially. If the cost table quotes ~20x,
scope it as "our eval harness" or a reviewer who runs vLLM will (fairly) call it
inflated. The defensible intrinsic claim is token-count asymmetry (~512 vs ~32
max_new_tokens), not wall-clock.

## 8. Missing baseline row: zero-shot Qwen3-4B on test. [owner call - costs one test consumption]

The money chart contrasts student vs teacher vs panel, but the "before
fine-tuning" bar (base Qwen3-4B, zero-shot, classify prompt) was only ever run
on val, if at all. infer.py supports it (`--adapter` omitted). Two cheap classify
passes (~6-7 min each) would anchor the chart's left edge on the same split as
everything else. This *is* a new consumption of test by a new model - legitimate
under the score-once rule (no selection involved), but it is your call whether
the chart needs it. If yes, decide before the paper freezes so it's one pass,
not an iterated one.

## 9. Small bookkeeping items

- The CLINC retention denominator in earlier chat/paper drafts was rounded to
  0.908; the panel file says 0.90779. findings.json now carries the exact value -
  make sure PAPER.md tables use it (0.9078) so retention percentages reproduce.
- test_recipe_b's 71 invalids are 1.3% of rows; the scorer counts them as wrong.
  If the paper quotes "macro-F1 0.8724," a footnote should say invalids are
  included as errors, or someone re-deriving from per-class F1 will get a
  slightly different number.
- The retry path's batch-4 hardcode (infer.py:174) is now a measured 2-hour cost
  center. The fix options are logged in chat (flag it, or constrain only the
  label field on retry); the second option changes what recipe_b "is" for
  retried rows, so it needs an owner decision before any M3 re-runs adopt it.
