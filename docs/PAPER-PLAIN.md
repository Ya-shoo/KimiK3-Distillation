# What We Built and What We Found — the plain-English version

> **DRAFT v2** (2026-07-30, all training runs complete — 3 random seeds per
> variant on both benchmarks). Companion to [`PAPER.md`](PAPER.md); assumes you've
> met the ideas in [`PRIMER.md`](PRIMER.md) (teacher/student, LoRA, ablation,
> macro-F1). ⟦Only the frontier-model showdown numbers are still pending.⟧

## The one-paragraph version

We taught a small, free-to-run AI model (Qwen3-4B) to sort customer messages into
categories by learning from a giant frontier model (Kimi K3). It works: our
student gets ~96–99% balanced accuracy on two very different benchmarks. But the
trendy part of the method — also teaching the student the big model's *reasoning*,
not just its answers — turned out to be a coin with two faces: on the tidy
synthetic benchmark it genuinely helps (a clear win, well beyond random
variation), and on the real-people benchmark it genuinely *hurts* (a clear loss,
also beyond random variation). And one reasoning variant quietly broke the most
business-critical behavior everywhere: knowing when to say *"this fits nothing —
send it to a human."*

## What we actually did

1. **Picked two exams.** Bitext: 27 customer-support intents, but synthetic —
   template-generated sentences, a bit too tidy. CLINC150: 151 categories of real
   human queries, including a special one, **`oos`** ("out of scope"), for
   messages that fit *nothing* — the "escalate to a human" signal.
2. **Had the teacher grade practice sets.** Kimi K3 labeled a few thousand
   examples per benchmark and wrote a one-sentence *why* for each. We kept only
   the ones where K3's answer matched the official answer (**gold-gating**,
   ~95% kept).
3. **Trained the same student three ways** — the controlled experiment:
   - **Ablation:** message → label. Nothing else. (The control.)
   - **Recipe B:** message → *reasoning, then* label, in one breath.
   - **Recipe A:** reasoning and labeling trained as two separate skills; at
     answer-time you only use the fast labeling skill.
4. **Graded on held-out questions after every epoch**, kept the test set locked
   away (it gets opened exactly once, later, when we grade the big models too).

## The scoreboard (validation; average over 3 random seeds, ± the seed wobble)

| | Bitext (synthetic) | CLINC (real) | CLINC "escalate" catch rate |
|---|---|---|---|
| Answers-only (control) | 97.9 ± 0.1% | **96.1 ± 0.1%** | 69% |
| Recipe A (two skills) | **99.2 ± 0.1%** | 95.6 ± 0.2% | 72% |
| Recipe B (think-then-answer) | 97.8 ± 0.2% | 93.2 ± 0.1% | **30%** |

The wobble columns matter: every gap discussed below is many times larger than
the ±, so none of this is luck-of-the-seed.

## The three findings

### 1. The reasoning recipe is a mirror: real win on fake data, real loss on real data

On the synthetic benchmark, Recipe A wins big — +1.3 points, replicated across
all three seeds, exactly what the famous "Distilling Step-by-Step" result
predicts. On the real benchmark the *same recipe loses* — −0.5 points, also
replicated across all three seeds, and still losing when we gave the control the
same total training compute. Our best explanation: synthetic data is written from
templates, and the teacher's explanations effectively teach the student the
template grammar — free extra signal. Real human queries have no grammar to
learn; the answer sits near the surface, and the reasoning detour spends the
student's limited capacity without adding information. The uncomfortable
implication for the field: method papers that evaluate reasoning-distillation on
clean benchmark data may be measuring the benchmark, not the method.

### 2. Think-first training broke the escalate button

Recipe B — the version that writes its reasoning *before* its answer — catches
only ~30% of out-of-scope messages (consistent across every seed: 26–35%), versus
~70% for everyone else. Writing an explanation first commits the model to
"evidence for some category…" — and the explanation format has no graceful way to
say "none of the above." So the model talks itself into the nearest category
instead of raising its hand. The same rambling occasionally runs past the token
budget before the answer ever appears — the only invalid outputs in the entire
project, all from this recipe. If your deployment depends on escalation (most
support systems do), that's disqualifying, whatever the headline accuracy says.

### 3. Our quality filter quietly deleted a whole category

Gold-gating (keep only teacher-answers-that-match-official-answers) sounds purely
protective. But K3 mislabeled *every single* `reminder_update` example as its
near-twin `reminder` — so the filter threw out 100% of that category's training
data, and the student, having never seen it, scores **zero** on it forever. A
filter meant to remove bad reasoning can also silently starve the student of
whole categories. (Fix queued: for filtered-out categories, feed the official
answers without the teacher's reasoning.)

### Bonus finding: we accused our own experiment of cheating — and it was innocent

A post-run audit found Recipe A's original synthetic-benchmark run had gotten
three hidden advantages: a learning-rate bug gave it a schedule the others didn't
get, its snapshots were taken at a more flattering moment, and it received twice
the training steps. Suspicious, we re-ran it with everything corrected — and the
score went *up* (99.2% vs the original 98.9%). The suspected inflation wasn't
there; the bug had actually been holding it back. We report this because the
process is the point: believing a result means trying to break it, and publishing
the audit either way — the accusation, the test, and the acquittal.

## What's still coming

- **The main event:** grading Kimi K3 itself and six other frontier/efficient
  models on the same locked test set, with real prices — the
  cost-vs-accuracy chart this project exists for. ⟦M3⟧
- A test of whether reasoning at least lets the student learn from *fewer*
  examples (its last remaining defense), and a fix for the deleted-category
  problem from finding 3.

## Why this matters

A model ~1000× smaller than the teacher, trained for a few dollars of API calls
and an evening on one gaming GPU, is already in the mid-90s on a 151-way real
benchmark. That's the economics of specialists: pay a frontier model once as a
teacher, not forever as an employee. And the controlled experiment did what
controlled experiments are for — it stopped us from shipping a fashionable
technique that, for this task, at best does nothing and at worst breaks the one
behavior a triage system cannot lose.
