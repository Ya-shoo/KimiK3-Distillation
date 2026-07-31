# What We Built and What We Found — the plain-English version

> **DRAFT v1** (2026-07-30). Companion to [`PAPER.md`](PAPER.md); assumes you've met
> the ideas in [`PRIMER.md`](PRIMER.md) (teacher/student, LoRA, ablation, macro-F1).
> ⟦Numbers still being measured are marked like this.⟧

## The one-paragraph version

We taught a small, free-to-run AI model (Qwen3-4B) to sort customer messages into
categories by learning from a giant frontier model (Kimi K3). It works: our student
gets ~96–99% balanced accuracy on two very different benchmarks. But the trendy
part of the method — also teaching the student the big model's *reasoning*, not
just its answers — didn't hold up when we tested it fairly on real data. On the
real benchmark, the plain version (answers only) beat both reasoning versions, and
one reasoning version quietly broke the most business-critical behavior: knowing
when to say *"this fits nothing — send it to a human."*

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

## The scoreboard (validation, one seed so far ⟦error bars coming⟧)

| | Bitext (synthetic) | CLINC (real) | CLINC "escalate" catch rate |
|---|---|---|---|
| Answers-only (control) | 98.0% | **96.2%** | 71% |
| Recipe A (two skills) | **98.9%** † | 95.7% | 72% |
| Recipe B (think-then-answer) | 97.6% | 93.2% | **33%** |

† Asterisk explained below — this number got an unfair boost.

## The three findings

### 1. The reasoning recipe lost the rematch on real data

On the synthetic benchmark, Recipe A looked like the winner — exactly what the
famous "Distilling Step-by-Step" result predicts. On the real benchmark it *lost*
to the plain control. We double-checked the loss three ways: we gave the control
the same total training compute as A (it still won); we fixed a scheduling bug
(below); and we selected every version's best epoch the same way. ⟦Seed-variance
runs will tell us if the gap is solid or wobble.⟧ The honest current read: on
short, real queries, the answer is close to the surface — the reasoning detour
doesn't add information the label doesn't already carry.

### 2. Think-first training broke the escalate button

Recipe B — the version that writes its reasoning *before* its answer — catches
only ~33% of out-of-scope messages, versus ~71% for everyone else. Writing an
explanation first commits the model to "evidence for some category…" — and the
explanation format has no graceful way to say "none of the above." So the model
talks itself into the nearest category instead of raising its hand. If your
deployment depends on escalation (most support systems do), that's disqualifying,
whatever the headline accuracy says.

### 3. Our quality filter quietly deleted a whole category

Gold-gating (keep only teacher-answers-that-match-official-answers) sounds purely
protective. But K3 mislabeled *every single* `reminder_update` example as its
near-twin `reminder` — so the filter threw out 100% of that category's training
data, and the student, having never seen it, scores **zero** on it forever. A
filter meant to remove bad reasoning can also silently starve the student of
whole categories. (Fix queued: for filtered-out categories, feed the official
answers without the teacher's reasoning.)

### Bonus finding: we caught our own experiment cheating

The synthetic-benchmark "win" for Recipe A (the †) got three hidden advantages,
found in a post-run audit: a learning-rate bug gave it a schedule the others
didn't get; its per-epoch snapshots were taken at a more flattering moment; and it
received twice the training steps. All three are fixed or controlled in the real-
benchmark runs — which is precisely where the win evaporated. ⟦A corrected
synthetic-benchmark re-run is happening tonight.⟧ Moral: before believing a
method's win, audit what *else* differed.

## What's still coming

- **The main event:** grading Kimi K3 itself and six other frontier/efficient
  models on the same locked test set, with real prices — the
  cost-vs-accuracy chart this project exists for. ⟦M3⟧
- Error bars (3 random seeds per recipe), the corrected synthetic re-run, and a
  test of whether reasoning at least lets the student learn from *fewer* examples
  (its last remaining defense).

## Why this matters

A model ~1000× smaller than the teacher, trained for a few dollars of API calls
and an evening on one gaming GPU, is already in the mid-90s on a 151-way real
benchmark. That's the economics of specialists: pay a frontier model once as a
teacher, not forever as an employee. And the controlled experiment did what
controlled experiments are for — it stopped us from shipping a fashionable
technique that, for this task, at best does nothing and at worst breaks the one
behavior a triage system cannot lose.
