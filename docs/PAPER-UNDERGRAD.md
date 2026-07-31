# The Undergrad Version — What We Built and What We Found

> **FINAL** (2026-07-31). This is the mid-level read: you've taken an intro ML
> course (you know what a train/test split, fine-tuning, and an F1 score are),
> but none of the jargon beyond that is assumed. The rigorous version with every
> caveat is [`PAPER.md`](PAPER.md); the no-background version is
> [`PAPER-ELI5.md`](PAPER-ELI5.md). *(This file supersedes the old `PAPER-PLAIN.md`.)*

## The one-paragraph version

We taught a small, free-to-run model (Qwen3-4B, runs on one gaming GPU) to sort
customer messages into categories by learning from a frontier model roughly
1000× its size (Kimi K3). On the locked-away test sets — opened exactly once —
the student didn't just approach the teacher, it **beat it on both benchmarks**,
and on the synthetic one it beat *every* system we scored, including four other
frontier/efficient cloud models. Marginal running cost: ~$0, versus $190–$2,200
per million messages for the cloud options. The trendy part of the method —
teaching the student the big model's *reasoning*, not just its answers — turned
out to be a coin with two faces: a genuine win on synthetic data, a genuine loss
on real data, both far beyond seed noise. And one reasoning variant quietly broke
the most business-critical behavior of all: knowing when to say "this fits
nothing — send it to a human."

## What we actually did

1. **Picked two exams.** Bitext: 27 customer-support intents, but synthetic —
   template-generated sentences, a bit too tidy. CLINC150: 151 categories of real
   human queries, including a special one, **`oos`** ("out of scope"), for
   messages that fit *nothing* — the "escalate to a human" signal.
2. **Had the teacher grade practice sets.** Kimi K3 labeled a few thousand
   examples per benchmark ($16 + $35, one time) and wrote a one-sentence *why*
   for each. We kept only the rows where K3's answer matched the dataset's
   official answer (**gold-gating**, ~95% kept). Note what this means: the
   student's training data is teacher-labeled but gold-*filtered* — remember
   this when the student later "beats" the teacher.
3. **Trained the same student three ways** — the controlled experiment:
   - **Ablation:** message → label. Nothing else. (The control.)
   - **Recipe B:** message → *reasoning, then* label, in one output.
   - **Recipe A:** reasoning and labeling trained as two separate skills; at
     answer-time you only use the fast labeling skill.
   Identical hyperparameters everywhere; the recipe is the only variable. Three
   random seeds per recipe so we can tell signal from luck.
4. **Kept test sacred.** All tuning and checkpoint selection used the validation
   split. The test split was scored once per system, at the very end, students
   and cloud models together, same scorer. (We also re-computed every headline
   test score independently from the raw prediction files. They match.)

## The final scoreboard (test sets, scored once — macro-F1)

| System | Bitext (synthetic, 27-way) | CLINC (real, 151-way) | Cost per 1k messages |
|---|---|---|---|
| **Student — Recipe A** | **99.2** 🥇 | 91.0 | ~$0 |
| **Student — ablation** | 98.5 | **92.2** 🥈 | ~$0 |
| **Student — Recipe B** | 98.0 | 87.2 ❌ | ~$0 |
| Kimi K3 (the teacher) | 95.0 | 90.8 | $2.20 |
| Gemini 3 Flash | 93.6 | **93.5** 🥇 | $0.39 |
| GPT-5.6 Luna | 93.6 | 89.6 | $0.35 |
| DeepSeek 3.2 | 93.2 | 86.7 | $0.19 |
| Haiku 4.5 | 87.9 | 88.3 | $0.79 |

The project's pass gate was "retain ≥97.5% of the teacher's score." Five of six
student runs didn't retain — they **exceeded**: 100.3–104.5% of the teacher. The
one failure (❌) is Recipe B on real data: 96.1%, the only gate miss in the
project. For scale, the same student model *without* fine-tuning scores 31% and
27% — the training is the entire product.

## The five findings

### 1. The reasoning recipe is a mirror: real win on fake data, real loss on real data

On synthetic Bitext, Recipe A wins big — +1.3 points over the label-only control
on validation (~8 standard deviations of seed noise), +0.75 on test — exactly
what the famous "Distilling Step-by-Step" paper predicts. On real CLINC the
*same recipe loses*: −0.5 points (~4σ) on validation, −1.2 on test, and it keeps
losing even when we gave the control the same total number of optimizer steps
(Recipe A's data file is 2× longer, so equal epochs ≠ equal compute — we
controlled for that). Same code, same hyperparameters, opposite conclusions —
the only thing that changed is the data. The uncomfortable implication: method
papers that evaluate reasoning-distillation only on clean, templated benchmarks
may be measuring the benchmark, not the method.

### 2. The entire synthetic win hides in one confusable label pair

Per-class autopsy of the Bitext gap: Recipe A's advantage lives almost entirely
in `delivery_period` vs `track_order` — the two intents even the teacher
confuses (its two worst classes, F1 0.70/0.77). Recipe A scores 0.99+ on both;
the control sits at ~0.87. Those two classes account for *more than the whole*
gap (the other 25 classes net slightly against A). So "rationales help" here
really means "rationales taught the student the boundary convention between two
overlapping labels." That's a much narrower — and more honest — claim.

### 3. Think-first training broke the escalate button

Recipe B — reasoning *before* the answer in one output — catches only 26% of
out-of-scope messages on test, versus ~60–69% for the other recipes. Writing an
explanation first commits the model to "evidence for some category…" — the
format has no graceful way to say "none of the above," so it talks itself into
the nearest category instead of raising its hand. Recipe B also produced the
project's only invalid outputs (rationales that overrun the token budget before
the answer appears, 71/5,500 on test, scored as wrong) and runs ~20× slower at
inference on our setup. It failed the retention gate, the escalation test, the
format test, and the cost test simultaneously. Buried.

The autopsy found something stranger, though: 85% of the rows where Recipe B's
output *broke* (invalid JSON on the first try) were exactly the out-of-scope
ones. The model does sense when a message fits nothing — but the signal comes
out as *malformed output* instead of the "out of scope" label. "The output
broke" detects out-of-scope messages with 75% recall; the model's actual
predictions manage 26%. The escalation signal exists; the format eats it.

But here's the twist the test set added: **every cloud flagship still beats
every student at escalation** (oos F1: Gemini 0.90, teacher 0.88, best student
0.76). Our specialist wins the categorization leaderboard while being worse at
knowing what it doesn't know — it saw at most 40 out-of-scope training examples.
That's the honest remaining gap, and the first thing to fix.

### 4. Our quality filter deleted a whole category — and the fix taught us something better

Gold-gating sounds purely protective, but K3 mislabeled *every single*
`reminder_update` example as its near-twin `reminder`, so the filter silently
deleted the entire category from training, and the student scored 0.0 on it.
The fix (feed the deleted categories their official labels directly, no
teacher reasoning) recovered it to F1 0.95 and produced our best CLINC
validation score, 97.2%. The kicker: on the test set, **every prompted model —
the 2.8T teacher and all four panel models — scores exactly 0.0 on that class.**
No amount of prompting elicits the distinction; twenty minutes of fine-tuning
learns it. Prompting ceilings are not fine-tuning ceilings.

### 5. The scary val→test drop turned out to be the exam changing shape, not the model failing

CLINC test scores are 4–6 points below validation, which normally smells like
overfitting. Decomposition says otherwise: the test set is 18.2% out-of-scope
messages vs 3.2% on validation. Restrict both splits to in-scope messages and
the control's macro-F1 moves just 96.8 → 95.8. The "drop" is mostly a thousand
oos messages misfiring into regular categories and denting per-class precision.
(On Bitext there's no drop at all — every test score is slightly *above* its
validation mean.) Lesson: before crying overfitting, check whether the two
splits are even the same exam.

### Bonus: we accused our own experiment of cheating — and acquitted it with data

A post-run audit found Recipe A's original synthetic-benchmark run had three
hidden advantages, including a learning-rate scheduler bug that gave it a
sawtooth schedule nobody intended. We re-ran it fully corrected — and the score
went *up* (99.2 vs 98.9). The bug had been holding it back, not helping it. We
report this because the process is the point: believing a result means trying
to break it, and publishing the audit either way.

## The economics (why anyone funds this)

Sorting 1M messages/month costs $2,200 with the teacher, $190–$790 with the
efficient cloud tier — and ~$0 (electricity) with the student, which outscores
all of them on Bitext and all but Gemini on CLINC, at ~14 messages/second on
one gaming GPU. Total one-time spend: ~$51 of teacher labeling + ~$22 of panel
evaluation + a few GPU-hours (each training run is 1–9 minutes). Break-even is
measured in days.

## What's left

Fix escalation (oversample oos / add a calibrated "not sure" threshold), rerun
the back-fill with all seeds and score it on test once, test whether rationales
at least let the student learn from *fewer* examples (their last remaining
defense), and then Phase 2: deploy on real product feedback.
