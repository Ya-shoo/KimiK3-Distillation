# The 5th-Grader Version - The Little Robot That Beat Its Teacher

> **FINAL** (2026-07-31). No background needed. The grown-up versions are
> [`PAPER-UNDERGRAD.md`](PAPER-UNDERGRAD.md) and [`PAPER.md`](PAPER.md).

## The story in one breath

There's a giant, super-smart AI that costs real money every single time you ask
it a question. We used it as a **teacher**: it did a few thousand practice
problems for us, one time. Then we used those practice problems to train a
tiny AI - about a thousand times smaller - that runs on a home gaming computer
for a couple of dollars of electricity. On the final exam, **the tiny student scored higher than its giant
teacher. On both exams. And on one exam it beat every AI we tested, period.**

## What was the job?

Sorting messages. A company gets thousands of messages like *"Where's my
package?"* or *"I want my money back"* - and each one needs to go in the right
bin (there were 27 bins in one exam, 151 in the other). Big AIs are great at
this, but paying a giant AI to sort a million messages costs about **$2,200 a
month**. Our tiny student does it for **a couple of dollars of electricity**.

## How did we teach it?

1. The giant teacher sorted a few thousand example messages and wrote one
   sentence explaining each choice. (This cost about $51. Once. Ever.)
2. We threw away the ~5% of examples where the teacher got the answer wrong -
   we could check, because these practice sets come with answer keys.
3. The tiny student studied those examples. Whole study session: a few minutes
   per try on one gaming computer.
4. Then came the final exam - questions the student had **never seen**, graded
   exactly once, no do-overs. The teacher and four other big AIs took the very
   same exam.

**Final scores (the real exam):** Student: **99 out of 100** on the easy exam
(best of everyone!) and **92** on the hard one (second place - one big AI got
93, the teacher got 91). Before studying, the student scored about 30. Studying
was everything.

## The three surprises

**1. Teaching the student to "explain its thinking" only helped on the fake
exam.** One exam used computer-generated practice sentences (very neat and
samey); the other used messy sentences from real people. When we also taught
the student the teacher's *explanations*, it got better on the neat fake exam -
but **worse** on the real-people exam. We triple-checked with three separate
training runs each. Explanations aren't magic; whether they help depends on
what you're studying.

**2. One version of the student forgot how to say "I don't know."** The hard
exam has trick questions that fit *no* bin - the right move is to raise your
hand and say "this one's for a human." The student trained to explain-first,
then-answer almost never raised its hand (it caught only about 1 in 4 trick
questions, versus 2 in 3 for the others). Writing an explanation first made it
talk itself into *some* answer every time. The funny part: on the trick
questions its handwriting turned to scribbles - its answers came out garbled
almost only on those. Deep down it *knew* something was off; it just couldn't
say so. If you build a robot helper, "I don't know, ask a person" is the one
skill it must never lose.

**3. Our safety filter accidentally erased a whole bin - and fixing it revealed
a superpower.** The teacher got *every single* practice problem wrong for one
tricky bin ("update my reminder" vs "set a reminder"), so our
throw-away-wrong-answers filter deleted that bin's practice problems entirely,
and the student scored zero on it. Here's the wild part: on the final exam,
**every big AI - including the giant teacher - also scored zero on that bin.**
You literally cannot *ask* your way to that distinction. But when we gave the
student the answer key for just that bin and let it study, it scored 95. Little
students that study can learn things giant AIs can't be talked into.

## Wait - how does a student beat its own teacher?

Two honest reasons. First, we deleted the teacher's mistakes from the study
materials, so the student studied a cleaner book than the teacher's own brain.
Second, the student got to *study for this exact test* while the big AIs walked
in cold, just reading the instructions. That's the whole point: for one narrow
job, a tiny specialist that studied beats a giant genius that didn't.

## Why it matters

Sorting a million messages a month: giant teacher, ~$2,200. Cheapest big-AI
option, ~$190. Our student: **about $2–3 of electricity** on a computer we
already own (and even if we rented a computer or paid a company to run it for
us, at most about $25), while scoring *higher* than almost all of them. Total
cost to build it: about $73 plus a few hours of computer time. It pays for
itself in days.

And one warning for the grown-ups: the student is still worse than the big AIs
at raising its hand on trick questions. Fixing that comes next.
