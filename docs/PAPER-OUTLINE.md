# Deliverable spec - the mini-paper + the resume artifact

Two end-of-project (M6) deliverables, specified here so the pieces are logged as we go
(especially by the 4090 during training) and nothing has to be reconstructed later.
Everything pulls from committed artifacts (hashes, `score.py` outputs, `charts.py`).

> Status: **spec only** - written 2026-07-28 while CLINC labeling runs. The paper is
> written once M2 (training) + M3 (eval) numbers exist. Placeholder = ⟦…⟧.

---

## Part A - The mini research paper  → `docs/PAPER.md`

**Thesis:** *You pay frontier prices for generality you don't need on a narrow task. We
bought ≥97.5% of Kimi K3's accuracy for ~1% of the cost - and here is the controlled
experiment, on two benchmarks, that proves it.*

| Section | Section | Argues / contains | Pulls from |
|---|---|---|---|
| 1 | **Abstract** | one paragraph: task, method, headline retention %, cost multiple, two benchmarks | final results |
| 2 | **Introduction** | the "generalist tax" on narrow NLP (triage/sentiment/routing); why a specialist wins | - |
| 3 | **Background** | knowledge distillation + rationale distillation ("Distilling Step-by-Step"); why rationales beat labels | `docs/PRIMER.md` |
| 4 | **Task & data** | Bitext-27 (synthetic) + CLINC-151 (real, +OOS); dedup-before-split (0% leakage); gold-gating | `artifacts/*/split_manifest.json`, `subsample_manifest.json` (hashes) |
| 5 | **Method** | teacher labeling → gold-gating → 3 recipes (A/B/ablation) → QLoRA Qwen3-4B; two-tier panel | `prompts/teacher*.md`, `configs/models.yaml` |
| 6 | **Setup** | model versions + **pinned prices** (`prices.yaml`, snapshot date), eval protocol (label-only), seeds, bootstrap CIs, test scored ONCE | `configs/prices.yaml` |
| 7 | **Results** | see the five sub-results below | `artifacts/*/eval/*.json` |
| 8 | **Cost analysis** | in-depth $ (see Part C chart set); marginal cost, savings-at-scale, break-even | `prices.yaml` + measured tokens |
| 9 | **Risks / threats to validity** | see checklist below | - |
| 10 | **Overfitting analysis** | train/val/test discipline; val learning curves; train-val gap; the two-benchmark + OOS generalization gap | 4090 logs (Part B) |
| 11 | **Limitations & future work** | synthetic-data caveat, single student size, Phase-2 deployment | - |
| 12 | **Conclusion** | restate the number; when to distill vs prompt | - |
| A | **Appendix / reproducibility** | frozen hashes, exact commands, `uv.lock`, price snapshot | repo |

### Section 7 Results - the five sub-results (each is a chart + a claim)
1. **Retention** - student macro-F1 ≥ 97.5% of K3, per benchmark. → `panel_leaderboard`
2. **Cost vs accuracy** (the hero) - student high-and-left; flagships high-and-right; efficient tier stuck between. → `money_chart`
3. **Ablation** - rationale recipes (A/B) beat label-only, beyond seed noise. → `model_bars` (error bars). *If ablation wins, report it honestly + the data-efficiency curve - see PRIMER.*
4. **Data efficiency** - macro-F1 vs #examples/class, rationale vs ablation; "the rationale buys an N× data reduction." → `data_efficiency` (to build)
5. **Generalization** - the recipe holds on BOTH a synthetic (Bitext) and a real (CLINC) benchmark; OOS-recall as the escalate signal.

### Section 9 Risk checklist (address each explicitly - naming them is the credibility move)
- Bitext is **synthetic/templated** → in-distribution number overstates; mitigated by CLINC (real) + OOS.
- **Label noise** (e.g. Bitext `delivery_period→track_order`, CLINC `order`/`shopping_list`) - quantify from the gold-gate drop set.
- **Prompt fairness** - same prompt/label space/test sample for all; report label-only regime + why.
- **Price drift** - pinned snapshot + eval date.
- **Overfitting** - Section 10.
- **Single seed / variance** - report mean ± CI over seeds.

---

## Part B - Findings the 4090 must LOG during training (feeds Section 7.3, Section 7.4, Section 10)

The training box is asked to emit these as JSON to `artifacts/<dataset>/eval/` so the
paper and charts read them directly (do NOT keep them only in a terminal scrollback):

- [ ] **Per-epoch val macro-F1** for each recipe (A / B / ablation), each dataset → *learning curves*.
- [ ] **Best epoch per recipe**, and **where diminishing returns start** (the epoch after which val F1 flattens/drops).
- [ ] **Every adjustment made and why** - LR, LoRA r/alpha, max_seq_len (did Recipe B's rationale get truncated?), loss-masking, batch/grad-accum. A short changelog.
- [ ] **Seed variance** - ≥2–3 seeds per recipe; report mean ± σ (this is what the error bars need).
- [ ] **Overfitting signals** - train-vs-val gap per epoch; note if/when val turned up while train kept falling.
- [ ] **Data-efficiency runs** (if done) - macro-F1 at 10/20/30/40 examples/class, rationale vs ablation.
- [ ] **Final test scores** - scored EXACTLY ONCE, all recipes, both datasets → `artifacts/<dataset>/eval/test_<recipe>.json`.

*(The initial 4090 run already found **Recipe A best at 2 epochs** on Bitext - record the
supporting curve so that claim is backed, not asserted.)*

---

## Part C - Charts the paper needs (`triage_distill.eval.charts`)

Built + working: **`money_chart`** (hero), **`panel_leaderboard`**, **`model_bars`** (recipe/error-bars).
To build once the logs exist:
- **`learning_curves`** - val macro-F1 vs epoch, one line per recipe; marks best epoch + the diminishing-returns knee.
- **`data_efficiency`** - macro-F1 vs #examples/class, rationale vs ablation (the "less data" proof).
- **`savings_at_scale`** - the **business-impact** chart: $ to classify 1M tickets/month, student vs each panel model; annotate "$X/mo saved vs the cheapest cloud option." This is the one that makes a hiring manager care - narrow tasks (triage, sentiment, routing) at company scale.

---

## Part D - The resume artifact  → `docs/RESUME.md` (built AFTER the paper)

Distinct from the paper - tight, skimmable, numbers-forward.

- **One-line bullet** (template):
  > *Distilled a frontier LLM (Kimi K3) into a 4B QLoRA specialist retaining **⟦≥97.5%⟧** of teacher macro-F1 on support-ticket triage across two benchmarks (Bitext-27, CLINC-151) at **⟦~1%⟧** of inference cost; ran a controlled rationale-distillation ablation and benchmarked against a two-tier panel of 7 frontier + efficient models with pinned versions and a live cost snapshot.*
- **2–3 sentence blurb** - problem, method, result, the business "so what" (**$⟦X⟧ saved per 1M tickets** vs the cheapest cloud option).
- **Links** - repo, the `money_chart` hero image, HF model card + weights.
- **Interview ammo** - the honest framing: *prompted generalists vs a fine-tuned specialist*; why the ablation matters; when NOT to distill.

---

*See `SPEC.md` (plan), `docs/PRIMER.md` (plain-English method), `docs/HANDOFF-M2-4090.md` (training + what to log).*
