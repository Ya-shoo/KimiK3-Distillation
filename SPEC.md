# Triage-Distill - Project Spec

**Distill Kimi K3 into a small, ~zero-cost support triage/escalation classifier that holds frontier-level accuracy at a fraction of the cost.**

> Portfolio thesis: *You pay frontier prices for generality you don't need on a narrow task. I bought ≥97.5% of the accuracy for ~1% of the cost - and here's the controlled experiment that proves it.*

---

## 1. Goal & success criteria

Prove **capability retention at a fraction of cost**, benchmarked against a panel of frontier models, then deploy the result as a real tool.

**Phase 1 pass gate (gates Phase 1 → Phase 2):**
- Student macro-F1 **≥ 97.5% of K3's** macro-F1 on the held-out benchmark test split, **AND**
- **≥ 20× cheaper** per 1,000 tickets (trivially cleared - local marginal cost ≈ electricity; the real gate is the accuracy number), **AND**
- Rationale-distilled student **beats the no-rationale ablation** (technique validated).

---

## 2. Two-phase architecture (generalize → specialize)

### Phase 1 - "the benchmark hero" (the résumé deliverable)
Distill K3 → small student on a **gold-labeled** public benchmark. Produce the accuracy-vs-cost chart against the frontier panel. **Freeze this checkpoint** - its numbers are the pitch.

### Phase 2 - "the deployed specialist" (the deployment story)
LoRA-adapt the frozen Phase-1 checkpoint on **DailyDles feedback**, add the `priority` + `escalate` heads (which never had public gold labels anyway), ship as a daily batch job. **Re-measure the benchmark after adapting** so the headline numbers stay honest.

*Why this split works:* gold-labeled `category` is exactly what Phase 1 needs; `priority`/`escalate` have no public gold labels, so they naturally belong to Phase 2 on owned data. The data reality and the sequencing agree.

---

## 3. Task definition

**Input:** a support/feedback message (string).
**Output:** compact JSON, produced in one forward pass:

```json
{ "category": "get_refund", "priority": "high", "escalate": true }
```

- `category` - multi-class intent (Bitext taxonomy). **Gold labels exist** → Phase 1.
- `priority` - ordinal (e.g. `low | medium | high | urgent`). **No gold** → Phase 2 (K3-silver + hand-labeled eval slice).
- `escalate` - binary; the business-critical head, tuned for **high recall** (missing a real escalation is the costly error). **No gold** → Phase 2.

Use **constrained/JSON-schema decoding** at inference so output is always parseable.

---

## 4. Data

| Source | Role | Labels |
|---|---|---|
| **Bitext customer-support** (~27 intents, ~27k rows) | Phase-1 benchmark spine | **Gold** category |
| *(optional)* CLINC150 (150 intents + out-of-scope) | robustness / "it generalizes" | Gold; OOS ≈ "escalate to human" |
| **DailyDles feedback (D1)** | Phase-2 dogfood + OOD eval + live demo | Self-defined (K3-silver + hand-labeled gold slice) |

Training data comes from **K3-labeled examples**, not hand-labeling - so low real-feedback volume is fine. Real feedback is the *live testbed* and a realistic eval, not the training fuel.

---

## 5. Method

### Rationale distillation ("Distilling Step-by-Step")
K3 (teacher) emits **label + reasoning trace** per example; the student trains on both (multi-task: predict rationale *and* label). This is the mechanism that lets a small model match a much larger one with less data - and it's the answer to the obvious interview jab *"why not just train on the gold labels?"*: (a) K3's rationales carry signal the labels don't, and (b) the real escalation task has **no gold labels at all**, so K3-as-teacher is load-bearing, not decorative.

Two inference modes to consider: keep the rationale at test time (slower, more accurate) vs. distill it into the weights and emit JSON directly (faster). Default: **train with rationale, emit JSON directly** at inference for speed; the rationale still improves the weights.

### Student model & training
- **Primary student:** Qwen3-4B, **QLoRA** (4-bit base + LoRA adapters) via Unsloth/TRL. Fits a 16 GB 4080 comfortably; ~7–8B is the practical QLoRA ceiling on 16 GB.
- *(optional extra axis)* also train **1.7B** → an "accuracy vs. size" frontier among your own students. Nice-to-have, not required.
- Teacher access: K3 via Together AI / Moonshot API. Label a few-thousand-example train subset (subsample for cost).

### The ablation (non-negotiable - turns "I fine-tuned a model" into "I ran a controlled experiment")
Train an identical student **without** rationales (labels only). The rationale-distilled student must beat it. One extra training run, large credibility payoff.

### Catastrophic-forgetting guard (protects the headline)
Phase 2 = a **separate LoRA adapter** on the frozen Phase-1 checkpoint (or mix a little benchmark data back in). **Re-measure the benchmark after adapting** - if the number moved, the pitch moved.

---

## 6. Evaluation - the money-chart

**Metrics**
- `category`: macro-F1 (headline), accuracy.
- `priority` (Phase 2): accuracy + quadratic-weighted kappa (ordinal).
- `escalate` (Phase 2): precision/recall/F1, reported **at a fixed high recall**.

**Frontier panel** (~6, evaluated few-shot, identical prompt + fixed label space, on the same test split):
1. **K3** - teacher / distillation source
2. **Anthropic flagship** (Fable 5 or Opus 4.8)
3. **OpenAI flagship** (GPT-5.6)
4. **Google Gemini flagship**
5. **A strong open model** (Kimi K2.x / DeepSeek / Qwen) - peer context
6. **A cheap mid-tier** (Haiku 4.5 / a mini) - the "even the cheap cloud option" point
7. **+ your student** (hero) and **the no-rationale ablation**

**Fairness framing (state it - it's the interview weapon, not a weakness):** this is *prompted generalists vs. a fine-tuned specialist* - the legitimate thesis of distillation. Same few-shot prompt for all; report each model's best reasonable prompt; **log model versions, dates, and a pricing snapshot** (APIs and prices are moving targets).

**The hero image:** scatter, **x = cost per 1k tickets (log scale), y = macro-F1**. Student sits high-and-left; frontier models high-and-right. Likely (and defensible) finding: the student **matches or beats** several prompted frontier models on this narrow task at ~0 marginal cost.

---

## 7. Deployment (Phase 2)

Because DailyDles sites are daily-refresh and static, the classifier runs as a **daily batch job on the 4080** - no always-on GPU server needed:

1. Pull new feedback from D1.
2. Local student → `{category, priority, escalate}` per item.
3. Write results back to D1 / surface a triage view.

*(Optional C-seed)* wrap the student behind a tiny inference API - the productizable "nutrition-facts-for-support-tickets" service.

---

## 8. Milestones (part-time, ~3–5 weekends for Phase 1 + 2)

- **M0 - Scaffold:** repo, download Bitext, eval harness, wire K3 + frontier APIs. *(~few days)*
- **M1 - Teacher labeling:** K3 rationales over train subset. *(~1–2 days)*
- **M2 - Train:** student QLoRA + no-rationale ablation. *(~2–3 days)*
- **M3 - Eval + chart:** frontier panel, accuracy-vs-cost chart, check **pass gate**. ← **Phase 1 done / portfolio deliverable.** *(~2–3 days)*
- **M4 - Dogfood data:** pull DailyDles feedback, K3-label priority/escalate, hand-label gold eval slice. *(~few days)*
- **M5 - Phase 2:** adapter + re-measure benchmark + deploy daily batch + tiny demo. *(~few days)*
- **M6 - Writeup:** blog post, README, model card, polish.

---

## 9. Cost

- K3 teacher labeling: ~$10–30
- Frontier panel eval (~500 examples × ~6 models): ~$10–20
- GPU: your own 4080
- **Total: < $50**

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Student can't hit 97.5% | Bigger student (up to ~7–8B QLoRA on 16 GB), more/cleaner teacher data, better prompts. Worst case: report the honest frontier - still a good result. |
| Catastrophic forgetting in Phase 2 | Frozen Phase-1 ckpt + separate adapter + re-measure benchmark. |
| Bitext is synthetic/easy → inflated scores | Add a harder real set (CLINC150 or Twitter customer-support) and/or the dogfood OOD slice. |
| Frontier API access/pricing drift | Pin model versions + snapshot prices; note the eval date. |
| JSON parse failures | Constrained / JSON-schema decoding. |

---

## 11. Portfolio deliverables

- **GitHub repo** - clean, reproducible, README leads with the chart.
- **Blog post / writeup** - narrative + method + honest limitations.
- **The accuracy-vs-cost chart** - the hero image.
- **Live demo** - paste a ticket → triage JSON, running the local student; plus the dogfood running on the sites.
- **Model card + distilled weights on Hugging Face** - publish the specialist.

---

## 12. Parked / optional

- Second benchmark (CLINC150) for cross-dataset robustness.
- Multiple student sizes (1.7B / 4B / 8B) → an accuracy-vs-size frontier of your own.
- API productization (the C-seed).
- KDA architecture project - deferred, revisit after this ships.
