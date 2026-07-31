"""Render the paper's chart set (PAPER-OUTLINE Part C) from committed artifacts.

Unlike `charts.py --out` (a placeholder demo), this reads REAL numbers from
`artifacts/` and renders every buildable spec'd chart, light + dark:

- money_chart          (hero: test macro-F1 vs $/1k, both benchmarks)
- panel_leaderboard    (8 systems x 2 benchmarks, test)
- model_bars_{bitext,clinc}  (recipe ablation, val mean +/- seed sigma)
- learning_curves      (val macro-F1 vs epoch, incl. the 6-epoch control)
- savings_at_scale     ($ per 1M tickets/month, honest student cost range)

data_efficiency stays unbuilt: the 10/20/30/40-per-class runs were never made.

    uv run python -m triage_distill.eval.render_charts          # -> artifacts/charts/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .charts import CAT_DARK, CAT_LIGHT, _apply_rc, _p, _save, _titles, model_bars, money_chart, panel_leaderboard

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "charts"


def _j(rel: str):
    return json.loads((ROOT / rel).read_text())


def _epochs(rel: str):
    d = _j(rel)
    eps = d.get("epochs", d) if isinstance(d, dict) else d
    return [(e["epoch"], (e.get("macro_f1") or e.get("val_macro_f1")) * 100) for e in eps]


# ---------------------------------------------------------------- data (committed)
BIT = _j("artifacts/eval/findings.json")
CLI = _j("artifacts/clinc/eval/findings.json")
PANEL = {
    "bitext": _j("artifacts/bitext/eval/panel/panel_summary.json"),
    "clinc": _j("artifacts/clinc/eval/panel/panel_summary.json"),
}

STUDENT_TEST = {  # (bitext, clinc) test macro-F1 x100, from findings.json test_eval
    "Student — recipe A": (BIT["test_eval"]["recipes"]["recipe_a"]["test_macro_f1"] * 100,
                           CLI["test_eval"]["recipes"]["recipe_a"]["test_macro_f1"] * 100),
    "Student — ablation": (BIT["test_eval"]["recipes"]["ablation"]["test_macro_f1"] * 100,
                           CLI["test_eval"]["recipes"]["ablation"]["test_macro_f1"] * 100),
    "Student — recipe B": (BIT["test_eval"]["recipes"]["recipe_b"]["test_macro_f1"] * 100,
                           CLI["test_eval"]["recipes"]["recipe_b"]["test_macro_f1"] * 100),
}
PANEL_ROWS = {  # label -> (bitext F1x100, clinc F1x100, $/1k mean of the two runs)
    m["label"]: [None, None, None] for m in PANEL["bitext"]["models"]
}
for ds_i, ds in enumerate(("bitext", "clinc")):
    for m in PANEL[ds]["models"]:
        PANEL_ROWS[m["label"]][ds_i] = m["macro_f1"] * 100
        PANEL_ROWS[m["label"]][2] = (PANEL_ROWS[m["label"]][2] or 0) + m["cost_per_1k_tickets_list"] / 2

# Student serving cost: plotted at the HOSTED-API CEILING ($0.03/1k) — the
# conservative end of the honest $0.002-$0.03 range (PAPER.md §8) — so the chart
# never flatters the specialist.
STUDENT_COST = 0.03
COST_NOTE = (r"Student plotted at its hosted-API cost ceiling (\$0.03/1k); owned-GPU "
             r"electricity is ~\$0.002/1k. Panel costs: pinned list prices × measured tokens.")


def render(dark: bool) -> None:
    sfx = "_dark" if dark else ""
    cats = CAT_DARK if dark else CAT_LIGHT

    # 1 — money chart (hero): best student per benchmark + full panel.
    # One shared display name for the student point so the hero highlight
    # matches in both panels (recipe A on Bitext, ablation on CLINC).
    money = {}
    for ds_i, bench in ((0, "Bitext-27 (synthetic) · test"), (1, "CLINC-151 (real) · test")):
        best = max(STUDENT_TEST.items(), key=lambda kv: kv[1][ds_i])
        NUDGE = {"Gemini 3 Flash": 9, "GPT-5.6 Luna": -2, "DeepSeek 3.2": -10} if ds_i == 0 else {}
        rows = [("Student (Qwen3-4B, local)", STUDENT_COST, best[1][ds_i], 0)]
        rows += [(lbl, v[2], v[ds_i], NUDGE.get(lbl, 0)) for lbl, v in PANEL_ROWS.items()]
        money[bench] = rows
    money_chart(
        money, hero="Student (Qwen3-4B, local)",
        title="Accuracy vs. cost — the specialist owns the cheap corner",
        subtitle="Held-out test, scored once · best student recipe per benchmark (recipe A on Bitext, ablation on CLINC)",
        dark=dark, note=COST_NOTE, out=OUT / f"money_chart{sfx}",
    )

    # 2 — panel leaderboard: all 8 systems, both benchmarks
    panels = {}
    for ds_i, bench in ((0, "Bitext-27 · macro-F1 (test)"), (1, "CLINC-151 · macro-F1 (test)")):
        rows = [(lbl, v[ds_i]) for lbl, v in STUDENT_TEST.items()]
        rows += [(lbl, v[ds_i]) for lbl, v in PANEL_ROWS.items()]
        panels[bench] = rows
    panel_leaderboard(
        panels, hero="Student — recipe A", teacher="Kimi K3 (teacher)",
        title="The frontier panel vs. the distilled students",
        subtitle="Held-out test, one scored pass per system, shared scorer · students in blue",
        dark=dark, note="Teacher in bold. Gate: ≥97.5% of teacher — recipe B fails it on CLINC (96.1%).",
        out=OUT / f"panel_leaderboard{sfx}",
    )

    # 3 — model bars: the controlled ablation, val mean ± seed σ
    sv_b = BIT["seed_variance"]["families"]
    model_bars(
        models=["Ablation", "Recipe A", "Recipe B"],
        values=[sv_b["ablation"]["macro_f1_mean"] * 100, sv_b["recipe_a"]["macro_f1_mean"] * 100,
                sv_b["recipe_b"]["macro_f1_mean"] * 100],
        errors=[sv_b["ablation"]["macro_f1_sigma"] * 100, sv_b["recipe_a"]["macro_f1_sigma"] * 100,
                sv_b["recipe_b"]["macro_f1_sigma"] * 100],
        title="Bitext-27: the rationale recipe WINS (+1.3 pt, ~8σ)",
        subtitle="Val macro-F1, mean ±1σ over seeds {42, 1337, 2024} · best epoch (2)",
        ylabel="Macro-F1", ylim=(97, 99.8), hero="Recipe A", categorical=False,
        dark=dark, out=OUT / f"model_bars_bitext{sfx}",
    )
    sv_c = CLI["seed_variance"]["families"]
    model_bars(
        models=["Ablation", "Recipe A", "Recipe B", "Abl · 6-ep ctrl", "Abl + back-fill"],
        values=[sv_c["ablation"]["macro_f1_mean"] * 100, sv_c["recipe_a"]["macro_f1_mean"] * 100,
                sv_c["recipe_b"]["macro_f1_mean"] * 100, sv_c["ablation_ctrl6"]["macro_f1_mean"] * 100, 97.15],
        errors=[sv_c["ablation"]["macro_f1_sigma"] * 100, sv_c["recipe_a"]["macro_f1_sigma"] * 100,
                sv_c["recipe_b"]["macro_f1_sigma"] * 100, 0, 0],
        title="CLINC-151: the rationale recipe LOSES (−0.5 pt, ~4σ)",
        subtitle="Val macro-F1, ±1σ over 3 seeds · best epoch (3) · ctrl + back-fill are n=1",
        ylabel="Macro-F1", ylim=(92.5, 98), hero="Ablation", categorical=False,
        dark=dark, out=OUT / f"model_bars_clinc{sfx}",
    )

    # 4 — learning curves (NEW, spec'd in Part C): val macro-F1 vs epoch
    p = _p(dark)
    _apply_rc(dark)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=200)
    series = {
        "Bitext-27 (seed 42, fixed schedule)": [
            ("Ablation", _epochs("artifacts/eval/epoch_scores_ablation.json")),
            ("Recipe A", _epochs("artifacts/eval/epoch_scores_bitext_recipe_a_s42v2.json")),
            ("Recipe B", _epochs("artifacts/eval/epoch_scores_recipe_b.json")),
        ],
        "CLINC-151 (seed 42)": [
            ("Ablation", _epochs("artifacts/clinc/eval/epoch_scores_clinc_ablation_s42.json")),
            ("Recipe A", _epochs("artifacts/clinc/eval/epoch_scores_clinc_recipe_a_s42.json")),
            ("Recipe B", _epochs("artifacts/clinc/eval/epoch_scores_clinc_recipe_b_s42.json")),
            ("Abl · 6-ep ctrl", _epochs("artifacts/clinc/eval/epoch_scores_clinc_ablation_ctrl6_s42.json")),
        ],
    }
    for ax, (bench, lines) in zip(axes, series.items()):
        for i, (name, pts) in enumerate(lines):
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            style = dict(color=cats[i % len(cats)], linewidth=2)
            if name.startswith("Abl ·"):
                style.update(color=cats[0], linestyle=(0, (3, 3)), linewidth=1.6)
            ax.plot(xs, ys, marker="o", markersize=5, markeredgecolor=p["surface"],
                    markeredgewidth=1.2, zorder=3, **style)
            best = max(pts, key=lambda q: q[1])
            ax.scatter([best[0]], [best[1]], s=90, color=style["color"], zorder=4,
                       edgecolor=p["surface"], linewidth=1.5)
            LDY = {"Ablation": 9, "Recipe A": -4}
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, LDY.get(name, 0) if len(lines) > 3 else 0),
                        textcoords="offset points",
                        va="center", fontsize=8.5, color=style["color"], fontweight="bold")
        ax.set_title(bench, fontsize=12, fontweight="bold", color=p["ink"], loc="left", pad=8)
        ax.set_xlabel("Epoch", fontsize=10, color=p["ink2"])
        ax.set_ylabel("Val macro-F1", fontsize=10, color=p["ink2"])
        ax.set_xticks(sorted({q[0] for _, pts in lines for q in pts}))
        ax.grid(True, color=p["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xmargin(0.22)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(p["axis"])
        ax.tick_params(length=0)
    fig.subplots_adjust(top=0.78, wspace=0.24, bottom=0.14)
    _titles(fig, "Learning curves — the knee is at epoch 2 (Bitext) / 3 (CLINC)",
            "Large dot = best epoch (val selection) · the 6-epoch control shows CLINC flattens after 3",
            "Bitext recipe A curve is the corrected fixed-schedule re-run (s42v2).", p,
            title_y=0.965, sub_y=0.885)
    _save(fig, OUT / f"learning_curves{sfx}")

    # 5 — savings at scale (NEW, spec'd in Part C): $/1M tickets/month
    _apply_rc(dark)
    rows = [("Kimi K3 (teacher)", 2200, None), ("Haiku 4.5", 790, None),
            ("Gemini 3 Flash", 390, None), ("GPT-5.6 Luna", 350, None),
            ("DeepSeek 3.2", 190, None),
            ("Student (Qwen3-4B)", 25, r"≤\$25 hosted-API ceiling · ~\$2–3 electricity on the owned 4090")]
    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=200)
    y = list(range(len(rows) - 1, -1, -1))
    vals = [r[1] for r in rows]
    colors = [p["accent"] if r[2] else p["bar"] for r in rows]
    ax.barh(y, vals, height=0.6, color=colors, zorder=3)
    for yi, (name, v, extra) in zip(y, rows):
        ax.text(v + 22, yi, rf"\${v:,}", va="center", fontsize=10.5,
                color=p["ink"], fontweight="bold" if extra else "normal")
        if extra:
            ax.text(v + 215, yi, extra, va="center", fontsize=8.5, color=p["ink2"], style="italic")
    ax.set_yticks(y)
    labels = ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=p["ink"])
    labels[-1].set_color(p["accent"])
    labels[-1].set_fontweight("bold")
    ax.set_xlim(0, 2560)
    ax.set_xlabel("USD per 1,000,000 tickets / month (list prices, measured tokens)",
                  fontsize=10, color=p["ink2"])
    ax.grid(axis="x", color=p["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.subplots_adjust(top=0.80, left=0.2, bottom=0.15)
    _titles(fig, "Savings at scale — 1M tickets a month",
            "Same job, same test set: the student outscores every model below the teacher's price", None, p,
            title_y=0.96, sub_y=0.87)
    _save(fig, OUT / f"savings_at_scale{sfx}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        render(dark)
    print(f"rendered chart set (light+dark, svg+png) to {OUT}")
