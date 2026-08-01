"""Frontier-lab-style release cards (the Twitter/X launch-graphic treatment).

Four dark 16:9 social cards rendered from the same committed artifacts as
`render_charts` — big numbers, accent-vs-gray bars, footnoted eval conditions:

  card1_bench     — the grouped benchmark bars (student in blue, field in gray)
  card2_headline  — the giant "104.5%" hero stat
  card3_cost      — "$2,200 -> $25" savings bars
  card4_scale     — the 2.8T vs 4B area-true circle visual

    uv run python -m triage_distill.eval.render_release_cards   # -> artifacts/cards/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .render_charts import PANEL_ROWS, STUDENT_TEST

OUT = Path(__file__).resolve().parents[3] / "artifacts" / "cards"

BG = "#141413"          # card ground (between the validated dark surface and black)
SURF = "#1a1a19"        # validated dark chart surface (panel fill)
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
BAR = "#3d3d3a"
ACCENT = "#3987e5"      # validated dark-mode slot-1 blue
FOOT = ("Held-out test sets, one scored pass per system, shared scorer · panel prompted "
        "label-only, reasoning off · pinned list prices · 2026-07-31")
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _fig(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG)
    plt.rcParams.update({"font.family": SANS})
    return fig


def _brand(fig, tagline):
    fig.text(0.055, 0.935, "Triage-Distill 4B", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.055, 0.885, tagline, fontsize=12.5, color=INK2)
    fig.text(0.055, 0.045, FOOT, fontsize=7.6, color=MUTED, style="italic")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=BG, bbox_inches=None)
    plt.close(fig)


def card1_bench():
    fig = _fig()
    _brand(fig, "A 4B student distilled from Kimi K3 (2.8T) — evaluated against the frontier field")
    order = ["Kimi K3 (teacher)", "Gemini 3 Flash", "GPT-5.6 Luna", "DeepSeek 3.2", "Haiku 4.5"]
    short = {"Kimi K3 (teacher)": "Kimi K3\n(teacher)", "Gemini 3 Flash": "Gemini 3\nFlash",
             "GPT-5.6 Luna": "GPT-5.6\nLuna", "DeepSeek 3.2": "DeepSeek\n3.2", "Haiku 4.5": "Haiku\n4.5"}
    benches = [("Bitext-27 · intent triage (macro-F1)", 0, "recipe A"),
               ("CLINC-151 + out-of-scope (macro-F1)", 1, "ablation")]
    for pi, (btitle, ds_i, brecipe) in enumerate(benches):
        ax = fig.add_axes([0.075 + pi * 0.48, 0.17, 0.40, 0.60])
        ax.set_facecolor(BG)
        stu = max(v[ds_i] for v in STUDENT_TEST.values())
        vals = [("Triage-\nDistill 4B", stu, True)] + [(short[m], PANEL_ROWS[m][ds_i], False) for m in order]
        x = range(len(vals))
        ax.bar(x, [v for _, v, _ in vals], width=0.68,
               color=[ACCENT if h else BAR for _, _, h in vals], zorder=3)
        for xi, (_, v, h) in zip(x, vals):
            ax.text(xi, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=15 if h else 12.5,
                    fontweight="bold", color=INK if h else INK2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([n for n, _, _ in vals], fontsize=9.5, color=INK2, linespacing=1.15)
        ax.get_xticklabels()[0].set_color(ACCENT)
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_ylim(0, 112)
        ax.set_yticks([])
        ax.set_title(btitle, fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0, color="#4a4a47", linewidth=1.2)
        ax.tick_params(length=0)
        note = f"student = {brecipe} checkpoint" + ("  ·  #1 of 8 systems" if pi == 0 else "  ·  #2 of 8, above the teacher")
        ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8.6, color=MUTED)
    _save(fig, "card1_bench")


def card2_headline():
    fig = _fig()
    _brand(fig, "A 4B student distilled from Kimi K3 (2.8T)")
    fig.text(0.5, 0.60, "104.5%", fontsize=104, fontweight="bold", color=ACCENT,
             ha="center", va="center")
    fig.text(0.5, 0.415, "of its 2.8-trillion-parameter teacher's accuracy —\nbeating the teacher outright on both benchmarks",
             fontsize=16.5, color=INK, ha="center", va="center", linespacing=1.45)
    stats = [("99.2", "Bitext-27 · test macro-F1\n#1 of 8 systems"),
             ("92.2", "CLINC-151 · test macro-F1\nteacher: 90.8"),
             (r"≤\$25", "per 1M tickets served\nvs \\$190–\\$2,200 cloud")]
    for i, (big, small) in enumerate(stats):
        cx = 0.235 + i * 0.265
        fig.text(cx, 0.235, big, fontsize=30, fontweight="bold", color=INK, ha="center")
        fig.text(cx, 0.155, small, fontsize=9.5, color=INK2, ha="center", linespacing=1.4)
    _save(fig, "card2_headline")


def card3_cost():
    fig = _fig()
    _brand(fig, "Classifying one million support tickets a month")
    rows = [("Kimi K3 (teacher)", 2200), ("Haiku 4.5", 790), ("Gemini 3 Flash", 390),
            ("GPT-5.6 Luna", 350), ("DeepSeek 3.2", 190), ("Triage-Distill 4B", 25)]
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.62])
    ax.set_facecolor(BG)
    y = list(range(len(rows) - 1, -1, -1))
    for yi, (name, v) in zip(y, rows):
        hero = name.startswith("Triage")
        ax.barh(yi, v, height=0.62, color=ACCENT if hero else BAR, zorder=3)
        ax.text(v + 28, yi, rf"\${v:,}", va="center", fontsize=15 if hero else 12.5,
                fontweight="bold", color=INK if hero else INK2)
        if hero:
            ax.text(365, yi, "hosted-API ceiling · ~\\$2–3 electricity on one RTX 4090",
                    va="center", fontsize=9.5, color=MUTED, style="italic")
    ax.set_yticks(y)
    labels = ax.set_yticklabels([r[0] for r in rows], fontsize=12.5, color=INK2)
    labels[-1].set_color(ACCENT)
    labels[-1].set_fontweight("bold")
    ax.set_xlim(0, 2560)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.055, 0.80, "88× cheaper than its teacher — while outscoring it",
             fontsize=17, fontweight="bold", color=INK)
    _save(fig, "card3_cost")


def card4_scale():
    fig = _fig()
    _brand(fig, "Distillation: pay a frontier model once as a teacher, not forever as an employee")
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.70])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    big_r, small_r = 2.55, 2.55 / 26.5  # area-true: 2.8T vs 4B params
    ax.add_patch(Circle((3.4, 3.1), big_r, color=BAR))
    ax.add_patch(Circle((9.2, 3.1), small_r, color=ACCENT))
    ax.add_patch(Circle((9.2, 3.1), 0.5, fill=False, edgecolor=ACCENT, linewidth=1.2, alpha=0.55))
    ax.annotate("", xy=(8.55, 3.1), xytext=(6.35, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.6))
    ax.text(6.9, 3.42, "distill", fontsize=11.5, color=INK2, ha="left")
    ax.text(3.4, 3.1, "Kimi K3\n2.8T params", fontsize=14, fontweight="bold",
            color=INK, ha="center", va="center", linespacing=1.5)
    ax.text(9.2, 4.25, "Triage-Distill 4B", fontsize=14, fontweight="bold", color=ACCENT, ha="center")
    ax.text(9.2, 1.95, "≈1/700th the parameters (area-true)\nhigher test accuracy on both benchmarks",
            fontsize=10, color=INK2, ha="center", linespacing=1.5)
    ax.text(6.0, 0.35, r"Built for <\$100 of API calls + a few GPU-hours on one RTX 4090",
            fontsize=11, color=INK2, ha="center")
    _save(fig, "card4_scale")


if __name__ == "__main__":
    for f in (card1_bench, card2_headline, card3_cost, card4_scale):
        f()
    print(f"rendered 4 release cards to {OUT}")
