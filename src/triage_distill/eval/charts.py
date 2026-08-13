"""Portfolio charts for the frontier-panel comparison (M3 deliverables).

Two styled, data-driven figures, rendered from the same numbers `score.py` produces.
Colors + marks follow a house style (validated slot-1 blue accent, the fixed
categorical hue order, neutral inks, thin recessive gridlines). This is PLUMBING -
the numbers are yours (from M2/M3 eval); this only renders them.

- `model_bars()` - vertical bars comparing models on one metric, with **error
  bars** (± across seeds / bootstrap CI). The "did A/B beat the ablation by more than
  noise?" view. (Reference style: grouped color bars with caps + value labels.)
- `panel_leaderboard()` - small-multiple horizontal ranked bars, one panel per eval
  slice, our student highlighted. The "where does the tiny specialist land?" view.

    uv run python -m triage_distill.eval.charts --out <dir>          # PLACEHOLDER demo (light)
    uv run python -m triage_distill.eval.charts --out <dir> --dark   # dark variant
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- house palette (from the dataviz reference; light + dark are both "selected") ---
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", track="#ececea", bar="#c9c8c2", accent="#2a78d6")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835", track="#242422", bar="#4a4a47", accent="#3987e5")
# categorical slots in the fixed, CVD-validated order - never cycled past what's defined.
CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

_SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _p(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _apply_rc(dark: bool) -> None:
    p = _p(dark)
    plt.rcParams.update({
        "font.family": _SANS,
        "figure.facecolor": p["surface"], "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"], "savefig.bbox": "tight",
        "text.color": p["ink"], "axes.labelcolor": p["ink2"],
        "xtick.color": p["ink2"], "ytick.color": p["ink2"],
    })


def _titles(fig, title: str, subtitle: str, note: str | None, p: dict,
            title_y: float = 0.99, sub_y: float = 0.925) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", color=p["ink"], y=title_y)
    if subtitle:
        fig.text(0.5, sub_y, subtitle, ha="center", fontsize=11, color=p["ink2"])
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8.5, color=p["muted"], style="italic")


def _save(fig, out: Path | None):
    if out is None:
        return fig
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    return out


def model_bars(models, values, errors=None, *, title, subtitle="", ylabel="Score",
               ylim=None, hero=None, categorical=True, dark=False, note=None, out=None):
    """Vertical bar comparison with optional error bars + on-top value labels."""
    p = _p(dark); _apply_rc(dark)
    cats = CAT_DARK if dark else CAT_LIGHT
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=200)
    x = list(range(len(models)))
    if hero is not None:                       # single-highlight mode (hero vs muted)
        colors = [p["accent"] if m == hero else p["bar"] for m in models]
    elif categorical:                          # one hue per model (fixed order)
        colors = [cats[i % len(cats)] for i in x]
    else:
        colors = [p["accent"]] * len(models)

    ax.bar(x, values, width=0.62, color=colors, zorder=3,
           yerr=errors, capsize=5,
           error_kw=dict(ecolor=p["ink"], elinewidth=1.4, capthick=1.4))
    span = (ylim[1] - ylim[0]) if ylim else (max(values) - min(values) or 1)
    for xi, v in zip(x, values):
        e = errors[xi] if errors is not None else 0
        ax.text(xi, v + (e or 0) + span * 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, color=p["ink"])

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11, color=p["ink2"])
    ax.set_ylabel(ylabel, fontsize=11, color=p["ink2"])
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=p["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(p["axis"])
    ax.tick_params(length=0)
    fig.subplots_adjust(top=0.82, bottom=0.1)
    _titles(fig, title, subtitle, note, p, title_y=0.975, sub_y=0.9)
    return _save(fig, out)


def panel_leaderboard(panels, hero, *, teacher=None, title, subtitle="", scale_max=100.0,
                      dark=False, note=None, ncols=2, out=None):
    """Small-multiple ranked horizontal bars; `hero` model highlighted in every panel.

    panels: {panel_title: [(model, value), ...]} - each panel sorted descending here.
    """
    p = _p(dark); _apply_rc(dark)
    items = list(panels.items())
    nrows = math.ceil(len(items) / ncols)
    max_rows = max((len(r) for r in panels.values()), default=6)  # scale height to the tallest panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, nrows * (0.33 * max_rows + 0.7) + 0.9), dpi=200)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for ax, (ptitle, rows) in zip(axes, items):
        rows = sorted(rows, key=lambda r: r[1], reverse=True)
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        y = list(range(len(rows) - 1, -1, -1))  # first row on top
        ax.barh(y, [scale_max] * len(rows), height=0.62, color=p["track"], zorder=1)
        colors = [p["accent"] if n == hero else p["bar"] for n in names]
        ax.barh(y, vals, height=0.62, color=colors, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v + scale_max * 0.012, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=9.5, color=p["ink2"], fontfamily="monospace")
        ax.set_yticks(y)
        labels = ax.set_yticklabels(names, fontsize=9.5, color=p["ink"])
        for lab, n in zip(labels, names):        # weight the hero (+ teacher) names
            if n == hero:
                lab.set_color(p["accent"]); lab.set_fontweight("bold")
            elif n == teacher:
                lab.set_fontweight("bold")
        ax.set_xlim(0, scale_max * 1.14)
        ax.set_title(ptitle, fontsize=11, fontweight="bold", color=p["ink"], loc="left", pad=8)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    for ax in axes[len(items):]:
        ax.set_visible(False)
    fig.subplots_adjust(top=0.83, hspace=0.5, wspace=0.5)
    _titles(fig, title, subtitle, note, p, title_y=0.985, sub_y=0.9)
    return _save(fig, out)


def money_chart(benchmarks, hero, *, title, subtitle="",
                xlabel="Cost per 1k tickets (USD, log scale)", ylabel="Macro-F1",
                dark=False, note=None, ncols=2, out=None):
    """The hero image: cost (log x) vs macro-F1 (y), one panel per benchmark, hero highlighted.

    benchmarks: {name: [(model, cost_per_1k, macro_f1[, label_dy]), ...]} - the optional
    4th element nudges that row's label vertically (in points) to dodge collisions.
    The student sits high-and-left (accurate + ~free); flagships high-and-right
    (accurate + expensive); efficient tier between.
    """
    p = _p(dark); _apply_rc(dark)
    items = list(benchmarks.items())
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.9 * nrows), dpi=200)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for ax, (bname, rows) in zip(axes, items):
        rows = [(r + (0,))[:4] for r in rows]
        costs = [c for _, c, _, _ in rows]
        f1s = [f for _, _, f, _ in rows]
        for model, cost, f1, dy in rows:
            h = model == hero
            ax.scatter([cost], [f1], s=170 if h else 70, zorder=5 if h else 3,
                       color=p["accent"] if h else p["bar"], edgecolor=p["surface"], linewidth=1.5)
            ax.annotate(model, (cost, f1), xytext=(9, dy), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5,
                        color=p["accent"] if h else p["ink2"], fontweight="bold" if h else "normal")
        ax.set_xscale("log")
        ax.set_xlim(min(costs) * 0.45, max(costs) * 4.5)   # room for right-side labels
        ax.set_ylim(min(f1s) - 2, max(f1s) + 2)
        ax.set_xlabel(xlabel, fontsize=10, color=p["ink2"])
        ax.set_ylabel(ylabel, fontsize=10, color=p["ink2"])
        ax.set_title(bname, fontsize=12, fontweight="bold", color=p["ink"], loc="left", pad=8)
        ax.grid(True, color=p["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(p["axis"])
        ax.tick_params(length=0)

    for ax in axes[len(items):]:
        ax.set_visible(False)
    fig.subplots_adjust(top=0.80, wspace=0.26, bottom=0.13)
    _titles(fig, title, subtitle, note, p, title_y=0.995, sub_y=0.9)
    return _save(fig, out)


# --------------------------------------------------------------------------- demo
def _demo(out_dir: Path, dark: bool) -> None:
    note = "ILLUSTRATIVE - placeholder numbers, NOT measured results (M2/M3 not run yet)."
    suffix = "_dark" if dark else ""

    # image-2 style: the internal controlled experiment - A vs B vs ablation (all local, free)
    model_bars(
        models=["Ablation", "Recipe B", "Recipe A"],
        values=[86.4, 89.3, 89.9],
        errors=[0.6, 0.4, 0.4],
        title="Which recipe wins? (student · Bitext val)",
        subtitle="Error bars = ±1σ over 3 seeds · Recipe A = label-only at inference",
        ylabel="Macro-F1", ylim=(80, 93), hero="Recipe A", categorical=False,
        dark=dark, note=note, out=out_dir / f"model_bars{suffix}",
    )

    # image-1 style: the finalized 7-model panel + all three students, per benchmark; Recipe A hero
    P = ["Kimi K3 (teacher)", "GPT-5.6 Sol", "Gemini 3 Flash", "Fable 5",
         "Haiku 4.5", "GPT-5.6 Luna", "DeepSeek 3.2",
         "Student - Recipe A", "Student - Recipe B", "Student - ablation"]
    panel = {
        "Bitext-27 · macro-F1 (test)": list(zip(P, [91.0, 90.4, 88.2, 89.6, 87.5, 87.0, 85.8, 89.9, 89.3, 86.4])),
        "CLINC-151 · macro-F1 (test)": list(zip(P, [88.0, 87.1, 84.0, 86.2, 85.0, 84.5, 82.5, 85.9, 85.4, 81.0])),
    }
    panel_leaderboard(
        panel, hero="Student - Recipe A", teacher="Kimi K3 (teacher)",
        title="The panel vs. our distilled student",
        subtitle="Two tiers (flagship + efficient) vs. a fine-tuned 4B specialist · macro-F1",
        scale_max=100.0, dark=dark, note=note, out=out_dir / f"panel_leaderboard{suffix}",
    )

    # the hero image: cost vs accuracy - student high-and-left, flagships high-and-right
    money = {
        "Bitext-27": [("Student - Recipe A", 0.02, 89.9), ("DeepSeek 3.2", 0.28, 85.8),
                      ("GPT-5.6 Luna", 0.55, 87.0), ("Gemini 3 Flash", 0.62, 88.2),
                      ("Haiku 4.5", 1.08, 87.5), ("Kimi K3 (teacher)", 3.23, 91.0),
                      ("GPT-5.6 Sol", 5.45, 90.4), ("Fable 5", 15.0, 89.6)],
        "CLINC-151": [("Student - Recipe A", 0.02, 85.9), ("DeepSeek 3.2", 0.28, 82.5),
                      ("GPT-5.6 Luna", 0.55, 84.5), ("Gemini 3 Flash", 0.62, 84.0),
                      ("Haiku 4.5", 1.08, 85.0), ("Kimi K3 (teacher)", 3.23, 88.0),
                      ("GPT-5.6 Sol", 5.45, 87.1), ("Fable 5", 15.0, 86.2)],
    }
    money_chart(
        money, hero="Student - Recipe A",
        title="Accuracy vs. cost - the specialist owns the cheap corner",
        subtitle="Cost per 1k tickets (list API prices) vs. macro-F1 · student runs at ~$0",
        dark=dark, note=note, out=out_dir / f"money_chart{suffix}",
    )
    print(f"rendered demo charts (svg+png) to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".", help="output directory for the demo charts")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    _demo(Path(args.out), args.dark)
