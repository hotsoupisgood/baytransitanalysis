#!/usr/bin/env python3
"""
plots.py -- the model figures, in the A0 house style (analysis/a0_plots.py:
one palette, left-aligned titles, a Scope footnote on every PNG). Every figure
from the route-stratified fit is stamped POOLED in title, footnote and
filename; every figure from the VADER stand-in says STAND-IN: VADER, not
stance in its title. Nothing here computes a statistic: tables in, pictures out.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import a0_plots as ap  # noqa: E402

FAM_COL = ap.FAM_COL
ROLE_COL = ap.ROLE_COL
ACC = "#2a78d6"
INK = ap.P["ink"]
MUTED = ap.P["muted"]
FS = ap.FS


def _prep(scope, pooled):
    ap.SCOPE = scope
    ap.POOLED = pooled
    plt.rcParams["font.family"] = ["system-ui", "-apple-system", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]


def _title(text, pooled, standin=None):
    if standin:
        text = f"{standin} -- {text}"
    return ("POOLED (route as stratum): " if pooled else "") + text


def _label(b):
    return ap.BODY_NAME.get(b, b.title())


# ----------------------------------------------------------- estimand 1 ------
def shrinkage(bodies, path, scope, pooled, subtitle=""):
    """Crude body rate against the posterior body rate with 90 % interval."""
    _prep(scope, pooled)
    df = bodies.dropna(subset=["raw_rate"]).copy()
    fig, axes = ap.figure(1, 1, _title("Shrinkage: crude rate vs posterior body rate", pooled),
                          subtitle or "episode-minutes per meeting-hour; dot = posterior median, bar = 90 % interval, size = exposure hours",
                          (9, 7.5))
    ax = axes[0, 0]; ap.style(ax, ygrid=True, xgrid=True)
    floor = 0.1
    x = np.maximum(df["raw_rate"], floor); y = df["post_median"]
    size = 18 + 2.2 * np.sqrt(df["exposure_h"])
    ax.vlines(x, df["lo90"], df["hi90"], color=ACC, alpha=0.35, linewidth=1.4)
    ax.scatter(x, y, s=size, color=ACC, edgecolors=ap.P["surface"], linewidths=1.0, zorder=3)
    lim = (floor * 0.8, max(x.max(), df["hi90"].max()) * 1.3)
    ax.plot(lim, lim, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    for _, r in df.iterrows():
        ax.annotate(_label(r["body"]), (max(r["raw_rate"], floor), r["post_median"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8.5, color=ap.P["ink2"])
    ax.set_xlabel("crude rate (minutes located / meeting-hours read), min per h; zeros drawn at 0.1", color=ap.P["ink2"])
    ax.set_ylabel("posterior body rate, min per meeting-hour", color=ap.P["ink2"])
    ap.footnote(fig, "Bodies with little exposure are pulled toward the mean (dashed = no shrinkage). Rate is the family mean at the "
                     "reference route, month-averaged, conditional on an active cell.")
    ap.save(fig, path, (0, 0.05, 1, 0.9))


def body_forest(bodies, path, scope, pooled, subtitle="", family_cols=None):
    """Ranked body rates with 90 % intervals; the two family rates as small dots."""
    _prep(scope, pooled)
    df = bodies.sort_values("post_median").reset_index(drop=True)
    h = max(5.0, 0.32 * len(df) + 2.2)
    fig, axes = ap.figure(1, 1, _title("Posterior body rates, ranked", pooled),
                          subtitle or "episode-minutes per meeting-hour; bar = 90 % interval; x = crude rate; coloured dots = density / transit",
                          (9, h))
    ax = axes[0, 0]; ap.style(ax, ygrid=False, xgrid=True)
    yy = np.arange(len(df))
    ax.hlines(yy, df["lo90"], df["hi90"], color=ACC, linewidth=2.0, alpha=0.6)
    ax.scatter(df["post_median"], yy, s=38, color=ACC, zorder=3)
    ax.scatter(df["raw_rate"].fillna(0).clip(upper=ax.get_xlim()[1] if False else None), yy, marker="x", s=30, color=INK, zorder=4, linewidths=1)
    for fam, col in FAM_COL.items():
        c = f"{fam}_median"
        if c in df.columns:
            ax.scatter(df[c], yy + (0.18 if fam == "transit" else -0.18), s=16, color=col, zorder=3, alpha=0.9)
    ax.set_yticks(yy); ax.set_yticklabels([f"{_label(b)}  ({int(round(e))} h)" for b, e in zip(df["body"], df["exposure_h"])])
    ax.set_xscale("log")
    ax.set_xlabel("minutes per meeting-hour (log scale)", color=ap.P["ink2"])
    handles = [plt.Line2D([], [], color=ACC, marker="o", linestyle="-", linewidth=2), plt.Line2D([], [], color=INK, marker="x", linestyle=""),
               plt.Line2D([], [], color=FAM_COL["density"], marker="o", linestyle=""), plt.Line2D([], [], color=FAM_COL["transit"], marker="o", linestyle="")]
    ap.legend(ax, handles, ["posterior median, 90 %", "crude rate", "density", "transit"], loc="lower right")
    ap.footnote(fig, "Exposure hours in brackets. A body with two cells and no episodes sits near the prior mean with a wide interval: that is the point.")
    ap.save(fig, path, (0, 0.04, 1, 0.92))


def contrast(contrast_df, per_route_rows, path, scope, pooled):
    """Transit / density rate ratio: pooled posterior and each per-route fit."""
    _prep(scope, pooled)
    fig, axes = ap.figure(1, 1, _title("Density vs transit: rate ratio transit / density", pooled),
                          "posterior median and 90 % interval; 1 = equal minutes per meeting-hour", (9, 4.8))
    ax = axes[0, 0]; ap.style(ax, ygrid=False, xgrid=True)
    rows = [("stratified (POOLED over routes)", contrast_df.iloc[0])] + per_route_rows
    yy = np.arange(len(rows))[::-1]
    for y, (lab, r) in zip(yy, rows):
        col = ACC if lab.startswith("stratified") else MUTED
        ax.hlines(y, r["ratio_lo90"], r["ratio_hi90"], color=col, linewidth=2.2)
        ax.scatter([r["ratio_median"]], [y], s=42, color=col, zorder=3)
    ax.axvline(1.0, color=INK, linewidth=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([lab for lab, _ in rows]); ax.set_xscale("log")
    ax.set_xlabel("rate ratio (log scale)", color=ap.P["ink2"])
    ap.footnote(fig, "Per-route rows are separate fits on one locate route each (never pooled); the top row pools with route as a stratum.")
    ap.save(fig, path, (0, 0.06, 1, 0.86))


def months(month_df, path, scope, pooled):
    """Monthly rate per family with 90 % band and the crude monthly rate."""
    _prep(scope, pooled)
    fig, axes = ap.figure(1, 1, _title("Monthly trajectory per family", pooled),
                          "random-walk month effect; line = posterior median, band = 90 %; dots = crude monthly rate (size = exposure)", (10, 5.2))
    ax = axes[0, 0]; ap.style(ax)
    for fam, g in month_df.groupby("family"):
        g = g.sort_values("month"); x = np.arange(len(g)); col = FAM_COL.get(fam, ACC)
        ax.fill_between(x, g["lo90"], g["hi90"], color=col, alpha=0.15, linewidth=0)
        ax.plot(x, g["rate_median"], color=col, linewidth=2.0, label=fam)
        ax.scatter(x, g["raw_rate"], s=10 + 0.6 * np.sqrt(g["exposure_h"].fillna(0)), color=col, alpha=0.7, edgecolors="none")
        ax.set_xticks(x); ax.set_xticklabels(list(g["month"]), rotation=45, ha="right")
    ax.set_ylabel("minutes per meeting-hour", color=ap.P["ink2"]); ax.set_ylim(bottom=0)
    ap.legend(ax, *ax.get_legend_handles_labels(), loc="upper left")
    ap.footnote(fig, "Each month's crude rate pools all bodies read that month, so it moves with WHICH bodies met; the model's body intercepts absorb that.")
    ap.save(fig, path, (0, 0.06, 1, 0.9))


def ppc(ppc_df, path, scope, pooled, likelihood="zinb", compare=None):
    """Observed vs replicated zero share and total minutes, all cells and per family."""
    _prep(scope, pooled)
    subsets = [s for s in ("all", "family=density", "family=transit") if s in set(ppc_df["subset"])]
    stats = [("zero_share", "share of zero cells"), ("total_minutes", "total minutes"), ("max_cell", "largest cell (min)")]
    fig, axes = ap.figure(1, len(stats), _title(f"Posterior predictive check ({likelihood.upper()})", pooled),
                          "bar = 90 % interval of the replicated statistic, dot = its median, x = observed" +
                          ("; grey = plain NB for comparison" if compare is not None else ""), (11, 4.6))
    for ax, (stat, lab) in zip(axes[0], stats):
        ap.style(ax, ygrid=False, xgrid=True)
        yy = np.arange(len(subsets))[::-1]
        for y, s in zip(yy, subsets):
            r = ppc_df[(ppc_df["subset"] == s) & (ppc_df["stat"] == stat)].iloc[0]
            ax.hlines(y, r["rep_lo90"], r["rep_hi90"], color=ACC, linewidth=2.4)
            ax.scatter([r["rep_median"]], [y], s=36, color=ACC, zorder=3)
            ax.scatter([r["observed"]], [y], s=60, marker="x", color=INK, zorder=4, linewidths=1.6)
            if compare is not None:
                c = compare[(compare["subset"] == s) & (compare["stat"] == stat)]
                if len(c):
                    c = c.iloc[0]
                    ax.hlines(y - 0.28, c["rep_lo90"], c["rep_hi90"], color=MUTED, linewidth=1.6)
                    ax.scatter([c["rep_median"]], [y - 0.28], s=20, color=MUTED, zorder=3)
        ax.set_yticks(yy); ax.set_yticklabels([s.replace("family=", "") for s in subsets])
        ap.panel_title(ax, lab)
        if stat != "zero_share":
            ax.set_xscale("log")
    ap.footnote(fig, "A replicated interval that misses the x is misfit. Cell maxima are bounded by meeting minutes in reality but not in the likelihood.")
    ap.save(fig, path, (0, 0.08, 1, 0.86))


# ----------------------------------------------------------- estimand 2 ------
def role_gap_forest(gap_rows, path, instrument, scope, outcome="stance", subtitle=""):
    """Public - dais latent gap per family, per role instrument; STAND-IN in the
    title when the rows come from VADER."""
    standin = instrument if instrument.startswith("STAND-IN") else None
    _prep(scope, False)
    title = _title(f"Role gap, {outcome}: public minus dais on the latent scale", False, standin)
    fig, axes = ap.figure(1, 2, title, subtitle or ("cumulative logit; dot = posterior median, bar = 90 %; right: difference in the "
                                                      f"'{gap_rows['tail'].iloc[0]}' share at the mean body / episode / length"), (11, 4.9))
    labels = [f"{r['family']}  [{r['role_source']}]  n={int(r['n_turns'])}" for _, r in gap_rows.iterrows()]
    yy = np.arange(len(labels))[::-1]
    ax = axes[0, 0]; ap.style(ax, ygrid=False, xgrid=True)
    for y, (_, r) in zip(yy, gap_rows.iterrows()):
        col = FAM_COL.get(r["family"], ACC)
        ax.hlines(y, r["gap_lo90"], r["gap_hi90"], color=col, linewidth=2.2)
        ax.scatter([r["gap_latent_median"]], [y], s=40, color=col, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8); ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_xlabel("log-odds shift (public - dais); > 0 = public higher on the scale", color=ap.P["ink2"])
    ap.panel_title(ax, "latent gap")
    ax = axes[0, 1]; ap.style(ax, ygrid=False, xgrid=True)
    for y, (_, r) in zip(yy, gap_rows.iterrows()):
        if np.isnan(r["p_tail_diff_median"]):
            continue
        col = FAM_COL.get(r["family"], ACC)
        ax.hlines(y, r["p_tail_diff_lo90"], r["p_tail_diff_hi90"], color=col, linewidth=2.2)
        ax.scatter([r["p_tail_diff_median"]], [y], s=40, color=col, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8); ax.set_yticks(yy); ax.set_yticklabels([""] * len(yy))
    ax.set_xlabel("public share - dais share", color=ap.P["ink2"])
    ap.panel_title(ax, "probability-scale gap")
    note = ("STAND-IN: the ordinal outcome is VADER compound binned into five quantile levels over turns >= 100 words, with log words "
            "as a covariate. VADER is valence, not stance; a higher level means more positive words, not support. Replace with "
            "turn_stance when the stance pass has run." if standin else
            "Outcome from the stance pass (pipeline/stance.py); stance -2..2 toward the item, hostility 0..4.")
    ap.footnote(fig, note)
    ap.save(fig, path, (0, 0.14, 1, 0.86))


def role_gap_ppc(ppc_df, path, instrument, scope, outcome="stance", levels=(-2, -1, 0, 1, 2)):
    standin = instrument if instrument.startswith("STAND-IN") else None
    _prep(scope, False)
    fig, axes = ap.figure(1, 2, _title(f"Role gap PPC, {outcome}: observed vs replicated level shares", False, standin),
                          "bars = observed share per level; dot + line = replicated median and 90 %", (10, 4.4))
    for ax, role in zip(axes[0], ("dais", "public")):
        ap.style(ax)
        g = ppc_df[ppc_df["role"] == role].sort_values("level")
        x = np.arange(len(g))
        ax.bar(x, g["observed_share"], color=ROLE_COL[role], alpha=0.45, width=0.7)
        ax.vlines(x, g["rep_lo90"], g["rep_hi90"], color=INK, linewidth=1.4)
        ax.scatter(x, g["rep_median"], s=28, color=INK, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([str(l) for l in g["level"]])
        ap.panel_title(ax, f"{role}  (n = {int(g['n'].iloc[0])})")
        ax.set_ylim(0, max(0.5, float(g["observed_share"].max()) * 1.25))
    ap.footnote(fig, "Proportional odds assumes one shift for all thresholds; a role whose observed shares bulge in the middle while the "
                     "replicate does not is a violation worth a category-specific model.")
    ap.save(fig, path, (0, 0.08, 1, 0.86))
