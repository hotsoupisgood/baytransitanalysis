#!/usr/bin/env python3
"""
analysis/freq_spread.py -- top content words with their month-to-month
distribution, as three lines per speaker.

    x axis: the words (top-N per speaker, union, ordered by overall rate)
    y axis: rate per million host tokens
    lines:  per speaker, the AVERAGE of the monthly rates (solid, marked) and
            the spread of the monthly rates as two thin lines: by default the
            10th and 90th percentile across months (--spread p10p90);
            --spread sd gives mean +/- 1 SD, --spread minmax the range.

Each month contributes one number per word (its pooled rate, METHODS.md 2),
so every month has the same weight regardless of how many tokens it has --
that is what "average by month" means here, and it is the right unit because
month is the stratum (plan/SAMPLING.md). The lines describe how much a
speaker's use of a word moves from month to month; they are not a confidence
interval on the mean.

    .venv/bin/python analysis/freq_spread.py --db corpus.sqlite --top 8
Output: <out-dir>/top_words_spread.png (+ .csv with mean/lo/hi per speaker x word)
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freq_trends as ft  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def summarise(rows, words, speakers, spread):
    out = {}
    for s in speakers:
        for w in words:
            v = np.array([r["rate_pm"] for r in rows if r["speaker"] == s and r["word"] == w
                          and r["rate_pm"] == r["rate_pm"]], float)
            if v.size == 0:
                out[(s, w)] = (np.nan, np.nan, np.nan, 0)
                continue
            mean = v.mean()
            if spread == "sd":
                lo, hi = max(0.0, mean - v.std(ddof=1) if v.size > 1 else mean), mean + (v.std(ddof=1) if v.size > 1 else 0)
            elif spread == "minmax":
                lo, hi = v.min(), v.max()
            else:
                lo, hi = np.percentile(v, [10, 90])
            out[(s, w)] = (mean, lo, hi, v.size)
    return out


def render(summary, words, speakers, path, title, spread, months_n, mode="light", dpi=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = ft.PALETTE[mode]
    x = np.arange(len(words))
    # width capped so 20-25 words still read on a phone; labels rotate instead
    fig, ax = plt.subplots(figsize=(min(15, max(7, 0.55 * len(words) + 3)), 6.2))
    fig.patch.set_facecolor(P["surface"]); ax.set_facecolor(P["surface"])
    k = len(speakers)
    for i, s in enumerate(speakers):
        off = (i - (k - 1) / 2) * 0.12                       # tiny x offset so speakers never overprint
        mean = np.array([summary[(s, w)][0] for w in words])
        lo = np.array([summary[(s, w)][1] for w in words])
        hi = np.array([summary[(s, w)][2] for w in words])
        c = P["series"][i]
        ax.plot(x + off, mean, color=c, linewidth=2.2, marker="o", markersize=6,
                markeredgecolor=P["surface"], markeredgewidth=1.5, label=f"{s} — mean of monthly rates")
        ax.plot(x + off, hi, color=c, linewidth=1, linestyle=(0, (4, 3)), alpha=0.9)
        ax.plot(x + off, lo, color=c, linewidth=1, linestyle=(0, (4, 3)), alpha=0.9)
        ax.vlines(x + off, lo, hi, color=c, linewidth=0.8, alpha=0.35)
    spread_label = {"p10p90": "10th–90th percentile of monthly rates", "sd": "mean ± 1 SD of monthly rates",
                    "minmax": "min–max of monthly rates"}[spread]
    ax.plot([], [], color=P["ink2"], linewidth=1, linestyle=(0, (4, 3)), label=spread_label)
    ax.set_xticks(x); ax.set_xticklabels(words, fontsize=9, color=P["ink"], rotation=45 if len(words) > 12 else 25,
                                         ha="right")
    ax.set_ylim(bottom=0); ax.set_xlim(-0.6, len(words) - 0.4)
    ax.tick_params(axis="y", labelsize=8, colors=P["muted"], length=0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=P["grid"], linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(P["axis"])
    ax.set_ylabel("per million host tokens", fontsize=8, color=P["ink2"])
    ax.set_title(title, loc="left", fontsize=12, color=P["ink"], fontweight="bold", pad=26)
    ax.text(0, 1.03, f"each month weighted equally; {months_n} months", transform=ax.transAxes,
            fontsize=8, color=P["ink2"])
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=P["ink2"])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=P["surface"])
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "trends"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--words", nargs="+")
    g.add_argument("--top", type=int, default=8)
    g.add_argument("--distinctive", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--spread", choices=["p10p90", "sd", "minmax"], default="p10p90")
    ap.add_argument("--all-speakers", action="store_true")
    ap.add_argument("--min-host-conf", type=float, default=0.0)
    ap.add_argument("--title", default="Top content words: monthly average and spread")
    a = ap.parse_args(argv)

    cells = ft.load_cells(a.db, host_only=not a.all_speakers, min_host_conf=a.min_host_conf)
    if not cells:
        sys.exit("no segments with a month in the corpus")
    words = ft.pick_words(cells, a)
    rows, _ = ft.compute(cells, words, n_boot=0, seed=7)   # no bootstrap needed here
    speakers = sorted({r["speaker"] for r in rows})
    months = sorted({r["month"] for r in rows})
    overall = {w: sum(r["count"] for r in rows if r["word"] == w) / max(1, sum(r["tokens"] for r in rows if r["word"] == w))
               for w in words}
    words = sorted(words, key=lambda w: -overall[w])
    summary = summarise(rows, words, speakers, a.spread)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "top_words_spread.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["speaker", "word", "mean_rate_pm", "lo", "hi", "months", "spread"])
        for s in speakers:
            for wd in words:
                m, lo, hi, n = summary[(s, wd)]
                w.writerow([s, wd, round(m, 2), round(lo, 2), round(hi, 2), n, a.spread])
    path = render(summary, words, speakers, out / "top_words_spread.png", a.title, a.spread, len(months))
    print(f"words: {', '.join(words)}")
    print(f"wrote {path} and {out / 'top_words_spread.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
