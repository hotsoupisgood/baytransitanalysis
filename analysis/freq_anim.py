#!/usr/bin/env python3
"""
analysis/freq_anim.py -- animated bar chart of the top content words over time.

    x axis: the words (a FIXED set: the top-N content words per speaker over the
            whole corpus, union, ordered by overall rate -- fixed so a bar is the
            same word in every frame and the eye can follow it)
    y axis: rate per million host tokens in that month (METHODS.md 2)
    time:   one frame per month, held, with short tweens between months
    bars:   one per speaker, side by side, series colours; thin whisker = 95%
            cluster-bootstrap interval when the cell has >= 2 videos

    .venv/bin/python analysis/freq_anim.py --db corpus.sqlite --top 8
    .venv/bin/python analysis/freq_anim.py --db corpus.sqlite --words trump gaza israel

Output: <out-dir>/top_words.gif (and top_words_last.png, the final frame).
Same tokenizer, cells and bootstrap as analysis/freq_trends.py.
"""
import argparse, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freq_trends as ft  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def frames_for(rows, words, speakers, months):
    """rate[month][speaker] -> np.array over words (nan when the cell is empty)."""
    idx = {w: i for i, w in enumerate(words)}
    out, lo, hi = {}, {}, {}
    for m in months:
        out[m] = {s: np.full(len(words), np.nan) for s in speakers}
        lo[m] = {s: np.full(len(words), np.nan) for s in speakers}
        hi[m] = {s: np.full(len(words), np.nan) for s in speakers}
    for r in rows:
        i = idx[r["word"]]
        out[r["month"]][r["speaker"]][i] = r["rate_pm"]
        if r["lo"] is not None:
            lo[r["month"]][r["speaker"]][i] = r["lo"]
            hi[r["month"]][r["speaker"]][i] = r["hi"]
    return out, lo, hi


def render_gif(rows, words, speakers, months, path, title, hold=10, tween=6, fps=12, mode="light", dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    P = ft.PALETTE[mode]
    rate, lo, hi = frames_for(rows, words, speakers, months)
    ymax = max(1.0, np.nanmax([np.nanmax(np.where(np.isnan(hi[m][s]), rate[m][s], hi[m][s]))
                               for m in months for s in speakers if not np.all(np.isnan(rate[m][s]))] or [1])) * 1.1
    n, k = len(words), len(speakers)
    width = 0.8 / k
    x = np.arange(n)

    # timeline: for each month, `tween` interpolation frames from the previous month, then `hold` frames
    seq = []
    for j, m in enumerate(months):
        if j > 0:
            for t in range(1, tween + 1):
                seq.append((j - 1, j, t / (tween + 1)))
        seq += [(j, j, 1.0)] * hold

    fig, ax = plt.subplots(figsize=(min(15, max(7, 0.55 * n + 3)), 5.6))
    fig.patch.set_facecolor(P["surface"]); ax.set_facecolor(P["surface"])
    bars = []
    for i, s in enumerate(speakers):
        b = ax.bar(x + (i - (k - 1) / 2) * width, np.zeros(n), width * 0.92, color=P["series"][i], label=s,
                   linewidth=0)
        bars.append(b)
    err = [ax.errorbar(x + (i - (k - 1) / 2) * width, np.zeros(n), yerr=np.zeros((2, n)), fmt="none",
                       ecolor=P["ink2"], elinewidth=0.8, capsize=0, alpha=0.6) for i in range(k)]
    ax.set_xticks(x); ax.set_xticklabels(words, fontsize=9, color=P["ink"], rotation=25, ha="right")
    ax.set_ylim(0, ymax); ax.set_xlim(-0.6, n - 0.4)
    ax.tick_params(axis="y", labelsize=8, colors=P["muted"], length=0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=P["grid"], linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(P["axis"])
    ax.set_ylabel("per million host tokens", fontsize=8, color=P["ink2"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9, labelcolor=P["ink2"],
              ncol=k, handlelength=1.2, columnspacing=1.2)
    ax.set_title(title, loc="left", fontsize=12, color=P["ink"], fontweight="bold", pad=14)
    month_txt = ax.text(0.99, 0.98, "", transform=ax.transAxes, ha="right", va="top", fontsize=22,
                        color=P["ink"], fontweight="bold")
    meta_txt = ax.text(0.99, 0.84, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=P["muted"])
    exposure = {(e["speaker"], e["month"]): e for e in ft_exposure(rows)}

    def draw(frame):
        a, b, t = seq[frame]
        ma, mb = months[a], months[b]
        for i, s in enumerate(speakers):
            va, vb = np.nan_to_num(rate[ma][s]), np.nan_to_num(rate[mb][s])
            v = va + (vb - va) * t
            for rect, h in zip(bars[i], v):
                rect.set_height(h)
            # whiskers only on the held frame
            lo_b, hi_b = lo[mb][s], hi[mb][s]
            show = (t >= 1.0) and not np.all(np.isnan(lo_b))
            segs = []
            for xi, l, h in zip(x + (i - (k - 1) / 2) * width, lo_b, hi_b):
                segs.append([(xi, 0 if np.isnan(l) else l), (xi, 0 if np.isnan(h) else h)] if show and not np.isnan(l) else [(xi, 0), (xi, 0)])
            err[i][2][0].set_segments(segs)
        month_txt.set_text(mb if t >= 1.0 else f"{ma} → {mb}")
        info = [f"{s}: {exposure[(s, mb)]['videos']} videos, {exposure[(s, mb)]['tokens']:,} tokens"
                for s in speakers if (s, mb) in exposure]
        meta_txt.set_text("\n".join(info))
        return [r for b in bars for r in b] + [month_txt, meta_txt]

    anim = FuncAnimation(fig, draw, frames=len(seq), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": P["surface"]})
    draw(len(seq) - 1)
    fig.savefig(str(path).replace(".gif", "_last.png"), dpi=dpi, facecolor=P["surface"])
    plt.close(fig)
    return path


def ft_exposure(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["speaker"], r["month"])
        if key not in seen:
            seen.add(key)
            out.append({"speaker": r["speaker"], "month": r["month"], "videos": r["videos"], "tokens": r["tokens"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "trends"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--words", nargs="+")
    g.add_argument("--top", type=int, default=8, help="top-N content words per speaker (union), fixed x order")
    g.add_argument("--distinctive", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--all-speakers", action="store_true")
    ap.add_argument("--min-host-conf", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hold", type=int, default=10, help="frames each month is held (fps 12 -> ~0.8 s)")
    ap.add_argument("--tween", type=int, default=6, help="interpolation frames between months")
    ap.add_argument("--title", default="Top content words by month")
    a = ap.parse_args(argv)

    cells = ft.load_cells(a.db, host_only=not a.all_speakers, min_host_conf=a.min_host_conf)
    if not cells:
        sys.exit("no segments with a month in the corpus")
    words = ft.pick_words(cells, a)
    rows, _ = ft.compute(cells, words, a.n_boot, a.seed)
    speakers = sorted({r["speaker"] for r in rows})
    months = sorted({r["month"] for r in rows})
    # order the x axis by overall rate so the tallest bars sit on the left
    overall = {w: sum(r["count"] for r in rows if r["word"] == w) / max(1, sum(r["tokens"] for r in rows if r["word"] == w))
               for w in words}
    words = sorted(words, key=lambda w: -overall[w])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = render_gif(rows, words, speakers, months, out / "top_words.gif", a.title, hold=a.hold, tween=a.tween)
    print(f"words: {', '.join(words)}")
    print(f"months: {months[0]} .. {months[-1]} ({len(months)}); speakers: {', '.join(speakers)}")
    print(f"wrote {path} and {str(path).replace('.gif', '_last.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
