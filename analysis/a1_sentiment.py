#!/usr/bin/env python3
"""
a1_sentiment.py -- Phase A1, lexicon pass (plan/ANALYSIS-transit.md 3, first
bullet). VADER compound per TURN and per SPEAKER (mean and min), plus a
hostility word-list rate. $0: no network, no model download, no Claude call.

THE UNIT. A turn is a maximal run of consecutive segments by one diarized
cluster inside one episode. The loader already counts turns this way
(load_episodes.py), so a turn here is the same object as `episode_speakers.turns`
there. A speaker is one cluster in one episode, with a role. Per-speaker mean
and min are over that speaker's TURN compounds, unweighted, which is what the
plan asks for; the words-weighted mean is carried alongside because a speaker
with one 400-word turn and six "thank you"s has two defensible means and they
disagree.

WHAT VADER MEASURES, AND WHAT IT DOES NOT. The compound score is valence, not
stance: "I am terrified of the traffic this will bring" is strongly negative and
is not hostile; "I strongly support this excellent project" is strongly positive
and is still opposition if the project is a fare increase. Nothing here measures
support for or against an item -- that is the Claude pass. Procedural speech
("thank you", "please", "welcome") is positive by construction, which is why the
dais/public comparison below is reported next to a procedural-term check.

FACETS. Role source and locate route are instruments, not variables
(analysis/facets.py), and A1 is a ROLE contrast, so pooling the two role
instruments would compare dais and public with two different rulers. This script
refuses a corpus spanning a facet unless --pool is given, exactly as a0_plots
does; analysis/facet_sweep.py runs it once per level.

Outputs (output/analysis/a1/ or the facet cell's folder):
  a1_turns.csv              one row per turn: compound, words, seconds, position
  a1_speakers.csv           one row per cluster: mean, min, weighted mean, rates
  a1_violins.png            compound distribution by role, and by body within role
  a1_sentiment_vs_length.png    turn compound against turn length
  a1_sentiment_by_frame.png     mean compound of turns using each frame, by role
  a1_sentiment_over_episode.png compound against position through the episode
  a1_hostility.csv          hostility hits per category, by role and body

    .venv/bin/python analysis/a1_sentiment.py --db ~/.local/share/ytpt/corpus_a0_pilot.sqlite
"""
import argparse, csv, os, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import word_freq as wf        # noqa: E402
import a0_terms as at         # noqa: E402
import a0_plots as ap         # noqa: E402  (one house style for every figure)
import facets                 # noqa: E402

ROLES = ("dais", "public")
MIN_TURNS = 20       # a violin or a frame mean under this many turns is a dot plot, not a distribution
MIN_SPK_TURNS = 3    # a speaker's mean over 1-2 turns is the turn, not a speaker summary
# plan/hostility.txt categories named `ambiguous_*` fired on non-hostile speech
# in the pilot KWIC read ("units already entitled", "eight active lawsuits").
# They are counted and written out, but kept OUT of the headline rate.
AMBIGUOUS = "ambiguous_"


def analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        sys.exit("Phase A1's lexicon pass needs VADER: .venv/bin/pip install vaderSentiment "
                 "(pure Python, no network at run time; it is in requirements.txt)")
    return SentimentIntensityAnalyzer()


def load_turns(db, frames, hostility, pool=False):
    """[(episode, speaker, role, body, family, turn_idx, start, end, words, text)] plus scope."""
    con = wf._open_readonly(db)
    con.row_factory = sqlite3.Row
    scope = facets.require_single(con, pool=pool, script="analysis/a1_sentiment.py")
    eps = {r["episode_id"]: dict(r) for r in con.execute("SELECT * FROM episodes")}
    segs = [dict(r) for r in con.execute(
        "SELECT episode_id, seg_idx, start_s, end_s, speaker, role, text FROM episode_segments "
        "ORDER BY episode_id, seg_idx")]
    con.close()

    va = analyzer()
    fpat = {k: [at.term_pattern(t) for t in ts] for k, ts in frames.items()}
    hpat = {k: [at.term_pattern(t) for t in ts] for k, ts in hostility.items()}

    turns, cur = [], None
    for s in segs:
        spk = s["speaker"] or "UNLABELLED"
        if cur and cur["episode_id"] == s["episode_id"] and cur["speaker"] == spk:
            cur["text"].append(s["text"]); cur["end"] = s["end_s"]
        else:
            if cur:
                turns.append(cur)
            cur = {"episode_id": s["episode_id"], "speaker": spk, "role": s["role"],
                   "start": s["start_s"], "end": s["end_s"], "text": [s["text"]]}
    if cur:
        turns.append(cur)

    # episode span, for "how far through the episode was this said"
    span = defaultdict(lambda: [float("inf"), 0.0])
    for t in turns:
        sp = span[t["episode_id"]]
        sp[0] = min(sp[0], t["start"]); sp[1] = max(sp[1], t["end"])

    idx = Counter()
    rows = []
    for t in turns:
        text = " ".join(t["text"]).strip()
        e = eps.get(t["episode_id"], {})
        words = len(wf.TOKEN_RE.findall(text.lower()))
        if not words:
            continue
        lo, hi = span[t["episode_id"]]
        dur = (hi - lo) or 1.0
        idx[t["episode_id"]] += 1
        fr = sorted(k for k, pats in fpat.items() if any(p.search(text) for p in pats))
        hh = {k: sum(len(p.findall(text)) for p in pats) for k, pats in hpat.items()}
        rows.append({
            "episode_id": t["episode_id"], "speaker": t["speaker"], "role": t["role"],
            "body": e.get("body", ""), "family": e.get("family", ""), "month": e.get("month", ""),
            "method": e.get("method", ""), "role_source": e.get("roles_source", ""),
            "turn": idx[t["episode_id"]], "start_s": round(t["start"], 1), "end_s": round(t["end"], 1),
            "talk_s": round(t["end"] - t["start"], 1), "words": words,
            "position": round((t["start"] - lo) / dur, 4),
            "compound": va.polarity_scores(text)["compound"],
            "frames": ";".join(fr),
            "hostility_hits": sum(v for k, v in hh.items() if not k.startswith(AMBIGUOUS)),
            "ambiguous_hits": sum(v for k, v in hh.items() if k.startswith(AMBIGUOUS)),
            **{f"host_{k}": v for k, v in hh.items()},
        })
    return rows, scope


def per_speaker(turns):
    """mean and min of the TURN compounds, per (episode, cluster)."""
    by = defaultdict(list)
    for t in turns:
        by[(t["episode_id"], t["speaker"])].append(t)
    out = []
    for (ep, spk), ts in sorted(by.items()):
        c = [t["compound"] for t in ts]
        w = sum(t["words"] for t in ts)
        out.append({"episode_id": ep, "speaker": spk, "role": ts[0]["role"], "body": ts[0]["body"],
                    "family": ts[0]["family"], "turns": len(ts), "words": w,
                    "talk_s": round(sum(t["talk_s"] for t in ts), 1),
                    "mean_compound": round(float(np.mean(c)), 4),
                    "min_compound": round(float(min(c)), 4),
                    "max_compound": round(float(max(c)), 4),
                    "weighted_mean_compound": round(sum(t["compound"] * t["words"] for t in ts) / w, 4) if w else 0.0,
                    "hostility_hits": sum(t["hostility_hits"] for t in ts),
                    "hostility_per_1000w": round(sum(t["hostility_hits"] for t in ts) / w * 1000, 3) if w else 0.0})
    return out


def boot_mean_ci(values, clusters, n_boot=2000, seed=7):
    """95 % CI for a mean, resampling whole EPISODES (METHODS.md 7: turns inside
    an episode are not independent)."""
    if len(set(clusters)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for v, c in zip(values, clusters):
        by[c].append(v)
    keys = list(by)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        pool = [v for i in pick for v in by[keys[i]]]
        if pool:
            means.append(float(np.mean(pool)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))) if means else (float("nan"),) * 2


def violin(ax, groups, labels, colors):
    """One violin per group, median dot, n under the axis. Groups with too few
    turns are drawn as the raw points -- a kernel density over 6 numbers is a
    picture of the kernel."""
    pos = np.arange(len(groups))
    fat = [i for i, g in enumerate(groups) if len(g) >= MIN_TURNS]
    if fat:
        parts = ax.violinplot([groups[i] for i in fat], positions=[pos[i] for i in fat],
                              widths=0.75, showextrema=False, showmedians=False)
        for b, i in zip(parts["bodies"], fat):
            b.set_facecolor(colors[i]); b.set_alpha(0.45); b.set_edgecolor(colors[i]); b.set_linewidth(1.2)
    for i, g in enumerate(groups):
        if not g:
            continue
        if i not in fat:
            jit = (np.random.default_rng(i).random(len(g)) - 0.5) * 0.25
            ax.scatter(pos[i] + jit, g, s=22, color=colors[i], alpha=0.8, edgecolors="none", zorder=3)
        ax.scatter([pos[i]], [np.median(g)], s=55, color=colors[i], edgecolors=ap.P["surface"], linewidths=1.5, zorder=4)
    ax.axhline(0, color=ap.P["axis"], linewidth=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_ylim(-1.05, 1.05)


def fig_violins(turns, out):
    fig, axes = ap.figure(1, 2, "VADER compound by role and body", "", (12.5, 6.5),
                          gridspec_kw={"width_ratios": [1, 2.4]})
    fig._a0_sub.set_text(f"n = {len(turns):,} turns; one violin per group, dot = median; "
                         f"groups under {MIN_TURNS} turns drawn as points")
    ax = axes[0][0]; ap.panel_title(ax, "By role")
    groups = [[t["compound"] for t in turns if t["role"] == r] for r in ROLES + ("unknown",)]
    violin(ax, groups, [f"{r}\n({len(g)})" for r, g in zip(ROLES + ("unknown",), groups)],
           [ap.ROLE_COL[r] for r in ROLES + ("unknown",)])
    ax.set_ylabel("VADER compound (turn)", color=ap.P["ink2"], fontsize=ap.FS["label"]); ap.style(ax)

    ax = axes[0][1]; ap.panel_title(ax, "By body, within role")
    bodies = [b for b in ap.BODIES if any(t["body"] == b for t in turns)]
    labels, groups, cols = [], [], []
    for b in bodies:
        for r in ROLES:
            g = [t["compound"] for t in turns if t["body"] == b and t["role"] == r]
            labels.append(f"{ap.BODY_NAME.get(b, b)}\n{r} ({len(g)})"); groups.append(g); cols.append(ap.ROLE_COL[r])
    violin(ax, groups, labels, cols)
    ax.tick_params(axis="x", labelsize=ap.FS["note"]); ap.style(ax)
    rows = [{"body": b, "role": r,
             "turns": len([t for t in turns if t["body"] == b and t["role"] == r]),
             "median": round(float(np.median([t["compound"] for t in turns if t["body"] == b and t["role"] == r])), 4),
             "mean": round(float(np.mean([t["compound"] for t in turns if t["body"] == b and t["role"] == r])), 4)}
            for b in bodies for r in ROLES
            if [t for t in turns if t["body"] == b and t["role"] == r]]
    ap.write_csv(out / "a1_violins.csv", rows, list(rows[0].keys()))
    ap.footnote(fig, "VADER measures VALENCE, not stance: an alarmed resident and an opponent both score negative, "
                     "and procedural politeness ('thank you', 'please') scores positive. Read the role gap with that in mind.")
    ap.save(fig, out / "a1_violins.png", (0, 0.04, 1, 0.93))


def fig_length(turns, out):
    fig, axes = ap.figure(1, 2, "Sentiment against talk length", "", (12.5, 6))
    fig._a0_sub.set_text(f"n = {len(turns):,} turns; line = median compound in each length bin, "
                         f"shaded = 25th-75th percentile")
    for ax, key, xlabel, bins in ((axes[0][0], "words", "words in the turn (log)", [1, 5, 15, 40, 100, 250, 600, 1e9]),
                                  (axes[0][1], "talk_s", "seconds of the turn (log)", [0.1, 2, 6, 15, 40, 100, 300, 1e9])):
        ap.panel_title(ax, xlabel.split(" (")[0].capitalize())
        for r in ROLES:
            ts = [t for t in turns if t["role"] == r and t[key] > 0]
            ax.scatter([t[key] for t in ts], [t["compound"] for t in ts], s=9, alpha=0.22,
                       color=ap.ROLE_COL[r], edgecolors="none")
            xs, med, lo, hi = [], [], [], []
            for a, b in zip(bins, bins[1:]):
                g = [t["compound"] for t in ts if a <= t[key] < b]
                if len(g) >= 8:
                    xs.append(float(np.sqrt(a * min(b, max(t[key] for t in ts)))))
                    med.append(float(np.median(g))); lo.append(float(np.percentile(g, 25))); hi.append(float(np.percentile(g, 75)))
            if xs:
                ax.fill_between(xs, lo, hi, color=ap.ROLE_COL[r], alpha=0.16, linewidth=0)
                ax.plot(xs, med, color=ap.ROLE_COL[r], linewidth=2, marker="o", markersize=5, label=r)
        ax.set_xscale("log"); ax.axhline(0, color=ap.P["axis"], linewidth=0.8)
        ax.set_xlabel(xlabel, color=ap.P["ink2"], fontsize=ap.FS["label"]); ax.set_ylim(-1.05, 1.05); ap.style(ax)
    axes[0][0].set_ylabel("VADER compound (turn)", color=ap.P["ink2"], fontsize=ap.FS["label"])
    h, l = axes[0][0].get_legend_handles_labels(); ap.legend(axes[0][0], h, l, loc="lower right")
    rows = [{"role": r, "bin_words_from": a, "bin_words_to": (b if b < 1e8 else ""),
             "turns": len([t for t in turns if t["role"] == r and a <= t["words"] < b]),
             "median_compound": round(float(np.median([t["compound"] for t in turns if t["role"] == r and a <= t["words"] < b])), 4)}
            for r in ROLES for a, b in zip([1, 5, 15, 40, 100, 250, 600], [5, 15, 40, 100, 250, 600, 1e9])
            if [t for t in turns if t["role"] == r and a <= t["words"] < b]]
    ap.write_csv(out / "a1_sentiment_vs_length.csv", rows, list(rows[0].keys()))
    ap.footnote(fig, "A one-word turn ('aye', 'thank') has almost no lexicon to score and piles up at 0.0; the short bins "
                     "are mostly that artefact, not calm speech.")
    ap.save(fig, out / "a1_sentiment_vs_length.png", (0, 0.04, 1, 0.93))


def fig_frames(turns, frames, out):
    keys = [k for k in frames if sum(1 for t in turns if k in t["frames"].split(";")) >= 8]
    fig, axes = ap.figure(1, 1, "Sentiment of the turns that use each frame", "", (12, 8.5))
    fig._a0_sub.set_text(f"mean VADER compound over the turns containing at least one term of the frame; "
                         f"95 % CI resamples whole episodes; frames with under 8 turns dropped")
    ax = axes[0][0]
    rows, y = [], []
    order = sorted(keys, key=lambda k: -float(np.mean([t["compound"] for t in turns if k in t["frames"].split(";")])))
    h = 0.36
    for i, k in enumerate(order):
        for j, r in enumerate(ROLES):
            ts = [t for t in turns if r == t["role"] and k in t["frames"].split(";")]
            if len(ts) < 5:
                continue
            m = float(np.mean([t["compound"] for t in ts]))
            lo, hi = boot_mean_ci([t["compound"] for t in ts], [t["episode_id"] for t in ts])
            yy = len(order) - 1 - i + (h / 2 if j == 0 else -h / 2)
            ax.barh(yy, m, h * 0.95, color=ap.ROLE_COL[r], label=r if i == 0 else None)
            if not np.isnan(lo):
                ax.plot([lo, hi], [yy, yy], color=ap.P["ink2"], linewidth=1.2, zorder=4)
            ax.text(m + (0.012 if m >= 0 else -0.012), yy, f"{m:+.2f} (n={len(ts)})",
                    va="center", ha="left" if m >= 0 else "right", fontsize=ap.FS["note"], color=ap.P["ink2"])
            rows.append({"frame": k, "role": r, "turns": len(ts), "mean_compound": round(m, 4),
                         "lo": round(lo, 4) if not np.isnan(lo) else "", "hi": round(hi, 4) if not np.isnan(hi) else ""})
        y.append(at.FRAME_LABELS.get(k, k))
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels(y, fontsize=ap.FS["label"])
    ax.axvline(0, color=ap.P["axis"], linewidth=0.9)
    ax.set_xlabel("mean VADER compound of turns using the frame", color=ap.P["ink2"], fontsize=ap.FS["label"])
    ap.style(ax, ygrid=False, xgrid=True)
    hh, ll = ax.get_legend_handles_labels(); ap.legend(ax, hh, ll, loc="lower right")
    if rows:
        ap.write_csv(out / "a1_sentiment_by_frame.csv", rows, list(rows[0].keys()))
    ap.footnote(fig, "A turn can use several frames and is counted under each. This is the sentiment of turns that "
                     "MENTION the frame, not sentiment about the frame -- VADER does not know what the negativity is aimed at.")
    ap.save(fig, out / "a1_sentiment_by_frame.png", (0, 0.04, 1, 0.93))


def fig_over_episode(turns, out):
    fig, axes = ap.figure(1, 2, "Sentiment through the course of an episode", "", (12.5, 6),
                          gridspec_kw={"width_ratios": [1.5, 1]})
    fig._a0_sub.set_text(f"position 0 = first turn of the episode, 1 = last; "
                         f"n = {len(turns):,} turns in {len({t['episode_id'] for t in turns})} episodes")
    ax = axes[0][0]; ap.panel_title(ax, "Compound against position")
    edges = np.linspace(0, 1, 11)
    rows = []
    for r in ROLES:
        ts = [t for t in turns if t["role"] == r]
        ax.scatter([t["position"] for t in ts], [t["compound"] for t in ts], s=9, alpha=0.2,
                   color=ap.ROLE_COL[r], edgecolors="none")
        xs, med, lo, hi = [], [], [], []
        for a, b in zip(edges, edges[1:]):
            g = [t["compound"] for t in ts if a <= t["position"] < b or (b == 1.0 and t["position"] == 1.0)]
            if len(g) >= 8:
                xs.append((a + b) / 2); med.append(float(np.median(g)))
                lo.append(float(np.percentile(g, 25))); hi.append(float(np.percentile(g, 75)))
                rows.append({"role": r, "position_bin": f"{a:.1f}-{b:.1f}", "turns": len(g),
                             "median_compound": round(float(np.median(g)), 4)})
        if xs:
            ax.fill_between(xs, lo, hi, color=ap.ROLE_COL[r], alpha=0.16, linewidth=0)
            ax.plot(xs, med, color=ap.ROLE_COL[r], linewidth=2, marker="o", markersize=5, label=r)
    ax.axhline(0, color=ap.P["axis"], linewidth=0.8)
    ax.set_xlabel("position through the episode", color=ap.P["ink2"], fontsize=ap.FS["label"])
    ax.set_ylabel("VADER compound (turn)", color=ap.P["ink2"], fontsize=ap.FS["label"])
    ax.set_ylim(-1.05, 1.05); ap.style(ax)
    h, l = ax.get_legend_handles_labels(); ap.legend(ax, h, l, loc="lower left")

    ax = axes[0][1]; ap.panel_title(ax, "Where each role speaks")
    for r in ROLES:
        ps = [t["position"] for t in turns if t["role"] == r]
        if ps:
            ax.hist(ps, bins=np.linspace(0, 1, 21), alpha=0.55, color=ap.ROLE_COL[r], label=f"{r} ({len(ps)})")
    ax.set_xlabel("position through the episode", color=ap.P["ink2"], fontsize=ap.FS["label"])
    ax.set_ylabel("turns", color=ap.P["ink2"], fontsize=ap.FS["label"]); ap.style(ax)
    h, l = ax.get_legend_handles_labels(); ap.legend(ax, h, l, loc="upper right")
    if rows:
        ap.write_csv(out / "a1_sentiment_over_episode.csv", rows, list(rows[0].keys()))
    ap.footnote(fig, "Position is within the EPISODE, not the meeting: an episode is already one agenda item, so this shows "
                     "staff presentation -> public comment -> dais deliberation inside the item, not time of night.")
    ap.save(fig, out / "a1_sentiment_over_episode.png", (0, 0.04, 1, 0.93))


def write_hostility(turns, hostility, out):
    rows = []
    for r in ROLES + ("unknown",):
        ts = [t for t in turns if t["role"] == r]
        w = sum(t["words"] for t in ts)
        if not w:
            continue
        rows.append({"scope": "role", "key": r, "turns": len(ts), "words": w,
                     "hits": sum(t["hostility_hits"] for t in ts),
                     "per_1000w": round(sum(t["hostility_hits"] for t in ts) / w * 1000, 4),
                     "turns_with_any": sum(1 for t in ts if t["hostility_hits"]),
                     "ambiguous_hits": sum(t["ambiguous_hits"] for t in ts),
                     **{k: sum(t[f"host_{k}"] for t in ts) for k in hostility}})
    for b in sorted({t["body"] for t in turns}):
        for r in ROLES:
            ts = [t for t in turns if t["body"] == b and t["role"] == r]
            w = sum(t["words"] for t in ts)
            if not w:
                continue
            rows.append({"scope": "body_role", "key": f"{b}/{r}", "turns": len(ts), "words": w,
                         "hits": sum(t["hostility_hits"] for t in ts),
                         "per_1000w": round(sum(t["hostility_hits"] for t in ts) / w * 1000, 4),
                         "turns_with_any": sum(1 for t in ts if t["hostility_hits"]),
                         "ambiguous_hits": sum(t["ambiguous_hits"] for t in ts),
                         **{k: sum(t[f"host_{k}"] for t in ts) for k in hostility}})
    ap.write_csv(out / "a1_hostility.csv", rows, list(rows[0].keys()))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus_a0_pilot.sqlite"))
    p.add_argument("--out-dir", default=str(ROOT / "output" / "analysis" / "a1"))
    p.add_argument("--frames", default=str(ROOT / "plan" / "frames.yaml"))
    p.add_argument("--hostility", default=str(ROOT / "plan" / "hostility.txt"))
    p.add_argument("--pool", action="store_true",
                   help="run over every instrument at once; refused by default (analysis/facets.py). "
                        "A1 is a ROLE contrast, so pooling the two role instruments is exactly the wrong move.")
    a = p.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    frames = at.load_frames(a.frames)
    hostility = at.load_frames(a.hostility)
    turns, scope = load_turns(a.db, frames, hostility, pool=a.pool)
    if not turns:
        sys.exit("no turns in this corpus -- run analysis/load_episodes.py first")
    ap.SCOPE = scope
    ap.POOLED = scope.startswith("POOLED")
    # a0_plots.load() normally fills these; A1 does not call it, so set them here.
    ap.BODIES = sorted({t["body"] for t in turns})
    for i, b in enumerate(ap.BODIES):
        ap.BODY_COL.setdefault(b, ap.SLOTS[i % len(ap.SLOTS)])
        ap.BODY_NAME.setdefault(b, b.title())
        ap.BODY_MARK[b] = ap.MARKERS[(i // len(ap.SLOTS)) % len(ap.MARKERS)]

    spk = per_speaker(turns)
    ap.write_csv(out / "a1_turns.csv", turns, list(turns[0].keys()))
    ap.write_csv(out / "a1_speakers.csv", spk, list(spk[0].keys()))
    plt.rcParams["font.family"] = ["system-ui", "-apple-system", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]
    fig_violins(turns, out)
    fig_length(turns, out)
    fig_frames(turns, frames, out)
    fig_over_episode(turns, out)
    host = write_hostility(turns, hostility, out)

    print(f"A1 lexicon pass -- scope: {scope}")
    print(f"  {len(turns):,} turns, {len(spk)} clusters, {sum(t['words'] for t in turns):,} words "
          f"in {len({t['episode_id'] for t in turns})} episodes")
    for r in ROLES + ("unknown",):
        ts = [t for t in turns if t["role"] == r]
        if not ts:
            continue
        c = [t["compound"] for t in ts]
        allspk = [s for s in spk if s["role"] == r]
        ss = [s for s in allspk if s["turns"] >= MIN_SPK_TURNS]
        lo, hi = boot_mean_ci(c, [t["episode_id"] for t in ts])
        # The per-speaker MIN is only a speaker statistic when the speaker has
        # several turns. A public commenter typically has ONE turn, so its min
        # is that turn again; say how many clusters clear the floor.
        mn = f"{np.mean([s['min_compound'] for s in ss]):+.3f}" if ss else "n/a"
        print(f"  {r:<8} {len(ts):>5} turns  mean {np.mean(c):+.3f} [{lo:+.3f}, {hi:+.3f}]  median {np.median(c):+.3f}  "
              f"neg-share {np.mean([x < -0.05 for x in c]):.1%}  "
              f"speaker-min mean {mn} ({len(ss)} of {len(allspk)} clusters have >= {MIN_SPK_TURNS} turns)")
    for r in host:
        if r["scope"] == "role":
            print(f"  hostility {r['key']:<8} {r['hits']:>3} hits / {r['words']:>7,} words = "
                  f"{r['per_1000w']:.3f} per 1,000w, in {r['turns_with_any']} of {r['turns']} turns"
                  f"  (+{r['ambiguous_hits']} ambiguous, not counted)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
