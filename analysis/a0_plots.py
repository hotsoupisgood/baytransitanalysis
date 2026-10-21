#!/usr/bin/env python3
"""
a0_plots.py -- Phase A0 exploratory figures on the T1 pilot (plan/ANALYSIS-transit.md
1-2; no models). Static matplotlib PNGs sized for a phone, one CSV twin per figure.

Units. A "speaker" is one diarized cluster inside one episode with its role
(dais / public / unknown). Every rate is per 1,000 WORDS (all word tokens of
the speech in question, word_freq.TOKEN_RE, before stop/filler filtering) so
keyword, frame and top-term rates share one denominator. Top-term lists use
the word_freq tokenizer (stop words and fillers removed) for the NUMERATOR.
Intervals are cluster bootstraps that resample whole EPISODES (METHODS.md 7):
tokens inside an episode are not independent.

Figures (output/analysis/a0/):
  a0_rates.png                 episode-min per meeting-h by body (stacked family),
                               public share of episode time, speakers per episode,
                               located vs mention-only episodes by body
  a0_top_terms_role.png        top-25 content terms, dais vs public, with 95% CIs
  a0_top_terms_public_body.png the same for public comment only, one panel per body
  a0_keyness_public_vs_dais.png G2 vs log ratio, public vs dais, same episodes
  a0_frames_role_body.png      frame rates per 1,000 words, role x body (heatmap)
  a0_frames_by_date*.png       frame rates per meeting date (small multiples)
  a0_speakers.png              one dot per speaker: talk seconds vs frame rate
  a0_density_vs_transit.png    keyness and frame rates, density vs transit episodes

    .venv/bin/python analysis/a0_plots.py --db ~/.local/share/ytpt/corpus.sqlite
"""
import argparse, csv, math, os, sys
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
import facets                 # noqa: E402
from freq_trends import PALETTE  # noqa: E402

SCOPE = ""      # facets.describe() of the loaded corpus; appended to every footnote
POOLED = False  # True when --pool drew the figures over more than one instrument

P = PALETTE["light"]
# Categorical slots from the dataviz reference palette, fixed per entity and
# validated with scripts/validate_palette.js (adjacent and all-pairs, light):
ROLE_COL = {"dais": "#2a78d6", "public": "#eb6834", "unknown": "#898781"}   # unknown = residual, muted on purpose
FAM_COL = {"density": "#1baf7a", "transit": "#4a3aa7", "both": "#eda100"}
BODY_COL = {"bart": "#2a78d6", "oakland": "#eb6834", "sanjose": "#1baf7a", "sanrafael": "#4a3aa7"}
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BODY_NAME = {"bart": "BART", "oakland": "Oakland", "sanjose": "San Jose", "sanrafael": "San Rafael"}
BODIES = ["bart", "oakland", "sanjose", "sanrafael"]   # the T1 pilot; load() rebuilds it from the data
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]   # body slots, validated all-pairs (4 colours only)
# The wide corpus has 10 bodies and the validated categorical palette has 4
# colours. Reusing a colour every 5th body would put two bodies on one legend
# swatch, so body is encoded as (colour, MARKER): 4 x 5 = 20 distinguishable
# combinations, and the legend carries the same pair. Do not drop the marker.
MARKERS = ["o", "s", "^", "D", "v"]
BODY_MARK = {}
FS = {"title": 15, "sub": 11, "panel": 13, "tick": 10.5, "label": 11, "note": 9.5}
DPI = 160
MIN_FRAME_HITS = 20
TOP_N = 30          # ANALYSIS-transit.md 2: "top-30 terms per role, per body, per family"
MIN_CELL_WORDS = 500   # below this a rate per 1,000 words is one or two utterances; plotted as "--"
NAMES = set()   # name-like tokens (a0_terms.name_tokens); counted, never labelled


# ------------------------------------------------------------------ data ------
def load(db, frames_path, pool=False):
    global SCOPE, POOLED
    con = wf._open_readonly(db)
    con.row_factory = __import__("sqlite3").Row
    # Before anything is counted: is this one instrument or two? (facets.py)
    SCOPE = facets.require_single(con, pool=pool, script="analysis/a0_plots.py")
    POOLED = SCOPE.startswith("POOLED")
    meetings = [dict(r) for r in con.execute("SELECT * FROM meetings")]
    episodes = {r["episode_id"]: dict(r) for r in con.execute("SELECT * FROM episodes")}
    speakers = {(r["episode_id"], r["speaker"]): dict(r) for r in con.execute("SELECT * FROM episode_speakers")}
    segs = con.execute("SELECT episode_id, speaker, role, text FROM episode_segments ORDER BY episode_id, seg_idx").fetchall()
    con.close()
    frames = at.load_frames(frames_path)
    fpat = {k: [at.term_pattern(t) for t in ts] for k, ts in frames.items()}
    global NAMES, BODIES
    seen = sorted({e["body"] for e in episodes.values()} | {m["body"] for m in meetings})
    BODIES = [b for b in BODIES if b in seen] + [b for b in seen if b not in BODIES]
    for i, b in enumerate(BODIES):
        BODY_COL.setdefault(b, SLOTS[i % len(SLOTS)]); BODY_NAME.setdefault(b, b.title())
        BODY_MARK[b] = MARKERS[(i // len(SLOTS)) % len(MARKERS)]
    NAMES = at.name_tokens([(r["episode_id"], 0, None, None, r["role"], r["speaker"], r["text"]) for r in segs])
    # per speaker: words, content-token Counter, frame hits
    text = defaultdict(list)
    for r in segs:
        text[(r["episode_id"], r["speaker"])].append(r["text"])
    for key, ts in text.items():
        joined = " ".join(ts)
        sp = speakers[key]
        sp["tokens"] = Counter(wf.tokens(joined))
        sp["frame_hits"] = {k: sum(len(rx.findall(joined)) for rx in pats) for k, pats in fpat.items()}
        sp["family"] = episodes[key[0]]["family"]
        sp["method"] = episodes[key[0]].get("method")
        sp["role_source"] = episodes[key[0]].get("roles_source")
        sp["body"] = episodes[key[0]]["body"]
        sp["meeting_date"] = episodes[key[0]]["meeting_date"]
        sp["month"] = episodes[key[0]]["month"]
    return meetings, episodes, speakers, frames


def pooled(speakers, keep):
    """Per-episode counts for a subset of speakers: {episode: (Counter tokens, words, {frame: hits})}."""
    out = {}
    for (ep, _s), sp in speakers.items():
        if not keep(sp):
            continue
        if ep not in out:
            out[ep] = [Counter(), 0, Counter()]
        out[ep][0].update(sp["tokens"]); out[ep][1] += sp["words"]; out[ep][2].update(sp["frame_hits"])
    return out


def rate_ci(counts, totals, n_boot=2000):
    n, t = sum(counts), sum(totals)
    rate = n / t * 1000 if t else float("nan")
    if len(counts) >= 2 and t:
        lo, hi = wf.cluster_bootstrap_ci(counts, totals, n_boot=n_boot)
        return rate, lo / 1000, hi / 1000
    return rate, float("nan"), float("nan")


# ----------------------------------------------------------------- style ------
def figure(nrow, ncol, title, subtitle, size, **kw):
    fig, axes = plt.subplots(nrow, ncol, figsize=size, squeeze=False, **kw)
    fig.patch.set_facecolor(P["surface"])
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=FS["title"], color=P["ink"], fontweight="semibold")
    fig._a0_sub = fig.text(0.01, 0.995 - 0.055 * 8 / size[1], subtitle, fontsize=FS["sub"], color=P["ink2"], va="top")
    return fig, axes


def style(ax, ygrid=True, xgrid=False):
    ax.set_facecolor(P["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["axis"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=P["muted"], labelsize=FS["tick"], length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(P["ink2"])
    if ygrid:
        ax.grid(axis="y", color=P["grid"], linewidth=0.6); ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis="x", color=P["grid"], linewidth=0.6); ax.set_axisbelow(True)
    ax.title.set_color(P["ink"])


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=FS["panel"], color=P["ink"], fontweight="semibold", pad=8)


def legend(ax_or_fig, handles, labels, **kw):
    lg = ax_or_fig.legend(handles, labels, frameon=False, fontsize=FS["label"], labelcolor=P["ink2"], **kw)
    return lg


def footnote(fig, text, y=0.012):
    # Every figure carries its instrument scope. A PNG that leaves this folder
    # has to say on its face which locate route and which role instrument it was
    # drawn from, or the facets only exist in the directory tree.
    if SCOPE:
        text = f"{text}\nScope: {SCOPE}."
    fig.text(0.01, y, text, fontsize=FS["note"],
             color=(P["ink"] if POOLED else P["muted"]), va="bottom", wrap=True)


def stamped(path):
    """POOLED figures and CSVs get it in the filename too, not just the caption."""
    path = Path(path)
    return path.with_name("POOLED_" + path.name) if POOLED else path


def save(fig, path, rect):
    path = stamped(path)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=DPI, facecolor=P["surface"])
    plt.close(fig)
    print(f"wrote {path}")


def write_csv(path, rows, fields):
    with open(stamped(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def hbar_ci(ax, labels, rates, los, his, color, ns=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rates, height=0.62, color=color, edgecolor=P["surface"], linewidth=1)
    ok = ~np.isnan(los)
    ax.errorbar(np.asarray(rates)[ok], y[ok], xerr=[np.asarray(rates)[ok] - np.asarray(los)[ok], np.asarray(his)[ok] - np.asarray(rates)[ok]],
                fmt="none", ecolor=P["ink2"], elinewidth=1, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=FS["tick"])
    ax.set_ylim(-0.7, len(labels) - 0.3)
    if ns is not None:
        for yi, n in zip(y, ns):
            ax.text(ax.get_xlim()[1] * 0.995, yi, f"n={n}", ha="right", va="center", fontsize=FS["note"] - 1, color=P["muted"])
    style(ax, ygrid=False, xgrid=True)


# --------------------------------------------------------------- figure a -----
def classified_meetings(meetings):
    """The exposure denominator: every meeting the classifier actually read.

    'failed' is NOT excluded. A meeting whose audio download failed was still
    classified from its caption track, so its hours are real exposure and its
    located episode-minutes are real numerator -- dropping it inflates the rate
    for that body (in the T1 pilot it moved San Jose from 3 meetings / 9.7 h to
    2 / 7.4 h). Only never-read statuses come out."""
    return [m for m in meetings if m["status"] not in ("discovered", "captioned")]


def fig_rates(meetings, episodes, speakers, out):
    fig, axes = figure(2, 2, "Episode rates", "", (12, 9.5))
    classified = classified_meetings(meetings)
    hours = defaultdict(float); core = defaultdict(lambda: defaultdict(float)); nmeet = Counter()
    n_about = Counter(); n_mention = Counter()
    for m in classified:
        hours[m["body"]] += (m["duration_s"] or 0) / 3600; nmeet[m["body"]] += 1
        for f in ("density", "transit", "both"):
            core[m["body"]][f] += m[f"{f}_core_s"] / 60
        n_about[m["body"]] += m["n_about"]; n_mention[m["body"]] += m["n_mention"]
    fig._a0_sub.set_text(f"n = {len(classified)} classified meetings, {sum(hours.values()):.1f} meeting-h, "
                          f"{len(episodes)} transcribed episodes, {len(speakers)} speakers, "
                          f"{sum(e['words'] for e in episodes.values()):,} words")
    # a1 stacked
    ax = axes[0][0]; panel_title(ax, "Episode-minutes per meeting-hour, by body")
    x = np.arange(len(BODIES)); bottom = np.zeros(len(BODIES)); rows = []
    for f in ("density", "transit", "both"):
        v = np.array([core[b][f] / hours[b] if hours[b] else 0 for b in BODIES])
        ax.bar(x, v, 0.6, bottom=bottom, color=FAM_COL[f], edgecolor=P["surface"], linewidth=1.5, label=f)
        bottom += v
    for i, b in enumerate(BODIES):
        # A body with no recorded meeting duration has NO rate, not a rate of
        # zero. Granicus meetings for samtrans and sausalito carry duration_s
        # NULL, and "0.0 min/h" next to BART's 28.3 would read as "they never
        # discuss it". Say the denominator is missing instead.
        if not hours[b]:
            ax.text(i, 0.3, "no meeting\nduration", ha="center", va="bottom", rotation=90,
                    fontsize=FS["note"], color=P["muted"], style="italic")
        else:
            ax.text(i, bottom[i] + 0.3, f"{bottom[i]:.1f}", ha="center", va="bottom", fontsize=FS["label"], color=P["ink2"])
        rows.append({"body": b, "meetings": nmeet[b], "meeting_hours": round(hours[b], 2),
                     **{f"{f}_min_per_h": (round(core[b][f] / hours[b], 2) if hours[b] else "") for f in ("density", "transit", "both")}})
    # The pilot fits four horizontal labels with the "N mtg, X h" line under
    # each. The wide corpus has ten, so the names collide and the second line
    # turns to mush: rotate, fold the exposure into the tick, drop the pad.
    wide = len(BODIES) > 4
    ax.set_xticks(x)
    ax.set_xticklabels([f"{BODY_NAME[b]}\n{nmeet[b]} mtg, {hours[b]:.0f} h" for b in BODIES],
                       rotation=38 if wide else 0, ha="right" if wide else "center",
                       fontsize=FS["note"] if wide else FS["tick"])
    ax.set_ylabel("episode-min / meeting-h", color=P["ink2"], fontsize=FS["label"])
    ax.set_ylim(0, (max(bottom) or 1) * 1.25); style(ax)
    h, l = ax.get_legend_handles_labels(); legend(ax, h, l, loc="upper right")
    write_csv(out / "a0_rates_episode_min.csv", rows, list(rows[0].keys()))
    # a2 public share
    ax = axes[0][1]; panel_title(ax, "Public-comment share of episode talk time")
    talk = defaultdict(lambda: Counter()); neps = defaultdict(set)
    for (ep, _s), sp in speakers.items():
        talk[sp["body"]][sp["role"]] += sp["talk_s"]; neps[sp["body"]].add(ep)
    # Only bodies with a transcribed episode: a route that located nothing at a
    # body has no talk-time composition, and a row of "0 %" reads as "no public
    # comment there" rather than "not measured here".
    shown = [b for b in BODIES if neps[b]]
    rows = []
    for i, b in enumerate(shown):
        tot = sum(talk[b].values()) or 1
        left = 0
        for role in ("public", "dais", "unknown"):
            v = talk[b][role] / tot * 100
            ax.barh(len(shown) - 1 - i, v, 0.6, left=left, color=ROLE_COL[role], edgecolor=P["surface"], linewidth=1.5, label=role if i == 0 else None)
            left += v
        ax.text(talk[b]["public"] / tot * 100 + 1, len(shown) - 1 - i, f"{talk[b]['public'] / tot * 100:.0f}%", va="center", fontsize=FS["label"], color=P["ink2"])
        rows.append({"body": b, "episodes": len(neps[b]), "talk_s": round(tot), **{f"{r}_share": round(talk[b][r] / tot, 3) for r in ("public", "dais", "unknown")}})
    ax.set_yticks(range(len(shown))[::-1]); ax.set_yticklabels([f"{BODY_NAME[b]} ({len(neps[b])} ep)" for b in shown])
    ax.set_xlim(0, 100); ax.set_xlabel("% of episode talk seconds", color=P["ink2"], fontsize=FS["label"]); style(ax, ygrid=False, xgrid=True)
    ax.set_ylim(-1.4, len(shown) - 0.4)           # an empty band under the last bar for the legend
    h, l = ax.get_legend_handles_labels(); legend(ax, h, l, loc="lower right", ncol=3)
    write_csv(out / "a0_rates_public_share.csv", rows, list(rows[0].keys()))
    # a3 speakers per episode
    ax = axes[1][0]; panel_title(ax, "Speakers (diarized clusters) per episode")
    ns = [e["n_speakers"] for e in episodes.values()]
    bins = np.arange(0, max(ns) + 2) - 0.5
    ax.hist(ns, bins=bins, color=ROLE_COL["dais"], edgecolor=P["surface"], linewidth=1.5)
    ax.axvline(np.median(ns), color=P["ink2"], linewidth=1)
    med = float(np.median(ns))
    ax.text(med + 0.3, ax.get_ylim()[1] * 0.92, f"median {med:g}", fontsize=FS["label"], color=P["ink2"])
    ax.set_xlabel("clusters in the episode", color=P["ink2"], fontsize=FS["label"]); ax.set_ylabel("episodes", color=P["ink2"], fontsize=FS["label"]); style(ax)
    # a4 located vs mention-only. NOT always "by the classifier": on an
    # agenda_index slice these episodes were read off agenda titles, and a panel
    # captioned "located by the classifier" would name the wrong instrument.
    route = SCOPE.split(" = ", 1)[1].split(" (")[0] if SCOPE.startswith("locate route") else "the locate route"
    ax = axes[1][1]; panel_title(ax, f"Episodes located by {route}, by body")
    w = 0.36
    ax.bar(x - w / 2, [n_about[b] for b in BODIES], w, color=ROLE_COL["dais"], label="about density/transit", edgecolor=P["surface"], linewidth=1)
    ax.bar(x + w / 2, [n_mention[b] for b in BODIES], w, color=P["axis"], label="mention only", edgecolor=P["surface"], linewidth=1)
    for i, b in enumerate(BODIES):
        ax.text(i - w / 2, n_about[b] + 0.3, str(n_about[b]), ha="center", fontsize=FS["label"], color=P["ink2"])
        ax.text(i + w / 2, n_mention[b] + 0.3, str(n_mention[b]), ha="center", fontsize=FS["label"], color=P["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels([BODY_NAME[b] for b in BODIES], rotation=38 if wide else 0,
                       ha="right" if wide else "center", fontsize=FS["note"] if wide else FS["tick"])
    ax.set_ylabel("episodes", color=P["ink2"], fontsize=FS["label"]); style(ax)
    h, l = ax.get_legend_handles_labels(); legend(ax, h, l, loc="upper right")
    footnote(fig, "Panel 1 counts every located about_* episode (transcribed or not), core minutes over the meeting's full length; "
                  f"panels 2-3 use the {len(episodes)} transcribed episodes. "
                  "Roles are cue-phrase labels (81-94% of speech time correct on T0 hand checks).")
    save(fig, out / "a0_rates.png", (0, 0.03, 1, 0.94))


# --------------------------------------------------------------- figure b -----
def top_terms(per_ep, top=TOP_N, min_eps=3):
    tot = Counter(); words = {ep: v[1] for ep, v in per_ep.items()}
    for ep, v in per_ep.items():
        tot.update(v[0])
    rows = []
    for w, n in tot.most_common(400):
        if w in NAMES:
            continue
        eps = sum(1 for ep in per_ep if per_ep[ep][0][w])
        if eps < min_eps:
            continue
        rate, lo, hi = rate_ci([per_ep[ep][0][w] for ep in per_ep], [words[ep] for ep in per_ep])
        rows.append({"term": w, "n": n, "episodes": eps, "rate_per_1000": round(rate, 2), "lo": round(lo, 2), "hi": round(hi, 2)})
        if len(rows) >= top:
            break
    return rows


def fig_top_terms(speakers, out):
    fig, axes = figure(1, 2, f"Top-{TOP_N} content terms, dais vs public comment", "", (12, 10.5))
    stats = []
    for i, role in enumerate(("dais", "public")):
        per_ep = pooled(speakers, lambda sp, r=role: sp["role"] == r)
        rows = top_terms(per_ep)
        nw = sum(v[1] for v in per_ep.values()); nsp = sum(1 for sp in speakers.values() if sp["role"] == role)
        stats.append(f"{role}: {len(per_ep)} episodes, {nsp} speakers, {nw:,} words")
        ax = axes[0][i]; panel_title(ax, f"{role}  ({len(per_ep)} ep, {nsp} spk, {nw:,} words)")
        if not rows:
            ax.text(0.5, 0.5, "no term in >= 3 episodes", ha="center", va="center", transform=ax.transAxes, color=P["muted"]); style(ax); continue
        hbar_ci(ax, [r["term"] for r in rows], [r["rate_per_1000"] for r in rows], np.array([r["lo"] for r in rows]), np.array([r["hi"] for r in rows]), ROLE_COL[role])
        ax.set_xlabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"])
        for r in rows:
            r["role"] = role
        write_csv(out / f"a0_top_terms_{role}.csv", rows, list(rows[0].keys()))
    fig._a0_sub.set_text("n = " + "; ".join(stats) + ". Stop words and fillers removed; terms in >= 3 episodes")
    footnote(fig, "Whiskers: 95% cluster bootstrap resampling whole episodes (2,000 replicates). Rate = occurrences / all words of that role x 1,000.")
    save(fig, out / "a0_top_terms_role.png", (0, 0.03, 1, 0.95))

    nrow = math.ceil(len(BODIES) / 2)
    fig, axes = figure(nrow, 2, f"Top-{TOP_N} content terms in PUBLIC COMMENT, by body", "", (12, 7.5 * nrow))
    stats = []
    for i, b in enumerate(BODIES):
        per_ep = pooled(speakers, lambda sp, b=b: sp["role"] == "public" and sp["body"] == b)
        nw = sum(v[1] for v in per_ep.values()); nsp = sum(1 for sp in speakers.values() if sp["role"] == "public" and sp["body"] == b)
        ax = axes[i // 2][i % 2]
        min_eps = 3 if len(per_ep) >= 6 else 2 if len(per_ep) >= 3 else 1
        rows = top_terms(per_ep, min_eps=min_eps)
        panel_title(ax, f"{BODY_NAME[b]}  ({len(per_ep)} ep, {nsp} spk, {nw:,} words; terms in >= {min_eps} ep)")
        stats.append(f"{BODY_NAME[b]} {nw:,}")
        if not rows:
            ax.text(0.5, 0.5, "no public comment transcribed", ha="center", va="center", transform=ax.transAxes, color=P["muted"]); style(ax); continue
        hbar_ci(ax, [r["term"] for r in rows], [r["rate_per_1000"] for r in rows], np.array([r["lo"] for r in rows]), np.array([r["hi"] for r in rows]), ROLE_COL["public"])
        ax.set_xlabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"])
        for r in rows:
            r["body"] = b
        write_csv(out / f"a0_top_terms_public_{b}.csv", rows, list(rows[0].keys()))
    for j in range(len(BODIES), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    fig._a0_sub.set_text("n = public-comment words: " + ", ".join(stats) + ". Stop words and fillers removed")
    footnote(fig, "Whiskers: 95% cluster bootstrap over episodes (none when a body has one episode). Oakland's three episodes are short: read its rates as anecdotes.")
    save(fig, out / "a0_top_terms_public_body.png", (0, 0.02, 1, 0.96))


# --------------------------------------------------------------- figure c -----
def keyness(a_counts, b_counts, min_total=5):
    ca, cb = sum(a_counts.values()), sum(b_counts.values())
    rows = []
    if not ca or not cb:
        # One side of the contrast is empty in this facet cell (e.g. a route
        # that located no transit episode at all). Keyness of X against nothing
        # is not a small number, it is undefined; return no rows and let the
        # figure say the cell is empty.
        return rows, ca, cb
    for w in set(a_counts) | set(b_counts):
        a, b = a_counts[w], b_counts[w]
        if a + b < min_total:
            continue
        rows.append({"term": w, "a": a, "b": b, "log_ratio": round(wf.log_ratio(a, b, ca, cb), 3),
                     "g2": round(wf.log_likelihood(a, b, ca, cb), 2), "name_like": int(w in NAMES)})
    return rows, ca, cb


def label_points(ax, pts, side, color):
    """pts = [(x, y, label)] on one side of zero. Labels go in a column just
    beyond the outermost point of that side, spread vertically so none
    overprint, each joined to its point by a hairline."""
    if not pts:
        return
    to_axes = lambda x, y: ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
    pts = sorted(pts, key=lambda p: to_axes(*p[:2])[1])
    ys = [to_axes(x, y)[1] for x, y, _ in pts]
    gap = 0.9 / max(len(pts), 1)
    lo = max(0.03, ys[0])
    for i in range(len(ys)):
        ys[i] = max(ys[i], lo + i * gap) if i else lo
    over = ys[-1] - 0.97
    if over > 0:
        ys = [y - over for y in ys]
    xa = max(to_axes(x, y)[0] for x, y, _ in pts) + 0.03 if side > 0 else min(to_axes(x, y)[0] for x, y, _ in pts) - 0.03
    for (x, y, lab), ya in zip(pts, ys):
        ax.annotate(lab, xy=(x, y), xycoords="data", xytext=(xa, ya), textcoords="axes fraction",
                    ha="left" if side > 0 else "right", va="center", fontsize=FS["tick"], color=P["ink"],
                    arrowprops=dict(arrowstyle="-", color=P["axis"], linewidth=0.5, shrinkA=0, shrinkB=3))


def fig_keyness(speakers, out):
    has = defaultdict(set)
    for (ep, _s), sp in speakers.items():
        has[ep].add(sp["role"])
    eps = {ep for ep, roles in has.items() if {"dais", "public"} <= roles}
    pub = pooled(speakers, lambda sp: sp["role"] == "public" and sp["episode_id"] in eps)
    dais = pooled(speakers, lambda sp: sp["role"] == "dais" and sp["episode_id"] in eps)
    pc, dc = Counter(), Counter()
    for v in pub.values():
        pc.update(v[0])
    for v in dais.values():
        dc.update(v[0])
    rows, cp, cd = keyness(pc, dc)
    rows.sort(key=lambda r: -r["log_ratio"])
    write_csv(out / "a0_keyness_public_vs_dais.csv", rows, ["term", "a", "b", "log_ratio", "g2", "name_like"])
    fig, axes = figure(1, 1, "Public comment vs dais: keyness on the same episodes", "", (12, 9.5))
    n_pw = sum(v[1] for v in pub.values()); n_dw = sum(v[1] for v in dais.values())
    fig._a0_sub.set_text(f"n = {len(eps)} episodes with both roles; public {n_pw:,} words, dais {n_dw:,} words; "
                          f"{len(rows)} content terms with >= 5 occurrences")
    ax = axes[0][0]
    x = np.array([r["log_ratio"] for r in rows]); y = np.array([max(r["g2"], 0.05) for r in rows])
    col = [ROLE_COL["public"] if v > 0 else ROLE_COL["dais"] for v in x]
    ax.scatter(x, y, s=42, c=col, alpha=0.75, edgecolors=P["surface"], linewidths=1)
    if not len(rows):
        # A thin facet cell can have NO term reaching the 5-occurrence floor in
        # both roles. Draw the empty axes and say why, rather than dying on
        # x.min() of an empty array -- the caller is a facet sweep and one thin
        # cell must not take the other cells' figures down with it.
        ax.text(0.5, 0.5, "no term occurs >= 5 times in both roles\nin this facet cell",
                ha="center", va="center", fontsize=FS["panel"], color=P["muted"], transform=ax.transAxes)
        ax.set_xlim(-1, 1)
    else:
        ax.set_xlim(x.min() - 1.6, x.max() + 1.6)  # room for the two label columns inside the axes
    ax.set_yscale("log"); ax.axvline(0, color=P["axis"], linewidth=0.8)
    ax.axhline(3.84, color=P["axis"], linewidth=0.8); ax.axhline(10.83, color=P["axis"], linewidth=0.8)
    ax.text(-0.08, 3.84, "p<0.05", ha="right", va="bottom", fontsize=FS["note"], color=P["muted"])
    ax.text(-0.08, 10.83, "p<0.001", ha="right", va="bottom", fontsize=FS["note"], color=P["muted"])
    top_pub = [r for r in rows if r["log_ratio"] > 0 and not r["name_like"]][:20]
    top_dais = [r for r in rows if r["log_ratio"] < 0 and not r["name_like"]][-20:]
    label_points(ax, [(r["log_ratio"], max(r["g2"], 0.05), r["term"]) for r in top_pub], +1, ROLE_COL["public"])
    label_points(ax, [(r["log_ratio"], max(r["g2"], 0.05), r["term"]) for r in top_dais], -1, ROLE_COL["dais"])
    ax.set_xlabel("log2 ratio  (< 0: more in dais speech     > 0: more in public comment)", color=P["ink2"], fontsize=FS["label"])
    ax.set_ylabel("G2 (log scale)", color=P["ink2"], fontsize=FS["label"])
    style(ax, ygrid=True, xgrid=True)
    from matplotlib.lines import Line2D
    legend(ax, [Line2D([], [], marker="o", color=ROLE_COL["public"], linestyle=""), Line2D([], [], marker="o", color=ROLE_COL["dais"], linestyle="")],
           ["more in public comment", "more in dais speech"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    footnote(fig, f"Labels: top 20 by |log ratio| each side (+0.5 smoothing; a term absent on one side caps near +/-5); {len(NAMES)} name-like tokens are never labelled. "
                  "Each term is its own test -- read log ratio first; G2 lines are a floor, not a finding (METHODS.md 3-4, 11).")
    save(fig, out / "a0_keyness_public_vs_dais.png", (0, 0.04, 1, 0.93))


# --------------------------------------------------------------- figure d -----
def frame_totals(speakers, frames):
    tot = Counter()
    for sp in speakers.values():
        tot.update(sp["frame_hits"])
    return tot


def fig_frames(speakers, frames, out):
    tot = frame_totals(speakers, frames)
    keep = [k for k in frames if tot[k] >= MIN_FRAME_HITS]
    if not keep:                                    # tiny corpus: show what there is, say so
        keep = [k for k in frames if tot[k] > 0]
    small = [f"{at.FRAME_LABELS.get(k, k)} ({tot[k]})" for k in frames if tot[k] < MIN_FRAME_HITS and k not in keep]
    cols = [(b, r) for b in BODIES for r in ("dais", "public")]
    mat = np.full((len(keep), len(cols)), np.nan); nwords = []; rows = []
    for j, (b, r) in enumerate(cols):
        sub = [sp for sp in speakers.values() if sp["body"] == b and sp["role"] == r]
        w = sum(sp["words"] for sp in sub); nwords.append(w)
        for i, k in enumerate(keep):
            h = sum(sp["frame_hits"][k] for sp in sub)
            mat[i, j] = h / w * 1000 if w >= 500 else np.nan
            rows.append({"frame": k, "body": b, "role": r, "hits": h, "words": w, "rate_per_1000": round(h / w * 1000, 2) if w else None})
    write_csv(out / "a0_frames_role_body.csv", rows, ["frame", "body", "role", "hits", "words", "rate_per_1000"])
    fig, axes = figure(1, 1, "Frame-term rates per 1,000 words, by body x role", "", (12, 8.5))
    fig._a0_sub.set_text(f"n = {len(speakers)} speakers in 46 episodes, {sum(nwords):,} words; frames with >= {MIN_FRAME_HITS} hits ({len(keep)} of {len(frames)})")
    ax = axes[0][0]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = min(np.nanmax(mat), 8.0)          # colour saturates at 8 so two small cells do not flatten the rest
    ax.imshow(np.nan_to_num(mat, nan=0), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(keep)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=FS["tick"], color=P["muted"])
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=FS["tick"], color="white" if v > vmax * 0.55 else P["ink"])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{BODY_NAME[b]}\n{r}\n{w:,} w" for (b, r), w in zip(cols, nwords)], fontsize=FS["tick"])
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([f"{at.FRAME_LABELS.get(k, k)} ({tot[k]})" for k in keep], fontsize=FS["tick"])
    for j in range(1, len(cols)):
        ax.axvline(j - 0.5, color=P["surface"], linewidth=2)
    for i in range(1, len(keep)):
        ax.axhline(i - 0.5, color=P["surface"], linewidth=2)
    ax.tick_params(length=0, colors=P["ink2"]); ax.set_facecolor(P["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    footnote(fig, "Cell = hits / words x 1,000; '--' = under 500 words in that cell; colour saturates at 8. Unknown-role speech is left out."
             + (" Frames under 20 hits, listed not plotted: " + ", ".join(small) + "." if small else ""))
    save(fig, out / "a0_frames_role_body.png", (0, 0.05, 1, 0.93))

    # by meeting date, small multiples, <= 6 panels per figure
    meet = defaultdict(lambda: [Counter(), 0])   # (body, date) -> hits, words
    for sp in speakers.values():
        m = meet[(sp["body"], sp["meeting_date"])]
        m[0].update(sp["frame_hits"]); m[1] += sp["words"]
    dates = sorted({d for _b, d in meet})
    rows = []
    for k in keep:
        for (b, d), (h, w) in meet.items():
            rows.append({"frame": k, "body": b, "meeting_date": d, "hits": h[k], "words": w, "rate_per_1000": round(h[k] / w * 1000, 2) if w else None})
    write_csv(out / "a0_frames_by_date.csv", rows, ["frame", "body", "meeting_date", "hits", "words", "rate_per_1000"])
    chunks = [keep[i:i + 6] for i in range(0, len(keep), 6)]
    from matplotlib.lines import Line2D
    for ci, chunk in enumerate(chunks):
        suffix = "" if len(chunks) == 1 else f"_{ci + 1}"
        fig, axes = figure(2, 3, f"Frame-term rates by meeting date{' (' + str(ci + 1) + '/' + str(len(chunks)) + ')' if len(chunks) > 1 else ''}", "", (13, 8.5))
        fig._a0_sub.set_text(f"n = {len(meet)} meetings (one dot each), {sum(m[1] for m in meet.values()):,} words; {dates[0]} to {dates[-1]}")
        for i, k in enumerate(chunk):
            ax = axes[i // 3][i % 3]; panel_title(ax, f"{at.FRAME_LABELS.get(k, k)}  ({tot[k]} hits)")
            for b in BODIES:
                pts = sorted((d, h[k] / w * 1000) for (bb, d), (h, w) in meet.items() if bb == b and w)
                if not pts:
                    continue
                xs = [dates.index(d) for d, _ in pts]; ys = [v for _, v in pts]
                ax.scatter(xs, ys, s=70, color=BODY_COL[b], marker=BODY_MARK[b],
                           edgecolors=P["surface"], linewidths=1.5, zorder=3)
            ax.set_xticks(range(len(dates))); ax.set_xticklabels([d[5:] for d in dates], rotation=60, fontsize=FS["note"])
            ax.set_ylim(bottom=0); style(ax)
            if i % 3 == 0:
                ax.set_ylabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"])
        for i in range(len(chunk), 6):
            axes[i // 3][i % 3].axis("off")
        legend(fig, [Line2D([], [], marker=BODY_MARK[b], color=BODY_COL[b], linestyle="") for b in BODIES],
               [BODY_NAME[b] for b in BODIES],
               loc="upper right", ncol=4, bbox_to_anchor=(0.99, 0.945))
        footnote(fig, "x = meeting date (MM-DD, 2026); one dot per meeting, all transcribed episodes of that meeting pooled, all roles. "
                      "Two meetings per body: this shows spread between meetings, not a trend.")
        save(fig, out / f"a0_frames_by_date{suffix}.png", (0, 0.03, 1, 0.9))


# --------------------------------------------------------------- figure e -----
def fig_speakers(speakers, frames, out):
    nrow = math.ceil(len(BODIES) / 2)
    fig, axes = figure(nrow, 2, "One dot per speaker: talk time vs frame-term rate", "", (12, 4.75 * nrow), sharey=True)
    rows = []
    for sp in speakers.values():
        hits = sum(sp["frame_hits"].values())
        rows.append({"episode_id": sp["episode_id"], "speaker": sp["speaker"], "body": sp["body"], "role": sp["role"],
                     "talk_s": sp["talk_s"], "words": sp["words"], "frame_hits": hits,
                     "rate_per_1000": round(hits / sp["words"] * 1000, 2) if sp["words"] else None})
    write_csv(out / "a0_speakers.csv", rows, list(rows[0].keys()))
    fig._a0_sub.set_text(f"n = {len(rows)} speakers (diarized clusters) in 46 episodes; y = hits on all {len(frames)} frames / words x 1,000; "
                          f"x is log-scaled")
    eligible = [r for r in rows if r["words"] >= 50 and r["rate_per_1000"] is not None]
    top5 = sorted(eligible, key=lambda r: -r["rate_per_1000"])[:5]
    from matplotlib.lines import Line2D
    for i, b in enumerate(BODIES):
        ax = axes[i // 2][i % 2]
        sub = [r for r in rows if r["body"] == b and r["talk_s"] > 0 and r["rate_per_1000"] is not None]
        panel_title(ax, f"{BODY_NAME[b]}  ({len(sub)} speakers)")
        for role in ("unknown", "dais", "public"):
            pts = [r for r in sub if r["role"] == role]
            ax.scatter([r["talk_s"] for r in pts], [r["rate_per_1000"] for r in pts], s=52, color=ROLE_COL[role],
                       edgecolors=P["surface"], linewidths=1.5, alpha=0.85, zorder=3 if role != "unknown" else 2)
        for r in top5:
            if r["body"] == b:
                ax.annotate(f"{r['episode_id']}/{r['speaker'][-2:]}", (r["talk_s"], r["rate_per_1000"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=FS["note"], color=P["ink"])
        ax.set_xscale("log"); ax.set_xlim(1, 4000)
        ax.set_xticks([1, 10, 60, 180, 600, 1800]); ax.set_xticklabels(["1 s", "10 s", "1 min", "3 min", "10 min", "30 min"])
        ax.set_ylim(0, max(r["rate_per_1000"] or 0 for r in rows) * 1.08); style(ax)
        if i % 2 == 0:
            ax.set_ylabel("frame terms per 1,000 words", color=P["ink2"], fontsize=FS["label"])
        if i >= (nrow - 1) * 2:
            ax.set_xlabel("talk time (log scale)", color=P["ink2"], fontsize=FS["label"])
    for j in range(len(BODIES), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    legend(fig, [Line2D([], [], marker="o", color=ROLE_COL[r], linestyle="") for r in ("dais", "public", "unknown")], ["dais", "public", "unknown"],
           loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
    footnote(fig, "Labelled: the 5 highest rates among speakers with >= 50 words (episode id / cluster). Speakers under ~30 s have noisy rates: a single hit in 20 words is 50 per 1,000.")
    save(fig, out / "a0_speakers.png", (0, 0.03, 1, 0.92))


# --------------------------------------------------------------- figure f -----
def fig_family(speakers, frames, out):
    den = pooled(speakers, lambda sp: sp["family"] == "density")
    tra = pooled(speakers, lambda sp: sp["family"] == "transit")
    dc, tc = Counter(), Counter()
    for v in den.values():
        dc.update(v[0])
    for v in tra.values():
        tc.update(v[0])
    rows, cd, ct = keyness(dc, tc)
    rows.sort(key=lambda r: -r["log_ratio"])
    write_csv(out / "a0_keyness_density_vs_transit.csv", rows, ["term", "a", "b", "log_ratio", "g2", "name_like"])
    rows = [r for r in rows if not r["name_like"]]
    nd, nt = sum(v[1] for v in den.values()), sum(v[1] for v in tra.values())
    fig, axes = figure(1, 2, "Density vs transit episodes", "", (13, 9.5), gridspec_kw={"width_ratios": [1, 1]})
    fig._a0_sub.set_text(f"n = density {len(den)} episodes / {nd:,} words; transit {len(tra)} episodes / {nt:,} words "
                          f"(the 1 about_both episode is left out)")
    ax = axes[0][0]; panel_title(ax, "Most distinctive terms (top 15 each side)")
    top_d = rows[:15]; top_t = rows[-15:][::-1]
    sel = top_d + top_t[::-1]
    y = np.arange(len(sel))[::-1]
    ax.barh(y, [r["log_ratio"] for r in sel], height=0.62, color=[FAM_COL["density"] if r["log_ratio"] > 0 else FAM_COL["transit"] for r in sel],
            edgecolor=P["surface"], linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['term']} ({r['a']}/{r['b']})" for r in sel], fontsize=FS["tick"])
    ax.axvline(0, color=P["axis"], linewidth=0.8)
    ax.set_xlabel("log2 ratio  (< 0: transit episodes     > 0: density episodes)", color=P["ink2"], fontsize=FS["label"])
    style(ax, ygrid=False, xgrid=True)
    ax = axes[0][1]; panel_title(ax, "Frame-term rates per 1,000 words")
    keys = sorted(frames, key=lambda k: -(sum(v[2][k] for v in den.values()) + sum(v[2][k] for v in tra.values())))
    frows = []
    for k in keys:
        for fam, per in (("density", den), ("transit", tra)):
            rate, lo, hi = rate_ci([v[2][k] for v in per.values()], [v[1] for v in per.values()])
            frows.append({"frame": k, "family": fam, "hits": sum(v[2][k] for v in per.values()), "words": sum(v[1] for v in per.values()),
                          "rate_per_1000": round(rate, 2), "lo": round(lo, 2), "hi": round(hi, 2)})
    write_csv(out / "a0_frames_by_family.csv", frows, list(frows[0].keys()))
    y = np.arange(len(keys))[::-1]; h = 0.36
    for off, fam in ((h / 2, "density"), (-h / 2, "transit")):
        r = [x for x in frows if x["family"] == fam]
        ax.barh(y + off, [x["rate_per_1000"] for x in r], height=h, color=FAM_COL[fam], edgecolor=P["surface"], linewidth=1, label=fam)
        ax.errorbar([x["rate_per_1000"] for x in r], y + off, xerr=[[x["rate_per_1000"] - x["lo"] for x in r], [x["hi"] - x["rate_per_1000"] for x in r]],
                    fmt="none", ecolor=P["ink2"], elinewidth=1, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([at.FRAME_LABELS.get(k, k) for k in keys], fontsize=FS["tick"])
    ax.set_xlabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"]); style(ax, ygrid=False, xgrid=True)
    hd, ld = ax.get_legend_handles_labels(); legend(ax, hd, ld, loc="lower right")
    footnote(fig, "Left: content tokens with >= 5 occurrences, +0.5 smoothing, counts shown as density/transit. Right: whiskers = 95% cluster bootstrap over episodes.")
    save(fig, out / "a0_density_vs_transit.png", (0, 0.03, 1, 0.94))


# --------------------------------------------------------------- figure g -----
def fig_rates_by_month(meetings, episodes, out):
    """ANALYSIS-transit.md 1, the month half: episode-minutes per meeting-hour by
    body AND month, stacked density/transit. In the T1 pilot each body has three
    classified meetings spread over two or three months, so a "month" here is one
    or two meetings -- month and meeting are confounded and the panel is a spread,
    not a trend. It is drawn anyway because the confound is the finding."""
    cell = defaultdict(lambda: {"h": 0.0, "n": 0, "density": 0.0, "transit": 0.0, "both": 0.0})
    for m in classified_meetings(meetings):
        c = cell[(m["body"], (m["month"] or "?")[:7])]
        c["h"] += (m["duration_s"] or 0) / 3600; c["n"] += 1
        for f in ("density", "transit", "both"):
            c[f] += m[f"{f}_core_s"] / 60
    months = sorted({mo for _b, mo in cell})
    rows = [{"body": b, "month": mo, "meetings": c["n"], "meeting_hours": round(c["h"], 2),
             **{f"{f}_min_per_h": round(c[f] / c["h"], 2) if c["h"] else None for f in ("density", "transit", "both")}}
            for (b, mo), c in sorted(cell.items())]
    write_csv(out / "a0_rates_by_month.csv", rows, list(rows[0].keys()))
    nrow = math.ceil(len(BODIES) / 2)
    fig, axes = figure(nrow, 2, "Episode-minutes per meeting-hour, by body and month", "", (12, 4.4 * nrow), sharex=True)
    fig._a0_sub.set_text(f"n = {sum(c['n'] for c in cell.values())} classified meetings over {len(months)} months "
                         f"({months[0]} to {months[-1]}); stacked by family. One or two meetings per bar.")
    x = np.arange(len(months))
    for i, b in enumerate(BODIES):
        ax = axes[i // 2][i % 2]
        bottom = np.zeros(len(months))
        for f in ("density", "transit", "both"):
            v = np.array([(cell[(b, mo)][f] / cell[(b, mo)]["h"]) if (b, mo) in cell and cell[(b, mo)]["h"] else 0.0
                          for mo in months])
            ax.bar(x, v, 0.6, bottom=bottom, color=FAM_COL[f], edgecolor=P["surface"], linewidth=1.2,
                   label=f if i == 0 else None)
            bottom += v
        for j, mo in enumerate(months):
            c = cell.get((b, mo))
            # The meeting count rides on the value label, not under the axis: a
            # second row of text below the ticks collides with the month labels.
            lab = f"{bottom[j]:.1f}  ({c['n']} mtg)" if c and c["h"] else "no meeting"
            ax.text(j, bottom[j] + 0.02 * max(bottom.max(), 1), lab, ha="center", va="bottom",
                    fontsize=FS["note"] - 0.5, color=P["ink2"] if c and c["h"] else P["muted"])
        panel_title(ax, f"{BODY_NAME[b]}")
        ax.set_xticks(x); ax.set_xticklabels(months, fontsize=FS["tick"])
        ax.set_ylim(0, max(bottom.max() * 1.35, 1)); style(ax)
        if i % 2 == 0:
            ax.set_ylabel("episode-min / meeting-h", color=P["ink2"], fontsize=FS["label"])
    for j in range(len(BODIES), nrow * 2):
        axes[j // 2][j % 2].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    legend(fig, h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
    footnote(fig, "Exposure = the full length of every CLASSIFIED meeting in that body-month; numerator = located episode core minutes, "
                  "transcribed or not. A body-month with no classified meeting is blank, not zero. "
                  "Month is confounded with meeting here: read spread between meetings, never a trend.")
    save(fig, out / "a0_rates_by_month.png", (0, 0.04, 1, 0.92))


# --------------------------------------------------------------- figure h -----
def fig_terms_by_cell(speakers, out):
    """ANALYSIS-transit.md 2, first bullet, in full: top-30 terms for every
    role x body x family cell that has enough words to carry a rate. Cells under
    MIN_CELL_WORDS are reported as thin rather than plotted -- in the pilot most
    of the 3-way grid is thin, which is itself the result."""
    cells, thin, rows = [], [], []
    for b in BODIES:
        for r in ("dais", "public"):
            for fam in ("density", "transit", "both"):
                per_ep = pooled(speakers, lambda sp, b=b, r=r, f=fam: sp["body"] == b and sp["role"] == r and sp["family"] == f)
                w = sum(v[1] for v in per_ep.values())
                rec = {"body": b, "role": r, "family": fam, "episodes": len(per_ep), "words": w}
                if w < MIN_CELL_WORDS:
                    thin.append(rec); continue
                cells.append((b, r, fam, per_ep, w))
                min_eps = 3 if len(per_ep) >= 6 else 2 if len(per_ep) >= 3 else 1
                for t in top_terms(per_ep, min_eps=min_eps):
                    rows.append({**rec, **t})
    write_csv(out / "a0_top_terms_role_body_family.csv", rows,
              ["body", "role", "family", "episodes", "words", "term", "n", "rate_per_1000", "lo", "hi"])
    write_csv(out / "a0_cells_thin.csv", sorted(thin, key=lambda r: -r["words"]), list(thin[0].keys()) if thin else
              ["body", "role", "family", "episodes", "words"])
    if not cells:
        print("no role x body x family cell reaches the word floor; nothing plotted"); return
    ncol = min(3, len(cells)); nrow = math.ceil(len(cells) / ncol)
    fig, axes = figure(nrow, ncol, f"Top-{TOP_N} terms by role x body x family", "", (4.6 * ncol, 8.2 * nrow))
    fig._a0_sub.set_text(f"n = {len(cells)} of {len(cells) + len(thin)} cells clear the {MIN_CELL_WORDS:,}-word floor; "
                         f"the other {len(thin)} are listed in a0_cells_thin.csv")
    for i, (b, r, fam, per_ep, w) in enumerate(cells):
        ax = axes[i // ncol][i % ncol]
        min_eps = 3 if len(per_ep) >= 6 else 2 if len(per_ep) >= 3 else 1
        tr = top_terms(per_ep, min_eps=min_eps)
        panel_title(ax, f"{BODY_NAME[b]} / {r} / {fam}\n{len(per_ep)} ep, {w:,} w, terms in >= {min_eps} ep")
        if not tr:
            ax.text(0.5, 0.5, "no term clears the floor", ha="center", va="center", transform=ax.transAxes, color=P["muted"]); style(ax); continue
        hbar_ci(ax, [t["term"] for t in tr], [t["rate_per_1000"] for t in tr],
                np.array([t["lo"] for t in tr]), np.array([t["hi"] for t in tr]), ROLE_COL[r])
        ax.set_xlabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"])
    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    footnote(fig, "Whiskers: 95% cluster bootstrap over episodes; absent where the cell has one episode. "
                  "Rates across panels are comparable (same denominator rule), counts are not.")
    save(fig, out / "a0_top_terms_role_body_family.png", (0, 0.03, 1, 0.94))


# --------------------------------------------------------------- figure i -----
def fig_frames_by_month(speakers, frames, out):
    """ANALYSIS-transit.md 2, frame bullet, the month half: frame rates per 1,000
    words by month, small multiples, one line per body, role split by marker."""
    tot = frame_totals(speakers, frames)
    keep = [k for k in frames if tot[k] >= MIN_FRAME_HITS] or [k for k in frames if tot[k] > 0]
    cell = defaultdict(lambda: [Counter(), 0])          # (body, month, role) -> hits, words
    for sp in speakers.values():
        if sp["role"] == "unknown":
            continue
        c = cell[(sp["body"], sp["month"], sp["role"])]
        c[0].update(sp["frame_hits"]); c[1] += sp["words"]
    months = sorted({mo for _b, mo, _r in cell})
    rows = [{"frame": k, "body": b, "month": mo, "role": r, "hits": h[k], "words": w,
             "rate_per_1000": round(h[k] / w * 1000, 2) if w else None}
            for k in keep for (b, mo, r), (h, w) in sorted(cell.items())]
    write_csv(out / "a0_frames_by_month.csv", rows, ["frame", "body", "month", "role", "hits", "words", "rate_per_1000"])
    chunks = [keep[i:i + 6] for i in range(0, len(keep), 6)]
    from matplotlib.lines import Line2D
    for ci, chunk in enumerate(chunks):
        suffix = "" if len(chunks) == 1 else f"_{ci + 1}"
        fig, axes = figure(2, 3, f"Frame-term rates by month{f' ({ci + 1}/{len(chunks)})' if len(chunks) > 1 else ''}", "", (13, 8.5))
        fig._a0_sub.set_text(f"n = {len(months)} months ({months[0]} to {months[-1]}); dais = filled, public = hollow; "
                             f"cells under {MIN_CELL_WORDS:,} words dropped")
        for i, k in enumerate(chunk):
            ax = axes[i // 3][i % 3]; panel_title(ax, f"{at.FRAME_LABELS.get(k, k)}  ({tot[k]} hits)")
            for b in BODIES:
                for r, fill in (("dais", True), ("public", False)):
                    pts = sorted((mo, h[k] / w * 1000) for (bb, mo, rr), (h, w) in cell.items()
                                 if bb == b and rr == r and w >= MIN_CELL_WORDS)
                    if not pts:
                        continue
                    xs = [months.index(mo) for mo, _ in pts]; ys = [v for _, v in pts]
                    ax.plot(xs, ys, color=BODY_COL[b], linewidth=1.2, alpha=0.75, zorder=2,
                            linestyle="-" if fill else "--")
                    ax.scatter(xs, ys, s=62, color=BODY_COL[b] if fill else P["surface"], marker=BODY_MARK[b],
                               edgecolors=BODY_COL[b], linewidths=1.8, zorder=3)
            ax.set_xticks(range(len(months))); ax.set_xticklabels(months, rotation=45, fontsize=FS["note"])
            ax.set_xlim(-0.4, len(months) - 0.6); ax.set_ylim(bottom=0); style(ax)
            if i % 3 == 0:
                ax.set_ylabel("per 1,000 words", color=P["ink2"], fontsize=FS["label"])
        for i in range(len(chunk), 6):
            axes[i // 3][i % 3].axis("off")
        legend(fig, [Line2D([], [], marker=BODY_MARK[b], color=BODY_COL[b], linestyle="") for b in BODIES],
               [BODY_NAME[b] for b in BODIES], loc="upper right", ncol=4, bbox_to_anchor=(0.99, 0.945))
        footnote(fig, "One point per body-month-role. In the T1 pilot a body has three meetings over two or three months, "
                      "so a line joins two points from two meetings: spread between meetings, not a time trend.")
        save(fig, out / f"a0_frames_by_month{suffix}.png", (0, 0.03, 1, 0.9))


# ------------------------------------------------------------------- main -----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus.sqlite"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "analysis" / "a0"))
    ap.add_argument("--frames", default=str(ROOT / "plan" / "frames.yaml"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: rates months terms cells keyness frames framemonths speakers family")
    ap.add_argument("--pool", action="store_true",
                    help="draw ONE figure set over every instrument in the corpus. Off by default: a corpus "
                         "spanning two locate routes or two role instruments is refused (analysis/facets.py). "
                         "Everything --pool writes is prefixed POOLED_ and captioned POOLED.")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    meetings, episodes, speakers, frames = load(a.db, a.frames, pool=a.pool)
    if not speakers:
        sys.exit("no episode speakers in the corpus -- run analysis/load_episodes.py first")
    plt.rcParams["font.family"] = ["system-ui", "-apple-system", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]
    steps = {"rates": lambda: fig_rates(meetings, episodes, speakers, out),
             "months": lambda: fig_rates_by_month(meetings, episodes, out),
             "terms": lambda: fig_top_terms(speakers, out),
             "cells": lambda: fig_terms_by_cell(speakers, out),
             "keyness": lambda: fig_keyness(speakers, out),
             "frames": lambda: fig_frames(speakers, frames, out),
             "framemonths": lambda: fig_frames_by_month(speakers, frames, out),
             "speakers": lambda: fig_speakers(speakers, frames, out),
             "family": lambda: fig_family(speakers, frames, out)}
    for k, fn in steps.items():
        if a.only is None or k in a.only:
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
