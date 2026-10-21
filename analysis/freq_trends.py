#!/usr/bin/env python3
"""
analysis/freq_trends.py -- word frequency over time, per speaker, for content
(non-filler, non-stop) words, with cluster-bootstrap intervals.

    .venv/bin/python analysis/freq_trends.py --db corpus.sqlite --top 12
    .venv/bin/python analysis/freq_trends.py --db corpus.sqlite --words trump gaza israel
    .venv/bin/python analysis/freq_trends.py --db corpus.sqlite --distinctive 8

What it computes (analysis/METHODS.md 1, 2, 7):
  * tokens = word_freq.tokens(): lower-cased, STOP and FILLERS removed, host
    speech only (is_host) by default -- the same tokenizer every other number
    in this project uses, so a trend here is comparable to `make overall`.
  * cell = speaker x month (the sampling stratum, plan/SAMPLING.md). Rate is
    pooled within the cell: sum(count) / sum(tokens) x 1e6 -- per million
    host tokens, as in METHODS.md 2.
  * the 95% interval resamples WHOLE VIDEOS within the cell (cluster
    bootstrap, METHODS.md 7). With 4-12 videos per cell the interval is wide
    and honest; a cell with one video has no interval.
  * word selection: --words (explicit); --top N (the N highest-rate words per
    speaker over the whole corpus, union); --distinctive N (the N words with
    the largest |log ratio| between the two speakers among words with >= 20
    occurrences, METHODS.md 4).

Outputs (in --out-dir, default output/trends):
  trends.csv    speaker, month, word, count, tokens, videos, rate_pm, lo, hi
  trends.html   self-contained small-multiples line chart, one panel per word,
                one line per speaker with the interval as a band, hover
                crosshair + tooltip, legend, table view, light/dark.
  months.csv    tokens and videos per speaker x month (the exposure)

Nothing here spends money or touches state.sqlite.
"""
import argparse, csv, html, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import word_freq as wf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- data ------
def load_cells(db, host_only=True, min_host_conf=0.0, keep_fillers=False):
    """{(speaker, month): {video_id: Counter}} plus per-video token totals."""
    con = wf._open_readonly(db)
    q = ("SELECT v.speaker, v.month, s.video_id, s.text, s.is_host, v.host_confidence "
         "FROM segments s JOIN videos v USING(video_id) ORDER BY s.video_id, s.seg_idx")
    texts = defaultdict(list)                     # (speaker, month, vid) -> [text]
    for sp, month, vid, text, is_host, conf in con.execute(q):
        if not month:
            continue
        if min_host_conf and (conf or 0) < min_host_conf:
            continue
        if host_only and is_host is not None and not is_host:
            continue
        texts[(sp, month[:7], vid)].append(text)
    con.close()
    cells = defaultdict(dict)
    for (sp, month, vid), ts in texts.items():
        cells[(sp, month)][vid] = Counter(wf.tokens(" ".join(ts), keep_fillers=keep_fillers))
    return cells


def pick_words(cells, args):
    if args.words:
        return [w.lower() for w in args.words]
    by_speaker = defaultdict(Counter)
    for (sp, _m), vids in cells.items():
        for c in vids.values():
            by_speaker[sp].update(c)
    speakers = sorted(by_speaker)
    if args.distinctive and len(speakers) >= 2:
        a, b = by_speaker[speakers[0]], by_speaker[speakers[1]]
        ta, tb = sum(a.values()), sum(b.values())
        scored = []
        for w in set(a) | set(b):
            if a[w] + b[w] < args.min_count:
                continue
            scored.append((abs(wf.log_ratio(a[w], ta, b[w], tb)), w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:args.distinctive]]
    out, seen = [], set()
    for sp in speakers:
        for w, _n in by_speaker[sp].most_common(args.top):
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def compute(cells, words, n_boot, seed):
    rows, exposure = [], []
    for (sp, month), vids in sorted(cells.items()):
        totals = {v: sum(c.values()) for v, c in vids.items()}
        exposure.append({"speaker": sp, "month": month, "videos": len(vids), "tokens": sum(totals.values())})
        for w in words:
            counts = [vids[v][w] for v in vids]
            tots = [totals[v] for v in vids]
            n, t = sum(counts), sum(tots)
            rate = n / t * 1e6 if t else float("nan")
            if len(vids) >= 2 and t and n_boot > 0:
                lo, hi = wf.cluster_bootstrap_ci(counts, tots, n_boot=n_boot, seed=seed)
            else:
                lo = hi = float("nan")
            rows.append({"speaker": sp, "month": month, "word": w, "count": n, "tokens": t,
                         "videos": len(vids), "rate_pm": round(rate, 2),
                         "lo": None if math.isnan(lo) else round(lo, 2),
                         "hi": None if math.isnan(hi) else round(hi, 2)})
    return rows, exposure


# ----------------------------------------------------------------- chart -----
PALETTE = {  # dataviz reference palette, categorical slots 1-4 (validated all-pairs)
    # Two speakers in the side corpus; the meeting corpus draws one series per
    # BODY (phase A0, analysis/a0_bridge.py), so a fourth slot exists. Same four
    # hues in the same order as a0_plots.SLOTS, so a body keeps its colour across
    # every A0 figure.
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
              "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#6a5bd0"]},
}


def render_html(rows, exposure, words, title, note):
    speakers = sorted({r["speaker"] for r in rows})
    months = sorted({r["month"] for r in rows})
    data = {"speakers": speakers, "months": months, "words": words, "rows": rows, "exposure": exposure}
    css_vars = lambda m: ";".join(f"--{k}:{v}" for k, v in PALETTE[m].items() if k != "series") + \
        ";" + ";".join(f"--series-{i + 1}:{c}" for i, c in enumerate(PALETTE[m]["series"]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
.viz-root{{color-scheme:light;{css_vars('light')};font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--surface);color:var(--ink);margin:0;padding:20px 24px;}}
@media (prefers-color-scheme: dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;{css_vars('dark')}}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;{css_vars('dark')}}}
h1{{font-size:18px;font-weight:600;margin:0 0 4px}} .sub{{color:var(--ink2);font-size:13px;margin:0 0 14px}}
.bar{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:0 0 14px;font-size:13px}}
.legend{{display:flex;gap:14px}} .legend span{{display:inline-flex;align-items:center;gap:6px}}
.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
button{{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--axis);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}}
.panel{{border:1px solid var(--grid);border-radius:8px;padding:10px 12px 6px;background:var(--surface)}}
.panel h2{{font-size:14px;font-weight:600;margin:0 0 2px}} .panel .m{{font-size:11px;color:var(--muted);margin:0 0 6px}}
svg{{width:100%;height:auto;display:block;overflow:visible}}
.tip{{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--axis);
  border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9}}
table{{border-collapse:collapse;font-size:12px;margin-top:14px;display:none}} th,td{{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right;font-variant-numeric:tabular-nums}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
.showtable table{{display:table}}
</style></head>
<body class="viz-root">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(note)}</p>
<div class="bar">
  <div class="legend" id="legend"></div>
  <button id="tbl">Table view</button>
  <button id="theme">Dark / light</button>
</div>
<div class="grid" id="grid"></div>
<div class="tip" id="tip"></div>
<table id="table"><thead><tr><th>word</th><th>speaker</th><th>month</th><th>count</th><th>tokens</th><th>videos</th><th>per million</th><th>95% lo</th><th>95% hi</th></tr></thead><tbody></tbody></table>
<script>
const D = {json.dumps(data)};
const css = getComputedStyle(document.body);
const col = i => css.getPropertyValue('--series-' + (i + 1)).trim();
const W = 360, H = 190, L = 44, R = 60, T = 12, B = 30;
const fmt = v => v == null ? '–' : (v >= 100 ? Math.round(v) : v.toFixed(1));
function build() {{
  const legend = document.getElementById('legend'); legend.innerHTML = '';
  D.speakers.forEach((s, i) => {{ legend.innerHTML += `<span><i class="sw" style="background:${{col(i)}}"></i>${{s}}</span>`; }});
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  const x = m => L + (W - L - R) * (D.months.length === 1 ? 0.5 : D.months.indexOf(m) / (D.months.length - 1));
  D.words.forEach(word => {{
    const rows = D.rows.filter(r => r.word === word);
    const ymax = Math.max(1, ...rows.map(r => Math.max(r.rate_pm || 0, r.hi || 0))) * 1.08;
    const y = v => T + (H - T - B) * (1 - v / ymax);
    let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{word}} per million host tokens by month">`;
    [0, .5, 1].forEach(f => {{ const v = ymax * f; svg += `<line x1="${{L}}" x2="${{W - R}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="var(--grid)"/>
      <text x="${{L - 6}}" y="${{y(v) + 4}}" font-size="10" text-anchor="end" fill="var(--muted)">${{fmt(v)}}</text>`; }});
    D.months.forEach((m, i) => {{ if (i % Math.ceil(D.months.length / 6) === 0 || i === D.months.length - 1)
      svg += `<text x="${{x(m)}}" y="${{H - 8}}" font-size="10" text-anchor="middle" fill="var(--muted)">${{m}}</text>`; }});
    D.speakers.forEach((s, i) => {{
      const pts = D.months.map(m => rows.find(r => r.speaker === s && r.month === m)).filter(Boolean);
      const band = pts.filter(p => p.lo != null);
      if (band.length > 1) svg += `<path d="M${{band.map(p => x(p.month) + ' ' + y(p.hi)).join(' L')}} L${{band.slice().reverse().map(p => x(p.month) + ' ' + y(p.lo)).join(' L')}} Z" fill="${{col(i)}}" fill-opacity=".12"/>`;
      if (pts.length > 1) svg += `<path d="M${{pts.map(p => x(p.month) + ' ' + y(p.rate_pm)).join(' L')}}" fill="none" stroke="${{col(i)}}" stroke-width="2" stroke-linejoin="round"/>`;
      pts.forEach(p => {{ svg += `<circle cx="${{x(p.month)}}" cy="${{y(p.rate_pm)}}" r="4" fill="${{col(i)}}" stroke="var(--surface)" stroke-width="2" data-s="${{s}}" data-m="${{p.month}}" data-w="${{word}}"/>`; }});
      if (pts.length) {{ const last = pts[pts.length - 1];
        svg += `<text x="${{x(last.month) + 8}}" y="${{y(last.rate_pm) + 4}}" font-size="11" fill="var(--ink2)">${{s}}</text>`; }}
    }});
    svg += `<line id="xh" x1="0" x2="0" y1="${{T}}" y2="${{H - B}}" stroke="var(--axis)" stroke-dasharray="3 3" style="display:none"/></svg>`;
    const tot = rows.reduce((a, r) => a + r.count, 0);
    grid.innerHTML += `<div class="panel"><h2>${{word}}</h2><p class="m">${{tot}} occurrences · per million host tokens · band = 95% cluster-bootstrap</p>${{svg}}</div>`;
  }});
  const tb = document.querySelector('#table tbody'); tb.innerHTML = D.rows.map(r =>
    `<tr><td>${{r.word}}</td><td>${{r.speaker}}</td><td>${{r.month}}</td><td>${{r.count}}</td><td>${{r.tokens}}</td><td>${{r.videos}}</td><td>${{fmt(r.rate_pm)}}</td><td>${{fmt(r.lo)}}</td><td>${{fmt(r.hi)}}</td></tr>`).join('');
}}
build();
const tip = document.getElementById('tip');
document.addEventListener('mousemove', e => {{
  const c = e.target.closest && e.target.closest('circle[data-w]');
  document.querySelectorAll('#xh').forEach(l => l.style.display = 'none');
  if (!c) {{ tip.style.display = 'none'; return; }}
  const r = D.rows.find(q => q.word === c.dataset.w && q.speaker === c.dataset.s && q.month === c.dataset.m);
  const xh = c.closest('svg').querySelector('#xh'); xh.setAttribute('x1', c.getAttribute('cx')); xh.setAttribute('x2', c.getAttribute('cx')); xh.style.display = '';
  tip.innerHTML = `<b>${{r.word}}</b> · ${{r.speaker}} · ${{r.month}}<br>${{fmt(r.rate_pm)}} per million (${{r.count}} in ${{r.tokens.toLocaleString()}} tokens, ${{r.videos}} videos)` +
    (r.lo != null ? `<br>95%: ${{fmt(r.lo)}} – ${{fmt(r.hi)}}` : '<br>no interval (1 video)');
  tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
}});
document.getElementById('tbl').onclick = () => document.body.classList.toggle('showtable');
document.getElementById('theme').onclick = () => {{ const r = document.documentElement;
  r.dataset.theme = r.dataset.theme === 'dark' ? 'light' : 'dark'; build(); }};
</script></body></html>
"""


def render_png(rows, words, title, note, path, mode="light", dpi=160):
    """Static small multiples with the same palette and marks as the HTML
    (for WhatsApp, papers, slides). Needs matplotlib; skipped if absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; no PNG (pip install matplotlib)")
        return None
    P = PALETTE[mode]
    speakers = sorted({r["speaker"] for r in rows})
    months = sorted({r["month"] for r in rows})
    ncol = 3 if len(words) > 4 else min(len(words), 2)
    nrow = math.ceil(len(words) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow + 0.9), squeeze=False)
    fig.patch.set_facecolor(P["surface"])
    xs = {m: i for i, m in enumerate(months)}
    step = max(1, math.ceil(len(months) / 5))
    shown = [i for i in range(0, len(months), step)]
    if len(months) - 1 - shown[-1] >= max(2, step // 2):
        shown.append(len(months) - 1)                  # last month, unless it collides with the previous tick
    for k, word in enumerate(words):
        ax = axes[k // ncol][k % ncol]
        ax.set_facecolor(P["surface"])
        ends = []
        for i, sp in enumerate(speakers):
            pts = [r for r in rows if r["word"] == word and r["speaker"] == sp]
            x = [xs[r["month"]] for r in pts]
            y = [r["rate_pm"] for r in pts]
            band = [r for r in pts if r["lo"] is not None]
            if len(band) > 1:
                ax.fill_between([xs[r["month"]] for r in band], [r["lo"] for r in band], [r["hi"] for r in band],
                                color=P["series"][i], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=P["series"][i], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=P["surface"], markeredgewidth=1.5, label=sp)
            if pts:
                ends.append([x[-1], y[-1], sp])
        # direct labels at the line ends, pushed apart so they never overprint
        ymax = max([r["hi"] or r["rate_pm"] or 0 for r in rows if r["word"] == word] + [1])
        ends.sort(key=lambda e: e[1])
        gap = 0.07 * ymax
        for j in range(1, len(ends)):
            if ends[j][1] - ends[j - 1][1] < gap:
                ends[j][1] = ends[j - 1][1] + gap
        for ex, ey, sp in ends:
            ax.annotate(sp, (ex, ey), xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=P["ink2"], va="center", annotation_clip=False)
        ax.set_title(word, fontsize=10, color=P["ink"], loc="left", fontweight="semibold")
        ax.set_xticks(shown)
        ax.set_xticklabels([months[i] for i in shown], fontsize=7, color=P["muted"])
        ax.tick_params(axis="y", labelsize=7, colors=P["muted"], length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=P["grid"], linewidth=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(P["axis"])
        ax.set_xlim(-0.3, len(months) - 0.7 if len(months) > 1 else 0.7)
    for k in range(len(words), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(speakers) >= 2:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, labelcolor=P["ink2"],
                   bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=P["ink"], fontweight="semibold")
    fig.text(0.01, 0.955, "per million host tokens, per speaker x month; band = 95% cluster bootstrap",
             fontsize=8, color=P["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, facecolor=P["surface"])
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main -----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "trends"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--words", nargs="+")
    g.add_argument("--top", type=int, default=10, help="top-N content words per speaker (union)")
    g.add_argument("--distinctive", type=int, default=0, help="N words with the largest |log ratio| between speakers")
    ap.add_argument("--min-count", type=int, default=20, help="for --distinctive: minimum pooled occurrences")
    ap.add_argument("--all-speakers", action="store_true", help="include guest speech (default: host only)")
    ap.add_argument("--min-host-conf", type=float, default=0.0)
    ap.add_argument("--keep-fillers", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="Content-word frequency over time")
    a = ap.parse_args(argv)

    cells = load_cells(a.db, host_only=not a.all_speakers, min_host_conf=a.min_host_conf, keep_fillers=a.keep_fillers)
    if not cells:
        sys.exit("no segments with a month in the corpus")
    words = pick_words(cells, a)
    rows, exposure = compute(cells, words, a.n_boot, a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trends.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out / "months.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "month", "videos", "tokens"])
        w.writeheader(); w.writerows(exposure)
    n_videos = sum(e["videos"] for e in exposure)
    n_tokens = sum(e["tokens"] for e in exposure)
    note = (f"{'Host speech only' if not a.all_speakers else 'All speech'}, STOP and FILLERS removed"
            f"{' (fillers kept)' if a.keep_fillers else ''}; {n_videos} videos, {n_tokens:,} tokens, "
            f"{len({e['month'] for e in exposure})} months; rate per million tokens per speaker x month; "
            f"95% intervals resample whole videos ({a.n_boot} replicates).")
    (out / "trends.html").write_text(render_html(rows, exposure, words, a.title, note), encoding="utf-8")
    png = render_png(rows, words, a.title, note, out / "trends.png")
    print(f"words: {', '.join(words)}")
    print(f"cells: {len(cells)}  videos: {n_videos}  tokens: {n_tokens:,}")
    print(f"wrote {out / 'trends.csv'}, {out / 'months.csv'}, {out / 'trends.html'}" + (f", {png}" if png else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
