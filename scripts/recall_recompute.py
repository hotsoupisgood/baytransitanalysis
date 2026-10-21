#!/usr/bin/env python3
"""
scripts/recall_recompute.py -- re-score the locate-recall check under the
CURRENT keyword list (or any keyword list), for $0 (plan/RECALL-CHECK.md
"Keyword screen expansion", 2026-10-18).

The exhaustive Sonnet labels in <src>/windows.jsonl are a property of the GPU
transcript and the codebook, not of the keyword screen, so they are reused as
they are. Everything that depends on the screen or on the pipeline's episodes
is recomputed with the keyword list given: screen_s / screen_gpu_s
(segment.candidate_windows on the locate track and on the GPU text),
hits_per_100w / family_rates / top_keywords, pipeline_label (classifier_calls,
bounds-matched) and episode_s / episode_tags (the `episodes` rows of the arm's
method). recall_check.analyze then runs unchanged, writing into a SEPARATE
directory so the pre-expansion artefacts are kept for comparison.

    python3 scripts/recall_recompute.py [--out output/validation/recall/post_kw2]
                                        [--keywords-git 35df7c7]   # the English pair at a commit
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state            # noqa: E402
import segment as sg    # noqa: E402
import classify as cl   # noqa: E402
import recall_check as rc   # noqa: E402


def keywords_at(rev, lang="en"):
    """The keyword pair for `lang` as committed at git revision `rev`."""
    d = Path(tempfile.mkdtemp(prefix="kw-"))
    paths = []
    for f in cl.keyword_files_for(lang):
        rel = f.relative_to(ROOT)
        p = d / f.name
        p.write_bytes(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{rev}:{rel.as_posix()}"]))
        paths.append(p)
    return sg.load_keywords(*paths)


def recompute(con, src, out, keywords, max_run=None):
    """Rebuild <out>/windows.jsonl + sample.json from <src> under `keywords`
    and the episodes as they are now, then rc.analyze into <out>. `max_run`
    (2026-10-18): the candidate granularity the pipeline's rows for these
    meetings were labelled at -- the default moved to 120 s that day, but the
    rows of the sampled meetings are 240-s readings (bare prompt_version), so
    the bounds-matched pipeline_label needs 240 here; None = cl.PV_MAX_RUN.
    The screen (screen_s / screen_gpu_s) does not depend on it."""
    src, out = Path(src), Path(out)
    p = dict(cl.DEFAULTS, max_run=float(cl.PV_MAX_RUN if max_run is None else max_run))
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (src / "windows.jsonl").read_text().splitlines() if l.strip()]
    sample = json.loads((src / "sample.json").read_text())
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r["video_id"], []).append(r)
    W, P = rc.WINDOW, rc.PAD
    for m in sample["meetings"]:
        vid = m["video_id"]
        if not m.get("transcript") or vid not in by_vid:
            continue
        doc = json.loads(Path(m["transcript"]).read_text())
        segs = [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]} for x in doc["segments"]]
        segs_asr = sg.read_segments(m["vtt"])
        cands, cand_labels, extra = rc.pipeline_verdicts(con, vid, segs_asr, keywords, p)
        cand_spans = [(c["start_s"], c["end_s"]) for c in cands]
        gpu_cands = sg.candidate_windows(segs, keywords, W, p["threshold"], p["max_run"])
        gpu_spans = [(c["start_s"], c["end_s"]) for c in gpu_cands]
        eps = [dict(r) for r in con.execute("SELECT episode_id, start_s, end_s, tags, status, label FROM episodes WHERE video_id = ? "
                                            "AND method = ? AND status != 'mention' ORDER BY start_s", (vid, rc.METHOD))]
        for r in by_vid[vid]:
            a_, b_ = r["start_s"], r["end_s"]
            c = sg.hits(r["text"], keywords)
            fh = sg.family_hits(r["text"], keywords)
            scr = rc.covered(cand_spans, a_, b_)
            lab = None
            for cnd in cands:
                if cnd["start_s"] <= a_ < cnd["end_s"] or (cnd["start_s"] < b_ and cnd["end_s"] > a_
                                                            and rc.covered([(cnd["start_s"], cnd["end_s"])], a_, b_) >= W / 2):
                    lab = cand_labels.get(cnd["id"])
                    break
            if lab is None and scr > 0:
                for x in extra:
                    if x["start_s"] < b_ and x["end_s"] > a_:
                        lab = x["label"]
            r.update({"hits_per_100w": round(sg.density_per_100w(r["words"], sum(c.values())), 2),
                      "family_rates": {f: round(sg.density_per_100w(r["words"], k), 2) for f, k in sorted(fh.items())},
                      "top_keywords": [t for t, _ in c.most_common(5)],
                      "screen_s": round(scr, 1), "screen_gpu_s": round(rc.covered(gpu_spans, a_, b_), 1),
                      "pipeline_label": lab,
                      "episode_s": round(rc.covered([(e["start_s"], e["end_s"]) for e in eps], a_, b_), 1),
                      "episode_tags": sorted({e["tags"] for e in eps if e["start_s"] < b_ and e["end_s"] > a_})})
        rc.log(f"  {vid}: screen marks {sum(1 for r in by_vid[vid] if r['screen_s'] >= W / 2)} min on the locate track, "
               f"{sum(1 for r in by_vid[vid] if r['screen_gpu_s'] >= W / 2)} on the GPU text; {len(eps)} located episodes = "
               f"{sum(e['end_s'] - e['start_s'] for e in eps) / 60:.1f} min; candidates {len(cands)}, "
               f"{sum(1 for v in cand_labels.values() if v is None)} unlabelled")
    sample["keywords_version"] = keywords.version
    sample["recomputed_from"] = str(src)
    (out / "sample.json").write_text(json.dumps(sample, indent=1))
    (out / "windows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    prev = rc.OUT
    rc.OUT = out
    try:
        rc.log(f"keyword list {keywords.version} ({len(keywords)} terms); analyzing into {out}")
        return rc.analyze(con)
    finally:
        rc.OUT = prev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(rc.OUT / "post_kw2"))
    ap.add_argument("--src", default=str(rc.OUT))
    ap.add_argument("--keywords-git", default=None, help="load the English keyword pair from this git revision instead of disk")
    ap.add_argument("--captions", default=None, help="videos.captions kind of the arm's locate track ('auto' adds the "
                    "caption-alias layer, classify.ALIAS_FILES); default: the plain list")
    ap.add_argument("--max-run", type=float, default=None,
                    help=f"candidate granularity the pipeline rows were labelled at (default {cl.PV_MAX_RUN:.0f})")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    keywords = keywords_at(a.keywords_git) if a.keywords_git else cl.keywords_for("en", a.captions)
    return recompute(con, a.src, a.out, keywords, a.max_run)


if __name__ == "__main__":
    sys.exit(main())
