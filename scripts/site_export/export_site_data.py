#!/usr/bin/env python
"""Export the dashboard's data files from the ytpt pipeline (READ-ONLY).

Run from the ytpt repo with its venv so pipeline.state can load .env:

    cd ~/ytpt && .venv/bin/python scripts/site_export/export_site_data.py

Writes, beside this repo's index.html:
    data/bodies.json            one row per body + meta (totals, instrument versions, QA, stance n)
    data/meetings.json          one row per meeting video, with its episode ids
    data/episodes.json          one light row per episode (no text); has_text marks gated ones
    data/episodes/<body>.json   the turns of every QA-gated episode of that body, loaded on demand
    data/terms.json             keyword + frame lexicons (legend only; spans are precomputed)
    data/figs/*.png             analysis/models figures, data/tables/*.csv the posterior tables
                                (kept in the repo and cited by REPORT.md; no page fetches them
                                since the topic-time panel came off the dashboard 2026-10-20)

Text rules (owner decision 2026-10-20 09:13, "go back to removing names" -- which
reverses the 2026-10-18 keep-names rule and puts plan/DECISIONS.md #15 back in force):
  * only QA-gated episodes (output/qa/latest.csv, kind=episode, status OK|WARN) carry text;
    that gate is exactly the 1,059-episode set loaded into `corpus.sqlite`;
  * every turn written under docs/ goes through `pipeline/redact.py` first: personal names
    (including the self-introduction cues), street addresses and contact details become
    `[name]`, `[address]`, `[contact]`. The redactor sees a whole turn at once, so a name
    split across two segments is still caught;
  * redaction happens BEFORE the keyword/frame spans are computed, so the highlight offsets
    index the published string rather than the original;
  * it runs on every body's shard, not just the one committed shard. The published set is a
    .gitignore keep-list, one `git add` away from changing, and nothing written under docs/
    should carry a name whether or not it is tracked today;
  * a turn = maximal run of consecutive segments by one diarized cluster
    (pipeline/stance.py episode_turns, the S1 pass's own function), indexed over turns with
    at least one word, so `turn_idx` is the join key to `turn_stance`;
  * stance is filled for **every scored turn** of the S1 pass (9,358 turns / 625 episodes,
    prompt_version st-7161e26ad2); stance_coverage = s1|none per episode.

Never prints .env values or the DB path. Never writes inside the ytpt repo.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

YTPT = Path("~/ytpt").expanduser()
SITE = Path(__file__).resolve().parents[2] / "docs"  # the GitHub Pages root of this repo
DATA = SITE / "data"
MODELS = YTPT / "output" / "analysis" / "models"
QA_CSV = YTPT / "output" / "qa" / "latest.csv"
CORPUS = Path("~/.local/share/ytpt/corpus.sqlite").expanduser()

sys.path.insert(0, str(YTPT))
from pipeline import state  # noqa: E402  (loads .env itself)
from pipeline import redact  # noqa: E402  (the pipeline's own redactor, used on all published text)

WORD_RE = re.compile(r"[A-Za-z0-9']+")


#: how many headings scrub_heading() changed, for the export's own redaction meta
HEADING_SCRUBS = Counter()


def scrub_heading(text):
    """Street addresses and contact details out of a published heading.

    `title` (the platform's own video title) and `agenda_item` are published free
    text, and agenda items name specific properties -- a design-review appeal is
    filed against an address. So the address and contact layers run on them.

    Most person layers deliberately do NOT. An agenda heading is Title Case
    throughout, which is the one shape spaCy's NER mislabels wholesale: it reads
    "Housing Element" and "Measure" as people, and redacting those would turn the
    line that tells a reader what the episode was about into placeholders. It is
    also the shape redact.py's cue regexes backtrack on without terminating, so
    the self-introduction layer cannot be run here -- and a heading is not where a
    speaker introduces themselves.

    The honorific cue IS run, since 2026-10-20. The claim this function used to
    make -- that spoken names are handled where they are actually said -- was
    wrong about headings: an agenda item names the official who authored it, and
    the publish scan found 57 of them here ("Report from Development Services
    Director Buizer", "From: Councilmember Houston", "State Senator Dr. Aisha
    Wahab"). That cue is anchored on a literal title word, so it is cheap on
    Title Case and cannot fire on "Housing Element". The given_name layer is
    still left out: on these headings it redacts firm names ("David Baker
    Architects"), which is meaning loss with no privacy gain.

    NOTE for whoever reads the published site: these are officials acting in
    their official capacity, named in an agenda the city publishes itself, and
    the episode's own `link` still points at that agenda. Removing the name here
    is Puck's 2026-10-20 rule applied literally; it does not make the official
    unidentifiable.
    """
    if not text:
        return text, 0
    spans = sorted(redact._spans_address(text) + redact._spans_contact(text)
                   + redact.spans_honorific(text))
    taken, keep = [], []
    for s, e, orig, rep, rule in spans:
        if any(s < te and e > ts for ts, te in taken):
            continue
        taken.append((s, e))
        keep.append((s, e, orig, rep, rule))
    for _s, _e, _o, _r, rule in keep:
        HEADING_SCRUBS[f"heading_{rule}"] += 1
    return redact.apply_spans(text, keep), len(keep)


# ── helpers ──────────────────────────────────────────────────────────────────
def rows(cur, sql, args=()):
    return [dict(r) for r in cur.execute(sql, args)]


def h(seconds):
    return round((seconds or 0) / 3600.0, 3)


def link_at(platform, source_url, s):
    """Deep link to the source video at second `s`.

    YouTube: &t=<int>s. Granicus MediaPlayer: &entrytime=<int>&autostart=1 -- the
    parameter pair the player page's own "embed" generator emits (probe of one
    santaclara MediaPlayer.php page, 2026-10-17; the page server-renders
    `start_time` into the Flowplayer options). Swagit: the page itself -- the
    player markup could not be read to find a seek parameter (petalumaca
    .new.swagit.com returns 403 to a plain GET, re-checked 2026-10-18), so the
    reader prints the offset to scrub to instead of inventing one."""
    if not source_url:
        return ""
    s = int(s or 0)
    if platform == "youtube":
        sep = "&" if "?" in source_url else "?"
        return f"{source_url}{sep}t={s}s"
    if platform == "granicus":
        sep = "&" if "?" in source_url else "?"
        return f"{source_url}{sep}entrytime={s}&autostart=1"
    return source_url


def load_keyword_file(path):
    """plan/keywords_*.txt loader: '#' comments, '@family', '? term' conditional."""
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("@"):
            continue
        cond = line.startswith("?")
        term = line.lstrip("? ").strip()
        if term:
            terms.append({"term": term, "conditional": cond})
    return terms


def load_frames(path):
    frames, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            cur = line[:-1].strip()
            frames[cur] = []
        elif line.strip().startswith("-") and cur:
            frames[cur].append(line.strip()[1:].strip())
    return frames


def term_pattern(term):
    """analysis/a0_terms.term_pattern: whole word, case-insensitive, optional 's, space ~ [\\s-]+."""
    body = r"[\s\-]+".join(re.escape(w) for w in term.lower().split())
    return re.compile(rf"(?<!\w){body}(?:'?s)?(?!\w)", re.I)


class SpanFinder:
    """Keyword and frame spans in a turn: [[start, end, cls], ...], cls in d|t|f:<frame>.
    Longest term wins an overlap; keywords win over frames at equal length."""

    def __init__(self, terms):
        self.items = []
        for fam, key in (("density", "d"), ("transit", "t")):
            for t in terms["keywords"][fam]:
                if not t["conditional"]:
                    self.items.append((term_pattern(t["term"]), key, len(t["term"])))
        for frame, lst in terms["frames"].items():
            for t in lst:
                self.items.append((term_pattern(t), "f:" + frame, len(t)))
        self.items.sort(key=lambda x: (-x[2], x[1]))

    def spans(self, text):
        taken, out = [], []
        for rx, cls, _n in self.items:
            for m in rx.finditer(text):
                s, e = m.start(), m.end()
                if any(s < te and e > ts for ts, te in taken):
                    continue
                taken.append((s, e))
                out.append([s, e, cls])
        out.sort()
        return out


def file_sha(path, n=10):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def turns_of(segs):
    """pipeline/stance.py episode_turns, over already-collapsed corpus rows:
    maximal runs of one cluster, indexed over runs with at least one word."""
    runs, cur = [], None
    for s in segs:
        spk = s["speaker"] or "UNLABELLED"
        if cur and cur["cluster"] == spk:
            cur["text"].append(s["text"])
            cur["end"] = float(s["end_s"])
        else:
            if cur:
                runs.append(cur)
            cur = {"cluster": spk, "start": float(s["start_s"]), "end": float(s["end_s"]), "text": [s["text"]]}
    if cur:
        runs.append(cur)
    out, idx = [], 0
    for r in runs:
        text = re.sub(r"\s{2,}", " ", " ".join(x.strip() for x in r["text"] if x)).strip()
        words = len(WORD_RE.findall(text.lower()))
        if not words:
            continue
        idx += 1
        out.append({"turn_idx": idx, "cluster": r["cluster"], "start_rel": round(r["start"], 1),
                    "end_rel": round(r["end"], 1), "words": words, "text": text})
    return out


def clean(o):
    """NaN/Inf -> None so the JSON the browser fetches is strict JSON."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    con = state.connect()
    cur = con.cursor()
    cp = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    cp.row_factory = sqlite3.Row

    geo = {r["body"]: r for r in csv.DictReader((DATA / "bodies_geo.csv").open(encoding="utf-8"))}
    for g in geo.values():
        g["lat"], g["lon"] = float(g["lat"]), float(g["lon"])

    videos = rows(cur, """select video_id, speaker as body, title, platform, source_url, meeting_id,
                                 meeting_date, upload_date, duration_s, status
                          from videos where kind='meeting'""")
    episodes = rows(cur, """select episode_id, meeting_id, video_id, start_s, end_s, tags, method,
                                   status, label, transcript_path, agenda_item, classifier_model, prompt_version
                            from episodes""")

    # roles: the corpus table is what every analysis read; state.speaker_roles is the fallback
    roles = defaultdict(dict)
    for r in rows(cur, "select episode_id, speaker, role, confidence, method, cues_version from speaker_roles"):
        roles[r["episode_id"]][r["speaker"]] = r
    for r in cp.execute("select episode_id, speaker, role, confidence, method, roles_source, talk_s, words, turns from episode_speakers"):
        d = roles[r["episode_id"]].setdefault(r["speaker"], {})
        d.update({"role": r["role"], "confidence": r["confidence"], "method": r["method"],
                  "roles_source": r["roles_source"]})

    stance_rows = rows(cur, """select episode_id, cluster, turn_idx, start_s, end_s, words, k, n_ok, n_numeric,
                                      stance, stance_median, stance_spread, stance_votes, na_share,
                                      hostility_median, hostility_spread, hostility_votes, frames, sub_topic,
                                      self_id, contains_instructions, prompt_version, model
                               from turn_stance""")
    stance_by_ep = defaultdict(dict)
    for r in stance_rows:
        stance_by_ep[r["episode_id"]][r["turn_idx"]] = r

    # role_audit (rule ra-1, plan/ROLE-AUDIT-2026-10-18.md) is the REPORTED role as of
    # 2026-10-18. The reader shows both when they differ, with the self_id that decided it,
    # because the audit is a second instrument and not a correction of a known-bad label:
    # a cue-phrase labeller cannot see self-identification, and 44 % of turns don't give one.
    audit_by_ep = defaultdict(dict)
    try:
        for r in rows(cur, """select episode_id, cluster, turn_idx, role_cue, self_id, role_audit,
                                     cluster_role_audit, rule, cluster_rule, audit_version
                              from role_audit"""):
            audit_by_ep[r["episode_id"]][r["turn_idx"]] = r
    except Exception as exc:                      # table absent on an older DB
        print(f"note: no role_audit table ({exc}); reader falls back to the cue role")

    # QA gate (status + the issue string, so the reader can show why a WARN warned)
    qa_status, qa_issues, qa_totals = {}, {}, Counter()
    with QA_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            qa_totals[(r["kind"], r["status"])] += 1
            if r["kind"] == "episode":
                qa_status[r["video_id"]] = r["status"]
                qa_issues[r["video_id"]] = (r.get("issues") or "").strip()
    gated = {eid for eid, s in qa_status.items() if s in ("OK", "WARN")}

    vid = {v["video_id"]: v for v in videos}
    derived_dates = 0
    for v in videos:
        if not v["meeting_date"]:
            v["meeting_date"] = v["meeting_id"].split(":", 1)[1] if v["meeting_id"] and ":" in v["meeting_id"] else None
            derived_dates += 1

    # ── terms + spans ───────────────────────────────────────────────────────
    plan = YTPT / "plan"
    terms = {
        "keywords": {
            "density": load_keyword_file(plan / "keywords_density.txt"),
            "transit": load_keyword_file(plan / "keywords_transit.txt"),
        },
        "frames": load_frames(plan / "frames.yaml"),
    }
    finder = SpanFinder(terms)

    # ── episodes (light index) ──────────────────────────────────────────────
    ep_by_meeting = defaultdict(list)
    ep_out = []
    for e in episodes:
        v = vid.get(e["video_id"])
        if not v:
            continue
        dur = (e["end_s"] or 0) - (e["start_s"] or 0)
        rl = roles.get(e["episode_id"], {})
        role_counts = Counter(r.get("role") for r in rl.values())
        row = {
            "episode_id": e["episode_id"],
            "meeting_id": e["meeting_id"],
            "body": v["body"],
            "start_s": round(e["start_s"] or 0, 1),
            "end_s": round(e["end_s"] or 0, 1),
            "hours": h(dur),
            "tags": e["tags"],
            "route": e["method"],
            "status": e["status"],
            "label": e["label"],
            "agenda_item": scrub_heading(e["agenda_item"])[0],
            "link": link_at(v["platform"], v["source_url"], e["start_s"] or 0),
            "speakers": {k: role_counts.get(k, 0) for k in ("dais", "public", "staff", "unknown")},
            "qa_status": qa_status.get(e["episode_id"]),
            "qa_issues": qa_issues.get(e["episode_id"]) or None,
            "has_text": e["episode_id"] in gated,
            "stance_coverage": "s1" if e["episode_id"] in stance_by_ep else "none",
        }
        ep_out.append(row)
        ep_by_meeting[e["meeting_id"]].append(row)
    ep_index = {r["episode_id"]: r for r in ep_out}
    ep_raw = {e["episode_id"]: e for e in episodes}

    # ── meetings ────────────────────────────────────────────────────────────
    meetings = []
    for v in videos:
        eps = [x for x in ep_by_meeting.get(v["meeting_id"], []) if x["meeting_id"] == v["meeting_id"]]
        meetings.append({
            "meeting_id": v["meeting_id"],
            "body": v["body"],
            "date": v["meeting_date"],
            "title": scrub_heading(v["title"])[0],
            "platform": v["platform"],
            "source_url": v["source_url"],
            "duration_h": h(v["duration_s"]),
            "episodes": [x["episode_id"] for x in sorted(eps, key=lambda x: x["start_s"])],
            "hours_done": round(sum(x["hours"] for x in eps if x["status"] == "done"), 3),
            "hours_located": round(sum(x["hours"] for x in eps if x["status"] == "located"), 3),
            "n_mention": sum(1 for x in eps if x["status"] == "mention"),
            "n_text": sum(1 for x in eps if x["has_text"]),
        })
    meetings.sort(key=lambda m: (m["body"], m["date"] or ""))

    # ── bodies ──────────────────────────────────────────────────────────────
    bodies = []
    for slug, g in geo.items():
        ms = [m for m in meetings if m["body"] == slug]
        eps = [e for e in ep_out if e["body"] == slug]
        done = [e for e in eps if e["status"] == "done"]
        dens = sum(e["hours"] for e in done if e["tags"] == "density") + 0.5 * sum(e["hours"] for e in done if e["tags"] == "both")
        tr = sum(e["hours"] for e in done if e["tags"] == "transit") + 0.5 * sum(e["hours"] for e in done if e["tags"] == "both")
        hours_done = dens + tr
        bodies.append({
            "body": slug, "name": g["name"], "county": g["county"], "kind": g["kind"],
            "lat": g["lat"], "lon": g["lon"], "site": g["site"],
            "platforms": sorted({m["platform"] for m in ms}),
            "n_meetings": len(ms),
            "meeting_hours": round(sum(m["duration_h"] for m in ms), 2),
            "n_episodes": len(eps),
            "n_done": len(done),
            "n_text": sum(1 for e in eps if e["has_text"]),
            "n_scored": sum(1 for e in eps if e["stance_coverage"] == "s1"),
            "routes": sorted({e["route"] for e in eps if e["route"]}),
            "hours_done": round(hours_done, 3),
            "hours_located": round(sum(e["hours"] for e in eps if e["status"] == "located"), 3),
            "hours_mention": round(sum(e["hours"] for e in eps if e["status"] == "mention"), 3),
            "density_share": round(dens / hours_done, 3) if hours_done > 0 else None,
            "date_min": min((m["date"] for m in ms if m["date"]), default=None),
            "date_max": max((m["date"] for m in ms if m["date"]), default=None),
        })

    # ── transcript text: every gated episode, redacted before anything reads it ──
    segs_by_ep = defaultdict(list)
    for r in cp.execute("select episode_id, seg_idx, start_s, end_s, speaker, role, text from episode_segments order by episode_id, seg_idx"):
        segs_by_ep[r["episode_id"]].append(dict(r))

    # Owner, 2026-10-20 09:13: names come back out of everything published. This is the
    # only place the corpus text enters the site, so it is the only place that has to be
    # right -- and it runs here, before turns_of() and before finder.spans(), so the
    # published `words`, `text` and highlight offsets all describe the redacted string.
    # redact_transcript() works a whole turn at a time (its own turns_of groups the
    # segments by speaker), which is what catches "My name is" / "Jane Smith" when the
    # cue and the name fall in different segments. A turn that was nothing but a name
    # goes wordless and drops out of the turn index; that is the documented reason the
    # stance join below keeps its (cluster, relative start) fallback, and the
    # stance_matched_by_position counter is what to watch if this changes.
    redaction = Counter()
    red_words = 0
    for eid in sorted(gated):
        segs = segs_by_ep.get(eid)
        if not segs:
            continue
        # publish=True: this is the publish path, so every allowlist tier is
        # capped at CORE. At the default FULL tier a name that is also a Bay Area
        # place keeps the place's protection and reaches the site in clear
        # ("Carlos", "Howard", "John"). CORE still protects the multi-word place,
        # so "San Carlos" survives and the person does not. It over-redacts a
        # handful of place and programme phrases; on published text that is the
        # right side to err on.
        new_doc, _audit = redact.redact_transcript({"segments": segs}, eid, publish=True)
        # _audit holds the original names. It is never written, printed or returned.
        segs_by_ep[eid] = new_doc["segments"]
        rinfo = new_doc.get("redaction") or {}
        redaction.update(rinfo.get("by_rule") or {})
        red_words += rinfo.get("words") or 0
    redaction.update(HEADING_SCRUBS)
    print(f"[redact] {redact.VERSION}: {sum(redaction.values())} replacements over "
          f"{red_words:,} words of {len(gated)} gated episodes; "
          f"by rule {dict(sorted(redaction.items())) or 'nothing'}")

    per_body = defaultdict(dict)
    stats = Counter()
    asr_versions = Counter()
    stance_unmatched = []
    for eid in sorted(gated):
        raw, row = ep_raw.get(eid), ep_index.get(eid)
        if not raw or not row:
            stats["gated_not_in_db"] += 1
            continue
        segs = segs_by_ep.get(eid)
        if not segs:
            stats["gated_no_corpus_text"] += 1
            row["has_text"] = False
            continue
        stats["text_from_corpus"] += 1
        base = raw["start_s"] or 0          # corpus segment clocks are episode-relative (checked: 0 of 1,059 absolute)
        rl = roles.get(eid, {})
        srows = stance_by_ep.get(eid, {})
        arows = audit_by_ep.get(eid, {})
        # Primary key is turn_idx. In 132 of 9,358 rows it slips: the S1 pass read the
        # redacted copies, where a turn that was nothing but a name became wordless and
        # dropped out of the index, shifting every later turn of that episode. Cluster +
        # relative start is identical in both, so it is the fallback and it closes the gap.
        by_pos = {(r["cluster"], round(r["start_s"] or 0, 1)): r for r in srows.values()}
        seen = set()

        out_turns = []
        for t in turns_of(segs):
            r = rl.get(t["cluster"], {})
            sr = srows.get(t["turn_idx"])
            if not (sr and sr["cluster"] == t["cluster"] and abs((sr["start_s"] or 0) - t["start_rel"]) <= 1.5):
                sr = by_pos.get((t["cluster"], t["start_rel"]))
            # role_audit shares turn_stance's keys, so it is looked up on the stance row
            # that actually matched -- not on the corpus turn_idx, which can have slipped.
            ar = arows.get(sr["turn_idx"]) if (sr and sr["cluster"] == t["cluster"]) else None
            if ar and ar["cluster"] != t["cluster"]:
                ar = None
            st = None
            if sr and sr["cluster"] == t["cluster"]:
                seen.add(sr["turn_idx"])
                stats["role_audit_matched"] += int(ar is not None)
                stats["role_audit_changed"] += int(bool(
                    ar and ar.get("cluster_role_audit")
                    and ar["cluster_role_audit"] != r.get("role")))
                stats["stance_matched_by_position"] += int(sr["turn_idx"] != t["turn_idx"])
                st = {
                    "median": sr["stance_median"], "spread": sr["stance_spread"],
                    "votes": sr["stance_votes"], "na_share": sr["na_share"], "label": sr["stance"],
                    "hostility": sr["hostility_median"], "hostility_spread": sr["hostility_spread"],
                    "frames": [f for f in (sr["frames"] or "").split(";") if f],
                    "sub_topic": sr["sub_topic"], "self_id": sr["self_id"],
                    "k": sr["k"], "n_ok": sr["n_ok"], "n_numeric": sr["n_numeric"],
                    "flagged": bool(sr["contains_instructions"]),
                }
                stats["stance_turns_matched"] += 1
            out_turns.append({
                "i": t["turn_idx"],
                "t_s": round(base + t["start_rel"], 1),
                "end_s": round(base + t["end_rel"], 1),
                "speaker": t["cluster"],
                # `role` is the REPORTED role: role_audit's cluster roll-up where the audit
                # has a verdict, the diarization-cue role otherwise (which is also what the
                # rule itself does for self_id = unknown). `role_cue` is always carried, and
                # `role_changed` lets the reader show both side by side where they differ.
                "role": (ar.get("cluster_role_audit") or r.get("role", "unknown")) if ar
                        else r.get("role", "unknown"),
                "role_cue": r.get("role", "unknown"),
                "role_audit_turn": ar.get("role_audit") if ar else None,
                "role_changed": bool(ar and ar.get("cluster_role_audit")
                                     and ar["cluster_role_audit"] != r.get("role")),
                "role_self_id": ar.get("self_id") if ar else (sr["self_id"] if sr else None),
                "role_rule": ar.get("cluster_rule") if ar else None,
                "role_audit_version": ar.get("audit_version") if ar else None,
                "role_conf": r.get("confidence"),
                "role_method": r.get("method"),
                "role_source": r.get("roles_source"),
                "words": t["words"],
                "spans": finder.spans(t["text"]),
                "text": t["text"],
                "stance": st,
            })
        for ti in srows:
            if ti not in seen:
                stance_unmatched.append((eid, ti))
        stats["turns"] += len(out_turns)
        stats["words"] += sum(t["words"] for t in out_turns)

        asr = {"model": None, "diarization": None, "pipeline_version": None, "route": raw["method"]}
        tp = Path(raw["transcript_path"]) if raw["transcript_path"] else None
        if tp and tp.exists():
            try:
                d = json.loads(tp.read_text(encoding="utf-8"))
                asr.update({"model": d.get("model"), "diarization": d.get("diarization_model"),
                            "pipeline_version": d.get("pipeline_version")})
                stk = d.get("stack") or {}
                asr_versions[(asr["model"], asr["diarization"], stk.get("pyannote.audio"), stk.get("faster_whisper"))] += 1
            except Exception:
                stats["transcript_unreadable"] += 1
        row["n_turns"] = len(out_turns)
        row["n_scored_turns"] = len(seen)
        row["words"] = sum(t["words"] for t in out_turns)
        per_body[row["body"]][eid] = {
            "episode_id": eid,
            "words": row["words"],
            "asr": asr,
            "roles_source": next((r.get("roles_source") for r in rl.values() if r.get("roles_source")), None),
            "roles_cues_version": next((r.get("cues_version") for r in rl.values() if r.get("cues_version")), None),
            "n_scored_turns": len(seen),
            "turns": out_turns,
        }

    # ── instrument versions (from the DB and files, not typed in) ──────────
    cls_versions = rows(cur, """select classifier_model as model, prompt_version, count(*) as n
                                from episodes where classifier_model is not null group by 1,2 order by n desc""")
    cls_calls = rows(cur, "select model, prompt_version, count(*) as n from classifier_calls group by 1,2 order by n desc")
    stance_versions = rows(cur, "select model, prompt_version, count(*) as n, count(distinct episode_id) as n_episodes from turn_stance group by 1,2")
    role_versions = rows(cur, "select cues_version, method, count(*) as n from speaker_roles group by 1,2 order by n desc")
    codebook_path = plan / "STANCE-CODEBOOK.md"
    codebook_txt = codebook_path.read_text(encoding="utf-8")
    marker = "--- CODEBOOK ---"
    codebook_block = (codebook_txt.split(marker, 1)[1].strip() + "\n") if marker in codebook_txt else ""
    codebook_hash_now = "st-" + hashlib.sha256(codebook_block.encode("utf-8")).hexdigest()[:10] if codebook_block else None
    asr_main = asr_versions.most_common(1)[0] if asr_versions else ((None,) * 4, 0)
    instruments = {
        "classifier": {"episodes_by_version": cls_versions, "calls_by_version": cls_calls},
        "asr": {"model": asr_main[0][0], "diarization": asr_main[0][1], "pyannote_audio": asr_main[0][2],
                "faster_whisper": asr_main[0][3], "n_episodes": asr_main[1],
                "variants": [{"model": k[0], "diarization": k[1], "pyannote_audio": k[2], "faster_whisper": k[3], "n": n}
                             for k, n in asr_versions.most_common()]},
        "roles": role_versions,
        "stance": {"runs": stance_versions, "codebook_hash_on_disk": codebook_hash_now,
                   "codebook_file_sha": file_sha(codebook_path), "n_turns": len(stance_rows),
                   "n_episodes": len(stance_by_ep), "kappa_gate": "not passed (no human has coded any turn)"},
        "redaction": {"version": redact.VERSION, "by_rule": dict(sorted(redaction.items())),
                      "total": sum(redaction.values()), "words": red_words,
                      "rule": "names, street addresses and contact details removed from every "
                              "published turn (owner, 2026-10-20 09:13; plan/DECISIONS.md #15)"},
    }

    # ── models figures and posterior tables ─────────────────────────────────
    # The PyMC topic-time panel came off the dashboard on 2026-10-20: it predates the
    # current corpus, the measured rates in the Rates panel are the live number, and the
    # role gap has its own panel. The figures and tables are still copied -- they stay in
    # the repo and REPORT.md cites them -- but data/models/summary.json is no longer
    # written and no page fetches it.
    if (MODELS / "summary.json").exists():
        (DATA / "figs").mkdir(parents=True, exist_ok=True)
        (DATA / "tables").mkdir(parents=True, exist_ok=True)
        figs = sorted(p.name for p in (MODELS / "figs").glob("*.png"))
        for f in figs:
            shutil.copyfile(MODELS / "figs" / f, DATA / "figs" / f)
        tables = sorted(p.name for p in (MODELS / "tables").glob("*.csv"))
        for t in tables:
            shutil.copyfile(MODELS / "tables" / t, DATA / "tables" / t)

    # ── write ───────────────────────────────────────────────────────────────
    with_hours = sorted([b for b in bodies if b["hours_done"] > 0], key=lambda b: -b["hours_done"])
    meta = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "exploratory": True,
        "redacted": True,
        "published": True,
        # What "published" covers, so the page can stop claiming nothing is: the dashboard
        # itself, the NIMBY/YIMBY quotes, and one body's episode shard. The owner asked to
        # keep "some auditable transcripts" readable (2026-10-20 09:13) -- redacted, and
        # every turn still links to the public recording at the same timestamp.
        "published_note": ("the dashboard, the NIMBY/YIMBY quotes and one body's episode shard are "
                           "published, all redacted; the other bodies' shards stay local "
                           "(the keep-list is docs/.gitignore)"),
        "derived_dates": derived_dates,
        "text": {
            "episodes_with_text": sum(1 for e in ep_out if e["has_text"]),
            "turns": stats["turns"],
            "words": stats["words"],
            "from_corpus": stats["text_from_corpus"],
            "gated_no_corpus_text": stats["gated_no_corpus_text"],
            "stance_turns_matched": stats["stance_turns_matched"],
            "stance_matched_by_position": stats["stance_matched_by_position"],
            "stance_turns_unmatched": len(stance_unmatched),
            "redaction": (f"{redact.VERSION}: names, street addresses and contact details removed "
                          f"from every published turn ({sum(redaction.values())} replacements), "
                          f"before the keyword spans were computed"),
        },
        "qa": {f"{k[0]}_{k[1]}": n for k, n in sorted(qa_totals.items())},
        "links": {
            "youtube": "&t=<s>s",
            "granicus": "&entrytime=<s>&autostart=1 (parameters the MediaPlayer page's own embed generator emits; playback at that offset not verified in a browser)",
            "swagit": "player page only; the page returns 403 to a plain GET so no seek parameter could be read -- the reader prints the offset to scrub to",
        },
        "instruments": instruments,
        "totals": {
            "bodies": len(bodies),
            "bodies_with_hours": len(with_hours),
            "bodies_with_text": sum(1 for b in bodies if b["n_text"]),
            "meetings": len(meetings),
            "meetings_by_platform": dict(Counter(m["platform"] for m in meetings)),
            "episodes": len(ep_out),
            "episodes_by_status": dict(Counter(e["status"] for e in ep_out)),
            "episodes_by_route": dict(Counter(e["route"] for e in ep_out if e["route"])),
            "gated_by_route": dict(Counter(e["route"] for e in ep_out if e["has_text"] and e["route"])),
            "episodes_done": sum(1 for e in ep_out if e["status"] == "done"),
            "episodes_located": sum(1 for e in ep_out if e["status"] == "located"),
            "episodes_with_text": sum(1 for e in ep_out if e["has_text"]),
            "episodes_scored": len(stance_by_ep),
            "stance_turns": len(stance_rows),
            "hours_done": round(sum(b["hours_done"] for b in bodies), 2),
            "hours_located": round(sum(b["hours_located"] for b in bodies), 2),
            "meeting_hours": round(sum(b["meeting_hours"] for b in bodies), 1),
        },
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "episodes").mkdir(exist_ok=True)
    for old in (DATA / "episodes").glob("*.json"):
        old.unlink()
    sizes = {}
    for slug, eps in per_body.items():
        p = DATA / "episodes" / f"{slug}.json"
        p.write_text(json.dumps(clean({"body": slug, "episodes": eps}), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        sizes[slug] = p.stat().st_size
    for b in bodies:
        b["text_file"] = f"data/episodes/{b['body']}.json" if b["body"] in per_body else None
        b["text_bytes"] = sizes.get(b["body"], 0)
    meta["episode_files"] = {"n": len(sizes), "bytes_total": sum(sizes.values()), "bytes_max": max(sizes.values(), default=0)}
    (DATA / "bodies.json").write_text(json.dumps(clean({"meta": meta, "bodies": bodies}), indent=1), encoding="utf-8")
    (DATA / "meetings.json").write_text(json.dumps(clean(meetings), indent=0), encoding="utf-8")
    (DATA / "episodes.json").write_text(json.dumps(clean(ep_out), indent=0, ensure_ascii=False), encoding="utf-8")
    (DATA / "terms.json").write_text(json.dumps(terms, indent=0), encoding="utf-8")

    print(json.dumps({k: v for k, v in meta.items() if k != "instruments"}, indent=1))
    print("instruments:", json.dumps({
        "classifier": instruments["classifier"]["episodes_by_version"],
        "asr": {k: v for k, v in instruments["asr"].items() if k != "variants"},
        "stance": instruments["stance"]["runs"], "codebook_now": codebook_hash_now,
        "roles": [r["cues_version"] for r in role_versions[:1]]}, indent=1))
    if stance_unmatched:
        print("UNMATCHED stance turns:", stance_unmatched[:10], "…" if len(stance_unmatched) > 10 else "")


if __name__ == "__main__":
    main()
