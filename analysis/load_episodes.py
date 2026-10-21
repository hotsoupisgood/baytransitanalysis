#!/usr/bin/env python3
"""
load_episodes.py -- add the T1 meeting episodes to corpus.sqlite, with roles.

WHY A SEPARATE LOADER: analysis/load_db.py builds `videos` / `segments` for the
Hasan/Tucker side corpus (host vs guest); that shape is untouched. A meeting
episode has a different unit of analysis -- a diarized cluster with a ROLE
(dais / public / unknown, plan/SCHEMA.md speaker_roles) inside an episode with
a family label, inside a meeting of a body -- so it gets its own tables in the
same corpus.sqlite (episodes, episode_speakers, episode_segments, meetings).
Everything here is derived: delete the tables and re-run.

Sources (read-only):
  * state.sqlite (STATE_DB_PATH): episodes rows with status='done' and their
    meeting row (body = videos.speaker, meeting_date), speaker_roles, and the
    mention_only / located rows for the exposure plots.
  * the episode transcript JSON (episodes.transcript_path).

Roles: speaker_roles rows when they exist (written by gpu_job.ingest_new since
b14). The b13 episodes were transcribed before the labeller was wired, so for
an episode with no speaker_roles rows the labels are computed here in memory
with pipeline/roles.label_transcript (text + timing cues, $0, no network) and
tagged roles_source='computed'. Nothing is written back to state.sqlite or the
JSON: this loader only reads them.

    .venv/bin/python analysis/load_episodes.py --db ~/.local/share/ytpt/corpus.sqlite
"""
import argparse, json, os, re, shutil, sqlite3, sys, tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "analysis"))

WORD_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")   # word_freq.TOKEN_RE, before any filtering

DDL = """
DROP TABLE IF EXISTS episode_segments;
DROP TABLE IF EXISTS episode_speakers;
DROP TABLE IF EXISTS episodes;
DROP TABLE IF EXISTS meeting_exposure;
DROP TABLE IF EXISTS meetings;
CREATE TABLE meetings (
  video_id TEXT PRIMARY KEY, body TEXT NOT NULL, meeting_date TEXT, month TEXT,
  duration_s REAL, status TEXT, platform TEXT,
  n_about INTEGER, n_mention INTEGER, n_done INTEGER,   -- episode rows located / mention_only / transcribed
  about_core_s REAL, density_core_s REAL, transit_core_s REAL, both_core_s REAL
);
-- The same exposure numbers split by locate route (episodes.method). meetings
-- holds the sum over routes; this holds one row per (meeting, route) so the
-- rate figures can be faceted without re-reading state.sqlite. The DENOMINATOR
-- (meetings.duration_s) is shared: a meeting is exposure for every route that
-- read it, so only the numerator splits. See analysis/facets.py.
CREATE TABLE meeting_exposure (
  video_id TEXT NOT NULL, method TEXT NOT NULL,
  n_about INTEGER, n_mention INTEGER, n_done INTEGER,
  about_core_s REAL, density_core_s REAL, transit_core_s REAL, both_core_s REAL,
  PRIMARY KEY (video_id, method)
);
CREATE TABLE episodes (
  episode_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, body TEXT NOT NULL,
  meeting_date TEXT, month TEXT, label TEXT, family TEXT,   -- family = tags: density | transit | both
  method TEXT,                                              -- FACET: locate route, classifier | agenda_index | local_asr
  start_s REAL, end_s REAL, core_start_s REAL, core_end_s REAL, duration_s REAL,
  agenda_item TEXT, transcript_path TEXT, roles_source TEXT,  -- FACET: 'db' | 'computed' | 'json'
  n_segments INTEGER, n_speakers INTEGER, words INTEGER, talk_s REAL
);
CREATE TABLE episode_speakers (
  episode_id TEXT NOT NULL, speaker TEXT NOT NULL,
  role TEXT NOT NULL, confidence REAL, method TEXT, roles_source TEXT,
  talk_s REAL, words INTEGER, turns INTEGER,
  PRIMARY KEY (episode_id, speaker)
);
CREATE TABLE episode_segments (
  episode_id TEXT NOT NULL, seg_idx INTEGER NOT NULL,
  start_s REAL, end_s REAL, speaker TEXT, role TEXT, text TEXT NOT NULL,
  PRIMARY KEY (episode_id, seg_idx)
);
CREATE INDEX idx_eseg_role ON episode_segments(role);
"""


def state_db_path(explicit=None):
    return Path(os.path.expanduser(explicit or os.environ.get("STATE_DB_PATH")
                                   or "~/.local/share/ytpt/state.sqlite"))


def roles_for(doc, db_roles):
    """{speaker: (role, confidence, method)} and where they came from."""
    if db_roles:
        return {k: (r, c, m) for k, (r, c, m) in db_roles.items()}, "db"
    if doc.get("roles"):
        return {k: (v["role"], v.get("confidence"), v.get("method")) for k, v in doc["roles"].items()}, "json"
    import roles as pr   # pipeline/roles.py, lazily: it pulls scripts/role_label.py and the cue file
    roles, _regions, _summ = pr.label_transcript(doc)
    return {k: (v["role"], v["confidence"], v["method"]) for k, v in roles.items()}, "computed"


#  T1 pilot (plan/ANALYSIS-transit.md 6, phase A0). The corpus has grown since;
#  a phase that says "the pilot" must name the pilot, not "whatever is done now".
#  The three parts are independent and all three are needed:
#    bodies  -- Santa Rosa and Sunnyvale were classified in the same 2026-09-21
#               run but never transcribed; they are not the pilot.
#    before  -- the classifier has run on 60+ further meetings since (T2).
#    batches -- b13 + b14/b15 are the pilot's GPU batches (b14 failed, b15 redid
#               the same 17 episodes; naming all three is order-independent).
PILOT = {"bodies": ("bart", "oakland", "sanjose", "sanrafael"),
         "classified_before": "2026-09-22",
         "batches": ("b13", "b14", "b15")}

#  A0-WIDE scope (plan/A0-WIDE-RESULTS.md): every body and every locate route
#  that is through the QA gate -- 99 episodes at the time of writing. Named by
#  QA report rather than by batch because batch ids overlap: the 30 b18
#  episodes also carry b17, so "--batches b13..b17" would silently pull in a
#  batch that has never been QA'd. The pilot stays pinned and untouched.
WIDE = {"qa_gate": "output/qa/a0_episodes.csv"}


def select_meetings(st, bodies=None, classified_before=None):
    """video_ids of the meetings that make up the exposure denominator."""
    keep = {r["video_id"] for r in st.execute("SELECT video_id FROM videos WHERE kind = 'meeting'")}
    if bodies:
        q = ",".join("?" * len(bodies))
        keep &= {r[0] for r in st.execute(
            f"SELECT video_id FROM videos WHERE kind = 'meeting' AND speaker IN ({q})", tuple(bodies))}
    if classified_before:
        keep &= {r[0] for r in st.execute(
            "SELECT DISTINCT video_id FROM classifier_calls WHERE at < ?", (classified_before,))}
    return keep


def select_batches(st, batches):
    """episode_ids transcribed under these GPU batches (batch_videos.video_id is
    the episode id for meeting episodes)."""
    q = ",".join("?" * len(batches))
    return {r[0] for r in st.execute(
        f"SELECT video_id FROM batch_videos WHERE batch_id IN ({q})", tuple(batches))}


def select_qa_gate(csv_path):
    """episode_ids that passed the QA gate in a qa_transcripts.py report.

    'Through the QA gate' is not the same set as status='done': a batch
    transcribed after the last `make qa` run has no QA row at all, and A0-RESULTS
    6 counts only the episodes that have one. Passing the report file (rather
    than a batch list) makes the wide scope say what it means and keeps it
    reproducible -- batch ids overlap (b17 covers two ingests), QA rows do not."""
    import csv as _csv
    keep = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("kind") == "episode" and r.get("status") in ("OK", "WARN"):
                keep.add(r["video_id"])
    return keep


def load(state, db, collapse, bodies=None, classified_before=None, batches=None, qa_gate=None,
         transcripts=None):
    """`transcripts`: a directory of <episode_id>.json to read INSTEAD of
    episodes.transcript_path -- the name-redacted copies in output/redacted/
    (pipeline/redact.py, DECISIONS #15: episode text passes redaction before
    anything loads it). An episode with no file there is skipped and listed,
    never silently read from output/whisper."""
    st = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    st.row_factory = sqlite3.Row
    keep_m = select_meetings(st, bodies, classified_before)
    keep_e = select_batches(st, batches) if batches else None
    if qa_gate is not None:
        keep_e = qa_gate if keep_e is None else (keep_e & qa_gate)
    meetings = {r["video_id"]: dict(r) for r in st.execute(
        "SELECT video_id, speaker AS body, meeting_date, month, duration_s, status, platform "
        "FROM videos WHERE kind = 'meeting'") if r["video_id"] in keep_m}
    eps = [dict(r) for r in st.execute(
        "SELECT episode_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, label, status, "
        "method, agenda_item, transcript_path FROM episodes ORDER BY episode_id") if r["video_id"] in keep_m]
    db_roles = defaultdict(dict)
    for r in st.execute("SELECT episode_id, speaker, role, confidence, method FROM speaker_roles"):
        db_roles[r["episode_id"]][r["speaker"]] = (r["role"], r["confidence"], r["method"])
    st.close()

    # meeting-level exposure: every located episode counts, transcribed or not.
    # Kept twice -- pooled over routes (agg) and split by route (agg_m) -- so a
    # per-method rate figure uses the per-method numerator and never the pooled
    # one. Both are built from the same loop, so they cannot drift apart.
    agg = defaultdict(lambda: defaultdict(float))
    agg_m = defaultdict(lambda: defaultdict(float))
    for e in eps:
        m = agg[e["video_id"]]
        mm = agg_m[(e["video_id"], e["method"] or "unknown")]
        core = (e["core_end_s"] or e["end_s"]) - (e["core_start_s"] or e["start_s"])
        if e["label"] == "mention_only":
            m["n_mention"] += 1; mm["n_mention"] += 1
        elif e["status"] != "failed":
            m["n_about"] += 1; mm["n_about"] += 1
            m["about_core_s"] += core; mm["about_core_s"] += core
            m[f"{e['tags']}_core_s"] += core; mm[f"{e['tags']}_core_s"] += core
        if e["status"] == "done":
            m["n_done"] += 1; mm["n_done"] += 1

    con = sqlite3.connect(db)
    con.executescript(DDL)
    for vid, m in meetings.items():
        a = agg.get(vid, {})
        con.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (vid, m["body"], m["meeting_date"], m["month"], m["duration_s"], m["status"], m["platform"],
                     int(a.get("n_about", 0)), int(a.get("n_mention", 0)), int(a.get("n_done", 0)),
                     a.get("about_core_s", 0.0), a.get("density_core_s", 0.0),
                     a.get("transit_core_s", 0.0), a.get("both_core_s", 0.0)))
    for (vid, meth), a in agg_m.items():
        if vid not in meetings:
            continue
        con.execute("INSERT INTO meeting_exposure VALUES (?,?,?,?,?,?,?,?,?)",
                    (vid, meth, int(a.get("n_about", 0)), int(a.get("n_mention", 0)), int(a.get("n_done", 0)),
                     a.get("about_core_s", 0.0), a.get("density_core_s", 0.0),
                     a.get("transit_core_s", 0.0), a.get("both_core_s", 0.0)))

    n_ep = n_seg = n_words = 0
    sources = defaultdict(int)
    redaction_versions = defaultdict(int)
    missing = []
    for e in eps:
        if e["status"] != "done" or not e["transcript_path"]:
            continue
        if keep_e is not None and e["episode_id"] not in keep_e:
            continue
        if transcripts is not None:
            path = Path(transcripts) / f"{e['episode_id']}.json"
            if not path.is_absolute():
                path = ROOT / path
        else:
            path = Path(e["transcript_path"])
            if not path.is_absolute():
                path = ROOT / path
        if not path.exists():
            print(f"  {e['episode_id']}: transcript missing at {path}; skipped")
            missing.append(e["episode_id"])
            continue
        doc = json.loads(path.read_text())
        if transcripts is not None:
            redaction_versions[(doc.get("redaction") or {}).get("version") or "UNREDACTED"] += 1
        m = meetings.get(e["video_id"]) or {"body": doc.get("body") or doc.get("speaker"),
                                              "meeting_date": None, "month": doc.get("month")}
        segs, _dropped = collapse(doc["segments"])
        roles, source = roles_for(doc, db_roles.get(e["episode_id"]))
        sources[source] += 1
        per_spk = defaultdict(lambda: {"talk_s": 0.0, "words": 0, "turns": 0})
        rows, prev = [], None
        for i, s in enumerate(segs):
            spk = s.get("speaker") or "UNLABELLED"
            role = roles.get(spk, ("unknown", None, "none"))[0]
            words = len(WORD_RE.findall(s["text"].lower()))
            p = per_spk[spk]
            p["talk_s"] += float(s["end"]) - float(s["start"])
            p["words"] += words
            if spk != prev:
                p["turns"] += 1
            prev = spk
            rows.append((e["episode_id"], i, s["start"], s["end"], spk, role, s["text"]))
        con.executemany("INSERT INTO episode_segments VALUES (?,?,?,?,?,?,?)", rows)
        for spk, p in per_spk.items():
            role, conf, method = roles.get(spk, ("unknown", None, "none"))
            con.execute("INSERT INTO episode_speakers VALUES (?,?,?,?,?,?,?,?,?)",
                        (e["episode_id"], spk, role, conf, method, source,
                         round(p["talk_s"], 1), p["words"], p["turns"]))
        words = sum(p["words"] for p in per_spk.values())
        talk = sum(p["talk_s"] for p in per_spk.values())
        con.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (e["episode_id"], e["video_id"], m["body"], m.get("meeting_date"), m.get("month"),
                     e["label"], e["tags"], e["method"] or "unknown",
                     e["start_s"], e["end_s"], e["core_start_s"], e["core_end_s"],
                     doc.get("duration_s") or (e["end_s"] - e["start_s"]), e.get("agenda_item"), str(path),
                     source, len(rows), len(per_spk), words, round(talk, 1)))
        n_ep += 1; n_seg += len(rows); n_words += words
    con.commit()
    con.execute("PRAGMA optimize")
    if transcripts is not None:
        print(f"  transcripts read from {transcripts}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(redaction_versions.items()))
              + (f"; {len(missing)} done episode(s) had no file there and were skipped: "
                 + " ".join(missing[:20]) + (" ..." if len(missing) > 20 else "") if missing else ""))
    return con, n_ep, n_seg, n_words, dict(sources), len(meetings)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus.sqlite"))
    ap.add_argument("--state", default=None, help="state.sqlite (default STATE_DB_PATH)")
    ap.add_argument("--pilot", action="store_true",
                    help=f"restrict to the T1 pilot: bodies {'/'.join(PILOT['bodies'])}, meetings classified "
                         f"before {PILOT['classified_before']}, batches {'+'.join(PILOT['batches'])}")
    ap.add_argument("--bodies", nargs="*", default=None, help="keep only meetings of these bodies")
    ap.add_argument("--classified-before", default=None, help="keep only meetings with a classifier call before this timestamp")
    ap.add_argument("--batches", nargs="*", default=None, help="keep only episodes transcribed in these GPU batches")
    ap.add_argument("--qa-gate", default=None, metavar="CSV",
                    help="keep only episodes with status OK/WARN in this qa_transcripts.py report "
                         "(output/qa/a0_episodes.csv); this is what 'through the QA gate' means")
    ap.add_argument("--wide", action="store_true",
                    help=f"the A0-WIDE scope: every body, every route, gated on {WIDE['qa_gate']}")
    ap.add_argument("--transcripts", default=None, metavar="DIR",
                    help="read <episode_id>.json from this directory instead of episodes.transcript_path "
                         "(output/redacted: the name-redacted copies, DECISIONS #15); an episode with no "
                         "file there is skipped and listed")
    a = ap.parse_args(argv)
    if a.wide and a.pilot:
        sys.exit("--pilot and --wide are different scopes; pick one")
    if a.wide:
        a.qa_gate = a.qa_gate or str(ROOT / WIDE["qa_gate"])
    if a.pilot:
        a.bodies = a.bodies or list(PILOT["bodies"])
        a.classified_before = a.classified_before or PILOT["classified_before"]
        a.batches = a.batches or list(PILOT["batches"])

    import load_db  # noqa: E402  (collapse_loops: same rule as the side corpus)
    state = state_db_path(a.state)
    if not state.exists():
        sys.exit(f"no state database at {state}")
    db_path = Path(a.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Build on local scratch and copy the finished file back, as load_db.py does
    # (the project folder may be a mount that cannot host SQLite locking). The
    # side-corpus tables already in the file are carried over untouched.
    tmpdir = tempfile.mkdtemp(prefix="corpusdb-")
    work = Path(tmpdir) / db_path.name
    if db_path.exists():
        shutil.copy(db_path, work)
    qa = select_qa_gate(a.qa_gate) if a.qa_gate else None
    if a.wide and not a.bodies:
        # The exposure denominator must be the same bodies as every other wide
        # figure. Without this the meetings table pulls in all 33 discovered
        # bodies (622 meetings) and the rate panel compares 10 bodies that have
        # transcripts against 20 that have none. Derived, never hardcoded.
        _st = sqlite3.connect(f"file:{state_db_path(a.state)}?mode=ro", uri=True)
        a.bodies = sorted({r[0] for r in _st.execute(
            "SELECT DISTINCT v.speaker FROM episodes e JOIN videos v ON v.video_id = e.video_id "
            "WHERE e.status = 'done'") if r[0]} & {r[0] for r in _st.execute(
            "SELECT DISTINCT v.speaker FROM episodes e JOIN videos v ON v.video_id = e.video_id "
            "WHERE e.episode_id IN (%s)" % ",".join("?" * len(qa)), tuple(qa))})
        _st.close()
    con, n_ep, n_seg, n_words, sources, n_meet = load(
        state, work, load_db.collapse_loops,
        bodies=a.bodies, classified_before=a.classified_before, batches=a.batches, qa_gate=qa,
        transcripts=a.transcripts)
    scope = ("T1 pilot" if a.pilot else "A0-WIDE" if a.wide else
             "selection" if (a.bodies or a.classified_before or a.batches or a.qa_gate) else "everything done")
    print(f"scope: {scope}"
          + (f"; bodies {','.join(a.bodies)}" if a.bodies else "")
          + (f"; classified before {a.classified_before}" if a.classified_before else "")
          + (f"; batches {','.join(a.batches)}" if a.batches else "")
          + (f"; QA gate {a.qa_gate} ({len(qa)} episodes OK/WARN)" if qa is not None else ""))
    print(f"loaded {n_ep} episodes, {n_seg} segments, {n_words:,} words from {n_meet} meetings -> {db_path}")
    print(f"  roles: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    for body, fam, ne, ns, w in con.execute(
            "SELECT body, family, COUNT(*), SUM(n_speakers), SUM(words) FROM episodes GROUP BY 1, 2 ORDER BY 1, 2"):
        print(f"  {body:<10} {fam:<8} {ne:>3} episodes {ns:>4} clusters {w:>8,} words")
    for role, n, t, w in con.execute(
            "SELECT role, COUNT(*), ROUND(SUM(talk_s)), SUM(words) FROM episode_speakers GROUP BY 1 ORDER BY 1"):
        print(f"  role {role:<8} {n:>4} clusters {t:>8.0f} s {w:>8,} words")
    print("  facets (method x role-source), episodes / words:")
    for meth, src, ne, w in con.execute(
            "SELECT method, roles_source, COUNT(*), SUM(words) FROM episodes GROUP BY 1, 2 ORDER BY 1, 2"):
        print(f"    {meth:<14} {src:<9} {ne:>3} episodes {w:>8,} words")
    con.close()
    shutil.copy(work, db_path)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
