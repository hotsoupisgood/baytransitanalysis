#!/usr/bin/env python3
"""
load_db.py — build the searchable corpus index from transcript JSON files.

WHY: the JSON files are the source of truth (SCHEMA.md), but they are a terrible
thing to query. This builds a derived SQLite index with FTS5 so that finding
every mention of a phrase is a ranked full-text query instead of a regex scan
over every file. The database is disposable by design: delete it and re-run.
"""
import argparse, json, shutil, sqlite3, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# NOTE: no WAL here. This derived index may live on a network/FUSE-mounted
# folder, where WAL needs shared memory the mount doesn't provide and SQLite
# fails with "disk I/O error". state.sqlite (SCHEMA.md) DOES want WAL and must
# therefore live on real local disk, never on a mounted or synced folder.
DDL = """
PRAGMA journal_mode=DELETE;
DROP TABLE IF EXISTS segments;
DROP TABLE IF EXISTS videos;
CREATE TABLE videos (
  video_id TEXT PRIMARY KEY, speaker TEXT, month TEXT,
  duration_s REAL, transcribed_s REAL, diarized INT,
  host_speaker TEXT, host_confidence REAL, model TEXT,
  loop_segments_dropped INT,  -- identical consecutive segments collapsed on load (collapse_loops)
  host_method TEXT            -- "embedding" (scripts/host_enroll.py; host_confidence is a cosine)
                              -- or "talk_time" (host_confidence is the host's share of speech)
);
CREATE TABLE segments (
  video_id TEXT NOT NULL, seg_idx INTEGER NOT NULL,
  start_s REAL, end_s REAL, speaker TEXT,
  is_host INTEGER,          -- the column analysis should filter on
  text TEXT NOT NULL,
  PRIMARY KEY (video_id, seg_idx)
);
CREATE INDEX idx_seg_video ON segments(video_id);
DROP TABLE IF EXISTS segments_fts;
CREATE VIRTUAL TABLE segments_fts USING fts5(
  text, video_id UNINDEXED, seg_idx UNINDEXED, tokenize='porter unicode61'
);
"""


def resolve_host(doc):
    """Map anonymous pyannote clusters to host/not-host.

    WHY A HEURISTIC AND NOT A LABEL: pyannote returns SPEAKER_00/01/... which are
    cluster ids, not identities, and the numbering is not stable across videos.
    The host is almost always the speaker with the most total speech time. We
    record the confidence (host share of speech) so low-confidence videos --
    the interview-heavy ones where this breaks -- can be flagged rather than
    silently trusted.
    """
    if doc.get("host_speaker"):
        return doc["host_speaker"], doc.get("host_confidence")
    talk = {}
    for s in doc["segments"]:
        talk[s.get("speaker")] = talk.get(s.get("speaker"), 0.0) + (s["end"] - s["start"])
    labelled = {k: v for k, v in talk.items() if k is not None}
    if not labelled:
        return None, None
    # The host is chosen among LABELLED clusters only. Before 2026-09-17 the
    # unlabelled bucket (segments WhisperX could not assign) competed too; when
    # it was largest, "host" became None and every segment of the video --
    # guests included -- was kept as if undiarized. The share still divides by
    # ALL speech, so unassigned time lowers the confidence instead of hiding.
    total = sum(talk.values())
    host = max(labelled, key=labelled.get)
    return host, round(labelled[host] / total, 3) if total else None


def host_labels(doc, host=None):
    """The set of cluster labels that ARE the host for this video.

    Since 2026-09-20 scripts/host_enroll.py writes `host_speakers`: every
    cluster whose voice embedding matches the channel's enrolled host (Hasan's
    impressions split him into several labels; all of them are him). Older or
    ambiguous files carry only `host_speaker`, so fall back to that single
    label (`host` = the resolve_host() answer, for files with neither).
    """
    hs = doc.get("host_speakers")
    if hs:
        return {h for h in hs if h is not None}
    single = doc.get("host_speaker") or host
    return {single} if single else set()


def collapse_loops(segments):
    """Collapse runs of consecutive segments with identical text (and the same
    speaker label) to the first one: its start is kept, its end is extended to
    the run's end. Returns (segments, n_dropped).

    WHY: a Whisper decoder loop emits the same line N times back to back; to
    every statistic downstream that is N genuine utterances of the words. The
    QA gate (analysis/qa_transcripts.py) only FAILs a run that spans more than
    DUP_FAIL_S of audio; shorter runs reach the loader and are neutralised
    here, so a five-line chant counts its words once. Same speaker only: a
    real "No." / "No." exchange between host and guest is two segments and
    stays two. Text is compared case-insensitively after strip, as in QA.
    """
    out, dropped = [], 0
    for s in segments:
        prev = out[-1] if out else None
        if (prev is not None and prev.get("speaker") == s.get("speaker")
                and str(prev["text"]).strip().lower() == str(s["text"]).strip().lower()):
            prev["end"] = max(prev["end"], s["end"])
            dropped += 1
            continue
        out.append(dict(s))
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="output_sim/whisper")
    ap.add_argument("--db",  default="output_sim/corpus.sqlite")
    a = ap.parse_args()

    src = ROOT / a.src
    db_path = ROOT / a.db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # WHY BUILD ELSEWHERE THEN COPY: the project folder may be a network/FUSE
    # mount (it is, when Claude reaches it remotely). SQLite needs POSIX byte-
    # range locking that such mounts often don't provide, and fails with a bare
    # "disk I/O error". Building on real local disk and copying the finished
    # single file across is reliable everywhere and costs nothing -- the file is
    # small and written once. Same reason state.sqlite must live on local disk.
    tmpdir = tempfile.mkdtemp(prefix="corpusdb-")
    work = Path(tmpdir) / db_path.name
    con = sqlite3.connect(work)
    con.executescript(DDL)

    nv = ns = nd = 0
    for f in sorted(src.glob("*.json")):
        doc = json.loads(f.read_text())
        host, conf = resolve_host(doc)
        hosts = host_labels(doc, host)
        segs, dropped = collapse_loops(doc["segments"])
        if dropped:
            print(f"  {doc['video_id']}: collapsed {dropped} looped segment(s) (identical consecutive text)")
            nd += dropped
        con.execute(
            "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (doc["video_id"], doc.get("speaker") or doc.get("channel"),
             doc.get("month"), doc.get("duration_s"), doc.get("transcribed_s"),
             int(bool(doc.get("diarized"))), host, conf, doc.get("model"), dropped,
             doc.get("host_method") or ("talk_time" if host else None)))
        rows = []
        for i, s in enumerate(segs):
            is_host = None if not hosts else int(s.get("speaker") in hosts)
            rows.append((doc["video_id"], i, s["start"], s["end"],
                         s.get("speaker"), is_host, s["text"]))
        con.executemany("INSERT OR REPLACE INTO segments VALUES (?,?,?,?,?,?,?)", rows)
        con.executemany("INSERT INTO segments_fts (text, video_id, seg_idx) VALUES (?,?,?)",
                        [(r[6], r[0], r[1]) for r in rows])
        nv += 1; ns += len(rows)
    con.commit()
    con.execute("PRAGMA optimize")

    print(f"loaded {nv} videos, {ns} segments -> {db_path}"
          + (f" ({nd} looped segments collapsed)" if nd else ""))
    # host_confidence is a share of speech under talk_time and a cosine under
    # embedding, so the mean is reported per method rather than mixed.
    for sp, m, nv_, n, h in con.execute(
            "SELECT v.speaker, v.host_method, COUNT(DISTINCT v.video_id), COUNT(*), ROUND(AVG(v.host_confidence),2) "
            "FROM segments s JOIN videos v USING(video_id) GROUP BY 1, 2 ORDER BY 1, 2"):
        print(f"  {sp:<12} {m or '-':<10} {nv_:>4} videos {n:>7} segments   mean host_confidence {h}")
    con.close()
    shutil.copy(work, db_path)
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
