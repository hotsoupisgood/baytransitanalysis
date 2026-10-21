#!/usr/bin/env python3
"""
pipeline/state.py -- the one place that opens state.sqlite and moves a video
between states. Every worker imports this; nothing else writes the DB.

Schema: plan/SCHEMA.md 1 (videos, spend, pods, batches, batch_videos, events).
State machine: plan/PLAN.md 3.

    discovered -> downloading -> downloaded -> transcoded -> queued
               -> transcribing -> done
                     | (any stage)
                   failed -> (requeue resets to the prior state)
                   dead   -> attempts >= MAX_ATTEMPTS; never retried automatically

WHY a conditional UPDATE for every claim: two workers (or one worker and a
human) can never hold the same row. rowcount 0 means someone else has it; move
on. WHY an events row per transition: videos.updated_at only remembers the last
change; the dashboard's throughput, ETA and stall detection read events.

Meeting corpus (2026-09-21, PROPOSAL-transit.md 5, T1): a `videos` row now has a
`kind` -- 'show' (the Hasan/Tucker frame, unchanged), 'meeting' (one council or
board meeting: a YouTube /streams VOD or a Granicus clip, in_frame = 0, never
touched by the launchd workers) or 'episode' (a cut clip of one meeting, the
unit the GPU transcribes, registered in-frame as 'transcoded' exactly like the
T0 test-5 clips). Meeting rows walk their own states:

    discovered -> captioned -> classified -> downloading -> downloaded -> cut
                            \\-> no_episodes              (nothing about density/transit)

plus `sources`, `episodes`, `speaker_roles` and `classifier_calls` (SCHEMA.md 1).
Columns and tables are only ever ADDED; `migrate` copies the DB to
~/.local/share/ytpt/backups/ first.

    python3 pipeline/state.py init        # create the DB (idempotent)
    python3 pipeline/state.py counts      # status counts, in-frame only
    python3 pipeline/state.py migrate     # backup, then add the 2026-09-21 columns/tables
"""
import os, re, shutil, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 3
STATES = ("discovered", "downloading", "downloaded", "transcoded", "queued",
          "transcribing", "done", "failed", "dead",
          # kind = 'meeting' only (2026-09-21):
          "captioned", "classified", "no_episodes", "cut")
KINDS = ("show", "meeting", "episode")
EPISODE_STATES = ("located", "mention", "cut", "queued", "transcribing", "done", "failed", "capped")

# Columns added after the 2026-09-18 DDL; connect() adds any that are missing
# (ALTER TABLE ADD COLUMN is the only schema change SQLite can do in place).
VIDEO_COLUMNS_ADDED = [
    ("kind", "TEXT NOT NULL DEFAULT 'show'"),      # show | meeting | episode
    ("platform", "TEXT NOT NULL DEFAULT 'youtube'"),  # youtube | granicus
    ("source_url", "TEXT"),                        # watch URL / Granicus player URL
    ("meeting_id", "TEXT"),                        # '<body>:<YYYY-MM-DD>'; episodes carry their meeting's
    ("meeting_date", "TEXT"),                      # 'YYYY-MM-DD' from the title, not the upload date
    ("captions", "TEXT"),                          # auto | manual | granicus | none
    ("captions_path", "TEXT"),
    ("episode_id", "TEXT"),                        # kind = 'episode' rows only
]
# Same rule for the other tables (T2, 2026-09-22).
COLUMNS_ADDED = {
    "videos": VIDEO_COLUMNS_ADDED,
    "classifier_calls": [("batch_id", "TEXT"),    # Message Batches API batch that carries this call (NULL = direct)
                         ("keywords_version", "TEXT")],   # stage-1 keyword list that produced the window (2026-10-18; NULL = before versioning)
    "sources": [("vocab", "TEXT")],               # WhisperX initial_prompt vocabulary for this body (<= 200 chars)
    "episodes": [("keywords_version", "TEXT")],   # keyword list(s) whose windows located the episode (2026-10-18)
}

DDL = """
CREATE TABLE IF NOT EXISTS videos (
  video_id      TEXT PRIMARY KEY,
  speaker       TEXT NOT NULL,
  channel_url   TEXT NOT NULL,
  month         TEXT NOT NULL,
  title         TEXT,
  upload_date   TEXT,
  date_exact    INTEGER DEFAULT 0,
  duration_s    REAL,
  view_count    INTEGER,
  like_count    INTEGER,
  raw_meta      TEXT,
  crawled_at    TEXT NOT NULL,
  in_frame      INTEGER DEFAULT 0,
  status        TEXT NOT NULL,
  audio_path    TEXT,
  opus_path     TEXT,
  transcribed_s REAL,
  attempts      INTEGER DEFAULT 0,
  last_error    TEXT,
  updated_at    TEXT NOT NULL,
  sponsor_segments TEXT,            -- SponsorBlock [[start,end],...] JSON, fetched at download (PLAN.md 4.2)
  kind          TEXT NOT NULL DEFAULT 'show',
  platform      TEXT NOT NULL DEFAULT 'youtube',
  source_url    TEXT,
  meeting_id    TEXT,
  meeting_date  TEXT,
  captions      TEXT,
  captions_path TEXT,
  episode_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_status  ON videos(status);
CREATE INDEX IF NOT EXISTS idx_frame   ON videos(in_frame, status);
CREATE INDEX IF NOT EXISTS idx_stratum ON videos(speaker, month);
-- 2026-09-21, PROPOSAL-transit.md 5. A body may live on several platforms over
-- time; channels.tsv is the human input, this is what discovery resolved it to.
CREATE TABLE IF NOT EXISTS sources (
  source_id   TEXT PRIMARY KEY,     -- 'sanjose:youtube:streams', 'oakland:granicus:2'
  body        TEXT NOT NULL,        -- slug; videos.speaker for meeting rows
  body_kind   TEXT NOT NULL,        -- council | county | transit_board | commission
  platform    TEXT NOT NULL,        -- youtube | granicus
  platform_id TEXT NOT NULL,        -- channel id + tab, or site + view_id
  url         TEXT NOT NULL,
  regular_re  TEXT,                 -- title regex that marks a regular meeting
  region      TEXT NOT NULL DEFAULT 'bayarea',
  active      INTEGER DEFAULT 1,
  note        TEXT,
  updated_at  TEXT NOT NULL,
  vocab       TEXT);                -- WhisperX initial_prompt vocabulary for the body (T2, pipeline/vocab.py)
-- One row per located span. The same span can be found by keyword density, the
-- classifier and an agenda item; n is reported per method / classifier_version.
CREATE TABLE IF NOT EXISTS episodes (
  episode_id   TEXT PRIMARY KEY,    -- '<video_id>_e<n>' (= the clip's videos.video_id once cut)
  meeting_id   TEXT NOT NULL,
  video_id     TEXT NOT NULL,       -- the meeting row
  start_s      REAL NOT NULL, end_s REAL NOT NULL,   -- padded, cue-snapped bounds actually cut
  core_start_s REAL, core_end_s REAL,               -- unpadded window bounds, for durations
  tags         TEXT NOT NULL,       -- density | transit | both
  method       TEXT NOT NULL,       -- classifier | agenda_index | keyword | manual
  score        REAL,                -- keyword hits per 100 words
  label        TEXT,                -- about_density | about_transit | about_both | mention_only
  classifier_model TEXT, prompt_version TEXT,
  agenda_item  TEXT,                -- Granicus index-point title overlapping the span, if any
  status       TEXT NOT NULL,       -- located -> cut -> queued -> transcribing -> done | mention | failed | capped
  audio_path TEXT, opus_path TEXT, transcript_path TEXT,
  attempts INTEGER DEFAULT 0, last_error TEXT, updated_at TEXT NOT NULL,
  keywords_version TEXT);           -- segment.load_keywords().version of the list(s) that located it (2026-10-18)
CREATE INDEX IF NOT EXISTS idx_ep_status  ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_ep_meeting ON episodes(meeting_id);
-- Roles are rows, not a key in the transcript: several methods, revisable, and
-- labels stay anonymous clusters (PROPOSAL 6, appendix 15).
CREATE TABLE IF NOT EXISTS speaker_roles (
  episode_id  TEXT NOT NULL,
  speaker     TEXT NOT NULL,        -- SPEAKER_00 ...
  role        TEXT NOT NULL,        -- dais | public | unknown
  confidence  REAL,
  method      TEXT NOT NULL,        -- cue_phrase | structure | recurrence | prior | none
  talk_s      REAL,
  cues_version TEXT,
  PRIMARY KEY (episode_id, speaker));
-- Every classifier call, so $ and tokens per meeting are a SELECT and a
-- re-run never re-bills a window it already has (SCHEMA.md: no summary tables).
-- Key = (id, prompt_version) since 2026-09-22 (review L2): a re-run under a
-- new prompt version or model adds rows, it never overwrites the old
-- measurement or its cost. prompt_version embeds the model id for every
-- model but the T0 instrument (classify.prompt_version_for).
CREATE TABLE IF NOT EXISTS classifier_calls (
  id            TEXT NOT NULL,      -- '<video_id>_c<nnn>'
  video_id      TEXT NOT NULL,
  start_s REAL NOT NULL, end_s REAL NOT NULL, pad_start_s REAL, pad_end_s REAL,
  words INTEGER, hits_per_100w REAL, family_rates TEXT, top_keywords TEXT,
  label TEXT, rationale TEXT, contains_instructions INTEGER,
  model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  input_tokens INTEGER, cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, latency_s REAL, at TEXT NOT NULL,
  batch_id TEXT,                    -- Message Batches API id when the call went through a batch
  keywords_version TEXT,            -- stage-1 keyword list version that produced this window (2026-10-18)
  PRIMARY KEY (id, prompt_version));
CREATE INDEX IF NOT EXISTS idx_cc_video ON classifier_calls(video_id);
-- One row per Message Batches API submission (T2, 2026-09-22): the calls it
-- carries are classifier_calls rows with label NULL and this batch_id until
-- `classify.py collect` writes the results back.
CREATE TABLE IF NOT EXISTS classifier_batches (
  batch_id     TEXT PRIMARY KEY,    -- Anthropic msgbatch_... id
  created_at   TEXT NOT NULL,
  status       TEXT NOT NULL,       -- submitted | ended | collected | failed
  model        TEXT NOT NULL, prompt_version TEXT NOT NULL,
  n_requests   INTEGER, n_succeeded INTEGER, n_errored INTEGER,
  est_usd REAL, cost_usd REAL,
  collected_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS spend (
  id INTEGER PRIMARY KEY, pod_id TEXT, started_at TEXT NOT NULL, ended_at TEXT,
  gpu_hours REAL, usd REAL, n_videos INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS pods (
  pod_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, terminated_at TEXT, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
  n_videos INTEGER, audio_hours REAL, est_usd REAL, approval_requested_at TEXT,
  approved_at TEXT, approved_via TEXT, pod_id TEXT, started_at TEXT,
  finished_at TEXT, gpu_minutes REAL, actual_usd REAL, note TEXT);
CREATE TABLE IF NOT EXISTS batch_videos (
  batch_id TEXT NOT NULL, video_id TEXT NOT NULL, PRIMARY KEY (batch_id, video_id));
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, from_status TEXT,
  to_status TEXT NOT NULL, at TEXT NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
-- 2026-09-22 (T2): a worker that stopped itself on a YouTube bot check / 429
-- writes WHEN it may ask again here, so a restart -- launchd, a human or the
-- next loop -- respects the backoff instead of walking straight back into the
-- challenge. One row per worker; cleared by hand or by the first success.
CREATE TABLE IF NOT EXISTS backoffs (
  worker TEXT PRIMARY KEY,          -- 'captions', 'downloader', ...
  until  TEXT NOT NULL,             -- ISO8601 UTC; no requests before this
  reason TEXT, set_at TEXT NOT NULL);
-- 2026-09-24: one row per request a worker actually made to a rate-limiting
-- host, so a daily cap and a minimum gap survive a restart. The 2026-09-23
-- caption run was 429'd after 5 requests in 41 min; the gap was only ever
-- enforced inside one process, so launchd's KeepAlive could restart the worker
-- and it would ask again immediately. Append-only; nothing reads it but the
-- rate limiter and the dashboard.
CREATE TABLE IF NOT EXISTS worker_requests (
  id     INTEGER PRIMARY KEY,
  worker TEXT NOT NULL,             -- 'captions', ...
  host   TEXT NOT NULL,             -- 'youtube' | 'granicus' -- the limit is per host
  at     TEXT NOT NULL,             -- ISO8601 UTC
  video_id TEXT);
CREATE INDEX IF NOT EXISTS idx_wr_worker_at ON worker_requests(worker, host, at);
"""


def load_env(path=ROOT / ".env"):
    """KEY=value lines; inline comments stripped; never overrides the real env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), re.sub(r"\s+#.*$", "", v.strip()).strip('"').strip("'"))


def setting(key, default):
    load_env()
    return os.environ.get(key) or default


def db_path():
    return Path(os.path.expanduser(setting("STATE_DB_PATH", "~/.local/share/ytpt/state.sqlite")))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=None, create=True):
    """Open the state DB in WAL mode and make sure the schema exists."""
    path = Path(path) if path else db_path()
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(f"no state DB at {path}")
    con = sqlite3.connect(path, timeout=30, isolation_level=None)   # autocommit; explicit BEGIN where needed
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(DDL)
    add_missing_columns(con)
    return con


def missing_columns(con, table="videos"):
    have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    return [(c, t) for c, t in COLUMNS_ADDED.get(table, []) if c not in have] if have else []


def add_missing_columns(con):
    """ALTER TABLE ADD COLUMN for every column in COLUMNS_ADDED that an older
    DB lacks. Add-only and idempotent; every old row keeps its values and gets
    the column default (kind='show', platform='youtube', batch_id NULL)."""
    added = []
    for table in COLUMNS_ADDED:
        for col, typ in missing_columns(con, table):
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            added.append(col if table == "videos" else f"{table}.{col}")
    return added


def classifier_calls_key(con):
    """The primary-key columns of classifier_calls, in order."""
    return [r[1] for r in sorted((r for r in con.execute("PRAGMA table_info(classifier_calls)") if r[5]), key=lambda r: r[5])]


def rebuild_classifier_calls_key(con):
    """(id) -> (id, prompt_version) as the key (2026-09-22, review L2). SQLite
    cannot change a primary key in place, so the table is rebuilt: create the
    new shape, copy every row, drop the old, rename. Every row and every
    dollar survives; only `migrate` calls this, after its backup. Returns True
    when a rebuild happened."""
    if classifier_calls_key(con) != ["id"]:
        return False
    cols = [r[1] for r in con.execute("PRAGMA table_info(classifier_calls)")]
    ddl = DDL[DDL.index("CREATE TABLE IF NOT EXISTS classifier_calls"):]
    ddl = ddl[:ddl.index(");") + 2].replace("classifier_calls", "classifier_calls__new", 1)
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(ddl)
        con.execute(f"INSERT INTO classifier_calls__new ({', '.join(cols)}) SELECT {', '.join(cols)} FROM classifier_calls")
        con.execute("DROP TABLE classifier_calls")
        con.execute("ALTER TABLE classifier_calls__new RENAME TO classifier_calls")
        con.execute("CREATE INDEX IF NOT EXISTS idx_cc_video ON classifier_calls(video_id)")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return True


def backup_db(path=None, dest_dir=None):
    """Consistent copy of the state DB (SQLite online backup, so a WAL with
    un-checkpointed pages is included) to backups/state-<date>[-n].sqlite."""
    path = Path(path) if path else db_path()
    dest_dir = Path(dest_dir) if dest_dir else path.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    dest, n = dest_dir / f"state-{stamp}.sqlite", 1
    while dest.exists():
        n += 1
        dest = dest_dir / f"state-{stamp}-{n}.sqlite"
    src = sqlite3.connect(path)
    dst = sqlite3.connect(dest)
    src.backup(dst)
    dst.close(); src.close()
    return dest


def migrate(path=None, backup=True):
    """The explicit migration: backup, add columns/tables, backfill `kind` for
    the T0 test-5 clip rows (registered 2026-09-21 with raw_meta.t0 set).
    Returns a dict describing what changed."""
    path = Path(path) if path else db_path()
    out = {"db": str(path), "backup": None, "columns_added": [], "tables_added": [], "episode_rows_backfilled": 0}
    if backup and path.exists():
        out["backup"] = str(backup_db(path))
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    before = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.execute("PRAGMA journal_mode=WAL")
    out["columns_added"] = add_missing_columns(con) if "videos" in before else []
    con.executescript(DDL)
    out["columns_added"] += add_missing_columns(con)          # tables that pre-dated a column added later
    if "videos" not in before:
        out["columns_added"] = []
    after = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out["tables_added"] = sorted(after - before)
    out["classifier_calls_rekeyed"] = rebuild_classifier_calls_key(con) if "classifier_calls" in before else False
    # T0 test-5 clips: '<yt_id>_e<n>' rows with raw_meta.t0 -- these are episodes
    # of a meeting, not shows; their meeting/episode ids come from raw_meta.
    rows = con.execute("SELECT video_id, raw_meta, speaker FROM videos WHERE kind = 'show' "
                       "AND video_id LIKE '%\\_e%' ESCAPE '\\' AND raw_meta LIKE '%\"t0\": \"test 5 episode clip\"%'").fetchall()
    import json
    for r in rows:
        try:
            meta = json.loads(r["raw_meta"] or "{}")
        except ValueError:
            meta = {}
        con.execute("UPDATE videos SET kind = 'episode', episode_id = ? WHERE video_id = ?",
                    (meta.get("episode_id") or r["video_id"], r["video_id"]))
        out["episode_rows_backfilled"] += 1          # meeting_id is filled when the meeting row is discovered
    con.close()
    return out


def set_episode(con, episode_id, status, note=None, **fields):
    """Move an episodes row (no claim semantics: one process cuts at a time).
    Writes an events row keyed on the episode id so the audit trail is complete."""
    if status not in EPISODE_STATES:
        raise ValueError(f"unknown episode state {status!r}")
    sets = ", ".join(["status = ?", "updated_at = ?"] + [f"{k} = ?" for k in fields])
    con.execute("BEGIN IMMEDIATE")
    try:
        old = con.execute("SELECT status FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        cur = con.execute(f"UPDATE episodes SET {sets} WHERE episode_id = ?", [status, now_iso(), *fields.values(), episode_id])
        if cur.rowcount != 1:
            con.execute("ROLLBACK")
            return False
        con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                    (episode_id, old["status"] if old else None, status, now_iso(), note))
        con.execute("COMMIT")
        return True
    except Exception:
        con.execute("ROLLBACK")
        raise


def transition(con, video_id, from_status, to_status, note=None, **fields):
    """Move one row from from_status to to_status atomically; write the event.
    Returns True if this call made the change, False if the row was not in
    from_status (someone else claimed it, or it moved on)."""
    if to_status not in STATES:
        raise ValueError(f"unknown state {to_status!r}")
    sets = ", ".join(["status = ?", "updated_at = ?"] + [f"{k} = ?" for k in fields])
    args = [to_status, now_iso(), *fields.values(), video_id]
    con.execute("BEGIN IMMEDIATE")
    try:
        if from_status is None:
            cur = con.execute(f"UPDATE videos SET {sets} WHERE video_id = ?", args)
        else:
            cur = con.execute(f"UPDATE videos SET {sets} WHERE video_id = ? AND status = ?",
                              args + [from_status])
        if cur.rowcount != 1:
            con.execute("ROLLBACK")
            return False
        con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                    (video_id, from_status, to_status, now_iso(), note))
        con.execute("COMMIT")
        return True
    except Exception:
        con.execute("ROLLBACK")
        raise


def claim(con, video_id, from_status, to_status, count_attempt=True):
    """Claim a row for work: status change plus attempts += 1 in one statement."""
    con.execute("BEGIN IMMEDIATE")
    try:
        cur = con.execute(
            "UPDATE videos SET status = ?, updated_at = ?, "
            f"attempts = attempts + {1 if count_attempt else 0} "
            "WHERE video_id = ? AND status = ?",
            (to_status, now_iso(), video_id, from_status))
        if cur.rowcount != 1:
            con.execute("ROLLBACK")
            return False
        con.execute("INSERT INTO events (video_id, from_status, to_status, at) VALUES (?,?,?,?)",
                    (video_id, from_status, to_status, now_iso()))
        con.execute("COMMIT")
        return True
    except Exception:
        con.execute("ROLLBACK")
        raise


def fail(con, video_id, from_status, error, max_attempts=MAX_ATTEMPTS):
    """Record a failure. After max_attempts the row is dead and needs a human
    (PLAN.md 3). Returns the new status."""
    row = con.execute("SELECT attempts FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    attempts = int(row["attempts"] or 0) if row else 0
    to = "dead" if attempts >= max_attempts else "failed"
    err = str(error).strip()[:1000]
    transition(con, video_id, from_status, to, note=err[:200], last_error=err)
    return to


def reclaim_stuck(con, status, back_to, note="reclaimed at worker start", kind=None, in_frame=None, speakers=None):
    """Rows left in an in-progress state by a killed worker go back to the
    state they were claimed from (crash safety, TESTING.md level 5). Only call
    this when no other worker of that kind is running; attempts are kept, so a
    row that keeps killing the worker still dies after MAX_ATTEMPTS.

    `kind` / `in_frame` restrict the sweep to the rows THIS worker owns
    (2026-09-21 review): the launchd downloader and episodes.py both use
    'downloading', so an unfiltered sweep by one moved the other's live row
    into a state it can never leave ('classified' for a show, 'discovered'
    for a meeting mid-fetch). `speakers` (2026-10-16) narrows it to the bodies
    this run was given: two episodes.py runs on disjoint --bodies lists may
    overlap, and the second must not reclaim the first's live meeting."""
    n = 0
    where, params = "status = ?", [status]
    if kind is not None:
        where += " AND kind = ?"; params.append(kind)
    if in_frame is not None:
        where += " AND in_frame = ?"; params.append(int(in_frame))
    if speakers:
        where += f" AND speaker IN ({','.join('?' * len(speakers))})"; params += list(speakers)
    for r in con.execute(f"SELECT video_id FROM videos WHERE {where}", params).fetchall():
        if transition(con, r["video_id"], status, back_to, note=note):
            n += 1
    return n


def counts(con, in_frame_only=True):
    where = "WHERE in_frame = 1" if in_frame_only else ""
    return {r["status"]: r["n"] for r in con.execute(
        f"SELECT status, COUNT(*) AS n FROM videos {where} GROUP BY status ORDER BY status")}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "counts"
    if cmd == "migrate":                    # before connect(): the backup must precede any schema change
        r = migrate()
        print(f"migrated {r['db']}: backup {r['backup']}; columns added {r['columns_added'] or 'none'}; "
              f"tables added {r['tables_added'] or 'none'}; T0 clip rows -> kind=episode: {r['episode_rows_backfilled']}; "
              f"classifier_calls re-keyed (id, prompt_version): {r['classifier_calls_rekeyed']}")
        return 0
    con = connect()
    if cmd == "init":
        print(f"state DB ready at {db_path()}")
    elif cmd == "counts":
        c = counts(con)
        print(f"{db_path()}  in-frame videos by status: " + (", ".join(f"{k}={v}" for k, v in c.items()) or "none"))
        m = {r["status"]: r["n"] for r in con.execute(
            "SELECT status, COUNT(*) AS n FROM videos WHERE kind = 'meeting' GROUP BY status ORDER BY status")}
        if m:
            print("  meetings by status: " + ", ".join(f"{k}={v}" for k, v in m.items()))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
