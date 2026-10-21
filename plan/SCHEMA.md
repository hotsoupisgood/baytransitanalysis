# Database design

Two databases, different jobs. Neither is the source of truth.

```
  per-video JSON files  ──►  corpus.duckdb   (analysis: aggregation, stats, R)
   (SOURCE OF TRUTH)     └─►  corpus.sqlite  (search: FTS5 full-text)

  state.sqlite  (pipeline state only — NOT transcripts, NOT rebuildable)
```

**Why files remain the source of truth.** A schema mistake should cost a reload,
not the corpus. The analysis DBs are derived indexes: `rm corpus.duckdb && make
load` must always be safe and must always reproduce. If you ever find yourself
with data that exists *only* in a database, that's a bug.

`state.sqlite` is the exception — it *is* authoritative, because pipeline state
has no file to be derived from. It holds no transcript text.

---

## 1. `state.sqlite` — pipeline state

Why a database and not files: two workers running concurrently, retries,
resumability after `kill -9`, and one bug class already realised in this repo
(`output/meanness/summary.json` holds 1 of 10 videos because a re-run rewrote
the whole file). A keyed table makes that failure structurally impossible.

```sql
PRAGMA journal_mode=WAL;   -- concurrent readers + one writer; without this the
                           -- downloader and transcoder block each other

CREATE TABLE videos (
  video_id     TEXT PRIMARY KEY,  -- YouTube ID: stable, global, natural key
  speaker      TEXT NOT NULL,     -- 'hasanabi' etc. Set at discovery, never
                                  -- inferred later; speakers.json went stale
                                  -- exactly that way
  channel_url  TEXT NOT NULL,     -- which channel it came from. A speaker may
                                  -- have several; needed to interpret "Hasan"
  month        TEXT NOT NULL,     -- 'YYYY-MM'. The stratum. Denormalised from
                                  -- upload_date so cell queries stay trivial
  title        TEXT,
  upload_date  TEXT,              -- may be approximate (see SAMPLING.md)
  date_exact   INTEGER DEFAULT 0, -- 0 = from flat-playlist approximation.
                                  -- Flags videos near a month boundary that
                                  -- might sit in the wrong cell
  duration_s   REAL,              -- full video length
  view_count   INTEGER,           -- captured now because it's free now and
  like_count   INTEGER,           -- impossible later; needed for any weighting
  raw_meta     TEXT,              -- full yt-dlp JSON. Cheap insurance against
                                  -- wanting a field we didn't think of. On a
                                  -- kind='episode' clip it is the episode's own
                                  -- dict (episodes.py register_clip), and since
                                  -- 2026-09-23 it also carries `audio_levels`
                                  -- (below). NO COLUMN WAS ADDED for that
  crawled_at   TEXT NOT NULL,     -- catalogs change; videos get deleted. Without
                                  -- this the corpus isn't reproducible

  in_frame     INTEGER DEFAULT 0, -- 1 = drawn into the sample. Rows not in the
                                  -- frame stay here as the sampling population
  status       TEXT NOT NULL,     -- see state machine in PLAN.md §3
  audio_path   TEXT,
  opus_path    TEXT,
  transcribed_s REAL,             -- how much was actually transcribed. Differs
                                  -- from duration_s under the 45-min cap; the
                                  -- truncation must be visible downstream
  attempts     INTEGER DEFAULT 0,
  last_error   TEXT,
  updated_at   TEXT NOT NULL
);
CREATE INDEX idx_status   ON videos(status);       -- workers poll on this
CREATE INDEX idx_frame    ON videos(in_frame, status);
CREATE INDEX idx_stratum  ON videos(speaker, month);  -- cell counts

-- Why: the spend guard must survive a restart. An in-memory counter resets and
-- silently stops guarding, which is worse than no guard because you trust it.
CREATE TABLE spend (
  id          INTEGER PRIMARY KEY,
  pod_id      TEXT,
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  gpu_hours   REAL,
  usd         REAL,
  n_videos    INTEGER,
  note        TEXT
);

-- Why: a pod created but not recorded is a 4090 billing unnoticed. Insert the
-- row BEFORE the create call returns, reconcile after.
CREATE TABLE pods (
  pod_id      TEXT PRIMARY KEY,
  created_at  TEXT NOT NULL,
  terminated_at TEXT,
  status      TEXT NOT NULL   -- creating | running | terminated | ORPHAN
);

-- Added 2026-09-17. Why a batch is a row and not a variable in a script: the
-- approval (DECISIONS.md #10) has to survive a crash, a restart and a different
-- process, and "who approved what, when, for how much" is the audit trail for
-- every dollar. gpu_batch.py launch refuses unless the row is 'approved', the
-- estimate keeps the month under MONTHLY_SPEND_CAP_USD, and no pod is live.
CREATE TABLE batches (
  batch_id     TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  status       TEXT NOT NULL,  -- proposed -> awaiting_approval -> approved ->
                               -- launching -> running -> done
                               --           \-> denied | expired | failed
  n_videos     INTEGER,
  audio_hours  REAL,           -- capped hours in the batch
  est_usd      REAL,           -- at the CURRENT measured realtime factor
  approval_requested_at TEXT,
  approved_at  TEXT,
  approved_via TEXT,           -- 'whatsapp' -- how the human said yes
  pod_id       TEXT,
  started_at   TEXT,
  finished_at  TEXT,
  gpu_minutes  REAL,           -- actual, for the next estimate
  actual_usd   REAL,
  note         TEXT
);
CREATE TABLE batch_videos (
  batch_id TEXT NOT NULL, video_id TEXT NOT NULL,
  PRIMARY KEY (batch_id, video_id)
);

-- Added 2026-09-17. videos.updated_at only remembers the LAST change, so it
-- cannot answer "how many hours did we download on Tuesday" or "has anything
-- finished in the last six hours". One append-only row per transition does, at
-- a few bytes each, and it is what the dashboard's throughput, ETA and stall
-- detection read.
CREATE TABLE events (
  id         INTEGER PRIMARY KEY,
  video_id   TEXT NOT NULL,
  from_status TEXT,
  to_status  TEXT NOT NULL,
  at         TEXT NOT NULL,
  note       TEXT
);
CREATE INDEX idx_events_at ON events(at);
```

### Meeting corpus additions (2026-09-21, PROPOSAL-transit.md §5, Phase T1)

Add-only. `pipeline/state.py migrate` copies the DB to
`~/.local/share/ytpt/backups/state-<date>.sqlite` first, then `ALTER TABLE
videos ADD COLUMN` for each new column and `CREATE TABLE IF NOT EXISTS` for the
new tables; `state.connect()` adds any missing column on every open, so no
worker can run against a half-migrated DB.

**Why `videos` is reused for meetings instead of a `meetings` table.** A
meeting here is one YouTube `/streams` VOD or one Granicus clip — one media
object with one id, exactly what a `videos` row is — and every worker, the
dashboard, `requeue.py` and the batch gate key on `videos`. A separate table
would have meant a second state machine and a second set of claim/transition
helpers for a case (one meeting, several uploads — Cupertino) that the pilot
does not contain; `meeting_id = '<body>:<YYYY-MM-DD>'` on the row already
groups such parts if it ever does. The episode clips the GPU transcribes are
*also* `videos` rows (`kind = 'episode'`), registered as in-frame
`'transcoded'` rows exactly like the T0 test-5 clips, so `gpu_batch.propose`,
`gpu_job` and `ingest_new` consume them unchanged.

```sql
-- videos: new columns (all rows that existed before get the defaults)
  kind          TEXT NOT NULL DEFAULT 'show',    -- show | meeting | episode
  platform      TEXT NOT NULL DEFAULT 'youtube', -- youtube | granicus
  source_url    TEXT,        -- watch URL or the Granicus MediaPlayer URL
  meeting_id    TEXT,        -- '<body>:<YYYY-MM-DD>'; an episode carries its meeting's
  meeting_date  TEXT,        -- 'YYYY-MM-DD' parsed from the TITLE (uploads lag a day)
  captions      TEXT,        -- auto | manual | granicus | none
  captions_path TEXT,        -- output/subs/<id>.en.vtt
  episode_id    TEXT         -- kind = 'episode' only; equals episodes.episode_id
```

Meeting rows (`kind = 'meeting'`) are inserted with `in_frame = 0` and walk
their own states — the launchd downloader and transcoder select `in_frame = 1`
rows only, so they never touch a meeting:

```
discovered -> captioned -> classified -> downloading -> downloaded -> cut
  (captions.py)  (classify.py)         (episodes.py: whole audio, then clips)
                       \-> no_episodes   (no about_* window; counted, no audio)
```

Granicus meeting ids are `gr-<site>-<clip_id>`; the whole-meeting audio for
YouTube lives in `output/audio/<id>.opus` (kept; 45 MB per meeting-hour), a
Granicus meeting has no whole audio — its episodes are audio-only HLS slices.

```sql
-- What discovery resolved a channels.tsv row to. A body may live on several
-- platforms over time (Solana Beach: Granicus to 2015, YouTube since).
CREATE TABLE sources (
  source_id   TEXT PRIMARY KEY,   -- 'sanjose:youtube:<channel_id>:streams', 'oakland:granicus:oakland:2'
  body        TEXT NOT NULL,      -- slug = videos.speaker for meeting rows
  body_kind   TEXT NOT NULL,      -- council | county | transit_board | commission
  platform    TEXT NOT NULL,      -- youtube | granicus
  platform_id TEXT NOT NULL,
  url         TEXT NOT NULL,
  regular_re  TEXT,               -- the title regex that marks a regular meeting
  region      TEXT NOT NULL DEFAULT 'bayarea',
  active      INTEGER DEFAULT 1,
  note        TEXT,
  updated_at  TEXT NOT NULL
);

-- One row per located span. The same span can be found by keyword density, the
-- classifier and an agenda item; n is reported per method / prompt_version,
-- never from what is on disk (SAMPLING.md, last paragraph).
CREATE TABLE episodes (
  episode_id   TEXT PRIMARY KEY,  -- '<video_id>_e<n>' (about_*) | '_m<n>' (mention) | '_a<n>' (agenda_index)
  meeting_id   TEXT NOT NULL,
  video_id     TEXT NOT NULL,     -- the meeting row
  start_s      REAL NOT NULL, end_s REAL NOT NULL,   -- padded, cue-snapped bounds actually cut
  core_start_s REAL, core_end_s REAL,               -- the classifier windows' bounds, for durations
  tags         TEXT NOT NULL,     -- density | transit | both
  method       TEXT NOT NULL,     -- the locate ROUTE: classifier | agenda_index | local_asr | swagit_transcript
                                  --   | fullasr (2026-10-18: every 120-s window of a whole-meeting GPU
                                  --   transcript read by codebook v2.1, no keyword screen; ids '_f<n>' /
                                  --   '_fm<n>'; scripts/fullasr.py, plan/FULLASR-GRANICUS.md) | keyword | manual
  score        REAL,              -- keyword hits per 100 words (max family rate)
  label        TEXT,              -- about_density | about_transit | about_both | mention_only
  classifier_model TEXT, prompt_version TEXT,
  agenda_item  TEXT,              -- Granicus index-point title overlapping the span most
  status       TEXT NOT NULL,     -- located -> cut -> queued -> transcribing -> done
                                  --   | mention (never downloaded) | failed | capped
  audio_path TEXT, opus_path TEXT, transcript_path TEXT,
  attempts INTEGER DEFAULT 0, last_error TEXT, updated_at TEXT NOT NULL
);

-- `audio_levels` (2026-09-23, additive, NO new column). How the clip's opus was
-- encoded, written by pipeline/transcode.encode onto the clip's own videos row
-- (kind='episode') under raw_meta.audio_levels -- that row IS the episode's row,
-- and it already carries a JSON dict, so nothing needed adding. Read it when a
-- clip fails QA:
--
--   {"af": "pan=mono|c0=c0,loudnorm=I=-16.0:TP=-1.5:LRA=11",
--    "downmix": "polarity-inverted stereo: sum -35.7 dB below channel 0",
--    "normalised": true, "threshold_dbfs": -30.0, "target_lufs": -16.0,
--    "pre_mean_dbfs": -42.1, "pre_max_dbfs": -7.2,      -- mono, before encoding
--    "post_mean_dbfs": -18.8, "post_max_dbfs": -1.6}    -- the opus on disk
--
-- `af` null and `normalised` false means the clip encoded exactly as clips did
-- before 2026-09-23. The thresholds are settings: NORMALISE_BELOW_DBFS (-30),
-- LOUDNORM_I (-16).

-- Roles are rows, not a key in the transcript: several methods, revisable,
-- anonymous clusters only (no names; appendix #15). A copy is also written
-- into the transcript JSON (`roles`, `roles_summary`, `roles_cues_version`)
-- so the file stays self-describing.
CREATE TABLE speaker_roles (
  episode_id  TEXT NOT NULL,
  speaker     TEXT NOT NULL,      -- SPEAKER_00 ...
  role        TEXT NOT NULL,      -- dais | public | unknown
  confidence  REAL,
  method      TEXT NOT NULL,      -- cue_phrase | structure | recurrence | prior | none
  talk_s      REAL,
  cues_version TEXT,              -- sha1 of plan/role_cues.txt
  PRIMARY KEY (episode_id, speaker)
);

-- Every classifier call. $ and tokens per meeting are a SELECT (no summary
-- table), and a re-run under the same model + prompt_version never re-bills a
-- window. The T0 results were imported here (classify.py import-jsonl).
CREATE TABLE classifier_calls (
  id            TEXT NOT NULL,     -- '<video_id>_c<nnn>'
  video_id      TEXT NOT NULL,
  start_s REAL NOT NULL, end_s REAL NOT NULL, pad_start_s REAL, pad_end_s REAL,
  words INTEGER, hits_per_100w REAL, family_rates TEXT, top_keywords TEXT,
  label TEXT, rationale TEXT, contains_instructions INTEGER,
  model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  input_tokens INTEGER, cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, latency_s REAL, at TEXT NOT NULL,
  batch_id TEXT,                   -- 2026-09-22: Message Batches API id while/after the call went through a batch
  PRIMARY KEY (id, prompt_version) -- 2026-09-22 (review L2): was (id); a new instrument adds rows, never overwrites
);

-- 2026-09-22 (T2): one row per Message Batches API submission. Its calls are
-- classifier_calls rows with label NULL and this batch_id until `classify.py
-- collect` writes the results back (status collected).
CREATE TABLE classifier_batches (
  batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,   -- submitted | ended | collected | failed
  model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  n_requests INTEGER, n_succeeded INTEGER, n_errored INTEGER, est_usd REAL, cost_usd REAL,
  collected_at TEXT, note TEXT
);
```

`sources.vocab TEXT` (2026-09-22): the body's WhisperX `initial_prompt`
vocabulary (`pipeline/vocab.py`, ≤ 200 chars; `plan/vocab.tsv` is the human
source, this column overrides it).

`prompt_version` (2026-09-22): the T0/T1 instrument (claude-sonnet-4-6) keeps
the bare codebook hash `t0-a4a8e55760`; every other model gets
`<hash>/<model id>` (`classify.prompt_version_for`), so the same codebook on
Haiku 4.5 is a different instrument with its own rows and cost.

**The one non-add-only change so far**: the `classifier_calls` key. SQLite
cannot alter a primary key, so `state.py migrate` (backup first, as always)
rebuilt the table — create the new shape, copy every row, drop, rename — on
2026-09-22; 316 rows / $1.1641 before and after. `connect()` never does this;
only the explicit `migrate` does.

Transcript JSON for an episode clip: unchanged shape, added top-level keys
`kind`, `episode_id`, `meeting_id`, `body`, `tags`, `label`, `agenda_item`,
`episode_start_s` (= `window_start_s`, the clip's offset in the meeting; it
used to be hardcoded 0), `roles`, `roles_summary`, `roles_cues_version`.

`dashboard/status.py` reads all of these read-only and tolerates every one of
them being absent — it has to work on day 1, before the pipeline has written a
row. `dashboard/demo_state.py` builds a synthetic database in exactly this shape.

**Claiming work safely.** Every worker takes a row with a conditional update, so
two workers can never hold the same video:

```sql
UPDATE videos SET status='downloading', attempts=attempts+1, updated_at=?
 WHERE video_id=? AND status='queued';
-- rowcount 0 means someone else got it. Move on; do not retry.
```

### `backoffs` (2026-09-22, T2)

```sql
CREATE TABLE backoffs (
  worker TEXT PRIMARY KEY,   -- 'captions', 'downloader', ...
  until  TEXT NOT NULL,      -- ISO8601 UTC; the worker makes no requests before this
  reason TEXT, set_at TEXT NOT NULL);
```

A worker that stops itself on a YouTube bot check / 429 writes **when it may
ask again**, so the decision survives the process. Without it, "stop and wait"
only held for as long as that process lived: launchd, a human or a heartbeat
could restart the worker a minute later and walk straight back into the
challenge. `pipeline/captions.py` reads the row before its first request and
exits 0 if it is still in force (`--backoff-status`, `--clear-backoff`). An
unparseable `until` counts as expired — a broken row must never wedge a worker
shut for good.

---

## 2. `corpus.sqlite` — search

Why SQLite FTS5: the current `stance_analysis.py` finds mentions with
`re.escape(target)` over segment text, so "Israel" misses IDF, Netanyahu,
Zionist, and every pronoun. A real full-text index with stemming and ranked
retrieval is a better instrument, and it's ~20 lines.

```sql
CREATE TABLE segments (
  video_id  TEXT NOT NULL,
  seg_idx   INTEGER NOT NULL,   -- order within video; needed for context windows
  start_s   REAL NOT NULL,
  end_s     REAL NOT NULL,
  speaker   TEXT,               -- SPEAKER_00 etc. NULL if not diarized
  is_host   INTEGER,            -- resolved by the §4.6 heuristic. The column
                                -- downstream analysis should actually filter on
  text      TEXT NOT NULL,
  PRIMARY KEY (video_id, seg_idx)
);

-- Rows, not JSON blobs. A blob column is an expensive folder — you can't index,
-- join, or aggregate it.
CREATE VIRTUAL TABLE segments_fts USING fts5(
  text, video_id UNINDEXED, seg_idx UNINDEXED,
  content='segments', content_rowid='rowid', tokenize='porter unicode61'
);
```

`porter` stemming so "attacking" matches "attack"; `bm25()` for ranked
retrieval; `snippet()` for KWIC-style qualitative reading, which is the
qualitative arm of the project done properly rather than by eye.

---

## 3. `corpus.duckdb` — analysis

Why DuckDB alongside SQLite: columnar, so `GROUP BY speaker, month` over
millions of segments is instant; reads and writes Parquet; and it queries
directly from R, so chunk scores reach `brms` without a CSV round-trip.

Same `segments` table, plus:

```sql
-- Why a table and not summary.json: the existing scripts rewrite that file from
-- only the current run's videos, which is how 9 of 10 videos were lost. An
-- upsert keyed on (video_id, chunk_idx) cannot do that.
CREATE TABLE meanness_chunks (
  video_id   VARCHAR NOT NULL,
  chunk_idx  INTEGER NOT NULL,
  start_s    DOUBLE, end_s DOUBLE,
  host_only  BOOLEAN,   -- was non-host speech excluded before scoring
  score      DOUBLE,    -- NULL for a failed parse, never 0 — a 0 reads as
                        -- "maximally polite" and silently drags the mean down
  n_samples  INTEGER,   -- k in the k-sample median. Records that this was
  score_sd   DOUBLE,    -- measured with known noise, not asserted
  model      VARCHAR, prompt_version VARCHAR,  -- results are only comparable
                                               -- within a prompt version
  PRIMARY KEY (video_id, chunk_idx)
);

CREATE TABLE stance_mentions (
  video_id VARCHAR, target VARCHAR, seg_idx INTEGER,
  stance VARCHAR,        -- FAVOR | AGAINST | NEUTRAL | UNPARSEABLE
                         -- UNPARSEABLE is its own label: the current code falls
                         -- back to NEUTRAL, silently inflating the neutral count
  window_segs INTEGER, model VARCHAR, prompt_version VARCHAR,
  PRIMARY KEY (video_id, target, seg_idx)
);
```

**No `summary` tables.** Summaries are `SELECT`s. Every stored aggregate is a
thing that can go stale relative to its inputs, which is the original bug in a
new costume.

## Analysis note

Chunks are nested in videos, nested in speakers. The natural query
(`GROUP BY speaker`) pools chunks as if independent and will make a difference
look far more certain than it is — ~10 videos of real information, not ~124
chunks. Keep `video_id` on every row so a random intercept for video is always
available. See `CLAUDE.md`.
