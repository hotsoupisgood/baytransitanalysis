# Proposal: Bay Area density + transit meetings (draft for Puck)

**Status:** draft, 2026-09-20. Document only — nothing in `DECISIONS.md`,
`SAMPLING.md`, `frame.csv` or the code changes until Puck has read this.
**Inputs fixed by Puck on 2026-09-20** (not reopened here): Bay Area only; two
keyword families (density, transit) feeding one episode extractor; 12-month
window Sep 2025 – Sep 2026; the "caption method" (locate with YouTube
auto-captions, GPU-transcribe only the episodes); the Hasan/Tucker corpus stays
as a finished side corpus; the operating rules (laptop downloads at ≤2–3
concurrent, texted approval per GPU batch) stay.

Every number below is either derived (arithmetic shown) or marked ASSUMPTION.
Nothing here has been run beyond today's two segmentation pilots.

---

## 1. Research question and unit of analysis

**Question.** In Bay Area city councils, county boards and transit boards over
the last twelve months, how much meeting time goes to density (housing element,
upzoning, ADUs, parking minimums…) and to transit (BART, Caltrain, SMART, bus
and bike lanes, road diets…), who speaks in those episodes — the dais or the
public — and how does the stance taken by each side vary across bodies and
months? The Hasan/Tucker question (how two commentators talk) becomes a side
corpus; this one is about how *deliberative bodies and their publics* talk.

**Unit of analysis.** The **episode**: one contiguous span of one meeting that
is about density, transit or both, with `[start_s, end_s]`, tags, and the
transcript of that span with anonymous speaker clusters and a role label per
cluster. Episodes nest in meetings, meetings in bodies. For rates and counts the
cell is **body × month**, with exposure = meeting-hours in that cell (the
analogue of the realised hours per cell in `SAMPLING.md`). A speaker cluster
within an episode is the unit for the public-vs-dais comparison; chunks within
a cluster are what the stance scorer rates, and the hierarchy
(chunk ⊂ cluster ⊂ episode ⊂ meeting ⊂ body) is what the model has to carry —
the same warning as `CLAUDE.md` "Statistical note", one level deeper.

**"NIMBY-style", operationalised.** Three measurable things, and nothing else:

1. *Topic*: the episode matches the density and/or transit keyword family and
   the LLM classifier (§4) says the window is *about* the topic, not a passing
   mention.
2. *Speaker role*: each diarized cluster is labelled `dais` (council/board
   members), `staff`, `public` (public comment), or `unknown` (§3g).
3. *Stance*: a chunk-level pro/anti-density (and pro/anti-transit) score from a
   written codebook, scored by a pinned Claude model with k samples and human
   agreement reported (`docs/MODELS.md` §2). Opposition frames — "neighborhood
   character", "traffic", "parking", "local control" — are counted as
   collocates, not asserted.

What it is **not**: a label on a person (no names are extracted, §6); general
sentiment; any mention of housing (a consent-calendar item read aloud is a
`mention`, not an episode); or a claim about outcomes of votes.

---

## 2. Sampling design

### 2.1 Population

All regular meetings of the bodies in `plan/sources_bayarea.md` between
2025-09-20 and 2026-09-20. Counting that file's rows by platform
(`sources_bayarea.md` lines 22–47, 61–70, 85–92):

| Group | YouTube `/videos` | YouTube `/streams` | Granicus | Not located | Total |
|---|---|---|---|---|---|
| Councils + county boards | 11 | 4 | 14 | 7 | 36 |
| Transit boards | 2 | 0 | 4 | 1 | 7 |
| **Total** | **13** | **4** | **18** | **8** | **43** |

YouTube councils: San José, Palo Alto, Sunnyvale, Cupertino, San Mateo, Daly
City, Concord, Milpitas, Los Altos, Petaluma, Cotati; `/streams`: San Rafael,
Santa Rosa, Healdsburg, Rohnert Park. Granicus councils: SF BoS, Oakland,
Berkeley (council view_id not yet located), Mountain View, Fremont (thin),
Santa Clara, Alameda, Richmond, Hayward, San Leandro, Walnut Creek, Sausalito,
Marin County (BoS view_id not located), Napa County. Boards: BART, VTA
(YouTube); AC Transit, SamTrans, MTC/ABAG, SFMTA (Granicus). Not located:
Redwood City, Pleasanton, Livermore, Vallejo, Windsor, Novato, Sonoma County
BoS, Caltrain. California HSR (line 92) is statewide and excluded.

**Meetings in the window** (ASSUMPTION: 2 regular meetings/month per council,
1.5/month per board; San Rafael's "16 in 2026" by Sep 8 is 1.9/month, so 2 is
about right):

- 29 located councils × 24 = 696; 6 located boards × 18 = 108 → **804 meetings**
- with the 8 unlocated bodies: + 7 × 24 + 18 = 186 → **990 meetings**

**Meeting length.** Today's 12 pilot meetings total 45.67 h (durations from
`output/segtest/*/*.info.json`: Solana Beach 1.81–6.27 h, North Bay 2.86–4.69
h) → mean **3.8 h**. Meeting-hours in the window: 804 × 3.8 = **3,055 h**
(990 × 3.8 = 3,762 h with the unlocated bodies).

**Episode share.** Pilots, each with a *single-agency* keyword set:

| Pilot | Meetings | Meeting-h | Episodes | Episode-min | Share |
|---|---|---|---|---|---|
| Solana Beach / Coaster, threshold 1.5 | 6 | 25.2 | 3 | 13 | 0.9 % |
| North Bay / SMART, threshold 0.8 | 6 | 20.5 | 7 | 27 | 2.2 % |
| Both | 12 | 45.7 | 10 | 40 | 1.5 % |

Two broad families (density has far more agenda time than one rail agency) will
score higher. ASSUMPTION, per Puck's brief: **2–5 % of meeting time**, with a
10 % row as the sensitivity case. Padding adds ~25 % (30 s each side on a
typical 4-min episode; `pipeline/segment.py:114` default `pad=30`).

### 2.2 Cost of a census under the caption method

GPU rate is now **measured**, not the PLAN.md estimate: batch b9 ran 50.67
audio-h in 129.7 GPU-min for $1.06 (RTFX 25.0x, ASR + align + diarize;
`batches.note`), i.e. **$0.021 per audio-hour all-in** ($1.06 / 50.67 h) at the
$0.49/h A40 rate ($1.0593 / 2.162 h). b11 agrees ($0.50 / 23.95 h = $0.021/h).

| Episode share | Episode-h (804 mtgs) | + 25 % pad | GPU-h @25x | GPU $ @ $0.49/h | × 1.5 contingency (`gpu_batch.py:60`) |
|---|---|---|---|---|---|
| 2 % | 61 | 76 | 3.1 | $1.6 | $2.4 |
| 5 % | 153 | 191 | 7.6 | $4.0 | $6.0 |
| 10 % | 306 | 382 | 15.3 | $8.0 | $12 |
| whole meetings (comparison) | 3,055 | — | 122 | $64 | $96 |

With the 8 unlocated bodies multiply by 1.23. The caption method saves ~$60–90
of GPU, but the larger saving is download volume: 3,055 h of audio at the
measured 58 MB/h (`PLAN.md` §4.3) is 177 GB and weeks of polite fetching;
191 h of sections is 11 GB and ~3 days.

**Unknown that matters:** RTFX on 3–10 min clips. b9's 25x was on 45-min
files; WhisperX and pyannote have per-file warm-up, so short clips may run at
half that. Even at 12x the 5 % row is $12. T1 measures it.

**Granicus bodies (no captions): 18 bodies, 14 × 24 + 4 × 18 = 408 meetings ×
3.8 h = 1,550 meeting-hours** need a first pass to *locate* episodes:

| Option | Cost | Notes |
|---|---|---|
| Granicus agenda index / Legistar item timestamps | $0 | Granicus clip pages usually carry per-item jump times (`sources_bayarea.md:14`). Match item titles to the keyword families → candidate items. Coverage across the 18 archives UNKNOWN — probe 3 in T1 |
| Granicus closed-caption track, if served | $0 | Many archives serve CC for accessibility; not checked on any of ours. Probe in T1 |
| ASR-only locate pass on the A40, `large-v3-turbo`, no alignment/diarization | 1,550 h ÷ 60x = 26 GPU-h → **$13**; at 40x $19, at 100x $8 (ASSUMPTION: ASR-only turbo runs 2–4x faster than the measured 25x full stack) | Same batch/approval machinery, a `mode=locate` job; its output is a caption-equivalent VTT that feeds the same segment stage |
| Mac `mlx-whisper` turbo (`docs/MODELS.md` §4) | $0, ~8 days of Mac time at an ASSUMED 8x realtime | fine for probes, not for 1,550 h on the download host |
| Groq ASR (`DECISIONS.md` #3: $0.04/h) | $62 | no |
| AssemblyAI ($0.15/h) | $232 | no |

Recommendation: probe the two free routes on Oakland, AC Transit and MTC in T1;
if either covers most archives, the paid pass shrinks to the remainder. Budget
the locate pass at **$10–30**. Also unknown: Granicus HLS is video+audio; a
3.8-h meeting at 1–2 Mbit/s is 1.7–3.4 GB of transfer per meeting (0.7–1.4 TB
for 408) unless the master playlist has an audio-only or low-bitrate rendition,
or the clip page links an MP4. `ffmpeg -vn` keeps only the audio on disk, but
the bytes still cross the wire. Check on the first archive before scaling.

### 2.3 Census or stratified sample — recommendation: census, with a valve

`DECISIONS.md` #9 chose a stratified sample because a census of *shows* is
dominated by whoever uploads most, and transcription cost scaled with everything
downloaded. Both premises change here:

- Locating episodes costs nothing on YouTube bodies (captions) and $10–30 on
  Granicus ones; transcription cost scales with episode-hours, which the table
  puts at $2–12. There is no budget reason to sample.
- The primary quantities are *rates* — episodes and episode-minutes per
  meeting-hour by body and month. Sampling meetings throws away exactly the
  counts the rates are made of. A stratified alternative (body × month, one
  meeting per cell = 35 × 12 = 420 meetings) halves the work and loses the
  within-cell variance that distinguishes "one big hearing" from "every
  meeting".
- The dominance problem (SF BoS or San José at 6–9 h per meeting, weekly) is
  real for *token-based* statistics. Handle it in the model (body random effect,
  meeting-hours as exposure offset; body-level summaries before pooling) rather
  than by discarding data — and add a **per-body monthly cap on transcribed
  episode-hours** (proposed 3 h, mirroring the cell quota) as a spend valve.
  Episodes beyond the cap keep their `episodes` row (located, counted) but are
  not transcribed until Puck lifts the cap. Counts stay a census; transcripts
  are capped.

Report n from the `episodes` table at `method`/`classifier_version`, never from
what is on disk (`SAMPLING.md` last paragraph, same reason).

---

## 3. Pipeline changes — build list

Effort in hours of a coding session, ASSUMPTIONS based on the sizes of the
existing modules (`pipeline/downloader.py` 302 lines, `gpu_job.py` 813,
`segment.py` 209).

| # | Change | Touches | Hours |
|---|---|---|---|
| a | **`kind=meeting` sources, no 45-min cap.** Third column in `plan/channels.tsv` (`show` \| `meeting`); `crawl_metadata.sh:41` already reads `speaker url _rest`, so the column is ignored by the crawl and safe to add now. `state.py` DDL (`pipeline/state.py:33–55`) gains `kind`; `discover.py:37` stores it; `transcode.py:39` `cap_seconds()` returns no cap when `kind='meeting'` (episodes are cut earlier, at download). `speaker` becomes the body slug for meeting rows — the column name stays because `load_db`, `word_freq` and `status` all read it | channels.tsv, state.py, discover.py, transcode.py | 3 |
| b | **Granicus adapter.** `pipeline/granicus.py`: listing (`ViewPublisher.php?view_id=N`) → clip ids + dates + titles → clip page → `playlist.m3u8` (+ agenda index if present) → rows in `videos` with `platform='granicus'`, `video_id = 'gr:<site>:<clip>'`. Downloader branch: `ffmpeg -vn` from the m3u8 instead of yt-dlp (`downloader.py:58` `ytdlp_command`). Note: the Solana Beach archive (`view_id=5`) stops in 2015 and current Solana Beach meetings are on YouTube (`sources_bayarea.md:13`), so unit tests can use that archive's listing but the live probe must be a Bay Area site (Oakland). Estimated at ~60 lines in `sources_bayarea.md:13`; the agenda-index scrape and audio-only rendition hunt are the rest | new granicus.py, downloader.py, state.py | 6 |
| c | **YouTube `/streams` discovery.** Only a URL change in `channels.tsv` for San Rafael, Santa Rosa, Healdsburg, Rohnert Park (`sources_bayarea.md:52–55`); `discover.py:89` already skips `is_live`. Verify that yt-dlp fetches captions and audio from live-stream archives without 403s (§6) | channels.tsv | 2 |
| d | **Caption fetch stage + segment stage.** New `pipeline/captions.py`: `yt-dlp --skip-download --write-auto-subs --sub-langs "en,en-orig"` (never `en.*` — `PLAN.md:180–183`, the 429 that killed the Tucker run) into `output/subs/<id>.en.vtt`; new state `captioned` between `discovered` and `downloading`. `segment.py` becomes importable (`find_episodes`, `segment.py:114`) and a stage that writes `episodes` rows (§5) with `method='keyword'`, then the classifier (§4) updates `label` and `score`. Same polite pacing as the downloader (`downloader.py:15`, 45–180 s gaps): 804 caption fetches ≈ 25 h wall-clock at concurrency 1 | new captions.py, segment.py, state.py | 6 |
| e | **Section-only audio download.** Per episode, not per video: `yt-dlp --download-sections "*<start>-<end>"` on `bestaudio` into `output/audio/<video_id>_e<n>.<ext>`; today's pilot already produced two such clips by hand (`output/segtest/Q1PCSNkISV8/episode_1.opus` 430 s, `iyh8o9tJgVI/episode_1.opus` 250 s — 48 kHz stereo opus straight from YouTube, i.e. the section fetch works). Granicus: `ffmpeg -ss/-to -vn` on the m3u8. The state machine moves to the episode row (`episodes.status`), `videos.status` stays for the whole-meeting stages. This is the largest change because `downloader.py:159 download_one` and `transcode.py:78 transcode_one` are keyed on `video_id` | downloader.py, transcode.py, state.py | 8 |
| f | **GPU job over episode clips.** `gpu_job.py:376 build_manifest` reads episode rows; the pod side is unchanged (one opus in, one JSON out). Output `output/whisper/<video_id>_e<n>.json` with `window_start_s = start_s` (already in the contract, `PLAN.md` §1) plus `episode_id`, `meeting_id`, `body`, `tags` — added keys only. `ingest_new` (`gpu_job.py:584`) writes back to `episodes` | gpu_job.py, gpu_batch.py | 4 |
| g | **Speaker-role labelling (dais vs public).** Post-batch script `scripts/role_label.py`, like `host_enroll.py`. Cues, in order: (1) transcript phrases at cluster boundaries — "public comment", "next speaker", "you have two/three minutes", "your time is up", "thank you, next"; (2) the timer beep, detectable as a short tone in the audio (energy in a narrow band) — cheap to try, unverified; (3) agenda-item boundaries from Legistar/Granicus index when present; (4) diarization structure: dais clusters recur across the whole meeting, public clusters speak once for ≤ 3 min. Writes `speaker_roles` rows (§5) and `role_method`. Host enrolment (`PLAN.md` §4.6) does not transfer — there is no single host — but the embedding code in `host_enroll.py` can enrol the clerk/mayor per body if cues (1)–(4) prove weak | new script; load_db.py | 8 + 4 validation |
| h | **QA gate adaptations.** `qa_transcripts.py:65` coverage thresholds (0.60 WARN / 0.30 FAIL) assume stream audio; a public-comment episode has dead air between speakers, so keep FAIL at 0.30 but make WARN 0.50 for `kind=meeting`. Add: hallucination on dead air (the loop check at `:53` already covers repeats; add a "text during a >30 s silence" WARN using `GAP_S`), and an edge check — speech in the first/last 5 s of a clip means the cut is mid-sentence (§6, pad) | qa_transcripts.py | 3 |
| i | **Dashboard / status.** `status.py:79 collect` reads `frame_cells.csv` for the speaker × month heatmap; add an `episodes` panel (located / classified / downloaded / transcribed per body-month) and alerts for "body with zero meetings in 60 days" (source dried up — Cotati, `sources_bayarea.md:66`). Approval messages must show *episode-hours*, not video count | status.py, page.html | 4 |
| j | **Analysis.** Carries over unchanged: `load_db.py` (add `episode_id`, `body`, `tags`, `role` columns; `is_host` becomes `role`), `word_freq.py` overall/kwic (whole-word node match, `word_freq.py:224`), cluster bootstrap by episode instead of video. `stance_analysis.py` carries over with new targets ("housing element", "BART", "parking") and a role filter. `meanness_analysis.py` is not the construct here — keep it runnable for the side corpus, do not run it on meetings. New: episode counts and minutes per body-month with exposure (SQL over `episodes` + `meetings`); pro/anti-density stance codebook scored on 350-word chunks (k = 3, median, `score_sd`), model and `prompt_version` pinned; public-vs-dais comparison as a hierarchical model with episode and body intercepts. Cost of scoring at 5 %: 191 h × 150 wpm × 60 = 1.7 M words ≈ 4,900 chunks × ~1,000 tokens ≈ 4.9 M input tokens → $15 on Sonnet 4.6, $44 at k = 3, $22 via the Batches API | load_db.py, word_freq.py, stance_analysis.py, new episode_rates.py | 12 |
| — | **Segmentation upgrade** (§4) | segment.py, new classify_windows.py | 6 |
| — | **Schema** (§5) + tests in `analysis/test_pipeline.py` | state.py, SCHEMA.md, test_pipeline.py | 4 |
| | **Total** | | **≈ 70 h** |

---

## 4. Segmentation: two stages instead of one threshold

**Why.** The density segmenter finds agenda items (Solana Beach: 20–40 min of
LOSSAN, `segment.py:6`) but the SMART pilot showed the other shape of the
signal: 1–5 min liaison reports and asides (`sources_bayarea.md:72–79`). At
threshold 1.5 it found one of them; at 0.8 it found all seven and still missed
a liaison report that mentioned the agency once in passing. A single density
threshold cannot separate "two minutes about SMART" from "two minutes of
unrelated business that used the word train twice". The keyword file already
carries the ambiguity list — bare `smart` (41/41 hits were the agency, but
"smart meter" will break that), `train` as a verb, `freight` as road freight,
`measure q` (`keywords_smart.txt:18–22, 39–40, 58–64`). With the density
family added, `traffic`, `parking` and `housing` make it worse.

**Proposal.** Stage 1, keyword windows at a **low threshold (0.5–0.8)** on the
union keyword file (`plan/keywords_density.txt` + `keywords_transit.txt`,
each tagged with its family) → candidate windows, deliberately over-inclusive.
Stage 2, a **Claude classifier** labels each candidate (its caption text ± pad)
as one of `about_density`, `about_transit`, `about_both`, `mention_only`, `no`,
with a one-line rationale. Per `docs/MODELS.md` §2: pinned model
(`claude-sonnet-4-6`, the current `SCORING_MODEL`, retirement not before
2027-02-17), `thinking={"type": "disabled"}`, no sampling parameters, a
structured-output JSON schema, prompt version recorded on every row. Caption
text goes in as quoted data, never as instructions (rule 5 in the supervisor
AGENTS.md). Adjacent `about_*` windows merge into one episode; `mention_only`
rows are kept in `episodes` with `status='mention'` and are never downloaded
— they are a count, which the SMART corridor showed is most of the signal.
Structural cues (chapters, description timestamps, Legistar or Granicus agenda
items) are added when present as a third vote (`segment.py:147
structural_chapters`, already implemented; none of the 12 pilot uploads had
chapters).

**Cost.** ASSUMPTION: 10–20 candidates per meeting on the union list (the SMART
set alone gave 1.2/meeting at 0.8 with zero false positives in the three
SMART-free meetings). Per candidate: ~500 tokens of cached instructions +
codebook, ~900 tokens of captions (4–6 min at ~150 wpm), ~60 tokens out. Per
meeting at 15 candidates: 21 k in, 0.9 k out → 21 k × $3/M + 0.9 k × $15/M =
**$0.077**. For 990 meetings: **$77** single-sample, **$38** through the
Batches API (50 % off), $115 if k = 3. Recommendation: k = 1 for this gate —
it is a screen, not the measurement — and measure its error instead: Puck and
the agent hand-label 100 candidates blind in T0, report precision/recall by
family. On `claude-sonnet-5` ($2/$10 but ~30 % more tokens, `MODELS.md:38`) the
saving is ~10 %; not worth switching instruments for.

**T0 acceptance for the segmenter:** recall ≥ 0.9 of hand-labelled `about_*`
windows at threshold 0.5, and the liaison-report case from the SMART pilot
caught.

---

## 5. Schema additions (`SCHEMA.md` style — add, never rename)

```sql
-- Why a sources table: a body may have several platforms over time (Solana
-- Beach: Granicus to 2015, YouTube since). channels.tsv stays the human-edited
-- input; this is what discovery resolved it to.
CREATE TABLE sources (
  source_id   TEXT PRIMARY KEY,   -- 'san_rafael:youtube:streams'
  body        TEXT NOT NULL,      -- slug; also written to videos.speaker for meeting rows
  body_kind   TEXT NOT NULL,      -- council | county | transit_board | commission
  platform    TEXT NOT NULL,      -- youtube | granicus | legistar
  platform_id TEXT NOT NULL,      -- channel id + tab, or granicus site + view_id
  region      TEXT NOT NULL,      -- 'bayarea' now; the expansion axis later
  active      INTEGER DEFAULT 1,
  note        TEXT
);

-- Why meetings is separate from videos: one meeting can be several uploads
-- (Cupertino multi-part) and one Granicus clip is not a YouTube video_id.
CREATE TABLE meetings (
  meeting_id   TEXT PRIMARY KEY,  -- '<body>:<YYYY-MM-DD>'
  source_id    TEXT NOT NULL,
  body         TEXT NOT NULL,
  meeting_date TEXT NOT NULL,     -- from title/agenda, not upload date: uploads lag a day
  month        TEXT NOT NULL,     -- the cell
  regular      INTEGER DEFAULT 1, -- special/closed sessions counted but excluded from rates
  video_ids    TEXT NOT NULL,     -- JSON list, in order
  duration_s   REAL,              -- exposure for the body x month cell
  captions     TEXT,              -- auto | manual | none | locate_pass
  agenda_url   TEXT,              -- Legistar/Granicus, when found
  crawled_at   TEXT NOT NULL
);

-- Why episodes carry method and score: the same span can be found by keyword
-- density, the classifier and an agenda item; the analysis must be able to
-- report n under each and re-run one without touching the others.
CREATE TABLE episodes (
  episode_id   TEXT PRIMARY KEY,  -- '<video_id>_e<n>'
  meeting_id   TEXT NOT NULL,
  video_id     TEXT NOT NULL,
  start_s      REAL NOT NULL, end_s REAL NOT NULL,   -- padded bounds actually downloaded
  core_start_s REAL, core_end_s REAL,                 -- unpadded, for durations
  tags         TEXT NOT NULL,     -- 'density' | 'transit' | 'both'
  method       TEXT NOT NULL,     -- keyword | classifier | agenda | manual
  score        REAL,              -- keyword hits/100w, or classifier confidence
  label        TEXT,              -- about_density | about_transit | about_both | mention_only | no
  classifier_model TEXT, prompt_version TEXT,
  agenda_item  TEXT,              -- Legistar item id or chapter title, if any
  status       TEXT NOT NULL,     -- located -> approved -> downloading -> downloaded -> transcoded
                                  --   -> queued -> transcribing -> done | mention | failed | dead | capped
  audio_path TEXT, opus_path TEXT, transcript_path TEXT,
  attempts INTEGER DEFAULT 0, last_error TEXT, updated_at TEXT NOT NULL
);
CREATE INDEX idx_ep_status ON episodes(status);
CREATE INDEX idx_ep_meeting ON episodes(meeting_id);

-- Why roles are rows, not a key in the transcript: several methods, revisable,
-- and the transcript JSON must stay the contract in PLAN.md 1 (keys added,
-- never nested). Labels are anonymous clusters; no names, by design (S6).
CREATE TABLE speaker_roles (
  episode_id  TEXT NOT NULL,
  speaker     TEXT NOT NULL,      -- SPEAKER_00 ...
  role        TEXT NOT NULL,      -- dais | staff | public | unknown
  confidence  REAL,
  method      TEXT NOT NULL,      -- cue_phrase | timer | agenda | structure | enrol | manual
  talk_s      REAL,
  PRIMARY KEY (episode_id, speaker)
);
```

Transcript JSON: unchanged shape, added keys `episode_id`, `meeting_id`,
`body`, `body_kind`, `tags`, `episode_start_s` (= `window_start_s`),
`roles` (a copy of `speaker_roles` for that file, so the file stays
self-describing). `batches`/`batch_videos` gain an `episode_id` column;
`audio_hours` already means what it says.

---

## 6. Risks and unknowns

| Risk | What we know | Mitigation |
|---|---|---|
| Granicus rate limits / blocking | Nothing — one listing and one clip fetched today | Same pacing as the downloader (45–180 s gaps), one archive at a time, stop on any 4xx and tell Puck; Granicus is not YouTube, so the residential-IP rule is not the constraint but the politeness rule stays |
| YouTube 403s on live-stream archives | `/streams` VODs are known to be pickier about player clients; the crawl today was metadata only | `downloader.py:79` already stops the worker at three 403s in a row; T0 tests one caption + one section fetch per `/streams` body before any bulk work |
| Auto-caption quality on council audio | Pilots: agency names mangled ("Hillsburg", "Snowoma", "Lurchburg", `keywords_smart.txt:69–75`); density still worked | The keyword files carry ASR variants; the classifier reads context; the *transcript* comes from WhisperX, captions only locate |
| Diarization on 20-speaker public comment | Unknown; pyannote clusters by voice, and 20 one-time speakers of 2 min each is the hard case (over-merging, `PLAN.md` §4.6) | Role labels lean on turn structure and cue phrases, not cluster identity; hand-check 10 min of public comment in T1 and record the observed failure rate before any claim about "the public" |
| "smart" / "train" / "measure q" ambiguity | Measured today: 41/41 `smart` hits were the agency in six meetings; `train` fired as a verb 3x; `measure q/i` 4/4 wrong and disabled | Stage-2 classifier; per-body keyword notes; re-check when a city with a smart-meter item appears |
| Clip boundaries cutting mid-sentence | Default pad 30 s; keyword windows are 60 s, so a boundary error of up to one window is expected | Pad 45–60 s, snap to the nearest caption-cue gap ≥ 1.5 s, QA edge check (§3h); record `core_*` and padded bounds separately |
| Short clips lower RTFX | b9's 25x is for 45-min files | T1 measures on the pilot batch; the cost table's worst case is $12 at half throughput |
| Granicus transfer volume | HLS is video+audio; 0.7–1.4 TB for 408 meetings if no audio rendition exists | Check the master playlist on the first archive; prefer the lowest rendition; never store video |
| Ethics — naming public commenters | Commenters state their names on the record; transcripts of identifiable people are personal data (`PLAN.md` §9 item 9) | **Speaker labels stay anonymous clusters with a role; no names are extracted, indexed or stored; a name-redaction pass over episode transcripts before `make load` (regex on "my name is …" + NER) is part of §3g.** Meetings are public record, which makes the *corpus* defensible, not the *people* searchable |
| Source drift | Cotati stopped in March 2026; Windsor and Novato post no full meetings | Dashboard alert on 60 days of silence per body; `meetings.regular` so gaps are visible in the exposure, not silently zero |
| Prompt injection via captions/agenda text | Untrusted by rule 5 | Classifier input is quoted data with a fixed schema output; nothing from captions reaches a shell or a URL |

---

## 7. Phased plan with gates

**Phase T0 — this week (Sep 21–27), no GPU, ~14 h.** Sources rows for three
YouTube bodies: **San Rafael** (`/streams`, the liaison-report shape), **San
José** (`/videos`, 2–9 h meetings, big-city density items), **BART** (a transit
board, dais-heavy). Build (a), (c), (d) and the union keyword files; fetch
captions for their 12-month window (≈ 24 + 24 + 18 = 66 meetings, ~2 h of
polite fetching); run stage 1 at 0.5 and the classifier on ~10 meetings (~$1);
hand-label 100 candidate windows blind (Puck 50, agent 50, then swap).
**Gate:** recall ≥ 0.9 on `about_*`, candidates/meeting and $/meeting measured,
the SMART false negative reproduced and caught, no 403 on a `/streams` caption
or section fetch.

**Phase T1 — Sep 28 to Oct 11, ~30 h, one GPU batch under $1.** Granicus
adapter (b) probed on Oakland, AC Transit, MTC, including the free locate
routes; `episodes`/`meetings`/`sources`/`speaker_roles` tables (§5); section
downloads (e); GPU job over episodes (f); role labelling (g) first cut; QA (h).
Pilot batch: the T0 episodes from ~10 meetings, ≈ 1–3 episode-hours → $0.05–
0.10 at the measured rate (under $1 even at a quarter of the throughput);
proposed with `gpu_batch.py propose`, approved by text, launched detached.
**Gate:** RTFX on clips measured and the cost table rewritten; zero QA FAIL on
the pilot; diarization and role labels hand-checked on 10 min of public comment
with the agreement recorded; Granicus first-pass method chosen with its cost.

**Phase T2 — Oct 12 to ~Nov 15, ~26 h of build plus running time.** Add the
remaining located bodies (35 total); spend ≤ 2 h locating the 8 missing ones
or drop them explicitly; captions for all YouTube meetings (~25 h wall-clock);
Granicus locate pass ($10–30, approved as batches); classifier ($40–80);
section downloads (~3 days); GPU batches at the `BATCH_TARGET_HOURS` cadence
(`gpu_batch.py:55`, each texted); dashboard (i); analysis (j). Calendar is
paced by polite fetching, not by compute. **Gate:** a coverage report per
body-month (meetings found vs expected, episodes located vs transcribed vs
capped), actual vs estimated spend, and the first episode-rate table before any
stance scoring.

Then expand (region column in `sources`, §5) — the SoCal corridor bodies from
the Coaster pilot are the obvious second region.

---

## 8. What Puck must decide next

Only items where different answers produce materially different work:

1. **Census with a per-body cap (recommended, §2.3) or a body × month sample?**
   And the cap value (proposed 3 transcribed episode-hours per body per month).
2. **Planning commissions / zoning boards** — Berkeley ZAB and Marin PC are the
   view_ids already located, and density is arguably *more* concentrated there.
   Recommended: not in T0–T2 (`body_kind='commission'`, `active=0`), add after.
3. **Public-comment-only episodes** (the public speaks, the dais does not
   respond) — include as episodes? Recommended: yes, tagged; they are the
   NIMBY-style signal.
4. **Mention-only rows** — keep as counts without transcripts (recommended), or
   transcribe them too (+ unknown hours, probably small).
5. **Granicus first pass** — approve probing the free routes, and a $10–30
   locate-pass budget if they fail.
6. **Names** — confirm anonymous speaker labels and a redaction pass; no name
   extraction, ever, even for council members (who are identifiable from
   agendas anyway).
7. **Drop or locate** the 8 bodies without a found archive (2 h of probing).
8. **Classifier instrument** — `claude-sonnet-4-6`, k = 1, Batches API
   (recommended) — or k = 3 at 3x the cost.
9. **California HSR** — excluded as statewide (recommended) or kept as a
   comparison board.

---

## Appendix — DECISIONS.md entries that would be *added* if approved

Drafted in the table's format; numbers continue from #11. Nothing existing
changes; #9 (stratified sample) stays true for the Hasan/Tucker corpus.

| # | Decision | Why | Reopen if |
|---|---|---|---|
| 12 | **Second corpus: Bay Area council, county and transit-board meetings, Sep 2025 – Sep 2026, a census of regular meetings; only density/transit episodes are transcribed, with a per-body monthly cap on transcribed episode-hours** *(2026-09-20, Puck; cap value TBD)* | Rates per meeting-hour need every meeting counted; locating costs ~nothing on captioned bodies and transcription scales with episode-hours ($2–12 at 2–10 %), so the reason for #9's sampling does not apply. The cap is the spend valve and keeps big cities from dominating tokens | A body's cap binds every month — then raise it for that body or stratify tokens in the model |
| 13 | **Caption method: locate before transcribing.** YouTube auto-captions (or a cheap ASR-only pass for Granicus bodies) find candidate windows; WhisperX with diarization runs only on the extracted, padded episodes | Whole meetings are 3.8 h of which 2–5 % is on topic; transcribing everything costs 20–50x the GPU and, worse, 177 GB and weeks of downloads for 11 GB of use | Auto-captions prove too poor to locate (T0 recall < 0.9) — then the ASR-only pass runs on YouTube bodies too |
| 14 | **The episode is the unit; it is defined by two keyword families at a low density threshold plus a pinned Claude classifier (model + prompt version recorded per row), tagged density / transit / both** | A single density threshold cannot tell a two-minute liaison report from two stray uses of "train" (SMART pilot). The classifier is a screen with a measured error, not the measurement | The classifier's recall or precision on the hand-labelled set drops below 0.9 / 0.8 on a new body type |
| 15 | **Speaker labels stay anonymous clusters with a role (dais / staff / public); no names are extracted, stored or indexed, and episode transcripts pass a name-redaction step before loading** | Public commenters are private individuals on a public record. The research question needs roles, not identities | Never for public commenters; a body-level "who on the dais" mapping only with an explicit ethics decision |
| 16 | **The Hasan/Tucker corpus (143 videos, `output/whisper`, `corpus.sqlite`) is frozen as a finished side corpus: not re-sampled, not deleted, `kind=show` rows and their analysis code untouched** | It is complete and its frame is drawn; reusing the pipeline for meetings must not change what that study measured | A new question needs more of it — then draw a new frame, per #9 |

The `SAMPLING.md` change, if approved, is a new section "Meeting corpus" (the
population, the census-with-cap rule, exposure and thin-cell rules per
body-month), not an edit to the speaker × month design.
