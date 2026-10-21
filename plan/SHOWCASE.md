# SHOWCASE — getting the transit corpus out the door

Draft 2026-10-16 (Wren II, for Puck). Companion to `ANALYSIS-transit.md`
(what is measured) and `T2-PLAN.md` (how the data is made). Every cost is an
estimate unless it cites a measurement. Nothing in this file changes
`DECISIONS.md`, `frame.csv` or the code; it is the plan Puck approves whole.

## 0. Deliverables

1. **Public GitHub repo** of the pipeline, cleaned, with README, LICENSE, CI
   running `make test` offline.
2. **GitHub Pages dashboard** (`hotsoupisgood.github.io/<repo>/`, same pattern
   as `AI-Graph-Response-Compare`: one static page, vanilla D3 v7 from the
   cdnjs CDN, JSON data files beside it):
   - **Geo map** of the Bay Area with one marker per body, sized by
     transcribed episode-hours, coloured by density/transit share. Click a
     body → its meetings. Click a meeting → the episodes, each with a link to
     the **source video at the episode's start timestamp** (YouTube `&t=`,
     Granicus MediaPlayer URL).
   - **Episode reader**: the transcript span we analysed, speaker turns
     labelled by role, keyword and frame terms highlighted, stance score per
     turn shown in the margin. Redacted (see §3).
   - **Flowchart** (D3, same style as `flow_iterate.html`): hypothesis →
     sources → discovery → captions → classifier → episodes → GPU transcribe
     → QA gate → stance pass → models → figures. Each node links to the doc
     or script that does it.
   - The model figures (§5): posterior topic-time by body, role gap.
3. **Write-up** as a post on lucashenry.xyz (Eleventy, TL;DR bullets first,
   then method, results, "which words drove it"-style honesty) linking to the
   dashboard and the repo.
4. **Resume bullets** (3, with numbers) and a 60-second talk track.

## 1. Estimands (from `ANALYSIS-transit.md` §0, §4, §8)

1. **Topic time.** Episode-minutes per meeting-hour, by body × month ×
   family (density / transit). Negative-binomial with exposure offset, body
   random intercepts, month smooth, locate route as a stratum. 90 % intervals.
2. **Role gap.** Stance (−2…+2) and hostility (0–4) of `public` vs `dais`
   turns within the same episode, ordinal, episode + body random intercepts.
   Tier A only: anonymous clusters, no names (`ANALYSIS-transit.md` §8).

Descriptive companions: frame-lexicon rates by role × body; keyword-window
sentiment (±N words around family terms, VADER, episode-bootstrap CIs) as an
appendix; BERTopic on turns as a check on the frame codebook.

## 2. Budget

Puck has **$100/month of Claude API credit** (2026-10-16). The Claude side is
therefore not the constraint; `CLAUDE_CAP_USD` in `.env` still gates every
run and is Puck's edit. RunPod stays at the $10 prepaid credit ($4.99 used).

| item | est. | note |
|---|---|---|
| Stance pass, Sonnet 4.6, all turns ≥ 5 words, k=3 | ~$33 | 6,600 turns today; grows a little to the freeze. Could be all 8,326 turns k=3 for ~$40 |
| Classifier over the remaining captioned meetings | ≤ $25 | `T2-PLAN.md` §3; only what lands before the freeze |
| Final GPU batch | ≤ $5 | needs `approve bN` |
| Everything else | $0 | local |

## 3. Privacy rule for anything published

`DECISIONS.md` #15: anonymous clusters, no names extracted, stored or
indexed; episode transcripts pass a **name-redaction step before loading**.
That step is **not built** (`a0_terms.py:redact_names` is a KWIC heuristic,
not a corpus pass). Publishing episode text on GitHub Pages therefore needs:

1. A redaction pass over every published turn: regex ("my name is …",
   "I'm … from …"), NER (spaCy `en_core_web_sm`, PERSON), and the
   self-identification field from the stance pass used as a cross-check.
   Output replaces names with `[name]`; a count of redactions per episode is
   kept and shown.
2. Publish **episode excerpts**, not whole-meeting transcripts. The link to
   the source video at the timestamp gives a reader the full context from
   the public record; the site itself never becomes the searchable copy.
3. No voice embeddings, no persistent speaker ids (Tier B is future work).
4. A hand check of 30 random published turns by Puck before the site goes
   live.

## 4. Plan (numbered; `approve plan` accepts the whole thing)

| # | Step | Effort | Cost | Why |
|---|---|---|---|---|
| 1 | **Freeze date 2026-10-24.** Captions soak, rediscovery lands 10-17, one last GPU batch if ≥ 15 h ready. Incomplete bodies reported as incomplete | 0 | ≤ $5 | A showcase needs a fixed corpus, not a finishing one |
| 1b | **Per-body consistency** (Puck, 2026-10-16). Today 398 episodes / 82.2 h are transcribed but 11 bodies hold all of it (Santa Clara 29.9 h, Petaluma 17.2 h … Oakland 0.9 h) and 16 bodies have **0 h done with 190 episodes / 29.5 h sitting at `located`** (San Leandro 7.9 h, Richmond 5.6 h, BART 3.2 h, Los Altos 2.4 h, Palo Alto 2.1 h, San Mateo 2.1 h, VTA 1.9 h …). Before the freeze: cut and transcribe **every located episode** (GPU at the measured $0.021/audio-h is < $1; Richmond and San Leandro are the Mac `local_asr` route), so every body with located episodes has *some* hours. In the analysis: the count model needs no cap (exposure is in the offset); the **role-gap sample is capped at 8 h per body**, drawn at random by episode, so Santa Clara and Petaluma cannot dominate the stance estimate, and the full set is reported as a sensitivity | 0 | < $1 | Body is the unit of comparison; 30 h vs 0.9 h is not a comparison |
| 2 | **Repo hygiene.** Commit the dirty tree; gitleaks over full history; root one-offs (`hasan_compare.py`, `meanness_analysis.py`, `stance_analysis.py`, `scrape_hasan_audio.py`, `Claude outputs/`) → `legacy/`; LICENSE; root README with architecture + numbers; GitHub Actions `make test` offline. Keep AGENTS.md/CLAUDE.md, framed as agent-supervised ops with hard money gates | 1 d | $0 | Nothing is visible until this exists |
| 3 | **Fix the open QA bug** (`window_start_s`, `BART-QIL-ASSESSMENT.md`), re-run `make qa` | ½ d | $0 | A reviewer who reads that doc sees an unfixed defect in the gate |
| 4a | **Stance pilot** (Puck, 2026-10-16: pilot before the full run). Prompt + JSON schema (stance, hostility, frames, sub-topic, self-id). **300 turns**, stratified: every body with transcribed episodes, public and dais, turns ≥ 5 words, k=3. Puck and Wren blind-code **150** of them from the codebook; report weighted κ for stance and hostility, per-turn k-spread, and the disagreements as KWIC. **Gate: κ ≥ 0.6 on stance** (or Puck accepts the disagreements as codebook fixes and the prompt is re-pinned) before 4b | 1 d | ~$1.50 | The instrument's credibility is the agreement number |
| 4b | **Stance pass, full.** Same prompt version as the passing pilot, Message Batches, all turns ≥ 5 words, k=3 | batch | ~$33 | Estimand 2 |
| 5 | **Redaction pass** (§3) and the published-data export: `bodies.json` (name, lat/lon, platform, meetings, hours), `meetings.json` (date, source_url, episodes), `episodes.json` (start/end, tags, route, turns with role, redacted text, keyword spans, stance) | 1 d | $0 | Gate for publishing any text |
| 6 | **Models.** Estimand 1 count model; estimand 2 ordinal model; PPCs; shrinkage plot. PyMC in the venv (or brms, Puck's call). Facets never pooled | 3–4 d | $0 | The Bayesian piece |
| 7 | **Dashboard** (§0.2): map, meeting/episode drill-down, episode reader with highlights, flowchart, model figures. Single `index.html` + `flow.html` + `data/*.json`, D3 v7, no build step. Separate public repo so the pipeline repo stays code-only | 3 d | $0 | The thing people click |
| 8 | **Write-up** on lucashenry.xyz: TL;DR bullets, question, data, instrument validation (8.7 % ASR disagreement; classifier vs T0; stance agreement), results with intervals, what broke (two 429 shutdowns, Granicus Flowplayer parser recovering 178 h, Haiku rejected at 0.758), limitations (platform selection, unlocated bodies, routes as strata) | 2 d | $0 | The narrative |
| 9 | **Package.** Resume bullets, talk track, pinned repos, screenshot | ½ d | $0 | |

Order: 1–3 now (2 and 3 in parallel with the soak), 4 starts once Puck has
read the preview, 5 after 4, 6 and 7 in parallel after 5, 8 after 6, 9 last.
About three weeks part-time.

## 5. Geo data

No coordinates exist in the repo. `plan/bodies_geo.csv` (body, display name,
county, lat, lon, kind council/board/transit) is hand-entered for the 33
bodies; transit boards get their HQ (BART Oakland, VTA San José, SamTrans San
Carlos). $0, 20 minutes.

## 6. Decisions for Puck

1. Freeze date (default 2026-10-24).
2. Stance pass row: default Sonnet 4.6, turns ≥ 5 words, k=3 (~$33).
   `CLAUDE_CAP_USD` → 60.
3. Modelling stack: PyMC (in venv, Python end to end) or brms/Stan (his R
   habit; the plan docs say brms).
4. Dashboard repo name under `hotsoupisgood`.
5. Published text: episode excerpts, redacted, with timestamp links (§3
   default) — confirm.

## 7. Future directions (documented, not scheduled)

`ANALYSIS-transit.md` §8 Tier B (persistent anonymous speaker ids via voice
embeddings; needs a DECISIONS entry) and Tier C (dais seat mapping only, with
an explicit ethics decision and the redaction step in place; public commenters
never). `ANALYSIS-transit.md` §5 extensions.
