# FUTURE-STEPS — the docket

Written 2026-10-18. Everything on this page is **not in the current phase**:
it is built or measured only as far as stated, costs money or a decision, and
waits for Puck. Each item gives the evidence, the cost (derivations in
`docs/COSTS.md`), the prerequisite, and the decision it needs. The current
phase is the production flowchart in `README.md`; nothing here changes
`plan/DECISIONS.md`, `plan/frame.csv` or the running pipeline.

| # | item | cost | decision needed from Puck |
|---|---|---|---|
| 1 | Full-ASR locate with codebook v2 | $105 point / $130 planning (Batches); $195–200 direct | adopt v2's transit definition; adjudicate 11 windows; Option 1 vs 2 of #13 |
| 2 | DECISIONS #13 / #14 amendments | $0 | choose Option 1 or 2; the 0.80 recall floor; how recall enters the model |
| 3 | A1 Claude pass | ~$100 | run it, or keep the lexicon pass as the appendix |
| 4 | Full stance pass (estimand 2) | $67–101 at k=1, $202–302 at k=3 on 30–45k turns | k; the κ gate result; 8 h/body cap |
| 5 | Public demo / showcase items | $0 | 30-turn redaction check; repo names; publish |
| 6 | The 10 unlocated bodies | $0 unless an archive appears | accept the thin board side, or drop them from the frame |
| 7 | Two broken-audio Richmond meetings | ≈ $0.05 GPU | re-fetch and retry, or exclude with a note |
| 8 | Transit-operations vocabulary on operator bodies | $0 to re-screen, cents to classify | what "transit" means on BART's own board |

---

## 1. Full-ASR locate with codebook v2

**What it is.** Transcribe every meeting whole on the GPU without
diarization, read every 120-s window with codebook v2 (`v2-ed4c72c9d6`) and
no keyword screen, then cut and diarize the episodes as now. Scripts exist as
pilot tooling only (`scripts/recall_fullasr_v2.py`, `classify.py --codebook v2
--max-run 120`); the pipeline's instrument is still v1 behind the screen.

**Evidence** (`plan/CODEBOOK-V2-PILOT-FULLASR.md`, 4 Richmond / San Leandro
meetings, 18.9 audio-h, 109 reference about-minutes):

| arm | recall any | recall transit | recall density | precision any |
|---|---|---|---|---|
| pipeline today, v1 + screen (`plan/RECALL-CHECK.md`, post-expansion) | 0.753 (0.654–0.852) | 0.435 | 0.843 | 0.761 |
| v2 @ 120 s + screen (`plan/CODEBOOK-V2-PILOT.md`) | 0.780 | 0.783 | 0.787 | 0.720 |
| **v2 + full ASR, no screen** | **0.927 (0.877–0.982)** | **1.000** | 0.910 | 0.711 |

It is the only arm whose "any" interval clears the 0.80 floor proposed in
`plan/DECISIONS-PROPOSED-13.md`. The YouTube auto-caption arm of the same
check (BART, Los Altos, Milpitas, Cupertino) put the caption route at 0.757,
with 61 of 63 misses never clearing the screen (15 of them auto-caption
homophones: fare→fair, ridership→wrership, YIMBY and rezoning dropped).

**Precision caveat.** Transit precision reads 0.355 against the v1 reference,
but all 9 over-called windows (8 transit) are things v2 *defines* as transit
and v1 called `mention_only`: the Richmond e-bike lending library
public-comment run, bike-share funding, pedestrian bridges for school access.
The over-calls are definitional, not noise, and a v1 reference cannot
adjudicate them. **11 windows await Puck's adjudication in
`plan/CODEBOOK-V2-PILOT.md` §5** (text in
`output/validation/codebook_v2/windows_v2_120.jsonl`): if ≥ 6 of 11 are
judged about, v2's precision is at least v1's. The 8 remaining density misses
(housing-authority budget, parkland fees, vacancy tax, tenant protections) are
codebook questions too, listed in
`output/validation/codebook_v2_fullasr/overcalls_and_misses.md`.

**Cost** (`docs/COSTS.md` A4): Sonnet $0.0407 per audio-h × 2,100 h = $85 at
the Batches rate ($171 direct); GPU without diarization $0.0102–0.0125/h =
$21–26; boot $3–5; re-runs 5 %. **$105 point, $130 planning, $195–200 if the
direct API is used.** `CLAUDE_CAP_USD` $80 has $61 of headroom, so the Sonnet
line alone needs a raise or phasing; RunPod has $3.43 left and needs the
top-up before any whole-meeting batch.

**Calendar constraint.** The whole audio of every YouTube meeting is needed,
not just the located clips: about 1,000 YouTube hours not yet on disk, one
yt-dlp request per meeting at the 12-meetings-per-day budget (the two 429s
this month came at 14–18 requests a day) is **≈ 25–30 days** of trickle from
this Mac, ~45 MB per meeting-hour. **Granicus bodies could go first**: their
audio comes over HLS / `ASX.php` with no YouTube budget, 237 meetings /
~704 h → Sonnet $29 + GPU $7–9 + boot ≈ **$45**, and Richmond / San Leandro /
Hayward already have Mac-ASR tracks whose adequacy for full-ASR locate is a
separate, unmeasured question.

**Prerequisites.** Puck's adjudication of the 11 windows; the #13 amendment
(item 2); a v2-coded reference sample with more transit minutes (the second
BART meeting 6.0 h and two Palo Alto meetings 11.6 h are on disk: ≈ $0.22 GPU
+ $0.9 Claude, `plan/RECALL-CHECK.md` "What more spend would buy"); RunPod
top-up; cap headroom. Never mid-measurement: v2 becomes the instrument only
from a clean start (`docs/MODELS.md`).

**Decision.** (a) Is active transportation and transit service planning
"transit" for this study (edit 1)? (b) Full-ASR for all bodies, Granicus
first, or stay with Option 1 (screen + measured recall as a thinning term)?

## 2. DECISIONS #13 / #14 amendments (pending)

`plan/DECISIONS-PROPOSED-13.md` has the two wordings. **Option 1**: keep
locate-first with a versioned keyword screen (`KEYWORDS_VERSION` on every
row), measure recall per route against the exhaustive reference, report it
beside every rate and enter it in the rate model as a measurement-error term;
reopen below 0.80. **Option 2**: whole-meeting ASR-only GPU pass as the
universal locate track (item 1 without dropping the screen gains no recall:
the screen's recall on GPU text was 0.826). The file recommended Option 1 on
2026-10-18 morning; the full-ASR arm measured later the same day is the case
for Option 2 with the screen dropped. #14 (codebook) is touched because
edits 1+2 of v2 are a new instrument and re-bill every window. Cost $0.
**Decision:** Option 1 or 2 (or neither); the 0.80 floor; whether recall
enters as a fixed thinning rate or a parameter with the bootstrap interval
as its prior.

## 3. A1 Claude pass

The lexicon A1 (VADER, hostility list) is run and reported as a
length-controlled appendix because VADER saturates with turn length
(`plan/A1-RESULTS.md`, `plan/WAVE1-RESULTS.md` §6). The Claude pass (per-turn
sentiment / frames with a pinned prompt) was never approved. **Cost** ~$100:
one Sonnet call per turn ≥ 5 words, k=1, ~45k turns × $0.00224. It overlaps
estimand 2's stance pass (item 4), which already returns frames and
hostility; running both is double-billing the same turns. **Prerequisite:**
corpus freeze; cap headroom. **Decision:** run it, fold it into item 4, or
keep the lexicon appendix.

## 4. Full stance pass (estimand 2)

Pilot done: 300 stratified turns × k=3, $2.01, self-consistency 97.8 % on
stance (`docs/REPORT.md` §4). The gate is **weighted κ ≥ 0.6 against 150
turns blind-coded by Puck** (`plan/SHOWCASE.md` 4a); the coding sheet exists,
the κ is pending. The S1 stratified sample was submitted by a sibling worker
at 16:32 UTC on 2026-10-18: 6,100 turns × k=3 in three Message Batches,
estimate $44.60, not yet collected, so its κ-independent coverage and its
bill are **PRELIMINARY** (refresh: `SELECT status, n_requests, est_usd,
cost_usd FROM stance_batches WHERE note LIKE 'stance-s1%'`).
**Cost** at the pilot rate $0.00224 per call: 30–45k turns = $67–101 at k=1,
$202–302 at k=3; the $100–150 quoted holds at k=1. **Prerequisites:** κ
result; corpus freeze; `CLAUDE_CAP_USD` headroom; role-gap sample capped at
8 h per body (`SHOWCASE.md` 1b). **Decision:** k (3 buys the spread
statistic, 1 buys coverage); the turn floor (≥ 5 words); whether S1 is enough
for the role-gap model or the full pass is wanted.

## 5. Public demo / showcase items not in the current phase

From `plan/SHOWCASE.md` §0 and `plan/SHOWCASE-EXEC.md`: the GitHub Pages
dashboard (`~/ytpt-site`: geo map, meeting → episode drill-down, episode
reader with redacted turns and timestamp deep links, D3 flowchart, model
figures), the lucashenry.xyz post (`docs/post-lucashenry.md` draft exists),
the public repo push and CI, resume bullets (`docs/RESUME.md` draft exists).
All $0. **Prerequisites:** Puck's 30-turn hand check of the redaction audit
(`output/redacted/`, DECISIONS #15), repo names under `hotsoupisgood`, the
"preview — not yet published" banner stays until then, corpus freeze and a
`make models` re-run so the figures are final. **Decision:** publish; repo
names; episode excerpts with timestamp links as the published form (§3
default).

## 6. The 8 unlocated bodies (was 10)

AC Transit, MTC/ABAG, Caltrain, Berkeley, Marin County BoS, Redwood City,
Windsor, Novato, Sonoma County BoS (`plan/T2-PLAN.md` §6,
`plan/WAVE1-RESULTS.md` §2): no live archive, or only commission views.

**2026-10-18: SFMTA and the SF Board of Supervisors came off this list.** Both
are on SFGovTV's Granicus at view ids the September pass never tried (it
stopped at 40): `sanfrancisco` view 10 (37 regular meetings, 116.9 h in window)
and `sfmta` view 55 (20 meetings, 80.6 h) — registered in `plan/channels.tsv`,
discovered into the state DB, and in scope for the full-ASR Granicus queue
(`plan/FULLASR-GRANICUS.md`), whose scope query is `kind='meeting' AND platform
!= 'youtube'` and so picked them up with no code change. Captions (`captions.vtt`)
and agenda index points are present on both. The regional-board side is now
BART, VTA, SamTrans **and SFMTA**, so the board comparison set is less thin.
Re-probing the remaining eight above view id 40 costs $0 and is the obvious
next move; any body found enters at the normal locate cost. The SF BoS
committees (Land Use and Transportation) sit on separate views and are
deliberately **not** registered — Puck's call.
**Decision:** accept the limitation as written in `docs/REPORT.md` §7, or
drop the remaining eight from `plan/channels.tsv` so the frame matches the data.

## 7. The two broken-audio Richmond meetings

`gr-richmond-6033` (2.15 h) and `gr-richmond-6038` (1.57 h): both Mac
mlx-whisper (0 located episodes, 1,118 words/h) and GPU WhisperX (b35,
decoder loop, 5 % coverage, 943 and 743 words) fail on them, so the audio or
Richmond's archive encode is the suspect, not the locate step
(`plan/RECALL-CHECK.md` "Two of six meetings failed"). They were the
zero-episode arm of the recall sample, so that arm never survived. **Cost:**
re-fetch via `ASX.php` → MP4 and re-transcode, then one no-diarization GPU
pass ≈ $0.05; listening to a minute of each is free. **Decision:** re-fetch
and retry, or mark them `dead` with a note and exclude them from Richmond's
exposure denominator.

## 8. Transit-operations vocabulary on operator bodies

On BART's board the exhaustive labeller called 115 of 117 about-minutes
`about_transit` (substation maintenance, service performance, fare gates),
and 46 of the 61 screen misses in the auto-caption arm are operations
vocabulary with no unconditional keyword ("train" is conditional). Whether
that is transit for the study is a definition, not a caption finding
(`plan/RECALL-CHECK.md` "A caveat specific to BART"). The homophone aliases
and the density-allocation terms (lottery, BMR, income tiers) are being
landed in the current phase by the re-screen worker; operations vocabulary
was not. Cost $0 to re-screen, cents to classify the new windows. **Decision:**
does the transit family on operator bodies mean "the agency's business" or
"transit policy as a city would debate it"?
