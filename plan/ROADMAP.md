# Roadmap — all plans in one place (2026-09-21)

Owner: Puck. Supervisor keeps this current; each row links to the full plan.

| Phase | What | Status | Gate / needs Puck | Cost | Doc |
|---|---|---|---|---|---|
| Polarisation corpus | 143 Hasan/Tucker videos transcribed, host-enrolled, QA 0 FAIL | **frozen side corpus** | — | $2.92 spent | `START_HERE.md`, `SAMPLING.md` |
| Pivot | Bay Area density + transit meetings, 12-month window, captions-first | **decided** | — | — | `PROPOSAL-transit.md` |
| T0 | 8 measurement tests on 9 YouTube meetings + 3 Granicus probes | **done** except test 3 (Puck's 100-row blind label sheet) | `output/t0/label_sheet.csv` | $1.37 | `T0-TESTPLAN.md`, `T0-RESULTS.md` |
| T1 | meeting pipeline built; pilot 12 meetings / 4 bodies → 46 episodes transcribed with roles | **done** (b15 passed) | approve DECISIONS #12–#16 | $0.19 | `PROPOSAL-transit.md` §3, `SCHEMA.md` |
| T2 part 1 | everything the full run needs: `gpu_batch.py reconcile` (M7) + Bearer auth (L1), spend ledger with hard stops (`pipeline/spend.py`), classifier `--model` / Batches API / `(id, prompt_version)` key (L2), Haiku 4.5 validated against the T0 Sonnet labels (0.758 → **Sonnet via Batches**), ingest tweaks (speaker tags, per-body vocab, QA vs meeting captions), sources for all located bodies + discovery for the 12-month window | **done 2026-09-22** (part 2 needs Puck's approvals) | approve DECISIONS #17–#19; drop or keep the unlocated bodies | Claude $0.52 (validation), RunPod $0 | `T2-PLAN.md` |
| T2 part 2 | captions → classify (Sonnet, Batches) → cut → GPU, per-body cap 3 episode-h/month. **Batch threshold 50 h → 15 h** (2026-09-22) | **running**: caption trickle worker **down** on a bot check (≈ 326 meetings left, untouched since 2026-09-22); Granicus located four ways — classifier (English), classifier (**Spanish**, Napa **now complete**, §3g), `method='agenda_index'`, `method='local_asr'` (Richmond + San Leandro running, **Hayward 27 meetings / 62.9 audio-h queued behind it**, plus the 3 large coverage re-reads); b17 **done and reconciled** (53 usable, 30 QA FAIL, cause found — §3g); **b18 awaits approval**: the same 30 Fremont clips re-encoded, 3.25 episode-h, est $0.18 (≈ $0.07 at the measured rate); the agenda_index cut run still going — 34 meetings / 105 located episodes / 30.4 h left to cut | **`approve b18`**; **resume the caption worker after the bot-check backoff (Puck's call, rule 2)**; **`NORMALISE_BELOW_DBFS` −30 dB normalises ~85 % of future clips — keep, or set −45?** (§3g); **Petaluma `UEV2mj4Qf-s` re-read is queued but blocked on the same backoff** | Claude $5.57, RunPod $3.62 | `T2-PLAN.md` §3b–3g |
| T2 caption audit | `scripts/caption_audit.py`: words the classifier sees ÷ meeting minutes, + coverage + language, per meeting and per body | **done 2026-09-23** — 622 meetings, 168 tracks; no truncation epidemic (medians 93–162 wpm); Hayward's artefact is 26 meetings with **no track**, not 9 % coverage; Santa Clara healthy and untouched. `MIN_COVERAGE` 0.50 → **0.95** (§3f): coverage is bimodal (median 1.000, p10 0.993), so the tighter line flags **5 of 168 (3.0 %)** for $0 and catches a track that stops early while reading a normal wpm | Petaluma decision above | $0 | `T2-PLAN.md` §3e–3f, `output/analysis/caption_audit.csv` |
| T2 Napa (Spanish) | `plan/keywords_density_es.txt` + `keywords_transit_es.txt`; stage 1 picks its keyword files from the track's own language (`classify.caption_language` — no new column); codebook unchanged, Spanish windows recorded under `prompt_version .../es` so the two recalls are never pooled | **done 2026-09-23** — the English pair found **2** candidate windows over 104 h, the Spanish pair finds **270**; **18 of 20 meetings classified, 32 located episodes / 194 core-min** (21 density, 9 transit, 2 both) where the English route had 0. The last 2 (`gr-napa-5713`, `gr-napa-5724`) are caption-audit failures at 71 % and 8.7 % coverage and are being located from audio instead, under `method='local_asr'` — not pooled with these | — | Claude $0.61 ($0.39 + **$0.2162** actual for the second half) | `T2-PLAN.md` §3f–3g |
| A0–A2 | exploratory plots only: rates, term frequency per speaker (role × body), sentiment (lexicon, then Claude); no models yet | **planned** | Puck: start A0 | ≤ $5 pilot, $15–45 full | `ANALYSIS-transit.md` |

Measured so far (T0/T1): episode share ~19 % of council time (16 % density,
7 % transit), 40 % at BART; classifier $0.08/meeting; GPU $0.02/audio-h at
25× on 3–60-min clips; whole-meeting audio ~50 s per meeting; roles 81–94 %
of speech time correct; AC Transit / MTC Granicus archives are dead — current
video elsewhere. Granicus locate routes, measured 2026-09-22: **captions** on
Oakland, Fremont, Santa Clara, Hayward (the last two behind a Flowplayer
config the parser had not read) and Napa (Spanish only); **agenda index
points** on Santa Clara, Napa, Sausalito, Hayward and 2 of 12 SamTrans — 3.0 %
of 2,066 agenda items match a keyword family; **neither** on Richmond and San
Leandro, whose audio is reachable only through `ASX.php`. Mac ASR
(`mlx-whisper` turbo) runs at **11.6× realtime**. Granicus HLS slices
(2026-09-23, 53 clips): wall clock **0.33× the slice's own length** at the
median, p90 1.31×, worst 2.25×, ~1 MB per clip-minute; a 30 s slice still
costs ~175 s because the manifest walk and seek are fixed cost. Classifier
tokens: **2.04 input tokens per English caption word, 2.74 per Spanish one**
(English ranges 1.69–2.24 across four batches, so it is a mid estimate, not a
bound); the codebook is 1,531 tokens and its cache only takes on a batch of a
few hundred requests or more — and even at 455 windows only 76 % of it is served
from cache. GPU, actual: **$0.0211/audio-h** on b17, $0.0229 on b13.
Audio: Fremont's Granicus MP4s are polarity-inverted stereo and cancel under an
`-ac 1` downmix; every other body's are dual-mono.

b17 (2026-09-23), corrected twice: 83 clips submitted, **53 usable**; all **30
Fremont clips failed QA at 1–26 % coverage**. Not silence, and not level either
— **our own downmix**. Fremont's Granicus MP4s are polarity-inverted stereo
(L = −R), so `-ac 1`, which averages the channels, cancels the speech: each
channel alone reads a healthy −42 dB with −7 dB peaks, the average reads −78 dB.
**32 of 32 Fremont clips lose 32–38 dB that way; 51 of 51 clips from the other
six bodies lose 0.0 dB.** The −20.9 dB figure for the passing set does not
reproduce — measured, they sit at −36.1 dB in the opus, and 45 of 53 of them are
below −30 dB, so level alone never separated the two groups. `transcode.py` now
measures before it encodes and takes ONE channel when the sum cancels, then
applies loudnorm (−16 LUFS) below `NORMALISE_BELOW_DBFS`; both, and the pre/post
levels, are recorded in `raw_meta.audio_levels`. The 30 clips re-encoded from
**−76.2 dB to −18.8 dB** median and are batch **b18** (3.25 episode-h, est
$0.18, ≈ $0.07 at b17's measured $0.0211/audio-h), awaiting Puck's approval.

`estimate_usd`: there are two, wrong in opposite directions. The **classifier**
one was low — it priced `words * 1.35` tokens and credited a codebook cache read
on every call; measured, it is **2.04 tokens per English caption word, 2.74 per
Spanish**, the codebook is 1,531 tokens, and the cache serves only 28 % of it at
93 windows against 76 % at 455, so below ~300 windows it does not pay off. Over
the five billed batches the estimate went from −50 %…+12 % to +1 %…+50 %, never
under the bill. `worst_case_usd`, the rail the spend cap is checked against, had
the same defect and is fixed. The **GPU** one is not low at all: it runs +68 % on
b17, +72 % on b13, +107 % on b15, by design (7-min boot × 1.5 contingency), and
was left alone.

YouTube captions stopped 2026-09-23 04:12 UTC on a **bot check**
(`hZ04_fO0htU`); 24 h backoff, `com.ytpt.captions` down, 326 meetings still
uncaptioned. Rule 2: it stays down until Puck says resume.

Cotati: 3 meetings discovered (2026-01-27 … 2026-03-24), none since, while every
other body is current to early September — likely the `City Council Meeting$`
title regex or a channel move. Not probed: no YouTube requests while the
bot-check holds.

Standing rules: one gate on GPU spend (Puck's texted `approve bN`); launches
run detached; YouTube from this Mac only, sequential, stop on 429/bot-check;
no names extracted from public comment.
