# Assessment: the two flagged BART episodes (`qIlAln3uR7U_e14`, `_e15`)

Written 2026-09-23. $0: no Claude call, no GPU, no RunPod, no YouTube request.
**This is a recommendation only. Nothing was re-transcribed, no batch was
proposed or launched, and no file was changed.** Puck decides.

Meeting: BART Board of Directors Regular Meeting, 2026-08-13
(`qIlAln3uR7U`, 21,454 s / 5 h 58 m, auto captions).

## Recommendation

**Accept both episodes without a caveat, and fix the QA gate instead.**

They are not damaged transcripts. Against the *correct* stretch of captions they
score **10.1 %** and **9.2 %** disagreement — among the best in the meeting. The
115.7 % and 771.1 % figures are a bug in `analysis/qa_transcripts.py`'s caption
windowing, triggered by a missing field in batch b13's transcript JSON. Nothing
about the audio, the decoder or the diarization is wrong.

Do **not** re-transcribe. There is nothing to fix on the GPU, and re-running
them would spend money to reproduce what is already correct.

---

## 1. What the flag actually was

`qa_transcripts.check_video` compares an episode clip against its meeting's
caption track, windowed by `meta["window_start_s"]`:

```python
w0 = float(meta.get("window_start_s") or 0.0)
cap = [(st - w0, t) for st, t in cap if w0 <= st < w0 + span]
```

**11 of the 15 BART episodes have `window_start_s = 0` in their transcript
JSON** even though they start hours into the meeting. With `w0 = 0` the
reference becomes "the first `span` seconds of the meeting" — so `_e14`, which
covers 11,790–15,450 s, was scored against the captions of 868–3,660 s. Two
unrelated 46-minute stretches of speech give an edit distance close to the
reference length, hence 115.7 %.

`_e15` (1,020 s span) got captions 868–1,020 s = **277 words**. `_e13` got the
same 277 words. So did `_e9`, a 2-minute episode. **The identical `ref_words`
across episodes of 2, 17 and 17 minutes is the fingerprint of the bug** — a
17-minute clip cannot have the same caption reference as a 2-minute one.
`_e15`'s 2,227 hypothesis words against those 277 give 7.71.

Re-scored with each episode's true start from `state.sqlite` (read-only):

| episode | span | `window_start_s` in JSON | true start | ref words @JSON | ref words @true | WER @JSON | **WER @true** |
|---|---|---|---|---|---|---|---|
| `_e1` | 242 s | 868 | 868 | 559 | 559 | 0.136 | 0.136 |
| `_e6` | 1200 s | **0** | 4890 | 785 | 2,739 | 3.094 | **0.098** |
| `_e9` | 117 s | 7291 | 7291 | 277 | 277 | 0.152 | 0.152 |
| `_e10` | 239 s | 7830 | 7830 | 672 | 672 | 0.083 | 0.083 |
| `_e12` | 121 s | 9690 | 9690 | 381 | 381 | 0.121 | 0.121 |
| `_e13` | 1020 s | **0** | 10650 | 277 | 2,413 | 8.148 | **0.085** |
| `_e14` | 3660 s | **0** | 11790 | 6,735 | 8,835 | 1.157 | **0.101** |
| `_e15` | 1020 s | **0** | 15630 | 277 | 2,344 | 7.711 | **0.092** |

Only `_e1`, `_e9`, `_e10` and `_e12` carry a real offset — and they are exactly
the four whose QA numbers looked sane. Every episode with `window_start_s = 0`
and a span under 868 s (the caption track's start) produced an *empty* reference
and silently got no caption metrics at all: `_e2`, `_e3`, `_e4`, `_e5`, `_e7`,
`_e8`, `_e11`. **Those seven were never caption-checked.** With the true offset
they score 0.066–0.115 — all fine.

The "15 minutes with >2× the captions' words at 0m, 1m, 2m, 3m, 4m, 5m" warning
is the same bug: constant across `_e6`, `_e13`, `_e14`, `_e15`, always the same
minutes, because it is measuring the same mis-aligned window every time.

## 2. Scope of the bug

**Batch b13 only.** Across all 129 `status='done'` episodes:

| batch | episodes | `window_start_s` matches `episodes.start_s` |
|---|---|---|
| b13 | 29 | **1 of 29** |
| b14+b15 | 17 | 17 of 17 |
| b16+b17 | 8 | 8 of 8 |
| b17 | 45 | 45 of 45 |
| b16+b17+b18 | 30 | 30 of 30 |

28 of 29 b13 episodes were written before the ingest recorded the clip offset.
b13 is the same batch whose episodes have no `speaker_roles` rows — the
"old ingest" batch, with both defects and no third.

So the caption check is unreliable for 28 episodes and correct for the other
101. It is not a transcription problem anywhere.

## 3. The long unknown-role turn is genuine speech

A0-RESULTS §4 flags three unknown-role clusters (1,414 / 986 / 967 words)
holding ~92 % of all unknown speech, the largest a single 475-second turn, and
asks whether it is a decoder loop.

It is not. `_e13` `SPEAKER_01`, 546 s of clip time, 1,442 words:

- **158 words/minute** — ordinary speech rate, not a loop.
- **Type-token ratio 0.382** (551 types / 1,442 tokens). A decoder loop has a
  TTR near zero.
- **No repeated 8-gram occurs more than twice** in the whole turn.
- **7.1 % disagreement against the captions for the same 546 seconds**
  (1,467 caption words vs 1,442 transcript words).
- The content is a BART staff presentation on an equipment procurement item,
  ending "…authorized to enter into an agreement … in an amount not to exceed
  22,091,052. Thank you." The captions say the same thing.

This is exactly what A0-RESULTS guessed: **a staff presentation the cue
labeller could not place.** The failure is in `pipeline/roles.py`, which has
`dais` / `public` / `unknown` and no cue for "staff presenter" — a person who
opens with neither the dais's procedural formula nor the public's "my name is".

## 4. Audio: clean

| | `_e13` | `_e14` | `_e15` |
|---|---|---|---|
| opus duration | 1020.01 s | 3660.01 s | 1020.01 s |
| episode window (`end_s − start_s`) | 1020.02 s | 3660.02 s | 1020.02 s |
| mean volume | −17.9 dB | −17.4 dB | −18.5 dB |
| max volume | 0.0 dB | 0.0 dB | 0.0 dB |
| format | 48 kHz mono | 48 kHz mono | 48 kHz mono |

Durations match the cut windows to 0.01 s, so **there is no duplicated audio**.
Levels are normal. The caption track itself is complete: it runs 868 → 21,374 s
against a 21,454 s video, so it does not stop early either.

## 5. Ruling out the alternatives asked for

| hypothesis | verdict | evidence |
|---|---|---|
| decoder loop | **no** | TTR 0.382, no 8-gram repeats, 158 wpm, §3 |
| duplicated audio | **no** | opus duration == cut window to 0.01 s, §4 |
| caption track stops early | **no** | captions cover 868–21,374 s of a 21,454 s video |
| genuine long staff presentation the labeller could not place | **yes** | §3 |
| QA windowing bug | **yes, and it is the whole flag** | §1, §2 |

## 6. Consequences for A0

A0-RESULTS §4 says "treat any number from `qIlAln3uR7U_e13/e14/e15` as
provisional". **That caveat can be lifted.** These are three of the better
transcripts in the pilot. They are also large — `_e14` alone is 8,422 words,
12 % of the pilot — so lifting it matters.

The separate A0 point stands unchanged: **3,655 unknown-role words are excluded
from the frame heat map**, so ~5 % of the pilot is silently dropped there. That
is a role-labelling gap, not a transcription one, and it is now known to be
real speech rather than noise.

## 7. What I would fix, if Puck wants it ($0, no GPU)

Not done — listed so the decision is a decision:

1. **`analysis/qa_transcripts.py`**: when `window_start_s` is 0/absent and the
   file is an episode, take the offset from `state.sqlite episodes.start_s`
   rather than assuming 0. Or refuse to emit a caption metric at all rather than
   emitting a wrong one — the current behaviour produced a 771 % number that
   looked like a finding.
2. **Re-run `make qa`** afterwards. 28 b13 episodes get their first honest
   caption check, and the 36 WARN count will change. Costs nothing and touches
   no audio.
3. **`pipeline/roles.py`**: add a staff-presenter cue ("staff recommends",
   "this item before you", "I'll walk you through") so 475-second presentations
   stop landing in `unknown`. This is the fix that recovers 5 % of the pilot for
   the frame analysis.

None of these needs a GPU, a batch, or a network call.

---

## 8. Closing note (2026-10-16, branch `showcase/qa-window-fix`)

Puck approved the fix as showcase step 3 (`plan/SHOWCASE.md` row 3). Item 1 of
§7 is done, item 2 is run, item 3 (`roles.py` staff-presenter cue) is **not**
touched here.

**What changed in `analysis/qa_transcripts.py`.** For an episode, a
`window_start_s` of 0 or missing is read as *unknown*, never as "the meeting's
first `span` seconds". `episode_window_start()` takes the offset from, in order,
`window_start_s` (> 0), `episode_start_s` in the JSON, or `episodes.start_s` in
`state.sqlite` for that `episode_id` (new `--state-db` flag, default
`STATE_DB_PATH` as `pipeline/state.py` resolves it; `none` disables). The DB row
is authoritative even at ≤ 0 (`9FM5406YagE_e1` is padded off the meeting's
start). With no source at all the caption block is skipped: `wer_vs_captions`
stays **empty** in the CSV, the row carries an `info:` note saying why, and the
status is not changed. Shows (`window_start_s` legitimately 0) are untouched.
Two new CSV columns: `window_source`, `window_start_s_used`.

**Tests.** Seven regression checks in `analysis/test_pipeline.py` section 23
(a synthetic meeting whose first 120 s of captions are unrelated speech, a clip
at 600 s with `window_start_s = 0`, an `episodes` row at 600.0, a row at −0.02,
a clip with no row, the `episode_start_s` fallback, an absent DB, and the CLI
through to the CSV). `make test`: **427 checks, all passed**. The fixture also
asserts the wrong window really would have scored > 90 %, so it is not a no-op.

**`make qa` re-run, 552 transcripts, offline** (before = `output/qa/latest_prefix_2026-10-16.csv`):

| | before | after |
|---|---|---|
| OK / WARN / FAIL | 456 / 96 / 0 | **472 / 80 / 0** |
| `qIlAln3uR7U_e14` | WARN, 115.7 % | WARN, **10.1 %** (`state_db`, 11,790 s) — the remaining WARN is its 6 speaker-tag segments, a separate, real finding |
| `qIlAln3uR7U_e15` | WARN, 771.1 % | OK, **9.2 %** (`state_db`, 15,630 s) |
| episodes re-windowed from the DB | — | 35, now 5.3–33.9 % |
| episodes caption-checked for the first time | — | 10 (previously an empty reference, silently) |
| rows WARN → OK | — | 16 (all b13; every one was the mis-aligned window) |
| no offset source at all | — | 5: `Chj4lSP9je4_e1–e5` (Sunnyvale, b13, no `episodes` row). Were 562–1,599 %; now no number, `info` note |

The §1 table is confirmed to the decimal by the re-run (`_e6` 9.8 %, `_e13`
8.5 %, `_e14` 10.1 %, `_e15` 9.2 %). The A0 caveat on `_e13/e14/e15` can be
lifted. Nothing under `output/` was deleted, nothing was re-transcribed, and
no network, GPU or Claude call was made.

Open: the five Sunnyvale clips have no `episodes` row, so they stay unmeasured
until someone records their offsets; and §7 item 3 (`roles.py`) is still a
proposal.
