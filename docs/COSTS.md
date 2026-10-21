# COSTS — how the estimates are made, how the money piled up, what finishing costs

Written 2026-10-18 from `~/.local/share/ytpt/state.sqlite` (tables `spend`,
`pods`, `batches`, `classifier_batches`, `classifier_calls`, `stance_calls`,
`stance_batches`), `pipeline/gpu_batch.py`, `pipeline/classify.py`,
`pipeline/stance.py`, `pipeline/spend.py` and the plan documents named inline.
Every number carries its source. Numbers marked **PRELIMINARY** will move
because two workers were still running when this was written (the 120-s /
density-vocabulary re-screen and the S1 stance sample); a `refresh` SQL
snippet sits beside each so it can be updated. Read-only: nothing here changes
a cap or a decision.

Ledger totals at writing (`pipeline/spend.py --json`, 2026-10-18 16:30 UTC):
RunPod **$6.57** recorded of the $10.00 credit (stop line $9.50, $3.43 left);
Claude **$19.15** of `CLAUDE_CAP_USD` $80.

**Update 2026-10-18 20:50 UTC (S1 stance pass collected):** Claude ledger
**$63.05** = `classifier_calls` $25.57 + `stance_calls` $37.48 of $80
($16.95 headroom). The S1 lines below (A3, B3, C) now carry actuals; the
classifier rows were not refreshed in this update (the 120-s re-screen landed
in between; use the refresh SQL).

**Corrected 2026-10-18 after validation (`plan/VALIDATION-2026-10-18.md` §7):
both lines above are stale and the caps they quote are no longer the live
ones. Read this block, not them.**

| | value | source |
|---|---|---|
| RunPod recorded | **$8.4337** (22 `spend` rows) | 2026-10-19T01:42:54Z snapshot |
| Claude recorded | **$69.4660** = `classifier_calls` **$31.9897** + `stance_calls` **$37.4763** (31 `classifier_batches`, 28,074 stance calls) | same snapshot |
| RunPod credit / stop line | **$31.50** / **$31.00** | `pipeline/spend.py --json`, live (the authority per `AGENTS.md`) |
| `CLAUDE_CAP_USD` | **$150.00** | same |
| position | $8.43 of $31.50 (**27 %**) and $69.47 of $150 (**46 %**); `runpod_blocked` and `claude_blocked` both false | same |

So **the 80 %-of-cap escalation is not triggered**, and the "$3.43 left" and
"$16.95 headroom" readings above are wrong in opposite directions from the
truth. The two total deltas are staleness, not miscomputation — every
*as-of-stamped* line below ties out exactly. The totals omit pods b39
($0.6533) and b40 ($1.2071), **$1.86** of RunPod, and 15 classifier batches
($9.1239), **$6.42** of Claude, all booked after the last stamp. Minor: pod
`6a8i5phufeukie` is attributed to b5 in §C below; `batches` says **b6** (no
dollar impact).

RunPod keeps moving while sibling workers launch pods: at 2026-10-19T01:55Z
`spend.py --json` already reads **$10.583** recorded, $20.92 left. Run the
refresh SQL rather than quoting any total on this page.

```sql
-- refresh: both ledger totals
SELECT (SELECT ROUND(SUM(usd),4) FROM spend) runpod,
       (SELECT ROUND(SUM(cost_usd),4) FROM classifier_calls)
     + (SELECT ROUND(SUM(cost_usd),4) FROM stance_calls) claude;
```

---

## A. How estimates are made

### A1. GPU batch estimate (`pipeline/gpu_batch.py estimate_usd`)

```
est_usd = (audio_hours / RTFX + boot_min / 60) × usd_per_hour × contingency
```

| input | value | where it comes from |
|---|---|---|
| `GPU_RTFX_ESTIMATE` | 25× | code default, "an ESTIMATE until Phase 0 measures it" (`gpu_batch.py` docstring, `PLAN.md` 5) |
| `GPU_BOOT_MINUTES` | 7 min | code default |
| `GPU_USD_PER_HOUR` | $0.49/h | `.env` (A40 SECURE, `plan/RECALL-CHECK.md` "Method" 2); code default is $0.34 (RTX 4090 community). Back-derived from `spend`: pod `9753hkj9hg3jdu` 0.472 GPU-h → $0.2314 = $0.49/h; the first four pods (b3–b8, 2026-09-18/19) billed $0.17/h |
| `GPU_COST_CONTINGENCY` | 1.5 | code default |

The estimate is frozen on the batch row (`batches.est_usd`) so a later measured
RTFX does not rewrite history. Measured RTFX, from `batches.note` ("measured
RTFX … ASR+align+diarize, excl. boot/deps"):

| configuration | batches | measured RTFX | $/audio-h actual |
|---|---|---|---|
| WhisperX large-v3 **with** pyannote diarization | b7 23.7, b8 24.2, b9 25.0, b11 24.9, b13 25.9, b15 24.9, b17 26.2, b18 25.2, b22 25.3, b31 25.0, b32 28.2, b33 29.3, **b34 25.1**, b37 24.7 | 24–29×, median 25.1× | transit batches b13–b37: $3.098 / 143.2 h = **$0.0216** |
| **without** diarization (validation arms) | **b35 58.6×** (22.64 h, $0.2313), **b38 56.2×** (15.0 h, $0.187) | 56–59× | **$0.0102** (b35) – **$0.0125** (b38) |

So the 25× guess was right for the diarized job and the estimator is ~1.6× the
bill there (the 1.5 contingency plus boot overhead); for a no-diarization job
it over-predicts ~3× because it still assumes 25×.

```sql
-- refresh: measured RTFX and $/audio-h per done batch
SELECT batch_id, audio_hours, est_usd, actual_usd, ROUND(actual_usd/audio_hours,4) usd_per_h,
       substr(note, instr(note,'measured RTFX'), 20) rtfx
FROM batches WHERE status='done' ORDER BY created_at;
```

### A2. Classifier estimate (`pipeline/classify.py estimate_usd`)

```
est_usd = [ caption_tokens × $3.00/M
          + n_windows × ( codebook_tokens × codebook_rate + 40 × $15.00/M ) ]
          × 0.5 if Message Batches
```

| input | value | source |
|---|---|---|
| tokens per caption word | 2.04 English, 2.74 Spanish | `classify.TOKENS_PER_WORD`, measured over 1,044 billed windows |
| tokens per padded word on WhisperX text | **1.34** | `plan/CODEBOOK-V2-PILOT-FULLASR.md` "Cost": 306,577 input tokens / 228,528 words |
| codebook tokens | 1,531 (v1 `t0-a4a8e55760`), ~2,014 (v2 `v2-ed4c72c9d6`) | `classify.py` comment; FULLASR 1,905 cache-read per call |
| `codebook_rate` | plain input $3.00/M under 300 windows; above, 0.75 × cache-read $0.30 + 0.25 × cache-write $3.75 = $1.16/M | `CACHE_PAYS_OFF_WINDOWS`, `CACHE_HIT_SHARE` |
| output per call | 40 tokens | `OUTPUT_TOKENS_PER_CALL`, measured 39.0–39.9 |
| Sonnet 4.6 prices per Mtok | input $3, output $15, cache write $3.75, cache read $0.30 | `classify.PRICES` |
| Batches discount | 0.5 | `BATCH_DISCOUNT` |
| worst case | the direct-API figure: no discount, codebook billed as a cache write (or plain input under Haiku's 4,096-token cache minimum) | `worst_case_usd`; `plan/T2-PLAN.md` §2 for the Haiku case |

Observed: the batch estimate ran 0.89× the bill on corpus batches (one Spanish
batch under-estimated 2×, see B2), 0.71× on the validation arms and 0.49× on
the pilots, because large warm-cache batches hit the cache on 100 % of calls
and WhisperX text bills 1.34, not 2.04, tokens per word.

### A3. Stance estimate (`pipeline/stance.py estimate_usd`)

Same shape as A2 (prompt words × tokens/word × input price + per call codebook
× cache-blend rate + output × price, × 0.5 batched). Pilot measurement,
`stance_calls` / `stance_batches`:

| | value |
|---|---|
| turns × k | 300 × 3 = 900 calls (10 smoke calls direct, 890 via one Message Batch `msgbatch_014Ycbj5NPxkDh8gERwmTgdh`) |
| estimate (batch) | $2.204 |
| **actual** | **$2.0142** ($0.0338 smoke + $1.9804 batch); ratio 0.90 |
| per call (batched) | $0.00224; per turn at k=3 $0.0067 |
| tokens | 280,934 input, 1,818,852 cache-read, 54,609 output for the 890 batched calls |

S1 pass measurement (collected 2026-10-18 16:40–16:56 UTC, same
prompt_version `st-7161e26ad2/claude-sonnet-4-6`, k = 3, six Message Batches
tagged `stance-s1`; `plan/STANCE-S1-RESULTS.md` §3):

| | value |
|---|---|
| turns × k | 9,058 new turns × 3 = **27,174 calls**, 0 errored, 0 unparseable (+ the 300 pilot turns reused: 9,358 turns in `turn_stance`) |
| estimate (batch, `stance.py`) | $66.16 over the six submissions ($15.00 + $14.98 + $14.63 + $7.15 + $7.35 + $7.05) |
| **actual** | **$35.46** ($8.52 + $7.90 + $7.72 + $3.70 + $3.90 + $3.73); ratio **0.54** |
| per call / per turn | **$0.001305** per call; **$0.00391** per turn at k = 3 |
| why under the pilot's $0.00224 per call | cache-read hit 98 % of calls (364,798 cache-write tokens over 27,174 calls vs 467,558 over 890 in the pilot): 7.98 M input, 69.4 M cache-read, 0.36 M cache-write, 1.65 M output tokens |

```sql
-- refresh: stance spend by prompt version and batch
SELECT prompt_version, batch_id IS NOT NULL batched, COUNT(*) calls,
       COUNT(DISTINCT episode_id||'#'||turn_idx) turns, ROUND(SUM(cost_usd),4) usd
FROM stance_calls GROUP BY 1,2;
```

### A4. Full-ASR locate projection (`plan/CODEBOOK-V2-PILOT-FULLASR.md`, `plan/RECALL-CHECK.md`)

| line | derivation | source |
|---|---|---|
| Sonnet, every 120-s window, codebook v2, Batches | $0.7696 / 18.91 audio-h = **$0.0407/h** × 2,100 h = **$85** | `classifier_batches` `msgbatch_01BBtcuGSe1yz…` (549 calls); FULLASR "Cost" |
| same, direct API (worst case) | × 2 = **$171** | A2 `BATCH_DISCOUNT` |
| GPU, whole-meeting WhisperX, no diarization | $0.0102/h (b35) – $0.0125/h (b38) × 2,100 h = **$21–26** | `batches` b35, b38 |
| pod boot | 7 min × $0.49/h = $0.057 per pod; 50–90 pods of 25–40 audio-h = **$3–5** | A1 |
| re-runs (QA FAIL, CUDA retries) | 5 % of Sonnet + GPU ≈ **$5** | b32/b33 lost 2 clips each; b35 lost 2 of 6 meetings |
| **point** | 85 + 21 = **$106 ≈ $105** | |
| **planning** | (85 + 26 + 5) × 1.05 = $122, rounded to **$130** | |
| **if direct API** | 171 + 21–26 + 3–5 = **$195–200** (quoted to Puck as $170–190; that figure took the low GPU line and no boot) | |

The 2,100 h is the corpus: 614 regular meetings / 2,099 meeting-h
(`plan/WAVE1-RESULTS.md` §3; `plan/T2-PLAN.md` says 2,284 h before dedup).
Nothing here is spent; see `docs/FUTURE-STEPS.md` item 1.

---

## B. How the costs piled up

### B1. RunPod ledger — every pod (`spend` joined to `batches`), $6.5736 total

| date (UTC) | pod | batch | purpose | audio-h | est | **actual** | what happened |
|---|---|---|---|---|---|---|---|
| 09-18 | w11po2erjc0lco | b3 | Hasan/Tucker | 1.92 | 0.05 | 0.0678 | 0/4 done, 4 pod FAIL (first real pod) |
| 09-18 | 6a8i5phufeukie | b5 | Hasan/Tucker | 1.92 | 0.21 | 0.0193 | launch failed on machine resources, 3-min pod |
| 09-18 | 8f43htpkwpx63m | b7 | Hasan/Tucker | 1.92 | 0.14 | 0.0754 | 4/4, RTFX 23.7× |
| 09-19 | eehi1xv3qdwntv | b8 | Hasan/Tucker | 1.92 | 0.14 | 0.0792 | 4/4, RTFX 24.2× |
| 09-20 | vjbkg7uh7vnpp1 | b9 | Hasan/Tucker | 50.67 | 1.58 | 1.0599 | 93/95, 2 QA FAIL, RTFX 25.0× |
| 09-20 | 6rboohe9ucl7kw | b10 | **orphan pod** | 23.95 | 0.79 | **1.1198** | driver killed by exec host 15 s after launch; pod idle 137 min; terminated by Puck in console; 0 transcripts |
| 09-20 | ohxlkkykm3plzk | b11 | Hasan/Tucker | 23.95 | 0.79 | 0.5016 | 45/48, 1 QA FAIL |
| 09-21 | emvwdu5zznf65q | b13 | transit corpus | 8.86 | 0.35 | 0.2036 | T1 pilot, 40/40 |
| 09-22 | vuv50kd22m16xh | b14 | transit (failed) | 1.70 | 0.14 | 0.1241 | no SSH after 900 s, community pod without public IP |
| 09-22 | 3wyew6wp075bnu | b15 | transit corpus | 1.70 | 0.14 | 0.0681 | 17/17 |
| 09-23 | s6o71k8pczjsvz | b17 | transit corpus | 14.11 | 0.50 | 0.2994 | 53/83, **30 QA FAIL** |
| 09-24 | z67br6cz6vte0x | b18 | transit corpus | 3.25 | 0.18 | 0.1007 | 30/30 |
| 09-24 | umxqgdkpk029ol | b22 | transit corpus | 37.64 | 1.19 | 0.7715 | 128/128 |
| 09-28 | ec7spjzt0hmu8s | b31 | transit corpus | 22.35 | 0.74 | 0.5024 | 141/143, after 7 supply-constraint failures b24–b30 |
| 10-17 | c6j7m3j7pmijy8 | b32 | transit corpus | 15.08 | 0.53 | 0.2948 | 92/94, 2 CUDA pod FAIL |
| 10-17 | sy3dezgu54jhd4 | b33 | transit corpus | 16.26 | 0.56 | 0.3156 | 86/88, 2 CUDA pod FAIL |
| 10-18 | 28byhvbqw5wh2g | b34 | transit corpus | 8.01 | 0.32 | 0.1945 | 46/46, RTFX 25.1× |
| 10-18 | 9753hkj9hg3jdu | b35 | **validation `recall`** | 22.64 | 0.75 | 0.2314 | whole-meeting, no diarization, RTFX 58.6×; 2 of 6 QA FAIL (Richmond audio) |
| 10-18 | h8efj4c8a2z7wm | b37 | transit corpus | 15.98 | 0.56 | 0.3573 | 92/92 |
| 10-18 | hce2tpbqsje6id | b38 | **validation `recallyt`** | 15.00 | 0.53 | 0.187 | no diarization, RTFX 56.2× |

Never billed: b1, b2, b4, b6, b16, b20, b21, b24–b30 (RunPod supply
constraint or SSH before a pod existed); denied: b12, b19, b23, b36.

| RunPod by purpose | pods | audio-h | est | **actual** | actual / est |
|---|---|---|---|---|---|
| Hasan/Tucker side corpus (b3–b11 excl. b10) | 6 | 80.4 | 2.70 (done only) | **1.8032** | 0.66 |
| orphan pod b10 | 1 | — | 0.79 | **1.1198** | — (idle billing, not an estimate miss) |
| transit corpus, done (b13–b37) | 10 | 143.2 | 5.07 | **3.0979** | 0.61 |
| transit, failed launch b14 | 1 | — | 0.14 | **0.1241** | — |
| validation arms (b35 `recall`, b38 `recallyt`) | 2 | 37.6 | 1.28 | **0.4184** | 0.33 (no-diarization job, 25× assumed) |
| **total** | 20 | | | **6.5734** | |

```sql
-- refresh: RunPod ledger with batch est vs actual
SELECT s.started_at, s.pod_id, b.batch_id, b.audio_hours, b.est_usd, s.usd, b.approved_via, substr(b.note,1,60)
FROM spend s LEFT JOIN batches b ON b.pod_id = s.pod_id ORDER BY s.started_at;
```

### B2. Claude ledger — $19.1461 total (`classifier_calls` $17.13 + `stance_calls` $2.01)

Corpus classifier by day (`classifier_calls`, rows not tagged
`recall*`/`cbv2`): 09-21 $1.16 (T0/T1 direct calls) · 09-22 $1.72 (incl. the
Haiku 4.5 test, 198 calls, **$0.5224**, not adopted) · 09-23 $2.37 · 09-24 $2.00
· 09-25 $0.26 · 10-17 $0.77 · **10-18 $4.87** (the 1,596-window fill
`msgbatch_01NE3ditTjdC2…` $4.10 and the keyword-expansion re-read $0.75).

Message Batches (`classifier_batches`, 16 rows; `pv` = prompt_version suffix):

| date | batch id (prefix) | purpose | pv | n | est | **actual** |
|---|---|---|---|---|---|---|
| 09-22 | 018NeiFpS6Y32 | corpus | v1 | 192 | 0.3515 | 0.6429 |
| 09-22 | 01F7rADU4TepF | corpus | v1 | 306 | 0.6166 | 0.5518 |
| 09-23 | 01472Yu1z6nHv | corpus | v1 | 455 | 0.8232 | 1.1050 |
| 09-23 | 01M5ELEadc9dr | corpus, Spanish | v1/es | 93 | 0.1939 | 0.3901 |
| 09-24 | 01RzosaEWXQQE | corpus, Spanish | v1/es | 59 | 0.3360 | 0.2162 |
| 09-24 | 01HwPvmjTfwNj | corpus, Swagit | v1/swagit | 318 | 1.0593 | 0.9015 |
| 10-17 | 015K1PynxMFHG | corpus | v1 | 181 | 0.8013 | 0.7736 |
| 10-18 | 01NE3ditTjdC2 | corpus fill | v1 | 1,596 | 5.1752 | 4.0985 |
| 10-18 | 01BEqiS6hQ1D3 | **validation `recall`** (b35 text) | v1 | 1,115 | 2.2671 | 1.8737 |
| 10-18 | 01K7KBCSGpcQs | corpus, kw-expansion re-read | v1, kw-24e8a4acd1 | 345 | 1.0747 | 0.6122 |
| 10-18 | 01UeiXiTXqgYv | corpus, kw-expansion Swagit | v1/swagit | 42 | 0.1945 | 0.1421 |
| 10-18 | 01WKezhAWcBG5 | **pilot `cbv2`** arm v1_120 | v1/mr120 | 111 | 0.4231 | 0.2177 |
| 10-18 | 01CqJqDutTjwi | **validation `recallyt`** (b38 text) | v1 | 709 | 1.4384 | 0.7659 |
| 10-18 | 014s6qJ4L3qgY | **pilot `cbv2`** arm v2_120 | v2/mr120 | 111 | 0.5033 | 0.2444 |
| 10-18 | 01UdWeKQ5cDGF | **pilot `cbv2`** arm v2_240 | v2 | 60 | 0.3118 | 0.1068 |
| 10-18 | 01BBtcuGSe1yz | **pilot `recallv2`** full-ASR | v2/mr120 | 549 | 1.5067 | 0.7696 |

| Claude by purpose | calls | est (batches) | **actual** | actual / est |
|---|---|---|---|---|
| corpus classifier, Sonnet 4.6 (direct + batches) | 4,434 | 10.63 (batched part) | **12.6312** | 0.89 |
| Haiku 4.5 classifier test (rejected) | 198 | — | **0.5224** | — |
| validation `recall` (Granicus / local_asr arm) | 1,115 | 2.27 | **1.8738** | 0.83 |
| validation `recallyt` (YouTube auto-caption arm) | 709 | 1.44 | **0.7660** | 0.53 |
| pilot `cbv2` (codebook v2, 3 arms) | 282 | 1.24 | **0.5689** | 0.46 |
| pilot `recallv2` (v2 + full ASR) | 549 | 1.51 | **0.7696** | 0.51 |
| stance pilot (300 turns × k=3) | 900 | 2.20 | **2.0142** | 0.90 |
| **total** | 8,187 | | **19.1461** | |
| *added 2026-10-18 16:56 UTC:* S1 stance pass (9,058 turns × k=3, six batches) | 27,174 | 66.16 | **35.4620** | 0.54 |

```sql
-- refresh: Claude by purpose
SELECT CASE WHEN video_id LIKE 'recall:%' THEN 'recall' WHEN video_id LIKE 'recallyt:%' THEN 'recallyt'
            WHEN video_id LIKE 'recallv2:%' THEN 'recallv2' WHEN video_id LIKE 'cbv2:%' THEN 'cbv2' ELSE 'corpus' END p,
       model, COUNT(*) n, ROUND(SUM(cost_usd),4) usd FROM classifier_calls GROUP BY 1,2
UNION ALL SELECT 'stance', model, COUNT(*), ROUND(SUM(cost_usd),4) FROM stance_calls GROUP BY 2;
```

### B3. Against the caps as they changed

| cap | value | what was spent under it | status |
|---|---|---|---|
| RunPod prepaid credit | $10.00 (`RUNPOD_CREDIT_USD`, stop line $9.50) | $6.57 recorded; $3.43 left | **PRELIMINARY**: a second $10 top-up was agreed with Puck but `RUNPOD_CREDIT_USD` still reads 10 in settings and the console balance was not checked for this file |
| `CLAUDE_CAP_USD` | $10 until 2026-10-17 (`plan/SHOWCASE-EXEC.md` §0), **$80** now (`spend.py --json`) | $8.29 of $10 on 10-17 18:55 UTC; $19.15 of $80 now | $60.85 headroom |
| day envelope 2026-10-17 | $20 (Puck, WhatsApp; **PRELIMINARY**, not in the DB) | 10-17: Claude $0.77 + $2.01 stance + RunPod $0.61 = **$3.39**; 10-18 so far: Claude $8.85 + RunPod $0.97 = **$9.82** | under |
| wave-2 budget | $10 (Puck, WhatsApp; **PRELIMINARY**) | the 10-18 corpus work: fill $4.10 + re-read $0.75 + GPU b34/b37 $0.55 = **$5.40** | under; the validation/pilot ceilings below were separate |
| recall check ceiling | $7 (`plan/RECALL-CHECK.md`) | $2.105 (b35 $0.231 + Claude $1.874) | 30 % |
| auto-caption arm ceiling | $4 | $0.953 (b38 $0.187 + Claude $0.766) | 24 % |
| codebook v2 pilot ceiling | $6 | $0.569 | 9 % |
| full-ASR arm ceiling | $4 | $0.770 | 19 % |
| stance smoke / pilot | ≤ $0.15 smoke; ~$1.50 pilot (`SHOWCASE.md` 4a) | $0.034 smoke; **$2.01 pilot** | pilot 1.3× its plan figure (the $1.50 predates the k=3 decision) |
| S1 stance pass | ~$35 quoted, $45 ceiling | **$35.46 actual** for the whole 9,358-turn frame (6,100-turn slice $24.14 + 2,958-turn top-up $11.33; `stance_batches` six rows tagged `stance-s1`, all `collected` 16:56 UTC) | under the ceiling; Claude ledger $63.05 of $80 after it (classifier spend also moved) |

```sql
-- refresh: spend by UTC day, both ledgers
SELECT d, ROUND(SUM(c),4) claude, ROUND(SUM(r),4) runpod FROM (
  SELECT substr(at,1,10) d, cost_usd c, 0 r FROM classifier_calls
  UNION ALL SELECT substr(at,1,10), cost_usd, 0 FROM stance_calls
  UNION ALL SELECT substr(started_at,1,10), 0, usd FROM spend) GROUP BY d ORDER BY d;
```

### B4. Estimate-vs-actual, for judging the contingency

| category | actual / estimate | read |
|---|---|---|
| GPU, diarized episode batches (transit) | 0.61 | the 1.5 contingency on top of a 25× RTFX that was correct; ~1.6× over-estimate |
| GPU, diarized (Hasan/Tucker) | 0.66 | same |
| GPU, no diarization (validation) | 0.33 | the estimator does not know the job skips pyannote |
| Claude classifier, corpus batches | 0.89 | about right; the Spanish and 455-window batches under-estimated (0.55× and 0.75×) before the 2.04 tokens/word fix |
| Claude classifier, validation arms | 0.53–0.83 | warm cache, WhisperX text at 1.34 tokens/word |
| Claude classifier, pilots | 0.46–0.51 | same, small batches with a pre-warmed prefix |
| Claude stance pilot | 0.90 | about right |
| Claude stance S1 pass | 0.54 | `stance.estimate_usd` assumes a 75 % cache-hit share (`CACHE_HIT_SHARE`); 6,000-call batches hit 98 %, so the codebook term is ~4× over-estimated. The constant is left as is: it is the right one for small direct runs |

---

## C. What finishing costs

The table given to Puck on 2026-10-18, with the arithmetic behind each line.

| item | quoted | derivation | prerequisite |
|---|---|---|---|
| **Locate, full-ASR with codebook v2** (not in the current phase) | $105 point / $130 planning | A4 | `docs/FUTURE-STEPS.md` 1 |
| **Episode transcription, remaining** | $5–10 | 380 `located` episodes / 65.0 h × $0.0216/h = $1.40; × 1.5 estimator = $2.10; the quoted range allows for the episodes the 120-s / density-vocabulary re-screen adds (**PRELIMINARY**: that worker had not landed) and 5–8 pod boots | YouTube downloads for the located clips, within the 12-request day budget |
| **Stance pass, full** | $100–150 via Batches on 30–45k turns (superseded) | **measured S1 rate $0.001305 per call** (A3), i.e. $0.00131 per turn at k=1, $0.00391 at k=3. The current QA-gated frame (9,358 turns) is **done**. Growing it: all 1,316 `done`+`cut`+`located` episodes (218.8 episode-h at the measured 77 turns ≥ 5 words per episode-hour) ≈ 16,900 turns = **$22 at k=1 / $66 at k=3**; the whole 2,099 meeting-h regular-meeting frame under full ASR ≈ 162,000 turns = **$211 / $633** (upper bound: whole meetings are less turn-dense than located episodes). The old 30–45k-turn figure is retired; `plan/STANCE-S1-RESULTS.md` §8 | κ ≥ 0.6 gate (`SHOWCASE.md` 4a) still unpassed; Puck keeps the phase exploratory |
| **S1 stance pass** (done) | ~$35 | **$35.46 actual** for 9,058 new turns × k=3 = 27,174 calls in six Message Batches (`stance_batches` note `stance-s1`, all `collected`); estimate $66.16, ratio 0.54 (98 % cache hits). Whole frame of 9,358 turns scored (300 pilot turns reused) | none; done |
| **A1 Claude pass** (optional) | ~$100 | one Sonnet call per turn ≥ 5 words, k=1, ~45k turns × $0.00224 = $101 | corpus freeze; cap headroom |
| everything else (QA, redaction, load, roles, A0/A1 lexicon, trends, models, dashboard, write-up, package) | $0 | local, `make test` / `make models` run on the Mac in minutes (`docs/REPORT.md` §9) | — |

```sql
-- refresh: remaining episode hours by status (located = not yet cut)
SELECT status, COUNT(*) n, ROUND(SUM(end_s-start_s)/3600,1) h FROM episodes GROUP BY status;
-- refresh: S1 stance sample once the sibling lands (new prompt_version or batch rows)
SELECT prompt_version, COUNT(*) calls, COUNT(DISTINCT episode_id||'#'||turn_idx) turns, ROUND(SUM(cost_usd),4) usd
FROM stance_calls GROUP BY 1;
```
