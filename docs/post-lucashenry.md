---
title: "Bay Area Transit Analysis: what Bay Area councils and transit boards say about housing and transit"
date: 2026-10-18
tags: ["statistics", "bayesian", "llm", "python"]
description: "Transcribing only the housing-density and transit stretches of a year of Bay Area public meetings, then measuring how much time each body gives the topic and whether public comment reads differently from the dais."
---

<!-- DRAFT 2026-10-18, nothing published. Supersedes the 2026-10-17 draft,
which was written at n = 177 pilot turns. EXPLORATORY phase: no gold set, the
κ ≥ 0.6 hand-coding gate is not passed, corpus freezes 2026-10-24. Every
number carries a source comment. Sources: plan/WAVE1-RESULTS.md as corrected
2026-10-18, plan/STANCE-S1-RESULTS.md, plan/VALIDATION-2026-10-18.md.
Replace {{DASHBOARD_URL}} and {{REPO_URL}} before this goes anywhere. -->

TLDR:

- I pulled a year of regular meetings from Bay Area city councils, county boards and transit boards, found the stretches that are *about* housing density or transit, and transcribed only those.

- Transit boards give the topic several times the airtime councils do — and which route read a body moved its rate nearly as much as the body itself did.

- I then scored every turn of that corpus for stance and hostility. Public comment comes out rated more hostile than the dais on both topics, from a low base on both sides.

- On stance there is no single shift: the public sits in both tails, more opposed *and* more supportive at once, while the dais stays at neutral most of the time.

- It is all on the [dashboard]({{DASHBOARD_URL}}); the instruments, the numbers behind each claim and the limitations are below.

The stereotype is that public comment on housing is where "neighborhood
character" and "parking" live, and the dais is where the state-mandate
language lives. Two questions, written down before anything ran: how much
meeting time goes to density and transit, by body, and do public commenters
take a different stance, or a more hostile tone, than the dais in the same
meeting. <!-- plan/PROPOSAL-transit.md §1 -->

**Finding the episodes without transcribing everything.** The window holds
671 meetings and 2,296 hours, which is too much for a side project, and most
of it is consent calendars. So the pipeline fetches each meeting's caption
track, slides keyword windows over it, and asks a pinned Claude Sonnet
classifier whether each window is *about* density or transit or only mentions
it. Contiguous "about" windows become an episode, and only that clip goes to
the GPU to be transcribed with speaker diarization. Granicus archives with no
caption track get two more routes, agenda-index timestamps and a local
Whisper pass on my Mac, and Petaluma has its own Swagit transcript. The route
is recorded on every episode and carried as a stratum in every model, never
averaged over, because the routes have measurably different recall. What came
out the other end is 628 episodes, 121.6 hours of audio and 1.03 million
words across 26 bodies, for $2.74 of rented GPU.
<!-- VALIDATION-2026-10-18.md §1 corpus after the coverage fix: 671 meetings / 2,296.5 h / 31 bodies, 315 located, 43 downloaded, 1 failed; WAVE1-RESULTS.md §1 routes: classifier 334 ep / 53.9 h, local_asr 161 / 25.4, agenda_index 53 / 27.4, swagit_transcript 80 / 14.9; WAVE1-RESULTS.md §1: 628 episodes loaded (633 passed QA), 121.6 h padded bounds, 1,029,427 words / 72,106 segments, 26 bodies, 9 done GPU batches $2.74 actual -->

![Topic minutes per meeting-hour by body and route, with 95% intervals](FIGURE-rates-by-body.png)

In the plot above the three transit boards sit clear at the top and the
councils cluster below them, and the same body moves when you change the
route that read it. BART spends 27.7 minutes of every meeting-hour on the
topic (95% interval 24.7 to 30.7) and VTA 26.4 (22.8 to 29.8), about five
times what a council gives it; VTA's lower bound of 23.7 sits above the
highest council's upper bound of 17.1, so that gap is not a ranking artefact.
The route spread is of the same order: Hayward reads 0.65 minutes per
meeting-hour by classifier, 1.73 by agenda index and 4.17 by the local
Whisper pass, on the same meetings.
<!-- WAVE1-RESULTS.md §5 / output/wave1/rates_by_body_route.csv, all 14 cells VERIFIED in VALIDATION-2026-10-18.md §1; VTA lower 23.7 > council upper 17.1; Hayward 0.65/1.73/4.17 -->
That route spread is a first-order effect, which is why there
is no single pooled number per body anywhere on the dashboard. Density
outweighs transit on every council on the classifier route except Hayward
(0.1 against 0.4), and not on Petaluma's Swagit route (7.2 against 8.2).
<!-- VALIDATION-2026-10-18.md §1 "Route is a first-order effect — VERIFIED and now stronger"; §3 correction to WAVE1-RESULTS.md:184, Hayward 0.1 vs 0.4 and Petaluma Swagit 7.2 vs 8.2 -->

Six of the eleven bodies I had called fully covered are not. Forty-three
meetings arrived after the run and sit downloaded but unread, so the rates
for Fremont, Hayward, Petaluma, Richmond, San Leandro and Santa Clara are
floors, not estimates. Five bodies are clean: BART, Napa, SamTrans,
Sausalito and VTA. The coverage test that said otherwise counted any status
it did not recognise as neither located nor pending; it now raises instead.
<!-- VALIDATION-2026-10-18.md §1 and §3.1: complete set 11 → 5, wave1_coverage.py:32-51 exhaustive whitelist, UnknownStatus -->

**The instrument had to earn its job, and partly hasn't.** The stance
codebook scores each turn from −2 to +2 toward the item in front of the body
and 0 to 4 for hostility toward people, with the codebook hash stamped on
every row and three samples per turn. The three samples agree at weighted
κ 0.98 and are identical on 97.6% of turns. That is self-consistency, not
validity. The validity number is agreement against hand coding, the gate is
κ ≥ 0.6, and it is not in, so everything below is a measurement with an
unvalidated instrument. <!-- STANCE-S1-RESULTS.md §0, §1, §6, §10: κ 0.98 weighted and exact, 97.6% identical on all three, κ ≥ 0.6 gate (SHOWCASE.md 4a) not passed -->

![Hostility by role, posterior intervals per family and role instrument](FIGURE-hostility-role.png)

**The hostility gap.** The frame is a census, not a sample: 9,358 turns in 26
bodies, three samples each, for $35.46. Public turns are rated more hostile
than dais turns in
every family and under both role instruments: log-odds **2.15** on density
(90% 1.85 to 2.46), an odds ratio near nine.
The absolute levels stay low either way — **5.7% of public turns and 0.3% of
dais turns** score hostility 2 or more.
*Corrected 2026-10-18 after role audit*: those read 1.70 (1.39 to 2.02), an
odds ratio near five, and 3.7% / 0.5%, before I audited which of those
speakers were really the public (below). The audit moved this result the
*other* way from the stance one: dais speech mislabelled as public had been
diluting the hostility gap, so taking it out made the gap larger.
<!-- STANCE-S1-RESULTS.md §0: 9,358 turns (8,289 dais / 1,069 public), 27,174 calls, k=3, $35.46, 0 unparseable; VALIDATION §2 recomputed 3.74% / 0.52% -->
The fits are healthy, max
R-hat 1.0042 and one divergence in 8,000 draws. The length control matters
here, because public turns are longer by construction — a commenter gets a
fixed slot and the dais interjects — and it is in the fit as a centred
log-words term. <!-- STANCE-S1-RESULTS.md §0, corrected 2026-10-18: max R-hat 1.0042 (hostility/db), 1 divergence (hostility/computed); VALIDATION §2: length control confirmed at role_gap.py:60,100, route enters as beta_strat -->

![Posterior predictive check, stance by role](FIGURE-stance-ppc.png)

**Why the stance gap is zero and why that is not the answer.** The
proportional-odds model puts the public−dais gap at 0.00 (90% −0.18 to 0.20).
Read literally that says the two sides are alike, and the check in the plot
above says why it cannot: the public distribution has two humps where the dais
has one hump at neutral. The model's own replicate puts 58% of public turns at
neutral against 23% observed. The honest statement is the model-free one:
among turns that take a position the public is **33** points more opposed on
density than the dais (90% 27 to 40) *and* **31** points more supportive (25
to 37) — more opposed and more supportive at once, on both topics, which is a
bimodal distribution against a unimodal one and not something one shift
parameter can describe.

**Then I found out who the public actually was.** The role labeller works off
cue phrases, and a councilmember who says "I live in this neighbourhood" or
"on behalf of my community" reads as a member of the public. Every turn also
carries the model's read of what the speaker says about *who they are* — a
second, independent instrument — so I audited one against the other. Where the
speaker says who they are, **one in five to two in five turns in the "public"
bucket say they are an elected member or staff.** Switching the reported role
to that audit moves the per-family stance gap on density from **−0.38 (90%
−0.62 to −0.13) to −0.06 (−0.30 to +0.20)**. The model-free tail shares go the
other way and get wider, because the clusters leaving the public bucket are
long hedged member speeches and the ones
arriving from the dais side are short and opinionated. Both facts are the same
fact.

Two things this audit is not. It is not a gold set: the self-identification is
another model's output, not a human's, so it says two instruments disagree and
which one has the better claim, not what is true. And it is not a cleanup —
44% of turns don't say who the speaker is and keep the old label, and the
audit moves clusters *both* ways, so the public base actually grows from 414
clusters to 457. It is a re-partition, and the numbers it changes were never
confirmatory to begin with.
<!-- STANCE-S1-RESULTS.md §0: gap 0.00 (−0.18, 0.20); public 33/23/44, dais neutral 73%; replicate 58% vs 23% observed -->

**YIMBY and NIMBY, with one body doing most of the work.** Among public
speakers with a stance on density items, clustered as anonymous speakers, the
split is **46% YIMBY, 43% NIMBY, 11% neutral** on 457 clusters, and **only
Oakland** has an interval on YIMBY minus NIMBY that excludes zero (+0.65, 90%
+0.11 to +0.85), leaning YIMBY. *Corrected 2026-10-18 after role audit*: the
split read 42 / 41 / 16 on 414 clusters before the audit, and **Petaluma on
the Swagit route had the only other starred result — it is withdrawn.** Its
net went +0.53 (90% +0.22 to +0.78) to +0.40 (−0.04 to +0.81) and its interval
now crosses zero. The single largest "public" speaker in that frame, 1,083
words, was an elected member advocating housing next to the SMART station.
Petaluma public-YIMBY is not a finding.

The split is fragile for an unrelated reason too. San Leandro is 36% of the
457-cluster public base, 76% of its clusters are affordability and
displacement talk — the sub-topic where this mapping inverts, because a
landlord opposing a rent-stabilisation ordinance is "opposed to the item" and
lands as NIMBY — and dropping it moves the split to 47 / 40 / 13. And 83%
of cluster labels rest on exactly one scored turn. <!-- STANCE-S1-RESULTS.md §7b; VALIDATION-2026-10-18.md §3 corrections: 42.0/41.5/16.4 → 42.2/38.3/19.5 without San Leandro, 117 of 158 clusters (74%), 1.10 scored turns per cluster, 83% on one turn -->

**Privacy.** Everything here is the public record: regular meetings that
the bodies themselves stream and archive. Speakers are diarization clusters
with a role, never linked across meetings; the transcript text is shown as
spoken, including names people give on the record. An earlier build ran a
name-redaction pass; I retired it once the corpus was clearly public-record
material, and the dashboard shows episode excerpts linked to the source video
at the timestamp rather than whole-meeting transcripts.
<!-- owner decision 2026-10-18 12:11 (names kept, no redaction); plan/DECISIONS.md #15 needs its amendment; the retired pass is documented in WAVE1-RESULTS.md §1a -->

**What broke.**

- **YouTube rate-limited the caption fetch twice**, HTTP 429 after 17 requests in 31 minutes and then after 5 in 41. The fix was a slower worker rather than a faster retry: one request an hour, a hard cap of 18 per rolling 24 hours written to a table *before* each call so a restart cannot burst, and the queue ordered by measured episode yield per body, 16.7 episodes per captioned meeting at BART down to 0.00 at Rohnert Park. <!-- docs/REPORT.md §6; START_HERE.md "T2 part 2"; plan/T2-PLAN.md §3h -->
- **Two Granicus bodies looked caption-less and weren't.** Santa Clara and Hayward serve a full English track through a Flowplayer config object with no `<track>` element, which was the only thing the parser read. Fixing it recovered 37 meetings and 178 meeting-hours, Santa Clara 34 of 40, which classified to 105 episodes for $1.18. Treat every parser as a hypothesis about how a thing can be served. <!-- docs/REPORT.md §6; plan/T2-PLAN.md §3d -->
- **Haiku was rejected as the classifier**, at 0.758 agreement against a 0.80 threshold written before the test. It was precise, 0.98, but it demoted 39 of 94 transit windows to "mention only", nearly all BART board items, which would have cut the transit rate at boards by a third. Its minimum cacheable prefix is 4,096 tokens, so the 1,531-token codebook never cached and the 198-call test cost $0.52 instead of about $0.10. <!-- docs/REPORT.md §6; docs/MODELS.md "Haiku 4.5 as the episode classifier — measured 2026-09-22, not adopted"; plan/T2-PLAN.md §2 -->
- **One of my own validation findings was wrong.** I reported that a quarter of episodes carried an unremoved 60-second pad and that BART and VTA were inflated by about 6%. They weren't: the unpadded window bounds are identical to the episode bounds on every row that has not been cut yet, by design, and the rate numerator only ever uses the unpadded ones. The finding was withdrawn before anything was recomputed, because recomputing would have corrupted correct data, and the invariant is now documented at both call sites. <!-- VALIDATION-2026-10-18.md §2: "finding WITHDRAWN, false positive, no row changed"; classify.merge_episodes:766-797, pipeline/episodes.py:107-114 -->
- **Petaluma's roles were someone else's labels, and they were wrong.** On the Swagit route the speaker roles come from the transcript as published, and they credit 1,014 turns to the dais against 56 to the public, far more lopsided than any other route. Petaluma's public-comment numbers rested on 17 speaker clusters. *Corrected 2026-10-18 after role audit*: this is the caveat that turned into the error. Auditing each speaker against what they say about themselves moves four clusters out of Petaluma's public bucket and thirteen in (17 → 25), and **withdraws the starred YIMBY lean** — +0.53 (+0.22, +0.78) becomes +0.40 (−0.04, +0.81). The lesson is not that Swagit's labels are bad; it is that a cue-phrase role labeller has no access to self-identification, and a starred body-level result can rest entirely on that gap. <!-- STANCE-S1-RESULTS.md §4 frame by route: swagit_transcript 1,014 dais / 56 public; §7b: swagit petaluma 17 clusters (9 scored), +0.53 (+0.22, +0.78), 151 turns -->

**Limitations, because they are the point at this stage.** Bodies enter the
corpus because their archive is machine-readable, so ten wanted bodies are
missing, including every regional transit board except BART, VTA and
SamTrans; these are rates for the bodies I could read, not for the Bay Area.
Episode boundaries are one pinned model's judgement of what counts as
"about", and a different codebook would move every rate. Routes are strata
and are never pooled. The hand-coding gate is not passed. Every figure on
the dashboard carries that stamp for those reasons. <!-- docs/REPORT.md §7; plan/DECISIONS.md #13 -->

**What I'd do differently.** Pick the caption fetch rate before the first
run, not after the second 429. Write the posterior predictive checks before
the model: the topic-time check over-predicts the biggest cell and the stance
check misses the bimodal public distribution, and both would have shaped the
priors. And build the hand-coded reference first, so the first role-gap fit
is a result rather than a demonstration that the code runs.
<!-- plan/WAVE1-RESULTS.md §6.4; STANCE-S1-RESULTS.md §10 -->

A future direction I would like to try is the missing half rather than more
data: hand-code a few hundred turns blind and report the stance numbers
against that, so they are measurements of something instead of the output of
an instrument nobody has checked.

Spend so far is $8.43 of GPU and $69.47 of model calls, split $31.99
classifier and $37.48 stance, against a $31 GPU stop line and a $150 model
cap. <!-- VALIDATION-2026-10-18.md §3, correcting docs/COSTS.md:23-47: $8.4337 RunPod (22 rows), $69.4660 Claude = classifier $31.9897 + stance $37.4763; caps $31.50 credit / $31.00 stop / $150 Claude; 80%-of-cap escalation not triggered -->

The corpus freezes on 2026-10-24 and the numbers above will move. The
dashboard is at [{{DASHBOARD_URL}}]({{DASHBOARD_URL}}) and the pipeline,
the models and the offline test suite are on [GitHub]({{REPO_URL}}).
