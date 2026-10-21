# SHOWCASE-EXEC — autonomous execution plan for the remaining steps

2026-10-17, Wren II. Puck is away for the day and asked for the remaining
`SHOWCASE.md` steps to be planned, reviewed by workers, revised, and executed
until the work is good. This file is the working plan; `SHOWCASE.md` stays the
approved scope.

## 0. Hard limits for every worker (non-negotiable)

- No GPU pod. `gpu_batch.py launch` only on Puck's texted `approve bN`.
- `.env` is never read into a conversation, printed, or edited. `CLAUDE_CAP_USD`
  stays $10 until Puck raises it. Claude ledger is $8.29 of $10 as of 18:55 UTC:
  the only Claude spend allowed today is one stance smoke run of **≤ 25 turns,
  ≤ $0.15**, by the stance worker, after a dry run shows the estimate.
- No YouTube requests except the pipeline scripts already running
  (`com.ytpt.captions`, the two `episodes.py` cut runs).
- Nothing is pushed to GitHub, no repo is created, nothing is published. The
  dashboard stays on localhost. "Push-ready" is the end state.
- Never delete under `output/`, never touch `plan/frame.csv` or
  `plan/DECISIONS.md`, never store names or persistent speaker ids (#15).
- Each worker works in its own git worktree off `main`
  (`git worktree add ../ytpt-<name> -b showcase/<name>`), uses
  `~/ytpt/.venv/bin/python`, runs `make test` before reporting,
  and does not merge. The supervisor merges after review.
- The DB (`state.sqlite`) is read-only for every worker except the stance
  worker, which may create and write its own new table(s) only.

## 1. State at start (18:55 UTC)

- Corpus: 496 episodes transcribed / 98.0 h, 644 transcripts QA-gated,
  559 OK / 85 WARN / 0 FAIL. b33 (88 clips, 16.26 h) awaits approval. Cuts
  running for 45 more clips (Richmond, San Leandro, Hayward, San Jose) → b34.
- Discovery is closed: window 2025-09-20 → 2026-09-20 by design; 0 new.
- Repo: main is clean, 433 checks, README/LICENSE/CI in. `~/ytpt-site` holds
  the dashboard sample (3 redacted episodes, SAMPLE banner).
- Blocked on Puck: cap raise (stance pilot and full pass), `approve b33`,
  `approve b34`, 30-turn redaction hand check, repo names, publish.

## 2. Work packages (WP), dependencies, acceptance

| WP | Step | Depends on | Deliverable | Acceptance |
|---|---|---|---|---|
| A | 4a code | — | `pipeline/stance.py`: per-turn Claude pass (stance −2…+2, hostility 0–4, frames multi-label, sub-topic, self-id from speech only), pinned codebook hashed into `prompt_version`, JSON-schema output, thinking disabled, Message Batches path + direct path, `--dry-run` cost estimate, `--sample N --stratify body,role` to draw the pilot set, `collect`, results in new `stance_calls` / `turn_stance` tables, `--cap-usd` via `pipeline/spend.py`. Turn definition = `analysis/a1_sentiment.py` (maximal run of segments by one cluster inside one episode). Codebook at `plan/STANCE-CODEBOOK.md`. Blind-coding sheet generator (`--coding-sheet out.csv`, 150 turns, no model columns). Agreement script (weighted κ, per-turn k-spread). | `make test` +≥6 checks; dry run prints estimate for 300 turns; **smoke ≤ 25 turns ≤ $0.15** returns parseable JSON for every turn; sheet has 150 rows; nothing else spent |
| B | 5 redaction | — | `pipeline/redact.py`: regex cues ("my name is", "I'm X and", "this is X", "Councilmember X", "Mayor X"…) + spaCy `en_core_web_sm` PERSON (install into venv) → `[name]`; street addresses → `[address]`; phone/email → `[contact]`. Per-turn, per-episode redaction counts. `--audit out.csv` of every replacement with 6-word context for Puck's hand check (30 random rows flagged). Report over-redaction rate on a 200-span sample read by the worker. Not applied in place: writes `output/redacted/<episode>.json` | `make test` +≥6 checks; runs over all gated episodes; audit CSV exists; worker's own read of 200 spans: ≤ 5 % obvious misses of self-introductions, over-redaction rate reported |
| C | 5 export | B | `site/export_site_data.py` (move from `~/ytpt-site/export/`): full `bodies.json`, `meetings.json`, `episodes.json` from **redacted** text, every gated episode, turn role + keyword/frame spans, stance columns present but null until 4b, Granicus `?t=`/`&autostart` deep-link if the player supports it (test on one URL, no login), per-episode redaction count, `qa_status`. Size budget: episodes.json ≤ 25 MB or split per body. | export runs from main checkout read-only; JSON validates; no `[name]`-free self-introduction found by grep of "my name is" |
| D | 6 models | — (re-run after freeze) | `analysis/models/topic_time.py` (PyMC: NB with log exposure offset, body random intercepts, month random walk or spline, family fixed effect, locate route as stratum, never pooled), `analysis/models/role_gap.py` (ordinal, ready to run on `turn_stance` when it exists; runs on the VADER-length-controlled stand-in now, labelled as such), PPCs, shrinkage plot, posterior tables as CSV, `make models`. `analysis/models/README.md` with the model statements and priors. | runs end to end on the current corpus in < 15 min on the Mac; PPC figures; convergence (R-hat < 1.01, no divergences after tuning) or an honest note |
| E | 7 dashboard | C (D for figures) | `~/ytpt-site` real version: full data, Granicus timestamps, model figures panel, flowchart nodes linking to repo paths, "preview — not yet published" banner (stays until Puck's hand check), redaction count shown per episode, methods footer with instrument versions. Lighthouse-level sanity: loads < 3 s locally. | screenshots 01–04 regenerated; console clean; click-through works on 3 bodies incl. a Granicus one |
| F | 8 write-up draft | D, A smoke | `docs/REPORT.md` (~2,000 words) + `docs/post-lucashenry.md` (Eleventy front matter, TL;DR bullets, house style of the "LLM chart sentiment" post). Numbers from the current corpus marked PRELIMINARY where the freeze or stance pass will change them. "What broke" section. Limitations. | reads clean; every number has a source comment; no invented results |
| G | 9 bullets | F | `docs/RESUME.md`: 3 bullets, 60-s talk track, 1-paragraph LinkedIn blurb | numbers match REPORT.md |
| H | ops | — | watch the cut runs; when done: `gpu_batch.py propose` for the new clips + the two requeued (b34); `make qa` after any batch; relay alerts | b34 proposed, nothing launched |

Order: A, B, D start now in parallel (independent). C after B. E after C (figures from D added when ready). F after D and A-smoke. G after F. H continuous.

## 3. Review protocol

1. Two reviewers critique this plan before any worker starts: one methods
   reviewer (estimands, model, instrument validity, redaction ethics), one
   engineering reviewer (repo structure, tests, concurrency, worktrees, what
   can break the running pipeline). Plan revised in place; changes logged in §5.
2. Every WP deliverable gets one reviewer pass on its branch (read the diff,
   run the tests, try to break it) before merge. Findings fixed by the same
   worker (continue the session), re-reviewed only if the fix is non-trivial.
3. After F exists, one final end-to-end reviewer reads REPORT.md against the
   code and data and lists every claim it cannot verify.

## 4. What Puck gets tomorrow

One message: what was built and merged, test count, the stance smoke output
(a few turns with their JSON), the redaction audit for his 30-turn check,
dashboard screenshots, the preliminary figures, the list of decisions still his
(cap raise, b33/b34, repo names, publish), and anything the reviewers flagged
that was not fixed.

## 5. Revision log

- v1 2026-10-17 18:58 UTC: initial.
- v2 2026-10-17 19:02 UTC: Puck clarified the "plan → team review → execute" instruction was about **GPU batch approvals**, which are now autonomous (USER.md 2026-10-17). §3 review protocol is dropped as a ceremony; WPs A–H stand as the work list and proceed under the approved `SHOWCASE.md` with ordinary review of each deliverable. §0 limits unchanged except: `gpu_batch.py approve/launch` no longer waits for a text, within the caps.
