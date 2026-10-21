# Common tasks. Run `make` with no target for the list.
.PHONY: help setup test smoke creds sim load load-episodes a0 a0-load a0-facets a1 a1-wide wave1-coverage a0-pooled a0-wide a0-wide-load models overall kwic trends anim spread crawl frame qa legacy dashboard dashboard-demo status package clean discover download transcode audio-check batches propose

help:
	@echo "setup           verify environment, install deps, run tests"
	@echo "test            end-to-end assertions on simulated data + regression checks"
	@echo "smoke           test of systems: offline + free credential checks (docs/TESTING.md)"
	@echo "                  SMOKE_FLAGS='--network --notify --paid' adds the optional live checks"
	@echo "creds           just the credential checks (Anthropic, Hugging Face, RunPod)"
	@echo "sim             regenerate the simulated corpus"
	@echo "load            build corpus.sqlite from transcript JSON  (SRC=, DB=)"
	@echo "load-episodes   add the T1 meeting episodes (+ roles) from state.sqlite to corpus.sqlite  (DB=)"
	@echo "a0              Phase A0 on the T1 pilot: terms, KWIC, figures, trends-by-body, parquet+DuckDB  (A0DB=)"
	@echo "                  drawn once per instrument facet (locate route, role source); see analysis/facets.py"
	@echo "a0-pooled       the same figures pooled over instruments -- stamped POOLED_, not for reporting"
	@echo "a0-wide         A0 on every QA-gated episode, separate DB + output tree  (A0WIDEDB= A0WIDEOUT=)"
	@echo "models          the two showcase Bayesian models -> output/analysis/models  (MODELSOUT=)"
	@echo "a1              Phase A1 lexicon sentiment (VADER + hostility list) on the pilot, per facet  (A1OUT=)"
	@echo "a1-wide         the same A1 pass on the wide corpus  (A1WIDEOUT=)"
	@echo "wave1-coverage  per-body coverage table (meetings discovered / located / transcribed) -> output/wave1/"
	@echo "overall         overall word frequency by speaker         (DB=)"
	@echo "kwic            collocates around a node word    (NODE=gaza WINDOW=2 DB=)"
	@echo "spread          top words: monthly mean + 10/90 pct as 3 lines per speaker -> output/trends/top_words_spread.png (SPREAD=sd|minmax)"
	@echo "anim            animated top-words bar chart by month -> output/trends/top_words.gif  (DB=, TOP=8)"
	@echo "trends          content-word frequency over time -> output/trends/trends.html  (DB=, TOP=10 | WORDS='a b' | DISTINCTIVE=8)"
	@echo "crawl           metadata-only crawl of plan/channels.tsv -> plan/catalog.tsv"
	@echo "frame           cell table (HOURS=3 per speaker-month); add DRAW=1 to write plan/frame.csv"
	@echo "qa              transcript QA gate on QA_SRC -> output/qa/latest.csv"
	@echo "legacy          wrap the exploratory transcripts as contract JSON (output/contract_legacy)"
	@echo "dashboard       local progress page on http://localhost:8765"
	@echo "dashboard-demo  the same page on synthetic data"
	@echo "status          the WhatsApp-sized status digest"
	@echo "discover        load plan/catalog.tsv (+ frame.csv) into state.sqlite"
	@echo "download        run the download worker once over the queue (MAX=n)"
	@echo "transcode       16 kHz mono opus, first TRANSCRIBE_CAP_MIN minutes, for every downloaded row"
	@echo "audio-check     pre-GPU acceptance table for every opus file"
	@echo "batches         list GPU batches and spend; propose: pipeline/gpu_batch.py propose"
	@echo "package         build a clean transfer tarball (no secrets, no audio)"

# DB defaults to local disk, not the project folder: a project folder on a
# synced or network mount cannot host a SQLite file (see START_HERE.md #4).
# Use the project venv's interpreter when setup.sh created one.
PY     := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

DB     ?= $(HOME)/.local/share/ytpt/corpus.sqlite
# Phase A0 runs on the T1 PILOT, not on whatever is transcribed today, and it
# gets its own database so a later wide run cannot silently redefine "the
# pilot" (the pilot set is pinned in analysis/load_episodes.py PILOT).
A0DB   ?= $(HOME)/.local/share/ytpt/corpus_a0_pilot.sqlite
# The WIDE run is a SEPARATE TARGET and a SEPARATE DATABASE: every body and
# every locate route that is through the QA gate. `make a0` keeps meaning the
# 46-episode pilot, and nothing here can overwrite A0DB.
A0WIDEDB  ?= $(HOME)/.local/share/ytpt/corpus_a0_wide.sqlite
A0WIDEOUT ?= output/analysis/a0_wide
SRC    ?= output_sim/whisper
NODE   ?= gaza
WINDOW ?= 2
HOURS  ?= 3
QA_SRC ?= output/whisper
QA_GLOB ?= *.json
SMOKE_FLAGS ?=

setup:   ; ./setup.sh
test:    ; $(PY) analysis/test_pipeline.py
smoke:   ; ./scripts/smoke_test.sh $(SMOKE_FLAGS)
creds:   ; $(PY) scripts/check_credentials.py
sim:     ; $(PY) analysis/simulate.py
load:    ; @mkdir -p $(dir $(DB)); $(PY) analysis/load_db.py --src $(SRC) --db $(DB)
# Episode text passes name redaction before it is loaded (DECISIONS #15):
# TRANSCRIPTS= is the directory of redacted copies (pipeline/redact.py run --all
# -> output/redacted) and QA_GATE= the qa_transcripts.py report that gates them.
# Both default to the wave-1 settings; TRANSCRIPTS= (empty) reads output/whisper.
TRANSCRIPTS ?= output/whisper
QA_GATE     ?= output/qa/latest.csv
LOAD_EP_FLAGS = $(if $(TRANSCRIPTS),--transcripts $(TRANSCRIPTS),) $(if $(QA_GATE),--qa-gate $(QA_GATE),)
load-episodes: ; @mkdir -p $(dir $(DB)); $(PY) analysis/load_episodes.py --db $(DB) $(LOAD_EP_FLAGS)
a0-load: ; @mkdir -p $(dir $(A0DB)); $(PY) analysis/load_episodes.py --pilot --db $(A0DB)
# The pilot spans two role instruments, so the unfaceted scripts refuse to draw
# (analysis/facets.py). a0-facets draws every figure once per facet level; the
# pooled set is still available, explicitly, as `make a0-pooled`.
a0:      a0-load a0-facets
a0-facets: ; $(PY) analysis/facet_sweep.py --db $(A0DB) --out-dir output/analysis/a0
a0-pooled:
	$(PY) analysis/a0_terms.py  --db $(A0DB) --pool
	$(PY) analysis/a0_plots.py  --db $(A0DB) --pool
	$(PY) analysis/a0_bridge.py --db $(A0DB) --pool
	$(PY) analysis/a0_export.py --db $(A0DB)
# --- wide: separate DB, separate output tree, same facet rules ---------------
# --wide gates on output/qa/a0_episodes.csv (the 2026-09-23 99-episode set) unless
# QA_GATE= says otherwise; the wave-1 default above re-points it at the current
# QA snapshot, so `make a0-wide` draws every QA-gated episode transcribed today.
a0-wide-load: ; @mkdir -p $(dir $(A0WIDEDB)); $(PY) analysis/load_episodes.py --wide --db $(A0WIDEDB) $(LOAD_EP_FLAGS)
a0-wide: a0-wide-load
	$(PY) analysis/facet_sweep.py --db $(A0WIDEDB) --out-dir $(A0WIDEOUT) --title "A0 wide"
# --- the two showcase models (plan/SHOWCASE.md 1): topic time + role gap -----
MODELSOUT ?= output/analysis/models
models:  ; $(PY) analysis/models/run.py --out $(MODELSOUT)
# --- A1: the $0 lexicon sentiment pass, same facet rules, same pilot DB ------
# The Claude pass (plan/ANALYSIS-transit.md 3, second bullet) is NOT here and
# is not run without Puck's approval: it costs money.
A1OUT ?= output/analysis/a1
a1: a0-load
	$(PY) analysis/facet_sweep.py --db $(A0DB) --out-dir $(A1OUT) --steps a1
# the same $0 lexicon pass over the wide (every QA-gated episode) corpus
A1WIDEOUT ?= output/analysis/a1_wide
a1-wide: a0-wide-load
	$(PY) analysis/facet_sweep.py --db $(A0WIDEDB) --out-dir $(A1WIDEOUT) --steps a1 --title "A1 wide"
# wave-1 coverage table: regular meetings discovered vs located vs transcribed, per body
wave1-coverage: ; @mkdir -p output/wave1; $(PY) analysis/wave1_coverage.py --out-dir output/wave1
overall: ; $(PY) analysis/word_freq.py --db $(DB) --mode overall
kwic:    ; $(PY) analysis/word_freq.py --db $(DB) --mode kwic --node $(NODE) --window $(WINDOW)
TOP    ?= 12
spread:  ; $(PY) analysis/freq_spread.py --db $(DB) $(if $(WORDS),--words $(WORDS),--top $(TOP)) $(if $(SPREAD),--spread $(SPREAD),)
anim:    ; $(PY) analysis/freq_anim.py --db $(DB) $(if $(WORDS),--words $(WORDS),--top $(TOP))
trends:  ; $(PY) analysis/freq_trends.py --db $(DB) $(if $(WORDS),--words $(WORDS),$(if $(DISTINCTIVE),--distinctive $(DISTINCTIVE),--top $(TOP)))
crawl:   ; ./plan/crawl_metadata.sh
frame:   ; $(PY) plan/build_frame.py --quota-hours $(HOURS) $(if $(DRAW),--draw,)
qa:      ; @mkdir -p output/qa; $(PY) analysis/qa_transcripts.py --src $(QA_SRC) --glob '$(QA_GLOB)' --subs output/subs --csv output/qa/latest.csv
legacy:  ; $(PY) analysis/legacy_to_contract.py
dashboard: ; $(PY) dashboard/serve.py
dashboard-demo: ; $(PY) dashboard/serve.py --demo
status:  ; $(PY) dashboard/status.py
discover: ; $(PY) pipeline/discover.py
download: ; $(PY) pipeline/downloader.py $(if $(MAX),--max $(MAX),)
transcode: ; $(PY) pipeline/transcode.py
audio-check: ; $(PY) pipeline/transcode.py --check
batches: ; $(PY) pipeline/gpu_batch.py list
propose: ; $(PY) pipeline/gpu_batch.py propose
package: ; ./package.sh
clean:   ; rm -rf analysis/__pycache__ __pycache__ dashboard/__pycache__ pipeline/__pycache__
