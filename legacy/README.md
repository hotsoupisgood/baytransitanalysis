# legacy/ — the Hasan/Tucker side-corpus one-offs (April–September 2026)

These are the exploratory scripts from the first study (`hasan_compare.py` WER/diff harness, `scrape_hasan_audio.py`, `meanness_analysis.py`, `stance_analysis.py`) plus a dashboard screenshot; they read `../output/` and `../.env` and still run (`../run.sh` wraps `hasan_compare.py`; `make test` imports `meanness_analysis.host_segments`).
They belong to the **frozen Hasan/Tucker side corpus** (`plan/DECISIONS.md` #16: finished, not re-sampled, not deleted, its analysis code untouched) and are not part of the transit pipeline under `pipeline/` and `analysis/`.
Moved here 2026-10-16 (`plan/SHOWCASE.md` §4 row 2); older docs cite them by their root-level names.
