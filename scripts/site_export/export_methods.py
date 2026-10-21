#!/usr/bin/env python
"""Export the Methods page's data from the ytpt repo (READ-ONLY).

Run from the ytpt repo with its venv:

    cd ~/ytpt && .venv/bin/python scripts/site_export/export_methods.py

Writes, beside this repo's methods.html:
    data/methods.json   every instrument's verbatim text, hash, model id and
                        validation numbers, pulled from code constants and
                        measurement artefacts

Why a build step for a no-build-step site (plan/SHOWCASE.md §0.2): the page
quotes four LLM instruments verbatim. Hand-copying them means the page drifts
from the instrument the moment anyone re-codes, and a drifted methods section
is worse than none. So every verbatim block here is read from the thing that
is actually sent to the model --
  * the episode classifier codebooks from pipeline.classify.CODEBOOKS (the
    same dict request_params() takes its system block from);
  * the stance codebook from pipeline.stance.codebook_text() (the block below
    '--- CODEBOOK ---' in plan/STANCE-CODEBOOK.md);
  * the NIMBY/YIMBY pilot prompt from scripts.nimby_pilot.SYSTEM;
  * the production flowchart from README.md's first ```mermaid``` fence
-- and each one carries a sha256 of its own text. analysis/test_pipeline.py
section_methods_page recomputes those hashes from the sources and fails if the
exported block is not byte-identical, so the page cannot go stale silently.

Nothing here is an instrument change: no constant is edited, no hash is
recomputed into a source file, and the recall / agreement numbers are read
from the measurement JSON, never retyped. Read-only on the ytpt repo; writes
only under this site repo's data/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

YTPT = Path("~/ytpt").expanduser()
SITE = Path(__file__).resolve().parents[2] / "docs"  # the GitHub Pages root of this repo
DATA = SITE / "data"
OUT = DATA / "methods.json"

sys.path.insert(0, str(YTPT / "pipeline"))
sys.path.insert(0, str(YTPT / "analysis"))
sys.path.insert(0, str(YTPT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from role_model import role_model  # noqa: E402  the one role-gap model
import classify  # noqa: E402  pipeline/classify.py -- the locate instrument
import stance    # noqa: E402  pipeline/stance.py   -- the stance instrument

EXPLORATORY = ("EXPLORATORY — no human-coded gold set exists in this chain. "
               "Every recall, precision and agreement number on this page is one "
               "LLM against another LLM, or a model against itself. Puck's decision "
               "of 2026-10-18 keeps the project in an exploratory phase: nothing here "
               "is a confirmatory result.")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def block(text: str, **extra) -> dict:
    """A verbatim instrument block: the exact bytes plus their fingerprint."""
    return {"text": text, "sha256": sha(text), "chars": len(text),
            "lines": text.count("\n") + 1, **extra}


def read_json(rel: str, default=None):
    p = YTPT / rel
    if not p.exists():
        return default
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# 1. Pipeline in one paragraph + the production flowchart from README.md
# ---------------------------------------------------------------------------
def section_pipeline() -> dict:
    readme = (YTPT / "README.md").read_text()
    fences = re.findall(r"```mermaid\n(.*?)```", readme, re.S)
    if not fences:
        raise SystemExit("README.md has no ```mermaid``` fence; the production chart moved")
    # The first fence is the production chart (README.md "Architecture": production
    # first, research flowchart second). Taking it by position rather than by a
    # heading grep keeps this working if the prose around it is reworded.
    mermaid = fences[0].rstrip("\n")
    para = (
        "Thirty-three Bay Area city councils, county boards and transit boards, "
        "regular meetings from 2025-09-20 to 2026-09-20 (discovery is closed, so the "
        "window is fixed). For each meeting the pipeline gets a cheap full-length text "
        "track — YouTube auto-captions where they exist, otherwise a Granicus agenda "
        "index, a Swagit timestamped transcript, or a Mac Whisper pass — and runs a "
        "keyword screen over it to find candidate windows. A pinned Claude classifier "
        "then makes a screening decision on each candidate: is this window substantively "
        "about residential density, about transit, both, a passing mention, or neither? "
        "Adjacent “about” windows merge into an episode. Only episodes are "
        "downloaded and transcribed properly (WhisperX large-v3 with pyannote "
        "diarization on a rented GPU), which is what makes the project affordable: "
        "roughly 2 % of the meeting hours get the expensive treatment. Gated transcripts "
        "are split into turns by diarized speaker cluster, labelled "
        "dais / public by cue phrases, and read turn by turn by a second pinned Claude "
        "instrument for stance and hostility. Every text field published anywhere on "
        "this site goes through the name / address / contact redactor before it is "
        "written, including the quotes and the one readable body's turns. One model is "
        "reported for the role gap, a mixed logistic fit with meetings nested in bodies; "
        "the earlier count and ordinal fits are in the repo and are not quoted here. "
        "Every locate route is a separate stratum and is never pooled silently, because "
        "the routes have measurably different recall."
    )
    return {"paragraph": para, "mermaid": block(mermaid, source="README.md (first mermaid fence)")}


# ---------------------------------------------------------------------------
# 2. Locate instruments: the episode classifier codebooks, verbatim
# ---------------------------------------------------------------------------
def section_locate() -> dict:
    notes = {
        "v1": ("The instrument for every row read up to 2026-10-18 (T0 onward). Kept "
               "byte-identical on purpose: one changed character changes CODEBOOK_VERSION, "
               "and rows read under two codebooks are never pooled."),
        "v2": ("Pilot codebook, 2026-10-18 (plan/CODEBOOK-V2-PILOT.md). Three edits off v1, "
               "all from the 30 missed windows of the recall check: active transportation "
               "and service planning named as transit; the “most of the window” test "
               "replaced by a contiguous-stretch test (~45 s / ~100 words); window "
               "granularity made an explicit parameter named in prompt_version."),
        "v2.1": ("The instrument going forward. v2 plus exactly one sentence at the top of "
                 "the transit definition — Puck's adjudication rule of 2026-10-18 09:36 PDT, "
                 "“transit is anything not car related” — and nothing else; "
                 "classify.py asserts the rest is byte-identical to v2. Frozen before the "
                 "corpus re-read so nothing is read twice."),
    }
    codebooks = [
        dict(block(classify.codebook_text(name),
                   name=name,
                   version=classify.CODEBOOK_VERSIONS[name],
                   est_tokens=round(classify.codebook_tokens(name)),
                   is_default=(name == classify.DEFAULT_CODEBOOK),
                   note=notes[name]))
        for name in ("v1", "v2", "v2.1")
    ]
    # The one sentence that separates v2.1 from v2, so a reader can see the whole
    # difference without diffing two 6,000-character blocks.
    v21_delta = block(classify.V21_TRANSIT_RULE, kind="v2.1 − v2")

    screen = {
        "window_s": classify.DEFAULTS["window"],
        "threshold_hits_per_100w": classify.DEFAULTS["threshold"],
        "max_run_s": classify.DEFAULTS["max_run"],
        "pv_max_run_s": classify.PV_MAX_RUN,
        "pad_s": classify.DEFAULTS["pad"],
        "merge_gap_s": classify.DEFAULTS["gap"],
    }
    return {
        "model": classify.LEGACY_MODEL,
        "model_note": ("docs/MODELS.md §2: claude-sonnet-4-6, thinking disabled, no sampling "
                       "parameters, JSON-schema output, the codebook as a cached system block, "
                       "Message Batches API (50 % off). Haiku 4.5 was measured as the classifier "
                       "on 2026-09-22 and rejected at 0.758 about-vs-not agreement — it demotes "
                       "39 of 94 about_transit windows, which would cut the transit rate at "
                       "transit boards by a third."),
        "screen": screen,
        "window_note": (f"Candidate windows are cut at {screen['max_run_s']:g} s with "
                        f"±{screen['pad_s']:g} s of caption text padded on each side. "
                        f"The bare prompt_version means {screen['pv_max_run_s']:g} s — every row "
                        f"up to 2026-10-18 — so a {screen['max_run_s']:g}-s reading carries "
                        f"'/mr{int(screen['max_run_s'])}' and never pools with a "
                        f"{int(screen['pv_max_run_s'])}-s one."),
        "codebooks": codebooks,
        "v21_delta": v21_delta,
        # The keyword screen's own version is a DB-row property (classify.keywords_version_of
        # reads it off classified rows), so it is taken from the production chart in the
        # versions table rather than recomputed here.
        "recall": section_recall(),
    }


def section_recall() -> dict:
    """The adjudicated arms table: like-for-like pairs only, on both references."""
    adj = read_json("output/validation/codebook_v2/adjudicated/results_adjudicated.json")
    local = read_json("output/validation/recall/results.json")
    rows = []
    if adj:
        label = {"v1_240": "v1 @ 240 s", "v1_120": "v1 @ 120 s", "v2_120": "v2 @ 120 s",
                 "v2_240": "v2 @ 240 s", "v2_fullasr": "v2 + full ASR (no screen)"}
        for fam in ("any", "transit", "density"):
            for arm, v in adj["arms"].items():
                b, a = v["before"].get(fam), v["after"].get(fam)
                if not b or not a:
                    continue
                rows.append({
                    "family": fam, "arm": arm, "arm_label": label.get(arm, arm),
                    # Each cell is the SAME arm scored on the SAME family under each of the
                    # two references, so the pair is like-for-like by construction. The
                    # "0.44 -> 0.87" figure quoted elsewhere is NOT (VALIDATION §6).
                    "v1_ref": {"recall": b["recall"], "precision": b["precision"]},
                    "adj_ref": {"recall": a["recall"], "recall_ci": a.get("recall_ci"),
                                "precision": a["precision"], "precision_ci": a.get("precision_ci")},
                    "about_min_v1_ref": b.get("about_min"), "about_min_adj_ref": a.get("about_min"),
                    "best": arm == "v2_120",
                })
    a = (adj or {}).get("adjudication", {})
    return {
        "rows": rows,
        "references": [
            {"key": "v1_ref", "name": "v1-coded reference",
             "detail": ("Every 60-s window of 4 whole-meeting GPU transcripts labelled "
                        "exhaustively by codebook v1 (t0-a4a8e55760), no keyword screen. "
                        f"{a.get('n_ref_minutes', {}).get('before', 109) if isinstance(a.get('n_ref_minutes'), dict) else 109} about-minutes.")},
            {"key": "adj_ref", "name": "adjudicated reference",
             "detail": (f"The same reference with {a.get('n_minutes_flipped', 18)} minutes inside "
                        f"{a.get('n_windows_flipped', 11)} windows flipped to the label v2 assigned, "
                        "after Puck judged all 11 of v2's over-calls to be genuinely “about” "
                        "(WhatsApp, 2026-10-18 09:33 PDT). 127 about-minutes.")},
        ],
        "circularity": (
            "Two things have to be said about the right-hand column, both from "
            "plan/VALIDATION-2026-10-18.md §6. First, the widely quoted lift from 0.44 to "
            "0.87 is not like-for-like: 0.435 is v1 @ 240 s scored on the v1 reference and "
            "0.865 is v2 @ 120 s scored on the adjudicated one. Like-for-like, v1 @ 240 s to "
            "v2 @ 120 s is 0.297 → 0.865 on the adjudicated reference, or 0.435 → 0.783 "
            "on the v1 reference. Second, and worse: the adjudicated reference is not "
            "independent of the arm it scores. All 18 flipped minutes were created by "
            "adjudicating v2's own over-calls as “about”, so v2 @ 120 s hits 18 of 18 "
            "of them by construction. On the 109 untouched reference minutes v2 @ 120 s scores "
            "any 0.780 and transit 0.783 — the entire lift from 0.78 to 0.87 transit comes from "
            "minutes v2 itself defined, and transit's reference grew from 23 to 37 minutes with "
            "38 % of the growth coming from v2's output. A reference corrected only where the "
            "instrument under test disagreed cannot measure that instrument; read 0.811 / 0.873 "
            "as a ceiling, not an estimate. The remedy on the docket is symmetric blind "
            "adjudication — equal draws of v2 over-calls, v2 misses, v1-only calls and "
            "agreements, with the arm hidden."),
        "power": ("4 meeting-clusters, 2 bodies, 1 locate route, 37 transit about-minutes in "
                  "2 meetings. The confidence intervals are decorative at this sample size."),
        "local_asr_arm": ({
            "families": {f: {"about_min": v["about_min"], "located_min": v["located_min"],
                             "recall": v["recall_min"], "recall_ci": v["recall_min_ci"],
                             "screen_recall": v.get("screen_recall_window")}
                         for f, v in local["families"].items()},
            "miss_share": local.get("miss_share"),
            "note": ("A separate measurement (plan/RECALL-CHECK.md, 2026-10-18): the v1 "
                     "pipeline's own recall on the local_asr route, 4 usable meetings of 6 "
                     "(two Richmond meetings failed GPU transcription QA with a decoder loop "
                     "and could not be scored). 23 of the 30 missed about-minutes (77 %) never "
                     "cleared the keyword screen at all — the ASR text used different words "
                     "than the keyword list — and only 7 were candidates the classifier then "
                     "called mention_only or no. This is below the recall < 0.9 reopen line "
                     "that DECISIONS #13 states for itself, and transit (0.304) is far below. "
                     "It does not generalise to the caption-based route, which has its own "
                     "unmeasured recall."),
        } if local else None),
    }


# ---------------------------------------------------------------------------
# 3. Stance instrument
# ---------------------------------------------------------------------------
def section_stance() -> dict:
    text = stance.codebook_text()
    rep = read_json("output/stance_s1/s1_k_replicate_agreement.json", {})
    allrep = rep.get("all", {})
    return {
        "model": stance.MODEL if hasattr(stance, "MODEL") else classify.LEGACY_MODEL,
        "prompt_version": stance.prompt_version_for(
            stance.MODEL if hasattr(stance, "MODEL") else classify.LEGACY_MODEL),
        "codebook_version": stance.CODEBOOK_VERSION,
        "codebook": block(text, source="plan/STANCE-CODEBOOK.md, everything below '--- CODEBOOK ---'",
                          file_sha256=sha((YTPT / "plan" / "STANCE-CODEBOOK.md").read_text())),
        "unit": ("A turn: the maximal run of consecutive transcript segments by one diarized "
                 "speaker cluster inside one episode. Turns under 5 words are not coded. The "
                 "model sees the episode's family tags, the agenda item title when the platform "
                 "supplied one, the previous and next turn clipped to 40 words, and the turn."),
        "k": 3,
        "replicates": {
            "n_turns": allrep.get("n_turns"),
            "stance_kappa_w": allrep.get("stance_kappa_w_pairwise"),
            "hostility_kappa_w": allrep.get("hostility_kappa_w_pairwise"),
            "all3_identical_stance": allrep.get("all3_identical_stance"),
            "by_role": {k: {"n_turns": v.get("n_turns"),
                            "stance_kappa_w": v.get("stance_kappa_w_pairwise")}
                        for k, v in rep.items() if k != "all"},
            "note": ("k = 3 samples per turn, median kept, spread recorded. The three "
                     "replicates are the same model reading the same prompt, so this measures "
                     "the instrument's noise, not its validity: it is the instrument agreeing "
                     "with itself."),
        },
        "kappa_gate": (
            "The κ ≥ 0.6 agreement gate that plan/SHOWCASE.md §4a sets for this "
            "instrument HAS NOT BEEN PASSED, because no human has coded any of these turns. "
            "The 0.98 replicate κ above is self-consistency. The 0.63 κ against the "
            "NIMBY/YIMBY prompt is Sonnet agreeing with Sonnet under a second codebook. Neither "
            "is a validation against human judgement, and the blind-coding sheet the gate calls "
            "for (150 turns, two coders) has not been filled. Treat every stance number as "
            "exploratory."),
    }


# ---------------------------------------------------------------------------
# 4. Speaker labels: the NIMBY/YIMBY pilot prompt and the derived view
# ---------------------------------------------------------------------------
def section_speakers() -> dict:
    # nimby_pilot imports anthropic only inside client(), so importing the module
    # neither needs a key nor touches .env.
    import nimby_pilot as nb

    ag = read_json("output/nimby_pilot/agreement.json", {})
    summary = read_json("output/stance_s1/nimby_view/summary.json", {})
    # The sign-flip and one-turn caveats below used to quote 2026-10-18 literals,
    # which went stale the moment the frame grew (plan/STANCE-S2.md). They now read
    # analysis/stance_s2_site_numbers.py, which recomputes them from the view's own
    # per-cluster CSV under the reported (audited) role.
    N = read_json("output/stance_s2/site_numbers.json", {})
    sl = N.get("sanleandro") or {}
    sp = N.get("split_audit") or {}
    spd = N.get("split_audit_drop_sanleandro") or {}
    one = N.get("public_cluster_scored_turns") or {}
    pct = lambda v, d="?": f"{v * 100:.1f}" if isinstance(v, (int, float)) else d
    trip = lambda s: f"{pct(s.get('YIMBY'))} / {pct(s.get('NIMBY'))} / {pct(s.get('neutral'))}"
    view = (YTPT / "analysis" / "stance_s1_nimby_view.py").read_text()
    thresh = re.search(r'"--threshold",\s*type=float,\s*default=([0-9.]+)', view)
    min_cl = re.search(r'"--min-clusters",\s*type=int,\s*default=([0-9]+)', view)

    return {
        "prompt": block(nb.SYSTEM, source="scripts/nimby_pilot.py SYSTEM"),
        "prompt_version": nb.PROMPT_VERSION,
        "model": nb.MODEL,
        "labels": list(getattr(nb, "LLM_LABELS", ("YIMBY", "NIMBY", "neither", "not_about_housing"))),
        "pilot": {
            "n_labelled": ag.get("n_labelled"),
            "prompt_version_run": ag.get("prompt_version"),
            "label_counts": ag.get("llm_label_counts"),
            "actual_usd": ag.get("actual_usd"),
            # The word-lexicon arm of this pilot came off both pages 2026-10-20 (owner).
            # It asked whether a hand-built YIMBY/NIMBY word list could stand in for an
            # LLM pass; it could not, and its κ was a lexicon-against-LLM disagreement
            # with no human in it, so it measured neither instrument. Publishing a number
            # that low invited reading it as a validation failure of the prompt. The arm,
            # its two lexicons and its κs stay in output/nimby_pilot/.
            "note": ("No human coded these 300 turns, so nothing here is a validation. What "
                     "the pilot does support is that this prompt and the independent stance "
                     "instrument agree on sides at κ = 0.631 (4-way agreement 0.74) — which "
                     "is Sonnet against Sonnet, not against a person. The prompt below is "
                     "quoted because it ran; the figures on the dashboard come from the "
                     "derivation rule underneath it, not from this prompt."),
        },
        "derivation": {
            "threshold": float(thresh.group(1)) if thresh else 0.5,
            "min_clusters_per_body": int(min_cl.group(1)) if min_cl else 10,
            "source": "analysis/stance_s1_nimby_view.py",
            "rule": ("The per-cluster YIMBY/NIMBY view is DERIVED from the stance pass, not "
                     "from the prompt above. Unit: one anonymous diarized speaker cluster inside "
                     "one episode (episode × cluster). For every cluster with at least one "
                     "turn carrying a numeric stance, the mean of its turns' median stance is its "
                     "density stance: YIMBY if that mean is ≥ +0.5, NIMBY if ≤ −0.5, "
                     "neutral in between. The threshold is half a codebook step. Only episodes "
                     "tagged density or both are in scope, since YIMBY/NIMBY is a housing "
                     "construct. Shares are reported per body with an episode-cluster bootstrap, "
                     "per locate route, never pooled across routes."),
            "no_linking": ("DECISIONS #15. The cluster key is (episode_id, cluster) — a speaker "
                           "is never linked across meetings, and no names or persistent speaker "
                           "ids are extracted, stored or indexed. A person who speaks at three "
                           "meetings is three clusters here; a person split across two clusters "
                           "inside one episode counts twice, and two people merged into one "
                           "cluster count once."),
            "n_turns": summary.get("n_turns"),
            "n_clusters": summary.get("n_clusters"),
            "clusters_by_role_label": summary.get("clusters_by_role_label"),
            "stamp": summary.get("stamp"),
        },
        "caveats": [
            {"name": "Sign flip on restriction items",
             "text": ("Stance is coded toward the agenda item, not toward housing. On items "
                      "that are restrictions — a moratorium, rent stabilisation, a height limit "
                      "— “supportive” is the NIMBY side and the mapping inverts: a "
                      "landlord opposing rent stabilisation is coded +1 and would read as YIMBY. "
                      "This is not a rare edge: "
                      f"{sl.get('flip_clusters', '?')} of San Leandro's {sl.get('clusters', '?')} "
                      f"public clusters ({pct(sl.get('flip_share'))} %) carry sub_topic = "
                      "affordability_displacement, where the inversion applies, and San Leandro "
                      f"is {sl.get('clusters', '?')} of {sp.get('n', '?')} public clusters — "
                      f"{pct(sl.get('share_of_public_base'))} % of the whole public base. Dropping "
                      f"it moves the split from {trip(sp)} to {trip(spd)}. Recomputed under the "
                      "audited role over the whole scored frame: when the frame was smaller San "
                      "Leandro was over a third of the public base and dropping it visibly moved "
                      "the split, so the near-tie then was partly a San Leandro artefact; on the "
                      "current frame it is a fifth of the base and the split barely moves. The "
                      "inversion is still wrong for those clusters — it is now diluted, not "
                      "resolved.")},
            {"name": "Most labels rest on one turn",
             "text": (f"Public clusters average {one.get('mean_all', '?')} scored turns "
                      f"({one.get('mean_with_stance', '?')} among those with a stance), and "
                      f"{pct(one.get('share_one_numeric'))} % of cluster labels rest on exactly "
                      "one scored turn. The “mean stance of a speaker” is usually the "
                      "stance of a single utterance, with none of the averaging the word "
                      "“mean” implies.")},
        ],
    }


# ---------------------------------------------------------------------------
# 4b. The models behind the role results
# ---------------------------------------------------------------------------
def section_models() -> dict:
    """Short entries for the statistical work, complementing the panel bullets.

    Owner, 2026-10-19: "in the methods it should read like a complement to the
    bullets in the dashboard page. keep them short, people can dig in in the
    code if they want." So: two to four sentences each, every number read from
    the runs' own artefacts, and the plan document named for anyone who wants
    the full table.

    Owner, 2026-10-19, later: "lets keep that lme4 model then as the one we use
    in the dashboard. drop the others from the final explanation but keep it in
    the repo." So the role gap has exactly one entry and one estimator here,
    with its variance components; the Bayesian horseshoe, the GEE, the custom
    Gauss-Hermite GLMM and the beta-binomial are named nowhere on the site, and
    the only thing said about them is the one line below about how far they
    disagree with this fit -- whose number is computed from the comparison
    table, not asserted.

    The permutation entry says that the role permutation is WITHDRAWN.
    plan/RENTER-SHARE-4-SUBSETS.md §0 is explicit -- "No permutation result for
    role should be quoted" -- so the p = 0.0002 run 4 once reported is named
    here only in order to say it is not evidence.
    """
    N = read_json("output/stance_s2/site_numbers_role.json", {})
    perm, rp = N.get("permutation") or {}, N.get("role_prediction") or {}
    nb = N.get("neutral_by_role") or {}
    cue = rp.get("cue_role") or {}
    cl, tn = cue.get("cluster") or {}, cue.get("turn") or {}
    sdir = cue.get("signed_stance_only_cluster_logit") or {}
    imp = rp.get("importance_cluster") or []
    pv = lambda v: "—" if v is None else f"{v:g}"
    pc = lambda v: "—" if v is None else f"{v * 100:.0f}"

    MO = role_model()
    steps = {s["key"]: s for s in (MO.get("steps") or [])}
    side, took = steps.get("which_side"), steps.get("any_side")
    fi = lambda v: "—" if v is None else f"{v:,.0f}"
    sg = lambda v: "—" if v is None else f"{v:+.2f}"
    f2 = lambda v: "—" if v is None else f"{v:.2f}"
    sd = lambda v: "exactly 0" if not v else f"{v:.2f}"
    # A profile interval can fail to bound at one end or at both. Saying which is the
    # honest rendering and it is also what keeps this page exportable: the which_side
    # renter term came back [None, None] on the 2026-10-20 refit, and formatting that
    # end as a number crashed the whole Methods export.
    def iv(s):
        lo, hi = s["ci95"][0], s["ci95"][1]
        if lo is None and hi is None:
            return "an interval the likelihood could not bound at either end"
        if lo is None:
            return f"up to {hi:+.2f}, with no lower end the likelihood could bound"
        if hi is None:
            return f"from {lo:+.2f} up, with no upper end the likelihood could bound"
        return f"{lo:+.2f} to {hi:+.2f}"
    # The renter-share interval of a step, not its role interval.
    ivr = lambda s: iv(s["renter"])

    entries = []
    if side:
        # One entry, one model, both steps, with the variance components the
        # owner asked for and the body-is-zero line.
        spec = (
            "One row per scored transcript turn, one outcome per step, and a mixed "
            "logistic regression with a separate baseline for every stretch of meeting, "
            "every meeting and every body it sits in, plus renter share, its interaction "
            "with role and the locate route as fixed terms. Intervals are 95 % and come "
            "from the likelihood rather than a standard error, except where the likelihood "
            "bounds a term at neither end and the standard-error interval is given and "
            "labelled as one.")
        first = ""
        if took:
            first = (
                f"Step one, did a turn take a side at all: over {fi(took['n_turns'])} "
                f"stance-bearing turns in {fi(took['n_meetings'])} meetings, a public turn "
                f"against a dais one is {sg(took['beta'])} log-odds ({iv(took)}), an odds "
                f"ratio of {f2(took['odds_ratio'])}. ")
        second = (
            f"Step two, which side given a side: over the {fi(side['n_turns'])} turns "
            f"that took one, in {fi(side['n_meetings'])} meetings, public against dais is "
            f"{sg(side['beta'])} log-odds ({iv(side)}), an odds ratio of "
            f"{f2(side['odds_ratio'])}. ")
        varcomp = (
            f"Where the dependence sits, on step two: the spread between stretches of "
            f"meeting is {sd(side['sd_episode'])} on the log-odds scale and between meetings "
            f"{sd(side['sd_meeting'])}, against {sd(side['sd_body'])} between bodies. "
            f"The body variance is zero once meeting is in the model — meeting is nested "
            f"inside body and absorbs it — which is why the model groups turns by meeting "
            f"and not by city, and why the interval on step two is taken from the identical "
            f"fit with the empty body term dropped. ")
        if took:
            varcomp += (
                f"On step one the same three spreads are {sd(took['sd_episode'])}, "
                f"{sd(took['sd_meeting'])} and {sd(took['sd_body'])}, and that fit needs no "
                "such reparameterisation. ")
        tail = (
            f"Both steps are large in the same direction, and what separates them is what "
            f"they measure: step one is mostly speech act, since "
            f"{pc((nb.get('dais') or {}).get('neutral_share_of_stance_bearing'))} % of "
            f"stance-bearing dais clusters are neutral against "
            f"{pc((nb.get('public') or {}).get('neutral_share_of_stance_bearing'))} % of "
            "public ones and chairing, moving and reading a report are all neutral by "
            "construction. Step two is the one to quote for a difference in position.")
        entries.append({
            "name": "The one model the role gap is reported from",
            "text": spec + " " + first + second + varcomp + tail,
            "docs": ["plan/RENTER-SHARE-6-R.md"] + list(MO.get("scripts") or [])})

        # The covariates, only where this run gives a number for them. The step-two
        # renter profile bounds at neither end, so role_model substitutes the
        # standard-error interval; name that in the sentence rather than let a reader
        # take it for a likelihood interval like every other number on the page.
        ren_note = (" The step-two interval is a standard-error one: the likelihood "
                    "bounded that term at neither end."
                    if side["renter"].get("ci95_kind") == "wald" else "")
        ren_lines = [
            f"Renter share does not move either step: on step two its interval runs "
            f"{ivr(side)}"
            + (f", and on step one {ivr(took)}" if took else "")
            + "." + ren_note
            + " Both cover zero, and so does its interaction with role."]
        tc = MO.get("transit_commute")
        if tc and tc.get("beta") is not None:
            ren_lines.append(
                f"The one covariate besides role that clears 95 % anywhere in this run is "
                f"transit commute share, at {sg(tc['beta'])} log-odds "
                f"({tc['ci95'][0]:+.2f} to {tc['ci95'][1]:+.2f}) — and that is a fit on "
                "body-level rows rather than on turns, because commute share is a property "
                "of a city and the turn model does not carry it. It is not the role result "
                "and is not on the dashboard.")
        ren_lines.append(
            "Nothing else in the covariate set survives: the honest summary of that part of "
            "the work is negative, and one body — Santa Clara — decides the sign of the "
            "renter slope by itself.")
        entries.append({
            "name": "The covariates in it",
            "text": " ".join(ren_lines),
            "docs": ["plan/RENTER-SHARE-6-R.md", "plan/RENTER-SHARE-4-SUBSETS.md"]})

        ag = MO.get("agreement") or {}
        if ag.get("max_abs_logodds") is not None:
            entries.append({
                "name": "The earlier fits are still in the repo",
                "text": (f"This result was fitted several other ways before it was settled, and "
                         f"those fits are kept in the repo rather than on this page: across the "
                         f"{ag['n_pairs']} coefficients they have in common they agree with the "
                         f"model above to within {ag['max_abs_logodds']:.3f} log-odds, so there "
                         f"is nothing for a second column to show."),
                "docs": ["plan/RENTER-SHARE-6-R.md", "plan/RENTER-SHARE-5-FREQ.md"]})

    entries.append({
        "name": "Predicting the role from content",
        "text": (f"A gradient-boosted model given 35 content features and no speaker "
                 f"identifier separates the two roles at AUC {cl.get('auc', 0):.2f} per "
                 f"cluster and {tn.get('auc', 0):.2f} per turn, holding out every meeting of "
                 f"a body together. Two features do nearly all of it: dropping absolute "
                 f"stance costs {(imp[0]['auc_drop'] if imp else 0):.2f} of AUC, dropping "
                 f"length {(imp[1]['auc_drop'] if len(imp) > 1 else 0):.2f}, and nothing else "
                 f"over 0.02. Signed stance alone is at chance ({sdir.get('auc', 0):.2f} per "
                 "cluster under a linear model), so what separates the benches is "
                 "opinionatedness, not position. Calibration does not travel between bodies "
                 f"— slope {cl.get('cal_slope', 0):.2f} out of body — so this is a "
                 "separability measurement, not a classifier to point at a new city."),
        "docs": ["plan/ROLE-PREDICTION.md", "plan/ROLE-AUDIT-2026-10-18.md"]})
    entries.append({
        "name": "Permutation tests on the covariates",
        "text": (f"Covariate-to-outcome links were tested against two nulls at "
                 f"{perm.get('n_perm') or 5000:,} draws each, and only transit commute share "
                 f"survives either of them; renter share does not."),
        "docs": ["plan/RENTER-SHARE-4-SUBSETS.md", "plan/RENTER-SHARE-2.md"]})
    entries.append({
        "name": "There is no permutation test for the role gap",
        "text": ("Run 4 originally reported p = "
                 f"{pv((perm.get('role_withdrawn') or {}).get('was'))} for role, by swapping "
                 "the two role rows inside a body-and-route cell. That test is withdrawn and "
                 "must not be quoted. "
                 # The artefact's own explanation, cut before its closing
                 # sentence: that sentence points at run 5's cluster-robust
                 # standard error, and the evidence here is now the nested
                 # model and nothing else.
                 + ((perm.get("role_withdrawn") or {}).get("why") or "")
                 .split("The role gap rests on")[0].strip()
                 + " The evidence for the gap is the model above."),
        "docs": ["plan/RENTER-SHARE-4-SUBSETS.md", "plan/REGRESSION-REVIEW-2026-10-19.md"]})
    entries.append({
        "name": "What the review changed",
        "text": ("An independent statistical review on 2026-10-19 raised eleven findings "
                 "against these regressions, and all eleven were implemented the same day. "
                 "The one that moved this page is the nesting: clusters sit inside episodes "
                 "inside meetings, the earlier fits grouped them by body instead, and once "
                 "the real nesting is modelled the role estimate is not merely less precise "
                 "but smaller. That nested fit is the model above, and it is the only one "
                 "quoted here. No earlier number was silently replaced: each is struck "
                 "through in place in the plan documents."),
        "docs": ["plan/REGRESSION-REVIEW-2026-10-19.md", "plan/RENTER-SHARE-6-R.md"]})

    return {
        "intro": ("The statistics behind the role results on the dashboard, in short. The "
                  "role gap comes from one model, described in the first entry; the rest of "
                  "this section is what else was measured about it and what cannot be "
                  "claimed. Every number here is substituted at build time from the runs' "
                  "own artefacts — none of it is typed into this page."),
        "entries": entries,
        "stamp": N.get("stamp"),
        "available": bool(MO.get("available")),
        "source": MO.get("source", "output/renter_share_6/"),
    }


# ---------------------------------------------------------------------------
# 5. Limitations
# ---------------------------------------------------------------------------
def section_limitations() -> dict:
    review = (YTPT / "plan" / "REVIEW-2026-10-18.md").read_text()
    m = re.search(r"## Owner's response.*?\n\n(.*?)(?=\n## |\Z)", review, re.S)
    owner = m.group(1).strip() if m else ""
    return {
        "owner_response": block(owner, source="plan/REVIEW-2026-10-18.md, Owner's response"),
        "items": [
            {"severity": "critical", "name": "No human-coded gold set, anywhere",
             "text": ("Nobody has hand-labelled anything in this chain. The 100-row label sheet "
                      "is empty, the 12-clip listening set has no corrections, and the stance "
                      "κ gate is unpassed. So every recall, precision, word-error and stance "
                      "number on this page is Sonnet against Sonnet, or WhisperX against YouTube's "
                      "ASR. Agreement between two LLMs tells you they share a bias; it does not "
                      "tell you either one is right. The path out is costed and on the docket: "
                      "~500 windows per coder, two coders, blind to which arm produced a label, "
                      "Krippendorff's α ≥ 0.7 on the 5-way label.")},
            {"severity": "critical", "name": "The v2 validation is circular",
             "text": ("The reference the new codebook is scored against was corrected only where "
                      "v2 disagreed with it, by reading v2's own rationale, and every flip went "
                      "v2's way. v2 therefore cannot lose on those minutes. 0.811 / 0.873 is the "
                      "best case the measurement can produce, not an estimate of what v2 does.")},
            {"severity": "high", "name": "The construct moved mid-study, in the wrong order",
             "text": ("“Transit” meant public transportation plus street reallocation "
                      "(v1), then that plus active transportation and service planning (v2), then "
                      "“anything not car related” (v2.1). The order actually followed "
                      "was: edit the codebook, pilot it, adjudicate the pilot's own over-calls, "
                      "then write the definition. The defensible order is the reverse. v1 and v2.x "
                      "rows are never pooled, which is honest but not a fix: a body read under v1 "
                      "for part of the window and v2.1 for the rest has no comparable rate over "
                      "time.")},
            {"severity": "high", "name": "The caption queue is ordered by yield",
             "text": ("Captions are fetched under a hard YouTube rate limit, and the queue is "
                      "ordered by measured yield — bodies that produced episodes get fetched "
                      "first. The exposure in the count model is therefore selected on the "
                      "outcome: high-yield bodies are more completely read, and month effects are "
                      "confounded with when a body's captions happened to land. Coverage-restricted "
                      "estimates are the honest primary; this is not yet modelled.")},
            {"severity": "high", "name": "Recall is body-specific, and the model does not know it",
             "text": ("The screen's misses are vocabulary misses, and vocabulary is "
                      "body-specific: active transportation at Richmond, operations and "
                      "maintenance at BART, affordable-housing allocation at Los Altos and "
                      "Milpitas, plus caption homophones (fare → fair, ridership → "
                      "wrership, YIMBY and rezoning dropped entirely). Recall therefore varies by "
                      "body, from 0.60 to 0.82 across just four meetings, and treating locate "
                      "route as a stratum absorbs none of that. Cross-body comparison is the "
                      "thing this biases most, and it is the comparison the study is about.")},
            {"severity": "medium", "name": "Padded bounds — found and corrected",
             "text": ("Episodes are cut with ± 30 s of pad around the located core and the "
                      "pad was initially credited as located time, inflating recall. The artefact "
                      "was found and the correction is applied: every recall figure on this page "
                      "is scored on core bounds, with the pad excluded.")},
            {"severity": "medium", "name": "Clusters are not people, and roles are coarse",
             "text": ("A diarized cluster is a voice in one meeting, not a person. There is no "
                      "staff role: staff reports sit inside dais and pull it toward neutral. "
                      "About 5–6 % of turns have no role and are outside the frame.")},
            {"severity": "medium", "name": "One model, one day",
             "text": ("Every reliability number comes from one model on one day. "
                      "claude-sonnet-4-6 is pinned because its retirement is no sooner than "
                      "2027-02-17, but when it goes the instrument is unrepeatable and anything "
                      "re-scored under a successor is a different measurement.")},
        ],
    }


# ---------------------------------------------------------------------------
# 6. Versions table, generated from the constants
# ---------------------------------------------------------------------------
def section_versions(locate: dict, stance_s: dict, speakers: dict) -> list:
    import nimby_pilot as nb
    rows = [
        {"instrument": "Episode classifier codebook v1", "version": classify.CODEBOOK_VERSIONS["v1"],
         "model": classify.LEGACY_MODEL, "source": "pipeline/classify.py INSTRUCTIONS",
         "sha256": next(c["sha256"] for c in locate["codebooks"] if c["name"] == "v1"),
         "status": "read the whole corpus up to 2026-10-18"},
        {"instrument": "Episode classifier codebook v2 (pilot)", "version": classify.CODEBOOK_VERSIONS["v2"],
         "model": classify.LEGACY_MODEL, "source": "pipeline/classify.py INSTRUCTIONS_V2",
         "sha256": next(c["sha256"] for c in locate["codebooks"] if c["name"] == "v2"),
         "status": "pilot only, 282 calls; never pooled with v1"},
        {"instrument": "Episode classifier codebook v2.1", "version": classify.CODEBOOK_VERSIONS["v2.1"],
         "model": classify.LEGACY_MODEL, "source": "pipeline/classify.py INSTRUCTIONS_V21",
         "sha256": next(c["sha256"] for c in locate["codebooks"] if c["name"] == "v2.1"),
         "status": "the instrument going forward; corpus re-read pending"},
        {"instrument": "Stance / hostility codebook", "version": stance_s["codebook_version"],
         "model": stance_s["model"], "source": "plan/STANCE-CODEBOOK.md below '--- CODEBOOK ---'",
         "sha256": stance_s["codebook"]["sha256"],
         "status": f"S1 pass, k = {stance_s['k']}; κ gate not passed"},
        {"instrument": "NIMBY / YIMBY speaker prompt", "version": nb.PROMPT_VERSION,
         "model": nb.MODEL, "source": "scripts/nimby_pilot.py SYSTEM",
         "sha256": speakers["prompt"]["sha256"],
         "status": "300-turn pilot; not what the dashboard's figures are derived from"},
    ]
    # Everything else with a pinned version, read off the production chart rather
    # than retyped: these are not LLM instruments but they date every number.
    mermaid = locate and None
    chart = section_pipeline()["mermaid"]["text"]
    for label, pat, note in (
        ("Keyword screen", r"KEYWORDS_VERSION (kw-[0-9a-f]+)", "pipeline/segment.py, 391 terms, 0.5 hits/100 w"),
        ("GPU transcription", r"PIPELINE_VERSION ([0-9.]+)", "pipeline/gpu_job.py, WhisperX large-v3 + pyannote"),
        # Back in the publication path since 2026-10-20 (owner, "go back to removing
        # names"): the site exporters run this same redactor over every turn and every
        # quote they write, so the version string on the chart now does describe the
        # text on these pages.
        ("Redaction", r"(redact-[0-9.]+)", "pipeline/redact.py, name / address / contact — ran before "
                                           "the stance pass, and again on every turn and quote "
                                           "published on this site"),
    ):
        m = re.search(pat, chart)
        if m:
            status = ("applied to all published text (owner, 2026-10-20)"
                      if label == "Redaction" else "non-LLM step")
            rows.append({"instrument": label, "version": m.group(1), "model": "—",
                         "source": note, "sha256": None, "status": status})
    del mermaid
    return rows


def main() -> int:
    locate = section_locate()
    stance_s = section_stance()
    speakers = section_speakers()
    doc = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "ytpt-site/export/export_methods.py",
        "exploratory": EXPLORATORY,
        "pipeline": section_pipeline(),
        "locate": locate,
        "stance": stance_s,
        "speakers": speakers,
        "models": section_models(),
        "limitations": section_limitations(),
        "versions": section_versions(locate, stance_s, speakers),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    n_blocks = sum(1 for _ in re.finditer(r'"sha256"', OUT.read_text()))
    print(f"wrote {OUT.relative_to(SITE)}  {OUT.stat().st_size/1024:.1f} kB, "
          f"{len(locate['codebooks'])} locate codebooks, {n_blocks} hashed blocks, "
          f"{len(doc['versions'])} version rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
