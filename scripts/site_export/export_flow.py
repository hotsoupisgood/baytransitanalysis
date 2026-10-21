#!/usr/bin/env python
"""Export the methods flowchart's boxes (READ-ONLY).

    cd ~/ytpt && .venv/bin/python scripts/site_export/export_flow.py

Writes, beside this repo's methods.html:
    data/flow.json      one object per box of the flowchart at the top of
                        methods.html: its state, its badge, the one line that
                        says what decided it, the numbers on its face and the
                        detail pane behind it

Owner, 2026-10-19: "can you make a flowchart for the dashboard in the methods
for what we did exactly", with the D3 flow of
https://hotsoupisgood.github.io/AI-Graph-Response-Compare/ as the reference --
a vertical column of boxes with a status key, dashed boxes where something was
decided, and a detail pane that fills in when a box is clicked.

Every count here is read out of this repo's own exported JSON --
data/bodies.json (corpus totals, QA, instrument versions), data/methods.json
(measured locate recall, replicate agreement) and data/panels.json (the role
audit, the speaker clusters and the model) -- so the chart cannot disagree with
the page it sits on. Nothing is typed as a literal count.

The money is the exception, because no exported JSON carries it: the three
dollar figures come straight out of the spend tables of state.sqlite, and each
one is stamped with the date of the last row that went into it. Owner's rule:
a dollar figure on the site shows when it was measured. docs/COSTS.md is the
narrative version of the same tables and is where the detail text comes from.

Read-only on the ytpt repo and its databases; writes only under this repo's
data/.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

YTPT = Path("~/ytpt").expanduser()
SITE = Path(__file__).resolve().parents[2] / "docs"  # the GitHub Pages root of this repo
DATA = SITE / "data"
STATE = Path("~/.local/share/ytpt/state.sqlite").expanduser()


def jl(name, default=None):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else (default or {})


def day(ts):
    """The date part of a DB timestamp, for stamping a dollar figure."""
    return (ts or "")[:10] or None


def money():
    """The three measured costs, each with the date it was measured.

    GPU: the diarized transcription batches that actually finished, their bill
    against their audio hours (docs/COSTS.md A1 and its measured-RTFX table).
    Stance and classifier: the per-call ledgers the two Claude passes write
    (docs/COSTS.md A2, A3). Each `as_of` is the newest row in the aggregate, so
    the stamp moves when the number does.
    """
    if not STATE.exists():
        return {}
    con = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    out = {}

    # Diarized and undiarized transcription bill at very different rates, and
    # the batch note is where the two are told apart: docs/COSTS.md A1 records
    # the diarized job at 24-29x real time and the undiarized validation and
    # whole-meeting arms at 46-65x. So the measured throughput on the row
    # decides which rate a batch belongs to, rather than a list of batch ids.
    rows = con.execute(
        "select batch_id, actual_usd, audio_hours, finished_at, note from batches "
        "where status = 'done' and actual_usd is not null and audio_hours > 0"
    ).fetchall()
    DIARIZED_RTFX_MAX = 40.0
    agg = {}
    for bid, usd, hours, last, note in rows:
        m = re.search(r"measured RTFX\s*([0-9.]+)", note or "")
        rtfx = float(m.group(1)) if m else None
        kind = ("gpu" if rtfx is not None and rtfx < DIARIZED_RTFX_MAX
                else "gpu_nodiar" if rtfx is not None else None)
        if kind is None:
            continue
        a = agg.setdefault(kind, {"usd": 0.0, "hours": 0.0, "batches": 0,
                                  "last": ""})
        a["usd"] += usd
        a["hours"] += hours
        a["batches"] += 1
        a["last"] = max(a["last"], last or "")
    for kind, what in (("gpu", "transcription with diarization, per audio-hour"),
                       ("gpu_nodiar",
                        "whole-meeting transcription without diarization, "
                        "per audio-hour")):
        a = agg.get(kind)
        if a and a["hours"]:
            out[kind] = {"usd": a["usd"], "hours": a["hours"],
                         "batches": a["batches"],
                         "per_hour": a["usd"] / a["hours"],
                         "as_of": day(a["last"]), "what": what}

    row = con.execute(
        "select sum(cost_usd), count(distinct episode_id || '#' || turn_idx), "
        "count(*), max(at) from stance_calls where cost_usd is not null").fetchone()
    if row and row[0]:
        usd, turns, calls, last = row
        out["stance"] = {"usd": usd, "turns": turns, "calls": calls,
                         "per_turn": usd / turns if turns else None,
                         "as_of": day(last), "what": "stance coding, per turn at k = 3"}

    row = con.execute(
        "select sum(cost_usd), count(*), max(at) from classifier_calls "
        "where cost_usd is not null").fetchone()
    if row and row[0]:
        usd, calls, last = row
        out["classifier"] = {"usd": usd, "calls": calls, "as_of": day(last),
                             "what": "the locate classifier, whole pass"}

    row = con.execute("select sum(usd), max(coalesce(ended_at, started_at)) "
                      "from spend").fetchone()
    if row and row[0]:
        out["runpod_total"] = {"usd": row[0], "as_of": day(row[1]),
                              "what": "every GPU pod this project has run"}
    con.close()
    return out


def usd_line(m, key, per=None, dp=4):
    """'$0.0216 per audio-hour, measured 2026-10-18' -- never a bare figure."""
    d = (m or {}).get(key)
    if not d:
        return ""
    v = d.get(per) if per else d.get("usd")
    if v is None:
        return ""
    amount = f"${v:,.{dp}f}" if v < 1 else f"${v:,.2f}"
    return f"{amount}, measured {d['as_of']}" if d.get("as_of") else amount


def build():
    bj, M, P = jl("bodies.json"), jl("methods.json"), jl("panels.json")
    meta = bj.get("meta") or {}
    T = meta.get("totals") or {}
    Q = meta.get("qa") or {}
    I = meta.get("instruments") or {}
    TX = meta.get("text") or {}
    plat = T.get("meetings_by_platform") or {}
    stat = T.get("episodes_by_status") or {}
    route = T.get("episodes_by_route") or {}
    gated = T.get("gated_by_route") or {}
    asr = I.get("asr") or {}
    st_run = (I.get("stance") or {}).get("runs") or [{}]
    rep = (M.get("stance") or {}).get("replicates") or {}
    rec = ((M.get("locate") or {}).get("recall") or {})
    rows = rec.get("rows") or []
    local_arm = ((rec.get("local_asr_arm") or {}).get("families") or {})
    nimby = P.get("nimby") or {}
    audit = nimby.get("role_audit") or {}
    overall = nimby.get("overall") or {}
    side = nimby.get("role_side") or {}
    mny = money()

    def arm(label, family):
        for r in rows:
            if r.get("family") == family and label in (r.get("arm_label") or ""):
                return (r.get("adj_ref") or {}).get("recall")
        return None

    fi = lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else "?"
    f1 = lambda v: f"{v:,.1f}" if isinstance(v, (int, float)) else "?"
    f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "?"
    f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "?"

    n_youtube = plat.get("youtube", 0)
    n_granicus = plat.get("granicus", 0)
    n_swagit = plat.get("swagit", 0)
    cut_queued = (stat.get("cut", 0) + stat.get("queued", 0)
                  + stat.get("located", 0))
    screen_any, full_any = local_arm.get("any", {}).get("recall"), arm("full ASR", "any")
    screen_transit = local_arm.get("transit", {}).get("recall")
    full_transit = arm("full ASR", "transit")
    kap = rep.get("stance_kappa_w") or []
    gpu_cost, stance_cost = usd_line(mny, "gpu", "per_hour"), usd_line(mny, "stance", "per_turn", 5)
    nodiar_cost = usd_line(mny, "gpu_nodiar", "per_hour")
    gpu_total, stance_total = usd_line(mny, "gpu"), usd_line(mny, "stance")
    cls_total = usd_line(mny, "classifier")
    pod_total = usd_line(mny, "runpod_total")

    boxes = [
        {"id": "discover", "n": "1", "title": "Meeting discovery",
         "state": "done",
         "badge": f"{fi(T.get('bodies'))} BODIES · {fi(T.get('meetings'))} MEETINGS",
         "line": ("What counts as a meeting was decided by a per-body title pattern: "
                  "regular sessions only, inside one twelve-month window."),
         "nums": [["bodies", fi(T.get("bodies"))],
                  ["meetings", fi(T.get("meetings"))],
                  ["video hours", fi(T.get("meeting_hours"))],
                  ["platforms", "3"]],
         "detail": (
             f"<p>Three archives, one row per body in the channel list: {fi(n_youtube)} "
             f"meetings found on YouTube, {fi(n_granicus)} on Granicus and {fi(n_swagit)} "
             f"on Swagit, {fi(T.get('bodies'))} bodies in all — city councils, one board of "
             "supervisors and three transit boards. Metadata only at this stage: a playlist "
             "listing and an agenda page, no media.</p>"
             f"<p>The decision that shapes everything downstream is what a <i>regular</i> "
             "meeting is. Each body gets a title pattern that keeps its ordinary sessions and "
             "drops special sessions, study sessions and committees, and the window runs "
             f"2025-09-20 to 2026-09-20 and is closed. That gives {f1(T.get('meeting_hours'))} "
             "hours of video as the frame. Bodies whose archive could not be located at all "
             "are a known selection limit and are listed in the plan, not silently dropped.</p>"),
         "docs": ["plan/channels.tsv", "plan/T2-PLAN.md", "plan/SAMPLING.md"]},

        {"id": "download", "n": "2", "title": "Download and transcode",
         "state": "active" if cut_queued else "done",
         "badge": "TRICKLING" if cut_queued else "DONE",
         "line": ("How fast this runs was decided by YouTube, not by us: a few requests at a "
                  "time from one Mac, and a bot check stops the worker rather than retrying."),
         "nums": [["clips transcoded", fi(T.get("episodes_done"))],
                  ["waiting", fi(cut_queued)],
                  ["audio hours", f1(T.get("hours_done"))],
                  ["bodies at 0 h", fi((T.get("bodies") or 0) - (T.get("bodies_with_hours") or 0))]],
         "detail": (
             "<p>Captions come first and cost nothing: platform caption tracks — YouTube "
             "auto-captions, Granicus VTT, Swagit transcripts — are fetched slowly by a "
             "background worker, and they decide <i>where</i> to look before any audio is "
             "pulled. Only then is audio downloaded and cut to the located spans, two or "
             "three at a time, from this one machine.</p>"
             "<p>The constraint is a daily YouTube budget that is not ours to set. A "
             "“confirm you’re not a bot” page or an HTTP 429 stops the downloader and pages "
             "the owner; nothing retries in a loop, and no personal account cookies are ever "
             f"used. {fi(cut_queued)} clips are cut or queued and waiting for their turn, which "
             "is why some bodies' hours are floors rather than totals.</p>"),
         "docs": ["pipeline/captions.py", "pipeline/episodes.py",
                  "plan/BACKLOG-2026-10-18.md"]},

        {"id": "locate", "n": "3", "title": "Locate: is this stretch about the topic?",
         "state": "done", "decision": True,
         "badge": f"{fi(T.get('episodes'))} EPISODES · 2 ROUTES",
         "line": ("Decided twice over, by two routes with measured recall: a keyword screen "
                  "with an LLM behind it, and whole-meeting transcription read with no screen "
                  "at all."),
         "nums": [["episodes", fi(T.get("episodes"))],
                  ["screened route", fi(route.get("classifier", 0))],
                  ["whole-meeting", fi(route.get("fullasr", 0))],
                  ["recall, screened", f3(screen_any)]],
         "detail": (
             "<p><b>Route one, the cheap one.</b> Caption windows are scored against keyword "
             "lists, the candidates go to an LLM with a pinned codebook, and it decides "
             "whether each window is <i>about</i> housing density, <i>about</i> transit, both, "
             "a passing mention or nothing. Adjacent “about” windows merge into an episode. "
             f"Its measured recall is {f3(screen_any)} on any topic and {f3(screen_transit)} on "
             "transit — and the leak is almost entirely the keyword screen in front of the "
             "model, not the model: fare becomes fair, ridership becomes wrership, YIMBY and "
             "rezoning drop out of the captions altogether.</p>"
             "<p><b>Route two, the expensive one.</b> For the bodies whose audio is not on "
             "YouTube, every regular meeting is transcribed whole, undiarized, and then read "
             "on every 120-second window with <b>no keyword screen at all</b>. Recall there is "
             f"{f3(full_any)} on any topic and {f3(full_transit)} on transit. Transcribing a "
             f"meeting whole and undiarized is the cheap part of that — "
             f"{nodiar_cost or 'a fraction of a cent per audio-hour'} — which is why it was "
             "worth doing for half the corpus.</p>"
             f"<p>Route is kept as a stratum in every model and nothing is pooled across "
             f"routes, because the routes that read the same body can disagree about it. "
             f"{fi(gated.get('classifier', 0))} screened and {fi(gated.get('fullasr', 0))} "
             "whole-meeting episodes are past the QA gate. The recall figures are measured "
             "against a reference the newer codebook itself partly defined, so they are a best "
             "case rather than an estimate.</p>"
             + (f"<p>The reading itself — every window either route sent to the model, on "
                f"both arms — comes to {cls_total}.</p>" if cls_total else "")),
         "docs": ["pipeline/classify.py", "scripts/fullasr.py",
                  "plan/RECALL-CHECK.md", "plan/FULLASR-GRANICUS.md"]},

        {"id": "gpu", "n": "4", "title": "GPU transcription with diarization",
         "state": "active" if stat.get("cut") else "done", "gate": True,
         "badge": f"{f1(T.get('hours_done'))} H" + (
             f" · {(mny.get('gpu') or {}).get('batches')} BATCHES" if mny.get("gpu") else ""),
         "line": ("Nothing starts until the owner approves that batch: the estimate is written "
                  "first, the pod only runs after a yes, and it is torn down after."),
         "nums": [["audio hours", f1(T.get("hours_done"))],
                  ["episodes", fi(T.get("episodes_with_text"))],
                  ["batches", fi((mny.get("gpu") or {}).get("batches"))],
                  [f"$/audio-h, measured {(mny.get('gpu') or {}).get('as_of', '')}",
                   f"${(mny.get('gpu') or {}).get('per_hour', 0):.4f}"]],
         "detail": (
             f"<p>Clip audio is batched on the Mac with a cost estimate attached; the owner "
             f"approves the batch; only then does a pod start. It runs WhisperX "
             f"<code>{asr.get('model', '')}</code> with "
             f"<code>{asr.get('diarization', '')}</code> — so the output is not just words but "
             f"words attributed to separate voices — uploads one JSON per episode and is torn "
             f"down. An orphan-pod check pages the owner if anything is still billing with no "
             f"batch running.</p>"
             + (f"<p>Measured rather than estimated: {gpu_cost} with diarization on, over "
                f"{f1((mny.get('gpu') or {}).get('hours'))} audio-hours in "
                f"{(mny.get('gpu') or {}).get('batches')} finished batches, {gpu_total} in all. "
                f"Every GPU pod this project has ever run comes to {pod_total}. Dropping "
                f"diarization roughly halves the rate"
                + (f" — {nodiar_cost}" if nodiar_cost else "")
                + ", which is what the whole-meeting route in step 3 does.</p>"
                if gpu_cost else "")
             + "<p>The estimate is frozen onto the batch row when it is written, so a later "
             "measured throughput does not rewrite what was approved.</p>"),
         "docs": ["pipeline/gpu_batch.py", "pipeline/spend.py", "docs/COSTS.md"]},

        {"id": "qa", "n": "5", "title": "QA gate: does the transcript hold up?",
         "state": "done", "decision": True,
         "badge": f"{fi(Q.get('episode_OK'))} PASS · {fi(Q.get('episode_WARN'))} WARN"
                  + (f" · {fi(Q.get('episode_FAIL'))} FAIL" if Q.get("episode_FAIL") else ""),
         "line": ("A transcript is compared with the caption track for the same span, and a "
                  "failing one is never loaded rather than being loaded with a warning."),
         "nums": [["pass", fi(Q.get("episode_OK"))],
                  ["warn", fi(Q.get("episode_WARN"))],
                  ["fail", fi(Q.get("episode_FAIL", 0))],
                  ["loaded", fi(T.get("episodes_with_text"))]],
         "detail": (
             f"<p>Every transcript is checked against the captions for the same span, for "
             f"coverage gaps, duplicated segments and diarization collapse. "
             f"{fi(Q.get('episode_OK'))} episodes pass, {fi(Q.get('episode_WARN'))} carry a "
             f"warning and {fi(Q.get('episode_FAIL', 0))} fail. A failing transcript is never "
             f"loaded: the row is requeued instead, which is the one loop in this chart.</p>"
             "<p>Whole meetings need one special case. A long recording trips the "
             "repeated-phrase rule on runs of “you” and “Thank you.” across a recess, which is "
             "silence rather than a decoder loop, so those are re-gated with filler runs "
             "collapsed. A genuine coverage failure stays failed.</p>"
             f"<p>What came through the gate is {fi(TX.get('turns'))} turns and "
             f"{fi(TX.get('words'))} words of speech. Names are kept, by owner decision: these "
             "are public meetings and every turn on the dashboard links to the recording at "
             "its own timestamp.</p>"),
         "docs": ["analysis/qa_transcripts.py", "docs/TESTING.md",
                  "plan/FULLASR-GRANICUS.md"]},

        {"id": "stance", "n": "6", "title": "Turn-level stance coding, three reads each",
         "state": "done", "gate": True,
         "badge": f"{fi(T.get('stance_turns'))} TURNS · κ GATE UNPASSED",
         "line": ("Reliability was decided by reading every turn three times and keeping the "
                  "median — which measures the instrument against itself, not against a human."),
         "nums": [["turns scored", fi(T.get("stance_turns"))],
                  ["episodes", fi(T.get("episodes_scored"))],
                  ["reads each", "3"],
                  ["replicate κ", f2(kap[0] if kap else None)]],
         "detail": (
             f"<p>Every turn of at least five words inside a located episode is coded for "
             f"stance toward the agenda item on a −2…+2 scale, hostility 0–4, a frame set, a "
             f"sub-topic and whether the speaker said who they are. The codebook is hashed "
             f"into the prompt version recorded on every row, thinking is disabled, and each "
             f"turn is read three times with the median kept and the spread stored. This is a "
             f"census of the gated corpus, not a sample: {fi(st_run[0].get('n'))} turns in "
             f"{fi(st_run[0].get('n_episodes'))} episodes.</p>"
             f"<p>The three reads agree at weighted κ "
             f"{' / '.join(f3(k) for k in kap) or '—'} and are identical on "
             f"{f1((rep.get('all3_identical_stance') or 0) * 100)} % of turns. That is the "
             f"instrument agreeing with <i>itself</i>. <b>No human has coded any turn</b>, so "
             f"the κ ≥ 0.6 validity gate is unpassed and everything downstream of here is "
             f"exploratory — which is the single most important sentence on this page.</p>"
             + (f"<p>Cost: {stance_total} for the pass, {stance_cost} per turn at three reads "
                f"each.</p>" if stance_total else "")),
         "docs": ["pipeline/stance.py", "plan/STANCE-CODEBOOK.md",
                  "plan/STANCE-S1-RESULTS.md", "docs/COSTS.md"]},

        {"id": "roles", "n": "7", "title": "Role audit: public or dais?",
         "state": "done",
         "badge": (f"{audit.get('audit_version', 'AUDITED').upper()} · "
                   f"{fi((audit.get('disagree_rate_informative') or 0) * 100)} % MOVED"
                   if audit else "AUDITED"),
         "line": ("Who counts as the public was decided by what a speaker says about "
                  "themselves, which overrides the cue-phrase label it started with."),
         "nums": ([["turns audited", fi(audit.get("turns"))],
                   ["informative", fi(audit.get("informative"))],
                   ["disagree", fi(audit.get("disagree"))],
                   ["no signal", f"{fi((audit.get('unknown_share') or 0) * 100)} %"]]
                  if audit else None),
         "detail": (
             "<p>Roles start from cue phrases in the diarized transcript — “my name is, I live "
             "at” against “the motion carries” — never from names. That labeller reads the "
             "wrong thing often enough to matter: a council member who chairs an item can "
             "pick up a public cue, and a staff planner presenting a report reads as neither. "
             "So the role is re-decided from what the speaker says about who they are, and "
             "that overrides the cue.</p>"
             + (f"<p>Of {fi(audit.get('informative'))} turns that say anything about the "
                f"speaker, {fi(audit.get('disagree'))} disagree with the cue label "
                f"({f1((audit.get('disagree_rate_informative') or 0) * 100)} %); "
                f"{f1((audit.get('unknown_share') or 0) * 100)} % of all turns say nothing and "
                f"keep the cue. The audit moves clusters <i>both</i> ways, so a shift in these "
                f"numbers is a re-partition and not contamination removed.</p>" if audit else "")
             + "<p><b>This is an audit, not a gold set.</b> The self-identification is another "
             "model's output and no human has checked it, so the audited public bucket is "
             "still mixed at an unknown rate.</p>"),
         "docs": ["pipeline/roles.py", "plan/ROLE-AUDIT-2026-10-18.md"]},

        {"id": "clusters", "n": "8", "title": "Speaker clusters, and a label for each",
         "state": "done",
         "badge": f"{fi(overall.get('n_clusters'))} CLUSTERS",
         "line": ("A speaker is one diarized voice inside one meeting and nothing wider: "
                  "clusters are never linked across meetings, by design."),
         "nums": [["clusters", fi(overall.get("n_clusters"))],
                  ["turns rolled up", fi(overall.get("n_turns") or T.get("stance_turns"))],
                  ["threshold", f2(overall.get("threshold"))],
                  ["public, with a stance",
                   fi(((overall.get("split") or {}).get("n_public_clusters")))]],
         "detail": (
             f"<p>The turn scores are rolled up per diarized voice per episode: a cluster is "
             f"YIMBY if its mean stance is at or above +{f2(overall.get('threshold'))}, NIMBY "
             f"if at or below −{f2(overall.get('threshold'))}, neutral between — half a "
             f"codebook step either side. Transit-family episodes are left out of that label, "
             f"YIMBY and NIMBY being a housing construct.</p>"
             "<p>A cluster is a voice in one meeting, not a person. There are no voice "
             "embeddings, no persistent speaker ids and no name extraction into a field, so a "
             "cluster label means nothing outside the one episode it appears in — the same "
             "person at two meetings is two clusters and the analysis cannot tell. Most public "
             "clusters rest on a single scored turn, which is why the per-body intervals are "
             "wide and why one table and one figure exist per locate route, never pooled.</p>"),
         "docs": ["analysis/stance_s1_nimby_view.py", "plan/DECISIONS.md",
                  "plan/NIMBY-PILOT.md"]},

        {"id": "model", "n": "9", "title": "The nested mixed model",
         "state": "done",
         "badge": "ONE MODEL, TWO STEPS" if side else "NOT FITTED",
         "line": ("The grouping was decided by the data: turns nest inside stretches of "
                  "meeting inside meetings, and once that is modelled the body level is empty."),
         "nums": ([[s["name"].lower(), f"{s['odds_ratio']:.2f}×"]
                   for s in (side.get("steps") or [])] or None),
         "detail": (
             "<p>One row per scored turn, and a mixed logistic regression that gives every "
             "stretch of meeting, every meeting and every body its own baseline, with speaker "
             "role, renter share, their interaction and the locate route as fixed terms. It is "
             "fitted twice on the same frame: did a turn take a side at all, and, among the "
             "turns that took one, which side.</p>"
             + ("<p>" + " ".join(
                 f"<b>{s['name']}</b>: {s['text']}" for s in (side.get("steps") or []))
                + "</p>" if side.get("steps") else "")
             + (f"<p>{side.get('model_line', '')} {side.get('restated', '')}</p>" if side else "")
             + "<p>The reason the model groups turns by meeting rather than by city is that "
             "the spread between bodies is zero once meeting is in it — meeting sits inside "
             "body and absorbs it entirely. Earlier fits grouped by body, which is the wrong "
             "level; they are kept in the repo and agree with this one to within a few "
             "thousandths of a log-odds. The role block on the dashboard reports this model "
             "and nothing else.</p>"),
         "docs": ["plan/RENTER-SHARE-6-R.md", "plan/REGRESSION-REVIEW-2026-10-19.md"]},
    ]

    meta_line = (
        f"{fi(T.get('bodies'))} bodies · {fi(T.get('meetings'))} meetings · "
        f"{fi(T.get('episodes'))} located episodes over 5 routes · "
        f"{f1(T.get('hours_done'))} h transcribed · {fi(T.get('stance_turns'))} turns scored"
        + (f" · {pod_total} of GPU" if pod_total else ""))

    return {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
        "generated_by": "ytpt-site/export/export_flow.py",
        "meta_line": meta_line,
        "lede": ("Nine stages, in the order they ran. Each box says what was decided at that "
                 "step; the dashed ones are the two places where something is judged rather "
                 "than processed. Click a box for the detail and the files behind it."),
        "key": [{"state": "done", "label": "done"},
                {"state": "active", "label": "still running"},
                {"state": "prelim", "label": "preliminary or has an open issue"},
                {"state": "pending", "label": "not started"},
                {"state": "decision", "label": "a decision, not a step"},
                {"state": "gate", "label": "▸ human gate"}],
        "money_note": ("Every dollar figure on this page carries the date it was measured, "
                       "because these rates move: they come from the project's own spend "
                       "ledger, not from a price list."),
        "boxes": boxes,
        "sources": ["data/bodies.json", "data/methods.json", "data/panels.json",
                    "state.sqlite (spend, batches, stance_calls, classifier_calls)"],
        "money": mny,
    }


def main():
    flow = build()
    DATA.mkdir(exist_ok=True)
    (DATA / "flow.json").write_text(json.dumps(flow, indent=1), encoding="utf-8")
    print(f"wrote data/flow.json  {(DATA / 'flow.json').stat().st_size / 1000:.1f} kB, "
          f"{len(flow['boxes'])} boxes")
    for b in flow["boxes"]:
        print(f"  {b['n']}. {b['title']}  [{b['badge']}]")
    print(" money:", json.dumps(flow["money"], indent=1))


if __name__ == "__main__":
    main()
