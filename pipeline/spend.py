#!/usr/bin/env python3
"""
pipeline/spend.py -- the spend ledger and the hard stops (T2, 2026-09-22;
plan/T2-PLAN.md). One place that answers "how much has this project put on
the meter", read by gpu_batch.propose / launch, classify.py and status.py.

RunPod   every dollar the DB knows about, per pod: the `spend` row teardown
         writes (pod uptime x $/h) or, when a pod has none, the batch's
         actual_usd; spend rows without a pod (hand-recorded) count once.
         The ledger matched RunPod's own balance on 2026-09-22 ($3.32 recorded,
         $10.00 - $6.67 on the console), which is the only check that matters.
Claude   SUM(cost_usd) over classifier_calls + stance_calls -- every call is a row before
         anything else happens (classify.py), so this is exact for the
         classifier. Scoring runs outside the DB (analysis/*) are NOT in it.

Caps (settings, .env or the defaults here):
  RUNPOD_CREDIT_USD          10     the prepaid balance Puck loaded
  RUNPOD_CREDIT_MARGIN_USD   0.50   stop proposing / launching once
                                    recorded + estimate >= credit - margin
                                    (the real ceiling is the balance; the
                                    margin covers a teardown that lands late)
  CLAUDE_CAP_USD             10     classify.py refuses a call that could pass it

    python3 pipeline/spend.py            # the two lines status.py shows
    python3 pipeline/spend.py --json
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402

DEFAULTS = {"RUNPOD_CREDIT_USD": 10.0, "RUNPOD_CREDIT_MARGIN_USD": 0.50, "CLAUDE_CAP_USD": 10.0}


def caps():
    return {k: float(state.setting(k, v)) for k, v in DEFAULTS.items()}


def _tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def runpod_total(con):
    """Recorded RunPod dollars (see module docstring). Tolerates a DB without
    the tables (day 1)."""
    t = _tables(con)
    per_pod, no_pod = {}, 0.0
    if "spend" in t:
        for r in con.execute("SELECT pod_id, COALESCE(SUM(usd), 0) u FROM spend GROUP BY pod_id"):
            if r["pod_id"]:
                per_pod[r["pod_id"]] = max(per_pod.get(r["pod_id"], 0.0), float(r["u"] or 0))
            else:
                no_pod += float(r["u"] or 0)
    if "batches" in t:
        for r in con.execute("SELECT pod_id, actual_usd FROM batches WHERE pod_id IS NOT NULL AND actual_usd IS NOT NULL"):
            per_pod[r["pod_id"]] = max(per_pod.get(r["pod_id"], 0.0), float(r["actual_usd"] or 0))
    return round(sum(per_pod.values()) + no_pod, 4)


def claude_total(con):
    """Every Claude dollar the DB knows about: the classifier (classifier_calls)
    and, since 2026-10-17, the stance pass (stance_calls, pipeline/stance.py);
    both write the row before the request, so this is exact."""
    t = _tables(con)
    total = 0.0
    for table in ("classifier_calls", "stance_calls"):
        if table in t:
            total += float(con.execute(f"SELECT COALESCE(SUM(cost_usd), 0) FROM {table}").fetchone()[0] or 0)
    return round(total, 4)


def runpod_stop_line(c=None):
    c = c or caps()
    return round(c["RUNPOD_CREDIT_USD"] - c["RUNPOD_CREDIT_MARGIN_USD"], 2)


def runpod_refusal(con, estimate_usd=0.0, c=None):
    """A string saying why a pod must not be rented, or None. The check is
    recorded + estimate >= stop line, so an estimate of 0 asks 'are we past
    the line already'."""
    c = c or caps()
    total, stop = runpod_total(con), runpod_stop_line(c)
    if total + float(estimate_usd or 0) >= stop:
        return (f"RunPod recorded ${total:.2f} + estimate ${float(estimate_usd or 0):.2f} reaches the stop line "
                f"${stop:.2f} (RUNPOD_CREDIT_USD ${c['RUNPOD_CREDIT_USD']:.2f} - margin ${c['RUNPOD_CREDIT_MARGIN_USD']:.2f}); "
                f"top up the credit and raise RUNPOD_CREDIT_USD, or stop")
    return None


def claude_refusal(con, next_call_usd=0.0, c=None):
    c = c or caps()
    total = claude_total(con)
    if total + float(next_call_usd or 0) > c["CLAUDE_CAP_USD"]:
        return (f"Claude classifier recorded ${total:.4f} + next ${float(next_call_usd or 0):.4f} exceeds CLAUDE_CAP_USD "
                f"${c['CLAUDE_CAP_USD']:.2f}")
    return None


def summary(con):
    c = caps()
    rp, cl = runpod_total(con), claude_total(con)
    return {"runpod_usd": rp, "runpod_credit_usd": c["RUNPOD_CREDIT_USD"], "runpod_stop_usd": runpod_stop_line(c),
            "runpod_left_usd": round(c["RUNPOD_CREDIT_USD"] - rp, 2), "runpod_blocked": runpod_refusal(con, 0, c) is not None,
            "claude_usd": cl, "claude_cap_usd": c["CLAUDE_CAP_USD"], "claude_blocked": claude_refusal(con, 0, c) is not None}


def lines(s):
    """The two status lines (dashboard/status.py as_text)."""
    return [f"RunPod ${s['runpod_usd']:.2f} of ${s['runpod_credit_usd']:.2f} credit (stop at ${s['runpod_stop_usd']:.2f}"
            f"{', BLOCKED' if s['runpod_blocked'] else ''}); Claude classifier ${s['claude_usd']:.2f} of ${s['claude_cap_usd']:.2f}"
            f"{' BLOCKED' if s['claude_blocked'] else ''}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state, create=False)
    s = summary(con)
    print(json.dumps(s, indent=1) if a.json else "\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
