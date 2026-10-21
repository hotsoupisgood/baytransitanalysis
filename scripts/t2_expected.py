#!/usr/bin/env python3
"""
scripts/t2_expected.py -- the T2 numbers from what discovery found (T2 part 1,
2026-09-22; plan/T2-PLAN.md 3). Reads kind=meeting rows in the window from the
state DB and applies the T0 rates:

  episode share      councils 18.8 %, boards 39.6 % of meeting time (T0-RESULTS test 2)
  pad                +9 % (T0 test 4 / cue snap)
  per-body cap       CAP_H transcribed episode-hours per body per month (PROPOSAL 2.3, default 3)
  classifier         $0.0222 per meeting-hour direct on Sonnet 4.6 -> x 0.5 via Batches
  GPU                $0.021 per audio-hour at the measured 25x (b9, b11, b13, b15)
  fetch              YouTube: one request per meeting, 7.7 s per meeting-h + a 45-180 s gap
                     (mean 112.5 s); Granicus: 410 MB per meeting-hour on the wire for the
                     episode minutes only

A meeting without a duration (Sausalito, SamTrans listings carry none) takes
the body's mean, else 3.8 h. Prints a Markdown table and the totals; --json for
the raw numbers.

    python3 scripts/t2_expected.py [--cap-h 3] [--since 2025-09-20] [--json]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state  # noqa: E402
import discover  # noqa: E402

SHARE = {"council": 0.188, "county": 0.188, "transit_board": 0.396}
PAD = 1.09
CLASSIFIER_USD_PER_H = 0.0222
BATCH_DISCOUNT = 0.5
GPU_USD_PER_AUDIO_H = 0.021
YT_S_PER_MEETING_H, YT_GAP_S = 7.7, 112.5
GR_MB_PER_MEETING_H = 410.0
DEFAULT_H = 3.8


def compute(con, cap_h=3.0, since="2025-09-20", until="2026-09-20"):
    rows = [dict(r) for r in con.execute(
        "SELECT video_id, speaker, platform, meeting_date, duration_s, captions, status FROM videos "
        "WHERE kind = 'meeting' AND meeting_date >= ? AND meeting_date <= ?", (since, until))]
    by_body = defaultdict(list)
    for r in rows:
        by_body[r["speaker"]].append(r)
    out = []
    for body, ms in sorted(by_body.items()):
        kind = discover.body_kind_of(body)
        known = [m["duration_s"] / 3600 for m in ms if m["duration_s"]]
        mean_h = sum(known) / len(known) if known else DEFAULT_H
        hours = sum((m["duration_s"] / 3600) if m["duration_s"] else mean_h for m in ms)
        share = SHARE[kind]
        raw_ep_h = hours * share * PAD
        months = defaultdict(float)
        for m in ms:
            months[m["meeting_date"][:7]] += ((m["duration_s"] / 3600) if m["duration_s"] else mean_h) * share * PAD
        capped_ep_h = sum(min(v, cap_h) for v in months.values())
        platform = ms[0]["platform"]
        out.append({"body": body, "kind": kind, "platform": platform, "meetings": len(ms), "months": len(months),
                    "meeting_h": round(hours, 1), "mean_h": round(mean_h, 2), "durations_missing": sum(1 for m in ms if not m["duration_s"]),
                    "captioned": sum(1 for m in ms if m["captions"] and m["captions"] != "none"),
                    "episode_h_raw": round(raw_ep_h, 1), "episode_h_capped": round(capped_ep_h, 1),
                    "cap_binds_months": sum(1 for v in months.values() if v > cap_h),
                    "classifier_usd_batch": round(hours * CLASSIFIER_USD_PER_H * BATCH_DISCOUNT, 2),
                    "gpu_usd": round(capped_ep_h * GPU_USD_PER_AUDIO_H, 2),
                    "fetch_h": round((len(ms) * YT_GAP_S + hours * 3600 * YT_S_PER_MEETING_H / 3600) / 3600, 1) if platform == "youtube" else None,
                    "wire_gb": round(hours * GR_MB_PER_MEETING_H * share / 1000, 1) if platform == "granicus" else round(hours * 45.1 / 1000, 1)})
    tot = {k: round(sum(b[k] for b in out), 1) for k in ("meetings", "meeting_h", "episode_h_raw", "episode_h_capped", "classifier_usd_batch", "gpu_usd", "wire_gb", "captioned")}
    tot["fetch_h_youtube"] = round(sum(b["fetch_h"] or 0 for b in out), 1)
    tot["bodies"] = len(out)
    tot["batches_at_50h"] = max(1, round(tot["episode_h_capped"] / 50 + 0.4999))
    return out, tot


def table(out, tot, cap_h):
    lines = ["| body | kind | platform | meetings | meeting-h | ep-h raw | ep-h capped | cap binds (months) | classifier $ (batch) | GPU $ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for b in out:
        lines.append(f"| {b['body']} | {b['kind']} | {b['platform']} | {b['meetings']} | {b['meeting_h']}{' *' if b['durations_missing'] else ''} | "
                     f"{b['episode_h_raw']} | {b['episode_h_capped']} | {b['cap_binds_months']}/{b['months']} | ${b['classifier_usd_batch']:.2f} | ${b['gpu_usd']:.2f} |")
    lines.append(f"| **total** | | | **{tot['meetings']:.0f}** | **{tot['meeting_h']:.0f}** | **{tot['episode_h_raw']:.0f}** | **{tot['episode_h_capped']:.0f}** | | "
                 f"**${tot['classifier_usd_batch']:.2f}** | **${tot['gpu_usd']:.2f}** |")
    lines.append(f"\n(* mean-duration fill for listings without durations; cap {cap_h:g} episode-h per body per month; "
                 f"YouTube fetch ≈ {tot['fetch_h_youtube']} h wall at one request per meeting; wire ≈ {tot['wire_gb']} GB; "
                 f"≈ {tot['batches_at_50h']} GPU batches of ~50 episode-h)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-h", type=float, default=3.0)
    ap.add_argument("--since", default="2025-09-20")
    ap.add_argument("--until", default="2026-09-20")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state, create=False)
    out, tot = compute(con, a.cap_h, a.since, a.until)
    print(json.dumps({"bodies": out, "totals": tot}, indent=1) if a.json else table(out, tot, a.cap_h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
