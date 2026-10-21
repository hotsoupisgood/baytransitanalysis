#!/usr/bin/env python3
"""
scripts/codebook_v2_adjudicate.py -- $0 re-score of the codebook v2 pilot
against the ADJUDICATED reference (plan/CODEBOOK-V2-PILOT.md §8-§9, 2026-10-18).

Puck judged all 11 v2@120 over-call windows (§5) "about". This script flips
the reference 60-s minutes those windows cover from mention_only / no to the
about_* label v2 assigned, writes the corrected reference to
output/validation/codebook_v2/adjudicated/reference_adjudicated.jsonl (first
line = provenance header, no `label` key; the v1-coded reference files are not
touched), and recomputes recall / precision per family, with the same
meeting-cluster percentile bootstrap, for every arm already scored on these
four meetings:

    v1_240, v1_120, v2_120, v2_240   from windows_reference_with_arms.jsonl
                                     (ep_<arm> / tags_<arm> columns; 4 clusters)
    v2_fullasr                       from ../codebook_v2_fullasr/results.json
                                     episodes (6 clusters, as that arm's doc)

Nothing is sent anywhere; no DB access; no file under output/validation/recall*
or the four pilot arm files is modified.
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import recall_check as rc   # noqa: E402  (FAMILIES, TAG_FAMILY, covered, ratio_ci, hms; no DB)

PILOT = ROOT / "output" / "validation" / "codebook_v2"
FULL = ROOT / "output" / "validation" / "codebook_v2_fullasr"
OUT = PILOT / "adjudicated"
ARMS = ["v1_240", "v1_120", "v2_120", "v2_240", "v21_120"]      # v21_120 (PILOT 10) scored when its columns exist
ABOUT = {"about_density", "about_transit", "about_both"}
HALF, W = 30.0, 60.0
ADJUDICATED_AT = "2026-10-18T09:33-07:00"


def s_of(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = [json.loads(l) for l in (PILOT / "windows_reference_with_arms.jsonl").read_text().splitlines() if l.strip()]
    pilot_res = json.loads((PILOT / "results.json").read_text())
    full_res = json.loads((FULL / "results.json").read_text())
    vids = pilot_res["meetings"]
    arms = [a for a in ARMS if f"ep_{a}" in ref[0]]
    over = pilot_res["arms"]["v2_120"]["overcalls"]
    assert len(over) == 11, len(over)

    # --- 1. flip the adjudicated minutes --------------------------------------------
    flipped = []
    for o in over:
        a, b = s_of(o["start"]), s_of(o["end"])
        for r in ref:
            if r["video_id"] == o["video_id"] and r["start_s"] >= a and r["end_s"] <= b:
                assert r["label"] not in ABOUT, (o, r["start_s"], r["label"])
                flipped.append({"video_id": r["video_id"], "start": rc.hms(r["start_s"]), "from": r["label"], "to": o["arm_label"],
                                "window": f"{o['start']}-{o['end']}"})
                r["label_v1"] = r["label"]
                r["label"] = o["arm_label"]
                r["adjudicated"] = {"by": "Puck", "at": ADJUDICATED_AT, "window": f"{o['video_id']} {o['start']}-{o['end']}",
                                    "v2_label": o["arm_label"], "v2_rationale": o["rationale"]}
    header = {"_header": "reference_adjudicated", "written": rc.state.now_iso(),
              "source": str(PILOT / "windows_reference_with_arms.jsonl"),
              "coding": "codebook v1 t0-a4a8e55760 on 60-s windows (output/validation/recall/post_kw2/windows.jsonl), "
                        "with the minutes inside the 11 v2@120 over-call windows of plan/CODEBOOK-V2-PILOT.md §5 flipped to the "
                        "about_* label codebook v2 (v2-ed4c72c9d6) assigned, per Puck's adjudication (WhatsApp, 2026-10-18 09:33 PDT, "
                        "'yes to all'; §8). Flipped rows carry `label_v1` and `adjudicated`.",
              "n_windows_flipped": 11, "n_minutes_flipped": len(flipped),
              "flipped_from": dict(Counter(f["from"] for f in flipped)), "flipped_to": dict(Counter(f["to"] for f in flipped)),
              "meetings": vids, "n_ref_minutes": len(ref), "ref_labels": dict(Counter(r["label"] for r in ref))}
    (OUT / "reference_adjudicated.jsonl").write_text(
        json.dumps(header, ensure_ascii=False) + "\n" + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ref))

    # --- 2. coverage per reference minute for the full-ASR arm -----------------------
    eps_full = {}
    for e in full_res["episodes"]:
        eps_full.setdefault(e["video_id"], []).append(e)
    for r in ref:
        eps = eps_full.get(r["video_id"], [])
        r["ep_v2_fullasr"] = rc.covered([(e["start_s"], e["end_s"]) for e in eps], r["start_s"], r["end_s"])
        r["tags_v2_fullasr"] = sorted({e["tags"] for e in eps if e["start_s"] < r["end_s"] and e["end_s"] > r["start_s"]})

    # --- 3. score every arm, before (v1-coded) and after (adjudicated) --------------
    full_clusters = [m["video_id"] for m in full_res["families"]["any"]["per_meeting"]]   # 6, two zero clusters

    def score(arm, label_key):
        clusters = full_clusters if arm == "v2_fullasr" else vids
        fams = {}
        for fam, labels in rc.FAMILIES.items():
            per_m = []
            for vid in clusters:
                rows = [r for r in ref if r["video_id"] == vid]
                ab = [r for r in rows if r[label_key] in labels]
                cov = [r for r in rows if r[f"ep_{arm}"] >= HALF and any(t in rc.TAG_FAMILY[fam] for t in r[f"tags_{arm}"])]
                per_m.append({"video_id": vid, "about_min": len(ab) * W / 60, "located_min": sum(r[f"ep_{arm}"] for r in ab) / 60,
                              "found_w": sum(1 for r in ab if r[f"ep_{arm}"] >= HALF), "n_w": len(ab),
                              "cov_w": len(cov), "cov_about_w": sum(1 for r in cov if r[label_key] in labels)})
            tot = {k: sum(p[k] for p in per_m) for k in ("about_min", "located_min", "found_w", "n_w", "cov_w", "cov_about_w")}
            fams[fam] = {"about_min": tot["about_min"], "located_min": round(tot["located_min"], 1),
                         "recall": round(tot["located_min"] / tot["about_min"], 3) if tot["about_min"] else None,
                         "recall_ci": rc.ratio_ci(per_m, "located_min", "about_min"),
                         "covered_w": tot["cov_w"], "covered_about_w": tot["cov_about_w"],
                         "precision": round(tot["cov_about_w"] / tot["cov_w"], 3) if tot["cov_w"] else None,
                         "precision_ci": rc.ratio_ci(per_m, "cov_about_w", "cov_w"), "per_meeting": per_m}
        return fams

    for r in ref:
        r.setdefault("label_v1", r["label"])
    res = {"adjudication": header, "arms": {}}
    for arm in arms + ["v2_fullasr"]:
        res["arms"][arm] = {"before": score(arm, "label_v1"), "after": score(arm, "label")}
        # over-calls left after adjudication: 60-s minutes covered >= 30 s by a family-tagged span whose label is still outside the family
        res["arms"][arm]["overcall_minutes_after"] = sum(
            1 for r in ref if r[f"ep_{arm}"] >= HALF and r["label"] not in ABOUT)
        res["arms"][arm]["overcall_minutes_before"] = sum(
            1 for r in ref if r[f"ep_{arm}"] >= HALF and r["label_v1"] not in ABOUT)
    (OUT / "results_adjudicated.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # --- 4. markdown table -------------------------------------------------------------
    f3 = lambda x: "—" if x is None else f"{x:.3f}"
    L = ["| family | arm | about-min before → after | recall before → after | 95 % CI after | precision before → after | 95 % CI after |",
         "|---|---|---|---|---|---|---|"]
    for fam in ("density", "transit", "any", "both_label"):
        for arm in arms + ["v2_fullasr"]:
            b, a = res["arms"][arm]["before"][fam], res["arms"][arm]["after"][fam]
            L.append(f"| {fam} | {arm} | {b['about_min']:.0f} → {a['about_min']:.0f} | {f3(b['recall'])} → **{f3(a['recall'])}** | "
                     f"{f3(a['recall_ci'][0])}–{f3(a['recall_ci'][1])} | {f3(b['precision'])} → **{f3(a['precision'])}** | "
                     f"{f3(a['precision_ci'][0])}–{f3(a['precision_ci'][1])} |")
    L += ["", "| arm | located minutes outside any about_* reference minute, before → after |", "|---|---|"]
    for arm in arms + ["v2_fullasr"]:
        L.append(f"| {arm} | {res['arms'][arm]['overcall_minutes_before']} → {res['arms'][arm]['overcall_minutes_after']} |")
    L += ["", f"Flipped minutes ({len(flipped)}): " + "; ".join(f"{f['video_id']} {f['start']} {f['from']}→{f['to']}" for f in flipped)]
    (OUT / "table_adjudicated.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
