#!/usr/bin/env python3
"""
compare.py -- build the run-6 comparison table: the run-5 custom Gauss-Hermite
GLMM vs lme4::glmer vs glmmTMB beta-binomial vs the run-3 Bayesian R1 fit, on
both denominators, and flag every parameter where two fits differ by more than
0.05 log-odds.

Reads output/renter_share_5/freq/freq_results.json (Python),
output/renter_share_6/r_results.json (R) and
output/renter_share_3/models/model_table.csv (Bayesian).
Writes output/renter_share_6/comparison_r.csv and comparison_gaps.csv.
"""
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "output" / "renter_share_6"
PY = ROOT / "output" / "renter_share_5" / "freq" / "freq_results.json"
BAY = ROOT / "output" / "renter_share_3" / "models" / "model_table.csv"

WANT = ["role", "renter_share", "role_x_renter", "transit_commute",
        "log_population", "log_income", "log_density"]
GAP = 0.05


def rname(p):
    return "role_num" if p == "role" else p


def rtable(rows):
    return {r["parameter"]: r for r in rows}


def main():
    py = json.loads(PY.read_text())
    R = json.loads((OUT / "r_results.json").read_text())

    bay = {}
    for r in csv.DictReader(open(BAY)):
        if r["outcome"] == "nimby" and r["spec"] == "R1" and r["paired_only"] == "0":
            key = {"delta_role": "role",
                   "theta_role_x_renter": "role_x_renter"}.get(
                       r["parameter"], r["parameter"])
            bay[key] = (float(r["median"]), float(r["ci95_lo"]),
                        float(r["ci95_hi"]))

    lines, gaps = [], []
    for tag in ("primary", "secondary"):
        rb = R[tag]
        gl = rtable(rb["glmer"]["wald"])
        prof = rtable(rb["glmer"].get("profile") or [])
        tmb = {k: rtable(v["table"]) for k, v in rb["glmmTMB"].items()}
        pyg = py[tag]["glmm"]["est"]
        pybb = py[tag]["betabinom"]["est"]

        for p in WANT:
            rp = rname(p)
            here = {}
            if p in pyg:
                e = pyg[p]
                here["Python GH-GLMM"] = (e["est"], e["lo"], e["hi"])
                lines.append(dict(denominator=tag, parameter=p,
                                  fit="Python GH-GLMM", est=e["est"],
                                  lo=e["lo"], hi=e["hi"]))
            if rp in gl:
                e = gl[rp]
                here["lme4::glmer"] = (e["est"], e["lo"], e["hi"])
                pl = prof.get(rp)
                lines.append(dict(denominator=tag, parameter=p,
                                  fit="lme4::glmer (Wald)", est=e["est"],
                                  lo=e["lo"], hi=e["hi"]))
                if pl:
                    lines.append(dict(denominator=tag, parameter=p,
                                      fit="lme4::glmer (profile)",
                                      est=e["est"], lo=pl["prof_lo"],
                                      hi=pl["prof_hi"]))
            for nm, lab in (("bb_ri", "glmmTMB beta-binom + RI"),
                            ("bb_fixed", "glmmTMB beta-binom")):
                if nm in tmb and rp in tmb[nm]:
                    e = tmb[nm][rp]
                    here[lab] = (e["est"], e["lo"], e["hi"])
                    lines.append(dict(denominator=tag, parameter=p, fit=lab,
                                      est=e["est"], lo=e["lo"], hi=e["hi"]))
            if p in pybb:
                e = pybb[p]
                lines.append(dict(denominator=tag, parameter=p,
                                  fit="Python beta-binom", est=e["est"],
                                  lo=e["lo"], hi=e["hi"]))
            if tag == "secondary" and p in bay:
                m, l, h = bay[p]
                here["Bayesian R1"] = (m, l, h)
                lines.append(dict(denominator=tag, parameter=p,
                                  fit="Bayesian R1", est=m, lo=l, hi=h))

            names = list(here)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    d = here[names[i]][0] - here[names[j]][0]
                    if abs(d) > GAP:
                        gaps.append(dict(denominator=tag, parameter=p,
                                         fit_a=names[i], fit_b=names[j],
                                         est_a=round(here[names[i]][0], 4),
                                         est_b=round(here[names[j]][0], 4),
                                         diff=round(d, 4)))

    for r in lines:
        for k in ("est", "lo", "hi"):
            r[k] = None if r[k] is None else round(r[k], 4)
    with open(OUT / "comparison_r.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["denominator", "parameter", "fit",
                                           "est", "lo", "hi"])
        w.writeheader(); w.writerows(lines)
    with open(OUT / "comparison_gaps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["denominator", "parameter", "fit_a",
                                           "fit_b", "est_a", "est_b", "diff"])
        w.writeheader(); w.writerows(sorted(gaps, key=lambda r: -abs(r["diff"])))

    print(f"{len(lines)} comparison rows, {len(gaps)} pairs over {GAP} log-odds")
    for g in sorted(gaps, key=lambda r: -abs(r["diff"]))[:15]:
        print(f"  {g['denominator']:9s} {g['parameter']:16s} "
              f"{g['fit_a']} vs {g['fit_b']}: {g['diff']:+.3f}")


if __name__ == "__main__":
    main()
