#!/usr/bin/env python3
"""
renter_share2_acs.py -- the WIDE per-body ACS covariate table for
plan/RENTER-SHARE-2.md.

Same source, same vintage, same geography rule as analysis/renter_share_acs.py
(which this imports BODY_GEO from, so the two tables cannot drift apart):
the no-key table-based ACS 5-year Summary File, ACS 2019-2023 5-year,
geography vintage 2023. No Census key, no api.census.gov, no .env.

  https://www2.census.gov/programs-surveys/acs/summary_file/2023/
    table-based-SF/data/5YRData/acsdt5y2023-<table>.dat

Already pulled by renter_share_acs.py (read from output/renter_share/raw/):
  B25003 tenure, B19013 median HH income, B01003 population,
  Geos20235YR.txt, 2023 Gazetteer ALAND.
Pulled new by output/renter_share_2/raw/dl.sh (read from there):
  B25077 median value, B25064 median gross rent, B25070 rent burden,
  B25034 year built, B25024 units in structure, B15003 education,
  B01002 median age, B03002 Hispanic origin by race, B08301 commute mode,
  B25002 occupancy status.

Covariates (all one row per body):
  renter_share        B25003_E003 / B25003_E001
  med_hh_income       B19013_E001
  pop_density         B01003_E001 / ALAND sq mi
  population          B01003_E001
  med_home_value      B25077_E001
  med_gross_rent      B25064_E001
  rent_burden_30      (B25070_E007..E010) / (B25070_E001 - B25070_E011)
                      i.e. share of renter households paying >= 30 % of
                      income on gross rent, among those for whom the ratio
                      is computed ("not computed" excluded from the base)
  built_2010_plus     (B25034_E002 + B25034_E003) / B25034_E001
  multiunit_5plus     (B25024_E005..E008) / B25024_E001
  ba_plus             (B15003_E022..E025) / B15003_E001   (pop 25+)
  median_age          B01002_E001
  nhwhite_share       B03002_E003 / B03002_E001
  transit_commute     B08301_E010 / B08301_E001
  vacancy_rate        B25002_E003 / B25002_E001

Geography rule, unchanged:
  council        -> census PLACE
  county board   -> COUNTY (napa)
  transit board  -> union of MEMBER COUNTIES. BART = Alameda + Contra Costa
                    + San Francisco; VTA = Santa Clara; SamTrans = San Mateo.

Aggregation over a multi-county union:
  * every count-based ratio is sum(numerator) / sum(denominator) -- exact.
  * every ACS MEDIAN (income, home value, gross rent, age) is a
    population-weighted mean of county medians, which is NOT a median. Those
    bodies carry `medians_are_weighted_means = 1`. Only `bart` is affected
    (vta and samtrans are single counties).

Writes output/renter_share_2/covariates_wide.csv and acs_wide.json.
No key, no .env, nothing committed.
"""
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from renter_share_acs import BODY_GEO, SUMLEVEL  # noqa: E402  single source of truth

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW1 = ROOT / "output" / "renter_share" / "raw"        # v1 pull
RAW2 = ROOT / "output" / "renter_share_2" / "raw"      # this pull
OUT = ROOT / "output" / "renter_share_2"

VINTAGE = "ACS 2019-2023 5-year (table-based summary file), geography 2023"
BASE = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
        "table-based-SF/data/5YRData/acsdt5y2023-%s.dat")
NEW_TABLES = ["b25077", "b25064", "b25070", "b25034", "b25024",
              "b15003", "b01002", "b03002", "b08301", "b25002"]


def jam(v):
    """ACS jam values (-666666666 and friends) -> None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f < -1e8 else f


def read_geos():
    out = {}
    with open(RAW1 / "geo-Geos20235YR.txt", encoding="utf-8-sig") as fh:
        head = fh.readline().rstrip("\n").split("|")
        i_sum, i_st = head.index("SUMLEVEL"), head.index("STATE")
        i_gid, i_nm = head.index("GEO_ID"), head.index("NAME")
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= i_nm or f[i_st] != "06" or f[i_sum] not in ("160", "050"):
                continue
            out[(f[i_sum], f[i_nm])] = f[i_gid]
    return out


def read_table(stem, cols, where):
    out = {}
    with open(where / f"acsdt5y2023-{stem}.dat", encoding="utf-8-sig") as fh:
        head = fh.readline().rstrip("\n").split("|")
        idx = {c: head.index(c) for c in cols}
        for line in fh:
            f = line.rstrip("\n").split("|")
            out[f[0]] = {c: jam(f[i]) if i < len(f) else None
                         for c, i in idx.items()}
    return out


def read_gaz():
    out = {}
    for fn, sl in (("2023_Gaz_place_national.txt", "160"),
                   ("2023_Gaz_counties_national.txt", "050")):
        with open(RAW1 / fn, encoding="latin-1") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r = {k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in r.items()}
                g = r.get("GEOID")
                if not g:
                    continue
                try:
                    out[f"{sl}0000US{g}"] = float(r["ALAND"])
                except (KeyError, ValueError):
                    pass
    return out


def rng(stem, a, b):
    return [f"{stem.upper()}_E{i:03d}" for i in range(a, b + 1)]


def main():
    geos, gaz = read_geos(), read_gaz()
    t = {}
    t["b25003"] = read_table("b25003", ["B25003_E001", "B25003_E003"], RAW1)
    t["b19013"] = read_table("b19013", ["B19013_E001"], RAW1)
    t["b01003"] = read_table("b01003", ["B01003_E001"], RAW1)
    t["b25077"] = read_table("b25077", ["B25077_E001"], RAW2)
    t["b25064"] = read_table("b25064", ["B25064_E001"], RAW2)
    t["b25070"] = read_table("b25070", rng("b25070", 1, 11), RAW2)
    t["b25034"] = read_table("b25034", rng("b25034", 1, 3), RAW2)
    t["b25024"] = read_table("b25024", rng("b25024", 1, 8), RAW2)
    t["b15003"] = read_table("b15003", ["B15003_E001"] + rng("b15003", 22, 25), RAW2)
    t["b01002"] = read_table("b01002", ["B01002_E001"], RAW2)
    t["b03002"] = read_table("b03002", ["B03002_E001", "B03002_E003"], RAW2)
    t["b08301"] = read_table("b08301", ["B08301_E001", "B08301_E010"], RAW2)
    t["b25002"] = read_table("b25002", ["B25002_E001", "B25002_E003"], RAW2)

    # body type: the non-ACS stratum covariate.
    BODY_TYPE = {"place": "council", "county": "county", "county_union": "transit"}

    recs, lookup, problems = [], [], []
    for body, (kind, names) in sorted(BODY_GEO.items()):
        sl = SUMLEVEL[kind]
        parts = []
        for nm in names:
            gid = geos.get((sl, nm))
            if gid is None:
                problems.append(f"{body}: no GEO_ID for {sl}/{nm}")
                continue
            g = lambda tb, c: t[tb].get(gid, {}).get(c)  # noqa: E731
            parts.append(dict(
                name=nm, geo_id=gid, aland_m2=gaz.get(gid),
                pop=g("b01003", "B01003_E001"),
                occ=g("b25003", "B25003_E001"), rent=g("b25003", "B25003_E003"),
                income=g("b19013", "B19013_E001"),
                value=g("b25077", "B25077_E001"),
                grent=g("b25064", "B25064_E001"),
                age=g("b01002", "B01002_E001"),
                burden_num=sum((g("b25070", c) or 0) for c in rng("b25070", 7, 10)),
                burden_den=((g("b25070", "B25070_E001") or 0)
                            - (g("b25070", "B25070_E011") or 0)),
                new_num=sum((g("b25034", c) or 0) for c in rng("b25034", 2, 3)),
                new_den=g("b25034", "B25034_E001"),
                mu_num=sum((g("b25024", c) or 0) for c in rng("b25024", 5, 8)),
                mu_den=g("b25024", "B25024_E001"),
                ba_num=sum((g("b15003", c) or 0) for c in rng("b15003", 22, 25)),
                ba_den=g("b15003", "B15003_E001"),
                nhw_num=g("b03002", "B03002_E003"),
                nhw_den=g("b03002", "B03002_E001"),
                tr_num=g("b08301", "B08301_E010"),
                tr_den=g("b08301", "B08301_E001"),
                vac_num=g("b25002", "B25002_E003"),
                vac_den=g("b25002", "B25002_E001")))
            lookup.append(dict(body=body, geo_kind=kind, census_name=nm,
                               geo_id=gid, sumlevel=sl))
        if not parts:
            continue

        def s(k):
            return sum(p[k] or 0 for p in parts)

        def ratio(n, d):
            dd = s(d)
            return round(s(n) / dd, 6) if dd else None

        def wmean(k):
            w = sum(p["pop"] or 0 for p in parts if p[k])
            if not w:
                return None
            return sum((p[k] or 0) * (p["pop"] or 0) for p in parts if p[k]) / w

        sqmi = s("aland_m2") / 2_589_988.110336 if s("aland_m2") else None
        inc, val, gr, age = wmean("income"), wmean("value"), wmean("grent"), wmean("age")
        recs.append(dict(
            body=body, geo_kind=kind, body_type=BODY_TYPE[kind], n_geos=len(parts),
            census_names="; ".join(p["name"] for p in parts),
            geo_ids="; ".join(p["geo_id"] for p in parts),
            medians_are_weighted_means=int(len(parts) > 1),
            occupied_units=int(s("occ")), renter_units=int(s("rent")),
            renter_share=ratio("rent", "occ"),
            population=int(s("pop")),
            med_hh_income=round(inc) if inc else None,
            land_sq_mi=round(sqmi, 4) if sqmi else None,
            pop_density_per_sq_mi=round(s("pop") / sqmi, 2) if sqmi else None,
            med_home_value=round(val) if val else None,
            med_gross_rent=round(gr) if gr else None,
            median_age=round(age, 2) if age else None,
            rent_burden_30=ratio("burden_num", "burden_den"),
            built_2010_plus=ratio("new_num", "new_den"),
            multiunit_5plus=ratio("mu_num", "mu_den"),
            ba_plus=ratio("ba_num", "ba_den"),
            nhwhite_share=ratio("nhw_num", "nhw_den"),
            transit_commute=ratio("tr_num", "tr_den"),
            vacancy_rate=ratio("vac_num", "vac_den"),
            parts=parts))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acs_wide.json").write_text(json.dumps(dict(
        vintage=VINTAGE,
        urls_new={k: BASE % k for k in NEW_TABLES},
        urls_reused_from_v1="see output/renter_share/acs.json",
        geography_rule=dict(
            council="census place", county_board="county (napa)",
            transit_board=("union of member counties; BART = Alameda + Contra "
                           "Costa + San Francisco, VTA = Santa Clara, "
                           "SamTrans = San Mateo -- identical to v1, no new "
                           "lookup was needed"),
            counts="sum(numerator)/sum(denominator) over the union",
            medians=("population-weighted mean of county medians, NOT a "
                     "median; flagged medians_are_weighted_means (bart only)")),
        definitions=dict(
            renter_share="B25003_E003/B25003_E001",
            med_hh_income="B19013_E001",
            pop_density_per_sq_mi="B01003_E001 / ALAND sq mi (2023 Gazetteer)",
            med_home_value="B25077_E001",
            med_gross_rent="B25064_E001",
            rent_burden_30="(B25070_E007..E010)/(B25070_E001 - B25070_E011)",
            built_2010_plus="(B25034_E002+E003)/B25034_E001",
            multiunit_5plus="(B25024_E005..E008)/B25024_E001",
            ba_plus="(B15003_E022..E025)/B15003_E001",
            median_age="B01002_E001",
            nhwhite_share="B03002_E003/B03002_E001",
            transit_commute="B08301_E010/B08301_E001",
            vacancy_rate="B25002_E003/B25002_E001"),
        lookup=lookup, bodies=recs, problems=problems), indent=2))

    cols = [c for c in recs[0] if c != "parts"]
    with open(OUT / "covariates_wide.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    for r in recs:
        print(f"{r['body']:12s} {r['body_type']:8s} rent={r['renter_share']:.3f} "
              f"inc={r['med_hh_income']:>7} val={r['med_home_value']} "
              f"gr={r['med_gross_rent']} burden={r['rent_burden_30']:.3f} "
              f"new={r['built_2010_plus']:.3f} mu5={r['multiunit_5plus']:.3f} "
              f"ba={r['ba_plus']:.3f} age={r['median_age']:.1f} "
              f"nhw={r['nhwhite_share']:.3f} tr={r['transit_commute']:.4f} "
              f"vac={r['vacancy_rate']:.4f}")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ", file=sys.stderr)
    print(f"\n{len(recs)} bodies, {len(cols)} columns -> {OUT/'covariates_wide.csv'}")


if __name__ == "__main__":
    main()
