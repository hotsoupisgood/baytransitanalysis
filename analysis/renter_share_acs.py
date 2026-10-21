#!/usr/bin/env python3
"""
renter_share_acs.py -- build the per-body ACS covariate table for
plan/RENTER-SHARE.md.

WHY A FLAT FILE AND NOT THE API. api.census.gov now 302s every unkeyed request
to /data/missing_key.html (checked 2026-10-18 20:54 UTC); no Census key is in
this project and getting one is a signup, not a light pull. The table-based
ACS 5-year Summary File is the same estimates, no key:

  https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/
    data/5YRData/acsdt5y2023-{b25003,b19013,b01003}.dat
    documentation/Geos20235YR.txt
  https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/
    2023_Gaz_{place,counties}_national.zip          (ALAND for density)

Vintage: ACS 2019-2023 5-year (the latest 5-year release), geography vintage
2023. Downloaded by output/renter_share/raw/dl.log + gaz.log.

Covariates per body:
  renter_share  B25003_E003 / B25003_E001  (renter-occupied / occupied units)
  med_hh_income B19013_E001
  pop_density   B01003_E001 / (ALAND m^2 -> sq mi)

Geography rule, one row per body:
  council        -> the city's census PLACE
  county board   -> the COUNTY (napa = Napa County Board of Supervisors)
  transit board  -> population-weighted mean over the district's MEMBER
                    COUNTIES. BART = Alameda + Contra Costa + San Francisco
                    (the three counties in the BART District; San Mateo is
                    served under contract and is not a member county).
                    VTA = Santa Clara County. SamTrans = San Mateo County.
                    Weight = B01003_E001. For income the weighted mean of
                    county medians is not a median -- it is flagged
                    `income_is_weighted_mean` in the output.

Writes output/renter_share/acs.json (with vintage, URLs, the lookup table and
every raw count) and output/renter_share/covariates.csv. No key, no .env, one
pull, nothing committed.
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "output" / "renter_share" / "raw"
OUT = ROOT / "output" / "renter_share"

VINTAGE = "ACS 2019-2023 5-year (table-based summary file), geography 2023"
URLS = {
    "b25003": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b25003.dat",
    "b19013": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b19013.dat",
    "b01003": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b01003.dat",
    "geos": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/documentation/Geos20235YR.txt",
    "gazetteer_place": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_place_national.zip",
    "gazetteer_county": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip",
}

# body -> (geo kind, census name as it appears in Geos20235YR.txt NAME)
# Councils are cities; `napa` is the county Board of Supervisors (channels.tsv
# regex "^Board of Supervisors on "); the three boards are county unions.
BODY_GEO = {
    "concord":      ("place", ["Concord city, California"]),
    "cupertino":    ("place", ["Cupertino city, California"]),
    "dalycity":     ("place", ["Daly City city, California"]),
    "fremont":      ("place", ["Fremont city, California"]),
    "hayward":      ("place", ["Hayward city, California"]),
    "healdsburg":   ("place", ["Healdsburg city, California"]),
    "livermore":    ("place", ["Livermore city, California"]),
    "losaltos":     ("place", ["Los Altos city, California"]),
    "oakland":      ("place", ["Oakland city, California"]),
    "paloalto":     ("place", ["Palo Alto city, California"]),
    "petaluma":     ("place", ["Petaluma city, California"]),
    "pleasanton":   ("place", ["Pleasanton city, California"]),
    "richmond":     ("place", ["Richmond city, California"]),
    "sanjose":      ("place", ["San Jose city, California"]),
    "sanleandro":   ("place", ["San Leandro city, California"]),
    "sanmateo":     ("place", ["San Mateo city, California"]),
    "sanrafael":    ("place", ["San Rafael city, California"]),
    "santaclara":   ("place", ["Santa Clara city, California"]),
    "santarosa":    ("place", ["Santa Rosa city, California"]),
    "sausalito":    ("place", ["Sausalito city, California"]),
    "sunnyvale":    ("place", ["Sunnyvale city, California"]),
    "vallejo":      ("place", ["Vallejo city, California"]),
    "cotati":       ("place", ["Cotati city, California"]),
    "milpitas":     ("place", ["Milpitas city, California"]),
    "rohnertpark":  ("place", ["Rohnert Park city, California"]),
    "napa":         ("county", ["Napa County, California"]),
    "vta":          ("county_union", ["Santa Clara County, California"]),
    "samtrans":     ("county_union", ["San Mateo County, California"]),
    "bart":         ("county_union", ["Alameda County, California",
                                      "Contra Costa County, California",
                                      "San Francisco County, California"]),
}
SUMLEVEL = {"place": "160", "county": "050", "county_union": "050"}


def read_geos():
    """NAME -> GEO_ID for California places (160) and counties (050)."""
    out = {}
    with open(RAW / "geo-Geos20235YR.txt", encoding="utf-8-sig") as fh:
        head = fh.readline().rstrip("\n").split("|")
        i_sum, i_st = head.index("SUMLEVEL"), head.index("STATE")
        i_gid, i_nm = head.index("GEO_ID"), head.index("NAME")
        for line in fh:
            f = line.rstrip("\n").split("|")
            if len(f) <= i_nm or f[i_st] != "06" or f[i_sum] not in ("160", "050"):
                continue
            out[(f[i_sum], f[i_nm])] = f[i_gid]
    return out


def read_table(stem, cols):
    """GEO_ID -> {col: int or None} from a table-based .dat file."""
    out = {}
    with open(RAW / f"acsdt5y2023-{stem}.dat", encoding="utf-8-sig") as fh:
        head = fh.readline().rstrip("\n").split("|")
        idx = {c: head.index(c) for c in cols}
        for line in fh:
            f = line.rstrip("\n").split("|")
            gid = f[0]
            rec = {}
            for c, i in idx.items():
                v = f[i] if i < len(f) else ""
                try:
                    rec[c] = float(v)
                except ValueError:
                    rec[c] = None          # ACS jam values (-666666666 etc.)
                if rec[c] is not None and rec[c] < -1e8:
                    rec[c] = None
            out[gid] = rec
    return out


def read_gaz():
    """GEO_ID -> ALAND (m^2) from the 2023 Gazetteer place + county files."""
    out = {}
    for fn, sl in (("2023_Gaz_place_national.txt", "160"),
                   ("2023_Gaz_counties_national.txt", "050")):
        p = RAW / fn
        with open(p, encoding="latin-1") as fh:
            rows = csv.DictReader(fh, delimiter="\t")
            for r in rows:
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


def main():
    geos = read_geos()
    b25 = read_table("b25003", ["B25003_E001", "B25003_E003", "B25003_M003"])
    b19 = read_table("b19013", ["B19013_E001", "B19013_M001"])
    b01 = read_table("b01003", ["B01003_E001"])
    gaz = read_gaz()

    lookup, recs, problems = [], [], []
    for body, (kind, names) in sorted(BODY_GEO.items()):
        sl = SUMLEVEL[kind]
        parts = []
        for nm in names:
            gid = geos.get((sl, nm))
            if gid is None:
                problems.append(f"{body}: no GEO_ID for {sl}/{nm}")
                continue
            occ = b25.get(gid, {}).get("B25003_E001")
            rent = b25.get(gid, {}).get("B25003_E003")
            pop = b01.get(gid, {}).get("B01003_E001")
            inc = b19.get(gid, {}).get("B19013_E001")
            aland = gaz.get(gid)
            parts.append(dict(name=nm, geo_id=gid, occupied_units=occ,
                              renter_units=rent, renter_moe=b25.get(gid, {}).get("B25003_M003"),
                              population=pop, med_hh_income=inc,
                              aland_m2=aland))
            lookup.append(dict(body=body, geo_kind=kind, census_name=nm,
                               geo_id=gid, sumlevel=sl))
        if not parts:
            continue
        occ = sum(p["occupied_units"] or 0 for p in parts)
        rent = sum(p["renter_units"] or 0 for p in parts)
        pop = sum(p["population"] or 0 for p in parts)
        aland = sum(p["aland_m2"] or 0 for p in parts)
        wsum = sum(p["population"] or 0 for p in parts if p["med_hh_income"])
        inc = (sum((p["med_hh_income"] or 0) * (p["population"] or 0)
                   for p in parts if p["med_hh_income"]) / wsum) if wsum else None
        sqmi = aland / 2_589_988.110336 if aland else None
        recs.append(dict(
            body=body, geo_kind=kind, n_geos=len(parts),
            census_names="; ".join(p["name"] for p in parts),
            geo_ids="; ".join(p["geo_id"] for p in parts),
            occupied_units=int(occ), renter_units=int(rent),
            renter_share=round(rent / occ, 6) if occ else None,
            population=int(pop),
            med_hh_income=round(inc) if inc else None,
            income_is_weighted_mean=int(len(parts) > 1),
            land_sq_mi=round(sqmi, 4) if sqmi else None,
            pop_density_per_sq_mi=round(pop / sqmi, 2) if sqmi else None,
            parts=parts,
        ))

    OUT.mkdir(parents=True, exist_ok=True)
    blob = dict(vintage=VINTAGE, urls=URLS,
                note=("Pulled from the no-key table-based summary file; "
                      "api.census.gov requires a key as of 2026-10-18."),
                method=dict(
                    council="census place",
                    county_board="county (napa)",
                    transit_board=("population-weighted mean over member "
                                   "counties; BART = Alameda + Contra Costa + "
                                   "San Francisco, VTA = Santa Clara, "
                                   "SamTrans = San Mateo"),
                    renter_share="B25003_E003 / B25003_E001",
                    income="B19013_E001 (weighted mean of county medians for unions)",
                    density="B01003_E001 / ALAND sq mi (2023 Gazetteer)"),
                lookup=lookup, bodies=recs, problems=problems)
    (OUT / "acs.json").write_text(json.dumps(blob, indent=2))

    cols = ["body", "geo_kind", "n_geos", "census_names", "geo_ids",
            "occupied_units", "renter_units", "renter_share", "population",
            "med_hh_income", "income_is_weighted_mean", "land_sq_mi",
            "pop_density_per_sq_mi"]
    with open(OUT / "covariates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    for r in recs:
        print(f"{r['body']:12s} {r['geo_kind']:12s} renter={r['renter_share']:.3f} "
              f"inc={r['med_hh_income']} dens={r['pop_density_per_sq_mi']:.0f} "
              f"occ={r['occupied_units']}")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ", file=sys.stderr)
    print(f"\n{len(recs)} bodies -> {OUT/'acs.json'}, {OUT/'covariates.csv'}")


if __name__ == "__main__":
    main()
