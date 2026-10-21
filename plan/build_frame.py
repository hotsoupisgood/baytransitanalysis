#!/usr/bin/env python3
"""
build_frame.py -- read plan/catalog.tsv, print the speaker x period cell table,
and with --draw write the sampling frame: the fixed definition of the corpus.

DESIGN (plan/SAMPLING.md; decided by Puck 2026-09-17)
  Stratified by speaker x month. Each cell gets a quota of transcribed
  AUDIO-HOURS, not a count of videos. Every video is capped at --cap-min minutes
  of transcription, so one long stream cannot fill a cell on its own.

  Within a cell, videos are put in a seeded random order and taken one at a time
  until BOTH the capped hours reach the quota AND at least --min-videos videos
  are in. The video that crosses the quota is kept. Why that matters (full
  derivation in SAMPLING.md): whether video i is drawn depends only on the
  durations of the videos placed BEFORE it, never on its own length -- so long
  and short videos in a cell are drawn with (almost exactly) equal probability,
  and the overshoot is at most one capped video. "Stop before exceeding" would
  instead favour short videos.

INCLUSION PROBABILITIES
  pi_hat in frame.csv is P(video drawn), estimated by re-running the same rule
  --pi-reps times on the same cell with an independent RNG (Monte Carlo SE
  <= 0.5/sqrt(reps), about 0.011 at 2000). Use it for design weights (1/pi) or
  to check that selection is close to equal-probability within a cell.

OUTPUTS (with --draw)
  <out>/frame.csv        one row per sampled video -- THE corpus definition
  <out>/frame_cells.csv  one row per cell: eligible, sampled, quota, shortfall
  <out>/frame_meta.json  every parameter + sha256 of the catalog (reproducibility)

SAFETY
  * Aborts if a speaker listed in plan/channels.tsv has no rows (a failed crawl
    must not silently become a two-speaker study). --allow-missing-speaker overrides.
  * Refuses to overwrite an existing frame.csv without --force: redrawing
    changes the study.
  * Each cell's draw uses its own seed (seed:speaker:period) over a pool sorted
    by video_id, so the draw does not depend on catalog row order, and adding or
    dropping one month does not change any other month's draw.
"""
import argparse, csv, hashlib, json, random, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(ROOT / "plan" / "catalog.tsv"))
    ap.add_argument("--channels", default=str(ROOT / "plan" / "channels.tsv"),
                    help="expected speakers are read from here (first column)")
    ap.add_argument("--out-dir", default=str(ROOT / "plan"))
    ap.add_argument("--quota-hours", type=float, default=3.0,
                    help="capped audio-hours per speaker x period cell")
    ap.add_argument("--cap-min", type=float, default=45,
                    help="transcribe at most this many minutes of any video")
    ap.add_argument("--min-videos", type=int, default=3,
                    help="a cell needs at least this many videos: sampled per cell, "
                         "and eligible for the period to be kept")
    ap.add_argument("--min-min", type=float, default=5, help="exclude videos shorter than this")
    ap.add_argument("--max-min", type=float, default=90,
                    help="exclude videos longer than this; 0 = no limit (see SAMPLING.md: "
                         "this decides whether multi-hour streams can enter at all)")
    ap.add_argument("--period", choices=["month", "quarter"], default="month")
    ap.add_argument("--from-period", default="", metavar="YYYY-MM",
                    help="first period to include (inclusive); '' = no lower bound")
    ap.add_argument("--to-period", default="", metavar="YYYY-MM",
                    help="last period to include (inclusive); '' = no upper bound")
    ap.add_argument("--exact-dates-only", action="store_true",
                    help="drop rows whose upload_date is the crawl's approximate one "
                         "(date_exact != 1; see scripts/resolve_dates.py)")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--pi-reps", type=int, default=2000,
                    help="Monte Carlo replicates for inclusion probabilities (0 = skip)")
    ap.add_argument("--rtfx", type=float, default=25.0,
                    help="ESTIMATED GPU realtime factor; Phase 0 measures the real one")
    ap.add_argument("--gpu-usd-hr", type=float, default=0.34)
    ap.add_argument("--draw", action="store_true", help="write the frame files")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frame.csv")
    ap.add_argument("--allow-missing-speaker", action="store_true")
    return ap.parse_args(argv)


def expected_speakers(channels_path, rows):
    p = Path(channels_path)
    if p.exists():
        sp = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name = line.split("\t")[0].strip()
            if name and name not in sp:
                sp.append(name)
        if sp:
            return sp
    return sorted({r["speaker"] for r in rows})


def period_of(yyyymmdd, period):
    y, m = int(yyyymmdd[:4]), int(yyyymmdd[4:6])
    if not (1 <= m <= 12):
        raise ValueError(yyyymmdd)
    return f"{y}-{m:02d}" if period == "month" else f"{y}-Q{(m - 1) // 3 + 1}"


def draw_cell(pool, quota_s, cap_s, min_videos, rng_key):
    """Seeded random order; take videos until capped total >= quota AND
    count >= min_videos. Returns [(row, capped_s, cumulative_s)]."""
    order = sorted(pool, key=lambda r: r["video_id"])
    random.Random(rng_key).shuffle(order)
    take, total = [], 0.0
    for r in order:
        if total >= quota_s and len(take) >= min_videos:
            break
        capped = min(r["_dur"], cap_s)
        total += capped
        take.append((r, capped, total))
    return take


def inclusion_probs(pool, quota_s, cap_s, min_videos, reps, seed_key):
    """pi_i = P(video i is drawn) under draw_cell's rule, by simulation.

    The rule includes video i iff, when i is reached in the random order, the
    videos placed BEFORE it have not yet reached the quota (or the minimum
    count). That event never involves i's own duration, which is why pi is
    nearly flat across long and short videos; the only dependence is that a
    long video is not among "the others", which tilts pi very slightly upward
    for long videos (O(1/N) in a cell of N)."""
    if reps <= 0:
        return {}
    caps = [min(r["_dur"], cap_s) for r in pool]
    hits = [0] * len(pool)
    idx = list(range(len(pool)))
    rng = random.Random(f"pi:{seed_key}")
    for _ in range(reps):
        rng.shuffle(idx)
        total = 0.0
        for k, i in enumerate(idx):
            if total >= quota_s and k >= min_videos:
                break
            hits[i] += 1
            total += caps[i]
    return {pool[i]["video_id"]: hits[i] / reps for i in range(len(pool))}


def main(argv=None):
    a = parse_args(argv)
    cat = Path(a.catalog)
    if not cat.exists():
        sys.exit(f"no catalog at {cat} -- run plan/crawl_metadata.sh first")
    raw = cat.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    need = {"speaker", "video_id", "upload_date", "duration_s"}
    if rows and not need <= set(rows[0]):
        sys.exit(f"{cat} is missing columns {sorted(need - set(rows[0]))}. "
                 "If every row sits in one column, the catalog came from the old "
                 "crawl script that wrote literal backslash-t -- re-crawl.")

    speakers = expected_speakers(a.channels, rows)
    cap_s, quota_s = a.cap_min * 60, a.quota_hours * 3600
    cells, skipped = defaultdict(list), defaultdict(int)
    per_speaker_rows = defaultdict(int)
    for r in rows:
        per_speaker_rows[r["speaker"]] += 1
        d = (r.get("upload_date") or "").strip()
        try:
            per = period_of(d, a.period)
        except (ValueError, IndexError):
            skipped["no_date"] += 1
            continue
        try:
            dur = float(r.get("duration_s"))
        except (TypeError, ValueError):
            skipped["no_duration"] += 1
            continue
        if dur < a.min_min * 60:
            skipped["shorter_than_min"] += 1
            continue
        if a.max_min and dur > a.max_min * 60:
            skipped["longer_than_max"] += 1
            continue
        if (a.from_period and per < a.from_period) or (a.to_period and per > a.to_period):
            skipped["outside_window"] += 1
            continue
        if a.exact_dates_only and (r.get("date_exact") or "0") != "1":
            skipped["approximate_date"] += 1
            continue
        r["_dur"] = dur
        cells[(r["speaker"], per)].append(r)

    extra = sorted(set(per_speaker_rows) - set(speakers))
    if extra:
        print(f"note: catalog speakers not in {a.channels} are ignored: {extra}")
    missing = [s for s in speakers if per_speaker_rows.get(s, 0) == 0]
    no_elig = [s for s in speakers if s not in missing
               and not any(sp == s for sp, _ in cells)]
    print(f"{len(rows)} catalog rows; skipped: {dict(skipped)}")
    for s in speakers:
        print(f"  {s:<14} {per_speaker_rows.get(s, 0):>6} rows")
    if (missing or no_elig) and not a.allow_missing_speaker:
        sys.exit(f"\nABORT: speakers with no rows: {missing}; with no eligible videos: {no_elig}.\n"
                 "A failed crawl would silently turn this into a smaller study. Fix the "
                 "crawl, or pass --allow-missing-speaker if dropping them is a decision.")
    speakers = [s for s in speakers if s not in missing and s not in no_elig]

    periods = sorted({p for _, p in cells})
    def elig(s, p):
        pool = cells.get((s, p), [])
        return len(pool), sum(min(r["_dur"], cap_s) for r in pool)

    kept = [p for p in periods if all(elig(s, p)[0] >= a.min_videos for s in speakers)]
    dropped = [p for p in periods if p not in kept]

    w = 16
    print(f"\ncell = eligible capped hours / videos   (quota {a.quota_hours:g} h, cap "
          f"{a.cap_min:g} min, >= {a.min_videos} videos)")
    print(f"{a.period:<9}" + "".join(f"{s:>{w}}" for s in speakers) + "   status")
    for p in periods:
        vals = [elig(s, p) for s in speakers]
        txt = "".join(f"{f'{h / 3600:.1f}h/{n}v':>{w}}" for n, h in vals)
        if p in dropped:
            status = "DROP (a speaker has < min videos)"
        elif all(h >= quota_s for _, h in vals):
            status = "full"
        else:
            status = "kept, short of quota"
        print(f"{p:<9}{txt}   {status}")
    share = len(dropped) / len(periods) if periods else 0
    print(f"\nperiods: {len(periods)} total, {len(kept)} kept, {len(dropped)} dropped ({share:.0%})")
    if share > 0.20 and a.period == "month":
        print("  over 20% dropped -- SAMPLING.md rule 4: re-run with --period quarter")

    if not a.draw:
        print("\n(dry run -- pass --draw to write the frame)")
        return 0
    if not kept:
        sys.exit("ABORT: no period has every speaker present; nothing to draw.")

    out = Path(a.out_dir)
    frame_path = out / "frame.csv"
    if frame_path.exists() and not a.force:
        sys.exit(f"ABORT: {frame_path} exists. It defines the study; redrawing changes it. "
                 "Pass --force only if that is the intent, and tell Puck.")

    frame, cell_rows = [], []
    for p in periods:
        for s in speakers:
            n_el, h_el = elig(s, p)
            base = {"speaker": s, "period": p, "eligible_videos": n_el,
                    "eligible_capped_h": round(h_el / 3600, 3), "quota_h": a.quota_hours}
            if p in dropped:
                cell_rows.append({**base, "sampled_videos": 0, "sampled_capped_h": 0,
                                  "shortfall_h": a.quota_hours, "status": "dropped_period"})
                continue
            key = f"{a.seed}:{s}:{p}"
            take = draw_cell(cells[(s, p)], quota_s, cap_s, a.min_videos, key)
            pis = inclusion_probs(cells[(s, p)], quota_s, cap_s, a.min_videos, a.pi_reps, key)
            got = take[-1][2] if take else 0.0
            cell_rows.append({**base, "sampled_videos": len(take),
                              "expected_videos": round(sum(pis.values()), 2) if pis else "",
                              "sampled_capped_h": round(got / 3600, 3),
                              "shortfall_h": round(max(0.0, quota_s - got) / 3600, 3),
                              "status": "full" if got >= quota_s else "short"})
            for i, (r, capped, cum) in enumerate(take, 1):
                frame.append({"speaker": s, "period": p, "video_id": r["video_id"],
                              "upload_date": r["upload_date"], "duration_s": r["_dur"],
                              "capped_s": capped, "draw_order": i,
                              "cum_capped_s": round(cum, 1),
                              "pi_hat": round(pis[r["video_id"]], 4) if pis else "",
                              "channel_url": r.get("channel_url", ""),
                              "title": r.get("title", "")})

    out.mkdir(parents=True, exist_ok=True)
    with frame_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(frame[0]))
        wr.writeheader()
        wr.writerows(frame)
    with (out / "frame_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(cell_rows[0]))
        wr.writeheader()
        wr.writerows(cell_rows)

    hrs = sum(r["capped_s"] for r in frame) / 3600
    raw_hrs = sum(r["duration_s"] for r in frame) / 3600
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": "stratified speaker x period, audio-hours quota (plan/SAMPLING.md)",
        "seed": a.seed, "period": a.period, "quota_hours": a.quota_hours,
        "cap_min": a.cap_min, "min_videos": a.min_videos,
        "min_min": a.min_min, "max_min": a.max_min, "speakers": speakers,
        "from_period": a.from_period, "to_period": a.to_period, "exact_dates_only": a.exact_dates_only,
        "catalog": str(cat), "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "catalog_rows": len(rows), "periods_kept": kept, "periods_dropped": dropped,
        "pi_reps": a.pi_reps,
        "n_videos": len(frame), "capped_audio_hours": round(hrs, 2),
        "raw_audio_hours": round(raw_hrs, 2),
        "short_cells": sum(1 for c in cell_rows if c["status"] == "short"),
    }
    (out / "frame_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {frame_path} -- {len(frame)} videos, seed={a.seed}")
    print(f"  {hrs:.0f} capped audio-hours to transcribe ({raw_hrs:.0f} h of full-length audio)")
    print(f"  est GPU: {hrs / a.rtfx:.1f} GPU-h -> ${hrs / a.rtfx * a.gpu_usd_hr:.2f} "
          f"at {a.rtfx:g}x realtime (ESTIMATE until Phase 0), ${a.gpu_usd_hr}/h")
    if meta["short_cells"]:
        print(f"  {meta['short_cells']} cells below quota -- see frame_cells.csv (rule 2: take all, record shortfall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
