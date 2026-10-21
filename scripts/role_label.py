#!/usr/bin/env python3
"""
scripts/role_label.py -- label every diarized speaker cluster of a meeting
episode as `dais` (members, staff, clerk, mayor), `public` (public-comment /
applicant speakers) or `unknown`. T0 test 6; design in PROPOSAL-transit.md 3(g).

WHY
---
The transit/density study compares what the dais says with what the public
says, so every transcript segment needs a role. pyannote gives anonymous
cluster labels (SPEAKER_00 ...) that are NOT stable across clips of the same
meeting, and there is no single "host" to enrol (host_enroll.py's trick). What
a meeting transcript does carry is a very regular script: the chair opens
public comment, the clerk calls names and times people out, members move and
second, staff say "this concludes staff's presentation", and public speakers
say "my name is ... I live on ...". Those phrases, plus the structure they
imply (a public-comment stretch during which each new voice speaks once for
<= 3 min), label most of the speech time without audio processing. Everything
here is text + timing; no model, no GPU, no network.

CUES (plan/role_cues.txt, in the order they are applied)
  1. cue phrases per cluster: floor / timer / vote / staff / intro / address
     -> dais points; public -> public points. Section weights and caps are
     WEIGHTS below. `intro` (job titles) counts only in the first 15 s of a
     cluster's speech; elsewhere "general manager" is someone being addressed.
     Opening or closing a public-comment region (below) is +1.5 dais per event.
  2. structure: an `opener` phrase starts a public-comment region that ends
     at the next `closer` (or clip end); a `timer` phrase ("that's your time",
     "followed by") with no region open means the clip began inside public
     comment and opens one retroactively. A cluster with >= 70 % of its talk
     inside regions, no floor/timer/vote cue, and <= 3.5 min per region is
     public (+2 if one stretch; +1 if it is in a dialogue there). A cluster
     with <= 30 % inside and >= 20 s of talk is dais (+1) -- it had the floor
     before the public did -- and one that speaks after the last closer is
     dais (+1): "back to the council". A clip with no region at all and no
     public cue on the cluster gets a dais prior (+1): outside public comment
     the floor belongs to the body and its staff (and invited presenters, the
     known residual).
  3. recurrence: >= 3 substantive turns (>= 8 s each) spanning >= 5 min ->
     dais (+1.5, +2 if >= 6 turns; halved when the turns are inside public
     comment -- a serial commenter or a Q&A). Cue 1 outranks cue 3: two
     `public` phrases (3 points) beat recurrence.
  A cluster is dais if dais - public >= 1, public if <= -1, else unknown.
  confidence = 1 - exp(-|margin| / 2.5)  (margin 1 -> 0.33, 2 -> 0.55, 4 -> 0.8).
  Measured on 10 hand-checked b13 clips (output/t0/roles/truth_10.json):
  first run 80.8 % of speech time strict / 90.5 % lenient; after the cue and
  region fixes described in T0-RESULTS test 6: 93.6 % / 95.8 %, public
  precision 0.94, recall 1.00. Known misses: a councilmember who says "I
  live in ..." or "on behalf of my community" reads as public.
  method = the family that contributed most points to the winning side
  (cue_phrase | structure | recurrence), so the DB column `role_method`
  (PROPOSAL 5) is filled without a second pass.

WHAT THIS DOES NOT DO
  * No embeddings, no cross-clip matching (labels are per clip; optional
    enrolment of the presiding officer is a separate step, see T0-RESULTS).
  * Applicants and their representatives are `public` (they are on the
    public side of the podium). Invited presenters from outside the body --
    a Caltrain executive briefing a council, a citizen oversight-committee
    chair -- are a third thing; there is no cue for them, they come out
    `dais` (prior / recurrence) and the truth file marks them `other` and
    excludes them (781 s of 8,018 s on the 10 hand-checked clips).
  * Merged clusters (one label carrying two people) get one role; the
    speech-time metric in --score is what tells you how much that costs.

OUTPUT  (never overwrites anything outside --out; writes new files only)
  <out>/<clip>.json      {clip, video_id, body, episode_id,
                          clip_start_in_meeting_s, duration_s, talk_s,
                          regions: [[start, end, opener_speaker, closed_by_cue]],
                          speakers: {label: {role, method, confidence, talk_s,
                                     turns, in_region_share, dais_points,
                                     public_points, evidence: [..]}},
                          summary: {public_share, dais_share, unknown_share,
                                    n_public, n_dais, n_unknown, n_clusters}}
  <out>/summary.json     the per-clip summaries + overall / per-body shares
  --score TRUTH.json     compares against hand labels {clip: {label: role}}
                         and prints speech-time accuracy (strict: unknown is
                         wrong; lenient: unknown excluded) plus per-cluster
                         precision / recall for `public`.

RUN
  .venv/bin/python scripts/role_label.py --src output/batches/b13/out --out output/t0/roles
  .venv/bin/python scripts/role_label.py --score output/t0/roles/truth_10.json
Transcript text is data: cue matching is regex-only, nothing in it is executed
or fed to a model.
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUES = ROOT / "plan" / "role_cues.txt"
DEFAULT_SRC = ROOT / "output" / "batches" / "b13" / "out"
DEFAULT_OUT = ROOT / "output" / "t0" / "roles"

# section -> (side, weight per hit, cap). opener / closer credit is given per
# region actually opened / closed (find_regions), not per phrase hit: a public
# speaker saying "during the public comment period" is not opening anything.
WEIGHTS = {
    "floor":   ("dais", 2.0, 6.0),
    "timer":   ("dais", 2.0, 6.0),
    "vote":    ("dais", 2.0, 4.0),
    "staff":   ("dais", 1.5, 3.0),
    "intro":   ("dais", 1.5, 3.0),     # only within INTRO_S of the cluster's first speech
    "address": ("dais", 0.75, 2.25),
    "public":  ("public", 1.5, 4.5),
}
REGION_CREDIT = 1.5              # per region the cluster opened or closed (cap 3)
INTRO_S = 15.0
PUBLIC_TURN_MAX_S = 210.0        # "speak once for <= 3.5 min"
PUBLIC_TURN_SLACK = 1.25         # applicants get 5 min in some cities; 3.5 * 1.25 = 4.4
IN_SHARE_PUBLIC, IN_SHARE_DAIS = 0.70, 0.30
MIN_TURN_S = 8.0                 # a "substantive" turn for recurrence
INTERJECT_S = 1.0                # a foreign segment shorter than this does not end a turn
RECUR_TURNS, RECUR_SPAN_S = 3, 300.0
POST_CLOSE_DAIS = 1.0            # talk after the last closer, outside any region: floor is back with the body
NO_REGION_PRIOR = 1.0            # clip with no public-comment stretch and no public cue: dais
DECIDE_MARGIN = 1.0
SHORT_S = 10.0                   # clusters below this are excluded from cluster P/R


# ---------------------------------------------------------------- cue file --

def load_cues(path=DEFAULT_CUES):
    """{section: [(pattern, weak)]}. A line starting with '~' is a weak cue:
    it only counts when the speaker has another floor/vote hit in the clip."""
    cues, section = defaultdict(list), None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section is None:
            raise ValueError(f"{path}: cue before any [section]: {line!r}")
        weak = line.startswith("~")
        if weak:
            line = line[1:].strip()
        body = re.sub(r" +", r"\\s+", line)
        if section == "intro":
            # self-introduction form only: "I'm the city manager", "my name is
            # X, deputy director", "Ryan Dyson, Housing Specialist"
            name_tok = r"(?!(?:and|or|directors|members|member|council|board|mayor|president|chair|vice|staff|general|the)\b)[\w.'-]+"
            pat = re.compile(r"(?:(?:\bi'?m\b|\bi am\b|\bmy name is\b|\bthis is\b)\s+(?:[\w.'-]+,?\s+){0,5}?"
                             r"|^\W*(?:good\s+(?:morning|afternoon|evening),?\s+)?(?:" + name_tok + r"\s+){1,3}" + name_tok + r",\s+)"
                             r"(?:the\s+)?(?:" + body + r")(?!\w)", re.I)
        else:
            pat = re.compile(r"(?<!\w)(?:" + body + r")(?!\w)", re.I)
        cues[section].append((pat, weak))
    return dict(cues)


def hits(text, patterns):
    """Number of (pattern, weak) entries that match text at least once, and the
    matched strings, so the evidence list is readable."""
    n, found, weak_only = 0, [], True
    for pat, weak in patterns:
        m = pat.search(text)
        if m:
            n += 1
            found.append(m.group(0))
            weak_only = weak_only and weak
    return n, found, weak_only


# ---------------------------------------------------------------- structure --

def _spk(seg):
    return seg.get("speaker") or None


def turns_of(segments, speaker):
    """Maximal runs of this speaker's segments; a foreign segment shorter than
    INTERJECT_S (a diarization slip or an 'mm-hm') does not break a run.
    [(start, end, talk_s)]"""
    out, cur = [], None
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if _spk(s) == speaker:
            if cur is None:
                cur = [st, en, en - st]
            else:
                cur[1] = en
                cur[2] += en - st
        elif cur is not None and en - st >= INTERJECT_S:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def cluster_stats(segments):
    """Per label: talk_s, turns (substantive), n_turns_all, first, last, span_s."""
    stats = {}
    for label in sorted({_spk(s) for s in segments if _spk(s)}):
        turns = turns_of(segments, label)
        big = [t for t in turns if t[2] >= MIN_TURN_S]
        talk = sum(t[2] for t in turns)
        first = min(t[0] for t in turns)
        last = max(t[1] for t in turns)
        span = (max(t[1] for t in big) - min(t[0] for t in big)) if big else 0.0
        stats[label] = {"talk_s": talk, "turns": len(big), "turns_all": len(turns),
                        "first_s": first, "last_s": last, "span_s": span, "turn_list": turns}
    return stats


def cue_hits(segments, cues):
    """{label: {section: [n_hits, [examples], weak_only]}} over the cluster's
    whole text, segment by segment (a phrase split across two segments is
    missed; fine). `intro` cues count only while the cluster has spoken for
    less than INTRO_S -- the self-introduction position."""
    per = defaultdict(lambda: defaultdict(lambda: [0, [], True]))
    spoken = defaultdict(float)
    for s in segments:
        label = _spk(s)
        if not label:
            continue
        for section, pats in cues.items():
            if section in ("opener", "closer"):
                continue
            if section == "intro" and spoken[label] >= INTRO_S:
                continue
            n, found, weak_only = hits(s.get("text", ""), pats)
            if n:
                cell = per[label][section]
                cell[0] += n
                if len(cell[1]) < 4:
                    cell[1].extend(found[: 4 - len(cell[1])])
                cell[2] = cell[2] and weak_only
        spoken[label] += float(s["end"]) - float(s["start"])
    return per


def find_regions(segments, cues, duration_s):
    """Public-comment stretches. Returns (regions, credit) where regions =
    [(start, end, opener_speaker, closed_by_cue)] and credit = {label: n} for
    the speakers whose segment actually opened or closed a region.
    An opener starts a region unless one is open; the next closer (a later
    segment) ends it. A `timer` cue with no region open means the clip began
    inside public comment: the region is opened retroactively from the end of
    the previous region (or 0). Weak openers ('~' lines) are only honoured
    from a speaker who also has a floor/vote hit somewhere in the clip."""
    openers, closers, timers = cues.get("opener", []), cues.get("closer", []), cues.get("timer", [])
    regions, credit = [], defaultdict(int)
    open_at = open_by = open_idx = None
    last_close = 0.0
    for i, s in enumerate(segments):
        text, label, st = s.get("text", ""), _spk(s), float(s["start"])
        if open_at is None:
            if hits(text, closers)[0]:
                continue                     # "we'll close public comment and turn to ..." opens nothing
            n, _, _ = hits(text, openers)
            if n:
                open_at, open_by, open_idx = st, label, i
                credit[label] += 1
                continue
            n, _, _ = hits(text, timers)
            if n:
                open_at, open_by, open_idx = last_close, label, i
                credit[label] += 1
        else:
            n, _, _ = hits(text, closers)
            if n and i > open_idx:
                regions.append((open_at, float(s["end"]), open_by, True))
                credit[label] += 1
                last_close = float(s["end"])
                open_at = open_by = None
    if open_at is not None:
        regions.append((open_at, float(duration_s), open_by, False))
    return regions, dict(credit)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def region_shares(stats, regions):
    """Per label: (share of talk inside public regions, largest talk inside
    any single region, number of substantive turns inside regions, share of
    talk after the last cue-closed region)."""
    out = {}
    closed_ends = [r[1] for r in regions if r[3]]
    last_close = max(closed_ends) if closed_ends else None
    for label, st in stats.items():
        inside, per_region, turns_in, after = 0.0, [0.0] * len(regions), 0, 0.0
        for t0, t1, talk in st["turn_list"]:
            dens = talk / max(t1 - t0, 1e-6)
            hit = False
            for i, (r0, r1, _, _) in enumerate(regions):
                o = overlap(t0, t1, r0, r1) * dens
                inside += o
                per_region[i] += o
                hit = hit or o > 0
            if hit and talk >= MIN_TURN_S:
                turns_in += 1
            if last_close is not None:
                after += overlap(t0, t1, last_close, 1e9) * dens
        tot = st["talk_s"] or 1e-6
        out[label] = (inside / tot, max(per_region) if per_region else 0.0, turns_in, (after - 0.0) / tot if last_close is not None else 0.0)
    return out


# ------------------------------------------------------------------ labelling --

def label_episode(segments, cues, duration_s=None):
    """The whole decision for one clip. Returns (speakers dict, regions)."""
    if duration_s is None:
        duration_s = max((float(s["end"]) for s in segments), default=0.0)
    stats = cluster_stats(segments)
    hitmap = cue_hits(segments, cues)
    regions, credit = find_regions(segments, cues, duration_s)
    shares = region_shares(stats, regions)
    out = {}
    for label, st in stats.items():
        pts = {"dais": defaultdict(float), "public": defaultdict(float)}
        evidence = []
        h = hitmap.get(label, {})
        for section, (side, w, cap) in WEIGHTS.items():
            n, examples, _ = h.get(section, (0, [], True))
            if n:
                v = min(n * w, cap)
                pts[side]["cue_phrase"] += v
                evidence.append(f"{section}x{n} +{v:g} {side}: " + " | ".join(examples[:3]))
        if credit.get(label):
            v = min(credit[label] * REGION_CREDIT, 3.0)
            pts["dais"]["cue_phrase"] += v
            evidence.append(f"opened/closed public comment x{credit[label]} +{v:g} dais")
        floor_like = bool(credit.get(label)) or any(h.get(k, [0])[0] for k in ("floor", "timer", "vote"))
        in_share, max_in_region, turns_in, after_share = shares[label]
        in_region_public = False
        if regions:
            if in_share >= IN_SHARE_PUBLIC and not floor_like and max_in_region <= PUBLIC_TURN_MAX_S * PUBLIC_TURN_SLACK:
                v = (2.0 if in_share >= 0.9 else 1.5) if turns_in <= 2 else 1.0
                pts["public"]["structure"] += v
                in_region_public = True
                evidence.append(f"structure +{v:g} public: {in_share:.0%} of talk inside public comment, "
                                f"<= {max_in_region:.0f} s per region, {turns_in} turns inside")
            elif in_share <= IN_SHARE_DAIS and st["talk_s"] >= 20.0:
                pts["dais"]["structure"] += 1.0
                evidence.append(f"structure +1 dais: {in_share:.0%} of talk inside public comment")
            if after_share >= 0.7 and in_share <= IN_SHARE_DAIS:
                pts["dais"]["structure"] += POST_CLOSE_DAIS
                evidence.append(f"structure +{POST_CLOSE_DAIS:g} dais: {after_share:.0%} of talk after public comment closed")
        elif not pts["public"]:
            pts["dais"]["prior"] += NO_REGION_PRIOR
            evidence.append(f"prior +{NO_REGION_PRIOR:g} dais: no public-comment stretch in this clip, no public cue")
        if st["turns"] >= RECUR_TURNS and st["span_s"] >= RECUR_SPAN_S:
            v = 2.0 if st["turns"] >= 6 else 1.5
            if in_share >= IN_SHARE_PUBLIC:
                v *= 0.5                       # dialogue inside public comment, or a serial commenter
            pts["dais"]["recurrence"] += v
            evidence.append(f"recurrence +{v:g} dais: {st['turns']} turns over {st['span_s'] / 60:.1f} min")
        d, p = sum(pts["dais"].values()), sum(pts["public"].values())
        margin = d - p
        if margin >= DECIDE_MARGIN:
            role, side = "dais", "dais"
        elif margin <= -DECIDE_MARGIN:
            role, side = "public", "public"
        else:
            role, side = "unknown", None
        method = max(pts[side], key=pts[side].get) if side and pts[side] else "none"
        out[label] = {
            "role": role, "method": method,
            "confidence": round(1.0 - math.exp(-abs(margin) / 2.5), 2) if role != "unknown" else 0.0,
            "talk_s": round(st["talk_s"], 1), "turns": st["turns"], "span_s": round(st["span_s"], 1),
            "in_region_share": round(in_share, 2), "dais_points": round(d, 2), "public_points": round(p, 2),
            "evidence": evidence,
        }
    return out, [(round(a, 1), round(b, 1), c, closed) for a, b, c, closed in regions]


def summarise(speakers):
    talk = defaultdict(float)
    n = defaultdict(int)
    for v in speakers.values():
        talk[v["role"]] += v["talk_s"]
        n[v["role"]] += 1
    tot = sum(talk.values()) or 1.0
    return {"talk_s": round(tot, 1), "n_clusters": len(speakers),
            "public_share": round(talk["public"] / tot, 3), "dais_share": round(talk["dais"] / tot, 3),
            "unknown_share": round(talk["unknown"] / tot, 3),
            "n_public": n["public"], "n_dais": n["dais"], "n_unknown": n["unknown"]}


# --------------------------------------------------------------------- scoring --

def score(pred, truth, short_s=SHORT_S):
    """pred, truth: {clip: {label: role}} (pred values may be dicts with 'role'
    and 'talk_s'; truth values are 'dais' | 'public', or anything else --
    'other' (invited presenter from outside the body), 'mixed' -- which is
    excluded from every metric and reported as excluded, with what the
    labeller called it). Returns the speech-time accuracy (strict / lenient),
    public precision / recall over clusters >= short_s, and the per-clip rows."""
    tot = ok = unk = mixed_s = 0.0
    tp = fp = fn = 0
    rows, errors, excluded = [], [], []
    for clip, tlabels in sorted(truth.items()):
        p = pred.get(clip, {})
        c_tot = c_ok = c_unk = 0.0
        for label, t in tlabels.items():
            pv = p.get(label, {})
            role = pv.get("role", "missing") if isinstance(pv, dict) else pv
            talk = float(pv.get("talk_s", 0.0)) if isinstance(pv, dict) else 0.0
            if t not in ("dais", "public"):
                mixed_s += talk
                excluded.append((clip, label, t, role, talk))
                continue
            c_tot += talk
            if role == t:
                c_ok += talk
            elif role == "unknown":
                c_unk += talk
            else:
                errors.append((clip, label, t, role, talk))
            if talk >= short_s:
                tp += (role == "public" and t == "public")
                fp += (role == "public" and t != "public")
                fn += (role != "public" and t == "public")
        tot += c_tot; ok += c_ok; unk += c_unk
        rows.append((clip, c_tot, c_ok, c_unk))
    strict = ok / tot if tot else 0.0
    lenient = ok / (tot - unk) if tot - unk > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"speech_s": round(tot, 1), "correct_s": round(ok, 1), "unknown_s": round(unk, 1), "mixed_excluded_s": round(mixed_s, 1),
            "accuracy_strict": round(strict, 4), "accuracy_lenient": round(lenient, 4),
            "public_tp": tp, "public_fp": fp, "public_fn": fn,
            "public_precision": round(prec, 4), "public_recall": round(rec, 4),
            "rows": rows, "errors": errors, "excluded": excluded}


# ------------------------------------------------------------------------ I/O --

def meeting_offsets(fetch_report, episodes):
    """{episode_id: clip start in meeting seconds}: fetch_report actual_start_s
    when present (section clips carry a 10-s lead-in), else episodes.json
    start_s - 30 s pad."""
    off = {}
    if episodes and Path(episodes).exists():
        for e in json.loads(Path(episodes).read_text()):
            off[e["episode_id"]] = float(e["start_s"]) - 30.0
    if fetch_report and Path(fetch_report).exists():
        for c in json.loads(Path(fetch_report).read_text()).get("clips", []):
            a = c.get("actual_start_s")
            if a is None:
                a = c.get("start_s")
            if a is not None:
                off[c["episode_id"]] = float(a)
    return off


def write_new(path, doc):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_label(a):
    cues = load_cues(a.cues)
    out_dir = Path(a.out)
    if not str(out_dir.resolve()).startswith(str((ROOT / "output" / "t0" / "roles").resolve())) and not a.allow_out:
        sys.exit(f"--out {out_dir} is outside output/t0/roles; pass --allow-out if you mean it")
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = meeting_offsets(a.fetch_report, a.episodes)
    files = sorted(glob.glob(str(Path(a.src) / a.glob)))
    if not files:
        sys.exit(f"no transcripts match {a.src}/{a.glob}")
    summaries, by_body = {}, defaultdict(lambda: defaultdict(float))
    for f in files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        clip = Path(f).stem
        speakers, regions = label_episode(doc["segments"], cues, doc.get("duration_s"))
        summ = summarise(speakers)
        body = doc.get("speaker") or doc.get("channel") or "?"
        rec = {"clip": clip, "video_id": clip.rsplit("_e", 1)[0], "body": body, "episode_id": clip,
               "clip_start_in_meeting_s": offsets.get(clip), "duration_s": doc.get("duration_s"),
               "talk_s": summ["talk_s"], "regions": regions, "speakers": speakers, "summary": summ,
               "cues_file": str(Path(a.cues).relative_to(ROOT)) if str(a.cues).startswith(str(ROOT)) else str(a.cues)}
        write_new(out_dir / f"{clip}.json", rec)
        summaries[clip] = dict(summ, body=body, n_regions=len(regions))
        for role in ("public", "dais", "unknown"):
            by_body[body][role] += summ[f"{role}_share"] * summ["talk_s"]
        by_body[body]["talk_s"] += summ["talk_s"]
        by_body["ALL"]["talk_s"] += summ["talk_s"]
        for role in ("public", "dais", "unknown"):
            by_body["ALL"][role] += summ[f"{role}_share"] * summ["talk_s"]
        print(f"{clip:<18} {body:<10} k={summ['n_clusters']:2d} regions={len(regions)} "
              f"dais {summ['dais_share']:.0%} public {summ['public_share']:.0%} unknown {summ['unknown_share']:.0%} "
              f"(n {summ['n_dais']}/{summ['n_public']}/{summ['n_unknown']})")
    bodies = {b: {"talk_h": round(v["talk_s"] / 3600, 2),
                  **{f"{r}_share": round(v[r] / v["talk_s"], 3) for r in ("dais", "public", "unknown")}}
              for b, v in by_body.items()}
    ks = sorted(s["n_clusters"] for s in summaries.values())
    overall = {"clips": len(files), "clusters_min": ks[0], "clusters_median": ks[len(ks) // 2], "clusters_max": ks[-1],
               "by_body": bodies, "clips_with_public_region": sum(1 for s in summaries.values() if s["n_regions"])}
    write_new(out_dir / "summary.json", {"overall": overall, "clips": summaries})
    print(f"\n{len(files)} clips; clusters min/median/max {ks[0]}/{ks[len(ks) // 2]}/{ks[-1]}; "
          f"{overall['clips_with_public_region']} clips with a public-comment region")
    for b, v in bodies.items():
        print(f"  {b:<10} {v['talk_h']:5.2f} h  dais {v['dais_share']:.1%}  public {v['public_share']:.1%}  unknown {v['unknown_share']:.1%}")
    return 0


def run_score(a):
    truth = json.loads(Path(a.score).read_text(encoding="utf-8"))
    truth = {k: v for k, v in truth.items() if not k.startswith("_")}
    pred = {}
    for clip in truth:
        p = Path(a.out) / f"{clip}.json"
        if p.exists():
            pred[clip] = json.loads(p.read_text(encoding="utf-8"))["speakers"]
    r = score(pred, truth)
    print(f"{len(truth)} clips, {r['speech_s']:.0f} s of dais/public speech ({r['mixed_excluded_s']:.0f} s other/mixed excluded)")
    if r["excluded"]:
        for clip, label, t, role, talk in r["excluded"]:
            print(f"  excluded {clip:<18} {label:<10} truth {t:<6} labeller said {role:<7} {talk:5.0f} s")
    print(f"speech-time accuracy: strict {r['accuracy_strict']:.1%} (unknown = wrong; {r['unknown_s']:.0f} s unknown), "
          f"lenient {r['accuracy_lenient']:.1%}")
    print(f"public clusters (>= {SHORT_S:.0f} s): precision {r['public_precision']:.2f} recall {r['public_recall']:.2f} "
          f"(tp {r['public_tp']} fp {r['public_fp']} fn {r['public_fn']})")
    for clip, t, ok, unk in r["rows"]:
        print(f"  {clip:<18} {ok / t if t else 0:6.1%} of {t:6.0f} s  (unknown {unk:4.0f} s)")
    if r["errors"]:
        print("wrong:")
        for clip, label, t, role, talk in sorted(r["errors"], key=lambda e: -e[4]):
            print(f"  {clip:<18} {label:<10} truth {t:<6} pred {role:<7} {talk:5.0f} s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--glob", default="*_e*.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--allow-out", action="store_true", help="permit --out outside output/t0/roles")
    ap.add_argument("--cues", default=str(DEFAULT_CUES))
    ap.add_argument("--episodes", default=str(ROOT / "output" / "t0" / "episodes.json"))
    ap.add_argument("--fetch-report", default=str(ROOT / "output" / "t0" / "fetch_report.json"))
    ap.add_argument("--score", metavar="TRUTH.json", help="compare <out>/<clip>.json against hand labels; no labelling")
    a = ap.parse_args(argv)
    if a.score:
        return run_score(a)
    return run_label(a)


if __name__ == "__main__":
    sys.exit(main())
