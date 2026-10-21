#!/usr/bin/env python3
"""
a0_terms.py -- Phase A0 step 1: do the keyword and frame terms actually occur?

Before any rate is plotted, every term in plan/keywords_density.txt,
plan/keywords_transit.txt and plan/frames.yaml is counted in the transcribed
episodes (whole word, case-insensitive, multi-word phrases across whitespace
or a hyphen, optional 's -- the same regex the classifier's keyword stage uses,
pipeline/segment.py load_keywords). A term that never occurs is reported as
such, not hidden: a plot of "frame rates" built on terms nobody said would be
a plot of nothing.

Outputs (output/analysis/):
  a0_term_counts.csv    one row per keyword term: total, episodes, by family, by role
  a0_frame_counts.csv   one row per frame term, plus the frame it belongs to
  a0_kwic.md            KWIC lines for the most frequent terms and the top frame
                        terms per role, so the counts can be read as real uses
                        (names redacted to initials: public commenters are
                        anonymous by policy)

    .venv/bin/python analysis/a0_terms.py --db ~/.local/share/ytpt/corpus.sqlite
"""
import argparse, csv, os, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "pipeline"))
import word_freq as wf  # noqa: E402

FAMILIES = ("density", "transit", "both")
ROLES = ("dais", "public", "unknown")
FRAME_LABELS = {
    "traffic_parking": "traffic / parking", "neighborhood_character": "neighborhood character",
    "safety_crime": "safety / crime", "environment_ceqa_fire": "environment / CEQA / fire",
    "affordability_displacement": "affordability / displacement", "property_values": "property values",
    "process_notification": "process / notification", "fiscal": "fiscal",
    "state_mandate": "state mandate / Sacramento", "yimby_pro_housing": "YIMBY / pro-housing",
    "transit_service": "transit service", "transit_safety": "transit safety",
}


# ------------------------------------------------------------------ terms -----
def term_pattern(term):
    """Whole-word, case-insensitive, optional 's; a space matches whitespace or
    a hyphen. Identical to pipeline/segment.py load_keywords so a count here is
    the count the classifier stage would have seen."""
    body = r"[\s\-]+".join(re.escape(w) for w in term.lower().split())
    return re.compile(rf"(?<!\w){body}(?:'?s)?(?!\w)", re.I)


def load_keyword_list(path):
    """[(term, family, conditional)] in file order; comments and @family lines handled."""
    out, fam = [], None
    for line in Path(path).read_text().splitlines():
        t = line.split("#")[0].strip().lower()
        if not t:
            continue
        if t.startswith("@family"):
            fam = t.split(None, 1)[1].strip() if " " in t else None
            continue
        cond = t.startswith("?")
        out.append((t.lstrip("?").strip(), fam, cond))
    return out


def load_frames(path):
    """{frame: [terms]} from the plan/frames.yaml subset: `key:` lines, `- term`
    items, `#` comments. Anything else is an error, on purpose -- the file is
    meant to stay simple enough to read without a YAML library."""
    frames, cur = {}, None
    for ln, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            cur = line[:-1].strip()
            frames[cur] = []
        elif line.strip().startswith("- ") and cur:
            term = line.strip()[2:].strip().strip("'\"").lower()
            if term:
                frames[cur].append(term)
        else:
            raise ValueError(f"{path}:{ln}: expected 'frame:' or '- term', got {raw!r}")
    return frames


# ----------------------------------------------------------------- corpus -----
def load_segments(db):
    """[(episode_id, seg_idx, body, family, role, speaker, text)] for the 46 episodes."""
    con = wf._open_readonly(db)
    rows = con.execute(
        "SELECT s.episode_id, s.seg_idx, e.body, e.family, s.role, s.speaker, s.text "
        "FROM episode_segments s JOIN episodes e USING(episode_id) ORDER BY s.episode_id, s.seg_idx").fetchall()
    con.close()
    return rows


def count_terms(terms, segments):
    """{term: {"total", "episodes": set, "by_family": Counter, "by_role": Counter}}.
    Counts are per segment (a phrase never spans a segment boundary; a
    segment is a Whisper line, so this loses almost nothing)."""
    pats = {t: term_pattern(t) for t in terms}
    out = {t: {"total": 0, "episodes": set(), "by_family": Counter(), "by_role": Counter()} for t in terms}
    for ep, _i, _body, fam, role, _spk, text in segments:
        for t, rx in pats.items():
            n = len(rx.findall(text))
            if n:
                o = out[t]
                o["total"] += n; o["episodes"].add(ep)
                o["by_family"][fam] += n; o["by_role"][role] += n
    return out


def count_rows(counts, extra):
    """Flatten count_terms() into CSV rows, sorted by total desc then term."""
    rows = []
    for t, o in counts.items():
        r = dict(extra[t]); r.update({"term": t, "total": o["total"], "n_episodes": len(o["episodes"])})
        for f in FAMILIES:
            r[f"fam_{f}"] = o["by_family"][f]
        for ro in ROLES:
            r[f"role_{ro}"] = o["by_role"][ro]
        rows.append(r)
    rows.sort(key=lambda r: (-r["total"], r["term"]))
    return rows


def bucket(rows):
    return {"ge10": [r for r in rows if r["total"] >= 10],
            "1to9": [r for r in rows if 1 <= r["total"] <= 9],
            "zero": [r for r in rows if r["total"] == 0]}


# ------------------------------------------------------------------- KWIC -----
TITLE_RE = re.compile(r"\b(Director|Councilmember|Council Member|Mayor|Vice Mayor|Supervisor|President|"
                      r"Vice President|Chair|Commissioner|Senator|Assemblymember|Mr|Ms|Mrs|Dr|Officer|Chief|"
                      r"General Manager|Board Member)\.?\s+([A-Z][a-z]+)(\s+[A-Z][a-z]+)?\b")
NAME_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")
# Capitalised pairs that are places, bodies or things, not people.
NOT_NAMES = set("""san jose rafael oakland francisco bay area bart caltrain muni vta samtrans marin sonoma santa
clara alameda contra costa golden gate silicon valley east west north south downtown city council board county
state california sacramento housing element planning commission public comment works department general plan
specific area transit district station street avenue road boulevard park bridge center measure item agenda
staff report mayor vice chair president director councilmember supervisor thank you good evening morning
afternoon madam chief police fire life safety uc berkeley university community development authority
program pass bay pass regional rapid transportation commission federal highway school elementary high middle
market rate affordable ami hcd rhna ceqa lake merritt fremont richmond concord dublin pleasanton walnut creek
hayward daly millbrae antioch berryessa warm springs union square civic embarcadero powell montgomery
santa fe canal terra linda northgate downtown san pedro monterey bernal berryessa capitol corridor amtrak
ace clipper bay fair coliseum airport sfo oak new next first second third last okay thank""".split())


def redact_names(text):
    """Replace what looks like a person's name with initials. Heuristic, and
    deliberately over-eager: a wrongly initialled place name costs nothing, a
    public commenter's name in a report does."""
    def title_sub(m):
        first, second = m.group(2), (m.group(3) or "").strip()
        if first.lower() in NOT_NAMES and not second:
            return m.group(0)
        init = first[0] + "." + (f" {second[0]}." if second and second.lower() not in NOT_NAMES else "")
        return f"{m.group(1)} {init}"
    text = TITLE_RE.sub(title_sub, text)

    def pair_sub(m):
        a, b = m.group(1), m.group(2)
        if a.lower() in NOT_NAMES or b.lower() in NOT_NAMES:
            return m.group(0)
        return f"{a[0]}. {b[0]}."
    return NAME_RE.sub(pair_sub, text)


def name_tokens(segments):
    """Lower-cased words the redaction heuristic would initialise anywhere in the
    corpus ("my name is Ann Lee" -> {ann, lee}; "Director Smith" -> {smith}).
    Used to keep name-like tokens out of every labelled term list; they are
    still counted in totals -- the point is the label, not the number."""
    in_name, total, capped = Counter(), Counter(), Counter()
    cap_re = re.compile(r"\b[A-Z][a-z']+\b")
    for s in segments:
        text = s[6]
        total.update(wf.TOKEN_RE.findall(text.lower()))
        capped.update(w.lower() for w in cap_re.findall(text))
        for m in TITLE_RE.finditer(text):
            first, second = m.group(2), (m.group(3) or "").strip()
            if first.lower() not in NOT_NAMES:
                in_name[first.lower()] += 1
            if second and second.lower() not in NOT_NAMES:
                in_name[second.lower()] += 1
        for m in NAME_RE.finditer(text):
            a, b = m.group(1).lower(), m.group(2).lower()
            if a not in NOT_NAMES and b not in NOT_NAMES:
                in_name[a] += 1; in_name[b] += 1
    # A word is name-like when at least half its uses are inside a name, or
    # when it was seen in two or more names and is otherwise (almost) always
    # capitalised ("Eric" addressed alone, first name only). "grant" in "Oscar
    # Grant" once and "grant program" nine times stays a word.
    return {w for w, n in in_name.items()
            if n >= 0.5 * total[w] or (n >= 2 and capped[w] >= 0.7 * total[w])}


def kwic(term, segments, n=5, width=70, role=None, seed=7):
    """n context lines for a term: (body, episode_id, role, line). Spread across
    episodes first so five lines are not five sentences of one speaker."""
    import random
    rx = term_pattern(term)
    hits = []
    for ep, _i, body, _fam, r, _spk, raw in segments:
        if role and r != role:
            continue
        if not rx.search(raw):
            continue
        # Redact BEFORE marking the match: the ** markers would otherwise split
        # "Oscar Grant" around a node like "grant" and hide the name from the regex.
        text = redact_names(raw.replace("\n", " "))
        for m in rx.finditer(text):
            a, b = max(0, m.start() - width), min(len(text), m.end() + width)
            line = ("…" if a > 0 else "") + text[a:m.start()] + "**" + text[m.start():m.end()] + "**" \
                   + text[m.end():b] + ("…" if b < len(text) else "")
            hits.append((body, ep, r, line))
    rng = random.Random(seed)
    rng.shuffle(hits)
    by_ep = defaultdict(list)
    for h in hits:
        by_ep[h[1]].append(h)
    out = []
    while len(out) < n and any(by_ep.values()):       # round-robin over episodes
        for ep in sorted(by_ep):
            if by_ep[ep] and len(out) < n:
                out.append(by_ep[ep].pop())
    return out


# ------------------------------------------------------------------- main -----
def run(db, out_dir, kw_paths, frames_path, kwic_n=5, kwic_frames_per_role=3, kwic_frame_lines=8, pool=False):
    import facets
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Before a term is counted: one locate route and one role instrument, or an
    # explicit --pool. A term table pooled over instruments is a term table of
    # two corpora (analysis/facets.py).
    con = wf._open_readonly(db)
    scope = facets.require_single(con, pool=pool, script="analysis/a0_terms.py")
    con.close()
    segments = load_segments(db)
    if not segments:
        sys.exit("no episode segments in the corpus -- run analysis/load_episodes.py first")
    n_ep = len({s[0] for s in segments})
    n_words = sum(len(wf.TOKEN_RE.findall(s[6].lower())) for s in segments)

    # 1. keyword lists
    kw, extra = [], {}
    for p in kw_paths:
        for t, fam, cond in load_keyword_list(p):
            if t in extra:
                continue
            kw.append(t); extra[t] = {"list": Path(p).stem.replace("keywords_", ""), "conditional": int(cond)}
    kc = count_terms(kw, segments)
    krows = count_rows(kc, extra)
    fields = ["term", "list", "conditional", "total", "n_episodes"] + [f"fam_{f}" for f in FAMILIES] + [f"role_{r}" for r in ROLES]
    with open(out_dir / "a0_term_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(krows)

    # 2. frames
    frames = load_frames(frames_path)
    fterms = sorted({t for ts in frames.values() for t in ts})
    fc = count_terms(fterms, segments)
    fextra = {t: {"frame": ";".join(k for k, ts in frames.items() if t in ts)} for t in fterms}
    frows = count_rows(fc, fextra)
    ffields = ["term", "frame", "total", "n_episodes"] + [f"fam_{f}" for f in FAMILIES] + [f"role_{r}" for r in ROLES]
    with open(out_dir / "a0_frame_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ffields); w.writeheader(); w.writerows(frows)
    frame_tot = {k: sum(fc[t]["total"] for t in ts) for k, ts in frames.items()}
    frame_eps = {k: len(set().union(*(fc[t]["episodes"] for t in ts))) if ts else 0 for k, ts in frames.items()}

    # 3. summary to stdout
    kb = bucket(krows)
    print(f"A0 term verification: {n_ep} episodes, {n_words:,} words, {len(kw)} keyword terms, "
          f"{len(fterms)} frame terms in {len(frames)} frames")
    print(f"  keyword terms: {len(kb['ge10'])} appear >= 10 times, {len(kb['1to9'])} appear 1-9 times, "
          f"{len(kb['zero'])} never")
    print("  top 25 keyword terms:")
    for r in krows[:25]:
        print(f"    {r['total']:5d}  eps={r['n_episodes']:2d}  dais={r['role_dais']:4d} public={r['role_public']:4d}  "
              f"{r['term']}{' (?)' if r['conditional'] else ''}")
    print("  1-9 times: " + ", ".join(f"{r['term']} {r['total']}" for r in kb["1to9"]))
    print("  never: " + ", ".join(r["term"] for r in kb["zero"]))
    print("  frames (hits, episodes):")
    for k in frames:
        zero = [t for t in frames[k] if fc[t]["total"] == 0]
        print(f"    {frame_tot[k]:5d}  eps={frame_eps[k]:2d}  {FRAME_LABELS.get(k, k)}"
              f"   ({len(frames[k]) - len(zero)}/{len(frames[k])} terms occur)")

    # 4. KWIC
    md = [f"# A0 KWIC -- {n_ep} episodes, {n_words:,} words", "", f"Scope: {scope}.", "",
          "Names replaced by initials (heuristic, `a0_terms.redact_names`). `**term**` marks the match.", ""]
    md.append("## Most frequent keyword terms (sanity check: are these real uses?)")
    for r in krows[:kwic_n]:
        md.append(f"\n### {r['term']} -- {r['total']} occurrences in {r['n_episodes']} episodes "
                  f"(dais {r['role_dais']}, public {r['role_public']})")
        for body, ep, role, line in kwic(r["term"], segments, n=5):
            md.append(f"- `{body}` `{ep}` {role}: {line}")
    md.append("\n## Top frame terms per role")
    for role in ("public", "dais"):
        md.append(f"\n### {role}")
        top = sorted(fterms, key=lambda t: -fc[t]["by_role"][role])[:kwic_frames_per_role]
        for t in top:
            n = fc[t]["by_role"][role]
            md.append(f"\n#### {t} ({fextra[t]['frame']}) -- {n} {role} occurrences")
            for body, ep, r, line in kwic(t, segments, n=kwic_frame_lines, role=role):
                md.append(f"- `{body}` `{ep}`: {line}")
    md.append("\n## Frame terms by frame, top 3 each (all roles)")
    for k, ts in frames.items():
        top = sorted(ts, key=lambda t: -fc[t]["total"])[:3]
        md.append(f"\n### {FRAME_LABELS.get(k, k)} -- {frame_tot[k]} hits in {frame_eps[k]} episodes")
        for t in top:
            if fc[t]["total"] == 0:
                continue
            md.append(f"\n**{t}** ({fc[t]['total']})")
            for body, ep, r, line in kwic(t, segments, n=3):
                md.append(f"- `{body}` `{ep}` {r}: {line}")
    (out_dir / "a0_kwic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'a0_term_counts.csv'}, {out_dir / 'a0_frame_counts.csv'}, {out_dir / 'a0_kwic.md'}")
    return krows, frows, frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus.sqlite"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "analysis"))
    ap.add_argument("--keywords", nargs="+", default=[str(ROOT / "plan" / "keywords_density.txt"),
                                                      str(ROOT / "plan" / "keywords_transit.txt")])
    ap.add_argument("--frames", default=str(ROOT / "plan" / "frames.yaml"))
    ap.add_argument("--pool", action="store_true",
                    help="count over every instrument in the corpus at once; refused by default "
                         "when the corpus spans two locate routes or two role instruments (analysis/facets.py)")
    a = ap.parse_args(argv)
    run(a.db, a.out_dir, a.keywords, a.frames, pool=a.pool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
