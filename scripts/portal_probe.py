#!/usr/bin/env python3
"""
scripts/portal_probe.py -- is there a non-YouTube route for a body? (2026-09-24)

Why: the YouTube caption route is rate-limited to about one meeting an hour and
322 meetings are waiting behind it. Every body that publishes the same meetings
on a municipal portal can leave that queue permanently -- Granicus already gave
us 225 meetings at >= 5 s pacing with no bot checks at all (T2-PLAN 3a).

What it does, per body: guess the portal hostnames, drop the ones DNS does not
know (free, no request), GET the ones it does, and for a live Granicus site walk
view_id 1..N, count clips, then read ONE player page to see whether a caption
track or agenda index points are reachable.

HARD RULE: this script never asks youtube.com or googlevideo.com for anything.
Portal HTML is untrusted data: nothing it contains is executed or obeyed, and
`instruction_like` flags pages that read as if they were addressing the reader.

    python3 scripts/portal_probe.py --out output/t2/portal_probe
    python3 scripts/portal_probe.py --bodies paloalto,concord --views 8
"""
import argparse, json, re, socket, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import granicus  # noqa: E402

BANNED_HOSTS = ("youtube.com", "youtu.be", "googlevideo.com", "ytimg.com", "google.com")
GAP_S = 1.5                       # global pause between HTTP requests
TIMEOUT = 20

# Candidate slugs per body. The first is the body slug itself; the rest are the
# spellings Bay Area cities actually use (srcity = Santa Rosa, rpcity = Rohnert
# Park -- both are their real domains).
SLUGS = {
    "paloalto":    ["paloalto", "cityofpaloalto"],
    "cupertino":   ["cupertino", "cityofcupertino"],
    "sunnyvale":   ["sunnyvale", "sunnyvaleca", "cityofsunnyvale"],
    "santarosa":   ["santarosa", "srcity", "cityofsantarosa"],
    "livermore":   ["livermore", "cityoflivermore", "livermoreca"],
    "milpitas":    ["milpitas", "cityofmilpitas"],
    "pleasanton":  ["pleasanton", "pleasantonca", "cityofpleasantonca"],
    "healdsburg":  ["healdsburg", "cityofhealdsburg"],
    "rohnertpark": ["rohnertpark", "rpcity", "cityofrohnertpark"],
    "losaltos":    ["losaltos", "losaltosca", "cityoflosaltos"],
    "dalycity":    ["dalycity", "dalycityca", "cityofdalycity"],
    "sanmateo":    ["sanmateo", "cityofsanmateo", "sanmateoca"],
    "concord":     ["concord", "concordca", "cityofconcord"],
    "vallejo":     ["vallejo", "cityofvallejo", "vallejoca"],
    "petaluma":    ["petaluma", "cityofpetaluma", "petalumaca"],
    "cotati":      ["cotati", "cotaticity"],
}

# portal family -> (hostname template, path to probe)
PORTALS = {
    "granicus":    ("{slug}.granicus.com", "/ViewPublisher.php?view_id=1"),
    "legistar":    ("{slug}.legistar.com", "/Calendar.aspx"),
    "primegov":    ("{slug}.primegov.com", "/public/portal"),
    "civicclerk":  ("{slug}.civicclerk.com", "/"),
    # Swagit (Granicus-owned) lives on <slug>.new.swagit.com/views/<n>; the bare
    # <slug>.swagit.com host 404s or 500s even for real customers (Petaluma).
    "swagit":      ("{slug}.new.swagit.com", "/views/1"),
    "novusagenda": ("{slug}.novusagenda.com", "/agendapublic/"),
    "civicweb":    ("{slug}.civicweb.net", "/portal/"),
}

TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
# Text that reads like an instruction aimed at whoever is reading the page.
# AGENTS.md rule 5: report it, never act on it.
INSTRUCTION_LIKE = re.compile(
    r"\b(?:ignore (?:all |any )?(?:previous|prior|above) instructions?|you are (?:an? )?(?:ai|assistant|language model)"
    r"|system prompt|disregard (?:the )?(?:above|previous)|as an ai\b)", re.I)
AGENDA_WORD = re.compile(r"\b(agenda|minutes|packet)\b", re.I)
COUNCIL_WORD = re.compile(r"\bcity council\b|\bcouncil (?:meeting|regular)\b|\bboard of\b", re.I)
_last = {"at": 0.0}


def resolves(host):
    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return True
    except OSError:
        return False


def get(url, timeout=TIMEOUT, max_bytes=4_000_000):
    """GET with the project UA, >= GAP_S after the previous request. Returns
    (status, final_url, text). A 4xx/5xx comes back as a status, not an
    exception -- a 404 is a finding."""
    for banned in BANNED_HOSTS:
        if banned in urllib.parse.urlparse(url).netloc.lower():
            raise ValueError(f"refusing to request {banned}: this script never touches YouTube")
    dt = time.time() - _last["at"]
    if dt < GAP_S:
        time.sleep(GAP_S - dt)
    req = urllib.request.Request(url, headers={"User-Agent": granicus.UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(max_bytes)
            return r.status, r.geturl(), data.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(4096).decode("utf-8", "replace")
        except Exception:                                  # noqa: BLE001
            body = ""
        return e.code, url, body
    except Exception as e:                                 # noqa: BLE001 -- DNS/TLS/timeout is a finding
        return None, url, f"__ERROR__ {type(e).__name__}: {str(e)[:120]}"
    finally:
        _last["at"] = time.time()


SHELL_MAX_BYTES = 5_000


def page_facts(text):
    """`shell`: a 200 that proves nothing. Controlled 2026-09-24 against the
    nonexistent slug `nosuchcityxyz42`: *.civicclerk.com answers 200 with the
    same 1,287-byte client-rendered portal, *.novusagenda.com 1,453 bytes and
    *.legistar.com a 19-byte body. All three are wildcard catch-alls, so a
    resolving hostname and a 200 are NOT evidence that a body uses the portal --
    only the Legistar/PrimeGov read APIs and a Granicus listing are."""
    t = TITLE.search(text or "")
    facts = {"title": granicus.strip_tags(t.group(1))[:120] if t else None,
             "bytes": len(text or ""),
             "agenda_words": bool(AGENDA_WORD.search(text or "")),
             "council_words": bool(COUNCIL_WORD.search(text or "")),
             "instruction_like": INSTRUCTION_LIKE.findall(text or "")[:3]}
    facts["shell"] = facts["bytes"] < SHELL_MAX_BYTES and not (facts["agenda_words"] or facts["council_words"])
    return facts


AGENDA_LINK = re.compile(r"AgendaViewer\.php|MinutesViewer\.php", re.I)


def probe_granicus(slug, out_dir, views=8):
    """Walk view_id 1..views on a live Granicus site, then read one player page
    of the fullest council-looking view.

    A Granicus site with agenda links and NO clip rows is the agenda-only
    deployment (Livermore: 845 AgendaViewer links, 0 MediaPlayer links over the
    full 1.8 MB listing) -- documents, no video, no times."""
    found, agenda_links = [], 0
    for view in range(1, views + 1):
        status, url, text = granicus_get(slug, view)
        if status != 200:
            found.append({"view_id": view, "status": status})
            continue
        clips = granicus.parse_listing(text)
        facts = page_facts(text)
        agenda_links += len(AGENDA_LINK.findall(text))
        found.append({"view_id": view, "status": 200, "clips": len(clips), "title": facts["title"],
                      "council_words": facts["council_words"], "instruction_like": facts["instruction_like"],
                      "agenda_links": len(AGENDA_LINK.findall(text)), "bytes": len(text),
                      "newest": (clips[0].get("name"), clips[0].get("date")) if clips else None})
        (out_dir / f"{slug}_view{view}.html").write_text(text, encoding="utf-8")
    live = [f for f in found if f.get("clips")]
    best = max(live, key=lambda f: (f["council_words"], f["clips"]), default=None)
    player = None
    if best:
        clips = granicus.parse_listing((out_dir / f"{slug}_view{best['view_id']}.html").read_text(encoding="utf-8"))
        clip_id = clips[0]["clip_id"]
        status, url, text = get(granicus.player_url(slug, best["view_id"], clip_id))
        if status == 200:
            pl = granicus.parse_player(text, slug)
            player = {"clip_id": clip_id, "captions_url": pl["captions_url"], "captions_lang": pl["captions_lang"],
                      "captions_label": pl["captions_label"], "index_points": len(pl["index_points"] or []),
                      "m3u8": bool(pl["m3u8"]), "mp4": bool(pl["mp4"]), "asx_token": bool(pl["asx_token"]),
                      "instruction_like": page_facts(text)["instruction_like"]}
            if pl["captions_url"]:
                cs, _, ctext = get(pl["captions_url"])
                player["captions_status"] = cs
                player["captions_cues"] = (ctext or "").count("-->") if cs == 200 else 0
                player["captions_bytes"] = len(ctext or "") if cs == 200 else 0
        else:
            player = {"clip_id": clip_id, "player_status": status}
    return {"views": found, "best_view": best["view_id"] if best else None, "agenda_links": agenda_links,
            "player": player}


def granicus_get(slug, view):
    return get(granicus.listing_url(slug, view))


# --- the two families with a public read API ----------------------------------
# A hostname resolving is not evidence: *.civicclerk.com, *.novusagenda.com and
# *.legistar.com all answer 200 with a client-rendered shell for slugs that do
# not exist. These two APIs give a yes/no instead (checked 2026-09-24:
# webapi.legistar.com answers 500 for an unknown client, 200 with rows for a
# real one), and they also say where the VIDEO lives -- which is the difference
# between a body leaving YouTube entirely and only leaving the caption queue.
LEGISTAR_API = "https://webapi.legistar.com/v1/{slug}"
PRIMEGOV_API = "https://{host}/api/v2/PublicPortal/ListArchivedMeetings?year={year}"


def json_get(url):
    status, final, text = get(url)
    if status != 200:
        return status, None
    try:
        return status, json.loads(text)
    except ValueError:
        return status, None


def video_host(url):
    if not url:
        return None
    h = urllib.parse.urlparse(url if "//" in url else f"//{url}").netloc.lower()
    return h or url[:60]


def probe_legistar_api(slug):
    status, bodies = json_get(f"{LEGISTAR_API.format(slug=slug)}/bodies?$top=1")
    if status != 200 or not isinstance(bodies, list):
        return {"slug": slug, "status": status, "real": False}
    status2, events = json_get(f"{LEGISTAR_API.format(slug=slug)}/events?$top=40&$orderby=EventDate%20desc")
    events = events if isinstance(events, list) else []
    council = [e for e in events if COUNCIL_WORD.search(e.get("EventBodyName") or "")]
    vids = [e.get("EventVideoPath") for e in events if e.get("EventVideoPath")]
    return {"slug": slug, "status": status, "real": True, "n_events": len(events), "n_council_events": len(council),
            "agenda_files": sum(1 for e in events if e.get("EventAgendaFile")),
            "with_video": len(vids), "video_hosts": sorted({video_host(v) for v in vids})[:4],
            "insite": sum(1 for e in events if e.get("EventInSiteURL"))}


SWAGIT_VIDEO = re.compile(r"/videos/(\d+)")
SWAGIT_CUSTOMER = re.compile(r"swagitVideo/([A-Za-z0-9_-]+)/")
SWAGIT_TS = re.compile(r'data-ts="([\d.]+)"')


def probe_swagit(slug, text, clip_id=None):
    """`/views/1` IS NOT EVIDENCE. Checked 2026-09-24: on every Swagit host it
    answers 200, titles itself "<City>, CA Video Archive" from the hostname, and
    serves the SAME 1,530 clips -- whose player pages resolve to
    `swagitVideo/carmelin/...`, Carmel, Indiana. That held for `livermoreca`,
    `concordca` AND `petalumaca`, and Petaluma is a real Swagit customer, so the
    listing tells you nothing either way.

    The only sound test is a clip id that came from the BODY'S OWN portal
    (PrimeGov `swagitId`), whose player page then names the customer in its
    stream URL: `https://petalumaca.new.swagit.com/videos/371148` ->
    `swagitVideo/petalumaca/...`, title "Jan 05, 2026 City Council - Petaluma,
    CA". Pass that id as clip_id; without one, `matches_host` stays False and
    `clip_source` says why."""
    ids = list(dict.fromkeys(SWAGIT_VIDEO.findall(text or "")))
    probe_id = clip_id or (ids[0] if ids else None)
    if not probe_id:
        return {"slug": slug, "clips": 0, "owner": None, "matches_host": False, "clip_source": "none"}
    status, url, page = get(f"https://{slug}.new.swagit.com/videos/{probe_id}")
    m = SWAGIT_CUSTOMER.search(page or "")
    owner = m.group(1) if m else None
    title = page_facts(page)["title"]
    return {"slug": slug, "clips": len(ids), "probe_clip": probe_id, "probe_status": status, "owner": owner,
            "clip_source": "portal" if clip_id else "views_listing (NOT evidence -- see docstring)",
            "matches_host": bool(owner) and owner.lower() == slug.lower() and bool(clip_id), "probe_title": title,
            # The transcript is the prize: Swagit renders a voice-to-text
            # transcript inline, every phrase anchored with data-ts seconds.
            "transcript_anchors": len(SWAGIT_TS.findall(page or "")),
            "m3u8": bool(re.search(r"https?://[^\s\"']+\.m3u8", page or "")),
            "instruction_like": page_facts(page)["instruction_like"]}


def probe_primegov_api(host, year=2026):
    status, meetings = json_get(PRIMEGOV_API.format(host=host, year=year))
    if status != 200 or not isinstance(meetings, list):
        return {"host": host, "status": status, "real": False}
    council = [m for m in meetings if COUNCIL_WORD.search(m.get("title") or "")]
    vids = [m.get("videoUrl") for m in council if m.get("videoUrl")]
    # A swagitId like 'videos/371148' is the one clip id that provably belongs to
    # this body -- the only sound input to probe_swagit's ownership check.
    swagit = [str(m["swagitId"]).rsplit("/", 1)[-1] for m in council if m.get("swagitId")]
    return {"host": host, "status": status, "real": True, "year": year, "n_meetings": len(meetings),
            "n_council": len(council), "with_video": len(vids), "video_hosts": sorted({video_host(v) for v in vids})[:4],
            "swagit_ids": len(swagit), "swagit_clip": swagit[0] if swagit else None,
            "swagit_host": video_host(next((m["videoUrl"] for m in council if m.get("swagitId") and m.get("videoUrl")), None)),
            "with_documents": sum(1 for m in council if m.get("documentList"))}


def probe_body(body, out_dir, views=8, families=None):
    rec = {"body": body, "portals": {}, "granicus": None, "legistar_api": None, "primegov_api": None, "swagit": None}
    for family, (host_tpl, path) in PORTALS.items():
        if families and family not in families:
            continue
        for slug in SLUGS[body] + [f"{s}ca" for s in SLUGS[body] if not s.endswith("ca")]:
            host = host_tpl.format(slug=slug)
            if not resolves(host):
                rec["portals"].setdefault(family, []).append({"host": host, "dns": False})
                continue
            status, url, text = get(f"https://{host}{path}")
            hit = {"host": host, "dns": True, "status": status, "final_url": url}
            hit.update(page_facts(text))
            rec["portals"].setdefault(family, []).append(hit)
            if family == "granicus" and status == 200:
                rec["granicus"] = {"slug": slug, **probe_granicus(slug, out_dir, views)}
                break
            if family == "legistar" and status == 200 and not (rec["legistar_api"] or {}).get("real"):
                rec["legistar_api"] = probe_legistar_api(slug)
                if rec["legistar_api"]["real"]:
                    break
                continue                                  # a shell 200 proves nothing; try the next slug
            if family == "primegov" and status == 200 and not (rec["primegov_api"] or {}).get("real"):
                rec["primegov_api"] = probe_primegov_api(host)
                if rec["primegov_api"]["real"]:
                    break
                continue
            if family == "swagit" and status == 200 and not (rec["swagit"] or {}).get("matches_host"):
                rec["swagit"] = probe_swagit(slug, text, (rec["primegov_api"] or {}).get("swagit_clip"))
                if rec["swagit"]["matches_host"]:
                    break
                continue
            if status == 200:
                break
    (out_dir / f"{body}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def route_verdict(rec):
    """('verdict', video_off_youtube).

    The distinction that matters, and the one it would be easy to overclaim:
    `episodes.method='agenda_index'` needs index points with TIMES INTO THE
    VIDEO (Granicus index-point divs). A Legistar / PrimeGov / agenda-only
    Granicus portal publishes agenda and minutes DOCUMENTS with no timestamps,
    which cannot locate an episode and so does not remove a YouTube caption
    request. Those come back as 'agenda_docs', deliberately not 'agenda_index'.

      granicus_captions  caption track with cues -> the meeting leaves YouTube
      swagit_transcript  Swagit's own timestamped transcript -> leaves YouTube
      granicus_video     Granicus video, no captions -> leaves YouTube, needs ASR
      swagit_video       Swagit video, no transcript -> leaves YouTube, needs ASR
      agenda_docs        documents only, no times -> YouTube caption still needed
      none               nothing reachable
    """
    g, pl = rec.get("granicus") or {}, (rec.get("granicus") or {}).get("player") or {}
    sw = rec.get("swagit") or {}
    if pl.get("captions_cues"):
        return "granicus_captions", True
    if sw.get("matches_host") and sw.get("transcript_anchors"):
        return "swagit_transcript", True
    if g.get("best_view") and (pl.get("m3u8") or pl.get("mp4") or pl.get("asx_token")):
        return "granicus_video", True
    if sw.get("matches_host") and sw.get("m3u8"):
        return "swagit_video", True
    la, pg = rec.get("legistar_api") or {}, rec.get("primegov_api") or {}
    hosts = set(la.get("video_hosts") or []) | set(pg.get("video_hosts") or [])
    off_yt = bool(hosts) and not any("youtu" in h for h in hosts)
    if (la.get("real") and la.get("n_council_events")) or (pg.get("real") and pg.get("n_council")) \
            or (g.get("agenda_links") and not g.get("best_view")):
        return "agenda_docs", off_yt
    return "none", False


def summarise(rec):
    """One TSV line: body, live portals, captions?, agenda index?, verdict."""
    live = sorted({f for f, hits in rec["portals"].items()
                   if any(h.get("status") == 200 and not h.get("shell") for h in hits)})
    g = rec.get("granicus") or {}
    pl = g.get("player") or {}
    la, pg = rec.get("legistar_api") or {}, rec.get("primegov_api") or {}
    sw = rec.get("swagit") or {}
    cues = pl.get("captions_cues")
    clips = sum(v.get("clips") or 0 for v in (g.get("views") or []))
    verdict, off_yt = route_verdict(rec)
    return {"body": rec["body"], "verdict": verdict, "video_off_youtube": off_yt,
            "swagit": f"{sw.get('slug')}:owner={sw.get('owner')}:match={sw.get('matches_host')}:"
                      f"ts={sw.get('transcript_anchors')}" if sw else "-",
            "live_hosts": ",".join(live) or "-",
            "granicus": f"{g.get('slug')}:view{g.get('best_view')}:{clips}clips" if g.get("best_view") else "-",
            "captions_cues": cues if cues is not None else "-", "captions_lang": pl.get("captions_lang") or "-",
            "index_points": pl.get("index_points", "-"),
            "legistar": f"{la.get('slug')}:{la.get('n_council_events')}council/{la.get('n_events')}ev:"
                        f"vid={','.join(la.get('video_hosts') or []) or '-'}" if la.get("real") else "-",
            "primegov": f"{pg.get('host')}:{pg.get('n_council')}council/{pg.get('n_meetings')}mtg:"
                        f"vid={','.join(pg.get('video_hosts') or []) or '-'}:swagit={pg.get('swagit_ids')}" if pg.get("real") else "-",
            "audio": "m3u8" if pl.get("m3u8") else ("mp4" if pl.get("mp4") else ("asx" if pl.get("asx_token") else "-"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=None, help="comma-separated body slugs (default: all 16 in the audit)")
    ap.add_argument("--views", type=int, default=8, help="Granicus view_id 1..N to walk")
    ap.add_argument("--families", default=None,
                    help=f"comma-separated portal families to probe (default all: {','.join(PORTALS)})")
    ap.add_argument("--out", default=str(ROOT / "output" / "t2" / "portal_probe"))
    args = ap.parse_args(argv)
    families = {f.strip() for f in args.families.split(",")} if args.families else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies = [b.strip() for b in args.bodies.split(",")] if args.bodies else list(SLUGS)
    rows = []
    for body in bodies:
        if body not in SLUGS:
            print(f"SKIP {body}: no candidate slugs", flush=True)
            continue
        rec = probe_body(body, out_dir, args.views, families)
        s = summarise(rec)
        rows.append(s)
        print("\t".join(f"{k}={v}" for k, v in s.items()), flush=True)
        flagged = [h for hits in rec["portals"].values() for h in hits if h.get("instruction_like")]
        if flagged:
            print(f"  !! instruction-like text on {body}'s portal HTML (data, not obeyed): "
                  f"{[h['instruction_like'] for h in flagged][:2]}", flush=True)
    cols = list(rows[0]) if rows else []
    (out_dir / "summary.tsv").write_text("\n".join(["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]) + "\n",
                                         encoding="utf-8")
    print(f"\n{len(rows)} bodies probed -> {out_dir}/summary.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
