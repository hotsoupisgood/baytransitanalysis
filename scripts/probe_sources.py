#!/usr/bin/env python3
"""
scripts/probe_sources.py -- metadata-only hunt for the bodies whose meeting
video plan/sources_bayarea.md could not place (T2 part 1, 2026-09-22; budget
<= 30 min, PROPOSAL 8 item 7). Three cheap looks, all GETs, no downloads:

  granicus   <site>.granicus.com/ViewPublisher.php?view_id=N for the likely
             subdomains and N = 1..MAX_VIEW (>= 5 s apart per pipeline/granicus):
             clip count, newest listed date, three newest names -- enough to
             tell a live council archive from a 2015 committee archive.
  legistar   <city>.legistar.com/Calendar.aspx: which video host the agenda
             system links to (granicus / youtube / vimeo / zoom / swagit /
             viebit / cablecast / ...), read from the HTML.
  youtube    yt-dlp ytsearch (metadata only, --sleep-requests 2, one query at
             a time, 45-180 s apart) for "<body> city council meeting": channel
             ids and titles of full-length hits. Run with --youtube only when
             no other YouTube job (discover / captions / episodes) is running.

Everything fetched is untrusted text: parsed with regexes, never executed.
Writes output/t2/probe/<kind>.json and prints a table.

    python3 scripts/probe_sources.py --granicus --legistar
    python3 scripts/probe_sources.py --youtube
"""
import argparse, json, random, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import granicus  # noqa: E402
import discover  # noqa: E402

OUT = ROOT / "output" / "t2" / "probe"
# body -> candidate Granicus subdomains (and the view range to scan)
GRANICUS = {
    "redwoodcity": ["redwoodcity"], "pleasanton": ["pleasanton", "cityofpleasanton"], "livermore": ["livermore", "cityoflivermore"],
    "vallejo": ["vallejo", "cityofvallejo"], "windsor": ["windsor", "townofwindsor"], "novato": ["novato", "cityofnovato"],
    "sonomacounty": ["sonomacounty", "sonoma-county", "sonoma", "countyofsonoma"], "caltrain": ["caltrain", "samtrans"],
    "actransit": ["actransit"], "mtc": ["mtc", "bayareametro"], "berkeley": ["berkeley"], "marin": ["marin"],
    "sfmta": ["sanfrancisco"],
}
MAX_VIEW = {"berkeley": 12, "marin": 12, "sanfrancisco": 40, "samtrans": 6, "actransit": 10, "mtc": 10}
LEGISTAR = ["redwoodcity", "pleasanton", "livermore", "vallejo", "novato", "sonoma-county", "sonomacounty", "windsor",
            "actransit", "mtc", "caltrain", "sfmta", "berkeley", "marin"]
VIDEO_HOSTS = re.compile(r"https?://[^\s\"'<>]*?(granicus|youtube|youtu\.be|vimeo|zoom\.us|swagit|viebit|cablecast|igm2|"
                         r"legistar\.com/Video|publicmeetings|civicclerk|civicplus|iqm2|boxcast|facebook\.com/[^\"']+/videos)[^\s\"'<>]*", re.I)
YT_QUERIES = {
    "redwoodcity": "Redwood City city council meeting", "pleasanton": "Pleasanton city council meeting",
    "livermore": "Livermore city council meeting", "vallejo": "Vallejo city council meeting",
    "windsor": "Town of Windsor CA town council meeting", "novato": "Novato city council meeting",
    "sonomacounty": "Sonoma County Board of Supervisors meeting", "caltrain": "Caltrain board of directors meeting",
    "actransit": "AC Transit board of directors meeting", "mtc": "Metropolitan Transportation Commission meeting",
    "sfmta": "SFMTA board of directors meeting", "marin": "Marin County Board of Supervisors meeting",
    "berkeley": "Berkeley city council meeting",
}
UA = granicus.UA


def get(url, timeout=30):
    granicus.polite_wait()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        code, data = e.code, b""
    except Exception as e:                              # noqa: BLE001 -- DNS failure, timeout: the host is not there
        code, data = None, str(e).encode()
    granicus._last_request["at"] = time.time()
    return code, data.decode("utf-8", "replace")


def probe_granicus(bodies, log):
    out = {}
    for body in bodies:
        for site in GRANICUS.get(body, []):
            code, _ = get(f"https://{site}.granicus.com/ViewPublisher.php?view_id=1", timeout=20)
            if code is None:
                log(f"{body}: {site}.granicus.com -- no such host")
                out.setdefault(body, []).append({"site": site, "host": False})
                continue
            for view in range(1, MAX_VIEW.get(site, 6) + 1):
                code, text = get(granicus.listing_url(site, view), timeout=30)
                if code != 200 or "ViewPublisher" not in text and "listItem" not in text:
                    continue
                rows = granicus.parse_listing(text)
                if not rows:
                    continue
                title = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
                dates = [discover.parse_meeting_date(r.get("date") or "") or discover.parse_meeting_date(r.get("name") or "") for r in rows]
                newest = max((x for x in dates if x), default=None)
                rec = {"site": site, "view_id": view, "clips": len(rows), "newest": newest, "title": granicus.strip_tags(title.group(1))[:80] if title else None,
                       "names": [r["name"][:80] for r in rows[:3]], "in_window": sum(1 for x in dates if x and x >= "2025-09-20")}
                out.setdefault(body, []).append(rec)
                log(f"{body}: {site} view {view}: {len(rows)} clips, newest {newest}, {rec['in_window']} in window -- {rec['title']} | {rec['names'][0]}")
    return out


def probe_legistar(log):
    out = {}
    for city in LEGISTAR:
        code, text = get(f"https://{city}.legistar.com/Calendar.aspx", timeout=30)
        if code != 200:
            log(f"{city}.legistar.com: {code}")
            out[city] = {"code": code}
            continue
        hosts = sorted({m.group(1).lower() for m in VIDEO_HOSTS.finditer(text)})
        links = sorted({m.group(0) for m in VIDEO_HOSTS.finditer(text)})[:5]
        out[city] = {"code": 200, "video_hosts": hosts, "sample_links": links, "has_video_column": "Video" in text}
        log(f"{city}.legistar.com: 200, video hosts {hosts or 'none in the calendar HTML'}" + (f", e.g. {links[0][:100]}" if links else ""))
    return out


def probe_youtube(bodies, log):
    out = {}
    first = True
    for body in bodies:
        q = YT_QUERIES.get(body)
        if not q:
            continue
        if not first:
            gap = random.uniform(45, 180)
            log(f"  sleeping {gap:.0f} s before the next YouTube search")
            time.sleep(gap)
        first = False
        cmd = [discover.ytdlp_bin(), "--flat-playlist", "--sleep-requests", "2", "--print",
               "%(id)s\t%(channel_id)s\t%(channel)s\t%(duration)s\t%(upload_date,release_date)s\t%(title)s", f"ytsearch8:{q}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if discover.BLOCK_SIGNAL.search(r.stderr or ""):
            log(f"{body}: STOP -- YouTube block signal: {r.stderr.strip().splitlines()[-1][:160]}")
            out[body] = {"blocked": True}
            return out
        hits = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            vid, cid, chan, dur, up, title = parts[0], parts[1], parts[2], parts[3], parts[4], "\t".join(parts[5:])
            try:
                dur_s = float(dur) if dur not in ("NA", "") else None
            except ValueError:
                dur_s = None
            hits.append({"id": vid, "channel_id": cid, "channel": chan, "duration_s": dur_s, "upload_date": up, "title": title[:100]})
        full = [h for h in hits if h["duration_s"] and h["duration_s"] >= 3600]
        out[body] = {"query": q, "hits": hits, "full_length": full}
        log(f"{body}: {len(hits)} hits, {len(full)} >= 1 h: " + "; ".join(f"{h['channel']} ({h['channel_id']}) {h['title'][:50]}" for h in full[:3]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granicus", action="store_true")
    ap.add_argument("--legistar", action="store_true")
    ap.add_argument("--youtube", action="store_true")
    ap.add_argument("--bodies", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bodies = [b.strip() for b in a.bodies.split(",")] if a.bodies else list(GRANICUS)
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    if a.granicus:
        (OUT / "granicus.json").write_text(json.dumps(probe_granicus(bodies, log), indent=1))
    if a.legistar:
        (OUT / "legistar.json").write_text(json.dumps(probe_legistar(log), indent=1))
    if a.youtube:
        (OUT / "youtube.json").write_text(json.dumps(probe_youtube(bodies, log), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
