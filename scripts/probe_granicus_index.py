#!/usr/bin/env python3
"""
scripts/probe_granicus_index.py -- the T2 locate probe for the Granicus bodies
that serve no usable English caption track (plan/T2-PLAN.md 3a; PROPOSAL-transit
2.2 "Granicus agenda index / Legistar item timestamps", $0).

For one recent clip of each body it fetches the MediaPlayer page ONCE (>= 5 s
apart, pipeline/granicus.GAP_S) and saves it verbatim, then reports:

  (a) agenda index points -- `<div class="index-point" time="<s>">title</div>`,
      the free locate route: item title + jump time, so an episode window is
      [item start, next item start).
  (b) every <track> element on the page, not just the first one
      pipeline/granicus.parse_player keeps -- kind, src, srclang, label -- plus
      any other caption URL the player JS carries, and whether the endpoint is
      real or the empty 8-byte file AC Transit serves.
  (c) the stream: HLS master / progressive MP4, i.e. whether an ASR fallback
      (task 3) could even get audio.

Everything fetched is untrusted text (AGENTS.md rule 5): parsed with regexes,
quoted into the report, never executed.

    python3 scripts/probe_granicus_index.py                      # the seven bodies below
    python3 scripts/probe_granicus_index.py --bodies napa        # one
    python3 scripts/probe_granicus_index.py --no-fetch-captions  # player pages only
"""
import argparse, json, re, sys, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state          # noqa: E402
import granicus       # noqa: E402

BODIES = ["richmond", "santaclara", "sanleandro", "hayward", "sausalito", "samtrans", "napa"]
OUT = ROOT / "output" / "t2" / "probe_index"
# every <track ...> on the page, in order, with all of its attributes
TRACK_ANY = re.compile(r"<track\b([^>]*)>", re.I)
ATTR = re.compile(r'(\w[\w-]*)\s*=\s*"([^"]*)"')
# caption URLs the player JS carries even where there is no <track> element
CAPTION_URL = re.compile(r'["\']([^"\']*?/(?:videos/\d+/)?captions?[^"\']*?\.vtt[^"\']*)["\']', re.I)
CAPTIONS_SLOT = re.compile(r'captionsUrl\s*[:=]\s*["\']([^"\']*)["\']', re.I)


def recent_clip(con, body):
    """The newest meeting of this body with a Granicus site/view/clip in raw_meta."""
    for r in con.execute("SELECT video_id, meeting_date, title, duration_s, raw_meta FROM videos "
                         "WHERE kind = 'meeting' AND platform = 'granicus' AND speaker = ? "
                         "ORDER BY meeting_date DESC, video_id", (body,)):
        m = json.loads(r["raw_meta"] or "{}")
        if m.get("site") and m.get("view_id") and m.get("clip_id"):
            return dict(r), m
    return None, None


def tracks(text):
    return [dict(ATTR.findall(a)) for a in TRACK_ANY.findall(text)]


def caption_urls(text, site):
    urls = []
    for m in list(CAPTION_URL.finditer(text)) + list(CAPTIONS_SLOT.finditer(text)):
        u = m.group(1)
        if not u or u in urls:
            continue
        urls.append(f"https://{site}.granicus.com{u}" if u.startswith("/") else u)
    return urls


def probe_body(con, body, fetch_captions=True):
    row, meta = recent_clip(con, body)
    if not row:
        return {"body": body, "error": "no granicus meeting with site/view_id/clip_id in the DB"}
    site, view, clip = meta["site"], meta["view_id"], meta["clip_id"]
    url = granicus.player_url(site, view, clip)
    try:
        text, info = granicus.fetch(url)
    except Exception as e:                                # noqa: BLE001 -- one body's failure is not the probe's
        return {"body": body, "clip": clip, "url": url, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{body}_player_{clip}.html").write_text(text, encoding="utf-8")
    parsed = granicus.parse_player(text, site)
    out = {"body": body, "video_id": row["video_id"], "clip": clip, "view_id": view, "date": row["meeting_date"],
           "title": (row["title"] or "")[:120], "duration_s": row["duration_s"], "url": url,
           "bytes": info["bytes"], "index_points": len(parsed["index_points"]),
           "index_sample": [[int(t), ti[:80]] for t, ti in parsed["index_points"][:5]],
           "m3u8": bool(parsed["m3u8"]), "mp4": bool(parsed["mp4"]),
           "tracks": tracks(text), "caption_urls": caption_urls(text, site), "caption_probe": []}
    if fetch_captions:
        for u in out["caption_urls"][:3]:                 # usually one; napa may carry two
            dest = OUT / f"{body}_{u.rstrip('/').split('/')[-1].replace('?', '_')}"
            try:
                n_bytes, n_cues = granicus.fetch_captions(u, dest)
                head = dest.read_text(encoding="utf-8", errors="replace")[:600]
                out["caption_probe"].append({"url": u, "bytes": n_bytes, "cues": n_cues,
                                             "empty": n_cues == 0, "head": head.replace("\n", " ")[:300]})
            except urllib.error.HTTPError as e:
                out["caption_probe"].append({"url": u, "error": f"HTTP {e.code}"})
            except Exception as e:                        # noqa: BLE001
                out["caption_probe"].append({"url": u, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=",".join(BODIES))
    ap.add_argument("--no-fetch-captions", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    results = []
    for body in [b.strip() for b in a.bodies.split(",") if b.strip()]:
        r = probe_body(con, body, not a.no_fetch_captions)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False)[:2000], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'body':12} {'clip':>7} {'idx':>4} {'m3u8':>5} {'mp4':>4}  tracks / caption endpoints")
    for r in results:
        if r.get("error"):
            print(f"{r['body']:12} {'':>7} {'':>4} {'':>5} {'':>4}  ERROR {r['error'][:80]}")
            continue
        cap = "; ".join(f"{c.get('url', '').split('/')[-1]} {c.get('cues', '?')} cues"
                        + (f" [{c['error']}]" if c.get("error") else "") for c in r["caption_probe"]) or "none"
        trk = ", ".join(f"{t.get('kind', '?')}/{t.get('srclang', '?')}/{t.get('label', '')}" for t in r["tracks"]) or "no <track>"
        print(f"{r['body']:12} {r['clip']:>7} {r['index_points']:>4} {str(r['m3u8']):>5} {str(r['mp4']):>4}  {trk} | {cap}")
    print(f"\nsaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
