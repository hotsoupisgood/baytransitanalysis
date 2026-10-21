#!/usr/bin/env python3
"""
pipeline/granicus.py -- the Granicus archive adapter (PROPOSAL-transit.md 3b,
T0 test 7; T1 build 2026-09-21).

A Granicus archive is three pages:

  listing   https://<site>.granicus.com/ViewPublisher.php?view_id=<N>
            one <tr> per clip: name, date, duration, agenda / minutes links.
            Three templates exist -- the new one (Oakland: `td class="listItem
            clipName"`), the 2010-era one (AC Transit, MTC: `td
            class="listItem" headers="Name"` with the epoch in a hidden span)
            and the 2026 `tr class="listingRow"` one (SamTrans, Hayward, SF
            BoS, Fremont, Alameda, Walnut Creek, Napa: `headers="Date <slug>"`,
            the MediaPlayer link inside an onClick; upcoming events without a
            clip_id are skipped).
  player    https://<site>.granicus.com/MediaPlayer.php?view_id=<N>&clip_id=<id>
            carries the HLS master (`.../playlist.m3u8`), sometimes a
            progressive MP4, a `<track kind="captions" src="/videos/<id>/captions.vtt">`
            when the site serves closed captions, and the agenda index points
            (`<div class="index-point" time="<s>">title</div>`).
  captions  the .vtt -- whole-meeting captions, the same input segment.py reads
            from a YouTube auto track (T0 test 7: 6,215 cues over 3 h 42 m).

Audio (T0 test 7): the HLS rendition is video+audio only (no EXT-X-MEDIA audio
group), ~410 MB per meeting-hour on the wire, and `ffmpeg -ss <s> -t <n> -vn`
on the master DOES seek (2-s segments; a 5-min slice opened 154 segments in
34 s). So episodes are fetched as audio-only SLICES of the stream, never the
whole meeting and never the MP4 (`audio_slice`). Politeness: >= 5 s between
page requests (GAP_S), one ffmpeg at a time.

Everything fetched is untrusted text: titles and captions are data, parsed
with regexes, never executed. Unit tests run on saved fixtures
(analysis/fixtures/granicus/*, from the T0 probe) -- nothing here needs the
network unless you call fetch_*.

    python3 pipeline/granicus.py listing oakland 2 [--limit 5]      # print recent clips (network)
    python3 pipeline/granicus.py player  oakland 2 7755              # stream / captions / index points (network)
"""
import argparse, html, json, re, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

# The Wowza stream host (archive-stream.granicus.com) answers 403 to any UA that
# does not start with "Mozilla/5.0" (checked 2026-09-21: 403 vs 200 on the same
# playlist URL); the identifier after it still says who we are.
UA = "Mozilla/5.0 (Macintosh) ytpt-research/1.0 (council meeting transcription; polite; one request at a time)"
GAP_S = 5.0                      # minimum pause between page requests to one site
_last_request = {"at": 0.0}

# Listing templates seen: odd/even rows (Oakland; AC Transit / MTC with a hidden
# epoch), "listingRow" (SamTrans, Hayward, Fremont, Napa), "listLine" (SF BoS),
# bare <tr> (Alameda, Walnut Creek). parse_listing no longer keys on the row
# class (2026-09-22): any <tr> with a clip_id link is a clip.
KEY = {"clipName": "name", "Name": "name", "clipDate": "date", "Date": "date", "clipDuration": "duration", "Duration": "duration"}
INDEX_POINT = re.compile(r'<div[^>]*class="index-point[^"]*"[^>]*\btime="(\d+(?:\.\d+)?)"[^>]*>(.*?)</div>', re.S)
TRACK = re.compile(r'<track[^>]*kind="captions"[^>]*src="([^"]+)"', re.I)
# Santa Clara and Hayward serve a full English track with NO <track> element:
# the player config carries it as a Flowplayer subtitles object (found by the
# 2026-09-22 T2 probe, after 69 of their meetings had been recorded
# captions='none' and sent to 'no_episodes'). lang/label are kept so a
# non-English track (Napa's Spanish one) is visible instead of silently used.
JS_TRACK = re.compile(r'kind:\s*"(?:subtitles|captions)"\s*,\s*lang:\s*"([^"]*)"\s*,'
                      r'\s*label:\s*"([^"]*)"\s*,\s*src:\s*"([^"]+)"', re.I)
M3U8 = re.compile(r'https?://[^\s"\'<>]+?\.m3u8[^\s"\'<>]*')
MP4 = re.compile(r'https?://archive-video\.granicus\.com/[^\s"\'<>]+?\.mp4')
# The 2015-era Flash player (Richmond, San Leandro) puts NO stream URL on the
# page: it hands its VideoUrl to ASX.php, which answers with an rtmp:// path on
# archive-video.granicus.com. The same path is served over HTTPS with byte
# ranges (checked 2026-09-22 on Richmond: 200, Accept-Ranges, 1.79 GB for
# 7.5 h), so audio for those archives is reachable after all -- which is what
# an ASR locate pass needs. The 32-hex `r` token on the page is part of the
# ASX query.
ASX_TOKEN = re.compile(r"(?:[?&]|%26|%3F)r(?:=|%3D)([0-9a-f]{16,})", re.I)
ASX_REF = re.compile(r'<REF\s+HREF="([^"]+)"', re.I)


def strip_tags(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


ANY_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
ANY_CELL = re.compile(r"<td\b([^>]*)>(.*?)</td>", re.S)
_DATE_WORDS = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2},?\s*\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)
_DURATION = re.compile(r"^\s*(?:\d{1,2}:\d{2}:\d{2}|\d+h\s*\d+m?|\d+\s*min)\s*$", re.I)


def parse_listing(text):
    """[{clip_id, name, date, duration, duration_s, epoch, agenda, minutes}] in
    listing order (newest first on every template seen; the 2010-era one
    carries an epoch we sort by when every row has one).

    Template-agnostic since 2026-09-22: every <tr> that links a clip_id is a
    clip. Cells are keyed by their `headers="Name|Date|Duration ..."` attribute
    or `class="clipName|clipDate|clipDuration"`; where a template has neither
    (SF BoS: `<td align="left">Wednesday, December 11, 2024</td>`) the first
    cell whose text is a date is the date, one that looks like HH:MM:SS or
    NNh NNm is the duration, and the first other non-empty, non-link cell is
    the name. Rows without a clip_id (upcoming events) are skipped."""
    rows = []
    for m in ANY_ROW.finditer(text):
        row = m.group(1)
        clip = re.search(r"clip_id=(\d+)", row)
        if not clip:
            continue
        cells, loose = {}, []
        for attrs, v in ANY_CELL.findall(row):
            cls = re.search(r'class="listItem(?: (\w+))?"', attrs)
            hdr = re.search(r'headers="(\w+)', attrs)
            k = (KEY.get(cls.group(1)) if cls else None) or (KEY.get(hdr.group(1)) if hdr else None)
            txt = strip_tags(v)
            if k and k not in cells:
                cells[k] = txt
            elif txt and "<a " not in v:
                loose.append(txt)
        for txt in loose:
            if "date" not in cells and _DATE_WORDS.search(txt):
                cells["date"] = txt
            elif "duration" not in cells and _DURATION.match(txt):
                cells["duration"] = txt
        if "name" not in cells:
            rest = [t for t in loose if t not in (cells.get("date"), cells.get("duration")) and not t.isdigit()]
            if rest:
                cells["name"] = rest[0]
        if "name" not in cells:
            # SF BoS: the name cell is blank and the clip's title survives only as
            # the slug in headers="Agenda<Title-With-Hyphens>" / "Minutes <slug>"
            slug = re.search(r'headers="(?:Agenda|Minutes)\s*([^"]+)"', row)
            if slug:
                cells["name"] = " ".join(slug.group(1).replace("-", " ").split())
        if "name" not in cells:
            continue
        ep = re.search(r'display:none;">(\d{9,10})<', row)
        date = re.sub(r"^\d{9,10}\s*", "", cells.get("date", ""))
        rows.append({"clip_id": clip.group(1), "name": cells["name"], "date": date,
                     "duration": cells.get("duration", ""), "duration_s": parse_duration(cells.get("duration", "")),
                     "epoch": int(ep.group(1)) if ep else None,
                     "agenda": "AgendaViewer" in row, "minutes": "MinutesViewer" in row})
    seen, uniq = set(), []                                   # one row per clip
    for r in rows:
        if r["clip_id"] not in seen:
            seen.add(r["clip_id"]); uniq.append(r)
    rows = uniq
    if rows and all(r["epoch"] for r in rows):
        rows.sort(key=lambda r: -r["epoch"])
    return rows


def parse_duration(s):
    """'03h 42m' -> 13320.0; '02h 05m' -> 7500.0; '01:24:22' -> 5062.0 (SF); '' -> None."""
    hms = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", s or "")
    if hms:
        return float(int(hms.group(1)) * 3600 + int(hms.group(2)) * 60 + int(hms.group(3)))
    m = re.search(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?", s or "")
    if not m or not any(m.groups()):
        return None
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return float(h * 3600 + mi * 60 + se)


def parse_player(text, site=None):
    """{m3u8, mp4, captions_url, captions_lang, captions_label, index_points:
    [(time_s, title)]} from a MediaPlayer.php page. captions_url is absolute
    when site is given. Two templates carry the track: a <track kind="captions">
    element (Oakland, Napa, Sausalito, SamTrans) and a Flowplayer subtitles
    object in the player config (Santa Clara, Hayward) -- the element wins when
    both are present, and either way the endpoint may still be the empty
    8-byte file some sites serve (fetch_captions returns 0 cues for those)."""
    m3u8 = sorted(set(M3U8.findall(text)))
    mp4 = sorted(set(MP4.findall(text)))
    trk = TRACK.search(text)
    cap = trk.group(1) if trk else None
    lang = label = None
    if not cap:
        js = JS_TRACK.search(text)
        if js:
            lang, label, cap = js.group(1) or None, js.group(2) or None, js.group(3)
            cap = cap.replace("\\/", "/")
    if cap and cap.startswith("/") and site:
        cap = f"https://{site}.granicus.com{cap}"
    pts = [(float(t), strip_tags(title)) for t, title in INDEX_POINT.findall(text)]
    pts.sort()
    tok = ASX_TOKEN.search(text)
    return {"m3u8": m3u8[0] if m3u8 else None, "mp4": mp4[0] if mp4 else None,
            "captions_url": cap, "captions_lang": lang, "captions_label": label, "index_points": pts,
            "asx_token": tok.group(1) if tok else None}


def asx_url(site, view_id, clip_id, token=None):
    q = f"intro=1&view_id={view_id}&clip_id={clip_id}" + (f"&r={token}" if token else "") \
        + f"&sn={site}.granicus.com&stream_type=rtmp"
    return f"https://{site}.granicus.com/ASX.php?{q}"


def archive_mp4_from_asx(text):
    """The https progressive MP4 an ASX.php response points at, or None.
    `rtmp://archive-video.granicus.com/<site>/<file>.mp4` is the same object as
    `https://archive-video.granicus.com/<site>/<file>.mp4`."""
    m = ASX_REF.search(text or "")
    if not m:
        return None
    ref = m.group(1).strip()
    if ref.startswith("rtmp://"):
        ref = "https://" + ref[len("rtmp://"):]
    return ref if ref.startswith("https://") and ref.endswith(".mp4") else None


def fetch_archive_mp4(site, view_id, clip_id, token=None):
    """One extra request for an archive whose player page carries no stream
    (parse_player m3u8 and mp4 both None). Returns the https MP4 URL or None."""
    text, _ = fetch(asx_url(site, view_id, clip_id, token))
    return archive_mp4_from_asx(text)


def agenda_item_for(index_points, start_s, end_s):
    """The index-point title whose span [time, next time) overlaps [start, end)
    the most; None when there are no index points."""
    best, best_ov = None, 0.0
    for i, (t, title) in enumerate(index_points):
        t_end = index_points[i + 1][0] if i + 1 < len(index_points) else float("inf")
        ov = max(0.0, min(end_s, t_end) - max(start_s, t))
        if ov > best_ov:
            best, best_ov = title, ov
    return best


def video_id_for(site, clip_id):
    return f"gr-{site}-{clip_id}"


def listing_url(site, view_id):
    return f"https://{site}.granicus.com/ViewPublisher.php?view_id={view_id}"


def player_url(site, view_id, clip_id):
    return f"https://{site}.granicus.com/MediaPlayer.php?view_id={view_id}&clip_id={clip_id}"


def parse_source_url(url):
    """'https://oakland.granicus.com/ViewPublisher.php?view_id=2' -> ('oakland', '2')."""
    u = urllib.parse.urlparse(url)
    site = u.netloc.split(".granicus.com")[0]
    view = urllib.parse.parse_qs(u.query).get("view_id", [None])[0]
    if not site or not view:
        raise ValueError(f"not a Granicus listing URL: {url}")
    return site, view


# --- network (polite) ----------------------------------------------------------

def polite_wait(gap=GAP_S):
    dt = time.time() - _last_request["at"]
    if dt < gap:
        time.sleep(gap - dt)


def fetch(url, timeout=60, dest=None):
    """GET with the project UA, >= GAP_S after the previous request. Returns
    (text, {status, bytes, s}). Raises urllib.error.HTTPError on 4xx/5xx."""
    polite_wait()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        info = {"status": r.status, "bytes": len(data), "s": round(time.time() - t0, 2)}
    _last_request["at"] = time.time()
    if dest:
        Path(dest).write_bytes(data)
    return data.decode("utf-8", "replace"), info


def fetch_listing(site, view_id):
    text, info = fetch(listing_url(site, view_id))
    return parse_listing(text), info


def fetch_player(site, view_id, clip_id):
    text, info = fetch(player_url(site, view_id, clip_id))
    return parse_player(text, site), info


def fetch_captions(captions_url, dest):
    """Save the caption track; returns (n_bytes, n_cues) -- (0, 0) when the
    endpoint exists but is empty (AC Transit's 8-byte file)."""
    text, info = fetch(captions_url, dest=dest)
    return info["bytes"], text.count("-->")


def audio_slice_command(m3u8_url, start_s, dur_s, dest):
    """Audio only, stream-copied AAC, seeking on the master playlist (T0 test 7:
    -ss before -i jumps to the right 2-s segment instead of reading from 0)."""
    return ["ffmpeg", "-nostdin", "-v", "error", "-y", "-user_agent", UA,
            "-ss", f"{start_s:.3f}", "-i", m3u8_url, "-t", f"{dur_s:.3f}",
            "-vn", "-c:a", "copy", str(dest)]


# How long one slice may take. A flat 1800 s was wrong in both directions and
# took the 2026-09-23 cut run down with it (T2-PLAN 3e): the 53 slices that run
# cut took a median 0.33 x their own length in wall clock and a worst case of
# 2.25 x, because the archive host's throughput varies by an order of magnitude
# through a session -- so 1800 s is generous for a 4-min slice and too tight for
# a 17-min one (gr-santaclara-2497_e1, 1020 s of audio, still running at 1800 s).
# The budget therefore scales with the slice:
#   ratio 4.0   1.8x the worst ratio measured, ~12x the median
#   floor 900 s a short slice is dominated by fixed cost (manifest + seek), not
#               by its length: the worst 2.9-min slice took 392 s
#   max 3 h     past this ffmpeg is hung, not slow; the agenda_index cap is a
#               60-min item (agenda_index.MAX_ITEM_S), whose budget is 4 h
#               without it
SLICE_TIMEOUT_RATIO = 4.0
SLICE_TIMEOUT_FLOOR_S = 900.0
SLICE_TIMEOUT_MAX_S = 3 * 3600.0


def slice_timeout(dur_s, ratio=SLICE_TIMEOUT_RATIO, floor_s=SLICE_TIMEOUT_FLOOR_S, max_s=SLICE_TIMEOUT_MAX_S):
    """Seconds one slice of `dur_s` audio may take before it is this clip's
    failure. See SLICE_TIMEOUT_RATIO for where the numbers come from."""
    return min(max_s, max(floor_s, ratio * float(dur_s or 0)))


def part_path(dest):
    """`.../x_e1.m4a` -> `.../x_e1.part.m4a`. The suffix is KEPT last: ffmpeg
    picks its muxer from the output file's extension, and writing to
    `x_e1.m4a.part` makes every slice fail with "Error opening output files:
    Invalid argument" (measured 2026-09-23, 26 slices)."""
    dest = Path(dest)
    return dest.with_name(f"{dest.stem}.part{dest.suffix}")


def audio_slice(m3u8_url, start_s, dur_s, dest, timeout=None):
    """Run audio_slice_command; returns (ok, detail, wall_s). One ffmpeg at a
    time; the caller spaces calls by GAP_S or more.

    ffmpeg writes to `<dest>.part` and the file is renamed onto `dest` only
    when it exited 0, so **a file at `dest` is always a complete slice** --
    which is what episodes.py assumes when it skips a clip that is already on
    disk. A killed or failed run leaves its bytes at `<dest>.part` for
    inspection (nothing under output/ is deleted; the next attempt overwrites
    that scratch file).

    A slice that runs past slice_timeout() is THIS CLIP's failure, returned
    like any other ffmpeg error. Letting subprocess.TimeoutExpired escape took
    the whole 2026-09-23 cut run down on one slow slice and stranded its
    meeting in 'downloading'."""
    timeout = slice_timeout(dur_s) if timeout is None else timeout
    part = part_path(dest)
    part.parent.mkdir(parents=True, exist_ok=True)
    polite_wait()
    t0 = time.time()
    try:
        r = subprocess.run(audio_slice_command(m3u8_url, start_s, dur_s, part), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _last_request["at"] = time.time()
        wall = time.time() - t0
        return False, (f"ffmpeg timed out after {timeout:.0f} s on a {dur_s:.0f} s slice "
                       f"(partial bytes kept at {part.name})"), wall
    _last_request["at"] = time.time()
    if r.returncode != 0:
        return False, (r.stderr or "").strip().splitlines()[-1:] or [f"ffmpeg exit {r.returncode}"], time.time() - t0
    part.replace(dest)
    return True, f"{Path(dest).stat().st_size / 1e6:.1f} MB", time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["listing", "player"])
    ap.add_argument("site"); ap.add_argument("view_id"); ap.add_argument("clip_id", nargs="?")
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == "listing":
        rows, info = fetch_listing(a.site, a.view_id)
        print(f"{len(rows)} clips ({info['bytes']} B, {info['s']} s)")
        for r in rows[:a.limit]:
            print(f"  {r['clip_id']:>6} {r['date']:<20} {r['duration']:<8} {r['name'][:90]}")
    else:
        if not a.clip_id:
            ap.error("player needs a clip_id")
        p, info = fetch_player(a.site, a.view_id, a.clip_id)
        print(json.dumps({**p, "index_points": p["index_points"][:12], "n_index_points": len(p["index_points"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
