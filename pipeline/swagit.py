#!/usr/bin/env python3
"""
pipeline/swagit.py -- the Swagit archive adapter and the fourth locate route,
`episodes.method = 'swagit_transcript'` (plan/SOURCES-AUDIT.md, 2026-09-24).

WHY THIS EXISTS. 322 of 385 meetings were waiting on a YouTube caption request
at one an hour when the worker was 429'd a second time. Of the 16 waiting bodies
audited, exactly one can leave YouTube: **Petaluma**, whose PrimeGov portal
points its council meetings at Swagit, which serves a voice-to-text transcript
with a timestamp on EVERY WORD plus an HLS stream on
`archive-stream.granicus.com` -- the host `granicus.audio_slice` already cuts.
Petaluma is rank 2 by measured yield (8.00 episodes per captioned meeting), so
the 15 meetings this takes off the queue are disproportionately productive ones.

A Swagit archive is two pages, and the listing is NOT one of them:

  listing   the body's own PrimeGov API,
            `https://<host>/api/v2/PublicPortal/ListArchivedMeetings?year=<YYYY>`
            -- one JSON object per meeting with `title`, `date`, `videoUrl` and
            `swagitId` ('videos/400885'). This is the only trustworthy source of
            a clip id (see TRAP 2).
  player    `https://<slug>.new.swagit.com/videos/<clip_id>` carries four things:
              * the HLS master on archive-stream.granicus.com (granicus.M3U8),
              * the customer slug in that URL (`swagitVideo/petalumaca/...`),
                which is the ONLY proof the clip belongs to this body,
              * a timed agenda index -- `<span class="playerControl"
                data-ts="40" data-end-ts="1297" data-title="CLOSED SESSION">`,
                i.e. start AND end per item, which Granicus index points do not
                give (agenda_index.item_windows has to infer the end),
              * the transcript, in `<div id="transcript-fragments">`: one
                `<a data-ts="2.035" onclick="seek(2.035)...">GOOD </a>` per
                WORD. 58,415 of them on the 7.27 h meeting of 2026-09-14.

TWO TRAPS THE AUDIT ALREADY PAID FOR. Do not rediscover them.

  1. The host pattern is `<slug>.new.swagit.com`. The bare `<slug>.swagit.com`
     404s or 500s even for a real customer: `petaluma.swagit.com` -> 404,
     `petalumaca.swagit.com` -> 500, `petalumaca.new.swagit.com` -> 200.
  2. `/views/1` SERVES CARMEL, INDIANA ON EVERY SWAGIT HOST, including the real
     customer's. `livermoreca`, `concordca` and `petalumaca` all answer 200,
     all title themselves "<City>, CA Video Archive" from the hostname, and all
     serve the same 1,530 clips, whose player pages resolve to
     `swagitVideo/carmelin/...`. So a clip id must come from the BODY'S OWN
     portal (PrimeGov `swagitId`) and the player page's customer slug must then
     match the host (`parse_player()['owner']`). This module never reads
     `/views/1` at all, and `fetch_player` REFUSES a page whose owner is not the
     slug it asked for.

WHAT THE INSTRUMENT IS. A separate locate route, never pooled with the others
(plan/T2-PLAN.md forbids pooling routes; `analysis/facets.py` enforces it):

  * The transcript is voice-to-text over the whole meeting, so unlike
    `agenda_index` it can find twenty minutes of housing comment inside an item
    called "Consent Calendar". It is the same KIND of instrument as a caption
    track, read by the same stage 1 (`segment.candidate_windows`) and the same
    stage 2 codebook (`classify.py`) -- but it is a different transcriber with
    its own error rate, so its episodes carry `method='swagit_transcript'` and
    its recall is reported on its own. `scripts/swagit_recall.py` measures that
    recall against the English keyword route on the meetings that have both.
  * The page's own agenda index fills `episodes.agenda_item`, exactly as the
    Granicus index points do. It does not locate anything here.

AUDIO NEEDS NO NEW CODE. The row is registered with `platform='swagit'` and the
m3u8 in `raw_meta`, which is exactly what `episodes.cut_granicus` reads -- so
the cutter takes the non-YouTube branch and slices the HLS master with
`granicus.audio_slice`.

EVERYTHING FETCHED IS UNTRUSTED DATA (AGENTS.md rule 5). Agenda titles and
transcript words are parsed with regexes, stored, counted and never executed;
`parse_player` returns `instruction_like` and `register` logs it. No URL on a
YouTube host is ever requested -- `fetch()` raises if one is passed.

    python3 pipeline/swagit.py listing petaluma                 # PrimeGov council meetings + clip ids
    python3 pipeline/swagit.py player petalumaca 400885         # owner / m3u8 / index / transcript size
    python3 pipeline/swagit.py register --body petaluma --plan  # what would be written, no DB change
    python3 pipeline/swagit.py register --body petaluma
"""
import argparse, html, json, re, sys, urllib.parse, urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
import granicus  # noqa: E402

ROOT = state.ROOT
METHOD = "swagit_transcript"      # episodes.method for this route
CAPTIONS = "swagit"              # videos.captions; classify.METHOD_BY_CAPTIONS maps it to METHOD
PLATFORM = "swagit"              # videos.platform; episodes.py takes the non-YouTube (HLS slice) branch
BANNED_HOSTS = ("youtube.com", "youtu.be", "googlevideo.com", "ytimg.com")
PRIMEGOV_API = "https://{host}/api/v2/PublicPortal/ListArchivedMeetings?year={year}"
PLAYER_URL = "https://{slug}.new.swagit.com/videos/{clip_id}"

# Bodies with a proven Swagit route: body -> (PrimeGov host, Swagit owner slug).
# `owner` is the slug the player page must name in its stream URL; anything else
# is trap 2 and fetch_player refuses it. Only Petaluma is proven
# (plan/SOURCES-AUDIT.md): Livermore and Concord answer on Swagit but have no
# portal that can hand us a clip id of their own, so they are unproven, not
# listed, and must not be guessed at.
BODIES = {
    "petaluma": {"primegov_host": "cityofpetaluma.primegov.com", "owner": "petalumaca"},
}
COUNCIL_RE = re.compile(r"\bcity council\b|\bcouncil (?:meeting|regular)\b", re.I)

TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
OWNER = re.compile(r"swagitVideo/([A-Za-z0-9_-]+)/")
# The timed agenda index. The <span> and the <a> carry the same four attributes,
# so key on the span and the item appears once.
INDEX_SPAN = re.compile(r'<span[^>]*class="playerControl"[^>]*\bdata-ts="([\d.]+)"[^>]*\bdata-end-ts="([\d.]+)"'
                        r'[^>]*\bdata-title="([^"]*)"', re.I)
# One transcript word. `onclick="seek(` is what separates a transcript anchor
# from an agenda-index one (which carries data-title instead).
WORD_A = re.compile(r'<a\b[^>]*\bdata-ts="([\d.]+)"[^>]*onclick="seek\([^>]*>(.*?)</a>', re.S)
# The voice-to-text disclaimer Swagit puts above the transcript; recorded so the
# provenance of this track is visible in the player JSON, never used as text.
DISCLAIMER = re.compile(r"This transcript was created by voice-to-text technology[^<]*", re.I)
INSTRUCTION_LIKE = re.compile(
    r"\b(?:ignore (?:all |any )?(?:previous|prior|above) instructions?|you are (?:an? )?(?:ai|assistant|language model)"
    r"|system prompt|disregard (?:the )?(?:above|previous)|as an ai\b)", re.I)

# --- turning per-word timings into caption cues ---------------------------------
# Stage 1 buckets cue text by start // window, so cue granularity barely changes
# a density; it changes two other things. episodes.clip_bounds SNAPS each clip
# edge to the nearest cue edge within 3 s, and a human reading the track wants
# phrases. So words are grouped into a cue at a pause, at a length, or at a word
# count, whichever comes first.
# A Swagit clip only replaces a YouTube meeting if it is plausibly a record of
# the SAME meeting. Two of Petaluma's 15 fail that: the special meetings of
# 2025-09-29 and 2025-11-24 are published on Swagit as ~2-minute clips (CALL TO
# ORDER -> CLOSED SESSION -> REPORT OUT OF CLOSED SESSION, 364 and 274 words),
# against YouTube videos of 42 min and 1.78 h for the same dates. Those clips are
# not truncated -- they are a complete record of the open portion -- but they are
# not the same object, and registering one would retire a 1.78 h meeting behind a
# 2-minute transcript. They stay on the YouTube queue and are reported; whether a
# closed-session special meeting belongs in the sample at all is a DECISIONS.md
# question for Puck, not this adapter's.
MIN_DURATION_SHARE = 0.5

CUE_GAP_S = 2.0            # a gap this long between two word starts ends the cue
CUE_MAX_S = 8.0            # ... and so does this much elapsed
CUE_MAX_WORDS = 14
WORD_TAIL_S = 1.5          # assumed length of the last word of a cue (and of the track)


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def subs_dir():
    import os
    return Path(os.path.expanduser(state.setting("SUBS_DIR", str(ROOT / "output" / "subs"))))


def video_id_for(slug, clip_id):
    """'sw-petalumaca-400885'. Distinct from the Granicus 'gr-<site>-<clip>' and
    from a YouTube id, so the two routes' rows can never collide."""
    return f"sw-{slug}-{clip_id}"


def player_url(slug, clip_id):
    return PLAYER_URL.format(slug=slug, clip_id=clip_id)


def strip_tags(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def fetch(url, timeout=120):
    """GET through granicus.fetch, so this module shares the >= 5 s pacing with
    the Granicus adapter (same company, and the stream host is literally
    archive-stream.granicus.com). Refuses a YouTube host outright."""
    host = urllib.parse.urlparse(url).netloc.lower()
    for banned in BANNED_HOSTS:
        if banned in host:
            raise ValueError(f"refusing to request {banned}: this module never touches YouTube")
    return granicus.fetch(url, timeout=timeout)


# --- the listing: the body's own PrimeGov API ------------------------------------

def parse_meeting_date(s):
    """'Sep 14, 2026' -> '2026-09-14'; None when the portal writes something else."""
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime((s or "").strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_primegov(meetings, council_re=COUNCIL_RE):
    """The council meetings of one ListArchivedMeetings payload that carry a
    Swagit clip: [{clip_id, date, title, video_url, host}]. A meeting with no
    `swagitId` is skipped -- there is no clip to read, and inventing one from
    the listing is trap 2."""
    out = []
    for m in meetings if isinstance(meetings, list) else []:
        title = " ".join(str(m.get("title") or "").split())
        if not council_re.search(title):
            continue
        sid = str(m.get("swagitId") or "").rsplit("/", 1)[-1].strip()
        if not sid.isdigit():
            continue
        url = m.get("videoUrl") or ""
        out.append({"clip_id": sid, "date": parse_meeting_date(m.get("date")), "title": title,
                    "video_url": url, "host": urllib.parse.urlparse(url if "//" in url else f"//{url}").netloc.lower()})
    out.sort(key=lambda r: (r["date"] or "", r["clip_id"]))
    return out


def fetch_listing(body, years):
    """[{clip_id, date, title, ...}] over several years, de-duplicated on clip_id.
    One request per year, paced by granicus.GAP_S."""
    cfg = BODIES.get(body)
    if not cfg:
        raise ValueError(f"no proven Swagit route for {body!r}; known: {', '.join(BODIES)} "
                         f"(plan/SOURCES-AUDIT.md -- livermore/concord are UNPROVEN, not absent)")
    rows, seen = [], set()
    for year in years:
        text, _ = fetch(PRIMEGOV_API.format(host=cfg["primegov_host"], year=int(year)))
        try:
            payload = json.loads(text)
        except ValueError:
            log(f"  {body} {year}: PrimeGov did not answer JSON; skipped")
            continue
        for r in parse_primegov(payload):
            if r["clip_id"] not in seen:
                seen.add(r["clip_id"])
                rows.append(r)
    rows.sort(key=lambda r: (r["date"] or "", r["clip_id"]))
    return rows


# --- the player page -------------------------------------------------------------

def parse_words(text):
    """[(start_s, word)] from the transcript anchors, in time order. Words are
    unescaped and stripped of markup; the `[SECTION]` headings between them are
    NOT words and are not collected (the agenda index carries those, with times)."""
    out = []
    for ts, body in WORD_A.findall(text or ""):
        w = strip_tags(body)
        if w:
            out.append((float(ts), w))
    out.sort(key=lambda p: p[0])
    return out


def words_to_cues(words, gap_s=CUE_GAP_S, max_s=CUE_MAX_S, max_words=CUE_MAX_WORDS, tail_s=WORD_TAIL_S):
    """[{start, end, text}] -- the segment.read_segments contract, so stage 1,
    the cutter's cue-edge snapping and caption_audit all read this track exactly
    as they read a .vtt. A cue ends at a pause of `gap_s`, after `max_s`, or
    after `max_words`. Its end is the next word's start (so cues tile the
    meeting) except at a break, where the last word gets `tail_s`."""
    cues = []
    cur = None
    for i, (t, w) in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else None
        if cur is None:
            cur = {"start": t, "end": t + tail_s, "words": [w], "last": t}
        else:
            cur["words"].append(w)
            cur["last"] = t
        breaks = (nxt is None or nxt - t > gap_s or t + tail_s - cur["start"] >= max_s
                  or len(cur["words"]) >= max_words)
        if breaks:
            cur["end"] = min(nxt, cur["last"] + tail_s) if nxt is not None else cur["last"] + tail_s
            cues.append({"start": round(cur["start"], 3), "end": round(max(cur["end"], cur["start"] + 0.1), 3),
                         "text": " ".join(cur["words"])})
            cur = None
    return cues


def parse_player(text, slug=None):
    """{owner, title, m3u8, index_points, words, cues, n_words, speech_s,
    disclaimer, instruction_like} from a Swagit player page.

    `index_points` is [(start_s, end_s, title)] -- Swagit gives the END of an
    agenda item, which Granicus does not. `granicus_index_points` converts it to
    the (time, title) pairs classify.py and agenda_index.py expect.
    `owner` is the customer slug in the stream URL: the ownership proof."""
    m3u8 = sorted(set(granicus.M3U8.findall(text or "")))
    owner = OWNER.search(text or "")
    title = strip_tags(TITLE.search(text or "").group(1)) if TITLE.search(text or "") else None
    # The page prints the agenda index TWICE (a desktop list and a mobile one),
    # so every item matches twice; de-duplicate on (start, end, title) or the
    # agenda_item lookup weights every item double.
    pts, seen = [], set()
    for a, b, t in INDEX_SPAN.findall(text or ""):
        t = strip_tags(t)
        key = (float(a), float(b), t)
        if t and key not in seen:
            seen.add(key)
            pts.append(key)
    pts.sort()
    words = parse_words(text)
    cues = words_to_cues(words)
    dis = DISCLAIMER.search(text or "")
    # Instruction-like text is looked for in what we would actually READ: the
    # agenda titles and the transcript. Never in the page's scripts or markup.
    payload = " ".join([t for _, _, t in pts] + [w for _, w in words[:20000]])
    return {"slug": slug, "owner": owner.group(1) if owner else None, "title": title,
            "m3u8": m3u8[0] if m3u8 else None, "index_points": pts, "words": words, "cues": cues,
            "n_words": len(words), "speech_s": round(words[-1][0] + WORD_TAIL_S, 1) if words else 0.0,
            "disclaimer": (dis.group(0).strip() if dis else None),
            "instruction_like": sorted(set(m.group(0)[:120] for m in INSTRUCTION_LIKE.finditer(payload)))}


def granicus_index_points(index_points):
    """[(time_s, title)] -- the shape granicus.agenda_item_for() takes, so the
    agenda item on an episode row is filled by the same code for both routes."""
    return [(float(a), t) for a, _, t in index_points or []]


def player_json(player, clip_id, date=None, listing_title=None):
    """What lands in output/subs/<vid>.player.json. The word list is NOT stored
    here -- the cues are the track, and they are on disk as the .vtt."""
    return {"source": "swagit", "slug": player.get("slug"), "clip_id": clip_id, "owner": player.get("owner"),
            "player_url": player_url(player.get("slug"), clip_id), "m3u8": player.get("m3u8"),
            "page_title": player.get("title"), "listing_title": listing_title, "meeting_date": date,
            "index_points": granicus_index_points(player.get("index_points")),
            "index_points_with_end": [[a, b, t] for a, b, t in player.get("index_points") or []],
            "n_index_points": len(player.get("index_points") or []),
            "transcript_words": player.get("n_words"), "transcript_cues": len(player.get("cues") or []),
            "speech_s": player.get("speech_s"), "transcript_disclaimer": player.get("disclaimer"),
            "instruction_like": player.get("instruction_like")}


def fetch_player(slug, clip_id, dest_html=None):
    """The player page, parsed. RAISES when the customer slug on the page is not
    `slug`: that is trap 2 (Carmel, Indiana under a local city's title), and a
    route built on it would silently ingest another state's meetings."""
    text, info = fetch(player_url(slug, clip_id))
    if dest_html:
        Path(dest_html).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_html).write_text(text, encoding="utf-8")
    p = parse_player(text, slug)
    if (p["owner"] or "").lower() != slug.lower():
        raise ValueError(f"clip {clip_id} on {slug}.new.swagit.com belongs to {p['owner']!r}, not {slug!r} "
                         f"(SOURCES-AUDIT trap 2: /views/1 serves Carmel, Indiana on every Swagit host); refusing it")
    return p, info


# --- the track on disk -----------------------------------------------------------

def vtt_time(s):
    s = max(0.0, float(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def write_vtt(cues, dest):
    """A WebVTT of the Swagit transcript. A .vtt and not a JSON blob because
    every reader downstream already takes one: segment.read_segments, the
    cutter's cue-edge snapping, scripts/caption_audit.py and
    analysis/qa_transcripts.py's reference window. Nothing is deleted: the file
    is written whole and replaces an older copy of the same track only."""
    # NO cue identifier lines. WebVTT allows a number before the timing line,
    # but segment.vtt_to_segments treats any non-timing line after a cue as
    # TEXT, so a numbered track reads back as twice the cues, every other one
    # being the digit '2', '3', ... (measured: 4,387 cues -> 8,773 segments,
    # +7.5 % words). YouTube's tracks carry no identifiers either.
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for c in cues:
        lines += [f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}", c["text"], ""]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


# --- registration ----------------------------------------------------------------

def existing_meetings(con, body):
    """{meeting_date: row} for the body's meetings already in the DB, whatever
    platform. The Swagit route only registers a clip that matches one of these:
    the frame is plan/frame.csv and the discovery window, and this adapter must
    not quietly widen the sample (AGENTS.md rule 3). Extra clips are REPORTED."""
    rows = con.execute("SELECT * FROM videos WHERE kind = 'meeting' AND speaker = ? ORDER BY meeting_date", (body,)).fetchall()
    return {r["meeting_date"]: r for r in rows if r["meeting_date"]}


def register_meeting(con, body, clip, player, out_dir, dry_run=False):
    """One videos row for one Swagit clip, straight to 'captioned': the
    transcript came down with the player page, so there is no separate caption
    request to make. Returns (video_id, message).

    platform='swagit' + raw_meta['m3u8'] is what makes the audio reachable:
    episodes.py sends any non-YouTube meeting to cut_granicus, which slices that
    m3u8 with granicus.audio_slice. captions='swagit' is what makes the ROUTE
    visible: classify.METHOD_BY_CAPTIONS maps it to method='swagit_transcript',
    so these episodes are never pooled with the classifier's or the agenda
    index's (plan/T2-PLAN.md)."""
    slug = BODIES[body]["owner"]
    vid = video_id_for(slug, clip["clip_id"])
    vtt = Path(out_dir) / f"{vid}.en.vtt"
    pj = Path(out_dir) / f"{vid}.player.json"
    meta = player_json(player, clip["clip_id"], clip.get("date"), clip.get("title"))
    dur = float(player.get("index_points")[-1][1]) if player.get("index_points") else player.get("speech_s") or 0.0
    dur = max(dur, player.get("speech_s") or 0.0)
    if dry_run:
        return vid, (f"DRY RUN {vid} {clip.get('date')}: {player['n_words']} words -> {len(player['cues'])} cues, "
                     f"{meta['n_index_points']} agenda items, {dur / 3600:.2f} h, m3u8 {'yes' if meta['m3u8'] else 'NO'}")
    write_vtt(player["cues"], vtt)
    pj.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    fields = {"speaker": body, "title": clip.get("title") or meta.get("page_title"),
              # channel_url is NOT NULL and means "the listing this row came from"
              # (a Granicus meeting carries its ViewPublisher.php URL); for Swagit
              # the listing is the body's own PrimeGov portal.
              "channel_url": f"https://{BODIES[body]['primegov_host']}/public/portal",
              "duration_s": dur, "raw_meta": json.dumps(meta, ensure_ascii=False),
              "kind": "meeting", "platform": PLATFORM, "source_url": meta["player_url"],
              "meeting_id": f"{body}:{clip.get('date')}", "meeting_date": clip.get("date"),
              "captions": CAPTIONS, "captions_path": str(vtt), "date_exact": 1,
              "upload_date": (clip.get("date") or "").replace("-", "") or None,
              # in_frame stays 0 on a MEETING row, like every Granicus meeting:
              # the launchd workers select in-frame rows, and only the episode
              # clips episodes.register_clip writes are flipped to 1.
              "month": (clip.get("date") or "")[:7] or None, "in_frame": 0}
    cols = ", ".join(fields)
    con.execute(f"INSERT INTO videos (video_id, crawled_at, status, attempts, updated_at, {cols}) "
                f"VALUES (?,?,?,0,?,{','.join('?' * len(fields))}) "
                "ON CONFLICT(video_id) DO UPDATE SET "
                + ", ".join(f"{k} = excluded.{k}" for k in fields)
                + ", updated_at = excluded.updated_at "
                # Never walk a row BACK: a meeting already classified or cut keeps
                # its status and its episodes; only the track and metadata refresh.
                "WHERE videos.status IN ('discovered', 'captioned')",
                (vid, state.now_iso(), "captioned", state.now_iso(), *fields.values()))
    row = con.execute("SELECT status FROM videos WHERE video_id = ?", (vid,)).fetchone()
    # One events row, like every other status write: the dashboard's throughput
    # and stall detection read `events`, not videos.updated_at (state.py 17).
    con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                (vid, row["status"], row["status"], state.now_iso(),
                 f"swagit: {player['n_words']} words, {len(player['cues'])} cues, "
                 f"{meta['n_index_points']} agenda items"[:300]))
    return vid, (f"{vid} ({clip.get('date')}): {player['n_words']} words -> {len(player['cues'])} cues, "
                 f"{meta['n_index_points']} agenda items, {dur / 3600:.2f} h, status {row['status']}")


def already_captioned(row):
    """Has this meeting's YouTube caption request already been spent?"""
    return (row["captions"] or "none") != "none"


def register(con, body, years, out_dir, dry_run=False, limit=0, dates=None, refetch=False,
             include_captioned=False, min_share=MIN_DURATION_SHARE):
    cfg = BODIES[body]
    slug = cfg["owner"]
    cache = ROOT / "output" / "t2" / "swagit" / body
    clips = fetch_listing(body, years)
    known = existing_meetings(con, body)
    matched = [c for c in clips if c["date"] in known and known[c["date"]]["platform"] != PLATFORM]
    extra = [c for c in clips if c["date"] not in known]
    # A meeting whose YouTube caption track is ALREADY on disk is not registered
    # by default. That request has been spent and the classifier has read the
    # track or is about to; a second transcript of the same meeting would put two
    # instruments on it -- allowed (the ids do not collide) but it buys a second
    # classifier pass and it is not what this route is for. Those meetings are
    # the recall sample instead (scripts/swagit_recall.py).
    captioned = [c for c in matched if already_captioned(known[c["date"]])]
    if not include_captioned:
        matched = [c for c in matched if not already_captioned(known[c["date"]])]
    if dates:
        matched = [c for c in matched if c["date"] in dates]
    already = [c for c in clips if c["date"] in known and known[c["date"]]["platform"] == PLATFORM]
    log(f"{body}: {len(clips)} council clip(s) on {slug}.new.swagit.com via PrimeGov; "
        f"{len(matched)} to register, {len(captioned)} with a YouTube track already on disk "
        f"({'skipped; --include-captioned overrides' if not include_captioned else 'INCLUDED'}), "
        f"{len(already)} already registered here, "
        f"{len(extra)} clip(s) have no DB meeting (NOT registered -- the frame is plan/frame.csv)")
    if extra:
        log(f"  clips with no meeting row (report to Puck, do not ingest): "
            + ", ".join(f"{c['date']}/{c['clip_id']}" for c in extra[:40]))
    stats = {"ok": 0, "failed": 0, "requests": len(years), "flagged": 0, "short": 0}
    for c in matched[:limit or None]:
        vid = video_id_for(slug, c["clip_id"])
        cached = cache / f"player_{c['clip_id']}.html"
        try:
            if cached.exists() and not refetch:
                player = parse_player(cached.read_text(encoding="utf-8", errors="replace"), slug)
                if (player["owner"] or "").lower() != slug.lower():
                    raise ValueError(f"cached page for {c['clip_id']} is owned by {player['owner']!r}, not {slug!r}")
            else:
                player, _ = fetch_player(slug, c["clip_id"], dest_html=cached)
                stats["requests"] += 1
        except Exception as e:                       # noqa: BLE001 -- one bad clip is not the run's failure
            stats["failed"] += 1
            log(f"FAILED {vid} ({c['date']}): {type(e).__name__}: {str(e)[:200]}")
            continue
        if not player["n_words"]:
            stats["failed"] += 1
            log(f"FAILED {vid} ({c['date']}): player page carries no transcript anchors; not registered")
            continue
        if player["instruction_like"]:
            stats["flagged"] += 1
            log(f"  !! {vid}: instruction-like text in the portal transcript/agenda, IGNORED as data "
                f"(tell Puck): {player['instruction_like'][:2]}")
        known_s = float(known[c["date"]]["duration_s"] or 0)
        clip_s = max(float(player["index_points"][-1][1]) if player["index_points"] else 0.0, player["speech_s"] or 0.0)
        if known_s and clip_s < min_share * known_s:
            stats["short"] += 1
            log(f"SHORT {vid} ({c['date']}): the Swagit clip is {clip_s / 60:.1f} min against a known meeting of "
                f"{known_s / 60:.1f} min ({clip_s / known_s:.0%}); NOT registered, the meeting stays on the YouTube "
                f"queue. Agenda: {' / '.join(t for _, _, t in player['index_points'][:4]) or '(none)'}")
            continue
        _, msg = register_meeting(con, body, c, player, out_dir, dry_run)
        stats["ok"] += 1
        log(("  " if dry_run else "registered ") + msg)
    log(f"done: {stats['ok']} meeting(s) {'planned' if dry_run else 'registered'}, {stats['failed']} failed, "
        f"{stats['short']} too short to replace their meeting (left on the YouTube queue), "
        f"{stats['requests']} network request(s) (0 to YouTube), {stats['flagged']} page(s) with instruction-like text"
        + (" -- nothing written (plan only)" if dry_run else ""))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("listing", help="the body's PrimeGov council meetings and their Swagit clip ids (network)")
    p.add_argument("body"); p.add_argument("--years", default="2025,2026")
    p = sub.add_parser("player", help="parse one player page (network unless it is cached)")
    p.add_argument("slug"); p.add_argument("clip_id")
    p = sub.add_parser("register", help="register the matching clips as 'captioned' meetings")
    p.add_argument("--body", default="petaluma")
    p.add_argument("--years", default="2025,2026")
    p.add_argument("--dates", default=None, help="only these meeting dates (YYYY-MM-DD, comma separated)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--refetch", action="store_true", help="ask for the player page again even if it is cached")
    p.add_argument("--min-duration-share", type=float, default=MIN_DURATION_SHARE,
                   help=f"a Swagit clip shorter than this share of the known meeting duration is NOT registered "
                        f"(default {MIN_DURATION_SHARE}); see MIN_DURATION_SHARE")
    p.add_argument("--include-captioned", action="store_true",
                   help="also register meetings whose YouTube caption track is already on disk "
                        "(a SECOND instrument on those meetings; costs a second classifier pass)")
    p.add_argument("--plan", action="store_true", help="print what would be written, change nothing")
    p.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "listing":
        rows = fetch_listing(a.body, [int(y) for y in a.years.split(",") if y.strip()])
        print(f"{len(rows)} council clip(s)")
        for r in rows:
            print(f"  {r['date']}  {r['clip_id']:>8}  {r['host']:<30} {r['title'][:70]}")
        return 0
    if a.cmd == "player":
        cache = ROOT / "output" / "t2" / "swagit" / f"player_{a.clip_id}.html"
        if cache.exists():
            p = parse_player(cache.read_text(encoding="utf-8", errors="replace"), a.slug)
        else:
            p, _ = fetch_player(a.slug, a.clip_id)
        print(json.dumps({**player_json(p, a.clip_id), "index_points": granicus_index_points(p["index_points"])[:8],
                          "index_points_with_end": None, "first_cues": p["cues"][:3]}, indent=1, ensure_ascii=False))
        return 0
    con = state.connect(a.state)
    out_dir = subs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in a.dates.split(",")] if a.dates else None
    try:
        register(con, a.body, [int(y) for y in a.years.split(",") if y.strip()], out_dir, a.plan, a.max,
                 dates, a.refetch, a.include_captioned, a.min_duration_share)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
