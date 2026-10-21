#!/usr/bin/env python3
"""Build the whole-meeting audio manifest: one descriptive B2 name per meeting.

Written 2026-10-20 for Puck's "whole meeting uploads with descriptive titles".

Every meeting row (videos.kind='meeting') is matched against the audio that
holds the WHOLE meeting. Episode/clip cuts (<id>_e3.opus, _c2, _f1) are not
whole meetings and are never listed here.

Output: a list of manifest rows, each with
  dest      transit_bay/meetings/<Body>/<date> <Body> - <title>.<ext>
  local     the local file that holds the audio
  b2_raw    the raw backup key under transit_bay/output/... if backup.sh covers
            that tree, else None (those must be uploaded directly)

Import as a module (ops/meeting_copier.py does) or run it to print a summary.
"""
from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get("STATE_DB_PATH", Path.home() / ".local/share/ytpt/state.sqlite"))
PREFIX = "transit_bay"

# Audio extensions that can hold a whole meeting, best quality first. m4a is the
# original YouTube download; flac/opus are what fullasr fetched from Granicus or
# Swagit; wav under t2/asr is the 16 kHz mono ASR prep, the last resort.
EXT_RANK = {"m4a": 0, "mp4": 1, "flac": 2, "opus": 3, "mp3": 4, "wav": 5}

# Where whole-meeting audio lives, and whether backup.sh already copies that
# tree to B2 (so we can server-side-copy instead of re-uploading).
#   (directory, backed_up_by_backup_sh)
AUDIO_DIRS = [
    ("output/audio", True),
    ("output/fullasr/audio", True),
    ("output/t2/asr", True),
    ("output/opus", False),  # regenerable, deliberately not in backup.sh
]

# A clip, not a whole meeting: <id>_e3, <id>_c12, <id>_f1 (pipeline cut names).
CLIP_RE = re.compile(r"_(?:e|c|f)\d+$")

# Characters that break file names on macOS, B2 or a Windows download. ':' and
# '/' carry meaning in meeting titles ("9:00 AM", "09/29/25"), so they are
# transliterated rather than blanked; the rest become spaces.
TRANSLIT = {":": ".", "/": "-", "\\": "-", "|": "-", '"': "'", "*": "+"}
BAD_CHARS = re.compile(r'[?<>\x00-\x1f]')
# Anything that reads as an instruction rather than a title (AGENTS.md #5).
INJECTION_RE = re.compile(
    r"(?i)\b(ignore (all |the )?(previous|above|prior)|disregard (all|the|previous)|"
    r"system prompt|you are now|new instructions?|forget (everything|all)|"
    r"</?(system|instructions?|tool)>)"
)

MAX_TITLE = 150


def open_db():
    if not DB.exists():
        sys.exit(f"state db not found: {DB}")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def body_names(con):
    """slug -> display name. docs/data/bodies_geo.csv is the published map."""
    names = {}
    geo = REPO / "docs/data/bodies_geo.csv"
    if geo.exists():
        for r in csv.DictReader(geo.open(encoding="utf-8")):
            if r.get("body") and r.get("name"):
                names[r["body"]] = r["name"].strip()
    # Bodies with meetings but no geo row still need a name; build one from the
    # sources table rather than guessing (sfmta on 2026-10-20).
    kinds = {
        "council": "City Council",
        "county": "Board of Supervisors",
        "transit_board": "Board of Directors",
        "commission": "Commission",
    }
    for r in con.execute("SELECT DISTINCT body, body_kind FROM sources"):
        names.setdefault(r["body"], f"{r['body'].upper()} {kinds.get(r['body_kind'], 'Meeting')}")
    return names


def clean_component(s: str) -> str:
    """One path component: no separators, no controls, no leading/trailing junk."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(str.maketrans(TRANSLIT))
    s = BAD_CHARS.sub(" ", s)
    s = s.replace(" ", " ").replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def clean_title(title: str, vid: str) -> tuple[str, bool]:
    """Returns (safe title, looked_like_instructions)."""
    suspicious = bool(INJECTION_RE.search(title or ""))
    t = clean_component(title or "")
    if len(t) > MAX_TITLE:
        t = t[:MAX_TITLE].rstrip(" .,-")
    if not t:
        t = f"untitled {vid}"
    return t, suspicious


def meeting_date(row) -> str:
    """YYYY-MM-DD. meeting_date is authoritative; upload_date is the fallback."""
    md = (row["meeting_date"] or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", md):
        return md
    ud = (row["upload_date"] or "").strip()
    if re.fullmatch(r"\d{8}", ud):
        return f"{ud[:4]}-{ud[4:6]}-{ud[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}", md):
        return md + "-01"
    return "0000-00-00"


def disk_index():
    """id -> [(ext_rank, ext, relative path, backed_up)] for whole-meeting audio."""
    idx: dict[str, list] = {}
    for rel, backed in AUDIO_DIRS:
        d = REPO / rel
        if not d.is_dir():
            continue
        for p in d.iterdir():  # top level only: inverted_pre_fix/ is a superseded rerun
            if not p.is_file():
                continue
            ext = p.suffix.lstrip(".").lower()
            if ext not in EXT_RANK:
                continue
            stem = p.name[: -(len(ext) + 1)]
            # t2/asr names are <id>.wav; its siblings are <id>.asr.json etc.
            if "." in stem:
                continue
            if CLIP_RE.search(stem):
                continue
            idx.setdefault(stem, []).append((EXT_RANK[ext], ext, f"{rel}/{p.name}", backed))
    for v in idx.values():
        v.sort()
    return idx


def b2_raw_key(rel: str) -> tuple[str, str, str]:
    """Where backup.sh puts this local file in B2, and what format it is there.

    backup.sh (commit 42ee9ac, 2026-10-20) excludes *.wav from `b2 sync` and
    uploads each WAV re-encoded as 16 kHz mono 24 kbps Opus under the same name
    with the extension swapped -- or <name>.wav.opus when a real .opus already
    sits beside it. So a meeting whose only local audio is a WAV gets a
    descriptive name ending .opus, pointing at those Opus bytes, not the WAV.

    Returns (b2 key, destination extension, note for the index).
    """
    ext = rel.rsplit(".", 1)[1].lower()
    if ext != "wav":
        return f"{PREFIX}/{rel}", ext, ""
    sibling = REPO / (rel[: -len("wav")] + "opus")
    key = rel[: -len("wav")] + "opus" if not sibling.exists() else rel + ".opus"
    return f"{PREFIX}/{key}", "opus", "WAV re-encoded to 24 kbps Opus by backup.sh"


def build():
    con = open_db()
    names = body_names(con)
    idx = disk_index()
    rows = con.execute(
        "SELECT video_id, speaker, platform, title, meeting_date, upload_date, "
        "       duration_s, source_url, status "
        "FROM videos WHERE kind='meeting' ORDER BY speaker, meeting_date, video_id"
    ).fetchall()
    con.close()

    manifest, missing, suspicious = [], [], []
    used: dict[str, str] = {}
    for r in rows:
        vid = r["video_id"]
        cands = idx.get(vid)
        if not cands:
            missing.append(dict(r))
            continue
        _, _ext, local, backed = cands[0]
        raw_key, ext, note = b2_raw_key(local)
        body = names.get(r["speaker"], r["speaker"])
        title, flag = clean_title(r["title"], vid)
        if flag:
            suspicious.append((vid, (r["title"] or "")[:120]))
        date = meeting_date(r)
        bodydir = clean_component(body) or r["speaker"]
        base = f"{date} {bodydir} — {title}"
        name = f"{base}.{ext}"
        if f"{bodydir}/{name}" in used:  # same body, date and title: disambiguate
            name = f"{base} [{vid}].{ext}"
        dest = f"{PREFIX}/meetings/{bodydir}/{name}"
        used[f"{bodydir}/{name}"] = vid
        manifest.append(
            {
                "dest": dest,
                "local": local,
                "b2_raw": raw_key if backed else None,
                "b2_note": note,
                "video_id": vid,
                "body": r["speaker"],
                "body_name": body,
                "date": date,
                "title": title,
                "platform": r["platform"],
                "source_url": r["source_url"] or "",
                "duration_s": r["duration_s"] or "",
                "status": r["status"],
            }
        )
    return manifest, missing, suspicious


INDEX_COLS = ["dest", "video_id", "body", "body_name", "date", "title", "platform",
              "source_url", "duration_s", "local", "b2_note"]


def write_index(manifest, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        for m in sorted(manifest, key=lambda m: m["dest"]):
            w.writerow(m)
    os.replace(tmp, path)
    return path


def main():
    manifest, missing, suspicious = build()
    from collections import Counter

    have = Counter(m["platform"] for m in manifest)
    gone = Counter(m["platform"] for m in missing)
    print(f"meetings with whole audio: {len(manifest)}   without: {len(missing)}")
    for p in sorted(set(have) | set(gone)):
        print(f"  {p:<9} have {have[p]:>4}   missing {gone[p]:>4}")
    print(f"  direct-upload needed (not in backup.sh): "
          f"{sum(1 for m in manifest if not m['b2_raw'])}")
    if suspicious:
        print(f"  TITLES THAT READ AS INSTRUCTIONS ({len(suspicious)}): {suspicious[:5]}")
    out = write_index(manifest, REPO / "ops/meetings_INDEX.csv")
    print(f"index -> {out}")
    for m in manifest[:3]:
        print("  e.g.", m["dest"])


if __name__ == "__main__":
    main()
