#!/usr/bin/env python3
"""
qa_transcripts.py -- automated quality gate for transcript JSON.

WHY THIS EXISTS
---------------
Every transcript that reaches load_db.py becomes "data". A hallucinated passage,
a decoder loop, an hour of dropped speech or a broken speaker split looks exactly
like real speech to every downstream statistic, and a WER average hides it
because it is rare and concentrated (plan/README.md, glossary: WER). So each
batch is screened per video BEFORE loading, and anything suspicious is listed
with the timestamp a human needs to go and listen.

It already paid for itself: on the 20 exploratory transcripts it found a decoder
loop in AAiFho_u_GI at 21:39 -- "Hassan piker" x45 over 30 s, where the audio has
a ~12 s supercut of the name followed by other speech that Whisper overwrote.
Since 2026-09-20 a run of identical segments is a FAIL by the AUDIO it spans
(> DUP_FAIL_S), not by how many segments it has: the loader collapses short runs,
and a five-segment chant is not a reason to throw away a transcript.

Run after every transcription batch, before load_db.py:
    python3 analysis/qa_transcripts.py --src output/whisper --subs output/subs --csv qa.csv

Reads both formats:
  * contract -- {"video_id": ..., "segments": [{start, end, text, speaker?}, ...]} (PLAN.md 1)
  * legacy   -- a bare list of segments in <id>.<model>.json (hasan_compare.py)

Status per video: OK / WARN (a human should look) / FAIL (do not load until explained).
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = nothing to check.

Every threshold carries its reason. Change a number only with a new reason.
"""
import argparse, csv, gzip, html, json, os, re, sqlite3, sys
from pathlib import Path

# ---------------------------------------------------------------- thresholds --
# Whisper's own fallback heuristic: text whose gzip compression ratio exceeds 2.4
# is treated as a repetition loop and re-decoded. Only segments >= 60 chars are
# rated, because short strings compress badly and give meaningless ratios.
CR_MAX, CR_MIN_CHARS = 2.4, 60
# Identical consecutive segments: two in a row happens in real speech ("No. No."),
# three or more is worth a look (WARN). A COUNT is the wrong unit for FAIL,
# though: "We're not gonna make it." x5 over 4.8 s (ak1Jt_FXGXM, b11) is a
# crowd chant that WhisperX's word-level speaker split cut into five short
# segments, and it cost a 28-min transcript; a real decoder loop runs for
# minutes and the loader collapses short runs to one row anyway
# (load_db.collapse_loops). So FAIL only when one run of identical segments
# SPANS more than DUP_FAIL_S of audio (first start to last end). Span rather
# than summed segment length: alignment gives hallucinated text arbitrary,
# often tiny, per-segment timings, so the sum understates what the loop
# overwrote, and a false FAIL from three real "Yeah."s across a silent minute
# is a human listening for a minute, not a hallucination in the corpus.
DUP_WARN, DUP_FAIL_S = 3, 60.0
# Characters per second inside one segment. Fast political speech is ~20-23 cps
# (250 wpm x ~5.5 chars per word including the space). Above 35 cps the audio
# could not contain the text: hallucination or a broken timestamp. Only segments
# >= 3 s are rated: Whisper segment times are loose (often snapped to whole
# seconds), so on a 1 s segment one boundary error doubles the apparent rate. On
# the 20 exploratory transcripts every >35 cps hit under 3 s was a normal
# sentence squeezed into a 1.0 s timestamp.
CPS_MAX, CPS_MIN_DUR = 35.0, 3.0
# Share of the transcribed span covered by segments. Stream audio legitimately
# has music and dead air, so low coverage is a WARN; under 30% on anything over
# two minutes means most of the audio produced no text at all.
COV_WARN, COV_FAIL, COV_MIN_DUR = 0.60, 0.30, 120
# Meeting episodes (kind = 'episode', 2026-09-21, PROPOSAL-transit.md 3h): a
# public-comment stretch has dead air between speakers, so WARN at 50%
# coverage; FAIL stays at 30%. Speech inside the first / last EDGE_S of a clip
# means the cut landed mid-sentence (the pad + cue snap should prevent it).
# A single segment stretched over > STRETCH_S of audio at under STRETCH_WPS
# words per second is text laid over silence -- Whisper's hallucination shape
# on dead air, which the loop check does not catch when the words differ.
MEETING_COV_WARN, EDGE_S, STRETCH_S, STRETCH_WPS = 0.50, 5.0, 30.0, 0.5
# Words per minute over the covered speech. Conversational English is ~120-200;
# outside 60-300 the text and the timestamps disagree with each other.
WPM_LO, WPM_HI = 60, 300
# A silence this long between segments is music, an ad break, or dropped speech.
GAP_S = 30
# Host share of speech time below this: the "host = most talk time" heuristic is
# unreliable (PLAN.md 4.6).
HOST_CONF_MIN = 0.55
# Unassigned speech above this share: speaker-based filtering loses real text.
UNLABELLED_MAX = 0.05
# Caption comparison windows. A minute where one side has more than double the
# other's words (+20 words slack for quiet minutes) is not ordinary ASR noise
# (~10% WER); it is a hallucinated, dropped, or uncaptioned passage.
WIN_S, WIN_RATIO, WIN_SLACK = 60, 2.0, 20

# Phrases Whisper is known to emit over silence or music. Hosts also say some of
# them for real (sign-offs), so a hit is a WARN, and a hit that the captions also
# contain within +/-60 s is dropped.
HALLUCINATION_PHRASES = [
    "thank you for watching", "thanks for watching", "please subscribe",
    "like and subscribe", "subtitles by", "amara.org", "transcribed by",
    "transcription by", "captions by", "www.", ".com/", "copyright",
    "all rights reserved",
]
# Subtitle-style speaker tags emitted as words: "Male Speaker 1 Director Foley?",
# "Female Speaker Yes.", "Speaker 1 I can start". Whisper learned them from
# captioned training data; the 2026-09-21 validation found them in 4 of 46
# meeting episodes, concentrated in roll-call votes (analysis/
# validate_transcription.py), and never in the captions. WARN.
SPEAKER_TAG = re.compile(r"^\s*(?:(?:male|female)\s+speaker\b|speaker\s+\d+\b)", re.I)
EPISODE_ID = re.compile(r"_[ea]\d+$")               # '<meeting>_e<n>' (classifier) / '_a<n>' (agenda_index): the b13 clips carry no kind key
NONSPEECH_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪+")
CENSOR = re.compile(r"\[\s*__\s*\]")        # YouTube's profanity mask "[ __ ]"
WORD = re.compile(r"[a-z0-9']+")
INLINE_TS = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")


# ------------------------------------------------------------------- helpers --
def norm_words(text):
    """Lowercase word list for WER. Unescapes HTML (YouTube VTT writes &nbsp;),
    drops YouTube's "[ __ ]" profanity mask and bracketed non-speech tags. Digit
    vs number-word mismatches ("7" vs "seven") still count as errors: a known,
    roughly constant upward bias."""
    t = html.unescape(text).replace("’", "'").replace("\xa0", " ").lower()
    t = CENSOR.sub(" ", t)
    t = NONSPEECH_TAG.sub(" ", t)
    return WORD.findall(t)


def edit_distance(a, b):
    """Word-level Levenshtein distance, Myers/Hyyro bit-parallel algorithm.

    WHY NOT THE TEXTBOOK DP: the O(n*m) table is 100M cells for two 10k-word
    transcripts -- minutes in pure Python, hours for a 3-hour stream. The
    bit-vector form packs a whole DP column into one Python int, so each word of
    `b` costs a handful of big-int operations: O(n * m / 64) machine work.

    For pattern `a` (length m) and each symbol c of `b` (bit i = DP row i):
        Eq = bitmask of rows whose pattern word == c
        Xv = Eq | Mv
        Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq
        Ph = Mv | ~(Xh | Pv)          horizontal +1 deltas
        Mh = Pv & Xh                  horizontal -1 deltas
        score += top-row bit of Ph, or -= top-row bit of Mh
        Ph = (Ph << 1) | 1            |1: row 0 of the DP grows by 1 per column
        Pv = (Mh << 1) | ~(Xv | Ph)
        Mv = Ph & Xv
    Checked against the plain DP by --self-test and test_pipeline.py.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    m = len(a)
    full = (1 << m) - 1
    top = 1 << (m - 1)
    peq = {}
    for i, w in enumerate(a):
        peq[w] = peq.get(w, 0) | (1 << i)
    pv, mv, score = full, 0, m
    for w in b:
        eq = peq.get(w, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & full
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = (mh | ~(xv | ph)) & full
        mv = ph & xv
    return score


def dp_distance(a, b):
    """Reference O(n*m) edit distance, for testing edit_distance only."""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def wer(ref_text, hyp_text):
    """WER = edits / reference words. The REFERENCE is the human (or caption)
    text and goes first. Swapping them divides by the hypothesis length, which
    makes a hallucinated passage LOWER the error rate instead of raising it."""
    ref, hyp = norm_words(ref_text), norm_words(hyp_text)
    return edit_distance(ref, hyp) / len(ref) if ref else 0.0


def load_transcript(path):
    """Return (video_id, meta dict, segments) for either on-disk format."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return path.name.split(".")[0], {"format": "legacy"}, doc
    if isinstance(doc, dict) and isinstance(doc.get("segments"), list):
        meta = {k: v for k, v in doc.items() if k != "segments"}
        meta["format"] = "contract"
        return doc.get("video_id") or path.name.split(".")[0], meta, doc["segments"]
    raise ValueError("neither a segment list nor an object with 'segments'")


def parse_vtt(path):
    """YouTube VTT -> ([(start_s, text)], kind). Auto captions repeat each phrase
    across rolling cues; only lines carrying inline <hh:mm:ss.mmm> word timings
    hold new text, so only those are kept. Manual captions have no inline timings
    and are taken whole. kind is "auto" or "manual", read from the CONTENT --
    yt-dlp names both <id>.en.vtt, so the filename cannot tell you."""
    raw = path.read_text(encoding="utf-8")
    auto = bool(INLINE_TS.search(raw))
    segs = []
    for block in re.split(r"\n{2,}", raw):
        lines = [l for l in block.split("\n") if l.strip()]
        ts = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts is None or ts + 1 >= len(lines):
            continue
        body = lines[ts + 1:]
        if auto:
            body = [l for l in body if INLINE_TS.search(l)]
            if not body:
                continue
        text = re.sub(r"<[^>]+>", "", " ".join(body)).strip()
        try:
            h, mnt, s = lines[ts].split("-->")[0].strip().split()[0].replace(",", ".").split(":")
            start = int(h) * 3600 + int(mnt) * 60 + float(s)
        except ValueError:
            continue
        if text:
            segs.append((start, text))
    return segs, ("auto" if auto else "manual")


def pick_captions(subs_dir, vid):
    """Prefer a human (manual) English track; fall back to YouTube's ASR track."""
    found = []
    for p in sorted(Path(subs_dir).glob(f"{vid}.en*.vtt")):
        segs, kind = parse_vtt(p)
        if segs:
            found.append((0 if kind == "manual" else 1, p.name, p, segs, kind))
    if not found:
        return None, None, None
    _, _, p, segs, kind = min(found)
    return p, segs, kind


def default_state_db():
    """state.sqlite as pipeline/state.py resolves it (STATE_DB_PATH from .env),
    or None when the pipeline package is not importable (e.g. on a pod)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
        import state                                    # noqa: E402
        return state.db_path()
    except Exception:                                   # noqa: BLE001
        p = os.environ.get("STATE_DB_PATH")
        return Path(os.path.expanduser(p)) if p else None


def episode_window_start(meta, vid, state_db=None):
    """Offset of an episode clip inside its meeting, in seconds, and where it
    came from: ("window_start_s" | "episode_start_s" | "state_db", w0) or
    (None, None) when no source knows it.

    WHY NOT TRUST window_start_s ALONE (2026-10-16, plan/BART-QIL-ASSESSMENT.md):
    28 of the 29 batch-b13 clips were written before the ingest recorded the
    clip offset, so their JSON says window_start_s = 0 although they start
    hours into the meeting. Windowing the captions at 0 then scores the clip
    against the first `span` seconds of the meeting -- two unrelated stretches
    of speech -- which is where qIlAln3uR7U_e14's 115.7 % and _e15's 771.1 %
    came from (10.1 % and 9.2 % against the right window). A clip never starts
    at 0 (the cut is padded and cue-snapped inside the meeting), so 0 is read
    as "unknown" and the offset is taken from `episode_start_s` or from
    `episodes.start_s` in state.sqlite, the row the clip was cut from. With no
    source at all the caller must emit NO caption metric rather than a wrong
    one that looks like a finding."""
    for key in ("window_start_s", "episode_start_s"):
        try:
            w0 = float(meta.get(key) or 0.0)
        except (TypeError, ValueError):
            w0 = 0.0
        if w0 > 0:
            return key, w0
    if state_db and Path(state_db).exists():
        try:
            con = sqlite3.connect(f"file:{Path(state_db)}?mode=ro", uri=True)
            try:
                row = con.execute("SELECT start_s FROM episodes WHERE episode_id = ?", (vid,)).fetchone()
            finally:
                con.close()
        except sqlite3.Error:
            row = None
        if row and row[0] is not None:
            # The row the clip was cut from is authoritative even at 0: a clip
            # padded off the start of the meeting has start_s = -0.02
            # (9FM5406YagE_e1), and that is a real offset, not an unknown one.
            return "state_db", max(0.0, float(row[0]))
    return None, None


# ---------------------------------------------------------------------- core --
def check_video(path, subs_dir=None, state_db=None):
    """state_db: state.sqlite, consulted only for an episode whose JSON lacks
    its window offset (see episode_window_start). None = no DB lookup."""
    vid, meta, segs = load_transcript(path)
    r = {"video_id": vid, "file": path.name, "format": meta["format"],
         "model": meta.get("model", ""), "fail": [], "warn": [], "info": []}

    # 1. contract and timestamps --------------------------------------------
    bad, clean = 0, []
    for s in segs:
        try:
            st, en, tx = float(s["start"]), float(s["end"]), s["text"]
            if en < st or st < 0 or not isinstance(tx, str):
                raise ValueError
            clean.append((st, en, tx.strip(), s.get("speaker")))
        except (KeyError, TypeError, ValueError):
            bad += 1
    nonmono = sum(1 for p, q in zip(clean, clean[1:]) if q[0] + 0.01 < p[0])
    r["n_segments"], r["bad_segments"], r["non_monotonic"] = len(segs), bad, nonmono
    if not clean:
        r["fail"].append("no usable segments")
        return r
    if bad:
        r["fail"].append(f"{bad} malformed segments (missing keys, end<start, negative)")
    if nonmono:
        r["warn"].append(f"{nonmono} segments start before the previous one")

    span = float(meta.get("transcribed_s") or meta.get("duration_s") or clean[-1][1])
    ivs = sorted((st, en) for st, en, _, _ in clean)
    covered, cur_s, cur_e = 0.0, ivs[0][0], ivs[0][1]
    for st, en in ivs[1:]:
        if st > cur_e:
            covered += cur_e - cur_s
            cur_s, cur_e = st, en
        else:
            cur_e = max(cur_e, en)
    covered += cur_e - cur_s
    words = sum(len(tx.split()) for _, _, tx, _ in clean)
    r["span_min"] = round(span / 60, 1)
    r["coverage"] = round(min(covered / span, 1.0), 3) if span else None
    r["wpm"] = round(words / (covered / 60), 1) if covered else None
    gaps = [(q[0] - p[1], p[1]) for p, q in zip(clean, clean[1:]) if q[0] - p[1] > GAP_S]
    r["gaps_over_30s"] = len(gaps)
    r["longest_gap_s"] = round(max((g for g, _ in gaps), default=0), 1)
    is_episode = meta.get("kind") == "episode" or bool(meta.get("episode_id")) or bool(EPISODE_ID.search(vid))
    r["kind"] = "episode" if is_episode else meta.get("kind", "show")
    cov_warn = MEETING_COV_WARN if is_episode else COV_WARN
    if r["coverage"] is not None and span >= COV_MIN_DUR:
        if r["coverage"] < COV_FAIL:
            r["fail"].append(f"coverage {r['coverage']:.0%} of {r['span_min']} min")
        elif r["coverage"] < cov_warn:
            r["warn"].append(f"coverage {r['coverage']:.0%}")
    if is_episode:
        # Recorded as `info`, not WARN: on the 40 b13 clips speech sits within
        # 5 s of both edges on 38 (the 30-s pad lands inside continuous meeting
        # speech), so as a WARN it would flag every clip and gate nothing. The
        # numbers go to the CSV; a cut that matters is one that eats the core
        # window, which the pad is there to prevent.
        first, last = min(st for st, _, _, _ in clean), max(en for _, en, _, _ in clean)
        r["edge_speech_start_s"], r["edge_speech_end_s"] = round(first, 2), round(span - last, 2)
        edges = []
        if first < EDGE_S:
            edges.append(f"speech at {first:.1f} s from the start")
        if span - last < EDGE_S:
            edges.append(f"speech until {span - last:.1f} s before the end")
        if edges:
            r["info"].append("edge: " + ", ".join(edges) + " (pad in continuous speech; core window unaffected)")
        stretched = [(st, en - st, len(tx.split())) for st, en, tx, _ in clean
                     if en - st > STRETCH_S and len(tx.split()) / (en - st) < STRETCH_WPS]
        r["stretched_segs"] = len(stretched)
        if stretched:
            st, d, w = stretched[0]
            r["warn"].append(f"{len(stretched)} segment(s) of text over >{STRETCH_S:.0f} s of near-silence, "
                             f"first at {st / 60:.1f} min ({w} words over {d:.0f} s) -- listen: hallucination on dead air?")
    if r["wpm"] and not (WPM_LO <= r["wpm"] <= WPM_HI):
        r["warn"].append(f"speaking rate {r['wpm']} wpm outside {WPM_LO}-{WPM_HI}")
    if gaps:
        g, at = max(gaps)
        r["warn"].append(f"{len(gaps)} gaps >{GAP_S}s (longest {g:.0f}s at {at / 60:.1f} min)")

    # 2. decoder failure signatures ------------------------------------------
    # runs of identical consecutive segments: (count, span_s, start_s)
    runs, i = [], 0
    while i < len(clean):
        j = i
        while j + 1 < len(clean) and clean[i][2] and clean[i][2].lower() == clean[j + 1][2].lower():
            j += 1
        if j > i:
            runs.append((j - i + 1, clean[j][1] - clean[i][0], clean[i][0]))
        i = j + 1
    longest = max(runs, key=lambda x: x[0], default=(1, 0.0, 0.0))     # by count
    widest = max(runs, key=lambda x: x[1], default=(1, 0.0, 0.0))      # by audio span
    r["max_dup_run"], r["max_dup_span_s"] = longest[0], round(widest[1], 1)
    if widest[1] > DUP_FAIL_S:
        r["fail"].append(f"{widest[0]} identical segments in a row spanning {widest[1]:.0f} s "
                         f"at {widest[2] / 60:.1f} min (decoder loop)")
    elif longest[0] >= DUP_WARN:
        r["warn"].append(f"{longest[0]} identical segments in a row ({longest[1]:.1f} s) at {longest[2] / 60:.1f} min")

    loops = [st for st, _, tx, _ in clean
             if len(tx) >= CR_MIN_CHARS
             and len(tx.encode()) / len(gzip.compress(tx.encode())) > CR_MAX]
    r["high_compression_segs"] = len(loops)
    if loops:
        r["warn"].append(f"{len(loops)} repetitive segments (gzip ratio >{CR_MAX}), "
                         f"first at {loops[0] / 60:.1f} min -- listen: likely a decoder loop")

    fast = [st for st, en, tx, _ in clean
            if en - st >= CPS_MIN_DUR and len(tx) / (en - st) > CPS_MAX]
    r["too_fast_segs"] = len(fast)
    if fast:
        r["warn"].append(f"{len(fast)} segments >{CPS_MAX:.0f} chars/s, first at {fast[0] / 60:.1f} min")

    hits = [(st, p) for st, _, tx, _ in clean for p in HALLUCINATION_PHRASES if p in tx.lower()]
    tags = [st for st, _, tx, _ in clean if SPEAKER_TAG.match(tx)]
    r["speaker_tag_segs"] = len(tags)
    if tags:
        r["warn"].append(f"{len(tags)} segment(s) start with a subtitle-style speaker tag ('Male Speaker 1 ...'), "
                         f"first at {tags[0] / 60:.1f} min -- hallucinated words, not speech")

    # 3. diarization ---------------------------------------------------------
    spk_time = {}
    for st, en, _, sp in clean:
        spk_time[sp] = spk_time.get(sp, 0.0) + (en - st)
    labelled = {k: v for k, v in spk_time.items() if k is not None}
    r["diarized"] = bool(labelled)
    if labelled:
        tot = sum(spk_time.values())
        # Host = the enrolled label set when scripts/host_enroll.py identified
        # it by voice embedding (host_method "embedding"); otherwise the
        # talk-time heuristic, whose share is what the WARN below is about.
        enrolled = meta.get("host_method") == "embedding"
        hosts = set(meta.get("host_speakers") or []) & set(labelled) if enrolled else set()
        if not hosts:
            enrolled, hosts = False, {max(labelled, key=labelled.get)}
        r["n_speakers"] = len(labelled)
        r["host_method"] = "embedding" if enrolled else "talk_time"
        r["host_share"] = round(sum(labelled[h] for h in hosts) / tot, 3) if tot else None
        r["unlabelled_share"] = round(spk_time.get(None, 0.0) / tot, 3) if tot else None
        # A meeting episode has no host: the chair, staff and the public share the
        # floor by design, and pipeline/roles.py labels clusters dais / public
        # instead. The share is still recorded; the WARN is for shows only.
        if r["host_share"] is not None and r["host_share"] < HOST_CONF_MIN and not enrolled and not is_episode:
            sim = meta.get("host_similarity")
            why = (f"embedding did not decide (best cos {sim:.2f}, runner-up {meta.get('host_runner_up')})"
                   if isinstance(sim, (int, float)) else "host heuristic unreliable")
            r["warn"].append(f"host share {r['host_share']:.0%} < {HOST_CONF_MIN:.0%}: {why}")
        if r["unlabelled_share"] and r["unlabelled_share"] > UNLABELLED_MAX:
            r["warn"].append(f"{r['unlabelled_share']:.0%} of speech has no speaker label")

    # 4. captions (optional) --------------------------------------------------
    if subs_dir:
        cap_path, cap, kind = pick_captions(subs_dir, vid)
        if not cap and is_episode:
            # 2026-09-22 (output/validation/REPORT.md 6.3): an episode clip
            # '<meeting>_e<n>' has no caption file of its own; its meeting's
            # track, windowed by window_start_s below, is the reference -- so
            # every batch gets the per-minute dropped / extra check, not just
            # the validation script.
            meeting_vid = EPISODE_ID.sub("", vid)
            if meeting_vid != vid:
                cap_path, cap, kind = pick_captions(subs_dir, meeting_vid)
                if cap:
                    r["caption_source"] = "meeting"
        if cap:
            # The captions cover the whole video; the transcript covers only
            # its window (first TRANSCRIBE_CAP_MIN minutes). Compare like with
            # like, or a 45-min transcript of an 82-min video scores ~48% WER
            # from "deletions" that were never in scope (b7, 2026-09-18).
            if is_episode:
                # An episode's offset is NOT trusted from window_start_s alone:
                # 0 means "unknown", and the fallback is the episode metadata
                # or episodes.start_s in state.sqlite (2026-10-16, see
                # episode_window_start). No source -> no caption metric at all.
                src, w0 = episode_window_start(meta, vid, state_db)
                r["window_source"] = src or "none"
                if src is None:
                    # `info`, not WARN: nothing is known to be wrong with the
                    # transcript, only unmeasured. The CSV keeps wer_vs_captions
                    # EMPTY for this row and the note says why.
                    r["info"].append("caption check skipped: window_start_s is 0/missing and no episodes.start_s row "
                                     "in state.sqlite -- offset into the meeting unknown (pass --state-db)")
                    cap = None
                else:
                    r["window_start_s_used"] = round(w0, 2)
            else:
                w0 = float(meta.get("window_start_s") or 0.0)
            # ... and on the transcript's clock: caption times are absolute in
            # the meeting, segment times start at 0 in the clip, so the
            # per-minute windows below only line up once w0 is subtracted
            # (2026-09-22; every show had w0 = 0, so it never showed).
            if cap:
                cap = [(st - w0, t) for st, t in cap if w0 <= st < w0 + span]
        if cap:
            ref = [w for _, t in cap for w in norm_words(t)]
            hyp = [w for _, _, t, _ in clean for w in norm_words(t)]
            r["caption_kind"], r["caption_file"] = kind, cap_path.name
            if ref:
                r["wer_vs_captions"] = round(edit_distance(ref, hyp) / len(ref), 4)
                r["ref_words"], r["hyp_words"] = len(ref), len(hyp)
            cap_norm = [(st, " ".join(norm_words(t))) for st, t in cap]
            hits = [(st, p) for st, p in hits
                    if not any(abs(cst - st) <= 60 and " ".join(norm_words(p)) in ct
                               for cst, ct in cap_norm)]
            nwin = int(max(clean[-1][1], cap[-1][0]) // WIN_S) + 1
            rw, hw = [0] * nwin, [0] * nwin
            for st, t in cap:
                rw[int(st // WIN_S)] += len(norm_words(t))
            for st, _, t, _ in clean:
                hw[int(st // WIN_S)] += len(norm_words(t))
            ins = [i for i in range(nwin) if hw[i] > WIN_RATIO * rw[i] + WIN_SLACK]
            dele = [i for i in range(nwin) if rw[i] > WIN_RATIO * hw[i] + WIN_SLACK]
            r["extra_text_minutes"], r["missing_text_minutes"] = len(ins), len(dele)
            if ins:
                r["warn"].append(f"{len(ins)} minutes with >2x the captions' words (possible hallucination) at "
                                 + ", ".join(f"{i}m" for i in ins[:6]))
            if dele:
                r["warn"].append(f"{len(dele)} minutes with <1/2 the captions' words (possible dropped speech) at "
                                 + ", ".join(f"{i}m" for i in dele[:6]))

    r["phrase_hits"] = len(hits)
    if hits:
        r["warn"].append("stock phrases: " + "; ".join(f"'{p}' @{st / 60:.1f}m" for st, p in hits[:4]))
    return r


def self_test():
    import random
    rng = random.Random(1)
    for _ in range(300):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 40))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 40))]
        assert edit_distance(a, b) == dp_distance(a, b), (a, b)
    assert edit_distance("the cat sat".split(), "the cat sat".split()) == 0
    assert edit_distance("a b c".split(), "a x c y".split()) == 2
    ref = "the senator said the bill would pass " * 10
    assert wer(ref, ref + "thanks for watching " * 20) == 60 / 70   # insertions raise WER
    print("self-test: edit_distance == DP on 300 random pairs; WER direction ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="output/whisper", help="folder of transcript JSON")
    ap.add_argument("--glob", default="*.json", help="e.g. '*.medium.json' for legacy files")
    ap.add_argument("--subs", default=None, help="folder of YouTube .vtt captions (optional)")
    ap.add_argument("--state-db", default=None,
                    help="state.sqlite for an episode's offset when its JSON has window_start_s = 0 "
                         "(default: STATE_DB_PATH as pipeline/state.py resolves it; 'none' to disable)")
    ap.add_argument("--csv", default=None, help="write one row per video")
    ap.add_argument("--quiet", action="store_true", help="print only WARN/FAIL videos and the summary")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0

    files = sorted(Path(a.src).glob(a.glob))
    if not files:
        print(f"no files matching {a.glob} in {a.src}")
        return 2
    state_db = None if a.state_db == "none" else (Path(os.path.expanduser(a.state_db)) if a.state_db else default_state_db())
    if a.subs and state_db and not Path(state_db).exists():
        print(f"NOTE: {state_db} not found; episodes with window_start_s = 0 get no caption metric")
    rows = []
    for f in files:
        try:
            r = check_video(f, a.subs, state_db)
        except Exception as e:            # an unreadable file is a FAIL, not a crash
            r = {"video_id": f.name.split(".")[0], "file": f.name,
                 "fail": [f"unreadable: {e}"], "warn": []}
        r["status"] = "FAIL" if r["fail"] else "WARN" if r["warn"] else "OK"
        rows.append(r)

    width = max(len(r["video_id"]) for r in rows)
    for r in rows:
        if a.quiet and r["status"] == "OK":
            continue
        extra = ""
        if "wer_vs_captions" in r:
            extra = f"  WER vs {r['caption_kind']} captions {r['wer_vs_captions']:.1%}"
        print(f"{r['status']:<5} {r['video_id']:<{width}}  {r.get('span_min', '?'):>6} min"
              f"  cov {r.get('coverage') or 0:.0%}  {r.get('wpm') or 0:>5.0f} wpm{extra}")
        for msg in r["fail"]:
            print(f"        FAIL  {msg}")
        for msg in r["warn"]:
            print(f"        warn  {msg}")
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("OK", "WARN", "FAIL")}
    print(f"\n{len(rows)} videos: {n['OK']} OK, {n['WARN']} WARN, {n['FAIL']} FAIL")
    if any(r.get("caption_kind") == "auto" for r in rows):
        print("NOTE: 'auto' captions are YouTube's own ASR. WER against them measures agreement "
              "between two ASR systems, not accuracy. Accuracy needs manual captions or a hand transcript.")

    if a.csv:
        keys = sorted({k for r in rows for k in r if k not in ("fail", "warn", "info", "status")})
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["status"] + keys + ["issues"])
            w.writeheader()
            for r in rows:
                row = {k: v for k, v in r.items() if k not in ("fail", "warn", "info")}
                row["issues"] = " | ".join(["FAIL: " + m for m in r["fail"]] + r["warn"] + ["info: " + m for m in r.get("info", [])])
                w.writerow(row)
        print(f"wrote {a.csv}")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
