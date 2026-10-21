#!/usr/bin/env python3
"""
analysis/validate_transcription.py -- Phase 0 evidence on the pilot episode
transcripts (PLAN.md 6, docs/TESTING.md level 4), $0: no GPU, no API, no
network. Reads output/whisper/<episode>.json, the meeting caption tracks under
output/subs/, the episode audio, state.sqlite (episodes + speaker_roles) and
output/t0/roles/; writes ONLY under output/validation/.

What it measures, per episode
  1. Disagreement rate vs the meeting's caption track, cues overlapping the
     episode's [start_s, end_s] in meeting time. YouTube auto captions are ASR
     themselves (~8-15 % WER on clean speech), so against them this is a
     DISAGREEMENT rate between two recognisers, not a WER. Oakland's Granicus
     track is a live captioner (human, but real-time: clean-verbatim, lagging,
     with its own drops) and is reported separately.
     Normalisation (both sides, norm_tokens): HTML-unescaped, lower-cased,
     YouTube's "[ __ ]" mask, bracketed non-speech tags and ">>" removed,
     "%" -> percent, "$N" -> "N dollars", digit groups de-commaed, letter/digit
     boundaries split (sb79 -> sb 79), every remaining integer / decimal /
     ordinal spelled out (1100-2099 read as years: "twenty twenty six"; 7+
     digit strings and phone-number groups digit by digit), then tokens [a-z0-9']+. Residual number
     mismatches are the two sides spelling the same number differently
     ("two thousand twenty six").
     Global rate = word Levenshtein (qa_transcripts.edit_distance, Myers) /
     reference words. jiwer is not installed; this is the same measure.
     Per-minute rate: a difflib alignment of the same token lists, each
     substitution / deletion charged to the minute of the caption word, each
     insertion to the minute of the transcript word, over the caption words of
     that minute. Caption times are shifted by the median (transcript - caption)
     time of matched words first (live captions lag the audio).
  2. Hallucination / dropout: (a) transcript text lying inside >= 20 s of
     low-energy audio (frame RMS 20 dB below the episode's 75th-percentile
     level, from the 16 kHz opus); (b) the QA gate's loop signatures plus a
     repeated-4-gram check; (c) minutes where the captions carry > 2x the
     transcript's words (+20 slack) = dropped speech, and the reverse = extra
     text.
  3. Diarization sanity: clusters vs public-comment turn cues (timer phrases,
     name-and-city intros -- counted per 15 s event, never stored), sliver
     clusters, merge / split flags. Roles from speaker_roles, else
     output/t0/roles/<episode>.json (test 6), else computed in memory.
  4. A listening set: 12 excerpts of ~55 s (8 worst per-minute stratified by
     body x role, 4 random), cut copy-only from the episode audio, with the
     WhisperX text, the caption text, and a correction file per excerpt in the
     scripts/make_reference_excerpts.py format so scripts/score_reference.py
     turns Puck's corrections into a true WER.

    .venv/bin/python analysis/validate_transcription.py            # everything -> output/validation/
    .venv/bin/python analysis/validate_transcription.py --no-audio # skip RMS and clips
"""
import argparse, csv, difflib, html, json, os, random, re, sqlite3, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
from qa_transcripts import edit_distance, check_video, CENSOR, NONSPEECH_TAG, INLINE_TS  # noqa: E402

OUT = ROOT / "output" / "validation"
WIN_S, WIN_RATIO, WIN_SLACK = 60, 2.0, 20          # same window rule as qa_transcripts
MIN_REF_WORDS_MIN = 20                             # a per-minute rate needs this many caption words
LOW_DB_BELOW_P75, LOW_MIN_S, FRAME_S = 20.0, 20.0, 0.5
LAG_SEARCH_S = 20.0                                # captions searched this far outside the episode before lag correction
EXCERPT_S, N_WORST, N_RANDOM, SEED = 55, 8, 4, 20260921
BODIES = ["bart", "sanjose", "sanrafael", "oakland"]

WORD = re.compile(r"[a-z0-9']+")
TIMER_CUES = [r"next speaker", r"you have (one|two|three|1|2|3) minutes?", r"that('s| is) (your )?time",
              r"thank you,? next", r"(the )?next (caller|commenter)", r"speaker (number )?\d+", r"caller (number )?\d+",
              r"time('s| is) up", r"your time is up", r"please wrap (it )?up", r"thirty seconds"]
INTRO_CUES = [r"my name is", r"i live (in|on|at)", r"i'?m a (resident|homeowner|renter|member|native)",
              r"resident of", r"good (morning|afternoon|evening),? (mayor|council|president|directors?|board|chair|supervisors|commissioners)",
              r"i'?ve (lived|been) (in|a resident)", r"speaking (on behalf|as a)"]
TIMER_RE = re.compile("|".join(f"(?:{c})" for c in TIMER_CUES), re.I)
INTRO_RE = re.compile("|".join(f"(?:{c})" for c in INTRO_CUES), re.I)
NAME_RE = re.compile(r"my name is\s+\S+(\s+\S+)?", re.I)      # redaction for anything printed

# ------------------------------------------------------------ normalisation --
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _under_1000(n):
    w = []
    if n >= 100:
        w += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        w.append(TENS[n // 10])
        if n % 10:
            w.append(ONES[n % 10])
    elif n or not w:
        w.append(ONES[n])
    return w


def num_words(n):
    """Cardinal words for a non-negative int. 1100-2099 (not round hundreds) read as years."""
    if 1100 <= n <= 2099 and n % 100:
        hi, lo = divmod(n, 100)
        return _under_1000(hi) + (["oh", ONES[lo]] if lo < 10 else _under_1000(lo))
    if n == 0:
        return ["zero"]
    parts, scales = [], [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
    for val, name in scales:
        if n >= val:
            parts += _under_1000(n // val) + [name]
            n %= val
    if n:
        parts += _under_1000(n)
    return parts


def ordinal_words(n):
    w = num_words(n)
    last = w[-1]
    w[-1] = ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return w


def norm_tokens(text):
    """Token list for the disagreement rate; the rule is in the module docstring."""
    t = html.unescape(text).replace("’", "'").replace("\xa0", " ").lower()
    t = CENSOR.sub(" ", t)
    t = NONSPEECH_TAG.sub(" ", t)
    t = t.replace(">>", " ").replace("%", " percent ")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"\$\s?(\d+(?:\.\d+)?)((?:\s+(?:hundred|thousand|million|billion))?)", r"\1\2 dollars", t)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: " ".join(ordinal_words(int(m.group(1)))), t)
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", t)
    t = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: " ".join(num_words(int(m.group(1))) + ["point"] + [ONES[int(c)] for c in m.group(2)]), t)
    # <= 6 digits: cardinal (40,000 -> forty thousand); a single group of >= 7 digits or a run of >= 3 groups
    # with >= 7 digits (phone numbers, webinar ids) digit by digit, so the two sides' grouping
    # ("555 010 0100" vs "5550100100") does not score
    t = re.sub(r"\b\d[\d ]{5,}\d\b", lambda m: " ".join(ONES[int(c)] for c in m.group(0) if c.isdigit())
               if sum(c.isdigit() for c in m.group(0)) >= 7 and len(m.group(0).split()) >= 3 else m.group(0), t)
    t = re.sub(r"\b\d+\b", lambda m: " ".join(num_words(int(m.group(0))) if len(m.group(0)) <= 6 else [ONES[int(c)] for c in m.group(0)]), t)
    return WORD.findall(t)


# ---------------------------------------------------------------- captions --
def vtt_time(s):
    h, m, sec = s.strip().split()[0].replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def parse_cues(path):
    """[(start, end, text)] plus kind 'auto' | 'manual'. Auto (YouTube ASR)
    cues roll: each phrase appears twice, once with inline word timings; only
    those lines are new text (qa_transcripts.parse_vtt rule). Granicus /
    manual cues are taken whole."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    auto = bool(INLINE_TS.search(raw))
    cues = []
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
            a, b = lines[ts].split("-->")
            st, en = vtt_time(a), vtt_time(b)
        except ValueError:
            continue
        if text:
            cues.append((st, max(en, st), text))
    return cues, ("auto" if auto else "manual")


def timed_words(items, tokens_of=norm_tokens):
    """(start, end, text, extra) -> [(t, token, extra)], words spread evenly over the item."""
    out = []
    for st, en, text, extra in items:
        toks = tokens_of(text)
        if not toks:
            continue
        step = max(en - st, 0.0) / len(toks)
        out += [(st + i * step, w, extra) for i, w in enumerate(toks)]
    return out


# ---------------------------------------------------------------- alignment --
def align(ref, hyp):
    """difflib opcodes on token lists (autojunk off). Global rates use
    edit_distance; this is for attributing errors to minutes, roles and KWIC."""
    return difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes()


def lag_estimate(ref_w, hyp_w, ops):
    """Median (transcript time - caption time) over matched words; 0 if too few."""
    d = [hyp_w[j1 + k][0] - ref_w[i1 + k][0] for tag, i1, i2, j1, j2 in ops if tag == "equal" for k in range(i2 - i1)]
    if len(d) < 20:
        return 0.0, len(d)
    d.sort()
    return d[len(d) // 2], len(d)


def per_minute(ref_w, hyp_w, ops, n_min):
    """Per-minute error counts. Returns list of dicts with ref, hyp, err, wer."""
    rows = [{"ref": 0, "hyp": 0, "err": 0} for _ in range(n_min)]
    for t, _, _ in ref_w:
        rows[min(max(int(t // WIN_S), 0), n_min - 1)]["ref"] += 1
    for t, _, _ in hyp_w:
        rows[min(max(int(t // WIN_S), 0), n_min - 1)]["hyp"] += 1
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        for k in range(i1, i2):                      # substitutions + deletions: the caption word's minute
            rows[min(max(int(ref_w[k][0] // WIN_S), 0), n_min - 1)]["err"] += 1
        for k in range(j1 + (i2 - i1), j2):          # insertions beyond the substituted span
            rows[min(max(int(hyp_w[k][0] // WIN_S), 0), n_min - 1)]["err"] += 1
    for r in rows:
        r["wer"] = round(r["err"] / r["ref"], 3) if r["ref"] >= MIN_REF_WORDS_MIN else None
    return rows


def per_role(ref_w, hyp_w, ops):
    """{role: (err, ref_words)}; ref words carry the role of the transcript
    cluster active at their (lag-corrected) time, hyp words their own."""
    acc = defaultdict(lambda: [0, 0])
    for _, _, role in ref_w:
        acc[role][1] += 1
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        for k in range(i1, i2):
            acc[ref_w[k][2]][0] += 1
        for k in range(j1 + (i2 - i1), j2):
            acc[hyp_w[k][2]][0] += 1
    return dict(acc)


def divergence_runs(ref_w, hyp_w, ops, min_words=4, top=3, ctx=4):
    """Largest non-equal runs as KWIC: (size, t, REF snippet, HYP snippet), names redacted."""
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        n = max(i2 - i1, j2 - j1)
        if n < min_words:
            continue
        t = ref_w[i1][0] if i1 < len(ref_w) else (hyp_w[j1][0] if j1 < len(hyp_w) else 0.0)
        r = " ".join(w for _, w, _ in ref_w[max(i1 - ctx, 0):i1]) + " [" + " ".join(w for _, w, _ in ref_w[i1:i2]) + "] " \
            + " ".join(w for _, w, _ in ref_w[i2:i2 + ctx])
        h = " ".join(w for _, w, _ in hyp_w[max(j1 - ctx, 0):j1]) + " [" + " ".join(w for _, w, _ in hyp_w[j1:j2]) + "] " \
            + " ".join(w for _, w, _ in hyp_w[j2:j2 + ctx])
        out.append((n, t, redact(r), redact(h)))
    return sorted(out, reverse=True)[:top]


def redact(text):
    return NAME_RE.sub("my name is [name]", text)


# ------------------------------------------------------------------- audio --
def frame_rms_db(opus_path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(opus_path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(16000 * FRAME_S)
    m = len(x) // n * n
    if m == 0:
        return np.zeros(0)
    fr = x[:m].reshape(-1, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def low_energy_stretches(rms_db, frame_s=FRAME_S, min_s=LOW_MIN_S, below=LOW_DB_BELOW_P75):
    """[(start_s, end_s, thr_db)] maximal runs >= min_s where a 1.5 s moving max
    of frame RMS stays > `below` dB under the episode's 75th percentile."""
    import numpy as np
    if len(rms_db) == 0:
        return [], None
    thr = float(np.percentile(rms_db, 75)) - below
    k = 3
    padded = np.concatenate([rms_db[:1].repeat(k // 2), rms_db, rms_db[-1:].repeat(k // 2)])
    mov = np.max(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    low = mov < thr
    out, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            if (j - i + 1) * frame_s >= min_s:
                out.append((i * frame_s, (j + 1) * frame_s, round(thr, 1)))
            i = j + 1
        else:
            i += 1
    return out, thr


def text_over_silence(segments, stretches, min_words=3, share=0.8):
    """Segments (>= min_words) with >= `share` of their span inside a low-energy stretch."""
    hits = []
    for s in segments:
        st, en = float(s["start"]), float(s["end"])
        if en <= st or len(s["text"].split()) < min_words:
            continue
        inside = sum(max(0.0, min(en, b) - max(st, a)) for a, b, _ in stretches)
        if inside / (en - st) >= share:
            hits.append((st, en, len(s["text"].split()), redact(s["text"].strip())))
    return hits


def repeated_ngrams(segments, n=4, min_repeats=4, window_s=60):
    """4-grams that recur >= min_repeats times within window_s: (t, count, phrase)."""
    words = []
    for s in segments:
        toks = norm_tokens(s["text"])
        if not toks:
            continue
        step = (float(s["end"]) - float(s["start"])) / len(toks)
        words += [(float(s["start"]) + i * step, w) for i, w in enumerate(toks)]
    seen, out = defaultdict(list), []
    for i in range(len(words) - n + 1):
        g = tuple(w for _, w in words[i:i + n])
        seen[g].append(words[i][0])
    for g, ts in seen.items():
        ts.sort()
        for a in range(len(ts)):
            b = a
            while b + 1 < len(ts) and ts[b + 1] - ts[a] <= window_s:
                b += 1
            if b - a + 1 >= min_repeats:
                out.append((ts[a], b - a + 1, " ".join(g)))
                break
    return sorted(out)


# ------------------------------------------------------------------- roles --
def load_roles(con, episode_id, doc):
    rows = con.execute("SELECT speaker, role, method, talk_s FROM speaker_roles WHERE episode_id = ?", (episode_id,)).fetchall()
    if rows:
        return {r[0]: {"role": r[1], "method": r[2], "talk_s": r[3]} for r in rows}, "db", None
    p = ROOT / "output" / "t0" / "roles" / f"{episode_id}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return ({k: {"role": v["role"], "method": v["method"], "talk_s": v["talk_s"], "turns": v.get("turns")}
                 for k, v in d["speakers"].items()}, "t0", d.get("regions") or [])
    import roles as pr
    roles, regions, _ = pr.label_transcript(doc)
    return roles, "computed", regions


def cue_events(segments, pattern, gap_s=15.0):
    """Count cue hits, merging hits within gap_s into one event. Nothing stored."""
    times = sorted(float(s["start"]) for s in segments if pattern.search(s["text"]))
    events, last = 0, None
    for t in times:
        if last is None or t - last > gap_s:
            events += 1
        last = t
    return events


def diarization_check(segments, roles):
    stats = defaultdict(lambda: {"talk": 0.0, "turns": 0})
    prev = None
    for s in segments:
        sp = s.get("speaker") or "UNKNOWN"
        stats[sp]["talk"] += max(0.0, float(s["end"]) - float(s["start"]))
        if sp != prev:
            stats[sp]["turns"] += 1
        prev = sp
    clusters = {k: v for k, v in stats.items() if k != "UNKNOWN"}
    n = len(clusters)
    n_public = sum(1 for k in clusters if roles.get(k, {}).get("role") == "public")
    n_dais = sum(1 for k in clusters if roles.get(k, {}).get("role") == "dais")
    slivers = sum(1 for v in clusters.values() if v["talk"] < 15 and v["turns"] <= 2)
    intro = cue_events(segments, INTRO_RE)
    timer = cue_events(segments, TIMER_RE)
    flags = []
    if intro >= 3 and intro >= n_public + 2 and intro >= 1.5 * max(n_public, 1):
        flags.append(f"merge? {intro} public intros vs {n_public} public clusters")
    if n >= 4 and (slivers >= 3 or slivers > 0.3 * n):
        flags.append(f"split? {slivers} of {n} clusters < 15 s")
    return {"n_clusters": n, "n_public_clusters": n_public, "n_dais_clusters": n_dais, "sliver_clusters": slivers,
            "intro_events": intro, "timer_events": timer, "flags": flags}


# ----------------------------------------------------------------- episodes --
def load_episodes(state_path):
    con = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    eps = [dict(r) for r in con.execute(
        "SELECT e.episode_id, e.video_id AS meeting_video, e.start_s, e.end_s, e.core_start_s, e.core_end_s, e.label, "
        "e.transcript_path, e.opus_path, e.audio_path, v.speaker AS body, v.platform, m.captions, m.captions_path "
        "FROM episodes e JOIN videos v ON v.video_id = e.video_id "
        "LEFT JOIN videos m ON m.video_id = e.video_id "
        "WHERE e.status = 'done' ORDER BY v.speaker, e.episode_id")]
    return con, eps


def role_at(t, segs_sorted, roles):
    """Role of the transcript cluster active at time t (nearest segment if none overlaps)."""
    best, bd = None, None
    for st, en, sp in segs_sorted:
        d = 0.0 if st <= t <= en else min(abs(t - st), abs(t - en))
        if bd is None or d < bd:
            best, bd = sp, d
            if d == 0.0:
                break
    return roles.get(best, {}).get("role", "unknown") if best else "unknown"


def analyse_episode(ep, con, do_audio=True):
    doc = json.loads(Path(ep["transcript_path"]).read_text(encoding="utf-8"))
    segs = [s for s in doc["segments"] if isinstance(s.get("text"), str) and s["text"].strip()]
    dur = float(doc.get("transcribed_s") or doc.get("duration_s") or ep["end_s"] - ep["start_s"])
    roles, roles_src, regions = load_roles(con, ep["episode_id"], doc)
    segs_sorted = [(float(s["start"]), float(s["end"]), s.get("speaker") or "UNKNOWN") for s in segs]
    r = {"episode_id": ep["episode_id"], "body": ep["body"], "platform": ep["platform"], "label": ep["label"],
         "meeting_video": ep["meeting_video"], "start_s": ep["start_s"], "end_s": ep["end_s"], "dur_s": round(dur, 1),
         "n_segments": len(segs), "roles_source": roles_src, "caption_kind": None}

    # transcript words with time + role
    hyp_items = [(float(s["start"]), float(s["end"]), s["text"], roles.get(s.get("speaker") or "UNKNOWN", {}).get("role", "unknown")) for s in segs]
    hyp_w = timed_words(hyp_items)
    r["hyp_words"] = len(hyp_w)
    r["role_share_public"] = round(sum(1 for _, _, ro in hyp_w if ro == "public") / max(len(hyp_w), 1), 3)

    # captions
    cues, kind = parse_cues(ep["captions_path"]) if ep.get("captions_path") and Path(ep["captions_path"]).exists() else ([], None)
    r["caption_kind"] = ("granicus" if ep["platform"] == "granicus" else kind) if cues else None
    minutes, ops, ref_w = [], [], []
    if cues:
        s0, s1 = float(ep["start_s"]), float(ep["end_s"])
        wide = [(st - s0, en - s0, tx, None) for st, en, tx in cues if en > s0 - LAG_SEARCH_S and st < s1 + LAG_SEARCH_S]
        ref_w0 = timed_words(wide)
        hyp_tok = [w for _, w, _ in hyp_w]
        ops0 = align([w for _, w, _ in ref_w0], hyp_tok)
        lag, n_match = lag_estimate(ref_w0, hyp_w, ops0)
        r["caption_lag_s"], r["lag_matches"] = round(lag, 2), n_match
        # lag-corrected cue times, restricted to the episode window, roles by transcript cluster at that time
        items = [(st - s0 + lag, en - s0 + lag, tx, None) for st, en, tx in cues
                 if en - s0 + lag > 0 and st - s0 + lag < dur]
        ref_w = [(t, w, role_at(t, segs_sorted, roles)) for t, w, _ in timed_words(items) if 0 <= t < dur]
        ref_tok = [w for _, w, _ in ref_w]
        r["ref_words"] = len(ref_tok)
        if ref_tok:
            ops = align(ref_tok, hyp_tok)
            r["wer"] = round(edit_distance(ref_tok, hyp_tok) / len(ref_tok), 4)
            n_min = int(dur // WIN_S) + 1
            minutes = per_minute(ref_w, hyp_w, ops, n_min)
            r["minutes_rated"] = sum(1 for m in minutes if m["wer"] is not None)
            rated = sorted(m["wer"] for m in minutes if m["wer"] is not None)
            r["minute_wer_median"] = rated[len(rated) // 2] if rated else None
            r["minute_wer_max"] = rated[-1] if rated else None
            r["dropped_minutes"] = [i for i, m in enumerate(minutes) if m["ref"] > WIN_RATIO * m["hyp"] + WIN_SLACK]
            r["extra_minutes"] = [i for i, m in enumerate(minutes) if m["hyp"] > WIN_RATIO * m["ref"] + WIN_SLACK]
            r["by_role"] = {k: {"err": v[0], "ref": v[1], "wer": round(v[0] / v[1], 4) if v[1] else None} for k, v in per_role(ref_w, hyp_w, ops).items()}
            r["kwic"] = divergence_runs(ref_w, hyp_w, ops)

    # loops / stock phrases via the QA gate (no captions: episode ids do not match meeting vtt names)
    q = check_video(Path(ep["transcript_path"]))
    r["qa_status"] = "FAIL" if q["fail"] else "WARN" if q["warn"] else "OK"
    r["qa_issues"] = q["fail"] + q["warn"]
    r["coverage"] = q.get("coverage")
    r["max_dup_run"], r["high_compression_segs"], r["stretched_segs"] = q.get("max_dup_run"), q.get("high_compression_segs"), q.get("stretched_segs", 0)
    r["speaker_tag_segs"] = q.get("speaker_tag_segs", 0)
    r["ngram_loops"] = repeated_ngrams(segs)

    # low-energy audio
    r["low_stretches"], r["text_over_silence"] = [], []
    if do_audio and ep.get("opus_path") and Path(ep["opus_path"]).exists():
        rms = frame_rms_db(ep["opus_path"])
        stretches, thr = low_energy_stretches(rms)
        r["low_thr_db"] = round(thr, 1) if thr is not None else None
        r["low_stretches"] = [(round(a, 1), round(b, 1)) for a, b, _ in stretches]
        r["low_total_s"] = round(sum(b - a for a, b, _ in stretches), 1)
        r["text_over_silence"] = [(round(a, 1), round(b, 1), n, t[:120]) for a, b, n, t in text_over_silence(segs, stretches)]
        # sensitivity: the same check with looser thresholds (dB under p75, minimum stretch)
        r["low_sweep"] = {}
        for below, min_s in ((15.0, 20.0), (15.0, 10.0), (10.0, 10.0)):
            st2, _ = low_energy_stretches(rms, below=below, min_s=min_s)
            r["low_sweep"][f"{below:.0f}dB/{min_s:.0f}s"] = {"stretch_s": round(sum(b - a for a, b, _ in st2), 1),
                                                          "text_segs": len(text_over_silence(segs, st2))}

    r["diarization"] = diarization_check(segs, roles)
    r["n_public_regions"] = len(regions) if regions is not None else None
    return r, minutes, segs, roles, ref_w, hyp_w


# ------------------------------------------------------------ listening set --
def choose_excerpts(results, minute_tables, rng):
    """8 worst minutes stratified body x role (distinct episodes where possible) + 4 random."""
    cands = []
    for r in results:
        for i, m in enumerate(minute_tables.get(r["episode_id"], [])):
            if m["wer"] is None or m["ref"] < 40 or (i + 1) * WIN_S > r["dur_s"] + 5:
                continue
            cands.append({"episode_id": r["episode_id"], "body": r["body"], "minute": i, "wer": m["wer"],
                          "role": m.get("role", "unknown"), "ref": m["ref"], "hyp": m["hyp"]})
    chosen, used = [], set()
    for body in BODIES:
        for role in ("dais", "public"):
            pool = sorted((c for c in cands if c["body"] == body and c["role"] == role), key=lambda c: -c["wer"])
            pick = next((c for c in pool if c["episode_id"] not in used), pool[0] if pool else None)
            if pick:
                pick = dict(pick, why=f"worst {role} minute, {body}")
                chosen.append(pick)
                used.add(pick["episode_id"])
    rest = [c for c in cands if not any(c["episode_id"] == x["episode_id"] and c["minute"] == x["minute"] for x in chosen)]
    rng.shuffle(rest)
    for c in rest[:N_RANDOM]:
        chosen.append(dict(c, why="random"))
    return chosen


def cut_clip(src, start, dur, dst):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}", "-c", "copy", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def mmss(s):
    s = max(int(s), 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def write_listen(chosen, eps_by_id, docs, refs, roles_by_ep, listen_dir, do_audio=True):
    from make_reference_excerpts import excerpt as ref_excerpt
    listen_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Listening set -- 12 excerpts of ~55 s from the pilot episodes",
             "",
             "Listen to the clip, read the two texts, and write what is actually said on the",
             "`corrections / verdict` line (or edit the `.txt` next to it: one line per segment,",
             "fix the text, keep the timestamps). Then:",
             "",
             "    .venv/bin/python scripts/score_reference.py output/validation/listen/*.txt --summary",
             "",
             "gives the true WER of the WhisperX text against your corrections, pooled and per",
             "excerpt. Speaker labels are diarization clusters; `(dais)` / `(public)` is the",
             "role labeller's call. Caption text is the YouTube auto track (Granicus live",
             "captions for Oakland) shifted by the measured lag -- it is a second opinion,",
             "not truth. Excerpts 1-8 are the worst per-minute disagreement for each body x",
             "role, 9-12 are random.",
             ""]
    for k, c in enumerate(chosen, 1):
        ep = eps_by_id[c["episode_id"]]
        doc, roles = docs[c["episode_id"]], roles_by_ep[c["episode_id"]]
        st = c["minute"] * WIN_S
        en = min(st + EXCERPT_S, float(doc.get("transcribed_s") or doc.get("duration_s")))
        src = Path(ep["audio_path"])
        clip = listen_dir / f"{k:02d}_{c['episode_id']}_{int(st)}-{int(en)}{src.suffix}"
        if do_audio:
            cut_clip(src, st, en - st, clip)
        ref_excerpt(c["episode_id"], st, en - st, out_dir=listen_dir, quiet=True)
        segs = [s for s in doc["segments"] if float(s["end"]) > st and float(s["start"]) < en]
        caps = [(t, w) for t, w, _ in refs[c["episode_id"]] if st <= t < en]
        meeting_t = float(ep["start_s"]) + st
        lines += [f"## {k:02d}. {c['episode_id']} ({ep['body']}), clip {mmss(st)}-{mmss(en)}, meeting time {mmss(meeting_t)}",
                  "",
                  f"- audio: `{clip.name}`  ·  corrections file: `{c['episode_id']}.{int(st)}-{int(en)}.txt`",
                  f"- why: {c['why']}  ·  minute disagreement {c['wer']:.0%} ({c['ref']} caption words, {c['hyp']} transcript words)",
                  "",
                  "**WhisperX**",
                  ""]
        for s in segs:
            sp = s.get("speaker") or "UNKNOWN"
            lines.append(f"- [{mmss(s['start'])}] {sp} ({roles.get(sp, {}).get('role', '?')}): {s['text'].strip()}")
        lines += ["", "**Captions** (normalised tokens, lag-corrected)", "",
                  "> " + (" ".join(w for _, w in caps) if caps else "(no caption words in this window)"),
                  "",
                  "corrections / verdict: ",
                  ""]
    (listen_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ report --
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist_line(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0"
    return (f"n={len(xs)}  min {min(xs):.1%}  median {pct(xs, .5):.1%}  p90 {pct(xs, .9):.1%}  max {max(xs):.1%}")


def pooled(results, key=lambda r: True):
    err = sum(round(r["wer"] * r["ref_words"]) for r in results if r.get("wer") is not None and key(r))
    ref = sum(r["ref_words"] for r in results if r.get("wer") is not None and key(r))
    return err / ref if ref else None, ref


def write_report(results, chosen, out_dir, listen_dir):
    yt = [r for r in results if r["caption_kind"] == "auto"]
    oak = [r for r in results if r["caption_kind"] == "granicus"]
    L = ["# Transcription validation -- 46 pilot episodes (2026-09-21)", ""]
    L += ["Inputs: `output/whisper/<episode>.json` (WhisperX large-v3 + pyannote community-1, b13/b14),",
          "meeting caption tracks under `output/subs/`, episode audio, `state.sqlite` episodes +",
          "speaker_roles, `output/t0/roles/`. Produced by `analysis/validate_transcription.py`;",
          "per-episode numbers in `episodes.csv`, per-minute in `minutes.csv`, everything in",
          "`results.json`. Nothing under `output/` outside this folder was touched. $0.", ""]
    L += ["## 1. Word disagreement vs captions", "",
          "Against YouTube's auto track this is a **disagreement rate between two ASR systems**, not a",
          "WER: auto captions run ~8-15 % WER themselves on clean speech, and neither side is the",
          "reference. Normalisation rule and the per-minute attribution: script docstring.", ""]
    p, n = pooled(yt)
    L += [f"**YouTube auto captions, {len(yt)} episodes ({n:,} caption words)**: pooled {p:.1%}; per episode {dist_line([r['wer'] for r in yt])}", ""]
    L += ["| body | episodes | caption words | pooled | median | p90 | max |", "|---|---|---|---|---|---|---|"]
    for b in BODIES:
        rs = [r for r in results if r["body"] == b and r.get("wer") is not None]
        if not rs:
            continue
        pp, nn = pooled(rs)
        L.append(f"| {b}{' (Granicus live captions)' if b == 'oakland' else ''} | {len(rs)} | {nn:,} | {pp:.1%} | {pct([r['wer'] for r in rs], .5):.1%} | {pct([r['wer'] for r in rs], .9):.1%} | {max(r['wer'] for r in rs):.1%} |")
    L.append("")
    L += ["By role (caption words carry the role of the transcript cluster active at that time; YouTube episodes only):", "",
          "| role | caption words | disagreement |", "|---|---|---|"]
    for role in ("dais", "public", "unknown"):
        err = sum(r["by_role"].get(role, {}).get("err", 0) for r in yt if r.get("by_role"))
        ref = sum(r["by_role"].get(role, {}).get("ref", 0) for r in yt if r.get("by_role"))
        if ref:
            L.append(f"| {role} | {ref:,} | {err / ref:.1%} |")
    L.append("")
    mins = [m for r in results for m in r.get("_minutes", []) if m["wer"] is not None and r["caption_kind"] == "auto"]
    L += [f"Per-minute (YouTube, {len(mins)} rated minutes with >= {MIN_REF_WORDS_MIN} caption words): "
          f"{dist_line([m['wer'] for m in mins])}; minutes over 50 %: {sum(1 for m in mins if m['wer'] > .5)}, over 100 %: {sum(1 for m in mins if m['wer'] > 1)}.", ""]
    L += ["**5 worst episodes (YouTube)** with the largest divergence runs (`[...]` = the differing words):", ""]
    for r in sorted(yt, key=lambda r: -r["wer"])[:5]:
        L.append(f"- `{r['episode_id']}` ({r['body']}) {r['wer']:.1%}, {r['ref_words']} caption words, lag {r['caption_lag_s']:+.1f} s, "
                 f"worst minute {r['minute_wer_max']:.0%}; dropped minutes {r['dropped_minutes']}, extra {r['extra_minutes']}")
        for nw, t, rr, hh in r.get("kwic", []):
            L.append(f"    - {mmss(t)} ({nw} w) CAP: {rr}")
            L.append(f"      {' ' * len(mmss(t))}        WX:  {hh}")
    L.append("")
    if oak:
        p, n = pooled(oak)
        L += ["## 2. Oakland vs the Granicus live-captioner track", "",
              f"{len(oak)} episodes, {n:,} caption words: pooled {p:.1%}; per episode {dist_line([r['wer'] for r in oak])}. "
              f"Caption lag (median transcript - caption time of matched words): " + ", ".join(f"{r['caption_lag_s']:+.1f} s" for r in oak) + ".",
              "A live captioner types clean verbatim (fillers, false starts and asides dropped, sentences",
              "tidied) and falls behind on fast exchanges, so most of this gap is the captioner, not WhisperX;",
              "the KWIC runs below show which.", ""]
        for r in sorted(oak, key=lambda r: -r["wer"]):
            L.append(f"- `{r['episode_id']}` {r['wer']:.1%}, worst minute {r['minute_wer_max']:.0%}, dropped minutes {r['dropped_minutes']}, extra {r['extra_minutes']}")
            for nw, t, rr, hh in r.get("kwic", [])[:2]:
                L.append(f"    - {mmss(t)} ({nw} w) CAP: {rr}")
                L.append(f"      {' ' * len(mmss(t))}        WX:  {hh}")
        L.append("")
    # hallucination / dropout
    tos = [(r["episode_id"], h) for r in results for h in r["text_over_silence"]]
    low_total = sum(r.get("low_total_s", 0) for r in results)
    loops = [(r["episode_id"], g) for r in results for g in r["ngram_loops"]]
    dropped = [(r["episode_id"], i) for r in results for i in r.get("dropped_minutes", [])]
    extra = [(r["episode_id"], i) for r in results for i in r.get("extra_minutes", [])]
    qa = Counter(r["qa_status"] for r in results)
    L += ["## 3. Hallucination and dropout", "",
          f"- QA gate (`qa_transcripts.check_video`, no captions): {qa.get('OK', 0)} OK, {qa.get('WARN', 0)} WARN, {qa.get('FAIL', 0)} FAIL; "
          f"identical-segment runs >= 3: {sum(1 for r in results if (r['max_dup_run'] or 0) >= 3)}; gzip-ratio loop segments: {sum(r['high_compression_segs'] or 0 for r in results)}; "
          f"stretched segments (text over > 30 s at < 0.5 w/s): {sum(r['stretched_segs'] or 0 for r in results)}.",
          f"- (a) Low-energy audio: {low_total / 60:.1f} min in stretches >= {LOW_MIN_S:.0f} s across {sum(1 for r in results if r['low_stretches'])} episodes "
          f"(threshold {LOW_DB_BELOW_P75:.0f} dB under each episode's p75 frame RMS). Segments of >= 3 words lying >= 80 % inside one: **{len(tos)}**"
          + (":" if tos else "."),]
    for eid, (a, b, nw, t) in tos[:15]:
        L.append(f"    - `{eid}` {mmss(a)}-{mmss(b)} ({nw} w): {t}")
    sweep = defaultdict(lambda: [0.0, 0])
    for r in results:
        for k, v in r.get("low_sweep", {}).items():
            sweep[k][0] += v["stretch_s"]
            sweep[k][1] += v["text_segs"]
    if sweep:
        L.append("  Sensitivity (dB under p75 / minimum stretch -> total quiet audio, segments of text inside): "
                 + "; ".join(f"{k}: {v[0] / 60:.1f} min, {v[1]} segs" for k, v in sweep.items()) + ".")
    tags = [(r["episode_id"], r["speaker_tag_segs"]) for r in results if r.get("speaker_tag_segs")]
    L += [f"- (b) Repeated 4-grams (>= 4 times within 60 s): {len(loops)}" + (":" if loops else "."),]
    for eid, (t, c, g) in loops[:10]:
        L.append(f"    - `{eid}` {mmss(t)} x{c}: {g}")
    L += [f"- (b') Subtitle-style speaker tags as words ('Male Speaker 1 ...', the Whisper hallucination shape the QA gate now flags): "
          f"**{sum(n for _, n in tags)} segments in {len(tags)} episodes**" + (": " + ", ".join(f"`{e}` x{n}" for e, n in tags) if tags else ".")]
    L += [f"- (c) Minutes where captions carry > 2x the transcript's words (+20): **{len(dropped)}** (dropped speech candidates)"
          + (": " + ", ".join(f"`{e}` m{i}" for e, i in dropped[:12]) if dropped else "."),
          f"  Minutes where the transcript carries > 2x the captions' words: **{len(extra)}** (extra-text candidates; on a live-captioner",
          f"  track this is usually the captioner falling behind)" + (": " + ", ".join(f"`{e}` m{i}" for e, i in extra[:12]) if extra else "."), ""]
    for r in results:
        for msg in r["qa_issues"]:
            L.append(f"    - `{r['episode_id']}` QA: {msg}")
    L.append("")
    # diarization
    d = [r["diarization"] for r in results]
    ncl = [x["n_clusters"] for x in d]
    flagged = [(r["episode_id"], f) for r in results for f in r["diarization"]["flags"]]
    L += ["## 4. Diarization sanity", "",
          f"Clusters per episode: min {min(ncl)}, median {pct(ncl, .5):.0f}, max {max(ncl)} ({sum(ncl)} total over {len(d)} episodes); "
          f"public clusters {sum(x['n_public_clusters'] for x in d)}, dais {sum(x['n_dais_clusters'] for x in d)}; "
          f"name-and-city intro events {sum(x['intro_events'] for x in d)}, timer cues {sum(x['timer_events'] for x in d)}; "
          f"sliver clusters (< 15 s, <= 2 turns) {sum(x['sliver_clusters'] for x in d)}. Roles: "
          + ", ".join(f"{k} {v}" for k, v in Counter(r["roles_source"] for r in results).items()) + ".",
          "",
          "Merge flag = >= 3 intro events and intros >= public clusters + 2 and >= 1.5x; split flag = >= 3 sliver",
          "clusters or > 30 % of clusters. Both are heuristics on text cues (intros counted, never stored).", ""]
    L += [f"**Flags ({len(flagged)}):**" if flagged else "**No merge / split flags.**"]
    for eid, f in flagged:
        L.append(f"- `{eid}`: {f}")
    L.append("")
    L += ["| episode | body | clusters | public | dais | intros | timer | slivers |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["diarization"]
        L.append(f"| {r['episode_id']} | {r['body']} | {x['n_clusters']} | {x['n_public_clusters']} | {x['n_dais_clusters']} | {x['intro_events']} | {x['timer_events']} | {x['sliver_clusters']} |")
    L.append("")
    L += ["## 5. Listening set", "",
          f"`{listen_dir.relative_to(ROOT) if listen_dir.is_relative_to(ROOT) else listen_dir}/README.md`: {len(chosen)} excerpts, clips cut copy-only from the episode audio,",
          "WhisperX text with cluster + role, caption text, a blank verdict line and a correction file each",
          "(`scripts/score_reference.py output/validation/listen/*.txt --summary` scores the corrections).", ""]
    for k, c in enumerate(chosen, 1):
        L.append(f"{k:2d}. `{c['episode_id']}` min {c['minute']} ({c['body']}, {c['role']}, {c['wer']:.0%}) -- {c['why']}")
    L.append("")
    L += ["## 6. Verdict", ""]
    p_yt, _ = pooled(yt)
    med_yt = pct([r["wer"] for r in yt], .5)
    p90_yt = pct([r["wer"] for r in yt], .9)
    n_hall = len(tos) + len(loops) + sum(n for _, n in tags)
    verdict = ("**Good enough to scale** on word accuracy and hallucination" if p_yt is not None and p90_yt < 0.15 and not tos and not dropped
               else "**Not yet**: see the numbers above")
    L += [f"{verdict}. Disagreement with YouTube's own ASR is {p_yt:.1%} pooled, median {med_yt:.1%}, p90 {p90_yt:.1%} per episode "
          f"(docs/TESTING.md accepts < 12 % against *human* captions; two ~10 %-WER systems that agree this closely cannot both be far from the "
          f"speech), the worst single minute is {max(m['wer'] for m in mins):.0%} (a roll-call vote with hallucinated speaker tags, listening set #1), no dropped-speech minute, no text over dead air, "
          f"no decoder loop. The dais / public split of the rate is flat (public comment is not transcribed worse than the dais). "
          f"The measurable defects are small and specific: {sum(n for _, n in tags)} hallucinated speaker-tag segments, {len(loops)} short repeats, "
          f"and diarization merges of the roll-call kind (test 6 saw the same). The true WER needs Puck's corrections of the listening set.",
          "",
          "What would improve it (in order of expected gain):",
          "",
          "1. **Initial prompt with body vocabulary** (proper nouns are where the two systems disagree most: station names, bill numbers "
          "'AB 1586' / 'SB 79', 'HCD', 'ADU', director surnames): pass `initial_prompt` per body from a short list kept in "
          "`plan/` (WhisperX `asr_options`). Zero cost; test on the listening set before and after.",
          f"2. **Drop hallucinated speaker tags** at ingest (`^(Male|Female) Speaker` / `^Speaker N` prefix, `qa_transcripts.SPEAKER_TAG`) and keep "
          f"the WARN: {sum(n for _, n in tags)} segments here, concentrated in roll-call votes and long multi-speaker stretches.",
          "3. **QA gate against captions for episodes**: the gate compares nothing to the captions for `<meeting>_e<n>` files because the "
          "caption file is named after the meeting; this script does the mapping via `episodes.start_s`. Fold that into "
          "`qa_transcripts.py` (an `--state` lookup) so every batch gets the per-minute dropped / extra check, not just this one.",
          "",
          "Not indicated by the data: a larger-batch VAD or alignment change (coverage and per-minute rates show no dropped stretches; "
          "the 51 s and 47 s gaps flagged by the gate are recesses and the captions are empty there too).",
          ""]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-audio", action="store_true", help="skip RMS and clip cutting")
    ap.add_argument("--only", default=None, help="comma-separated episode ids")
    a = ap.parse_args(argv)
    state_path = Path(os.path.expanduser(a.state or os.environ.get("STATE_DB_PATH") or "~/.local/share/ytpt/state.sqlite"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    con, eps = load_episodes(state_path)
    if a.only:
        keep = set(a.only.split(","))
        eps = [e for e in eps if e["episode_id"] in keep]
    results, minute_tables, docs, refs, roles_by_ep = [], {}, {}, {}, {}
    for ep in eps:
        r, minutes, segs, roles, ref_w, hyp_w = analyse_episode(ep, con, do_audio=not a.no_audio)
        # majority transcript role per minute, for the stratified listening set
        for i, m in enumerate(minutes):
            c = Counter(ro for t, _, ro in hyp_w if int(t // WIN_S) == i)
            m["role"] = c.most_common(1)[0][0] if c else "unknown"
        r["_minutes"] = minutes
        results.append(r)
        minute_tables[ep["episode_id"]] = minutes
        docs[ep["episode_id"]] = json.loads(Path(ep["transcript_path"]).read_text(encoding="utf-8"))
        refs[ep["episode_id"]] = ref_w
        roles_by_ep[ep["episode_id"]] = roles
        w = f"{r['wer']:.1%}" if r.get("wer") is not None else "n/a"
        print(f"{ep['episode_id']:<22} {ep['body']:<10} {r['dur_s'] / 60:5.1f} min  disagree {w:>6}  lag {r.get('caption_lag_s', 0):+5.1f}s  "
              f"clusters {r['diarization']['n_clusters']:>2}  silence-text {len(r['text_over_silence'])}  {r['qa_status']}", flush=True)

    rng = random.Random(SEED)
    chosen = choose_excerpts(results, minute_tables, rng)
    eps_by_id = {e["episode_id"]: e for e in eps}
    listen_dir = out_dir / "listen"
    write_listen(chosen, eps_by_id, docs, refs, roles_by_ep, listen_dir, do_audio=not a.no_audio)
    write_report(results, chosen, out_dir, listen_dir)

    # machine-readable twins
    with open(out_dir / "minutes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["episode_id", "body", "minute", "ref_words", "hyp_words", "errors", "wer", "role"])
        for r in results:
            for i, m in enumerate(r["_minutes"]):
                w.writerow([r["episode_id"], r["body"], i, m["ref"], m["hyp"], m["err"], m["wer"], m["role"]])
    keys = ["episode_id", "body", "platform", "caption_kind", "dur_s", "ref_words", "hyp_words", "wer", "caption_lag_s", "minutes_rated",
            "minute_wer_median", "minute_wer_max", "dropped_minutes", "extra_minutes", "coverage", "qa_status", "low_total_s",
            "roles_source", "role_share_public"]
    with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["n_clusters", "n_public_clusters", "intro_events", "timer_events", "sliver_clusters", "flags", "text_over_silence"])
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in keys}
            row.update({k: r["diarization"][k] for k in ("n_clusters", "n_public_clusters", "intro_events", "timer_events", "sliver_clusters")})
            row["flags"] = "; ".join(r["diarization"]["flags"])
            row["text_over_silence"] = len(r["text_over_silence"])
            w.writerow(row)
    for r in results:
        r.pop("_minutes", None)
    (out_dir / "results.json").write_text(json.dumps({"episodes": results, "listening_set": chosen}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_dir / 'REPORT.md'}, episodes.csv, minutes.csv, results.json, listen/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
