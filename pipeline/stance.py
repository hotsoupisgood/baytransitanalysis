#!/usr/bin/env python3
"""
pipeline/stance.py -- the per-turn Claude stance pass (plan/SHOWCASE.md 4a/4b,
plan/ANALYSIS-transit.md 3 and 8; estimand 2 of the write-up).

UNIT. A turn is a maximal run of consecutive segments by one diarized cluster
inside one episode -- the same object analysis/a1_sentiment.py scores with
VADER -- taken from the QA-gated transcripts (output/qa/latest.csv OK / WARN),
with the role of its cluster (speaker_roles, or pipeline/roles.py when an
episode predates the labeller, exactly as analysis/load_episodes.py does).
Turns under --min-words carry no stance and are never sent.

INSTRUMENT (docs/MODELS.md 2). Pinned model (default SCORING_MODEL =
claude-sonnet-4-6), thinking disabled, no sampling parameters, JSON-schema
output, the codebook (plan/STANCE-CODEBOOK.md below its `--- CODEBOOK ---`
line) as a cached system block. prompt_version = 'st-<sha256 of the codebook
block>[:10]/<model>'; one changed byte is a new instrument. k separate calls
per turn (default 3); the median is the measurement and the spread is the
noise, both stored.

MONEY. Every call is a `stance_calls` row BEFORE the request (direct) or
before the batch is created (Message Batches), so a killed run resumes without
re-billing and the ledger (pipeline/spend.py claude_total, which now sums this
table too) is exact. --cap-usd bounds one run; CLAUDE_CAP_USD bounds the
project. New tables only: stance_calls, stance_batches, turn_stance. Nothing
else in state.sqlite is touched.

    stance.py sample --n 300 --stratify body,role --min-words 5 --out output/stance/pilot_turns.csv
    stance.py run --turns output/stance/pilot_turns.csv --dry-run           # estimate, no API
    stance.py run --turns ... --k 1 --direct --max-turns 10 --cap-usd 0.15  # smoke
    stance.py run --turns ... --k 3 --batch --cap-usd 2.5                   # submit one Message Batch
    stance.py collect [--wait]                                              # results -> stance_calls, turn_stance
    stance.py report [--turns CSV] [--out output/stance/pilot_results.csv]
    stance.py coding-sheet --turns CSV --n 150 --out output/stance/coding_sheet.csv
    stance.py agreement --sheet-a a.csv --sheet-b b.csv [--model]
"""
import argparse, csv, hashlib, json, random, re, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402

ROOT = state.ROOT
CODEBOOK_PATH = ROOT / "plan" / "STANCE-CODEBOOK.md"
FRAMES_PATH = ROOT / "plan" / "frames.yaml"
QA_CSV = ROOT / "output" / "qa" / "latest.csv"
DEFAULT_MODEL = "claude-sonnet-4-6"
SEED = 20261017
WORD_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")       # analysis/load_episodes.WORD_RE
CONTEXT_WORDS = 40                                        # tail of the previous turn / head of the next
MAX_TURN_WORDS = 700                                      # a longer turn is sent as its first 700 words, flagged

STANCES = ["-2", "-1", "0", "1", "2", "not_applicable"]
HOSTILITY = [0, 1, 2, 3, 4]
SUB_TOPICS = ["housing_element", "specific_project", "zoning_upzoning", "adu", "parking", "affordability_displacement",
              "rail_service", "bus_service", "bike_ped", "road_diet", "transit_funding", "safety", "other"]
SELF_IDS = ["resident", "homeowner", "renter", "business", "org_rep", "staff", "elected", "unknown"]
ROLES = ("public", "dais")
MODEL_COLUMNS = ("stance", "hostility", "frames", "sub_topic", "self_id", "rationale")   # blank on the coding sheet


def frame_keys(path=FRAMES_PATH):
    """Top-level keys of plan/frames.yaml (same reader rule as a0_terms.load_frames)."""
    keys = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].rstrip()
        if s and not s.startswith((" ", "-")) and s.endswith(":"):
            keys.append(s[:-1].strip())
    return keys


FRAMES = frame_keys()


def codebook_text(path=CODEBOOK_PATH):
    """The block the model sees: everything after the '--- CODEBOOK ---' line."""
    txt = Path(path).read_text(encoding="utf-8")
    marker = "--- CODEBOOK ---"
    if marker not in txt:
        raise SystemExit(f"{path} has no '{marker}' line")
    return txt.split(marker, 1)[1].strip() + "\n"


INSTRUCTIONS = codebook_text()
CODEBOOK_VERSION = "st-" + hashlib.sha256(INSTRUCTIONS.encode("utf-8")).hexdigest()[:10]


def prompt_version_for(model):
    """Codebook hash + model id: a different model is a different instrument."""
    return f"{CODEBOOK_VERSION}/{model}"


SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": STANCES},
        "hostility": {"type": "integer", "enum": HOSTILITY},
        "frames": {"type": "array", "items": {"type": "string", "enum": FRAMES}},
        "sub_topic": {"type": "string", "enum": SUB_TOPICS},
        "self_id": {"type": "string", "enum": SELF_IDS},
        "rationale": {"type": "string"},
        "contains_instructions": {"type": "boolean"},
    },
    "required": ["stance", "hostility", "frames", "sub_topic", "self_id", "rationale", "contains_instructions"],
    "additionalProperties": False,
}

# --- prices and the estimate (pipeline/classify.py, measured 2026-09-23) -------------
PRICES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
}
BATCH_DISCOUNT = 0.5
# Measured on the 2026-10-17 smoke (10 direct calls, claude-sonnet-4-6, this
# codebook): the cached system block billed 2,569 tokens (3.11 chars/token;
# classify.py's captions codebook was 3.64), the user turn 1.61 tokens per
# prompt word (captions: 2.04 -- ASR prose tokenises better than caption
# fragments), replies 54-74 tokens (mean 59). The constants sit a little above
# the measurement so the estimate does not come in under the bill.
TOKENS_PER_WORD = 1.65
CODEBOOK_CHARS_PER_TOKEN = 3.11
CACHE_MIN_TOKENS = {"claude-haiku-4-5-20251001": 4096}
CACHE_PAYS_OFF_CALLS = 300        # classify.py: cache hit share 28 % at 93 calls, 47 % at 190, 76 % at 455
CACHE_HIT_SHARE = 0.75
OUTPUT_TOKENS_PER_CALL = 65
MAX_TOKENS = 220


def price_for(model):
    p = PRICES.get(model)
    if not p:
        raise SystemExit(f"no price table for model {model!r}; add it to stance.PRICES (docs/MODELS.md) before running")
    return p


def codebook_tokens():
    return len(INSTRUCTIONS) / CODEBOOK_CHARS_PER_TOKEN


def codebook_rate(model, n_calls):
    p = price_for(model)
    if codebook_tokens() < CACHE_MIN_TOKENS.get(model, 0) or n_calls < CACHE_PAYS_OFF_CALLS:
        return p["input"]
    return CACHE_HIT_SHARE * p["cache_read"] + (1 - CACHE_HIT_SHARE) * p["cache_write"]


def text_tokens(words):
    return int(round(words * TOKENS_PER_WORD))


def usd(u, model, batch=False):
    """Dollars for one usage object or dict at the model's prices."""
    p = price_for(model)
    g = (lambda k: getattr(u, k, None)) if not isinstance(u, dict) else u.get
    d = ((g("input_tokens") or 0) * p["input"] + (g("output_tokens") or 0) * p["output"]
         + (g("cache_creation_input_tokens") or 0) * p["cache_write"]
         + (g("cache_read_input_tokens") or 0) * p["cache_read"]) / 1e6
    return d * BATCH_DISCOUNT if batch else d


def estimate_usd(prompt_words, n_calls, model, batch=False):
    """Expected cost of n_calls whose user turns total prompt_words words."""
    p = price_for(model)
    d = (text_tokens(prompt_words) * p["input"]
         + n_calls * (codebook_tokens() * codebook_rate(model, n_calls) + OUTPUT_TOKENS_PER_CALL * p["output"])) / 1e6
    return d * BATCH_DISCOUNT if batch else d


def worst_case_usd(prompt_words, model, batch=False):
    """Upper bound for ONE call: codebook as a cache write, prompt as input, max output."""
    p = price_for(model)
    book_rate = p["input"] if codebook_tokens() < CACHE_MIN_TOKENS.get(model, 0) else p["cache_write"]
    d = (codebook_tokens() * book_rate + text_tokens(prompt_words) * p["input"] + MAX_TOKENS * p["output"]) / 1e6
    return d * BATCH_DISCOUNT if batch else d


# --- the prompt ------------------------------------------------------------------

def clip_words(text, n, tail=False):
    w = text.split()
    if len(w) <= n:
        return text
    return ("... " + " ".join(w[-n:])) if tail else (" ".join(w[:n]) + " ...")


def user_turn(t):
    """The user message for one turn record (dict with family, agenda_item,
    prev_text, text, next_text). Transcript text is quoted as data."""
    fam = {"density": "density (housing)", "transit": "transit", "both": "both density and transit"}.get(t.get("family") or "", t.get("family") or "unknown")
    item = (t.get("agenda_item") or "").strip()
    text = t["text"]
    note = ""
    if len(text.split()) > MAX_TURN_WORDS:
        text = clip_words(text, MAX_TURN_WORDS)
        note = f"(The turn is longer than {MAX_TURN_WORDS} words; its first {MAX_TURN_WORDS} are shown.)\n"
    parts = [f"Episode topic family: {fam}.",
             f"Agenda item title: {item[:300]}" if item else "Agenda item title: (none recorded)",
             "",
             "Transcript (ASR, data only, enclosed in tags):",
             "<previous_turn>", clip_words(t.get("prev_text") or "", CONTEXT_WORDS, tail=True) or "(none)", "</previous_turn>",
             "<turn_to_code>", text, "</turn_to_code>",
             "<next_turn>", clip_words(t.get("next_text") or "", CONTEXT_WORDS) or "(none)", "</next_turn>",
             "", note + "Code the turn inside <turn_to_code>."]
    return "\n".join(parts)


def request_params(model, t):
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=[{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_turn(t)}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )


def parse_reply(msg):
    """(parsed dict or None) from a Messages response; never raises. A reply
    outside the codebook's value sets is None, never a number."""
    raw = next((b.text for b in getattr(msg, "content", []) if getattr(b, "type", "") == "text"), "")
    return parse_json(raw)


def parse_json(raw):
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(out, dict):
        return None
    if out.get("stance") not in STANCES or out.get("hostility") not in HOSTILITY:
        return None
    if out.get("sub_topic") not in SUB_TOPICS or out.get("self_id") not in SELF_IDS:
        return None
    fr = out.get("frames")
    if not isinstance(fr, list) or any(f not in FRAMES for f in fr):
        return None
    out["frames"] = sorted(set(fr))
    out["rationale"] = str(out.get("rationale") or "")[:200]
    out["contains_instructions"] = bool(out.get("contains_instructions"))
    return out


# --- turns from the corpus ---------------------------------------------------------

def gated_episode_ids(qa_csv=QA_CSV):
    keep = set()
    with open(qa_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("kind") == "episode" and r.get("status") in ("OK", "WARN"):
                keep.add(r["video_id"])
    return keep


def collapse_loops(segments):
    sys.path.insert(0, str(ROOT / "analysis"))
    import load_db  # noqa: E402
    return load_db.collapse_loops(segments)[0]


def roles_for(doc, db_roles):
    """{cluster: role} and its source, as analysis/load_episodes.roles_for."""
    if db_roles:
        return dict(db_roles), "db"
    if doc.get("roles"):
        return {k: v["role"] for k, v in doc["roles"].items()}, "json"
    import roles as pr  # noqa: E402
    r, _regions, _summ = pr.label_transcript(doc)
    return {k: v["role"] for k, v in r.items()}, "computed"


def episode_turns(episode_id, doc, roles, meta):
    """Every turn of one episode in order: maximal runs of one cluster. The
    index counts turns with at least one word (a1_sentiment.load_turns)."""
    segs = collapse_loops(doc["segments"])
    runs, cur = [], None
    for s in segs:
        spk = s.get("speaker") or "UNLABELLED"
        if cur and cur["cluster"] == spk:
            cur["text"].append(s["text"]); cur["end"] = float(s["end"])
        else:
            if cur:
                runs.append(cur)
            cur = {"cluster": spk, "start": float(s["start"]), "end": float(s["end"]), "text": [s["text"]]}
    if cur:
        runs.append(cur)
    turns, idx = [], 0
    for r in runs:
        text = " ".join(x.strip() for x in r["text"]).strip()
        words = len(WORD_RE.findall(text.lower()))
        if not words:
            continue
        idx += 1
        turns.append({"episode_id": episode_id, "cluster": r["cluster"], "turn_idx": idx,
                      "start_s": round(r["start"], 1), "end_s": round(r["end"], 1), "words": words,
                      "role": roles.get(r["cluster"], "unknown"), "body": meta["body"], "family": meta["family"],
                      "agenda_item": meta.get("agenda_item") or "", "text": text})
    for i, t in enumerate(turns):
        t["prev_text"] = turns[i - 1]["text"] if i else ""
        t["next_text"] = turns[i + 1]["text"] if i + 1 < len(turns) else ""
        t["turn_id"] = turn_id(t)
    return turns


def turn_id(t):
    return f"{t['episode_id']}__t{int(t['turn_idx'])}"


def load_corpus_turns(con, qa_csv=QA_CSV, min_words=5, log_fn=None):
    """All turns of every QA-gated 'done' episode, with roles. Reads state.sqlite
    and the transcript JSON only."""
    gated = gated_episode_ids(qa_csv)
    db_roles = defaultdict(dict)
    for r in con.execute("SELECT episode_id, speaker, role FROM speaker_roles"):
        db_roles[r["episode_id"]][r["speaker"]] = r["role"]
    rows = con.execute("SELECT e.episode_id, e.transcript_path, e.tags, e.agenda_item, v.speaker AS body "
                       "FROM episodes e JOIN videos v ON v.video_id = e.video_id "
                       "WHERE e.status = 'done' AND e.transcript_path IS NOT NULL ORDER BY e.episode_id").fetchall()
    out, sources, n_ep = [], Counter(), 0
    for r in rows:
        if r["episode_id"] not in gated:
            continue
        path = Path(r["transcript_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        roles, src = roles_for(doc, db_roles.get(r["episode_id"]))
        sources[src] += 1
        n_ep += 1
        for t in episode_turns(r["episode_id"], doc, roles, {"body": r["body"], "family": r["tags"], "agenda_item": r["agenda_item"]}):
            if t["words"] >= min_words:
                out.append(t)
    if log_fn:
        log_fn(f"{n_ep} gated episodes, {len(out)} turns >= {min_words} words; roles " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    return out


# --- the sampler -------------------------------------------------------------------

def allocate(cell_sizes, n, floor=4):
    """Proportional allocation over non-empty cells with a floor, exactly n in
    total when the cells can hold it. Largest-remainder rounding; a cell never
    gets more than it has; the floor is applied first and the remainder is
    spread in proportion to cell size."""
    cells = {k: v for k, v in cell_sizes.items() if v > 0}
    if not cells:
        return {}
    alloc = {k: min(floor, v) for k, v in cells.items()}
    left = n - sum(alloc.values())
    if left <= 0:
        # the floor alone exceeds n: scale the floor down evenly, largest cells first
        order = sorted(cells, key=lambda k: (-cells[k], k))
        alloc = {k: 0 for k in cells}
        i = 0
        while sum(alloc.values()) < min(n, sum(cells.values())):
            k = order[i % len(order)]
            if alloc[k] < cells[k]:
                alloc[k] += 1
            i += 1
        return alloc
    room = {k: cells[k] - alloc[k] for k in cells}
    total_room = sum(room.values())
    if total_room <= 0:
        return alloc
    share = {k: left * room[k] / total_room for k in cells}
    extra = {k: min(room[k], int(share[k])) for k in cells}
    rem = left - sum(extra.values())
    for k in sorted(cells, key=lambda k: (-(share[k] - int(share[k])), -cells[k], k)):
        if rem <= 0:
            break
        if extra[k] < room[k]:
            extra[k] += 1; rem -= 1
    # if rounding left slack (cells saturated), keep filling any cell with room
    for k in sorted(cells, key=lambda k: (-cells[k], k)):
        while rem > 0 and alloc[k] + extra[k] < cells[k]:
            extra[k] += 1; rem -= 1
    return {k: alloc[k] + extra[k] for k in cells}


def draw(turns, n, stratify=("body", "role"), floor=4, seed=SEED, roles=ROLES):
    """Stratified draw of n turns over the cells of `stratify`; roles outside
    `roles` are excluded. Deterministic for a seed and the same turn set in
    any order."""
    pool = [t for t in turns if t["role"] in roles]
    key = lambda t: tuple(str(t[k]) for k in stratify)       # noqa: E731
    cells = defaultdict(list)
    for t in pool:
        cells[key(t)].append(t)
    alloc = allocate({k: len(v) for k, v in cells.items()}, n, floor)
    rng = random.Random(seed)
    out = []
    for k in sorted(cells):
        rows = sorted(cells[k], key=lambda t: t["turn_id"])
        rng.shuffle(rows)
        for t in rows[:alloc.get(k, 0)]:
            out.append(dict(t, cell="|".join(k)))
    out.sort(key=lambda t: t["turn_id"])
    return out, alloc


SAMPLE_COLS = ["turn_id", "episode_id", "cluster", "turn_idx", "start_s", "end_s", "words", "role", "body", "family",
               "agenda_item", "cell", "prev_text", "text", "next_text"]


def write_csv(path, rows, cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_turns_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["turn_idx"] = int(r["turn_idx"]); r["words"] = int(r["words"])
        r["turn_id"] = r.get("turn_id") or turn_id(r)
    return rows


# --- the DB side --------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS stance_calls (
  call_id TEXT NOT NULL,            -- '<episode_id>__t<turn_idx>__k<k>'
  episode_id TEXT NOT NULL, cluster TEXT NOT NULL, turn_idx INTEGER NOT NULL, k INTEGER NOT NULL,
  model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  stance TEXT, hostility INTEGER, frames TEXT, sub_topic TEXT, self_id TEXT, rationale TEXT, contains_instructions INTEGER,
  input_tokens INTEGER, cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, latency_s REAL, at TEXT NOT NULL, batch_id TEXT, error TEXT,
  PRIMARY KEY (call_id, prompt_version));
CREATE INDEX IF NOT EXISTS idx_sc_turn ON stance_calls(episode_id, turn_idx);
CREATE TABLE IF NOT EXISTS stance_batches (
  batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,   -- submitted | ended | collected | failed
  model TEXT NOT NULL, prompt_version TEXT NOT NULL,
  n_requests INTEGER, n_succeeded INTEGER, n_errored INTEGER, est_usd REAL, cost_usd REAL, collected_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS turn_stance (
  episode_id TEXT NOT NULL, cluster TEXT NOT NULL, turn_idx INTEGER NOT NULL, prompt_version TEXT NOT NULL,
  model TEXT NOT NULL, start_s REAL, end_s REAL, words INTEGER, role TEXT, body TEXT, family TEXT, agenda_item TEXT,
  k INTEGER, n_ok INTEGER, n_numeric INTEGER,
  stance TEXT,                      -- median of the numeric votes as text ('-2'..'2', half-steps possible) or 'not_applicable'
  stance_median REAL, stance_spread REAL, stance_votes TEXT, na_share REAL,
  hostility_median REAL, hostility_spread REAL, hostility_votes TEXT,
  frames TEXT, sub_topic TEXT, self_id TEXT, rationale TEXT, contains_instructions INTEGER, disagree INTEGER,
  cost_usd REAL, updated_at TEXT NOT NULL,
  PRIMARY KEY (episode_id, cluster, turn_idx, prompt_version));
"""

CALL_COLS = ("call_id", "episode_id", "cluster", "turn_idx", "k", "model", "prompt_version", "stance", "hostility", "frames",
             "sub_topic", "self_id", "rationale", "contains_instructions", "input_tokens", "cache_read_input_tokens",
             "cache_creation_input_tokens", "output_tokens", "cost_usd", "latency_s", "at", "batch_id", "error")


def ensure_tables(con):
    con.executescript(DDL)


def call_id(t, k):
    return f"{turn_id(t)}__k{k}"


def call_row(t, k, model, pv, res=None, usage=None, cost=0.0, latency=None, batch_id=None, error=None):
    u = {kk: (getattr(usage, kk, 0) or 0) for kk in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")} if usage else {}
    return {"call_id": call_id(t, k), "episode_id": t["episode_id"], "cluster": t["cluster"], "turn_idx": int(t["turn_idx"]), "k": k,
            "model": model, "prompt_version": pv,
            "stance": res["stance"] if res else None, "hostility": res["hostility"] if res else None,
            "frames": ";".join(res["frames"]) if res else None, "sub_topic": res["sub_topic"] if res else None,
            "self_id": res["self_id"] if res else None, "rationale": res["rationale"] if res else None,
            "contains_instructions": int(res["contains_instructions"]) if res else None,
            "input_tokens": u.get("input_tokens"), "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"), "output_tokens": u.get("output_tokens"),
            "cost_usd": round(cost, 6), "latency_s": latency, "at": state.now_iso(), "batch_id": batch_id, "error": error}


def insert_call(con, row, replace=False):
    """INSERT OR IGNORE by default; REPLACE only overwrites a row of the same
    instrument and never puts an unlabelled row over a labelled one."""
    if replace and row.get("stance") is None:
        replace = False
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    return con.execute(f"{verb} INTO stance_calls ({', '.join(CALL_COLS)}) VALUES ({', '.join('?' * len(CALL_COLS))})",
                       [row[c] for c in CALL_COLS]).rowcount


def existing_calls(con, pv, turn_ids):
    """{call_id: row} for every stance_calls row of these turns under pv."""
    out = {}
    for r in con.execute("SELECT * FROM stance_calls WHERE prompt_version = ?", (pv,)):
        tid = r["call_id"].rsplit("__k", 1)[0]
        if tid in turn_ids:
            out[r["call_id"]] = dict(r)
    return out


def pending_batch_ids(con):
    return {r["batch_id"] for r in con.execute("SELECT batch_id FROM stance_batches WHERE status IN ('submitted', 'ended')")}


# --- aggregation ---------------------------------------------------------------------

def aggregate(votes):
    """k parsed replies (dicts) -> the turn_stance measurement. Stance: median
    of the numeric votes when at least half the parsed votes are numeric, else
    'not_applicable'; spread = max - min of the numeric votes. Hostility:
    median and spread over all parsed votes. Frames: those in at least half
    the votes. sub_topic / self_id: the mode (ties -> the value of the first
    vote that has it). disagree = any vote differs from another on stance."""
    ok = [v for v in votes if v]
    if not ok:
        return None
    num = [int(v["stance"]) for v in ok if v["stance"] != "not_applicable"]
    na = len(ok) - len(num)
    numeric = len(num) * 2 >= len(ok) and num
    sm = float(statistics.median(num)) if numeric else None
    hs = [int(v["hostility"]) for v in ok]
    fc = Counter(f for v in ok for f in v["frames"])
    frames = sorted(f for f, c in fc.items() if c * 2 >= len(ok))

    def mode(key):
        c = Counter(v[key] for v in ok)
        top = max(c.values())
        return next(v[key] for v in ok if c[v[key]] == top)
    return {"n_ok": len(ok), "n_numeric": len(num),
            "stance": (f"{sm:g}" if numeric else "not_applicable"),
            "stance_median": sm, "stance_spread": (float(max(num) - min(num)) if num else None),
            "stance_votes": ";".join(v["stance"] for v in ok), "na_share": round(na / len(ok), 3),
            "hostility_median": float(statistics.median(hs)), "hostility_spread": float(max(hs) - min(hs)),
            "hostility_votes": ";".join(str(h) for h in hs),
            "frames": ";".join(frames), "sub_topic": mode("sub_topic"), "self_id": mode("self_id"),
            "rationale": ok[0]["rationale"], "contains_instructions": int(any(v["contains_instructions"] for v in ok)),
            "disagree": int(len({v["stance"] for v in ok}) > 1)}


def write_turn_stance(con, t, model, pv, k, calls):
    """Aggregate the labelled calls of one turn into turn_stance. Returns the
    aggregate or None when nothing is labelled."""
    votes = [parse_json(json.dumps({"stance": c["stance"], "hostility": c["hostility"], "frames": (c["frames"] or "").split(";") if c["frames"] else [],
                                    "sub_topic": c["sub_topic"], "self_id": c["self_id"], "rationale": c["rationale"] or "",
                                    "contains_instructions": bool(c["contains_instructions"])}))
             for c in calls if c["stance"] is not None]
    agg = aggregate(votes)
    if not agg:
        return None
    cost = sum(c["cost_usd"] or 0 for c in calls)
    con.execute("INSERT OR REPLACE INTO turn_stance (episode_id, cluster, turn_idx, prompt_version, model, start_s, end_s, words, role, body, "
                "family, agenda_item, k, n_ok, n_numeric, stance, stance_median, stance_spread, stance_votes, na_share, hostility_median, "
                "hostility_spread, hostility_votes, frames, sub_topic, self_id, rationale, contains_instructions, disagree, cost_usd, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t["episode_id"], t["cluster"], int(t["turn_idx"]), pv, model, t.get("start_s"), t.get("end_s"), t.get("words"), t.get("role"),
                 t.get("body"), t.get("family"), (t.get("agenda_item") or None), k, agg["n_ok"], agg["n_numeric"], agg["stance"],
                 agg["stance_median"], agg["stance_spread"], agg["stance_votes"], agg["na_share"], agg["hostility_median"],
                 agg["hostility_spread"], agg["hostility_votes"], agg["frames"], agg["sub_topic"], agg["self_id"], agg["rationale"],
                 agg["contains_instructions"], agg["disagree"], round(cost, 6), state.now_iso()))
    return agg


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


# --- run ---------------------------------------------------------------------------------

def plan_calls(con, turns, model, pv, k):
    """(todo, have, pending): the (turn, k) pairs still to send, the labelled
    rows, and the rows sitting in an uncollected batch."""
    ids = {t["turn_id"] for t in turns}
    have = existing_calls(con, pv, ids)
    live = pending_batch_ids(con)
    todo, pending, done = [], [], []
    for t in turns:
        for kk in range(1, k + 1):
            cid = call_id(t, kk)
            r = have.get(cid)
            if r and r["stance"] is not None:
                done.append(cid)
            elif r and r["batch_id"] and r["batch_id"] in live:
                pending.append(cid)
            else:
                todo.append((t, kk))
    return todo, done, pending


def run(args, client_factory=None):
    con = state.connect(args.state)
    ensure_tables(con)
    import spend  # noqa: E402
    model = args.model or state.setting("SCORING_MODEL", DEFAULT_MODEL)
    price_for(model)
    pv = prompt_version_for(model)
    turns = read_turns_csv(args.turns)
    if args.max_turns:
        turns = turns[:args.max_turns]
    todo, done, pending = plan_calls(con, turns, model, pv, args.k)
    words = sum(len(user_turn(t).split()) for t, _ in todo)
    batch = args.batch and not args.direct
    est = estimate_usd(words, len(todo), model, batch=batch)
    worst = sum(worst_case_usd(len(user_turn(t).split()), model, batch=batch) for t, _ in todo)
    db_total = spend.claude_total(con)
    cap = spend.caps()["CLAUDE_CAP_USD"]
    log(f"model {model}  prompt_version {pv}  codebook ~{codebook_tokens():.0f} tokens  {len(turns)} turns x k={args.k}: "
        f"{len(done)} labelled, {len(pending)} pending in a batch, {len(todo)} calls to make ({words:,} prompt words); "
        f"estimate ${est:.2f} (worst case ${worst:.2f}) {'batch' if batch else 'direct'}; --cap-usd ${args.cap_usd:.2f}; "
        f"Claude ledger ${db_total:.2f} of CLAUDE_CAP_USD ${cap:.2f}" + (" (dry run)" if args.dry_run else ""))
    if not todo:
        log("nothing to send")
        finish_turns(con, turns, model, pv, args.k)
        con.close()
        return 0
    if args.dry_run:
        log(f"DRY RUN: {len(todo)} calls, estimated ${estimate_usd(words, len(todo), model):.2f} direct / "
            f"${estimate_usd(words, len(todo), model, batch=True):.2f} via the Batches API; nothing sent")
        con.close()
        return 0
    why = spend.claude_refusal(con, worst if batch else 0)
    if why:
        log(f"REFUSED: {why}")
        con.close()
        return 2
    if batch:
        if est > args.cap_usd:
            log(f"REFUSED: estimate ${est:.2f} exceeds --cap-usd ${args.cap_usd:.2f}; use --max-turns or a smaller k")
            con.close()
            return 2
        rc = submit_batch(con, todo, model, pv, est, client_factory)
        con.close()
        return rc
    rc = run_direct(con, todo, turns, model, pv, args.k, args.cap_usd, (db_total, cap), client_factory)
    con.close()
    return rc


def run_direct(con, todo, turns, model, pv, k, cap_usd, hard_cap, client_factory=None):
    client = (client_factory or _real_client)()
    spent = 0.0
    for t, kk in todo:
        worst = worst_case_usd(len(user_turn(t).split()), model)
        if spent + worst > cap_usd:
            log(f"STOP: ${spent:.4f} spent this run, next call could reach --cap-usd ${cap_usd:.2f}; re-run resumes from the DB")
            finish_turns(con, turns, model, pv, k)
            return 2
        if hard_cap[0] + spent + worst > hard_cap[1]:
            log(f"STOP: CLAUDE_CAP_USD ${hard_cap[1]:.2f}: ${hard_cap[0] + spent:.4f} recorded, next call could pass it")
            finish_turns(con, turns, model, pv, k)
            return 2
        # the row FIRST, then the request: a killed run finds the row and does not re-bill
        insert_call(con, call_row(t, kk, model, pv, error="in flight"))
        t0 = time.time()
        res, usage, err = None, None, None
        try:
            msg = client.messages.create(**request_params(model, t))
            res, usage = parse_reply(msg), msg.usage
            if res is None:
                err = "unparseable reply"
        except Exception as e:  # noqa: BLE001 -- 429/5xx/400: record, do not retry blindly
            err = f"{type(e).__name__}: {str(e)[:160]}"
            log(f"  {call_id(t, kk)}: API error {err}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(20)
        cost = usd(usage, model) if usage else 0.0
        spent += cost
        row = call_row(t, kk, model, pv, res, usage, cost, round(time.time() - t0, 2), error=err)
        con.execute(f"UPDATE stance_calls SET {', '.join(f'{c} = ?' for c in CALL_COLS[7:])} WHERE call_id = ? AND prompt_version = ?",
                    [row[c] for c in CALL_COLS[7:]] + [row["call_id"], pv])
        if res and res["contains_instructions"]:
            log(f"  !! {call_id(t, kk)}: model flagged instruction-like text in the transcript (coded on content; tell Puck)")
        log(f"  {call_id(t, kk)} {t['role']}/{t['body']} -> {res['stance'] if res else 'none'} h{res['hostility'] if res else '-'} "
            f"{res['sub_topic'] if res else ''} {res['self_id'] if res else ''}  ${cost:.4f} (run ${spent:.4f})")
    n = finish_turns(con, turns, model, pv, k)
    log(f"done: run spend ${spent:.4f}; {n} turn(s) aggregated into turn_stance")
    return 0


def _real_client():
    import anthropic
    return anthropic.Anthropic()


def finish_turns(con, turns, model, pv, k):
    """turn_stance for every turn with at least one labelled call."""
    ids = {t["turn_id"] for t in turns}
    have = existing_calls(con, pv, ids)
    by = defaultdict(list)
    for cid, r in have.items():
        by[cid.rsplit("__k", 1)[0]].append(r)
    n = 0
    for t in turns:
        calls = by.get(t["turn_id"], [])
        if any(c["stance"] is not None for c in calls) and write_turn_stance(con, t, model, pv, k, calls):
            n += 1
    return n


# --- Message Batches ------------------------------------------------------------------------

def submit_batch(con, todo, model, pv, est, client_factory=None):
    """Rows first (stance NULL, provisional batch id), then the batch, then the real id."""
    now = state.now_iso()
    provisional = f"pending-{now}"
    reqs = [{"custom_id": call_id(t, kk), "params": request_params(model, t)} for t, kk in todo]
    con.execute("INSERT INTO stance_batches (batch_id, created_at, status, model, prompt_version, n_requests, est_usd) VALUES (?,?,?,?,?,?,?)",
                (provisional, now, "submitted", model, pv, len(reqs), round(est, 4)))
    for t, kk in todo:
        insert_call(con, call_row(t, kk, model, pv, batch_id=provisional), replace=True)
    try:
        mb = (client_factory or _real_client)().messages.batches.create(requests=reqs)
    except Exception as e:  # noqa: BLE001 -- nothing billed; undo the provisional rows
        con.execute("DELETE FROM stance_calls WHERE batch_id = ? AND stance IS NULL", (provisional,))
        con.execute("DELETE FROM stance_batches WHERE batch_id = ?", (provisional,))
        log(f"batch create failed: {type(e).__name__}: {str(e)[:200]}; provisional rows removed")
        return 1
    con.execute("UPDATE stance_batches SET batch_id = ?, note = ? WHERE batch_id = ?", (mb.id, f"processing_status {mb.processing_status}", provisional))
    con.execute("UPDATE stance_calls SET batch_id = ? WHERE batch_id = ?", (mb.id, provisional))
    log(f"submitted {mb.id}: {len(reqs)} requests, prompt_version {pv}, status {mb.processing_status}; run `stance.py collect`")
    return 0


def apply_result(con, batch_id, pv, cid, res, usage, cost, error=None):
    if res:
        con.execute("UPDATE stance_calls SET stance = ?, hostility = ?, frames = ?, sub_topic = ?, self_id = ?, rationale = ?, "
                    "contains_instructions = ?, input_tokens = ?, cache_read_input_tokens = ?, cache_creation_input_tokens = ?, "
                    "output_tokens = ?, cost_usd = ?, at = ?, error = NULL WHERE call_id = ? AND prompt_version = ? AND batch_id = ?",
                    (res["stance"], res["hostility"], ";".join(res["frames"]), res["sub_topic"], res["self_id"], res["rationale"],
                     int(res["contains_instructions"]), getattr(usage, "input_tokens", 0) or 0, getattr(usage, "cache_read_input_tokens", 0) or 0,
                     getattr(usage, "cache_creation_input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0, round(cost, 6),
                     state.now_iso(), cid, pv, batch_id))
        return True
    con.execute("UPDATE stance_calls SET batch_id = NULL, cost_usd = ?, error = ? WHERE call_id = ? AND prompt_version = ? AND batch_id = ? AND stance IS NULL",
                (round(cost, 6), (error or "unparseable reply")[:200], cid, pv, batch_id))
    return False


def collect(args, client_factory=None):
    con = state.connect(args.state)
    ensure_tables(con)
    client = (client_factory or _real_client)()
    while True:
        batches = [dict(r) for r in con.execute("SELECT * FROM stance_batches WHERE status IN ('submitted', 'ended') ORDER BY created_at")]
        if not batches:
            log("no submitted batches")
            break
        still = 0
        for b in batches:
            bid, model, pv = b["batch_id"], b["model"], b["prompt_version"]
            try:
                mb = client.messages.batches.retrieve(bid)
            except Exception as e:  # noqa: BLE001
                log(f"{bid}: retrieve failed: {type(e).__name__}: {str(e)[:160]}")
                still += 1
                continue
            if mb.processing_status != "ended":
                rc = getattr(mb, "request_counts", None)
                log(f"{bid}: {mb.processing_status}" + (f" ({rc.processing} processing, {rc.succeeded} succeeded, {rc.errored} errored)" if rc else ""))
                still += 1
                continue
            con.execute("UPDATE stance_batches SET status = 'ended' WHERE batch_id = ?", (bid,))
            ok = bad = 0
            cost_total = 0.0
            flagged = []
            for result in client.messages.batches.results(bid):
                cid = result.custom_id
                if result.result.type == "succeeded":
                    msg = result.result.message
                    res, usage = parse_reply(msg), msg.usage
                    cost = usd(usage, model, batch=True) if usage else 0.0
                    cost_total += cost
                    if apply_result(con, bid, pv, cid, res, usage, cost):
                        ok += 1
                    else:
                        bad += 1
                    if res and res["contains_instructions"]:
                        flagged.append(cid)
                else:
                    bad += 1
                    err = getattr(getattr(result.result, "error", None), "type", result.result.type)
                    apply_result(con, bid, pv, cid, None, None, 0.0, error=f"batch {result.result.type}: {err}")
            con.execute("UPDATE stance_batches SET status = 'collected', n_succeeded = ?, n_errored = ?, cost_usd = ?, collected_at = ? WHERE batch_id = ?",
                        (ok, bad, round(cost_total, 4), state.now_iso(), bid))
            log(f"{bid}: collected {ok} labels, {bad} errored / unparseable, ${cost_total:.4f}"
                + (f"; {len(flagged)} reply(ies) flagged instruction-like transcript text (coded on content; tell Puck)" if flagged else ""))
            # aggregate every turn this batch touched
            rows = [dict(r) for r in con.execute("SELECT DISTINCT episode_id, cluster, turn_idx FROM stance_calls WHERE batch_id = ?", (bid,))]
            turns = turns_from_db_rows(con, rows, args.turns)
            kmax = con.execute("SELECT MAX(k) FROM stance_calls WHERE batch_id = ?", (bid,)).fetchone()[0] or 1
            n = finish_turns(con, turns, model, pv, kmax)
            log(f"{bid}: {n} turn(s) aggregated into turn_stance")
        if not still or not args.wait:
            break
        time.sleep(max(60, args.poll_s))
    con.close()
    return 0


def turns_from_db_rows(con, rows, turns_csv=None):
    """Turn dicts for aggregation: from the --turns CSV when given (full
    metadata), else the minimum the stance_calls rows carry."""
    meta = {}
    if turns_csv and Path(turns_csv).exists():
        meta = {t["turn_id"]: t for t in read_turns_csv(turns_csv)}
    out = []
    for r in rows:
        t = {"episode_id": r["episode_id"], "cluster": r["cluster"], "turn_idx": int(r["turn_idx"])}
        t["turn_id"] = turn_id(t)
        out.append(meta.get(t["turn_id"], t))
    return out


# --- report ---------------------------------------------------------------------------------------

RESULT_COLS = ["turn_id", "episode_id", "cluster", "turn_idx", "start_s", "end_s", "words", "role", "body", "family", "agenda_item",
               "k", "n_ok", "stance", "stance_median", "stance_spread", "stance_votes", "na_share", "hostility_median", "hostility_spread",
               "hostility_votes", "frames", "sub_topic", "self_id", "rationale", "contains_instructions", "disagree", "cost_usd", "prompt_version"]


def report(args):
    con = state.connect(args.state)
    ensure_tables(con)
    model = args.model or state.setting("SCORING_MODEL", DEFAULT_MODEL)
    pv = args.prompt_version or prompt_version_for(model)
    rows = [dict(r) for r in con.execute("SELECT * FROM turn_stance WHERE prompt_version = ? ORDER BY episode_id, turn_idx", (pv,))]
    if args.turns and Path(args.turns).exists():
        keep = {t["turn_id"] for t in read_turns_csv(args.turns)}
        rows = [r for r in rows if turn_id(r) in keep]
    for r in rows:
        r["turn_id"] = turn_id(r)
    calls = con.execute("SELECT COUNT(*) n, COALESCE(SUM(cost_usd), 0) usd, COALESCE(SUM(output_tokens), 0) o, COALESCE(SUM(input_tokens), 0) i, "
                        "COALESCE(SUM(cache_read_input_tokens), 0) cr, COALESCE(SUM(cache_creation_input_tokens), 0) cw, "
                        "SUM(stance IS NULL) unl FROM stance_calls WHERE prompt_version = ?", (pv,)).fetchone()
    print(f"prompt_version {pv}: {calls['n']} calls, {calls['unl']} unlabelled, ${calls['usd']:.4f}; tokens in {calls['i']:,} "
          f"cache-read {calls['cr']:,} cache-write {calls['cw']:,} out {calls['o']:,}"
          + (f"; {calls['o'] / max(1, calls['n'] - (calls['unl'] or 0)):.0f} output tokens/call" if calls['n'] else ""))
    print(f"{len(rows)} turns in turn_stance")
    if rows:
        print("\nstance by role:")
        for role in sorted({r["role"] for r in rows}, key=str):
            rs = [r for r in rows if r["role"] == role]
            print(f"  {str(role):<8} n={len(rs):<4} " + dist_line(rs))
        print("\nstance by body (public | dais):")
        for body in sorted({r["body"] for r in rows}, key=str):
            parts = []
            for role in ROLES:
                rs = [r for r in rows if r["body"] == body and r["role"] == role]
                if rs:
                    parts.append(f"{role} n={len(rs)} {dist_line(rs, short=True)}")
            print(f"  {str(body):<12} " + " | ".join(parts))
        print(f"\nnot_applicable share: {sum(1 for r in rows if r['stance'] == 'not_applicable') / len(rows):.1%}")
        sp = Counter(r["stance_spread"] if r["stance_spread"] is not None else "n/a" for r in rows)
        print("k-spread of stance (max - min of numeric votes): " + ", ".join(f"{k}: {v}" for k, v in sorted(sp.items(), key=lambda x: str(x[0]))))
        print(f"turns where the k samples disagree on stance (incl. NA vs number): {sum(r['disagree'] for r in rows)} of {len(rows)}")
        hc = Counter(r["hostility_median"] for r in rows)
        print("hostility median: " + ", ".join(f"{k:g}: {v}" for k, v in sorted(hc.items())))
        print("self_id: " + ", ".join(f"{k}: {v}" for k, v in Counter(r["self_id"] for r in rows).most_common()))
        print("sub_topic: " + ", ".join(f"{k}: {v}" for k, v in Counter(r["sub_topic"] for r in rows).most_common()))
        fc = Counter(f for r in rows for f in (r["frames"] or "").split(";") if f)
        print("frames: " + ", ".join(f"{k}: {v}" for k, v in fc.most_common()))
    if args.out:
        write_csv(args.out, rows, RESULT_COLS)
        print(f"wrote {args.out}")
    con.close()
    return 0


def dist_line(rows, short=False):
    c = Counter(r["stance"] for r in rows)
    keys = ["-2", "-1.5", "-1", "-0.5", "0", "0.5", "1", "1.5", "2", "not_applicable"]
    num = [r["stance_median"] for r in rows if r["stance_median"] is not None]
    mean = f" mean {statistics.mean(num):+.2f}" if num else ""
    if short:
        return " ".join(f"{k if k != 'not_applicable' else 'NA'}:{c[k]}" for k in keys if c[k]) + mean
    return " ".join(f"{k if k != 'not_applicable' else 'NA'}:{c[k]:<3}" for k in keys if c[k]) + mean


# --- coding sheet and agreement -----------------------------------------------------------------

SHEET_COLS = ["turn_id", "body", "role", "family", "agenda_item", "prev_text", "text", "next_text",
              "stance", "hostility", "frames", "sub_topic", "self_id", "notes"]


def coding_sheet(turns, n, seed=SEED):
    """n of the turns, shuffled by seed, every model column blank."""
    rng = random.Random(seed)
    rows = sorted(turns, key=lambda t: t["turn_id"])
    rng.shuffle(rows)
    out = []
    for t in rows[:n]:
        out.append({k: t.get(k, "") for k in ("turn_id", "body", "role", "family", "agenda_item", "prev_text", "text", "next_text")}
                   | {k: "" for k in ("stance", "hostility", "frames", "sub_topic", "self_id", "notes")})
    return out


def weighted_kappa(a, b, categories):
    """Quadratic-weighted Cohen's kappa over paired codes (lists of equal length)."""
    idx = {c: i for i, c in enumerate(categories)}
    n, m = len(a), len(categories)
    if n == 0 or m < 2:
        return float("nan")
    obs = [[0.0] * m for _ in range(m)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(obs[i]) for i in range(m)]
    rb = [sum(obs[i][j] for i in range(m)) for j in range(m)]
    w = [[((i - j) ** 2) / ((m - 1) ** 2) for j in range(m)] for i in range(m)]
    po = sum(w[i][j] * obs[i][j] for i in range(m) for j in range(m)) / n
    pe = sum(w[i][j] * ra[i] * rb[j] / n for i in range(m) for j in range(m)) / n
    return float("nan") if pe == 0 else 1 - po / pe


def krippendorff_alpha_interval(a, b):
    """Krippendorff's alpha, interval metric, two coders, no missing values."""
    n = len(a)
    if n < 2:
        return float("nan")
    vals = list(a) + list(b)
    do = sum((x - y) ** 2 for x, y in zip(a, b)) * 2 / (2 * n)          # within-unit, pairs counted both ways / (m-1)=1
    mean = sum(vals) / len(vals)
    de = sum((v - mean) ** 2 for v in vals) * 2 / (len(vals) - 1)
    return float("nan") if de == 0 else 1 - do / de


def agreement(sheet_a, sheet_b, model_rows=None):
    """Agreement statistics between two coding sheets (and the model's
    turn_stance rows when given). Stance kappa is over turns both coders
    gave a numeric stance; NA agreement is reported separately."""
    A = {r["turn_id"]: r for r in sheet_a}
    B = {r["turn_id"]: r for r in sheet_b}
    both = [t for t in A if t in B and A[t].get("stance", "") != "" and B[t].get("stance", "") != ""]

    def num(v):
        v = (v or "").strip()
        return None if v in ("", "not_applicable", "NA", "na") else int(round(float(v)))
    out = {"n_turns": len(both)}
    sa, sb = [num(A[t]["stance"]) for t in both], [num(B[t]["stance"]) for t in both]
    na_agree = sum((x is None) == (y is None) for x, y in zip(sa, sb))
    out["na_agreement"] = na_agree / len(both) if both else float("nan")
    pairs = [(x, y) for x, y in zip(sa, sb) if x is not None and y is not None]
    out["n_numeric_pairs"] = len(pairs)
    out["stance_kappa_w"] = weighted_kappa([p[0] for p in pairs], [p[1] for p in pairs], [-2, -1, 0, 1, 2])
    out["stance_exact"] = sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")
    out["stance_alpha"] = krippendorff_alpha_interval([p[0] for p in pairs], [p[1] for p in pairs])
    hp = [(int(A[t]["hostility"]), int(B[t]["hostility"])) for t in both if str(A[t].get("hostility", "")).strip() != "" and str(B[t].get("hostility", "")).strip() != ""]
    out["hostility_kappa_w"] = weighted_kappa([p[0] for p in hp], [p[1] for p in hp], HOSTILITY)
    out["hostility_exact"] = sum(x == y for x, y in hp) / len(hp) if hp else float("nan")
    out["disagreements"] = [(t, A[t]["stance"], B[t]["stance"]) for t in both if (A[t]["stance"] or "").strip() != (B[t]["stance"] or "").strip()]
    if model_rows:
        M = {turn_id(r): r for r in model_rows}
        for name, S in (("a", A), ("b", B)):
            ts = [t for t in both if t in M and M[t].get("stance") is not None]
            mp = [(num(S[t]["stance"]), None if M[t]["stance"] == "not_applicable" else int(round(float(M[t]["stance"])))) for t in ts]
            mp = [(x, y) for x, y in mp if x is not None and y is not None]
            out[f"model_vs_{name}_kappa_w"] = weighted_kappa([p[0] for p in mp], [p[1] for p in mp], [-2, -1, 0, 1, 2])
            out[f"model_vs_{name}_n"] = len(mp)
            hpm = [(int(S[t]["hostility"]), int(round(M[t]["hostility_median"]))) for t in ts if str(S[t].get("hostility", "")).strip() != ""]
            out[f"model_vs_{name}_hostility_kappa_w"] = weighted_kappa([p[0] for p in hpm], [p[1] for p in hpm], HOSTILITY)
        spread = [M[t]["stance_spread"] for t in both if t in M and M[t].get("stance_spread") is not None]
        out["model_k_spread"] = dict(Counter(spread))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "run", "collect", "report", "coding-sheet", "agreement"])
    ap.add_argument("--n", type=int, default=None, help="sample: turns to draw (300); coding-sheet: rows (150)")
    ap.add_argument("--with-model", action="store_true", help="agreement: also compare each sheet with turn_stance")
    ap.add_argument("--stratify", default="body,role")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--floor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--qa-csv", default=str(QA_CSV))
    ap.add_argument("--out", default=None)
    ap.add_argument("--turns", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--cap-usd", type=float, default=float(state.setting("STANCE_CAP_USD", 2.5)))
    ap.add_argument("--wait", action="store_true", help="collect: poll until every submitted batch has ended")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--sheet-a", default=None)
    ap.add_argument("--sheet-b", default=None)
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        con = state.connect(a.state, create=False)
        turns = load_corpus_turns(con, a.qa_csv, a.min_words, log)
        con.close()
        rows, alloc = draw(turns, a.n or 300, tuple(a.stratify.split(",")), a.floor, a.seed)
        out = a.out or str(ROOT / "output" / "stance" / "pilot_turns.csv")
        write_csv(out, rows, SAMPLE_COLS)
        cells = Counter(r["cell"] for r in rows)
        log(f"{len(rows)} turns drawn over {len(cells)} non-empty cells (floor {a.floor}, seed {a.seed}): "
            + ", ".join(f"{k} {v}" for k, v in sorted(cells.items())))
        log(f"wrote {out}")
        return 0
    if a.cmd == "run":
        if not a.turns:
            ap.error("run needs --turns CSV")
        if not (a.batch or a.direct or a.dry_run):
            ap.error("run needs --batch or --direct (or --dry-run)")
        return run(a)
    if a.cmd == "collect":
        return collect(a)
    if a.cmd == "report":
        return report(a)
    if a.cmd == "coding-sheet":
        if not a.turns:
            ap.error("coding-sheet needs --turns CSV")
        rows = coding_sheet(read_turns_csv(a.turns), a.n or 150, a.seed)
        out = a.out or str(ROOT / "output" / "stance" / "coding_sheet.csv")
        write_csv(out, rows, SHEET_COLS)
        log(f"wrote {out}: {len(rows)} turns, columns {', '.join(SHEET_COLS)}")
        return 0
    if a.cmd == "agreement":
        if not (a.sheet_a and a.sheet_b):
            ap.error("agreement needs --sheet-a and --sheet-b")
        with open(a.sheet_a, newline="", encoding="utf-8") as f:
            A = list(csv.DictReader(f))
        with open(a.sheet_b, newline="", encoding="utf-8") as f:
            B = list(csv.DictReader(f))
        model_rows = None
        if a.with_model or a.model is not None or a.prompt_version:
            con = state.connect(a.state, create=False)
            ensure_tables(con)
            pv = a.prompt_version or prompt_version_for(a.model or state.setting("SCORING_MODEL", DEFAULT_MODEL))
            model_rows = [dict(r) for r in con.execute("SELECT * FROM turn_stance WHERE prompt_version = ?", (pv,))]
            con.close()
        res = agreement(A, B, model_rows)
        for k, v in res.items():
            if k == "disagreements":
                continue
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        print(f"disagreements ({len(res['disagreements'])}):")
        for t, x, y in res["disagreements"]:
            print(f"  {t}: A={x} B={y}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
