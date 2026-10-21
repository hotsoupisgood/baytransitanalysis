#!/usr/bin/env python3
"""
redact.py — the name-redaction pass (plan/DECISIONS.md #15, plan/SHOWCASE.md §3).

Episode transcripts pass through here before anything loads, exports or
publishes their text. It never edits output/whisper; it writes
output/redacted/<episode_id>.json with the same structure, the segment text
redacted, and a `redaction` block (counts by rule, total words).

Layered rules, applied to each TURN (maximal run of consecutive segments by one
speaker, the definition in analysis/a1_sentiment.py load_turns), all on the
ORIGINAL text so a later layer cannot be fooled by an earlier one's edit:

  cue_name_is      "my name is X", "name's X", "name for the record is X"
  cue_im           "I'm X and|with|from|,", "I am X and|with|from"
  cue_this_is      "this is X speaking|with|from|,"
  cue_resident     "X, (a) resident of", "X, longtime resident"
  cue_honorific    Councilmember|Mayor|Director|Chair|Mr.|Ms.|Dr.|... + Capitalised token(s)
  cue_record       "for the record, X"
  cue_speaker      "next speaker (is) X", "followed by X", "speaker X"
  cue_thank_you    "thank you, X"          (comma required; "Thank you. Moving on" is not a name)
  person           spaCy en_core_web_sm PERSON entity
  given_name       a capitalised token in plan/given_names.txt with no cue and
                    no PERSON hit (catches a bare first name spaCy misses,
                    e.g. "Gaurav leaves and..."). flag=name redacts alone;
                    flag=ambiguous ("Grant", "Hope", "Will") only redacts as
                    part of a >=2-token capitalised run, same as every other
                    name span (CALIBRATION.md 2026-10-17 #2)
  ci_name_is       CASE-INSENSITIVE strong self-introduction cue: "my name is",
                    "my name's", "name is", "name for the record is", "call me"
  ci_intro         CASE-INSENSITIVE weak cue: "i'm", "i am", "this is", "it's",
                    only when the run is anchored (given-name lexicon hit, or a
                    self-introduction context word right after it)
  ci_given_pair    CASE-INSENSITIVE given_name: a plan/given_names.txt flag=name
                    token plus one surname-like token, no cue needed
  address          number + street name + St|Ave|Blvd|Rd|Dr|Way|Ct|Lane|...   -> [address]
  contact          phone numbers, e-mail addresses                             -> [contact]

The three `ci_` layers exist because every layer above them keys on a leading
capital, so the lowercase routes (local_asr / mlx-whisper, YouTube auto-captions)
published names in clear: 57 self-introductions over 30 people in the
2026-10-20 export, found by the case-insensitive residual scan in
scripts/redaction_scan.py and fixed the same day. They carry no casing
signal, so they never guess: each span is anchored on an explicit cue or on the
given-name lexicon, and every token is tested against the stop lists rather than
against its capitalisation. They run LAST, so the capitalised layers keep their
own calibration on capitalised text.

Every candidate span is trimmed against an ALLOWLIST of place, agency and
institution names (plan/channels.tsv bodies, plan/vocab.tsv, plan/keywords_*.txt,
plan/frames.yaml terms, a Bay Area place list, a0_terms.NOT_NAMES) plus a STOP
list of capitalised sentence openers and titles; what remains, if it still
looks like a name, becomes [name]. Earlier layers win overlaps. Redacting the
output again changes nothing (placeholders match no rule).

Public API for the site exporter and the stance report:

  redact_text(text) -> (redacted_text, replacements)
      replacements: [{"rule", "start", "end", "original", "replacement"}]
      offsets refer to the INPUT text.
  redact_transcript(doc) -> (new_doc, replacements_with_turn_idx)

CLI (see --help):
  redact.py run --all [--audit output/redacted/audit.csv]
                      [--sample-for-review 30 --out output/redacted/review_30.csv]
  redact.py run --episode <episode_id>
  redact.py run --residual-scan           # scan the REDACTED output for a cue + a name-like
                                          # token, capitalised OR not (see also
                                          # scripts/redaction_scan.py, which scans any file or tree)

The audit CSV holds the original spans (names) for Puck's hand check. It is a
temporary file under output/ (never committed); delete it after the check.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "plan"
SRC_DEFAULT = ROOT / "output" / "whisper"
OUT_DEFAULT = ROOT / "output" / "redacted"
QA_DEFAULT = ROOT / "output" / "qa" / "latest.csv"

# 1.3: a `tier` on the given_name layer so the publish path can run it at CORE,
# where a first name that doubles as a Bay Area place stops being protected, and
# spans_honorific() for agenda headings. Both came out of the 2026-10-20 publish
# scan, which found names the cue-anchored scan reports as clean.
VERSION = "redact-1.3"
NAME, ADDRESS, CONTACT = "[name]", "[address]", "[contact]"
PLACEHOLDER_RE = re.compile(r"\[(?:name|address|contact)\]")

# ----------------------------------------------------------------- lexicon ---
# A name token: Capitalised, at least one lower-case letter (so BART, VTA, SB
# are not names), apostrophes / hyphens / inner capitals allowed (O'Brien,
# McDonald, Garcia-Lopez).
CAP = r"[A-Z][a-zA-Z'’\-]*[a-z][a-zA-Z'’\-]*"
CAP_RUN = rf"{CAP}(?:\s+{CAP}){{0,3}}"
# particles that may sit INSIDE a name without a capital: de la Cruz, van der Berg
PARTICLES = {"de", "la", "del", "da", "di", "van", "der", "von", "le", "du"}   # not "al"/"el": Al is a first name, El Cerrito a city

HONORIFICS = [
    "Council\\s?member", "Council\\s?woman", "Council\\s?man", "Council President",
    "Mayor Pro Tem", "Vice Mayor", "Mayor", "Supervisor", "Vice President", "President",
    "Vice Chair(?:woman|man|person)?", "Chair(?:woman|man|person)?",
    "Director", "Commissioner", "Board\\s?member", "Trustee", "Senator", "Assembly\\s?member",
    "Congress(?:woman|man|member)", "General Manager", "City Manager", "City Attorney", "City Clerk",
    "District Secretary", "Chief", "Captain", "Sergeant", "Officer", "Reverend", "Pastor", "Professor",
    "Mr\\.?", "Ms\\.?", "Mrs\\.?", "Dr\\.?", "Miss",
    # all-caps titles: CAP needs a lowercase letter, so these never match as the
    # name slot themselves and are safe to treat purely as cues. Added 2026-10-20
    # for "Impact Report by CEO Howard Chang" in a published agenda heading.
    "CEO", "CFO", "COO",
]
HONORIFIC_RE = "(?:" + "|".join(HONORIFICS) + ")"

STREET_WORDS = (r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Way|Court|Ct|Lane|Ln|Place|Pl|"
                r"Terrace|Ter|Circle|Cir|Highway|Hwy|Parkway|Pkwy|Alley|Loop|Trail|Expressway|Expy|Freeway)")
# words that must not be the "street name" between the number and the type
STREET_STOP = set("""of the a an and in on to for by per percent more less feet foot units unit people homes miles
mile day days year years one two three four five six seven eight nine ten hundred thousand million dollar dollars
minute minutes hour hours week weeks month months time times way""".split())

# capitalised tokens that start sentences, titles, generic nouns: never a name by themselves,
# and trimmed off the ends of a candidate span
STOP = set("""a an and the this that these those it its we i you he she they them our your my me so but or if then
when where what who why how yes no not okay ok all right thank thanks good evening morning afternoon night hi hello
hey welcome please next first second third last item items measure measures agenda motion second vote roll call
public comment comments speaker speakers caller council city county district board committee commission staff
department mayor vice chair chairman chairwoman chairperson president director councilmember councilwoman councilman
supervisor commissioner senator assemblymember assemblywoman assemblyman congresswoman congressman trustee
boardmember member members madam mister sir ma'am mr ms mrs dr honorable clerk manager attorney
secretary general executive officer chief deputy assistant associate senior principal planning housing transit
transportation community development economic finance budget police fire parks recreation library water utilities
works services safety health human resources information technology office program project plan policy
report presentation item update study session workshop hearing meeting minutes consent calendar closed open
ordinance resolution amendment appeal application permit zoning element site area neighborhood downtown
street avenue boulevard road drive way court lane place park bridge station center square plaza building
school university college elementary high middle hospital church temple mosque
north south east west northern southern eastern western central
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday today tonight tomorrow yesterday
state federal regional local national american california bay area
proposition prop bill senate assembly act code section
move moving motion approve approved adopt adopted continue continued
i'm i've i'd i'll we're we've we'd we'll you're you've you'd you'll they're they've they'd they'll
he's he'd he'll she's she'd she'll it's that's there's here's what's who's let's don't didn't doesn't
isn't wasn't weren't won't can't couldn't wouldn't shouldn't haven't hasn't hadn't
mister doctor ma'am mgmt""".split())

# ------------------------------------------- case-insensitive name lexicons ---
# A name token with NO casing signal: letters, apostrophes and hyphens, any case.
ANY_TOK = r"[A-Za-z][A-Za-z'’\-]*"

# Function words, auxiliaries, common verbs and predicate adjectives. On
# capitalised text a leading capital tells a name from a sentence word; on
# lowercase text nothing does, so this list is the whole guard. A token here is
# never part of a name in any position, which is what keeps "i'm here to speak",
# "my name is going to be on the list" and "it's important" out of the output.
FUNCTION_STOP = set("""
a an the and or but nor so if then than that this these those there here where when what which who whom whose why how
i me my mine myself you your yours we us our ours they them their theirs he him his she her hers it its
am is are was were be been being do does did doing done have has had having
can could shall should must would gonna wanna let lets
go goes going went gone come comes coming came get gets getting got gotten make makes making made
say says saying said speak speaks speaking spoke spoken talk talks talking talked tell tells telling told
live lives living lived work works working worked support supports supporting supported
oppose opposes opposing opposed think thinks thinking thought know knows knowing knew known
want wants wanting wanted need needs needing needed like likes liking liked feel feels feeling felt
see sees seeing saw seen look looks looking looked ask asks asking asked
bring brings bringing brought take takes taking took taken give gives giving gave given put puts putting
try tries trying tried hope hopes hoping hoped urge urges urging urged
very really just only also still even too much many more most less least not no never always already
about above across after against along among around at before behind below beneath beside between beyond
by down during for from in inside into near of off on onto out outside over past since through
throughout to toward towards under until up upon with within without because while whether
please thank thanks yes yeah yep nope okay ok hi hello hey good morning evening afternoon night
first second third fourth next last another other same such own both few all any each every some none
one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand million
sorry glad happy sure afraid excited concerned worried grateful proud disappointed confused curious
available unable able ready fine done finished retired here's speaking calling representing resident renter
homeowner neighbor neighbour member parent student teacher volunteer resident's
""".split())

# Given names that are also ordinary English words. Two jobs, both ways round:
# the ci_given_pair layer may not SEED on them (a bare "will", "mark" or "june"
# is the word far more often than the person), but a STRONG self-introduction
# cue overrides that, so they are subtracted from the stop set inside a
# "my name is ..." run -- "my name is will chen" must still redact.
COMMON_NAME_WORDS = set("""will mark may june july august april grant bill art rose faith hope joy sue pat guy
frank chase dawn gene jack ray rich sandy summer autumn winter bob buck bud cliff dale dean dick don drew earl
earnest ernest felix flora forest frances gay glen gordon hazel holly hunter iris ivy jean jewel jim joe john
king knight lane lee lily mae major mason max mercy miles mills minnie misty moss nick noble north olive page
paige pearl penny perry pierce price prince reed reid reign river robin rock rocky rod royal rue rusty sage
sonny star sterling stone stormy sunny tiger tracy trinity true van victor viola violet ward wade wall wells
wilder woody young homer marsh""".split())

_CI_STRONG_STOP_0 = (STOP | FUNCTION_STOP) - COMMON_NAME_WORDS
_CI_STRONG_STOP_N = FUNCTION_STOP - COMMON_NAME_WORDS
_CI_WEAK_STOP_0 = STOP | FUNCTION_STOP
_CI_WEAK_STOP_N = FUNCTION_STOP

# Bay Area places, agencies, institutions, transit names: kept as they are.
BAY_AREA = """
san francisco oakland san jose san josé berkeley richmond san leandro hayward fremont sunnyvale santa rosa san rafael
alviso diridon berryessa milpitas mountain view palo alto redwood city daly city millbrae walnut creek concord dublin
pleasanton livermore antioch pittsburg marin sonoma napa solano alameda contra costa san mateo santa clara petaluma
novato sausalito mill valley cupertino campbell los gatos saratoga gilroy morgan hill emeryville albany el cerrito
vallejo fairfield benicia martinez lafayette orinda moraga danville san ramon union city newark los altos menlo park
burlingame san bruno south san francisco brisbane pacifica half moon bay castro valley rockridge fruitvale macarthur
lake merritt temescal embarcadero mission tenderloin excelsior bayview hunters point treasure island presidio marina
sacramento california bay area silicon valley peninsula east bay north bay south bay tri-valley coyote valley
alum rock willow glen japantown evergreen almaden cambrian santa teresa blossom hill tamien capitol warm springs
irvington niles centerville decoto ashby pleasant hill bay fair coliseum lawrence mathilda moffett park moffett
el camino real stevens creek winchester monterey market valencia van ness geary telegraph broadway international
hegenberger san pablo shattuck university bancroft terra linda northgate canal larkspur corte madera tiburon
santa venetia belmont san carlos foster city atherton woodside portola valley east palo alto los altos hills
sebastopol healdsburg windsor rohnert park cotati sonoma county napa county marin county solano county
santa clara county san mateo county alameda county contra costa county san francisco county
west oakland east oakland north oakland downtown oakland jack london uptown chinatown laurel dimond montclair
piedmont glenview eastmont elmhurst sobrante rodeo crockett hercules pinole el sobrante point richmond
hilltop marina bay iron triangle north richmond san pablo avenue macdonald avenue cutting boulevard
santa rosa avenue roseland railroad square coddingtown fountaingrove bennett valley rincon valley oakmont
petaluma boulevard lakeville east washington mcdowell
balboa park glen park colma west portal civic center powell montgomery 16th street 24th street 12th street
19th street pittsburg bay point el cerrito del norte el cerrito plaza north berkeley downtown berkeley south hayward
west dublin dublin pleasanton north concord oakland airport san francisco airport 4th and king 22nd street bayshore
hillsdale hayward park california avenue san antonio college park san jose diridon san martin coleman avenue coleman
mcallister folsom howard harrison bryant brannan townsend king street divisadero fillmore castro noe valley
potrero dogpatch soma south of market north beach russian hill nob hill pacific heights richmond district sunset
ocean beach lake merced parkmerced stonestown balboa ingleside visitacion valley portola bernal heights
sonoma marin area rail transit golden gate highway 101 interstate 880 interstate 580 interstate 680 interstate 280
highway 37 highway 12 highway 24 highway 4 highway 1 pacific coast
yosemite tahoe los angeles san diego fresno stockton modesto tracy manteca merced
nevada oregon washington texas new york chicago seattle portland boston
bart caltrain vta muni sfmta ac transit samtrans smart golden gate transit marin transit mtc abag hcd amtrak
capitol corridor ace altamont clipper link21 valley link weta ferry caltrans fta fhwa
sierra club yimby greenbelt alliance spur transform urban habitat working partnerships habitat for humanity
midpen eden housing bridge housing mercy housing related google apple meta tesla nvidia stanford uc berkeley
san jose state santa clara university sonoma state dominican kaiser sutter
sb 79 sb 9 sb 35 sb 423 ab 2011 ab 2097 sb 63 rhna ceqa ami adu hud fema epa dot
granicus swagit zoom youtube facebook twitter nextdoor
cal cities
cesar chavez martin luther king mlk rosa parks frederick douglass thurgood marshall
john f kennedy jfk george washington thomas jefferson abraham lincoln benjamin franklin
susan b anthony harriet tubman sojourner truth
"""
# Historical namesakes recur as street/school names ("Cesar Chavez Middle School",
# "MLK Way"); they identify a building, not a meeting participant, so the given_name
# layer must not treat them as a speaker (CALIBRATION.md 2026-10-17 #2). This is a
# short, Bay-Area-skewed list, not exhaustive, and does not cover place names outside
# the Bay Area allowlist's own scope (a SoCal comparison like "Santa Monica" is not
# protected and can still be redacted; a known, documented gap, not fixed here).


def _plan_lines(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _load_allowlist():
    """Two tiers of lower-cased phrases that are never a person.

    core  : multi-word place / agency / body phrases, every whole term of
            plan/vocab.tsv (the things actually said in these meetings), and
            agency acronyms. Applied to EVERY candidate, including strong
            self-introduction cues.
    full  : core plus every single token of the Bay Area list, the keyword and
            frame terms and a0_terms.NOT_NAMES. Applied to the weak layers
            (spaCy PERSON, "thank you, X", "next speaker X"), where a lone
            "Campbell" or "Hill" is more often the place than a person.
    The split exists because many Bay Area places are also surnames
    (Lawrence, Campbell, Richmond, Hayward): "my name is Lawrence Campbell"
    must still be redacted; "Thank you, Campbell" (the city) should not.
    """
    intro, core, full = set(), set(), set()
    for line in BAY_AREA.strip().splitlines():
        toks = line.split()
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                full.add(g)
                if n > 1:
                    intro.add(g)
    # channels.tsv body slugs
    for line in _plan_lines(PLAN / "channels.tsv"):
        intro.add(line.split("\t")[0].strip().lower())
    # vocab.tsv: "<body>\t<comma separated vocabulary>" — bodies, dais titles, agencies, places
    for line in _plan_lines(PLAN / "vocab.tsv"):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            for term in re.split(r"[,.;]", parts[1]):
                term = term.strip().lower()
                if term:
                    core.add(term)
                    if " " in term:
                        intro.add(term)
                    for t in term.split():
                        if len(t) > 2:
                            full.add(t)
    for kw in PLAN.glob("keywords_*.txt"):
        for line in _plan_lines(kw):
            full.add(line.lower())
    for line in _plan_lines(PLAN / "frames.yaml"):
        if line.startswith("- "):
            full.add(line[2:].strip().lower())
    try:
        sys.path.insert(0, str(ROOT / "analysis"))
        import a0_terms  # noqa: E402
        full |= set(a0_terms.NOT_NAMES)
    except Exception:
        pass
    # agency / programme acronyms and brands are never a person in any tier
    intro |= set("""bart caltrain vta muni sfmta samtrans smart mtc abag hcd amtrak ace clipper link21 weta caltrans
    fta fhwa rhna ceqa ami adu hud fema epa yimby spur midpen granicus swagit zoom youtube google apple meta tesla
    nvidia stanford kaiser sutter""".split())
    intro.discard(""); core.discard(""); full.discard("")
    core |= intro
    full |= core
    return intro, core, full


_ALLOW = None


def allowlist(tier=2):
    """tier: INTRO (0) < CORE (1) < FULL (2); see _load_allowlist."""
    global _ALLOW
    if _ALLOW is None:
        _ALLOW = _load_allowlist()
    return _ALLOW[tier]


def _norm(tok):
    return tok.lower().strip(".,;:!?\"'()").replace("’", "'")


def protected_mask(tokens, tier=2):
    """True for every token covered by an allowlisted 1-3 gram (longest first)."""
    allow = allowlist(tier)
    low = [_norm(t) for t in tokens]
    mask = [False] * len(tokens)
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            if " ".join(low[i:i + n]) in allow:
                for j in range(i, i + n):
                    mask[j] = True
    return mask


def name_runs(span_text, span_start, tier=2):
    """Split a candidate span into the maximal runs of tokens that are still
    plausibly a name after allowlist / stop-word protection. Returns
    [(abs_start, abs_end, text)]."""
    toks = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", span_text)]
    if not toks:
        return []
    words = [t[2] for t in toks]
    prot = protected_mask(words, tier)
    # In the INTRO tier (self-introductions) only the FIRST token is checked
    # against STOP: "I am Dana Park" is a name even though "park" is a stop
    # word. Multi-word place phrases and acronyms (the INTRO allowlist) still
    # protect anywhere in the span.
    keep = []
    for i, (s, e, w) in enumerate(toks):
        core = _norm(w)
        is_cap = re.fullmatch(CAP, w.strip(".,;:!?\"'()")) is not None
        particle = core in PARTICLES
        stop = core in STOP and (tier != INTRO or i == 0)
        ok = (not prot[i]) and (not stop) and (is_cap or particle)
        if "[" in w or "]" in w:
            ok = False
        keep.append(ok)
    runs, i = [], 0
    while i < len(toks):
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < len(toks) and keep[j]:
            j += 1
        # a run must contain at least one real capitalised token, and may not start or end on a particle
        a, b = i, j
        while a < b and _norm(words[a]) in PARTICLES:
            a += 1
        while b > a and _norm(words[b - 1]) in PARTICLES:
            b -= 1
        if a < b:
            s, e = toks[a][0], toks[b - 1][1]
            # strip trailing punctuation from the run
            while e > s and span_text[e - 1] in ".,;:!?\"'()":
                e -= 1
            if e > s:
                runs.append((span_start + s, span_start + e, span_text[s:e]))
        i = j
    return runs


# ------------------------------------------------------------------- rules ---
def _cue(pattern):
    # case-SENSITIVE overall: the name part must be capitalised; the cue words
    # are wrapped in (?i:...) by ci() below.
    return re.compile(pattern)


def ci(s):
    return f"(?i:{s})"


# Each cue: (rule name, compiled regex, tier); group "x" is the candidate name
# span, re-checked by name_runs against the allowlist tier:
#   INTRO  self-introduction: only multi-word places, acronyms and STOP words protect
#   CORE   honorific: INTRO plus the body vocabulary (plan/vocab.tsv terms)
#   FULL   weak cues and spaCy PERSON: everything, including single place tokens
INTRO, CORE, FULL = 0, 1, 2
_XC = rf"(?P<x>{CAP}(?:\s+(?:{CAP}|de|la|del|van|der|von))*)"
_C_NAME_IS = ci(r"(?:my\s+name\s+(?:is|'s)|name\s+for\s+the\s+record\s+is)")
_C_IM_AFTER = ci(r"(?:,|\.|and\b|with\b|from\b|here\b|speaking\b|calling\b|a\s+resident\b|representing\b|on\s+behalf\b)")
_C_THIS_IS = ci(r"this\s+is")
_C_THIS_AFTER = ci(r"(?:,|speaking\b|with\b|from\b|calling\b|here\b|on\s+behalf\b|representing\b|a\s+resident\b)")
_C_RESIDENT = ci(r"(?:a\s+|an?\s+)?(?:long[\s-]?time\s+|lifelong\s+|proud\s+|\d+[\s-]year\s+)?resident\b")
_C_RECORD = ci(r"for\s+the\s+record,?\s+(?:it's\s+|it\s+is\s+|this\s+is\s+)?")
_C_SPEAKER = ci(r"(?:(?:next|first|last|following|our\s+next)\s+speaker(?:\s+is|\s+will\s+be|\s+up\s+is)?[,:]?"
                r"|followed\s+by|speaker\s+(?:is|number\s+\d+[,:]?))")
_C_THANKS = ci(r"thank\s+you(?:\s+(?:very|so)\s+much)?")
CUES = [
    ("cue_name_is",   _cue(rf"\b{_C_NAME_IS}\s+{_XC}"), INTRO),
    ("cue_im",        _cue(rf"\b(?:I'm|I\s+am|i'm)\s+{_XC}(?=\s*{_C_IM_AFTER})"), INTRO),
    ("cue_this_is",   _cue(rf"\b{_C_THIS_IS}\s+{_XC}(?=\s*{_C_THIS_AFTER})"), INTRO),
    ("cue_resident",  _cue(rf"\b{_XC}(?=\s*,\s*{_C_RESIDENT})"), INTRO),
    ("cue_honorific", _cue(rf"\b{ci(HONORIFIC_RE)}\s+{_XC}"), CORE),
    ("cue_record",    _cue(rf"\b{_C_RECORD}{_XC}"), INTRO),
    ("cue_speaker",   _cue(rf"\b{_C_SPEAKER}\s+{_XC}"), FULL),
    ("cue_thank_you", _cue(rf"\b{_C_THANKS},\s+{_XC}"), FULL),
]

ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}[A-Za-z]?\s+(?:(?:[NSEW]\.?|North|South|East|West)\s+)?"
    rf"(?P<name>(?:[A-Za-z][A-Za-z'\-]*\.?\s+){{1,3}})"
    rf"{STREET_WORDS}\b(?:,?\s*(?:Apt|Apartment|Unit|Suite|Ste|#)\s*\.?\s*[\w-]+)?",
    re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")

_NLP = None


def nlp():
    """spaCy en_core_web_sm with only the NER stack; None when not installed."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "attribute_ruler", "tagger"])
        except Exception:
            _NLP = False
    return _NLP or None


def _spans_cues(text, cap=FULL):
    """`cap` is the loosest allowlist tier any cue may use.

    The weak cues sit at FULL because internally a lone "Campbell" or "Hill" is
    more often the place than the person. On the publish path the caller caps
    them at CORE: at FULL, "followed by John Smith" dropped "John" (a FULL-tier
    allowlist entry, from St. John / John Muir) and published the first name in
    clear. A cue's own tier is never loosened, only tightened."""
    out = []
    for rule, rx, tier in CUES:
        for m in rx.finditer(text):
            for s, e, orig in name_runs(m.group("x"), m.start("x"), min(tier, cap)):
                out.append((s, e, orig, NAME, rule))
    return out


GIVEN_NAMES_PATH = PLAN / "given_names.txt"
_GIVEN = None
_GIVEN_RUN_RE = re.compile(rf"\b{CAP}(?:\s+(?:{CAP}|{'|'.join(sorted(PARTICLES))}))*\b")
# A given-name SEED token sitting right after a CA place prefix ("San Tomas",
# "Santa Barbara", "North Carolina") or within the next two words of a
# street/place/business-name suffix ("Reed Street", "Hunter Storm Project",
# "Marcus and Millichap Company") is a street, neighbourhood or company, not a
# meeting participant, far more often than it is a person in this corpus: a
# hand-read 50-row sample was 52% over-redaction before this filter
# (CALIBRATION.md 2026-10-17 #2). Checked against the SEED token's own
# position, not the whole capitalised run, because the suffix word
# ("Street", "Project"...) is itself capitalised and usually ends up INSIDE
# the run the regex already matched, not after it. This is a context veto,
# not a positive identification of what the phrase IS; it only suppresses
# the given_name layer, which never had cue-level confidence to begin with.
GIVEN_NAME_PREFIX_VETO = {"san", "santa", "saint", "st", "el", "los", "las", "north", "south", "east", "west"}
GIVEN_NAME_SUFFIX_VETO = {
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "drive", "dr", "way", "court", "ct",
    "lane", "ln", "place", "pl", "terrace", "ter", "circle", "cir", "highway", "hwy", "parkway", "pkwy",
    "alley", "loop", "trail", "expressway", "expy", "freeway",
    "hill", "hills", "heights", "valley", "ridge", "grove", "meadows", "commons", "village", "garden", "gardens", "creek",
    "project", "projects", "station", "company", "co", "corporation", "corp", "llc", "inc", "apartments", "apts",
    "development", "partners", "associates", "group", "properties", "coalition", "committee", "commission",
    "center", "centre",
}


def _adjacent_word(text, pos, forward=True, skip=0):
    """The word touching `text[pos]` (forward) or `text[:pos]` (backward),
    normalised; `skip` jumps over that many words first (0 = the nearest)."""
    if forward:
        words = re.findall(r"\S+", text[pos:pos + 40])
    else:
        words = re.findall(r"\S+", text[max(0, pos - 20):pos])[::-1]
    if skip >= len(words):
        return ""
    return _norm(words[skip])


def _load_given_names(path=GIVEN_NAMES_PATH):
    out = {}
    for line in _plan_lines(path):
        parts = line.split("\t")
        if len(parts) >= 3:
            name, _origin, flag = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if name and flag in ("name", "ambiguous"):
                out[name.lower()] = flag
    return out


def given_names():
    global _GIVEN
    if _GIVEN is None:
        _GIVEN = _load_given_names()
    return _GIVEN


def _spans_given_name(text, tier=FULL):
    """A capitalised run containing a lexicon hit, with no cue word and
    (usually) no PERSON entity around it. flag=name: that token alone seeds
    a run. flag=ambiguous: only seeds inside a run of >=2 capitalised
    tokens, so "Grant funding was approved" keeps Grant but "Grant Hoffman"
    is a candidate.

    `tier` is the allowlist tier handed to name_runs. It defaults to FULL,
    where a lone token that is also a Bay Area place is protected, because
    this layer has the least confidence of any here. The publish path passes
    CORE instead: at FULL a first name that doubles as a place name survives
    into published text ("Carlos" <- San Carlos, "Howard", "Clara"), and for
    text that leaves this machine Puck's 2026-10-20 rule outranks the
    over-redaction risk. CORE still protects the multi-word place, so
    "San Carlos" keeps its 2-gram protection while bare "Carlos" loses it.

    A seed is often also a Bay Area place name (Clara, Scott, Mateo, June,
    Francisco...) and gets stripped by name_runs' own allowlist/STOP check,
    the same way a protected token is dropped anywhere else in this file.
    When that happens the run must not survive on an unrelated leftover
    capitalised neighbour ("the Clara Gardens project" -> "Gardens" alone is
    not a name, it is what is left after "Clara" is correctly recognised as
    Santa Clara): keep a returned run only if it still overlaps at least one
    of the ORIGINAL seed token positions (CALIBRATION.md 2026-10-17 #2)."""
    out = []
    lex = given_names()
    if not lex:
        return out
    for m in _GIVEN_RUN_RE.finditer(text):
        span_text = m.group(0)
        tok_matches = list(re.finditer(r"\S+", span_text))
        flags = [lex.get(_norm(tm.group(0))) for tm in tok_matches]
        seed_idx = [i for i, f in enumerate(flags) if f == "name" or (f == "ambiguous" and len(tok_matches) > 1)]
        if not seed_idx:
            continue
        seed_spans = []
        for i in seed_idx:
            s0, e0 = m.start() + tok_matches[i].start(), m.start() + tok_matches[i].end()
            if _adjacent_word(text, s0, forward=False) in GIVEN_NAME_PREFIX_VETO:
                continue
            # up to 3 words ahead: "Reed Street" (0), "Hunter Storm Project" (1),
            # "Marcus and Millichap Company" (2, "and" between)
            if any(_adjacent_word(text, e0, forward=True, skip=k) in GIVEN_NAME_SUFFIX_VETO for k in (0, 1, 2)):
                continue
            seed_spans.append((s0, e0))
        if not seed_spans:
            continue
        for s, e, orig in name_runs(span_text, m.start(), tier):
            if any(s < se and e > ss for ss, se in seed_spans):
                out.append((s, e, orig, NAME, "given_name"))
    return out


#: the honorific cue alone, pre-compiled. scrub_heading() in
#: scripts/site_export/export_site_data.py needs the "Councilmember X" /
#: "Director X" shape on Title Case agenda headings, where running the whole
#: CUES list backtracks without terminating. This cue is anchored on a literal
#: title word, so it is cheap and it cannot fire on "Housing Element".
#: Titles stack in agenda prose -- "State Senator Dr. Aisha Wahab", "Mayor Pro
#: Tem Dr. Lee" -- so the cue repeats. It deliberately does NOT capture the name:
#: matching the name too would make finditer resume past it, and a heading lists
#: officials back to back ("Councilmember Ramachandran And Council President
#: Jenkins"), where the first run swallows the second cue and the second name
#: then never gets looked at. Cue first, run afterwards, from m.end().
_HONORIFIC_CUE_RE = re.compile(rf"\b{ci(HONORIFIC_RE)}(?:\s+{ci(HONORIFIC_RE)})*\s+")
_NAME_RUN_RE = re.compile(rf"{CAP}(?:\s+(?:{CAP}|{'|'.join(sorted(PARTICLES))}))*")


def spans_honorific(text, tier=CORE):
    """Just the honorific cue: the name that follows a title word.

    Agenda headings name officials in their official capacity ("Report from
    Development Services Director Buizer", "From: Councilmember Houston"), and
    the 2026-10-20 publish scan found 57 of them in docs/data/episodes.json.
    The person layers cannot be run wholesale on a heading -- see
    scrub_heading() -- but this one can."""
    out = []
    for m in _HONORIFIC_CUE_RE.finditer(text):
        rm = _NAME_RUN_RE.match(text, m.end())
        if not rm:
            continue
        run = _cut_at_structure(rm.group(0), text, rm.start())
        if not run:
            continue
        for s, e, orig in name_runs(run, rm.start(), tier):
            out.append((s, e, orig, NAME, "honorific"))
    return out


def _cut_at_structure(run, text, start):
    """Trim a candidate name run where an agenda heading's own structure starts.

    A heading is Title Case throughout, so the capitalised run after a title
    word happily swallows the next structural word: "Senator Jesse Arreguin
    From:" and "Councilmember Houston Recommendation:" each redacted one token
    too many and left a dangling colon. STOP catches "Report" and "Update" but
    not "From", "Subject" or "Recommendation" -- "from" is in FUNCTION_STOP
    instead, and a word that the heading follows with a colon is a label.

    The colon is tested against `text`, not `run`: CAP cannot match ":", so the
    punctuation always falls outside the matched run."""
    toks = list(re.finditer(r"\S+", run))
    for i, t in enumerate(toks):
        labelled = text[start + t.end():start + t.end() + 1] == ":"
        if _norm(t.group(0)) in FUNCTION_STOP or labelled:
            return run[:toks[i - 1].end()] if i else ""
    return run


# ------------------------------------------- case-insensitive name layers ----
# Strong cues: the name slot after them is a name or the sentence is not English.
_CI_STRONG = re.compile(r"\b(?:my\s+name(?:\s+is|'s|’s)\s+|name\s+for\s+the\s+record\s+is\s+|name\s+is\s+"
                        r"|call\s+me\s+|for\s+the\s+record,?\s+(?:it'?s\s+|it\s+is\s+|this\s+is\s+)?)",
                        re.IGNORECASE)
# Weak cues: common enough in ordinary speech that a hit alone means nothing, so
# _spans_ci_cues only keeps a run that is also anchored (lexicon hit, or one of
# the self-introduction context words right after it).
_CI_WEAK = re.compile(r"\b(?:i'm|i’m|i\s+am|this\s+is|it's|it’s)\s+", re.IGNORECASE)
_CI_CUES = [("ci_name_is", _CI_STRONG, True), ("ci_intro", _CI_WEAK, False)]
# Context words that follow a self-introduction. The capitalised cue_im /
# cue_this_is also accept a comma here, because a leading capital has already
# done the work of telling a name from a sentence word. Without casing it has
# not: "it's raining, not surprising" is the shape, and a trailing comma made
# the 2026-10-20 export read "it's [name], not surprising" on the landing view.
# So the comma — and the bare conjunctions "and" / "with" / "from", which sit
# after any predicate at all — are not anchors here. What is left either names
# the act of introducing oneself ("speaking", "calling", "representing") or
# gives the speaker's standing, and a one-word run cannot use even those: the
# anchor has to sit after something that looks like a full name, or after a
# token the given-name lexicon vouches for.
_CI_AFTER_RE = re.compile(r"\s*(?:speaking\b|calling\b|representing\b|on\s+behalf\b"
                          r"|a\s+(?:resident|renter|homeowner|neighbor|neighbour|member|parent|student|teacher"
                          r"|volunteer|voter|constituent)\b)", re.IGNORECASE)
_CI_TOK_RE = re.compile(ANY_TOK)
_CLAUSE_END = ".,;:!?"


def _ci_name_run(text, start, strong=True, max_names=3):
    """The 1-3 name-like tokens beginning at `text[start]`, ignoring case.

    Walks forward and stops at the FIRST token that is not name-like; if that is
    the first token, there is no name here and this returns None. Only the run
    touching the cue is ever returned, which is what keeps "i'm here to talk
    about housing" from redacting "housing" once "here" has stopped the run.
    Particles ("de la Cruz") pass through without counting against max_names.
    Returns (start, end, text) with the offsets into `text`."""
    stop_0 = _CI_STRONG_STOP_0 if strong else _CI_WEAK_STOP_0
    stop_n = _CI_STRONG_STOP_N if strong else _CI_WEAK_STOP_N
    toks = [(start + m.start(), start + m.end(), m.group(0))
            for m in re.finditer(r"\S+", text[start:start + 120])][:max_names + 2]
    if not toks:
        return None
    prot = protected_mask([t[2] for t in toks], INTRO)
    kept, names = 0, 0
    for i, (s, e, w) in enumerate(toks):
        bare = w.strip("“”\"'()[]" + _CLAUSE_END)
        core = _norm(w)
        if "[" in w or "]" in w:                       # already a placeholder
            break
        if core in PARTICLES and kept:                 # inside a name only
            kept = i + 1
            continue
        if names >= max_names or prot[i] or not _CI_TOK_RE.fullmatch(bare):
            break
        if core in (stop_0 if i == 0 else stop_n):
            break
        kept, names = i + 1, names + 1
        if w.rstrip("“”\"')") and w.rstrip("“”\"')")[-1] in _CLAUSE_END:
            break                                      # the name does not cross a clause break
    while kept and _norm(toks[kept - 1][2]) in PARTICLES:
        kept -= 1
    if not kept:
        return None
    s, e = toks[0][0], toks[kept - 1][1]
    while e > s and text[e - 1] in _CLAUSE_END + "\"')“”":
        e -= 1
    return (s, e, text[s:e]) if e > s else None


def _ci_anchored(orig):
    """True when the given-name lexicon vouches for the run's first token.

    flag=name on its own is enough to believe a weak cue. flag=ambiguous needs a
    >=2-token run, which is the same bar the capitalised given_name layer sets
    for an ambiguous seed (CALIBRATION.md 2026-10-17 #2) and the reason "i'm
    jose hernandez" redacts while "we are in san jose" does not: half the
    ambiguous flags are there because the name is also a place or a common word.
    An ordinary English word that happens to be a given name is never an anchor
    in either case; a strong cue is what overrides that."""
    first = _norm(re.match(r"\S+", orig).group(0))
    flag = given_names().get(first)
    if first in COMMON_NAME_WORDS:
        return False
    return flag == "name" or (flag == "ambiguous" and len(orig.split()) > 1)


def _spans_ci_cues(text):
    out = []
    for rule, rx, strong in _CI_CUES:
        for m in rx.finditer(text):
            run = _ci_name_run(text, m.end(), strong=strong)
            if run is None:
                continue
            s, e, orig = run
            # A context word alone only vouches for a run that is already shaped
            # like a name: two tokens or more. One token plus "speaking" is
            # "it's pouring speaking" as readily as "i'm dana speaking", and the
            # lexicon is the only thing that can tell those apart.
            if not strong and not (_ci_anchored(orig)
                                   or (len(orig.split()) > 1 and _CI_AFTER_RE.match(text, e))):
                continue
            out.append((s, e, orig, NAME, rule))
    return out


#: single words that make up HONORIFICS, so a stacked title does not redact its
#: own second half ("mr chairman", "vice mayor", "council president").
_HONORIFIC_WORDS = {
    "council", "councilmember", "councilwoman", "councilman", "member", "members", "president",
    "mayor", "pro", "tem", "vice", "chair", "chairman", "chairwoman", "chairperson", "supervisor",
    "director", "commissioner", "boardmember", "board", "trustee", "senator", "assemblymember",
    "assembly", "congressman", "congresswoman", "congressmember", "general", "manager", "city",
    "attorney", "clerk", "district", "secretary", "chief", "captain", "sergeant", "officer",
    "reverend", "pastor", "professor", "mr", "ms", "mrs", "dr", "miss", "madam", "ma'am", "sir",
    "ceo", "cfo", "coo", "staff", "speaker", "deputy", "assistant", "interim", "acting",
}
_CI_HONORIFIC_RE = re.compile(rf"\b(?:{HONORIFIC_RE})\.?\s+", re.IGNORECASE)


def _spans_ci_honorific(text):
    """A title word plus the name after it, ignoring case.

    cue_honorific keys on CAP, so on the lowercase routes it reads nothing: the
    2026-10-20 publish scan found "good evening mayor gonzalez council members"
    in a published quote, where the capitalised form would have been redacted.
    A title directly before a name is as strong a cue as "my name is", so this
    uses the strong stop lists; the guard that matters is _HONORIFIC_WORDS, or
    "mr chairman" and "vice mayor" would redact their own second word.

    Surnames are not in plan/given_names.txt, which is given names only, so
    neither the lexicon layers nor ci_given_pair can reach this shape -- the
    cue is the whole anchor."""
    out = []
    for m in _CI_HONORIFIC_RE.finditer(text):
        # Only where the casing signal is actually absent. If the title itself
        # carries a capital the text is cased, cue_honorific is competent on it,
        # and this layer must stay out of its calibration -- otherwise "The Chair
        # recognizes BART" reads "recognizes" as the chair's surname.
        if any(ch.isupper() for ch in m.group(0)):
            continue
        run = _ci_name_run(text, m.end(), strong=True, max_names=2)
        if run is None:
            continue
        s, e, orig = run
        # Trim at the first title word in ANY position, not just the first:
        # "mayor gonzalez council members" is one name followed by the body it
        # is addressing, and a 2-token run happily took "gonzalez council".
        toks = list(re.finditer(r"\S+", orig))
        for i, t in enumerate(toks):
            if _norm(t.group(0)) in _HONORIFIC_WORDS:
                if i == 0:
                    orig = ""
                    break
                orig = orig[:toks[i - 1].end()]
                e = s + len(orig)
                break
        if not orig:
            continue
        out.append((s, e, orig, NAME, "ci_honorific"))
    return out


def _spans_ci_given_pair(text):
    """A plan/given_names.txt flag=name token plus ONE surname-like token, any
    case, no cue needed: "dana ortiz spoke against the rezoning".

    A pair is required because without casing a bare lowercase "aaron" or "dana"
    is indistinguishable from any other word and the given_name layer's own
    50-row calibration (CALIBRATION.md 2026-10-17 #2) was already 52 % over-
    redaction on bare seeds WITH a capital. Both tokens must clear the FULL
    allowlist and the stop lists, and the seed must clear the same street /
    company context vetoes the capitalised layer uses. KNOWN GAP: a surname that
    is itself an ordinary word or a COMMON_NAME_WORDS entry ("dana park",
    "dana rose") stops the pair, so a cue-less lowercase name of that shape
    still survives this layer; the cue layers above catch it when a cue is
    present."""
    out = []
    lex = given_names()
    if not lex:
        return out
    toks = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]
    prot = protected_mask([t[2] for t in toks], FULL)
    hard = STOP | FUNCTION_STOP
    for i in range(len(toks) - 1):
        (s0, e0, w0), (s1, e1, w1) = toks[i], toks[i + 1]
        c0, c1 = _norm(w0), _norm(w1)
        if lex.get(c0) != "name" or c0 in COMMON_NAME_WORDS or c0 in hard:
            continue
        if prot[i] or prot[i + 1] or "[" in w0 + w1 or "]" in w0 + w1:
            continue
        if not _CI_TOK_RE.fullmatch(w0.strip("“”\"'()" + _CLAUSE_END)):
            continue
        if w0.rstrip("“”\"')") and w0.rstrip("“”\"')")[-1] in _CLAUSE_END:
            continue                                   # seed ends the clause: no surname follows
        if c1 in hard or c1 in COMMON_NAME_WORDS or c1 in GIVEN_NAME_SUFFIX_VETO:
            continue
        if not _CI_TOK_RE.fullmatch(w1.strip("“”\"'()" + _CLAUSE_END)):
            continue
        if _adjacent_word(text, s0, forward=False) in GIVEN_NAME_PREFIX_VETO:
            continue
        if any(_adjacent_word(text, e1, forward=True, skip=k) in GIVEN_NAME_SUFFIX_VETO for k in (0, 1)):
            continue
        e = e1
        while e > s0 and text[e - 1] in _CLAUSE_END + "\"')“”":
            e -= 1
        if e > s0:
            out.append((s0, e, text[s0:e], NAME, "ci_given_pair"))
    return out


def _spans_person(text, doc=None):
    out = []
    model = nlp()
    if model is None:
        return out
    doc = doc if doc is not None else model(text)
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        for s, e, orig in name_runs(ent.text, ent.start_char):
            out.append((s, e, orig, NAME, "person"))
    return out


def _spans_address(text):
    out = []
    for m in ADDRESS_RE.finditer(text):
        mids = [_norm(w) for w in m.group("name").split()]
        if any(w in STREET_STOP for w in mids):
            continue
        out.append((m.start(), m.end(), m.group(0), ADDRESS, "address"))
    return out


def _spans_contact(text):
    out = [(m.start(), m.end(), m.group(0), CONTACT, "contact") for m in EMAIL_RE.finditer(text)]
    out += [(m.start(), m.end(), m.group(0), CONTACT, "contact") for m in PHONE_RE.finditer(text)]
    return out


def find_spans(text, doc=None, publish=False):
    """All replacements for `text`, non-overlapping, earlier layers winning.
    [(start, end, original, replacement, rule)] sorted by start.

    `publish=True` caps every allowlist tier at CORE. The tiers exist to hold
    down over-redaction on text that stays on this machine; for text that goes
    on the public site Puck's 2026-10-20 rule outranks that, and at FULL a name
    that is also a Bay Area place keeps the place's protection and reaches the
    page in clear (Carlos, Howard, John, Clara). CORE still protects the
    multi-word place, so "San Carlos" and "Santa Clara" survive."""
    tier_cap = CORE if publish else FULL
    # the case-insensitive layers go LAST: on capitalised text the layers above
    # keep their 2026-10-17 calibration, and these only add what casing hid.
    layers = [_spans_cues(text, tier_cap), _spans_address(text), _spans_contact(text),
              _spans_person(text, doc), _spans_given_name(text, tier_cap),
              _spans_ci_cues(text), _spans_ci_honorific(text), _spans_ci_given_pair(text)]
    taken, chosen = [], []
    for layer in layers:
        for s, e, orig, rep, rule in sorted(layer):
            if PLACEHOLDER_RE.search(orig):
                continue
            # A later span that STRICTLY CONTAINS a span spaCy took supersedes it.
            # spaCy frequently tags only the surname of "John Smith", and under
            # plain overlap-skip the wider given_name run was then discarded, so
            # the published text kept the first name in clear ("John [name]").
            # Only the `person` layer is superseded this way: it is the one layer
            # whose spans are a model's guess at the extent of a name. Widening
            # over any layer broke "Gaurav leaves" -> ci_given_pair's 2-token run.
            covered = [i for i, (ts, te, ru) in enumerate(taken)
                       if s <= ts and e >= te and ru == "person" and e - s > te - ts]
            if any(s < te and e > ts for i, (ts, te, _ru) in enumerate(taken) if i not in covered):
                continue
            for i in sorted(covered, reverse=True):
                taken.pop(i)
                chosen.pop(i)
            taken.append((s, e, rule))
            chosen.append((s, e, orig, rep, rule))
    chosen.sort()
    return chosen


def apply_spans(text, spans):
    out, pos = [], 0
    for s, e, _orig, rep, _rule in spans:
        out.append(text[pos:s]); out.append(rep); pos = e
    out.append(text[pos:])
    return "".join(out)


def redact_text(text, doc=None, publish=False):
    """(redacted_text, replacements). Offsets in `replacements` refer to the input.
    `publish=True` tightens the allowlist tiers; see find_spans."""
    spans = find_spans(text, doc, publish)
    reps = [{"rule": rule, "start": s, "end": e, "original": orig, "replacement": rep}
            for s, e, orig, rep, rule in spans]
    return apply_spans(text, spans), reps


WORD_RE = re.compile(r"[A-Za-z']+")


def count_words(text):
    return len(WORD_RE.findall(text))


# --------------------------------------------------------------- transcripts --
def turns_of(segments):
    """[(turn_idx, [segment indices])]: maximal runs of consecutive segments by one speaker."""
    turns, cur, cur_spk = [], [], None
    for i, s in enumerate(segments):
        spk = s.get("speaker") or "UNLABELLED"
        if cur and spk == cur_spk:
            cur.append(i)
        else:
            if cur:
                turns.append(cur)
            cur, cur_spk = [i], spk
    if cur:
        turns.append(cur)
    return list(enumerate(turns))


def _context(text, s, e, n=6):
    before = WORD_RE.findall(text[:s])[-n:]
    after = WORD_RE.findall(text[e:])[:n]
    return " ".join(before), " ".join(after)


def redact_transcript(doc, episode_id=None, publish=False):
    """Redact every segment of a transcript JSON (dict). Returns (new_doc, audit_rows).
    The turn text is the segments joined by one space; a span that crosses a
    segment boundary leaves the placeholder in the first segment.

    `publish` reaches find_spans; the two site exporters pass True."""
    segs = doc.get("segments") or []
    new_segs = [dict(s) for s in segs]
    roles = (doc.get("roles") or {})
    by_rule, audit = Counter(), []
    model = nlp()
    turn_list = turns_of(segs)
    texts, offsets = [], []
    for _ti, idxs in turn_list:
        parts, offs, pos = [], [], 0
        for i in idxs:
            t = segs[i].get("text") or ""
            offs.append((i, pos, pos + len(t)))
            parts.append(t); pos += len(t) + 1
        texts.append(" ".join(parts)); offsets.append(offs)
    docs = model.pipe(texts, batch_size=32) if model is not None else [None] * len(texts)
    words = 0
    for (ti, idxs), text, offs, sdoc in zip(turn_list, texts, offsets, docs):
        words += count_words(text)
        spans = find_spans(text, sdoc, publish)
        if not spans:
            continue
        spk = segs[idxs[0]].get("speaker") or "UNLABELLED"
        role = (roles.get(spk) or {}).get("role", "")
        edits = {i: [] for i in idxs}
        for s, e, orig, rep, rule in spans:
            by_rule[rule] += 1
            b, a = _context(text, s, e)
            audit.append({"episode_id": episode_id or doc.get("episode_id") or doc.get("video_id"),
                          "turn_idx": ti, "speaker": spk, "role": role, "rule": rule, "original": orig,
                          "before": b, "after": a, "replacement": rep})
            first = True
            for i, ls, le in offs:
                if s < le and e > ls:
                    edits[i].append((max(s, ls) - ls, min(e, le) - ls, rep if first else ""))
                    first = False
        for i, ed in edits.items():
            t = segs[i].get("text") or ""
            for ls, le, rep in sorted(ed, reverse=True):
                t = t[:ls] + rep + t[le:]
            new_segs[i]["text"] = re.sub(r"[ \t]{2,}", " ", t).strip()
    new_doc = dict(doc)
    new_doc["segments"] = new_segs
    new_doc["redaction"] = {"version": VERSION, "by_rule": dict(sorted(by_rule.items())),
                            "total": sum(by_rule.values()), "words": words, "n_turns": len(turn_list)}
    return new_doc, audit


# -------------------------------------------------------------------- gating --
def gated_episodes(qa_csv=QA_DEFAULT):
    """Episode transcript files (kind=episode, status OK|WARN) from the QA snapshot."""
    files = []
    with open(qa_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("kind") == "episode" and r.get("status") in ("OK", "WARN"):
                files.append((r["file"], r["status"]))
    return files


def write_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


AUDIT_COLS = ["episode_id", "turn_idx", "speaker", "role", "rule", "original", "before", "after", "replacement"]


def run(files, src, out_dir, audit_path=None, sample_n=0, sample_out=None, seed=20261017, quiet=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    totals, all_audit, zero, words, n = Counter(), [], [], 0, 0
    for fname, _status in files:
        p = src / fname
        if not p.exists():
            print(f"  missing transcript: {fname}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        eid = doc.get("episode_id") or p.stem
        new_doc, audit = redact_transcript(doc, eid)
        write_json_atomic(out_dir / f"{eid}.json", new_doc)
        red = new_doc["redaction"]
        totals.update(red["by_rule"]); words += red["words"]; n += 1
        all_audit.extend(audit)
        if red["total"] == 0:
            zero.append(eid)
        if not quiet:
            print(f"  {eid}: {red['total']} replacements / {red['words']} words")
    summary = {"episodes": n, "words": words, "replacements": sum(totals.values()),
               "by_rule": dict(sorted(totals.items())), "episodes_zero": len(zero), "zero_ids": zero}
    if audit_path:
        audit_path = Path(audit_path); audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=AUDIT_COLS, extrasaction="ignore")
            w.writeheader(); w.writerows(all_audit)
        summary["audit"] = str(audit_path)
    if sample_n and sample_out:
        rng = random.Random(seed)
        rows = rng.sample(all_audit, min(sample_n, len(all_audit)))
        sample_out = Path(sample_out); sample_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=AUDIT_COLS + ["verdict", "note"], extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(dict(r, verdict="", note=""))
        summary["sample"] = str(sample_out)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# ------------------------------------------------------------ residual scan ---
# The same cue words, followed by a Capitalised token that is NOT a placeholder.
RESIDUAL_CUES = [
    ("name_is", _cue(rf"\b{_C_NAME_IS}\s+(?P<x>{CAP})")),
    ("im", _cue(rf"\bI'm\s+(?P<x>{CAP})(?=\s*{ci(r'(?:,|and\b|with\b|from\b)')})")),
    ("this_is", _cue(rf"\b{_C_THIS_IS}\s+(?P<x>{CAP})(?=\s*{ci(r'(?:,|speaking\b|with\b|from\b)')})")),
    ("resident", _cue(rf"\b(?P<x>{CAP})(?=\s*,\s*{ci(r'(?:a\s+)?(?:long[\s-]?time\s+)?resident\b')})")),
    ("honorific", _cue(rf"\b{ci(HONORIFIC_RE)}\s+(?P<x>{CAP})")),
    ("record", _cue(rf"\b{_C_RECORD}(?P<x>{CAP})")),
]


def ci_cue_leaks(text, weak=False):
    """[(cue, start, end, token)]: a self-introduction cue in REDACTED text still
    followed by a name-like token, CAPITALISED OR NOT.

    This is the check docs/REPORT.md's old scan could not make: RESIDUAL_CUES
    below all require a leading capital, so on lowercase text it returned 0 by
    construction, and reported "0 residual" over 57 live leaks on 2026-10-20.
    A check that cannot fail is not evidence. `weak=False` reports
    only the strong cues ("my name is X", "name is X", "call me X"), where a
    surviving name-like token is a miss and nothing else; weak cues are
    advisory, since "it's important" legitimately matches one."""
    hits = []
    for rule, rx, strong in _CI_CUES:
        if not strong and not weak:
            continue
        for m in rx.finditer(text):
            if PLACEHOLDER_RE.match(text, m.end()):
                continue
            run = _ci_name_run(text, m.end(), strong=strong)
            if run is None:
                continue
            s, e, tok = run
            hits.append((rule, s, e, tok))
    return hits


def residual_scan(out_dir):
    """[(episode_id, turn_idx, cue, token, context)] in REDACTED output: probable misses."""
    hits = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        segs = doc.get("segments") or []
        for ti, idxs in turns_of(segs):
            text = " ".join(segs[i].get("text") or "" for i in idxs)
            eid = doc.get("episode_id") or p.stem
            found = []
            for cue, rx in RESIDUAL_CUES:
                for m in rx.finditer(text):
                    tok = m.group("x")
                    tail = text[m.end("x"):].lstrip()
                    one_more = re.sub(r"^\S+\s*", "", tail, count=1)
                    if PLACEHOLDER_RE.match(tail) or PLACEHOLDER_RE.match(one_more):
                        # backtrack artefact: a multi-word honorific (e.g. "Mayor Pro Tem")
                        # whose last word or two didn't fit {CAP} because the real name right
                        # after it IS a placeholder; the regex fell back to an internal word
                        # ("Pro") as `x`, but the actual name was already redacted. Not a miss.
                        continue
                    if name_runs(tok, 0, INTRO):     # survives allowlist / stop words -> probable miss
                        found.append((cue, m.start("x"), m.end("x"), tok))
            # the same miss seen by both scans (the capitalised cue matches one
            # token, the case-insensitive one the whole run): report it once
            for cue, s, e, tok in ci_cue_leaks(text):
                if not any(s < te and e > ts for _c, ts, te, _t in found):
                    found.append((cue, s, e, tok))
            for cue, s, e, tok in sorted(found, key=lambda r: r[1]):
                b, a = _context(text, s, e)
                hits.append((eid, ti, cue, tok, f"{b} <<{tok}>> {a}"))
    return hits


# --------------------------------------------------------------------- CLI ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="redact transcripts into output/redacted/")
    r.add_argument("--all", action="store_true", help="every kind=episode transcript OK/WARN in the QA snapshot")
    r.add_argument("--episode", action="append", default=[], help="one episode id (repeatable)")
    r.add_argument("--src", type=Path, default=SRC_DEFAULT)
    r.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    r.add_argument("--qa", type=Path, default=QA_DEFAULT)
    r.add_argument("--audit", type=Path, help="CSV: one row per replacement with 6 words of context")
    r.add_argument("--sample-for-review", type=int, default=0, metavar="N", help="N random audit rows for a hand check")
    r.add_argument("--out", type=Path, help="path of the review sample CSV")
    r.add_argument("--seed", type=int, default=20261017)
    r.add_argument("--residual-scan", action="store_true", help="scan the REDACTED output for cue + Capitalised token")
    r.add_argument("--quiet", action="store_true")
    t = sub.add_parser("text", help="redact a string from stdin (debugging)")
    a = ap.parse_args(argv)

    if a.cmd == "text":
        red, reps = redact_text(sys.stdin.read())
        print(red)
        for x in reps:
            print(f"  {x['rule']}: {x['original']!r} -> {x['replacement']}", file=sys.stderr)
        return 0

    files = []
    if a.all:
        files = gated_episodes(a.qa)
    for eid in a.episode:
        files.append((eid if eid.endswith(".json") else f"{eid}.json", "requested"))
    if files:
        if nlp() is None:
            print("WARNING: spaCy en_core_web_sm not available; PERSON layer skipped", file=sys.stderr)
        s = run(files, a.src, a.out_dir, a.audit, a.sample_for_review, a.out, a.seed, quiet=a.quiet)
        print(f"episodes={s['episodes']} words={s['words']} replacements={s['replacements']} "
              f"zero_episodes={s['episodes_zero']}")
        for k, v in s["by_rule"].items():
            print(f"  {k:15s} {v}")
        if a.audit:
            print(f"audit: {s['audit']}")
        if s.get("sample"):
            print(f"review sample: {s['sample']}")
    if a.residual_scan:
        hits = residual_scan(a.out_dir)
        print(f"residual scan: {len(hits)} probable misses")
        for eid, ti, cue, tok, ctx in hits:
            print(f"  {eid} turn {ti} [{cue}] {ctx}")
        return 0
    if not files and not a.residual_scan:
        ap.error("nothing to do: --all, --episode or --residual-scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
