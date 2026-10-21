#!/usr/bin/env python
"""redaction_scan.py -- case-INSENSITIVE residual scan over any file or tree.

WHY THIS EXISTS. docs/REPORT.md:182-183 reported "0 residual" from a scan that
looked for a self-introduction cue followed by a CAPITALISED token. The two
lowercase routes (local_asr / mlx-whisper, YouTube auto-captions) emit no
capitals, so that scan returned 0 by construction while 57 self-introductions
over 30 people sat in clear in docs/data/ on 2026-10-20.
A check that cannot fail is not evidence. This one carries no casing signal at
all, and it reads the PUBLISHED artefacts -- JSON, CSV, HTML, Markdown, text --
not the redactor's own output tree, so it tests what a reader can actually see.

WHAT IT REPORTS, per file and in total:
  cue        a self-introduction cue ("my name is X", "name is X", "call me X",
             "for the record, X") still followed by a name-like token that is
             not a [name] placeholder. This is the number that must be 0.
  weak       --weak only: the same for "i'm X" / "this is X" / "it's X".
             ADVISORY: "it's important to note" legitimately matches one, so a
             non-zero count here is a list to read, not a failure.
  address    pipeline/redact.py's own ADDRESS_RE (already case-insensitive)
  contact    phone numbers and e-mail addresses

It never prints the matched text. A leak report that quotes the leak is another
copy of the leak: output is counts, `file:line:col`, the rule that fired and the
match length only.

Usage:
  scripts/redaction_scan.py docs/data                      # a tree
  scripts/redaction_scan.py docs/data/nimby_points.json    # one file
  scripts/redaction_scan.py docs --weak --list 40
  scripts/redaction_scan.py - < some.txt                   # stdin

Exit status: 1 when any cue, address or contact leak is found (0 for weak-only
hits, which are advisory), so `make`/CI can gate a publish on it.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import redact as R  # noqa: E402

#: scanned by default; every text format anything is published in
EXTS = (".json", ".csv", ".tsv", ".md", ".txt", ".html", ".htm", ".js", ".yaml", ".yml")
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}


def line_col(text, pos, starts):
    """1-based (line, col) of `pos`, from a sorted list of line-start offsets."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1, pos - starts[lo] + 1


def scan_text(text, weak=False):
    """[(rule, start, end)] for every leak in `text`. Offsets into `text`.

    Line-by-line would be wrong twice over: a minified JSON export is one line
    of megabytes, and a cue can straddle a newline in a wrapped file. So the
    whole file is one subject and the position is resolved afterwards."""
    hits = [(rule, s, e) for rule, s, e, _tok in R.ci_cue_leaks(text, weak=weak)]
    for m in R.ADDRESS_RE.finditer(text):
        mids = [R._norm(w) for w in m.group("name").split()]
        if not any(w in R.STREET_STOP for w in mids):
            hits.append(("address", m.start(), m.end()))
    for rx in (R.EMAIL_RE, R.PHONE_RE):
        for m in rx.finditer(text):
            hits.append(("contact", m.start(), m.end()))
    # the cue layers can report the same span under a strong and a weak cue
    hits.sort(key=lambda h: (h[1], -h[2]))
    out = []
    for rule, s, e in hits:
        if any(s < te and e > ts for _r, ts, te in out):
            continue
        out.append((rule, s, e))
    return out


def targets(paths, exts):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file() and q.suffix.lower() in exts and not (set(q.parts) & SKIP_DIRS):
                    yield q
        elif p.is_file():
            yield p
        else:
            print(f"  missing: {raw}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="never prints the matched text; counts and file:line:col only")
    ap.add_argument("paths", nargs="+", help="files, directories, or - for stdin")
    ap.add_argument("--weak", action="store_true",
                    help="also report the advisory weak cues (i'm X, this is X, it's X)")
    ap.add_argument("--list", type=int, default=20, metavar="N",
                    help="print at most N locations per rule (0 = counts only)")
    ap.add_argument("--ext", default=",".join(EXTS), help="comma-separated extensions scanned inside a directory")
    a = ap.parse_args(argv)

    exts = tuple(e if e.startswith(".") else "." + e for e in a.ext.split(",") if e)
    by_rule, locs, n_files, n_dirty = {}, {}, 0, 0

    def account(label, text):
        nonlocal n_files, n_dirty
        n_files += 1
        hits = scan_text(text, weak=a.weak)
        if not hits:
            return
        n_dirty += 1
        starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        for rule, s, e in hits:
            by_rule[rule] = by_rule.get(rule, 0) + 1
            ln, col = line_col(text, s, starts)
            locs.setdefault(rule, []).append(f"{label}:{ln}:{col} ({e - s} chars)")

    if a.paths == ["-"]:
        account("<stdin>", sys.stdin.read())
    else:
        for p in targets(a.paths, exts):
            try:
                account(str(p), p.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                print(f"  unreadable: {p}: {exc.strerror}", file=sys.stderr)

    print(f"redaction_scan ({R.VERSION}): {n_files} files scanned, {n_dirty} with at least one leak")
    hard = 0
    for rule in ("ci_name_is", "ci_intro", "address", "contact"):
        n = by_rule.get(rule, 0)
        if rule != "ci_intro":
            hard += n
        if not n and rule == "ci_intro" and not a.weak:
            continue
        tag = "  (advisory)" if rule == "ci_intro" else ""
        print(f"  {rule:12s} {n}{tag}")
        for line in locs.get(rule, [])[:a.list]:
            print(f"      {line}")
        if a.list and len(locs.get(rule, [])) > a.list:
            print(f"      ... {len(locs[rule]) - a.list} more")
    print(f"cue leaks (must be 0): {by_rule.get('ci_name_is', 0)}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
