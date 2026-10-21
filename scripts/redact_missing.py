#!/usr/bin/env python3
"""
scripts/redact_missing.py -- run pipeline/redact.py over exactly the episode ids
listed in a file, one per line, leaving every existing output/redacted/*.json
untouched. `redact.py run --all` rewrites the whole directory; the S1 stance
frame is joined against the files it wrote, so a top-up must only add.

    .venv/bin/python scripts/redact_missing.py output/stance_s2/to_redact.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import redact  # noqa: E402

SRC = ROOT / "output" / "whisper"
OUT = ROOT / "output" / "redacted"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    ids = [x.strip() for x in Path(argv[0]).read_text().splitlines() if x.strip()]
    skipped = [e for e in ids if (OUT / f"{e}.json").exists()]
    todo = [e for e in ids if not (OUT / f"{e}.json").exists()]
    print(f"{len(ids)} ids, {len(skipped)} already redacted (skipped), {len(todo)} to do", flush=True)
    if redact.nlp() is None:
        print("REFUSED: spaCy en_core_web_sm unavailable; the PERSON layer would be skipped "
              "and these files would not match the S1 ones", file=sys.stderr)
        return 2
    s = redact.run([(f"{e}.json", "requested") for e in todo], SRC, OUT, quiet=True)
    print(f"episodes={s['episodes']} words={s['words']} replacements={s['replacements']} "
          f"zero_episodes={s['episodes_zero']}", flush=True)
    for k, v in sorted(s["by_rule"].items()):
        print(f"  {k:15s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
