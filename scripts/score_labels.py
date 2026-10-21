#!/usr/bin/env python3
"""
scripts/score_labels.py -- precision / recall of the window classifier against
blind hand labels (PROPOSAL-transit.md 4 "T0 acceptance"; T0-TESTPLAN.md test 6).

    .venv/bin/python scripts/score_labels.py output/t0/label_sheet.csv output/t0/label_key.csv

The sheet is the file Puck labelled (column `your_label`, one of about_density,
about_transit, about_both, mention_only, no; blank rows are skipped and
counted). The key is the model's label per id, written by
classify_windows.py --make-sheet and not opened before labelling.

Reported per family, where "density-positive" means about_density or about_both
and "transit-positive" means about_transit or about_both:
  precision = model-positive windows that the human also marked positive
  recall    = human-positive windows the model also marked positive
plus exact 5-way agreement, Cohen's kappa, and the confusion matrix. The gate in
the proposal is recall >= 0.9 and precision >= 0.8 on about_* per family.
"""
import argparse, csv, json, sys
from collections import Counter

LABELS = ["about_density", "about_transit", "about_both", "mention_only", "no"]
FAMILY_POS = {"density": {"about_density", "about_both"}, "transit": {"about_transit", "about_both"},
              "any": {"about_density", "about_transit", "about_both"}}


def read(path, col):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            v = (r.get(col) or "").strip().lower()
            if v:
                out[r["id"]] = v
    return out


def prf(pairs, positive):
    tp = sum(1 for h, m in pairs if h in positive and m in positive)
    fp = sum(1 for h, m in pairs if h not in positive and m in positive)
    fn = sum(1 for h, m in pairs if h in positive and m not in positive)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "n_human_pos": tp + fn, "n_model_pos": tp + fp}


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for h, m in pairs if h == m) / n
    ch, cm = Counter(h for h, _ in pairs), Counter(m for _, m in pairs)
    pe = sum(ch[k] * cm[k] for k in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def score(sheet_path, key_path, human_col="your_label"):
    human, model = read(sheet_path, human_col), read(key_path, "model_label")
    bad = {k: v for k, v in human.items() if v not in LABELS}
    ids = [i for i in human if i in model and i not in bad]
    pairs = [(human[i], model[i]) for i in ids]
    res = {"n_scored": len(pairs), "n_unlabelled": len(model) - len(human), "n_bad_labels": len(bad),
           "exact_agreement": (sum(1 for h, m in pairs if h == m) / len(pairs)) if pairs else None,
           "kappa": kappa(pairs), "families": {f: prf(pairs, pos) for f, pos in FAMILY_POS.items()},
           "confusion": {h: {m: sum(1 for hh, mm in pairs if hh == h and mm == m) for m in LABELS} for h in LABELS}}
    return res, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet"); ap.add_argument("key")
    ap.add_argument("--human-col", default="your_label")
    ap.add_argument("--json", help="also write the result here")
    a = ap.parse_args(argv)
    res, bad = score(a.sheet, a.key, a.human_col)
    if bad:
        print(f"ignored {len(bad)} rows with labels outside {LABELS}: {bad}", file=sys.stderr)
    print(f"{res['n_scored']} windows scored, {res['n_unlabelled']} still unlabelled")
    if res["exact_agreement"] is not None:
        print(f"exact 5-way agreement {res['exact_agreement']:.2f}, Cohen's kappa {res['kappa']:.2f}")
    for f, r in res["families"].items():
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
        print(f"{f:8} precision {fmt(r['precision'])}  recall {fmt(r['recall'])}  f1 {fmt(r['f1'])}   "
              f"(human+ {r['n_human_pos']}, model+ {r['n_model_pos']}, tp {r['tp']} fp {r['fp']} fn {r['fn']})")
    print("confusion (rows = human, cols = model):")
    print(" " * 14 + " ".join(f"{l[:8]:>8}" for l in LABELS))
    for h in LABELS:
        print(f"{h:14}" + " ".join(f"{res['confusion'][h][m]:8d}" for m in LABELS))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
