#!/usr/bin/env python3
"""
scripts/validate_classifier.py -- re-label the T0 candidate windows with a
second model and score it against the T0 instrument (claude-sonnet-4-6,
prompt t0-a4a8e55760). T2 part 1 (2026-09-22), plan/T2-PLAN.md decision rule.

The 198 windows come from output/t0/candidates.csv (ids, bounds, Sonnet
labels); their text is rebuilt exactly as T0 built it -- the caption cues
starting inside [pad_start_s, pad_end_s) of the meeting's .en.vtt -- so both
models see byte-identical input. Calls are direct (198 is small), one row per
window appended to output/t2/validate/<model>/<video_id>.jsonl as it lands
(resume-safe), and the run stops before the next call could pass --cap-usd.

Report: agreement overall, per Sonnet label, on the binary about_* vs not,
Cohen's kappa, the 5x5 confusion matrix, cost and cache behaviour ->
output/t2/validate/<model>/report.md + report.json.

    python3 scripts/validate_classifier.py --model claude-haiku-4-5-20251001 --cap-usd 3
    python3 scripts/validate_classifier.py --model claude-haiku-4-5-20251001 --score-only
"""
import argparse, csv, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state  # noqa: E402
import classify  # noqa: E402
import segment as sg  # noqa: E402

T0 = ROOT / "output" / "t0"
CANDS = T0 / "candidates.csv"


def vtt_for(vid):
    for p in (T0 / vid / f"{vid}.en.vtt", ROOT / "output" / "subs" / f"{vid}.en.vtt"):
        if p.exists():
            return p
    return None


def load_windows():
    rows = list(csv.DictReader(CANDS.open(newline="", encoding="utf-8")))
    segs = {}
    out = []
    for r in rows:
        vid = r["video_id"]
        if vid not in segs:
            p = vtt_for(vid)
            if not p:
                sys.exit(f"no caption file for {vid}")
            segs[vid] = sg.read_segments(p)
        text = classify.window_text(segs[vid], float(r["pad_start_s"]), float(r["pad_end_s"]))
        out.append({"id": r["id"], "video_id": vid, "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                    "pad_start_s": float(r["pad_start_s"]), "pad_end_s": float(r["pad_end_s"]), "words": int(r["words"] or 0),
                    "ref_label": r["label"], "ref_model": r["model"], "ref_prompt_version": r["prompt_version"], "text": text})
    return out


def load_done(out_dir):
    done = {}
    for p in sorted(Path(out_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("label"):
                    done[rec["id"]] = rec
    return done


def run_calls(windows, model, out_dir, cap_usd, log=print):
    import anthropic
    client = anthropic.Anthropic()
    done = load_done(out_dir)
    spent = sum(r.get("cost_usd") or 0 for r in done.values())
    todo = [w for w in windows if w["id"] not in done]
    pv = classify.prompt_version_for(model)
    log(f"{model} ({pv}): {len(done)} done, {len(todo)} to call, ${spent:.4f} spent so far, cap ${cap_usd}")
    for w in todo:
        worst = classify.worst_case_usd(w["text"], model)
        if spent + worst > cap_usd:
            log(f"STOP: ${spent:.4f} spent, next call could reach ${cap_usd}")
            return False
        t0 = time.time()
        try:
            res, usage = classify.classify_one(client, model, w["text"])
        except Exception as e:                            # noqa: BLE001
            log(f"  {w['id']}: API error {type(e).__name__}: {str(e)[:160]}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(20)
            continue
        cost = classify.usd(usage, model) if usage else 0.0
        spent += cost
        rec = {**{k: w[k] for k in ("id", "video_id", "start_s", "end_s", "pad_start_s", "pad_end_s", "words", "ref_label")},
               "label": res["label"] if res else None, "rationale": res["rationale"][:200] if res else None,
               "contains_instructions": res.get("contains_instructions") if res else None,
               "model": model, "prompt_version": pv,
               "usage": {k: getattr(usage, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                                 "cache_read_input_tokens")} if usage else None,
               "cost_usd": round(cost, 6), "latency_s": round(time.time() - t0, 2), "at": state.now_iso()}
        with (Path(out_dir) / f"{w['video_id']}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"  {w['id']} ref {w['ref_label']:13} -> {rec['label']:13} ${cost:.4f} (run ${spent:.4f}) "
            f"cache_read {rec['usage']['cache_read_input_tokens'] if rec['usage'] else '-'}")
    return True


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def score(windows, out_dir, model):
    done = load_done(out_dir)
    labels = classify.LABELS
    pairs = [(w["ref_label"], done[w["id"]]["label"]) for w in windows if w["id"] in done]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    about = lambda l: l in classify.ABOUT
    bin_pairs = [(about(a), about(b)) for a, b in pairs]
    bin_agree = sum(a == b for a, b in bin_pairs)
    per_label = {}
    for lab in labels:
        sub = [(a, b) for a, b in pairs if a == lab]
        per_label[lab] = {"n": len(sub), "agree": sum(a == b for a, b in sub),
                          "agree_frac": round(sum(a == b for a, b in sub) / len(sub), 3) if sub else None,
                          "binary_agree_frac": round(sum(about(a) == about(b) for a, b in sub) / len(sub), 3) if sub else None}
    conf = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        conf[a][b] += 1
    # about_* as the positive class, Sonnet as the reference
    tp = sum(1 for a, b in bin_pairs if a and b); fp = sum(1 for a, b in bin_pairs if not a and b)
    fn = sum(1 for a, b in bin_pairs if a and not b); tn = sum(1 for a, b in bin_pairs if not a and not b)
    # family agreement among windows both call about_*
    fam = lambda l: {"about_density": {"density"}, "about_transit": {"transit"}, "about_both": {"density", "transit"}}.get(l, set())
    both_about = [(a, b) for a, b in pairs if about(a) and about(b)]
    fam_agree = sum(fam(a) == fam(b) for a, b in both_about)
    recs = list(done.values())
    cost = sum(r.get("cost_usd") or 0 for r in recs)
    cache_reads = sum(1 for r in recs if (r.get("usage") or {}).get("cache_read_input_tokens"))
    lat = sorted(r.get("latency_s") or 0 for r in recs)
    rep = {"model": model, "prompt_version": classify.prompt_version_for(model), "reference": "claude-sonnet-4-6 / t0-a4a8e55760",
           "n_windows": len(windows), "n_scored": n, "agreement": round(agree / n, 4) if n else None,
           "kappa": round(kappa(pairs), 4) if n else None,
           "binary_about_agreement": round(bin_agree / n, 4) if n else None, "binary_kappa": round(kappa(bin_pairs), 4) if n else None,
           "about_recall_vs_sonnet": round(tp / (tp + fn), 4) if tp + fn else None,
           "about_precision_vs_sonnet": round(tp / (tp + fp), 4) if tp + fp else None,
           "binary_confusion": {"sonnet_about&haiku_about": tp, "sonnet_about&haiku_not": fn, "sonnet_not&haiku_about": fp, "sonnet_not&haiku_not": tn},
           "family_agreement_when_both_about": {"n": len(both_about), "agree": fam_agree,
                                                "frac": round(fam_agree / len(both_about), 4) if both_about else None},
           "per_reference_label": per_label, "confusion": conf,
           "cost_usd": round(cost, 4), "calls_with_cache_read": cache_reads, "flagged_instructions": sum(1 for r in recs if r.get("contains_instructions")),
           "latency_s_median": lat[len(lat) // 2] if lat else None,
           "about_minutes_ref": round(sum(w["end_s"] - w["start_s"] for w in windows if about(w["ref_label"])) / 60, 1),
           "about_minutes_model": round(sum(w["end_s"] - w["start_s"] for w in windows if w["id"] in done and about(done[w["id"]]["label"])) / 60, 1)}
    return rep, pairs


def decision(rep):
    b = rep["binary_about_agreement"] or 0
    if b >= 0.90:
        return f"binary agreement {b:.3f} >= 0.90 -> Haiku 4.5 is the T2 instrument (Sonnet stays the T0/T1 reference)"
    if b >= 0.80:
        return f"binary agreement {b:.3f} in [0.80, 0.90) -> Haiku 4.5 for T2 with a Sonnet re-check of its mention_only / borderline rows"
    return f"binary agreement {b:.3f} < 0.80 -> Sonnet 4.6 via the Batches API for T2 (tell Puck the cost)"


def write_report(rep, out_dir):
    labels = classify.LABELS
    lines = [f"# Classifier validation: {rep['model']} vs {rep['reference']}", "",
             f"{rep['n_scored']} of {rep['n_windows']} T0 windows scored; ${rep['cost_usd']:.4f}; "
             f"{rep['calls_with_cache_read']} calls had a cache read; median latency {rep['latency_s_median']} s; "
             f"{rep['flagged_instructions']} windows flagged as containing instructions.", "",
             f"- exact agreement (5 labels): **{rep['agreement']}** (kappa {rep['kappa']})",
             f"- about_* vs not: **{rep['binary_about_agreement']}** (kappa {rep['binary_kappa']}); "
             f"recall of Sonnet's about_* {rep['about_recall_vs_sonnet']}, precision {rep['about_precision_vs_sonnet']}",
             f"- family (density / transit / both) agreement when both say about_*: {rep['family_agreement_when_both_about']['agree']}/"
             f"{rep['family_agreement_when_both_about']['n']} = {rep['family_agreement_when_both_about']['frac']}",
             f"- about-minutes over the 198 windows: Sonnet {rep['about_minutes_ref']}, {rep['model']} {rep['about_minutes_model']}",
             f"- **decision rule (plan/T2-PLAN.md): {decision(rep)}**", "",
             "## Per Sonnet label", "", "| Sonnet label | n | same label | about/not same |", "|---|---|---|---|"]
    for lab in labels:
        p = rep["per_reference_label"][lab]
        lines.append(f"| {lab} | {p['n']} | {p['agree']} ({p['agree_frac']}) | {p['binary_agree_frac']} |")
    lines += ["", "## Confusion (rows = Sonnet, columns = " + rep["model"] + ")", "",
              "| Sonnet \\ model | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for a in labels:
        lines.append(f"| {a} | " + " | ".join(str(rep["confusion"][a][b]) for b in labels) + " |")
    (Path(out_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(out_dir) / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--cap-usd", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else ROOT / "output" / "t2" / "validate" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    if a.max:
        windows = windows[:a.max]
    log = lambda m: print(f"{state.now_iso()} {m}", flush=True)
    if not a.score_only:
        state.load_env()
        run_calls(windows, a.model, out_dir, a.cap_usd, log)
    rep, _ = score(windows, out_dir, a.model)
    print(write_report(rep, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
