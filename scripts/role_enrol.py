#!/usr/bin/env python3
"""
scripts/role_enrol.py -- T0 test 6, optional step 4: does speaker-embedding
enrolment add anything to the text cues in role_label.py?

WHY
---
pyannote cluster labels are not stable across the clips of one meeting, so
role_label.py decides each clip on its own and leaves clusters with no cue and
no structure as `unknown` (~6 % of talk on b13). The same voices recur across
a body's clips -- the mayor / president opens every item, the clerk calls every
speaker, and (at BART) the same two or three residents comment on every item.
host_enroll.py already showed that a wespeaker anchor separates one voice from
the rest at cos ~0.6. Two uses are tried here, both per body:
  presiding  one anchor from the cluster with the most floor / opener credit
             (the chair or the clerk); every cluster within THRESH of it is
             tagged, and an unknown / structure-only cluster that matches is
             relabelled dais, method `enrol`.
  bank       every cluster role_label.py decided from cue phrases with
             confidence >= BANK_CONF becomes an anchor of its role; an unknown
             cluster whose nearest anchor is >= THRESH and >= GAP clearer than
             the nearest anchor of the other role takes that role, method
             `enrol`. Serial public commenters are caught this way too.
Text labels are never overridden: enrolment only fills `unknown` (and the
presiding anchor may promote a `public` label that came from structure alone,
i.e. a chair who spoke inside public comment). The point of the test is the
delta, so --score prints accuracy before and after against the same truth file.

I/O   reads  output/t0/roles/<clip>.json (role_label.py output) and the clip
             audio output/t0/<video_id>/e<n>.opus
      writes output/t0/roles/embed/<clip>.npz   (cluster embeddings, cached)
             output/t0/roles/enrol/<clip>.json  (same schema as roles/, with
                                                 the enrolment applied)
             output/t0/roles/enrol/summary.json
      never touches roles/<clip>.json, the audio, the state DB or .env
      (HF_TOKEN is read by pipeline/state.setting inside the process, as
      host_enroll.py does; it is not printed).

RUN   ~/.local/share/ytpt/venv-embed/bin/python scripts/role_enrol.py            (~15 s per clip, CPU)
      ~/.local/share/ytpt/venv-embed/bin/python scripts/role_enrol.py --score output/t0/roles/truth_10.json
"""
import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pipeline"))
import host_enroll as he      # noqa: E402  decode / pick_segments / embed_spans / anchor_from
import role_label as rl       # noqa: E402  score()

ROLES = ROOT / "output" / "t0" / "roles"
EMBED = ROLES / "embed"
OUT = ROLES / "enrol"
THRESH, GAP = 0.60, 0.10
BANK_CONF = 0.55
MIN_CLUSTER_S, PER_CLUSTER_S, MIN_SEG_S = 10.0, 120.0, 1.5


def clip_audio(clip):
    vid, e = clip.rsplit("_e", 1)
    p = ROOT / "output" / "t0" / vid / f"e{e}.opus"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def embed_clip(clip, inference):
    """{label: unit mean embedding} for clusters >= MIN_CLUSTER_S; cached."""
    cache = EMBED / f"{clip}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    doc = json.loads((ROOT / "output" / "batches" / "b13" / "out" / f"{clip}.json").read_text())
    segs = doc["segments"]
    _, talk = he.talk_shares(segs)
    wave = he.decode(clip_audio(clip))
    out = {}
    for label, secs in talk.items():
        if label is None or secs < MIN_CLUSTER_S:
            continue
        spans, _ = he.pick_segments(segs, label, PER_CLUSTER_S, MIN_SEG_S)
        vecs = he.embed_spans(inference, wave, spans)
        if len(vecs):
            out[label] = he.anchor_from(vecs)
    EMBED.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def floor_credit(spk):
    """How much 'runs the floor' evidence a cluster has (from the evidence lines)."""
    pts = 0.0
    for e in spk.get("evidence", []):
        if e.startswith(("floorx", "timerx", "opened/closed")):
            pts += float(e.split("+")[1].split()[0])
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", metavar="TRUTH.json")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    import state                                      # noqa: E402
    token = state.setting("HF_TOKEN", "")
    if not token:
        sys.exit("HF_TOKEN not set (.env)")
    t0 = time.monotonic()
    import torch
    from pyannote.audio import Inference, Model
    model = Model.from_pretrained(he.EMBED_MODEL, token=token)
    inference = Inference(model, window="whole", device=torch.device(a.device))
    print(f"model {he.EMBED_MODEL} loaded in {time.monotonic() - t0:.1f} s; thresh {a.thresh} gap {a.gap}", flush=True)

    docs = {p.stem: json.loads(p.read_text()) for p in sorted(ROLES.glob("*_e*.json"))}
    by_body = defaultdict(list)
    for clip, d in docs.items():
        by_body[d["body"]].append(clip)

    embeds = {}
    for clip in docs:
        t1 = time.monotonic()
        embeds[clip] = embed_clip(clip, inference)
        print(f"  {clip:<18} {len(embeds[clip])} clusters embedded ({time.monotonic() - t1:.0f} s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"bodies": {}, "changed": [], "presiding_tags": 0, "bank_tags": 0}
    for body, clips in sorted(by_body.items()):
        # -- presiding anchor: most floor credit, then most talk, across the body's clips
        cands = [(floor_credit(s), s["talk_s"], clip, lab) for clip in clips for lab, s in docs[clip]["speakers"].items()
                 if lab in embeds[clip] and s["role"] == "dais"]
        cands.sort(reverse=True)
        pres = None
        if cands and cands[0][0] > 0:
            _, _, pclip, plab = cands[0]
            pres = embeds[pclip][plab]
        # -- bank: confident cue-phrase labels
        bank = {"dais": [], "public": []}
        for clip in clips:
            for lab, s in docs[clip]["speakers"].items():
                if lab in embeds[clip] and s["method"] == "cue_phrase" and s["confidence"] >= BANK_CONF and s["role"] in bank:
                    bank[s["role"]].append((clip, lab, embeds[clip][lab]))
        body_stats = {"clips": len(clips), "presiding_from": None if pres is None else f"{cands[0][2]}:{cands[0][3]}",
                      "bank_dais": len(bank["dais"]), "bank_public": len(bank["public"]),
                      "presiding_matches": 0, "changed": 0}
        for clip in clips:
            d = json.loads(json.dumps(docs[clip]))          # deep copy
            for lab, s in d["speakers"].items():
                v = embeds[clip].get(lab)
                if v is None:
                    continue
                s["cos_presiding"] = None if pres is None else round(float(v @ pres), 3)
                near = {}
                for role, rows in bank.items():
                    sims = [float(v @ e) for c, l, e in rows if not (c == clip and l == lab)]
                    near[role] = max(sims) if sims else None
                s["cos_bank_dais"], s["cos_bank_public"] = (None if near["dais"] is None else round(near["dais"], 3)), (None if near["public"] is None else round(near["public"], 3))
                old = s["role"]
                if pres is not None and s["cos_presiding"] >= a.thresh:
                    body_stats["presiding_matches"] += 1
                    summary["presiding_tags"] += 1
                    if old == "unknown" or (old == "public" and s["method"] == "structure"):
                        s["role"], s["method"], s["confidence"] = "dais", "enrol", round(s["cos_presiding"], 2)
                        s["evidence"].append(f"enrol: cos {s['cos_presiding']:.3f} to the presiding-officer anchor")
                if s["role"] == "unknown":
                    nd, npub = near["dais"], near["public"]
                    best_role = None
                    if nd is not None and nd >= a.thresh and (npub is None or nd - npub >= a.gap):
                        best_role, best = "dais", nd
                    elif npub is not None and npub >= a.thresh and (nd is None or npub - nd >= a.gap):
                        best_role, best = "public", npub
                    if best_role:
                        s["role"], s["method"], s["confidence"] = best_role, "enrol", round(best, 2)
                        s["evidence"].append(f"enrol: cos {best:.3f} to a known {best_role} voice of this body")
                        summary["bank_tags"] += 1
                if s["role"] != old:
                    body_stats["changed"] += 1
                    summary["changed"].append((clip, lab, old, s["role"], s["talk_s"], s["method"]))
            d["summary"] = rl.summarise(d["speakers"])
            rl.write_new(OUT / f"{clip}.json", d)
        summary["bodies"][body] = body_stats
        print(f"{body:<10} presiding anchor {body_stats['presiding_from']} -> {body_stats['presiding_matches']} matches; "
              f"bank {body_stats['bank_dais']} dais / {body_stats['bank_public']} public voices; {body_stats['changed']} labels changed", flush=True)
    rl.write_new(OUT / "summary.json", summary)
    print(f"\n{len(summary['changed'])} labels changed by enrolment:")
    for clip, lab, old, new, talk, m in summary["changed"]:
        print(f"  {clip:<18} {lab:<10} {old:<8} -> {new:<7} {talk:5.0f} s")

    if a.score:
        truth = {k: v for k, v in json.loads(Path(a.score).read_text()).items() if not k.startswith("_")}
        for name, src in (("text cues only", ROLES), ("with enrolment", OUT)):
            pred = {c: json.loads((src / f"{c}.json").read_text())["speakers"] for c in truth if (src / f"{c}.json").exists()}
            r = rl.score(pred, truth)
            print(f"{name:<16} strict {r['accuracy_strict']:.1%}  lenient {r['accuracy_lenient']:.1%}  unknown {r['unknown_s']:.0f} s  "
                  f"public P {r['public_precision']:.2f} R {r['public_recall']:.2f}")
    print(f"wall {(time.monotonic() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
