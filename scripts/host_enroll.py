#!/usr/bin/env python3
"""
scripts/host_enroll.py -- identify the host cluster by speaker-embedding
enrolment instead of "host = most talk time" (PLAN.md 4.6).

WHY
---
pyannote returns anonymous cluster labels per video. The GPU job picks the host
as the cluster with the most speech time, which breaks on interview-heavy
videos (23 of 143 done videos had a host share under 0.55) and on Hasan, whose
impressions split him into several clusters. Speaker embeddings are what
diarization clusters on anyway; one anchor per host, enrolled from a video
where the talk-time heuristic is certain, identifies that host's cluster(s) in
the videos where it is not. Prototyped 2026-09-20 (commit 6c4cb1f), promoted to
a run mode the same day after Puck approved it.

DECISION RULE (per video, over clusters with >= --min-cluster-s of speech)
    cos(c)      = cosine between the mean unit embedding of cluster c (first
                  --test-seconds of its segments >= --min-seg-s) and the anchor
                  of the video's channel
    best        = max cos(c)
    candidates  = {c : cos(c) >= THRESH}            if best >= THRESH
                = {argmax}                          otherwise
    runner-up   = max cos(c) over c NOT in candidates (None if every embedded
                  cluster is a candidate)
    embedding   iff best >= THRESH (0.60) and (runner-up is None or
                best - runner-up >= GAP (0.15))
                -> host_speakers = candidates (all of them: Hasan over-splitting
                   into 3-4 labels is merged into one host label set),
                   host_speaker = the candidate with the highest cos,
                   host_confidence = host_similarity (see below)
    talk_time   otherwise -> host_speakers = [talk-time host], host_speaker and
                host_confidence unchanged (share of speech), host_ambiguous=true
Why these numbers: on the prototype's reference videos the host's own segments
score cos ~0.75-0.85 against their anchor and other people score ~0.2-0.45, so
0.60 sits between the two populations; the 0.15 gap refuses the case where two
voices both sit near the line and the choice would be noise. Both are cheap to
revisit: re-run with --force and different --thresh/--gap.

JSON CONTRACT (keys ADDED to output/whisper/<id>.json; nothing renamed/nested)
    host_method       "embedding" | "talk_time"
    host_similarity   best cos (float)
    host_runner_up    cos of the best non-host cluster (float | null)
    host_speakers     list of cluster labels judged host (>= 1 label)
    host_ambiguous    true when the rule did not decide and talk time was kept
    host_embed_model  the embedding model, for reproducibility
Existing keys:
    host_speaker      kept; under "embedding" = highest-similarity host cluster
    host_confidence   kept; under "embedding" it IS host_similarity (a cosine,
                      typically 0.6-0.95), no longer a share of speech. Under
                      "talk_time" it stays the host's share of speech.
Readers (analysis/load_db.py, meanness_analysis.py, qa_transcripts.py) use
host_speakers when present and fall back to host_speaker.

SAFETY
    * --apply first copies every original to output/whisper_pre_enroll/<id>.json
      (never overwritten on a re-run: the first copy is the pre-enrolment file)
      and writes each modified JSON atomically (tmp + rename).
    * The state DB is opened read-only (mode=ro) and only supplies the done
      list and opus paths. Audio is never touched.
    * A video whose audio cannot be decoded or embedded is left untouched and
      listed at the end.
    * Videos that already carry host_method are skipped unless --force, so a
      killed run can be resumed by running the same command again.

RUN
    ~/.local/share/ytpt/venv-embed/bin/python scripts/host_enroll.py --apply
    ~/.local/share/ytpt/venv-embed/bin/python scripts/host_enroll.py \\
        --test -HhuK2610uQ --test 8_t2btcYv9w          # probe mode: prints only
Anchors: Hasan <- C0LMNHE-OF4, Tucker <- q72s8zwwGg4 (--ref SPEAKER=VIDEO_ID),
cached under output/host_anchors/<speaker>.npy (--recompute-anchors to redo).
~15 s per video on this Mac's CPU: ~35 min for 143 videos; run under nohup.

Dependencies are NOT in .venv on purpose (requirements.txt: torch stays off the
laptop). The scratch venv:
    python3.13 -m venv ~/.local/share/ytpt/venv-embed
    ~/.local/share/ytpt/venv-embed/bin/pip install pyannote.audio==4.0.7 torch==2.8.0 \\
        torchaudio==2.8.0 huggingface-hub==0.36.2
Audio is decoded with the ffmpeg CLI (as pipeline/transcode.py does) because
the torchcodec wheel that pyannote 4 relies on does not load against torch
2.8 on this Mac; pyannote accepts in-memory waveforms, so nothing is lost.
Model: pyannote/wespeaker-voxceleb-resnet34-LM (open, 26.6 MB; HF_TOKEN from
.env via pipeline/state.setting). pyannote/embedding is gated and NOT used.
"""
import argparse, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state  # noqa: E402

SR = 16000
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
THRESH, GAP = 0.60, 0.15
DEFAULT_REFS = ["hasanabi=C0LMNHE-OF4", "tucker=q72s8zwwGg4"]
WHISPER_DIR = ROOT / "output" / "whisper"
BACKUP_DIR = ROOT / "output" / "whisper_pre_enroll"
ANCHOR_DIR = ROOT / "output" / "host_anchors"


def decode(opus_path):
    """Whole file -> float32 mono 16 kHz numpy, via ffmpeg (no torchcodec)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(opus_path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_doc(vid):
    return json.loads((WHISPER_DIR / f"{vid}.json").read_text(encoding="utf-8"))


def write_atomic(path, doc):
    """Same shape as pipeline/gpu_job.write_atomic: compact JSON, tmp + rename."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def open_state_ro():
    """The state DB, read-only. Not state.connect(): that runs the schema DDL."""
    con = sqlite3.connect(f"file:{state.db_path()}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def opus_for(con, vid):
    row = con.execute("SELECT opus_path FROM videos WHERE video_id = ?", (vid,)).fetchone()
    p = Path(row["opus_path"]) if row and row["opus_path"] else ROOT / "output" / "opus" / f"{vid}.opus"
    if not p.exists():
        raise FileNotFoundError(f"{vid}: no opus at {p}")
    return p


def pick_segments(segments, speaker, max_s, min_seg_s):
    """That speaker's segments, in order, until max_s of audio."""
    out, tot = [], 0.0
    for s in segments:
        if s.get("speaker") != speaker:
            continue
        d = float(s["end"]) - float(s["start"])
        if d < min_seg_s:
            continue
        out.append((float(s["start"]), float(s["end"])))
        tot += d
        if tot >= max_s:
            break
    return out, tot


def embed_spans(inference, wave, spans):
    """One L2-normalised embedding per span; (n, dim)."""
    import torch
    vecs = []
    for st, en in spans:
        a, b = int(st * SR), min(int(en * SR), len(wave))
        if b - a < int(0.5 * SR):
            continue
        chunk = torch.from_numpy(wave[a:b]).unsqueeze(0)          # (channel, time)
        e = np.asarray(inference({"waveform": chunk, "sample_rate": SR}), dtype=np.float32).reshape(-1)
        n = np.linalg.norm(e)
        if n > 0:
            vecs.append(e / n)
    return np.stack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)


def anchor_from(vecs):
    m = vecs.mean(axis=0)
    return m / (np.linalg.norm(m) or 1.0)


def talk_shares(segments):
    talk = {}
    for s in segments:
        talk[s.get("speaker")] = talk.get(s.get("speaker"), 0.0) + (float(s["end"]) - float(s["start"]))
    tot = sum(talk.values()) or 1.0
    return {k: v / tot for k, v in talk.items()}, talk


def talk_time_host(segments):
    """The heuristic the GPU job used: longest LABELLED cluster, share of ALL
    speech (same as analysis/load_db.resolve_host). (label, share) or (None, None)."""
    shares, talk = talk_shares(segments)
    labelled = {k: v for k, v in talk.items() if k is not None}
    if not labelled:
        return None, None
    host = max(labelled, key=labelled.get)
    return host, round(shares[host], 3)


def decide(rows, thresh=THRESH, gap=GAP):
    """The decision rule from the docstring. rows: [(label, cos)] for every
    embedded cluster. Returns (method, host labels sorted by cos desc, best cos,
    runner-up cos or None). Pure, so analysis/test_pipeline.py can pin it."""
    if not rows:
        return "talk_time", [], None, None
    ranked = sorted(rows, key=lambda r: -r[1])
    best = ranked[0][1]
    cands = [(l, c) for l, c in ranked if c >= thresh] if best >= thresh else [ranked[0]]
    cand_labels = {l for l, _ in cands}
    others = [c for l, c in ranked if l not in cand_labels]
    runner = max(others) if others else None
    ok = best >= thresh and (runner is None or best - runner >= gap)
    return ("embedding" if ok else "talk_time"), [l for l, _ in cands], float(best), (None if runner is None else float(runner))


def embed_clusters(inference, wave, segments, talk, a):
    """[(label, used_secs, unit vecs)] for every cluster with enough speech."""
    out = []
    for label, secs_total in sorted(talk.items(), key=lambda kv: -kv[1]):
        if label is None or secs_total < a.min_cluster_s:
            continue
        spans, secs = pick_segments(segments, label, a.test_seconds, a.min_seg_s)
        vecs = embed_spans(inference, wave, spans)
        if len(vecs) == 0:
            continue
        out.append((label, secs, vecs))
    return out


def build_anchors(a, con, inference):
    """One anchor per speaker from --ref; cached as output/host_anchors/<spk>.npy."""
    anchors = {}
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    for spec in a.ref:
        spk, _, vid = spec.partition("=")
        cache = ANCHOR_DIR / f"{spk}.npy"
        if cache.exists() and not a.recompute_anchors:
            anchors[spk] = np.load(cache)
            print(f"anchor {spk}: loaded {cache.relative_to(ROOT)}", flush=True)
            continue
        doc = load_doc(vid)
        host, conf = talk_time_host(doc["segments"])
        if host is None or conf is None or conf < a.min_ref_conf:
            sys.exit(f"{vid}: talk-time host {host} @ {conf} is below --min-ref-conf {a.min_ref_conf}; pick another reference")
        t0 = time.monotonic()
        wave = decode(opus_for(con, vid))
        t_dec = time.monotonic() - t0
        spans, secs = pick_segments(doc["segments"], host, a.ref_seconds, a.min_seg_s)
        t0 = time.monotonic()
        vecs = embed_spans(inference, wave, spans)
        t_emb = time.monotonic() - t0
        anc = anchor_from(vecs)
        self_sim = vecs @ anc
        anchors[spk] = anc
        np.save(cache, anc)
        print(f"anchor {spk} <- {vid} host {host} @ {conf}: {len(vecs)} segments, {secs:.0f} s of speech; "
              f"self-consistency cos median {np.median(self_sim):.3f} (min {self_sim.min():.3f}); "
              f"decode {t_dec:.1f} s ({len(wave) / SR / 60:.0f} min audio), embed {t_emb:.1f} s; "
              f"cached {cache.relative_to(ROOT)}", flush=True)
    return anchors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", default=None, metavar="SPEAKER=VIDEO_ID",
                    help=f"anchor sources (default: {' '.join(DEFAULT_REFS)})")
    ap.add_argument("--test", action="append", default=[], metavar="VIDEO_ID", help="print-only probe of these videos")
    ap.add_argument("--apply", action="store_true", help="every done video in the state DB; WRITES output/whisper/*.json")
    ap.add_argument("--force", action="store_true", help="with --apply: redo videos that already have host_method")
    ap.add_argument("--recompute-anchors", action="store_true")
    ap.add_argument("--thresh", type=float, default=THRESH, help="cos >= this = host candidate")
    ap.add_argument("--gap", type=float, default=GAP, help="best - best-non-host must be >= this")
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps")
    ap.add_argument("--ref-seconds", type=float, default=180.0, help="<= this much host speech per anchor")
    ap.add_argument("--test-seconds", type=float, default=120.0, help="<= this much speech per cluster")
    ap.add_argument("--min-seg-s", type=float, default=1.5, help="ignore shorter segments (too little voice)")
    ap.add_argument("--min-cluster-s", type=float, default=10.0, help="skip clusters with less speech than this")
    ap.add_argument("--min-ref-conf", type=float, default=0.8)
    a = ap.parse_args(argv)
    if a.ref is None:
        a.ref = list(DEFAULT_REFS)
    if not a.apply and not a.test:
        ap.error("give --apply or at least one --test VIDEO_ID")

    token = state.setting("HF_TOKEN", "")
    if not token:
        sys.exit("HF_TOKEN not set (.env)")
    con = open_state_ro()

    t_start = time.monotonic()
    import torch
    from pyannote.audio import Inference, Model
    model = Model.from_pretrained(a.model, token=token)
    inference = Inference(model, window="whole", device=torch.device(a.device))
    print(f"model {a.model} on {a.device}, loaded in {time.monotonic() - t_start:.1f} s "
          f"(torch {torch.__version__}, threads {torch.get_num_threads()}); thresh {a.thresh} gap {a.gap}", flush=True)

    anchors = build_anchors(a, con, inference)

    if a.apply:
        vids = [r["video_id"] for r in con.execute("SELECT video_id FROM videos WHERE status = 'done' ORDER BY video_id")]
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\napply: {len(vids)} done videos; backups -> {BACKUP_DIR.relative_to(ROOT)}/", flush=True)
    else:
        vids = a.test

    stats = {"embedding": 0, "talk_time": 0}
    changed, sims, skipped, failed = [], [], [], []
    for vid in vids:
        t0 = time.monotonic()
        try:
            doc = load_doc(vid)
        except (OSError, ValueError) as e:
            failed.append((vid, f"cannot read JSON: {e}"))
            print(f"{vid}: FAILED cannot read JSON: {e}", flush=True)
            continue
        if a.apply and not a.force and doc.get("host_method"):
            skipped.append(vid)
            continue
        spk = doc.get("speaker") or doc.get("channel")
        anc = anchors.get(spk)
        if anc is None:
            failed.append((vid, f"speaker {spk} has no anchor"))
            print(f"{vid}: FAILED speaker {spk} has no anchor", flush=True)
            continue
        shares, talk = talk_shares(doc["segments"])
        tt_host, tt_share = talk_time_host(doc["segments"])
        if tt_host is None:
            failed.append((vid, "no labelled speech (undiarized?)"))
            print(f"{vid}: FAILED no labelled speech", flush=True)
            continue
        try:
            wave = decode(opus_for(con, vid))
            clusters = embed_clusters(inference, wave, doc["segments"], talk, a)
        except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError, ValueError) as e:
            failed.append((vid, f"embed failed: {e}"))
            print(f"{vid}: FAILED embed: {e}", flush=True)
            continue
        if not clusters:
            failed.append((vid, "no cluster with enough speech to embed"))
            print(f"{vid}: FAILED no embeddable cluster", flush=True)
            continue
        rows = [(label, float(anchor_from(vecs) @ anc)) for label, _, vecs in clusters]
        method, hosts, best, runner = decide(rows, a.thresh, a.gap)
        cos_by = dict(rows)

        if not a.apply:
            print(f"\n{vid} ({spk}, {len(wave) / SR / 60:.0f} min, talk-time host {tt_host} @ {tt_share})")
            print(f"  {'cluster':<11} {'share':>6} {'used':>7} {'n':>3}  {'cos':>6}  {'p10':>6}  {'p90':>6}")
            for label, secs, vecs in clusters:
                s = vecs @ anc
                print(f"  {label:<11} {shares[label]:>6.1%} {secs:>6.0f}s {len(vecs):>3}  {cos_by[label]:>6.3f}  "
                      f"{np.percentile(s, 10):>6.3f}  {np.percentile(s, 90):>6.3f}")
            print(f"  -> {method}: host_speakers {hosts} best {best:.3f} runner-up "
                  f"{'none' if runner is None else f'{runner:.3f}'}; "
                  f"{'same as' if hosts[0] == tt_host else 'DIFFERS from'} talk-time host {tt_host}")
            continue

        # ---- apply: back up, then write the extended doc atomically ----------
        backup = BACKUP_DIR / f"{vid}.json"
        if not backup.exists():
            shutil.copy2(WHISPER_DIR / f"{vid}.json", backup)
        if method == "embedding":
            doc["host_speaker"] = hosts[0]
            doc["host_confidence"] = round(best, 3)
        else:
            hosts = [tt_host]
            doc["host_speaker"] = tt_host
            doc["host_confidence"] = tt_share
        doc["host_method"] = method
        doc["host_similarity"] = round(best, 3)
        doc["host_runner_up"] = None if runner is None else round(runner, 3)
        doc["host_speakers"] = hosts
        doc["host_ambiguous"] = method != "embedding"
        doc["host_embed_model"] = a.model
        write_atomic(WHISPER_DIR / f"{vid}.json", doc)

        new_share = sum(shares.get(h, 0.0) for h in hosts)
        stats[method] += 1
        sims.append(best)
        if set(hosts) != {tt_host}:
            changed.append((vid, spk, tt_host, hosts, best, runner, tt_share, new_share))
        print(f"{vid} {spk:<8} {method:<9} sim {best:.3f} runner-up {'  none' if runner is None else f'{runner:.3f}'} "
              f"host {tt_host}->{','.join(hosts)} share {tt_share:.2f}->{new_share:.2f} "
              f"({len(clusters)} clusters, {time.monotonic() - t0:.0f} s)", flush=True)

    if a.apply:
        n = stats["embedding"] + stats["talk_time"]
        print(f"\nSUMMARY: {n} videos written: {stats['embedding']} embedding, {stats['talk_time']} talk_time fallback; "
              f"{len(skipped)} skipped (already enrolled), {len(failed)} failed (untouched); "
              f"wall {(time.monotonic() - t_start) / 60:.1f} min")
        if sims:
            print(f"similarity min {min(sims):.3f} median {float(np.median(sims)):.3f} max {max(sims):.3f}")
        if changed:
            print(f"host changed from talk-time guess in {len(changed)}:")
            for vid, spk, old, new, best, runner, os_, ns in changed:
                print(f"  {vid} {spk:<8} {old} -> {','.join(new)} sim {best:.3f} "
                      f"runner-up {'none' if runner is None else f'{runner:.3f}'} share {os_:.2f}->{ns:.2f}")
        for vid, why in failed:
            print(f"  FAILED {vid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
