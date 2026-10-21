#!/usr/bin/env python3
"""
pipeline/gpu_job.py -- the GPU batch job (PLAN.md 4.5, 4.6, 6 Phase 2).

One file, two roles, because the same contract code must run on both ends:

  ON THE POD  (`--on-pod`, started by gpu_bootstrap.sh; no state DB, no network
  but Hugging Face):
      for each video in in/manifest.json: WhisperX large-v3 (faster-whisper
      backend) -> wav2vec2 alignment -> pyannote community-1 diarization ->
      out/<video_id>.json in the PLAN.md 1 contract shape, written atomically
      the moment it finishes. out/timing.jsonl has the GPU seconds per video;
      out/FAILED/<id>.txt a per-video failure; out/DONE the exit code.

  ON THIS MAC (imported by gpu_batch.py launch, never run by hand):
      create_pod(con, batch_id)  -> pod_id   pods row BEFORE the create call
      run(con, batch_id, pod_id) -> fields   push, poll, pull per file, QA
                                             gate, state transitions, cost
      preflight / describe_plan             what --dry-run prints

  The GPU never writes state.sqlite (PLAN.md 3). Results are files; this Mac
  reads them back, runs analysis/qa_transcripts.py on each one, and only then
  moves a video to 'done'. A FAIL stays in output/batches/<id>/out/ and the
  video goes to 'failed' with the reason -- it is never loaded.

Transport is pipeline/gpu_transport.py (v1 rsync over SSH). Pod lifecycle is
RunPod's GraphQL API through gpu_batch.runpod_graphql; teardown is
gpu_batch.teardown in launch's finally:, on every exit path.

Secrets: HF_TOKEN and the SSH public key go to the pod as env at create time,
inside the request body. Nothing here prints or logs either.

Pod-side smoke without a GPU (contract only):
    python3 pipeline/gpu_job.py --selftest
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# --- the instrument (docs/MODELS.md "a model used to measure is an instrument")
PIPELINE_VERSION = "1.1.0"   # 1.1: segments split at word-level speaker changes
ASR_MODEL = "large-v3"                               # A/B large-v3-turbo in Phase 0, not here
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
LANGUAGE = "en"
CONTRACT_KEYS = ("video_id", "speaker", "channel", "channel_url", "month", "duration_s",
                 "transcribed_s", "window_start_s", "model", "diarization_model",
                 "pipeline_version", "diarized", "host_speaker", "host_confidence", "segments")

# --- pod spec (PLAN.md 4.5: RTX 4090, community tier). ALL UNVERIFIED until a
# pod has actually been created with them; overridable from .env.
POD_DEFAULTS = {
    "GPU_IMAGE": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",   # (unverified tag)
    "GPU_TYPE_ID": "NVIDIA GeForce RTX 4090",
    "GPU_CLOUD_TYPE": "COMMUNITY",
    "GPU_CONTAINER_DISK_GB": "40",               # models ~6 GB + deps ~8 GB + audio
    "GPU_MIN_VCPU": "4",
    "GPU_MIN_MEMORY_GB": "16",
    "GPU_SSH_KEY": "~/.ssh/ytpt_runpod",         # ed25519, no passphrase, pods only
    "GPU_BOOT_TIMEOUT_S": "900",
    "GPU_POLL_S": "45",
    "GPU_JOB_TIMEOUT_MIN": "",                   # empty = computed from audio hours
    "YTPT_JOB_DIR": "/workspace/ytpt",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# 1. The contract -- pure functions, identical on both ends
# =============================================================================

def resolve_host(segments):
    """host = LABELLED speaker with the most talk time; confidence = its share
    of ALL speech, unlabelled included (same rule as analysis/load_db.py, so
    the loader and the file agree). (None, None) when nothing is labelled."""
    talk = {}
    for s in segments:
        talk[s.get("speaker")] = talk.get(s.get("speaker"), 0.0) + (float(s["end"]) - float(s["start"]))
    labelled = {k: v for k, v in talk.items() if k is not None}
    if not labelled:
        return None, None
    total = sum(talk.values())
    host = max(labelled, key=labelled.get)
    return host, round(labelled[host] / total, 3) if total else None


def split_by_word_speaker(segments):
    """One segment per speaker RUN, using WhisperX's per-word labels.

    WhisperX labels a segment with whoever spoke most in it, so a rapid
    exchange (Hasan playing a clip, a guest interrupting Tucker) becomes one
    block under one name -- 15 of 48 lines were relabelled by hand in the
    2026-09-18 reference check, all of them this. The word list carries the
    right label per word; cut there. Words without a timestamp (numbers,
    punctuation tokens) or without a label stay with the run they are in.
    Segments that have no word list pass through unchanged."""
    out = []
    for s in segments:
        words = s.get("words") or []
        if not words:
            out.append(s)
            continue
        run = None
        for w in words:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            spk = w.get("speaker")
            if run is None or (spk is not None and run["speaker"] is not None and spk != run["speaker"]):
                if run is not None:
                    # A run that opened on an untimed word carries the SEGMENT's
                    # end; clamp it to where the next run starts so runs never
                    # overlap (2026-09-21 review).
                    if w.get("start") is not None and float(run["end"]) > float(w["start"]):
                        run["end"] = max(float(w["start"]), float(run["start"]))
                    out.append(run)
                run = {"start": w.get("start", s["start"]), "end": w.get("end", s["end"]),
                       "speaker": spk if spk is not None else s.get("speaker"), "text": token}
            else:
                run["text"] += " " + token
                if run["speaker"] is None and spk is not None:
                    run["speaker"] = spk
                if w.get("end") is not None:
                    run["end"] = w["end"]
        if run is not None:
            run["end"] = max(float(run["end"]), float(run["start"]))
            out.append(run)
    return out


# Subtitle-style speaker tags Whisper hallucinates on roll-call votes and long
# multi-speaker stretches ("Male Speaker 1 director", "Female Speaker: thank
# you"; 19 segments in 5 of the 46 validated episodes, output/validation/REPORT.md
# 3b'). They are not speech. Stripped at the START of a segment only; the QA
# gate (qa_transcripts.SPEAKER_TAG) still warns about any that remain.
SPEAKER_TAG_PREFIX = re.compile(r"^\s*(?:(?:male|female)\s+speaker(?:\s+\d+)?|speaker\s+\d+)\b[\s:,.\-]*", re.I)


def strip_speaker_tags(segments):
    """Remove a leading speaker-tag prefix from each segment's text (in place).
    Returns (segments, n_stripped, n_emptied): a segment whose text was ONLY
    the tag keeps the row with empty text so make_doc's empty filter drops it."""
    n = emptied = 0
    for s in segments:
        text = s.get("text") or ""
        new, k = SPEAKER_TAG_PREFIX.subn("", text, count=1)
        if k:
            n += 1
            s["text"] = new.strip()
            if not s["text"]:
                emptied += 1
    return segments, n, emptied


def make_doc(meta, segments, gpu_seconds=None, stack=None, batch_id=None):
    """Contract JSON (PLAN.md 1) from a manifest entry and WhisperX segments.
    Segments keep exactly {start, end, speaker, text}; keys are added at the
    top level only (PLAN.md 1: adding keys is safe, renaming or nesting is
    not)."""
    segs = []
    split = split_by_word_speaker(segments)
    split, n_tags, n_emptied = strip_speaker_tags(split)         # 2026-09-22: hallucinated 'Male Speaker 1' prefixes
    for s in split:
        text = (s.get("text") or "").strip()
        if not text:
            continue                                    # empty segments carry no data and skew coverage
        segs.append({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                     "speaker": s.get("speaker"), "text": text})
    host, conf = resolve_host(segs)
    doc = {
        "video_id": meta["video_id"],
        "speaker": meta.get("speaker"),
        "channel": meta.get("speaker"),                 # kept for the 2026-04 readers
        "channel_url": meta.get("channel_url"),
        "month": meta.get("month"),
        "duration_s": meta.get("duration_s"),
        "transcribed_s": meta.get("transcribed_s"),
        "window_start_s": meta.get("window_start_s", 0),
        "model": ASR_MODEL,
        "diarization_model": DIARIZATION_MODEL,
        "pipeline_version": PIPELINE_VERSION,
        "diarized": host is not None,
        "host_speaker": host,
        "host_confidence": conf,
        "segments": segs,
        # extra, top level only:
        "batch_id": batch_id,
        "transcribed_at": now_iso(),
        "gpu_seconds": round(gpu_seconds, 1) if gpu_seconds is not None else None,
        "stack": stack or {},
        "speaker_tags_stripped": n_tags,                # 2026-09-22; n_emptied of them were only the tag
        "initial_prompt": meta.get("initial_prompt"),   # the per-body vocabulary WhisperX was given (None = none)
    }
    # meeting episodes (2026-09-21): where the clip sits in its meeting and what it is about
    for k in ("kind", "episode_id", "meeting_id", "body", "tags", "label", "agenda_item"):
        if meta.get(k) is not None:
            doc[k] = meta[k]
    if meta.get("episode_id"):
        doc["episode_start_s"] = doc["window_start_s"]
    return doc


def validate_doc(doc, expect_video_id=None):
    """Problems with a returned file, as strings. Empty = shaped right. The QA
    gate (analysis/qa_transcripts.py) judges the CONTENT; this only checks
    that the loader and the gate will be able to read it at all."""
    p = []
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    for k in CONTRACT_KEYS:
        if k not in doc:
            p.append(f"missing key {k}")
    if expect_video_id and doc.get("video_id") != expect_video_id:
        p.append(f"video_id {doc.get('video_id')!r} != {expect_video_id!r}")
    segs = doc.get("segments")
    if not isinstance(segs, list):
        p.append("segments is not a list")
    else:
        for i, s in enumerate(segs[:5000]):
            if not (isinstance(s, dict) and {"start", "end", "text"} <= set(s)):
                p.append(f"segment {i} lacks start/end/text")
                break
    if doc.get("model") != ASR_MODEL or doc.get("pipeline_version") != PIPELINE_VERSION:
        p.append(f"model/pipeline_version {doc.get('model')}/{doc.get('pipeline_version')} "
                 f"!= {ASR_MODEL}/{PIPELINE_VERSION} (one corpus, one instrument)")
    return p


def write_atomic(path, doc):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)                               # rsync never sees a half file


# =============================================================================
# 2. Pod side
# =============================================================================

def _stack_versions():
    out = {}
    for mod in ("torch", "whisperx", "faster_whisper", "ctranslate2", "pyannote.audio"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[mod] = getattr(m, "__version__", "?")
        except Exception:                               # noqa: BLE001 -- versions are informational
            out[mod] = "?"
    return out


def load_stack(device, compute_type, batch_size):
    """WhisperX ASR + alignment + diarization, loaded once per batch. Imports
    live here so this module is importable on a Mac with none of it installed.
    (unverified against a real pod: the DiarizationPipeline location moved
    between WhisperX 3.3 and 3.4, and pyannote 4 renamed use_auth_token; both
    are handled, neither has been exercised.)"""
    import torch, whisperx                              # noqa: E402
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not in environment")
    asr_opts = {"condition_on_previous_text": False}    # docs/MODELS.md 3: hallucination mitigation
    try:
        asr = whisperx.load_model(ASR_MODEL, device, compute_type=compute_type, language=LANGUAGE,
                                  asr_options=asr_opts)
    except TypeError:                                   # older signature without asr_options
        asr = whisperx.load_model(ASR_MODEL, device, compute_type=compute_type, language=LANGUAGE)
    align_model, align_meta = whisperx.load_align_model(language_code=LANGUAGE, device=device)
    diar_cls = getattr(getattr(whisperx, "diarize", None), "DiarizationPipeline", None) \
        or getattr(whisperx, "DiarizationPipeline")
    try:
        diar = diar_cls(model_name=DIARIZATION_MODEL, use_auth_token=token, device=device)
    except TypeError:
        diar = diar_cls(model_name=DIARIZATION_MODEL, token=token, device=device)
    return {"whisperx": whisperx, "torch": torch, "asr": asr, "align": (align_model, align_meta),
            "diar": diar, "batch_size": batch_size}


def set_initial_prompt(asr, prompt, log=print):
    """Per-file Whisper initial_prompt (the body's vocabulary: agency names,
    titles, place names -- output/validation/REPORT.md 6.1). WhisperX's
    FasterWhisperPipeline keeps its decoding options in `asr.options`, a
    faster_whisper TranscriptionOptions namedtuple, so `_replace` sets it
    without reloading the model. UNVERIFIED on a pod until the next batch: any
    failure is logged and the file is transcribed without a prompt. Returns
    True when the prompt was set."""
    try:
        opts = getattr(asr, "options", None)
        if opts is None or not hasattr(opts, "_replace"):
            log(f"[pod] initial_prompt: asr.options is not a namedtuple ({type(opts).__name__}); prompt not applied", flush=True)
            return False
        asr.options = opts._replace(initial_prompt=prompt or None)
        return True
    except Exception as e:                              # noqa: BLE001 -- a vocabulary hint is never a reason to fail a file
        log(f"[pod] initial_prompt: {type(e).__name__}: {str(e)[:120]}; prompt not applied", flush=True)
        return False


def transcribe_one(stack, audio_path, device, initial_prompt=None, diarize=True):
    """(segments, timings) for one file. Every stage is timed separately so
    Phase 0 can see where the GPU seconds go. diarize=False (2026-10-17,
    plan/RECALL-CHECK.md: whole-meeting validation transcripts) skips the
    pyannote stage; ASR and alignment -- the text -- are unchanged."""
    wx = stack["whisperx"]
    t = {}
    t0 = time.monotonic()
    audio = wx.load_audio(str(audio_path))
    t["load"] = time.monotonic() - t0
    t0 = time.monotonic()
    want = initial_prompt or None
    if not stack.get("prompt_disabled") and stack.get("prompt_applied", "unset") != want:
        # set once per change (a batch sorted by body pays nothing per file);
        # after one failure the feature is off for the rest of the batch
        if set_initial_prompt(stack["asr"], want):
            stack["prompt_applied"] = want
        else:
            stack["prompt_disabled"] = True
    result = stack["asr"].transcribe(audio, batch_size=stack["batch_size"], language=LANGUAGE)
    t["asr"] = time.monotonic() - t0
    t0 = time.monotonic()
    model_a, meta_a = stack["align"]
    result = wx.align(result["segments"], model_a, meta_a, audio, device, return_char_alignments=False)
    t["align"] = time.monotonic() - t0
    t0 = time.monotonic()
    if diarize:
        diar_segments = stack["diar"](audio)
        result = wx.assign_word_speakers(diar_segments, result)
    t["diarize"] = time.monotonic() - t0
    return result["segments"], t


def _is_cuda_error(e):
    return isinstance(e, RuntimeError) and "cuda" in str(e).lower()


def transcribe_with_cuda_retry(stack, src, device, vid, log=print, initial_prompt=None, diarize=True):
    """transcribe_one, retried ONCE after a CUDA RuntimeError (out of memory,
    invalid device ordinal, ...) with the cache emptied and batch_size halved
    (min 1). Returns (segments, stages, note); note is None when the first
    attempt succeeded. batch_size is restored afterwards so one bad file does
    not slow the rest of the batch. Two of the three b11 failures were exactly
    this and the audio was fine (UNVERIFIED on a real pod until the next
    batch runs)."""
    try:
        segs, stages = transcribe_one(stack, src, device, initial_prompt, diarize)
        return segs, stages, None
    except RuntimeError as e:
        if not _is_cuda_error(e):
            raise
        first = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        stack["torch"].cuda.empty_cache()
    except Exception:                                   # noqa: BLE001 -- best effort, torch may be absent in tests
        pass
    old = stack.get("batch_size", 16)
    stack["batch_size"] = max(1, old // 2)
    log(f"[pod] {vid}: {first} -- cache emptied, retrying once at batch_size {stack['batch_size']}", flush=True)
    try:
        segs, stages = transcribe_one(stack, src, device, initial_prompt, diarize)
    finally:
        stack["batch_size"] = old
    return segs, stages, f"retried after CUDA error ({first}) at batch_size {max(1, old // 2)}"


def run_on_pod(batch_id, manifest_path, in_dir, out_dir, device="cuda", compute_type="float16", batch_size=16):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    failed_dir = out_dir / "FAILED"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(exist_ok=True)
    timing = out_dir / "timing.jsonl"
    todo = [m for m in manifest["videos"] if not (out_dir / f"{m['video_id']}.json").exists()]
    print(f"[pod] batch {batch_id}: {len(manifest['videos'])} videos, {len(todo)} to do, device={device}", flush=True)
    if not todo:
        return 0
    t0 = time.monotonic()
    stack = load_stack(device, compute_type, batch_size)
    stack_v = _stack_versions()
    print(f"[pod] models loaded in {time.monotonic() - t0:.0f} s: {stack_v}", flush=True)
    n_fail = 0
    for m in todo:
        vid = m["video_id"]
        src = in_dir / m["opus"]
        t0 = time.monotonic()
        try:
            if not src.exists():
                raise FileNotFoundError(f"{src.name} was not pushed")
            segs, stages, pod_note = transcribe_with_cuda_retry(stack, src, device, vid, initial_prompt=m.get("initial_prompt"),
                                                                diarize=m.get("diarize", True) is not False)
            gpu_s = time.monotonic() - t0
            doc = make_doc(m, segs, gpu_seconds=gpu_s, stack=stack_v, batch_id=batch_id)
            if m.get("diarize") is False:
                doc["diarization_model"] = None             # whole-meeting validation file: never diarized
            if m.get("initial_prompt") and stack.get("prompt_applied") != m.get("initial_prompt"):
                doc["initial_prompt"] = None                # asked for, not applied (see set_initial_prompt's log line)
                doc["initial_prompt_requested"] = m.get("initial_prompt")
            if pod_note:
                doc["pod_note"] = pod_note                  # extra top-level key; ingest_new copies it to the state note
            write_atomic(out_dir / f"{vid}.json", doc)
            with timing.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"video_id": vid, "audio_s": m.get("transcribed_s"), "gpu_s": round(gpu_s, 1),
                                     "stages": {k: round(v, 1) for k, v in stages.items()},
                                     "n_segments": len(doc["segments"]), "host_confidence": doc["host_confidence"],
                                     "note": pod_note, "at": now_iso()}) + "\n")
            rt = (m.get("transcribed_s") or 0) / gpu_s if gpu_s else 0
            print(f"[pod] {vid}: {len(doc['segments'])} segments, {gpu_s:.0f} s GPU, {rt:.1f}x realtime, "
                  f"host {doc['host_speaker']} @ {doc['host_confidence']}", flush=True)
        except Exception as e:                          # noqa: BLE001 -- one bad file must not sink the batch
            n_fail += 1
            (failed_dir / f"{vid}.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            print(f"[pod] {vid}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            if "CUDA" in str(e) and "out of memory" in str(e):
                try:
                    stack["torch"].cuda.empty_cache()
                except Exception:                       # noqa: BLE001
                    pass
    return 1 if n_fail else 0


# =============================================================================
# 3. Laptop side -- called by gpu_batch.launch; imports state lazily
# =============================================================================

def _state():
    sys.path.insert(0, str(HERE))
    import state                                        # noqa: E402
    return state


def pod_cfg():
    st = _state()
    return {k: st.setting(k, v) for k, v in POD_DEFAULTS.items()}


def ssh_key_path(cfg=None):
    return Path(os.path.expanduser((cfg or pod_cfg())["GPU_SSH_KEY"]))


def batch_dir(batch_id):
    st = _state()
    return Path(os.path.expanduser(st.setting("BATCH_DIR", str(ROOT / "output" / "batches")))) / batch_id


def whisper_dir():
    st = _state()
    return Path(os.path.expanduser(st.setting("WHISPER_DIR", str(ROOT / "output" / "whisper"))))


def build_manifest(con, batch_id):
    """The pod's worklist, from the batch's rows. Also the list of opus paths
    to push. Problems (missing files, wrong status) come back as strings."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(videos)")}
    extra = ", v.kind, v.episode_id, v.meeting_id, v.raw_meta" if "kind" in cols else ", 'show' AS kind, NULL AS episode_id, NULL AS meeting_id, NULL AS raw_meta"
    rows = [dict(r) for r in con.execute(
        "SELECT v.video_id, v.speaker, v.channel_url, v.month, v.duration_s, v.transcribed_s, v.opus_path, v.status" + extra +
        " FROM batch_videos b JOIN videos v USING (video_id) WHERE b.batch_id = ? ORDER BY v.video_id", (batch_id,))]
    has_episodes = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episodes'").fetchone() is not None
    problems, videos, paths = [], [], []
    for r in rows:
        p = Path(r["opus_path"] or "")
        if not r["opus_path"] or not p.exists():
            problems.append(f"{r['video_id']}: opus file missing ({r['opus_path']})")
            continue
        if r["status"] not in ("queued", "transcribing"):
            problems.append(f"{r['video_id']}: status '{r['status']}', expected 'queued'")
        v = {"video_id": r["video_id"], "speaker": r["speaker"], "channel_url": r["channel_url"],
             "month": r["month"], "duration_s": r["duration_s"], "transcribed_s": r["transcribed_s"],
             "window_start_s": 0, "opus": p.name}
        if r.get("kind") == "episode":
            # 2026-09-22: the body's WhisperX vocabulary (pipeline/vocab.py), <= 200 chars
            try:
                import vocab                                 # noqa: E402
                v["initial_prompt"] = vocab.for_body(con, r["speaker"]) or None
            except Exception:                                # noqa: BLE001 -- a vocabulary hint never blocks a batch
                v["initial_prompt"] = None
            # 2026-09-21 (PROPOSAL 3f): the clip's offset in the meeting, so the
            # transcript's timestamps can be placed; added keys only.
            ep = con.execute("SELECT start_s, meeting_id, tags, label, agenda_item FROM episodes WHERE episode_id = ?",
                             (r["episode_id"] or r["video_id"],)).fetchone() if has_episodes else None
            meta = {}
            try:
                meta = json.loads(r.get("raw_meta") or "{}")
            except ValueError:
                pass
            start = ep["start_s"] if ep and ep["start_s"] is not None else meta.get("window_start_s", 0)
            v.update({"window_start_s": float(start or 0), "kind": "episode",
                      "episode_id": r["episode_id"] or r["video_id"],
                      "meeting_id": (ep["meeting_id"] if ep else None) or r.get("meeting_id") or meta.get("meeting_id"),
                      "body": r["speaker"], "tags": (ep["tags"] if ep else None) or meta.get("tags") or meta.get("label"),
                      "label": (ep["label"] if ep else None) or meta.get("label"),
                      "agenda_item": (ep["agenda_item"] if ep else None) or meta.get("agenda_item")})
        elif r.get("kind") == "validation":
            # 2026-10-17 (plan/RECALL-CHECK.md): a whole-meeting audio file
            # transcribed to measure the locate step's recall. Same ASR and
            # alignment as the corpus, the body's vocabulary, no diarization
            # unless raw_meta says so. Never ingested into output/whisper --
            # scripts/recall_check.py runs launch with WHISPER_DIR / BATCH_DIR
            # pointed at output/validation/recall/.
            meta = {}
            try:
                meta = json.loads(r.get("raw_meta") or "{}")
            except ValueError:
                pass
            v.update({"kind": "validation", "meeting_id": r.get("meeting_id") or meta.get("meeting_id"),
                      "body": meta.get("body") or r["speaker"], "initial_prompt": meta.get("initial_prompt") or None,
                      "diarize": bool(meta.get("diarize", False))})
        videos.append(v)
        paths.append(p)
    manifest = {"batch_id": batch_id, "created_at": now_iso(), "asr_model": ASR_MODEL,
                "diarization_model": DIARIZATION_MODEL, "pipeline_version": PIPELINE_VERSION,
                "videos": videos}
    return manifest, paths, problems


def preflight(con, batch_id):
    """Everything that would make the real launch fail AFTER money was spent,
    checked for $0. Called by gpu_batch.launch_checks once the artefacts exist."""
    import gpu_transport                                # noqa: E402
    problems = gpu_transport.local_tools_ok()
    cfg = pod_cfg()
    key = ssh_key_path(cfg)
    if not key.exists() or not Path(str(key) + ".pub").exists():
        problems.append(f"SSH key {key} (+ .pub) missing -- create it once: "
                        f"ssh-keygen -t ed25519 -N '' -f {key} -C ytpt-runpod")
    _, _, mp = build_manifest(con, batch_id)
    problems += mp
    try:
        d = batch_dir(batch_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".write-test").write_text("ok"); (d / ".write-test").unlink()
        whisper_dir().mkdir(parents=True, exist_ok=True)
    except OSError as e:
        problems.append(f"cannot write output dirs: {e}")
    if subprocess.run(["bash", "-n", str(HERE / "gpu_bootstrap.sh")], capture_output=True).returncode != 0:
        problems.append("gpu_bootstrap.sh does not parse (bash -n)")
    freeze = ROOT / "plan" / "gpu-freeze.txt"
    if not any(l.strip() and not l.startswith("#") and "==" in l for l in freeze.read_text().splitlines()):
        problems.append("plan/gpu-freeze.txt has no name==version lines")
    return problems


def describe_plan(con, batch_id):
    """What `launch` would do, without doing any of it (--dry-run)."""
    cfg = pod_cfg()
    manifest, paths, _ = build_manifest(con, batch_id)
    mb = sum(p.stat().st_size for p in paths) / 1e6
    hours = sum(float(v["transcribed_s"] or 0) for v in manifest["videos"]) / 3600
    freeze = ROOT / "plan" / "gpu-freeze.txt"
    unverified = "UNVERIFIED" in freeze.read_text()[:600]
    lines = [
        f"plan for {batch_id} (dry run -- nothing below has happened):",
        f"  1. pods row 'creating' -> RunPod podFindAndDeployOnDemand: {cfg['GPU_TYPE_ID']}, "
        f"{cfg['GPU_CLOUD_TYPE']}, image {cfg['GPU_IMAGE']}, {cfg['GPU_CONTAINER_DISK_GB']} GB disk, "
        f"ports 22/tcp, env HF_TOKEN + PUBLIC_KEY ({ssh_key_path(cfg)}.pub)",
        f"  2. wait <= {cfg['GPU_BOOT_TIMEOUT_S']} s for a public SSH port, then rsync push "
        f"{len(paths)} opus files ({mb:.1f} MB, {hours:.2f} capped audio-h) + manifest + job + bootstrap + freeze"
        f" -> {cfg['YTPT_JOB_DIR']}",
        f"  3. videos 'queued' -> 'transcribing'; nohup bash gpu_bootstrap.sh {batch_id} on the pod "
        f"(pip install -r gpu-freeze.txt{' [UNVERIFIED pins]' if unverified else ''}, then gpu_job.py --on-pod)",
        f"  4. every {cfg['GPU_POLL_S']} s: rsync pull out/ -> {batch_dir(batch_id)}/out; each new <id>.json is "
        f"contract-checked, run through analysis/qa_transcripts.py, and on OK/WARN copied to {whisper_dir()} "
        f"and moved to 'done'; a FAIL stays in the batch dir and the video goes to 'failed'",
        f"  5. on out/DONE (or timeout {job_timeout_min(hours, cfg):.0f} min): batch gpu_minutes/actual_usd "
        f"from pod uptime x ${_state().setting('GPU_USD_PER_HOUR', 0.34)}/h; WhatsApp summary; "
        f"finally: gpu_batch.teardown terminates the pod and records spend",
    ]
    return "\n".join(lines)


def job_timeout_min(audio_hours, cfg=None):
    cfg = cfg or pod_cfg()
    if cfg["GPU_JOB_TIMEOUT_MIN"]:
        return float(cfg["GPU_JOB_TIMEOUT_MIN"])
    st = _state()
    rtfx = float(st.setting("GPU_RTFX_ESTIMATE", 25))
    boot = float(st.setting("GPU_BOOT_MINUTES", 7))
    # boot + deps/model download + 6x the estimated work, capped at 12 h: a
    # wedged pod must not bill for a weekend.
    return min(boot + 25 + max(15.0, 6 * audio_hours * 60 / rtfx), 12 * 60)


# --- RunPod calls (all through gpu_batch.runpod_graphql, the one HTTP path) ---

def _graphql():
    sys.path.insert(0, str(HERE))
    import gpu_batch                                    # noqa: E402
    return gpu_batch


CREATE_MUTATION = """
mutation ($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) { id name desiredStatus imageName costPerHr machineId }
}"""
POD_QUERY = """
query ($id: String!) {
  pod(input: {podId: $id}) { id name desiredStatus costPerHr
    runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } }
}"""
MY_PODS_QUERY = "query { myself { pods { id name desiredStatus } } }"


def create_pod(con, batch_id):
    """Rent the pod. The pods row is inserted BEFORE the create call under a
    provisional id and reconciled after (SCHEMA.md `pods`): a create that
    succeeds and then crashes must still be visible as a live pod."""
    gb = _graphql()
    st = _state()
    cfg = pod_cfg()
    pub = Path(str(ssh_key_path(cfg)) + ".pub").read_text(encoding="utf-8").strip()
    hf = st.setting("HF_TOKEN", "")
    if not hf:
        raise RuntimeError("HF_TOKEN is not set")
    name = f"ytpt-{batch_id}"
    provisional = f"pending-{batch_id}-{int(time.time())}"
    con.execute("INSERT INTO pods VALUES (?,?,?,?)", (provisional, st.now_iso(), None, "creating"))
    variables = {"input": {
        "cloudType": cfg["GPU_CLOUD_TYPE"], "gpuCount": 1, "volumeInGb": 0,
        "containerDiskInGb": int(cfg["GPU_CONTAINER_DISK_GB"]),
        "minVcpuCount": int(cfg["GPU_MIN_VCPU"]), "minMemoryInGb": int(cfg["GPU_MIN_MEMORY_GB"]),
        "gpuTypeId": cfg["GPU_TYPE_ID"], "name": name, "imageName": cfg["GPU_IMAGE"],
        "ports": "22/tcp", "volumeMountPath": "/workspace", "supportPublicIp": True, "startSsh": True,
        "env": [{"key": "HF_TOKEN", "value": hf}, {"key": "PUBLIC_KEY", "value": pub},
                {"key": "YTPT_JOB_DIR", "value": cfg["YTPT_JOB_DIR"]}],
    }}
    try:
        data = gb.runpod_graphql(CREATE_MUTATION, variables, timeout=120)
        pod = (data or {}).get("podFindAndDeployOnDemand") or {}
        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError(f"create returned no pod id: {json.dumps(data)[:300]}")
    except Exception as e:
        # We do not know whether RunPod created something. Look; adopt or clear.
        adopted = None
        try:
            mine = (gb.runpod_graphql(MY_PODS_QUERY, timeout=60).get("myself") or {}).get("pods") or []
            adopted = next((p["id"] for p in mine if p.get("name") == name), None)
        except Exception as e2:                         # noqa: BLE001
            con.execute("UPDATE pods SET status = 'ORPHAN' WHERE pod_id = ?", (provisional,))
            raise RuntimeError(f"pod create failed ({e}) AND the reconcile query failed ({e2}): "
                               f"check the RunPod console NOW; pods row {provisional} left as ORPHAN") from e
        if adopted:
            con.execute("UPDATE pods SET pod_id = ? WHERE pod_id = ?", (adopted, provisional))
            gb.teardown(con, adopted, n_videos=0, note=f"create call errored after creating {name}; torn down")
        else:
            con.execute("DELETE FROM pods WHERE pod_id = ?", (provisional,))
        raise
    con.execute("UPDATE pods SET pod_id = ?, status = 'running' WHERE pod_id = ?", (pod_id, provisional))
    print(f"pod {pod_id} created ({name}, {pod.get('imageName')}, ${pod.get('costPerHr')}/h reported by RunPod)")
    return pod_id


def pod_status(pod_id):
    """(desiredStatus, ip, public_port_22, uptime_s) -- ip/port None until the
    public port is mapped."""
    data = _graphql().runpod_graphql(POD_QUERY, {"id": pod_id}, timeout=60)
    pod = (data or {}).get("pod")
    if not pod:
        return None, None, None, 0
    rt = pod.get("runtime") or {}
    ip = port = None
    for p in rt.get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type", "tcp") == "tcp":
            ip, port = p.get("ip"), p.get("publicPort")
            break
    return pod.get("desiredStatus"), ip, port, rt.get("uptimeInSeconds") or 0


def pod_status_retry(pod_id, tries=3, sleep_s=10, log=print):
    """pod_status, retried on a transport-level failure (DNS blip, 5xx, a
    Cloudflare HTML page). One failed control-plane call used to fail the whole
    batch and tear down a pod that was booting fine (2026-09-21 review); a
    3x10 s retry is nothing against a 15-min boot. Re-raises after the last try."""
    for i in range(tries):
        try:
            return pod_status(pod_id)
        except Exception as e:                          # noqa: BLE001 -- see docstring
            if i + 1 == tries:
                raise
            log(f"pod status query failed ({type(e).__name__}: {str(e)[:120]}); retry {i + 1}/{tries - 1} in {sleep_s} s")
            time.sleep(sleep_s)


def wait_for_ssh(pod_id, key, timeout_s, poll_s=15):
    import gpu_transport                                # noqa: E402
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < timeout_s:
        status, ip, port, up = pod_status_retry(pod_id)
        if status is None:
            raise RuntimeError(f"pod {pod_id} no longer exists (reclaimed during boot?)")
        if status in ("EXITED", "TERMINATED"):
            raise RuntimeError(f"pod {pod_id} is {status} before it ever answered")
        if ip and port:
            tr = gpu_transport.SshRsync(ip, port, key)
            if tr.ok():
                print(f"pod {pod_id} answers on {ip}:{port} after {time.monotonic() - t0:.0f} s")
                return tr
            last = f"ssh not ready on {ip}:{port}"
        else:
            last = f"status {status}, no public port 22 yet (uptime {up} s)"
        time.sleep(poll_s)
    raise RuntimeError(f"pod {pod_id}: no SSH after {timeout_s:.0f} s ({last}). Community machines without a "
                       f"public IP cannot take rsync; try again or set GPU_CLOUD_TYPE=SECURE")


# --- results -----------------------------------------------------------------

def _qa():
    sys.path.insert(0, str(ROOT / "analysis"))
    import qa_transcripts                               # noqa: E402
    return qa_transcripts


def ingest_new(con, batch_id, staging, seen, log=print):
    """Look at every <id>.json / FAILED/<id>.txt pulled so far that we have
    not handled; gate it; move the video. Returns counts for this pass."""
    st, qa = _state(), _qa()
    out = {"done": 0, "qa_fail": 0, "pod_fail": 0, "warn": 0}
    dest = whisper_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(staging).glob("*.json")):
        vid = f.stem
        if vid in seen or f.name == "manifest.json":
            continue
        seen.add(vid)
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as e:
            reason = f"unreadable result JSON: {e}"
            st.fail(con, vid, "transcribing", reason)
            _episode_writeback(con, st, {}, vid, "failed", reason)
            out["qa_fail"] += 1
            continue
        if "speaker_tags_stripped" not in doc and isinstance(doc.get("segments"), list):
            # a pod running pre-2026-09-22 gpu_job.py: strip the hallucinated
            # 'Male Speaker 1' prefixes here, same rule as make_doc
            _, n_tags, _ = strip_speaker_tags(doc["segments"])
            doc["segments"] = [s for s in doc["segments"] if (s.get("text") or "").strip()]
            doc["speaker_tags_stripped"] = n_tags
            if n_tags:
                write_atomic(f, doc)                    # the staged file is what QA reads next
        problems = validate_doc(doc, expect_video_id=vid)
        try:
            r = qa.check_video(f) if not problems else {"fail": problems, "warn": []}
        except Exception as e:                          # noqa: BLE001 -- one odd file must not fail the batch and re-buy the rest
            r = {"fail": [f"QA gate crashed on this file: {type(e).__name__}: {str(e)[:200]}"], "warn": []}
        if r["fail"]:
            reason = "QA FAIL: " + "; ".join(r["fail"])[:900]
            st.fail(con, vid, "transcribing", reason)
            _episode_writeback(con, st, doc, vid, "failed", reason)
            out["qa_fail"] += 1
            log(f"  {vid}: {reason} -- kept in {staging}, NOT loaded")
            continue
        note = f"batch {batch_id}; {doc.get('gpu_seconds')} GPU-s; host {doc.get('host_speaker')} @ {doc.get('host_confidence')}"
        if doc.get("pod_note"):
            note += "; " + str(doc["pod_note"])[:200]
        if r["warn"]:
            note += "; QA WARN: " + "; ".join(r["warn"])[:300]
            out["warn"] += 1
        if doc.get("kind") == "episode" or doc.get("episode_id"):
            # 2026-09-21 (PROPOSAL 3g): dais / public roles per cluster, into the
            # file (added keys) and speaker_roles; the episodes row follows.
            try:
                import roles                                    # noqa: E402
                rs = roles.annotate(doc)
                roles.write_roles(con, doc.get("episode_id") or vid, rs, doc["roles_cues_version"])
                summ = doc["roles_summary"]
                note += f"; roles dais {summ['dais_share']:.0%} public {summ['public_share']:.0%} unknown {summ['unknown_share']:.0%}"
                write_atomic(dest / f"{vid}.json", doc)
            except Exception as e:                              # noqa: BLE001 -- roles are an annotation, never a reason to lose a transcript
                log(f"  {vid}: role labelling failed ({e!r}); transcript loaded without roles")
                shutil.copy2(f, dest / f"{vid}.json")
            _episode_writeback(con, st, doc, vid, "done", None, transcript_path=str(dest / f"{vid}.json"))
        else:
            shutil.copy2(f, dest / f"{vid}.json")
        st.transition(con, vid, "transcribing", "done", note=note, transcribed_s=doc.get("transcribed_s"),
                      last_error=None)
        out["done"] += 1
        log(f"  {vid}: done ({len(doc['segments'])} segments{', QA WARN' if r['warn'] else ''})")
    for f in sorted((Path(staging) / "FAILED").glob("*.txt")) if (Path(staging) / "FAILED").exists() else []:
        vid = f.stem
        if vid in seen:
            continue
        seen.add(vid)
        reason = "pod: " + f.read_text(encoding="utf-8").strip()[:900]
        st.fail(con, vid, "transcribing", reason)
        _episode_writeback(con, st, {}, vid, "failed", reason)
        out["pod_fail"] += 1
        log(f"  {vid}: FAILED on the pod -- {f.read_text()[:120].strip()}")
    return out


def _episode_writeback(con, st, doc, vid, status, error, **fields):
    """Mirror a clip's outcome onto its episodes row, if it has one."""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episodes'").fetchone():
        return
    ep_id = doc.get("episode_id") or vid
    if not con.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (ep_id,)).fetchone():
        return
    st.set_episode(con, ep_id, status, note=(error or "transcript ingested")[:200],
                   last_error=(error or None) and error[:300], **fields)


def settle_lost(con, batch_id, why):
    """Every video of the batch still 'transcribing' when the job ended (pod
    reclaimed, timeout, exited without a file) -> 'failed' with `why`, and its
    episodes row with it. Returns the list of ids. Without the write-back the
    episodes row stayed 'transcribing' for ever: launch's release path and
    episodes.py --requeue-failed only touch queued/transcribing videos and
    'failed' episodes (2026-09-21 review)."""
    st = _state()
    lost = [r["video_id"] for r in con.execute(
        "SELECT video_id FROM videos WHERE status = 'transcribing' AND video_id IN "
        "(SELECT video_id FROM batch_videos WHERE batch_id = ?)", (batch_id,))]
    for vid in lost:
        st.fail(con, vid, "transcribing", why)
        _episode_writeback(con, st, {}, vid, "failed", why)
    return lost


def run(con, batch_id, pod_id):
    """Drive one approved, created pod to the end. Returns the batches fields
    launch writes (gpu_minutes, actual_usd, note). Raises on anything that
    means the batch did not complete; launch records that and tears down."""
    st = _state()
    import gpu_transport                                # noqa: E402
    from notify import notify                           # noqa: E402
    cfg = pod_cfg()
    usd_hr = float(st.setting("GPU_USD_PER_HOUR", 0.34))
    key = ssh_key_path(cfg)
    job_dir = cfg["YTPT_JOB_DIR"]
    bdir = batch_dir(batch_id)
    staging = bdir / "out"
    bdir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=True)
    manifest, paths, problems = build_manifest(con, batch_id)
    if problems:
        raise RuntimeError("manifest: " + "; ".join(problems))
    (bdir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    hours = sum(float(v["transcribed_s"] or 0) for v in manifest["videos"]) / 3600
    n = len(manifest["videos"])
    b = dict(con.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone())
    t_start = time.monotonic()

    # 1. boot + transport
    tr = wait_for_ssh(pod_id, key, float(cfg["GPU_BOOT_TIMEOUT_S"]), poll_s=15)
    tr.ensure_rsync()
    tr.run(f"mkdir -p {job_dir}/in {job_dir}/out")
    tr.push([HERE / "gpu_job.py", HERE / "gpu_bootstrap.sh", ROOT / "plan" / "gpu-freeze.txt"], job_dir)
    tr.push([bdir / "manifest.json", *paths], f"{job_dir}/in")
    print(f"pushed {n} files ({sum(p.stat().st_size for p in paths) / 1e6:.1f} MB) to {tr.describe()}:{job_dir}/in")

    # 2. claim the videos, start the job detached (a dropped SSH must not kill it)
    for v in manifest["videos"]:
        st.transition(con, v["video_id"], "queued", "transcribing", note=f"pod {pod_id}")
    _graphql().episodes_follow(con, [v["video_id"] for v in manifest["videos"] if v.get("kind") == "episode"], "transcribing", f"pod {pod_id}")
    # setsid + explicit exit: on b6 (secure cloud, 2026-09-18) sshd kept the
    # session open for the backgrounded child and the plain `nohup ... &` hung
    # until our 60 s timeout, which failed the batch with the job actually
    # started. If the start call still times out, look for bootstrap.log before
    # declaring failure -- a started job is a started job.
    start = (f"cd {job_dir} && rm -f bootstrap.log && "
             f"setsid nohup bash gpu_bootstrap.sh {batch_id} > bootstrap.out 2>&1 < /dev/null & disown; exit 0")
    try:
        tr.run(start, timeout=90)
    except subprocess.TimeoutExpired:
        time.sleep(5)
        probe = tr.run(f"test -s {job_dir}/bootstrap.log && echo started || echo missing", timeout=60, check=False)
        if "started" not in (probe.stdout or ""):
            raise
        print("bootstrap start call timed out but bootstrap.log exists on the pod; continuing")
    print(f"bootstrap started on the pod; polling every {cfg['GPU_POLL_S']} s")

    # 3. poll: pull per file, gate, move state
    seen, totals = set(), {"done": 0, "qa_fail": 0, "pod_fail": 0, "warn": 0}
    timeout_s = job_timeout_min(hours, cfg) * 60
    poll_s = float(cfg["GPU_POLL_S"])
    exit_code, reason = None, ""
    while True:
        time.sleep(poll_s)
        # Checked BEFORE the pull: the `continue`s below used to skip it, so a
        # pod RunPod still called RUNNING with a dead sshd was polled (and
        # billed) for ever, past the 12-h cap (2026-09-21 review).
        if time.monotonic() - t_start > timeout_s:
            reason = f"job timeout after {timeout_s / 60:.0f} min with {totals['done']}/{n} done"
            break
        try:
            tr.pull(f"{job_dir}/out", staging, include=["*.json", "*.jsonl", "*.txt", "DONE", "FAILED/", "FAILED/*"])
        except gpu_transport.TransportError as e:
            try:
                status, *_ = pod_status_retry(pod_id)
            except Exception as e2:                     # noqa: BLE001 -- the control plane is down, the pod may be fine
                print(f"pull failed AND the pod status query failed ({type(e2).__name__}); will retry both "
                      f"(job timeout still bounds this batch)")
                continue
            if status is None or status in ("EXITED", "TERMINATED"):
                reason = f"pod {pod_id} disappeared mid-batch (status {status}): reclaimed?"
                break
            print(f"pull failed, will retry: {str(e)[-160:]}")
            continue
        got = ingest_new(con, batch_id, staging, seen)
        for k in totals:
            totals[k] += got[k]
        if (staging / "DONE").exists():
            exit_code = (staging / "DONE").read_text().strip()
            break
        if int(time.monotonic() - t_start) % 600 < poll_s:
            print(f"  {(time.monotonic() - t_start) / 60:.0f} min: {totals['done']}/{n} done, "
                  f"{totals['qa_fail'] + totals['pod_fail']} failed")

    # 4. settle whatever never came back
    lost = settle_lost(con, batch_id, reason or f"pod exited {exit_code} without a result for this video")
    try:                                                # the pod's logs, for the post-mortem
        tr.pull(f"{job_dir}", bdir, include=["bootstrap.log", "job.log", "bootstrap.out"])
    except Exception:                                   # noqa: BLE001 -- logs are nice-to-have
        pass

    # 5. cost and the measurement Phase 0 exists for
    row = con.execute("SELECT created_at FROM pods WHERE pod_id = ?", (pod_id,)).fetchone()
    started = datetime.fromisoformat(row["created_at"]) if row else datetime.now(timezone.utc)
    gpu_min = (datetime.now(timezone.utc) - started).total_seconds() / 60
    actual = round(gpu_min / 60 * usd_hr, 4)
    gpu_s = audio_s = 0.0
    tl = staging / "timing.jsonl"
    if tl.exists():
        for line in tl.read_text(encoding="utf-8").splitlines():
            try:
                t = json.loads(line)
                gpu_s += float(t.get("gpu_s") or 0); audio_s += float(t.get("audio_s") or 0)
            except ValueError:
                pass
    rtfx = (audio_s / gpu_s) if gpu_s else None
    note = (f"{totals['done']}/{n} done, {totals['qa_fail']} QA FAIL, {totals['pod_fail']} pod FAIL, "
            f"{len(lost)} lost; {gpu_min:.1f} GPU-min to job end (teardown adds a little); ")
    note += (f"measured RTFX {rtfx:.1f}x over {audio_s / 3600:.2f} audio-h (ASR+align+diarize, excl. boot/deps)"
             if rtfx else "no timing data")
    if reason:
        note += f"; ended early: {reason}"
    msg = (f"batch {batch_id} finished: {totals['done']}/{n} videos done "
           f"({totals['warn']} QA WARN, {totals['qa_fail']} QA FAIL, {totals['pod_fail'] + len(lost)} failed), "
           f"{gpu_min:.0f} GPU-min, ${actual:.2f} actual vs ${float(b.get('est_usd') or 0):.2f} est"
           + (f", measured {rtfx:.0f}x realtime" if rtfx else "") + (f". {reason}" if reason else "")
           + (". QA FAILs are NOT loaded; see output/batches/" + batch_id if totals["qa_fail"] else ""))
    print(msg)
    notify("warn" if (reason or totals["qa_fail"]) else "info", msg)
    if reason and totals["done"] == 0:
        raise RuntimeError(reason)
    return {"gpu_minutes": round(gpu_min, 2), "actual_usd": actual, "note": note[:1000]}


# =============================================================================
# 4. CLI
# =============================================================================

def selftest():
    meta = {"video_id": "T1", "speaker": "tucker", "channel_url": "u", "month": "2026-09",
            "duration_s": 100.0, "transcribed_s": 60.0}
    segs = [{"start": 0, "end": 30, "text": " hello ", "speaker": "SPEAKER_00"},
            {"start": 30, "end": 40, "text": "there", "speaker": "SPEAKER_01"},
            {"start": 40, "end": 50, "text": "", "speaker": "SPEAKER_01"},
            {"start": 50, "end": 60, "text": "unassigned"}]
    doc = make_doc(meta, segs, gpu_seconds=3.0, batch_id="bX")
    assert validate_doc(doc, "T1") == [], validate_doc(doc, "T1")
    assert doc["host_speaker"] == "SPEAKER_00" and doc["host_confidence"] == 0.6, doc
    assert len(doc["segments"]) == 3 and doc["segments"][0]["text"] == "hello"
    assert doc["diarized"] is True and doc["channel"] == "tucker"
    bad = dict(doc); bad["model"] = "medium"
    assert any("instrument" in p for p in validate_doc(bad))
    # word-level split: a 3-speaker exchange inside one WhisperX segment
    mixed = [{"start": 0, "end": 6, "text": "a b c d e f", "speaker": "SPEAKER_00",
              "words": [{"word": "a", "start": 0, "end": 1, "speaker": "SPEAKER_00"},
                        {"word": "b", "start": 1, "end": 2, "speaker": "SPEAKER_00"},
                        {"word": "c", "start": 2, "end": 3, "speaker": "SPEAKER_01"},
                        {"word": "100", "speaker": None},                       # no timestamp: stays in run
                        {"word": "d", "start": 3, "end": 4, "speaker": "SPEAKER_01"},
                        {"word": "e", "start": 4, "end": 5, "speaker": "SPEAKER_02"},
                        {"word": "f", "start": 5, "end": 6, "speaker": "SPEAKER_02"}]},
             {"start": 6, "end": 7, "text": "g", "speaker": "SPEAKER_00"}]        # no word list: unchanged
    parts = split_by_word_speaker(mixed)
    assert [(p["speaker"], p["text"], p["start"], p["end"]) for p in parts] == [
        ("SPEAKER_00", "a b", 0, 2), ("SPEAKER_01", "c 100 d", 2, 4), ("SPEAKER_02", "e f", 4, 6),
        ("SPEAKER_00", "g", 6, 7)], parts
    print("gpu_job selftest: contract shape, host heuristic, instrument pin, word-level speaker split ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on-pod", action="store_true", help="pod side: transcribe the manifest")
    ap.add_argument("--batch", default="")
    ap.add_argument("--manifest", default="in/manifest.json")
    ap.add_argument("--in", dest="in_dir", default="in")
    ap.add_argument("--out", dest="out_dir", default="out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if a.on_pod:
        return run_on_pod(a.batch, a.manifest, a.in_dir, a.out_dir, a.device, a.compute_type, a.batch_size)
    ap.error("this file is driven by pipeline/gpu_batch.py launch on the Mac, or --on-pod on the pod")


if __name__ == "__main__":
    sys.exit(main())
