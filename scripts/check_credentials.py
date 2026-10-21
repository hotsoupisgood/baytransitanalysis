#!/usr/bin/env python3
"""
scripts/check_credentials.py -- prove every key in .env works, for $0, before
anything is rented or downloaded.

WHY: each credential fails differently and late. A bad HF token surfaces as a
401 deep inside pyannote on a rented GPU with the meter running; an unaccepted
model licence looks exactly the same. A RunPod key with a pod still running
from last week bills quietly. Checking all of it here takes seconds.

Free calls only, unless --paid is given:
  Anthropic  GET  /v1/models                       key valid; the models this project names exist
  HF         GET  /api/whoami-v2                   token valid
             GET  gated config of each pyannote model   licence accepted by THIS token's account
  RunPod     POST graphql  myself { ... }          key valid; running pods and $/hr (anything > 0 is billing)
--paid adds ONE Anthropic request (5 output tokens, ~$0.0001) with the exact
request shape meanness_analysis.py uses, which catches parameter changes
(e.g. Claude 5 models reject a non-default temperature with HTTP 400).

Secrets are read from the environment or .env and never printed or put on a
command line (argv is visible to every process on the machine).

Exit code: 0 all good, 1 something failed.
"""
import argparse, json, os, re, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANTHROPIC_VERSION = "2023-06-01"
# Models the project refers to (docs/MODELS.md). A missing one is a warning:
# the account may not have access yet, or the name changed.
EXPECTED_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-sonnet-4-6"]
# The diarization model WhisperX 3.8.x loads, plus the 3.1 fallback pair.
HF_GATED = [("pyannote/speaker-diarization-community-1", True),
            ("pyannote/speaker-diarization-3.1", False),
            ("pyannote/segmentation-3.0", False)]

results = {"ok": 0, "warn": 0, "fail": 0}


def say(kind, msg):
    results[{"ok": "ok", "warn": "warn", "FAIL": "fail"}[kind]] += 1
    print(f"  [{kind:<4}] {msg}")


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), re.sub(r"\s+#.*$", "", v.strip()).strip('"').strip("'"))


def http(method, url, headers=None, body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    # Cloudflare in front of api.runpod.io rejects the default Python-urllib UA (error 1010).
    req.add_header("User-Agent", "ytpt-check-credentials/1.0")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, str(e).encode()


def check_anthropic(paid):
    print("\n== Anthropic ==")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return say("FAIL", "ANTHROPIC_API_KEY not set")
    h = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    code, raw = http("GET", "https://api.anthropic.com/v1/models?limit=100", h)
    if code != 200:
        return say("FAIL", f"GET /v1/models -> {code}: {raw[:160]!r}")
    ids = {m.get("id") for m in json.loads(raw).get("data", [])}
    say("ok", f"key valid; {len(ids)} models visible")
    for m in EXPECTED_MODELS + [os.environ.get("SCORING_MODEL") or "claude-sonnet-4-6"]:
        if m in ids:
            say("ok", f"model available: {m}")
        else:
            say("warn", f"model not listed for this key: {m}")
    if paid:
        model = os.environ.get("SCORING_MODEL") or "claude-sonnet-4-6"
        body = {"model": model, "max_tokens": 5, "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Reply with only the integer 7."}]}
        code, raw = http("POST", "https://api.anthropic.com/v1/messages", h, body, timeout=60)
        if code != 200:
            return say("FAIL", f"scoring-shaped request to {model} -> {code}: {raw[:200]!r}")
        text = "".join(b.get("text", "") for b in json.loads(raw).get("content", []) if b.get("type") == "text")
        say("ok" if text.strip().startswith("7") else "warn",
            f"scoring-shaped request to {model} returned {text.strip()[:20]!r}")


def check_hf():
    print("\n== Hugging Face (pyannote licences) ==")
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return say("FAIL", "HF_TOKEN not set")
    h = {"Authorization": f"Bearer {tok}"}
    code, raw = http("GET", "https://huggingface.co/api/whoami-v2", h)
    if code != 200:
        return say("FAIL", f"whoami -> {code}: token invalid or revoked")
    say("ok", f"token valid for account '{json.loads(raw).get('name')}'")
    for repo, required in HF_GATED:
        code, _ = http("GET", f"https://huggingface.co/{repo}/resolve/main/config.yaml", h)
        if code == 200:
            say("ok", f"licence accepted: {repo}")
        else:
            say("FAIL" if required else "warn",
                f"{repo} -> HTTP {code}: open https://huggingface.co/{repo} signed in as that account "
                f"and accept the conditions{'' if required else ' (only needed for the 3.1 fallback)'}")


def check_runpod():
    print("\n== RunPod ==")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        return say("FAIL", "RUNPOD_API_KEY not set")
    h = {"Authorization": f"Bearer {key}"}
    rich = "query { myself { id clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr } } }"
    plain = "query { myself { id pods { id name desiredStatus } } }"
    for q in (rich, plain):              # field names drift; fall back to the minimum
        code, raw = http("POST", "https://api.runpod.io/graphql", h, {"query": q})
        try:
            data = json.loads(raw)
        except ValueError:
            data = {}
        me = (data.get("data") or {}).get("myself")
        if code == 200 and me:
            break
    else:
        return say("FAIL", f"GraphQL myself -> {code}: {raw[:200]!r}")
    say("ok", "key valid")
    if me.get("clientBalance") is not None:
        say("ok" if float(me["clientBalance"]) >= 5 else "warn",
            f"prepaid balance ${float(me['clientBalance']):.2f} (keep it small: it is the hard ceiling on a runaway pod)")
    running = [p for p in me.get("pods") or [] if p.get("desiredStatus") in ("RUNNING", "CREATED", "RESTARTING")]
    spend = me.get("currentSpendPerHr")
    if running or (spend and float(spend) > 0):
        say("FAIL", f"{len(running)} pod(s) running, ${float(spend or 0):.2f}/hr: "
                    + ", ".join(f"{p.get('id')} {p.get('name')}" for p in running)
                    + " -- terminate them unless a batch is running right now")
    else:
        say("ok", "no pods running ($0.00/hr)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paid", action="store_true", help="also send one ~$0.0001 Anthropic request")
    ap.add_argument("--only", choices=["anthropic", "hf", "runpod"])
    a = ap.parse_args(argv)
    load_env()
    for name, fn in (("anthropic", lambda: check_anthropic(a.paid)), ("hf", check_hf), ("runpod", check_runpod)):
        if not a.only or a.only == name:
            fn()
    print(f"\ncredentials: {results['ok']} ok, {results['warn']} warnings, {results['fail']} failures")
    return 1 if results["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
