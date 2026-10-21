---
name: ytpt-ops
description: Runbooks for operating Puck's YouTube transcription pipeline (status, GPU batch approval, QA failures, YouTube bot checks, stalled downloads, orphan pods). Use when a WhatsApp message or heartbeat alert concerns the ytpt pipeline.
metadata: {"openclaw": {"os": ["darwin"], "requires": {"bins": ["python3", "ffmpeg", "launchctl"]}}}
---

# ytpt operations runbooks

Repo: `~/ytpt`. Python: `~/ytpt/.venv/bin/python` (below: `PY`). Always run
from the repo root. Read `START_HERE.md` if a runbook does not cover the case.

## Status

1. `PY dashboard/status.py` → send the text as is.
2. For "why" questions, `PY dashboard/status.py --json` and look at `alerts`,
   `batches`, `errors`, `qa`.

## A GPU batch needs approval

The pipeline proposes a batch when ready-for-GPU hours pass `BATCH_TARGET_HOURS`
and messages Puck ("[ytpt APPROVAL NEEDED] ...").

1. When Puck replies `approve <id>`: run `PY dashboard/status.py --json` and
   confirm batch `<id>` is still `awaiting_approval`, spend + estimate stays
   under the cap, and `pods_active` is empty. If any check fails, tell Puck
   which and stop.
2. Run `PY pipeline/gpu_batch.py approve <id> --via whatsapp` (bookkeeping: the
   batch row becomes `approved`; nothing is rented). Then launch it DETACHED
   (never in the foreground: the exec host kills foreground commands after
   ~15 s and that orphaned pod b10 on 2026-09-20):

       mkdir -p output/batches/<id> && nohup PY pipeline/gpu_batch.py launch <id> \
         > output/batches/<id>/launch.log 2>&1 &

   If OpenClaw's exec-approval gate is enabled for this command, Puck answers
   `/approve <request-id> allow-once` (he turned that gate off on 2026-09-20;
   the DB row set by `approve` is then the only gate -- still never launch
   without his texted `approve <id>` for that batch). `launch` itself refuses
   (exit 2, reasons printed) if the row is not approved, the month's spend
   plus the estimate is over the cap, a pod is live, or the Phase 0/2
   artefacts are missing -- relay the printed reasons to Puck.
3. `launch` runs until the pod is torn down (minutes to hours; it prints
   progress every ~10 min); tail `output/batches/<id>/launch.log`. On finish it
   messages Puck itself with videos done, GPU minutes and actual $ against
   the estimate (PLAN.md 4.7); relay anything it reports as failed or lost.
   Transcripts that pass QA land in `output/whisper/`; a QA FAIL stays in
   `output/batches/<id>/out/` and its video is `failed` with the reason.
   The pod's own logs come back to `output/batches/<id>/`.
4. If `launch` dies without printing "spend recorded" (Mac asleep, killed),
   treat it as the orphan-pod runbook: the pods row may still say running.

Never approve on Puck's behalf. "yes", "ok" or a thumbs-up to a different
message is not approval of a batch: ask him to reply `approve <id>`.

Related: `PY pipeline/gpu_batch.py deny <id>` for `deny <id>`;
`PY pipeline/gpu_batch.py list` shows every batch and this month's spend.

## QA gate FAIL

1. `make qa` (writes `output/qa/latest.csv`).
2. Send Puck each FAIL line: video id, the issue, the timestamp.
3. Those transcripts must not be loaded (`make load`) until Puck decides:
   re-transcribe with VAD, exclude, or accept.

## YouTube bot check / HTTP 429 / 403 storm

1. Stop the downloader: `launchctl bootout gui/$(id -u)/com.ytpt.downloader`.
2. Tell Puck: the error line, how many downloads succeeded today, and that
   the downloader is stopped.
3. Wait for Puck. Options he may pick: wait 24 h; reduce to concurrency 1 with
   longer sleeps; update yt-dlp (`.venv/bin/pip install -U "yt-dlp[default,curl-cffi]"`);
   a PO-token provider plugin (docs/MODELS.md); cookies from a throwaway account.

## Downloader stalled (no finished download for > 6 h)

1. `launchctl print gui/$(id -u)/com.ytpt.downloader | head -20` → is it running?
2. `tail -50 ~/Library/Logs/ytpt/downloader.log`.
3. If it simply died: `launchctl kickstart -k gui/$(id -u)/com.ytpt.downloader`
   and tell Puck you restarted it and why. If the log shows a bot check, use
   that runbook instead.

## Orphan pod (CRITICAL)

A pod is running with no running batch: it is billing. Message Puck
immediately with the pod id and "terminate it in the RunPod console:
https://www.runpod.io/console/pods". `PY pipeline/gpu_batch.py teardown <pod_id>`
exists (terminates the pod and records the spend); ask Puck for permission
before running it -- it is not on the allowlist.

## Batch stuck in launching / running / failed with a live pods row

(2026-09-22, review M7.) `launch` was killed mid-way, or failed, and the DB
still shows a `creating` / `running` pods row for that batch, so the next
`launch` refuses with "a pod is already live". Run, in this order:

    PY pipeline/gpu_batch.py reconcile <batch_id> --dry-run   # read-only: asks RunPod what exists, prints the plan
    PY pipeline/gpu_batch.py reconcile <batch_id>             # pod gone -> pods row terminated, batch failed, videos released

If RunPod says the pod is LIVE, reconcile reports it and changes nothing;
tell Puck and wait. Only with his go-ahead:
`PY pipeline/gpu_batch.py reconcile <batch_id> --terminate` (teardown + spend
row, then the same settling). It never creates a pod.

## Spend caps (2026-09-22)

`status` shows "RunPod $x of $10.00 credit (stop at $9.50); Claude classifier
$y of $10.00". `propose` / `launch` refuse at the RunPod stop line and
`classify.py` at CLAUDE_CAP_USD; when either says BLOCKED, tell Puck the
totals and stop -- raising a cap is his decision (`.env`: RUNPOD_CREDIT_USD,
CLAUDE_CAP_USD).

## Retry a video (`retry <video_id>`)

`PY pipeline/requeue.py <video_id>`: a failed/dead row goes back to the last
state its files support (opus -> transcoded, m4a -> downloaded, else
discovered) with attempts reset. Dead rows are only ever revived this way.

## Weekly

- `./scripts/smoke_test.sh` → send the summary line; details only for failures.
- `.venv/bin/pip list --outdated | grep -i yt-dlp` → if outdated, tell Puck.
