# HEARTBEAT.md — ytpt (copy into the OpenClaw workspace)

Every heartbeat, in order. Keep it cheap: run the scripts, don't explore.

1. `cd ~/ytpt && .venv/bin/python dashboard/status.py --new-alerts`
   - Empty output: nothing new since the last heartbeat.
   - Otherwise: send those lines to Puck on WhatsApp as they are.
2. `.venv/bin/python pipeline/notify.py --flush`
   (resends messages that queued while the gateway was down; say nothing if it prints "outbox empty").
3. If step 1 printed `NEW CRITICAL` for `orphan_pod`, also tell Puck the pod id
   and to terminate it in the RunPod console now. Do not terminate it yourself
   unless `pipeline/gpu_batch.py teardown` exists.

If nothing needs attention, reply HEARTBEAT_OK.
