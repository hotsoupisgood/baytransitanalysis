# OpenClaw supervisor setup (macOS)

The supervisor is an OpenClaw agent on the orchestrator Mac. It watches the
pipeline, runs the repo's scripts, and talks to Puck on WhatsApp. It does not
transcribe, download in a loop, or spend money on its own
(`plan/DECISIONS.md` #6 and #10).

OpenClaw's CLI moves fast (the `cron` command became `automations` in 2026). If
a command here is rejected, run `openclaw <command> --help` and `openclaw
doctor` rather than guessing; the shapes below were taken from docs.openclaw.ai
on 2026-09-17.

---

## 1. A dedicated macOS user

Run it as its own non-admin user (`ytpt`), not Puck's account. It will hold a
RunPod key, an HF token, an Anthropic key and shell access.

```bash
# as an admin: System Settings -> Users & Groups -> add "ytpt" (Standard)
sudo -u ytpt -i           # then everything below runs as ytpt
```

Put the working copy in that user's home — **not** in `~/Documents`,
`~/Desktop` or `~/Downloads`. Those are privacy-protected (TCC) and a launchd
job in them fails with "Operation not permitted" after appearing to start.

```bash
git clone <the repo>  ~/ytpt      # or: tar xzf ytpt-handoff-*.tar.gz && mv ytpt-handoff-* ~/ytpt
cd ~/ytpt && cp .env.example .env && chmod 600 .env    # fill it in
./setup.sh && make test && make smoke
```

`STATE_DB_PATH` must be on real local disk: `~/.local/share/ytpt/state.sqlite`
(the default) is correct; a synced folder is not.

## 2. Install OpenClaw

Node ≥ 24.16 (26.1+ recommended). `node --version` first.

```bash
curl -fsSL https://openclaw.ai/install.sh | bash      # runs the onboarding wizard
openclaw onboard --anthropic-api-key "sk-ant-..."     # API key auth, not the Claude subscription
openclaw gateway install                              # LaunchAgent, starts at login
openclaw gateway status                               # expects: listening on 18789
openclaw doctor
```

Use a **separate Anthropic API key** in its own Console workspace with a monthly
spend limit. Subscription reuse through the Claude CLI works but draws on the
signed-in subscription's limits and expects the same host.

## 3. Config

Start from `ops/openclaw/openclaw.example.json5`, merge into
`~/.openclaw/openclaw.json`, then `openclaw doctor`. What matters:

- `gateway.bind: "loopback"` and a long random `gateway.auth.token`.
- `channels.whatsapp.dmPolicy: "allowlist"` with only Puck's number in
  `allowFrom`, and `groupPolicy: "disabled"`. Without this, anyone who messages
  the linked number is talking to a shell.
- `agents.defaults.model.primary: "anthropic/claude-sonnet-5"` with
  `claude-opus-5` as a fallback (see `docs/MODELS.md`).
- `agents.defaults.heartbeat`: `every: "30m"`, `isolatedSession: true`,
  `lightContext: true`, `activeHours` — a full-context heartbeat every 30
  minutes is the single easiest way to spend real money on nothing.
- `approvals.exec`: forward approval prompts to WhatsApp (§5).

## 4. Link WhatsApp

```bash
openclaw channels login --channel whatsapp     # QR code; scan from the phone
openclaw message send --channel whatsapp --target "+15551234567" --message "ytpt supervisor online"
```

A separate number is recommended by the docs; Puck's own number with self-chat
also works. Then set in `.env`:

```
NOTIFY_BACKEND=openclaw
OPENCLAW_NOTIFY_CHANNEL=whatsapp
OPENCLAW_NOTIFY_TARGET=+15551234567
```

and prove the pipeline's own path works:

```bash
python3 pipeline/notify.py --dry-run info "test"     # prints the command
./scripts/smoke_test.sh --notify                      # actually sends one
```

If the gateway is down when the pipeline notifies, the message goes to
`~/.local/share/ytpt/notify_outbox.jsonl` and the heartbeat's
`pipeline/notify.py --flush` sends it later. Nothing is lost silently.

## 5. Exec approvals — the spend gate

This is the mechanism behind `DECISIONS.md` #10, and it is enforced by OpenClaw,
not by a sentence in a prompt.

```bash
openclaw approvals set --stdin < ops/openclaw/exec-approvals.example.json   # edit the paths first
openclaw approvals get
openclaw exec-policy show
```

- `security: "allowlist"`, `ask: "on-miss"`, `askFallback: "deny"`.
- The allowlist holds only routine, read-only or reversible commands
  (`dashboard/status.py`, `notify.py`, `qa_transcripts.py`, `make qa|status|test|smoke`,
  `launchctl print|kickstart`, `tail` on the log dir).
- **`pipeline/gpu_batch.py launch` is deliberately not on it.** When the agent
  runs it, OpenClaw sends Puck the approval request on WhatsApp; he answers
  `/approve <id> allow-once`. Unanswered requests expire (30 min default) as
  denials.
- Never allowlist bare `python3`, `bash`, `sh`, `curl`, `osascript` or
  `runpodctl`: any of them turns the allowlist into "run anything", including a
  direct RunPod API call that bypasses the gate.
- `openclaw security audit` after every config change.

Defence in depth, because no single layer is enough:

| Layer | Stops |
|---|---|
| OpenClaw exec approval | the agent launching a pod on its own |
| `gpu_batch.py` refuses without an approved batch row + under-cap estimate | a mistaken or replayed launch |
| `spend` table + `MONTHLY_SPEND_CAP_USD` | a slow leak across many batches |
| Small RunPod prepaid balance | everything above failing at once |
| `pods` row written **before** the create call, plus a `finally:` teardown | a crash leaving a pod billing |

## 6. Workspace files

```bash
cp ops/openclaw/AGENTS.md    ~/.openclaw/workspace/AGENTS.md
cp ops/openclaw/HEARTBEAT.md ~/.openclaw/workspace/HEARTBEAT.md
mkdir -p ~/.openclaw/workspace/skills
cp -r ops/openclaw/skills/ytpt-ops ~/.openclaw/workspace/skills/
```

`AGENTS.md` is injected into every session — keep it short. The runbooks live in
the skill, which loads on demand. Edit them in the repo and copy, so the
supervisor's instructions stay under version control.

## 7. Scheduled work

The heartbeat handles "is anything new?". Use automations for things that should
happen at a time:

```bash
# Twice-daily digest, straight from the status script (no model in the loop)
openclaw automations create "0 8,20 * * *" --name "ytpt digest" \
  --command "/Users/ytpt/ytpt/.venv/bin/python /Users/ytpt/ytpt/dashboard/status.py" \
  --command-cwd /Users/ytpt/ytpt \
  --announce --channel whatsapp --to "+15551234567"

openclaw automations list
openclaw automations run <job-id> --wait
```

Workers (downloader, transcoder, captions, dashboard) belong to **launchd**,
not to OpenClaw: they must survive the gateway being restarted.

```bash
./ops/launchd/install_dashboard.sh          # http://localhost:8765
./ops/launchd/install_workers.sh            # downloader + transcoder + captions
./ops/launchd/install_workers.sh com.ytpt.captions    # just one of them
launchctl print gui/$(id -u)/com.ytpt.dashboard | head
```

Pause/resume for the workers, once they exist:

```bash
launchctl bootout   gui/$(id -u)/com.ytpt.downloader     # pause
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ytpt.downloader.plist
```

`com.ytpt.captions` (2026-09-22) is the caption trickle worker: **one meeting an
hour with a hard cap of 18 requests per rolling 24 h** since 2026-09-24, after a
second 429 at 5 requests in 41 min (T2-PLAN §3h). Both limits are read from the
`worker_requests` table, so a `KeepAlive` restart cannot burst through them;
`--rate-status` prints where the window stands and `CAPTIONS_MIN_GAP_S` /
`CAPTIONS_MAX_GAP_S` / `CAPTIONS_DAILY_CAP` in `.env` change them without
touching the plist. Its queue is ordered by measured yield, highest first
(`--yield-table`), not by date. Both it and the downloader exit **0** on a YouTube bot check so
`KeepAlive` leaves them down; the captions worker also writes a 24 h row in
`backoffs`, so bringing it back early changes nothing:

```bash
.venv/bin/python pipeline/captions.py --backoff-status   # how long is left
.venv/bin/python pipeline/captions.py --clear-backoff    # Puck's call, then:
launchctl kickstart -k gui/$(id -u)/com.ytpt.captions
```

## 8. Prove the supervisor works

1. WhatsApp `status` → the digest arrives and matches http://localhost:8765.
2. Stop the dashboard agent; next heartbeat reports it; restart it; the alert clears.
3. `python3 dashboard/status.py --new-alerts` twice → the second call prints
   nothing (that is what keeps the heartbeat quiet).
4. Ask the agent to run something not on the allowlist (`ls /`); confirm the
   approval request reaches WhatsApp and that denying it stops the command.
5. Confirm a message from a number **not** in `allowFrom` is ignored.
6. `openclaw security audit` → clean, or every finding understood.

## 9. When something looks wrong

| Symptom | Check |
|---|---|
| No messages at all | `openclaw gateway status`; `pipeline/notify.py --flush`; is the outbox growing? |
| Agent says a script does not exist | it probably doesn't yet — `START_HERE.md` lists what is built |
| Approval never arrives | `openclaw approvals get` (targets), and that the request did not already expire |
| Dashboard 404/refused | `launchctl print gui/$(id -u)/com.ytpt.dashboard`, `~/Library/Logs/ytpt/dashboard.log`, TCC (§1) |
| Heartbeat chatty | `isolatedSession`, `lightContext`, and that it uses `--new-alerts` |
| Costs climbing | heartbeat interval and model; `openclaw automations list` for jobs nobody remembers |
