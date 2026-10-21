# AGENTS.md

Read **`START_HERE.md`** first — current state, reading order, and the mistakes
that cost real money. Repo rules and the Claude API constraints are in
`CLAUDE.md`. Design decisions are in `plan/DECISIONS.md`; don't reopen one
without telling Puck why.

Before changing anything: `./setup.sh`, `make test`, `make smoke`.
After changing anything: `make test` again (it pins the bugs found in the
2026-09-17 review), and `make qa` if transcripts are involved.

If you are the OpenClaw supervisor rather than a coding session, your operating
instructions are `ops/openclaw/AGENTS.md` (copied into the OpenClaw workspace),
and your runbooks are `ops/openclaw/skills/ytpt-ops/SKILL.md`.
