#!/usr/bin/env python3
"""
pipeline/gpu_transport.py -- how files move between this Mac and a pod
(PLAN.md 4.4). v1: rsync over SSH, laptop-driven. The pod never reaches back
to the laptop and never touches state.sqlite; this Mac pushes the inputs,
polls, and pulls each result file as soon as it exists.

One class, four verbs -- push / pull / run / ok -- so the v2 upgrade (Cloudflare
R2: laptop uploads, pod pulls, neither needs the other awake) is a second class
here and a one-line change in gpu_job.py, nothing else.

SSH settings: pod IPs and host keys are recycled between RunPod customers, so
host-key pinning is meaningless; BatchMode so a prompt can never hang an
unattended run; ConnectTimeout so a reclaimed pod fails fast.
"""
import shlex, subprocess, sys
from pathlib import Path

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


class TransportError(RuntimeError):
    pass


class SshRsync:
    def __init__(self, host, port, key, user="root"):
        self.host, self.port, self.key, self.user = host, int(port), str(key), user

    # -- building blocks ------------------------------------------------------
    def _ssh_base(self):
        return ["ssh", *SSH_OPTS, "-i", self.key, "-p", str(self.port)]

    def _rsh(self):
        return " ".join(shlex.quote(a) for a in self._ssh_base())

    def _target(self):
        return f"{self.user}@{self.host}"

    def run(self, command, timeout=120, check=True):
        """Run a shell command on the pod; returns CompletedProcess."""
        r = subprocess.run([*self._ssh_base(), self._target(), command],
                           capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise TransportError(f"ssh {self.host}:{self.port} '{command[:60]}' -> exit {r.returncode}: "
                                 f"{(r.stderr or r.stdout)[-400:]}")
        return r

    def ok(self, timeout=30):
        """True when sshd answers and a shell runs. Used to wait for boot."""
        try:
            return self.run("echo ok", timeout=timeout, check=False).stdout.strip() == "ok"
        except (subprocess.TimeoutExpired, OSError):
            return False

    def ensure_rsync(self):
        """The Mac's rsync needs an rsync on the far end. Install it if the
        image lacks one (runpod/pytorch images vary)."""
        self.run("command -v rsync >/dev/null 2>&1 || "
                 "(apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null)",
                 timeout=300)

    def _rsync(self, args, timeout):
        cmd = ["rsync", "-az", "--partial", "--timeout=60", "-e", self._rsh(), *args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise TransportError(f"rsync -> exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
        return r

    # -- verbs ----------------------------------------------------------------
    def push(self, local_paths, remote_dir, timeout=3600):
        """Copy files/dirs to <remote_dir>/ on the pod (created if missing)."""
        self.run(f"mkdir -p {shlex.quote(remote_dir)}")
        paths = [str(p) for p in local_paths]
        if not paths:
            return
        self._rsync([*paths, f"{self._target()}:{remote_dir.rstrip('/')}/"], timeout)

    def pull(self, remote_dir, local_dir, timeout=1800, include=None):
        """Copy <remote_dir>/ contents down to local_dir. --partial and
        atomic temp+rename on the pod side mean a half-written file is never
        seen as complete (the job writes <id>.json.tmp then renames)."""
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        args = []
        if include:
            for pat in include:
                args += ["--include", pat]
            args += ["--exclude", "*"]
        self._rsync([*args, f"{self._target()}:{remote_dir.rstrip('/')}/", str(local_dir).rstrip("/") + "/"], timeout)

    def describe(self):
        return f"ssh -i {self.key} -p {self.port} {self._target()}"


def local_tools_ok():
    """Problems with this Mac's side of the transport, as strings."""
    problems = []
    for tool in ("ssh", "rsync", "ssh-keygen"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            problems.append(f"{tool} not found on this Mac")
    return problems


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
