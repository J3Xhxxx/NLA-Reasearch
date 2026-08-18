#!/usr/bin/env python3
"""Remote command runner for the AutoDL (seetacloud) GPU server.

Used both interactively by the operator and programmatically as a helper.
Reads host/port/user/key from the local ``Host autodl`` SSH config entry.
No credential is stored in this repository.

Usage:
    python remote.py "<shell command>"          # run one command, stream output
    python remote.py --put LOCAL REMOTE         # upload a file (sftp)
    python remote.py --get REMOTE LOCAL         # download a file (sftp)
    python remote.py --shell                    # not implemented; use connect.ps1
"""
import sys
import time
from pathlib import Path
import paramiko

# Windows consoles default to GBK; force UTF-8 so emoji/CJK in remote output
# don't crash the helper.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SSH_ALIAS = "autodl"


def _ssh_settings():
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        raise FileNotFoundError(f"SSH config not found: {config_path}")
    config = paramiko.SSHConfig()
    with config_path.open(encoding="utf-8") as handle:
        config.parse(handle)
    entry = config.lookup(SSH_ALIAS)
    identities = [
        str(Path(item).expanduser()) for item in entry.get("identityfile", [])
    ]
    return {
        "hostname": entry.get("hostname", SSH_ALIAS),
        "port": int(entry.get("port", 22)),
        "username": entry.get("user"),
        "key_filename": identities or None,
    }


def _client(retries=5):
    last = None
    settings = _ssh_settings()
    for attempt in range(retries):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(
                **settings,
                timeout=30, banner_timeout=45, auth_timeout=45,
                look_for_keys=True, allow_agent=True,
            )
            return c
        except Exception as e:  # transient banner / rate-limit hiccups
            last = e
            time.sleep(2 + attempt * 2)
    raise last


def run(cmd, get_pty=False):
    c = _client()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, get_pty=get_pty, timeout=None)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return code
    finally:
        c.close()


def run_script(path):
    """Upload a local script and execute it under a login bash shell."""
    remote_path = "/tmp/_nla_run.sh"
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.put(path, remote_path)
        sftp.chmod(remote_path, 0o755)
        sftp.close()
        chan = c.get_transport().open_session()
        chan.exec_command(f"bash -l {remote_path}")
        while True:
            if chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
                sys.stdout.flush()
            if chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(65536).decode("utf-8", "replace"))
                sys.stderr.flush()
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
            time.sleep(0.05)
        return chan.recv_exit_status()
    finally:
        c.close()


def put(local, remote):
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.put(local, remote)
        print(f"uploaded {local} -> {remote}")
    finally:
        c.close()


def get(remote, local):
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.get(remote, local)
        print(f"downloaded {remote} -> {local}")
    finally:
        c.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--put":
        put(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--get":
        get(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--script":
        sys.exit(run_script(sys.argv[2]))
    else:
        sys.exit(run(sys.argv[1]))
