# Codex AutoDL Runbook

This is the operational workflow Codex should use to edit locally, sync to the
AutoDL server, run the GPU code, and fetch results.

In the Codex sandbox, bare `ssh` / `scp` may resolve to deny wrappers. Use the
Windows OpenSSH binaries explicitly:

```text
C:\Windows\System32\OpenSSH\ssh.exe
C:\Windows\System32\OpenSSH\scp.exe
```

On the remote AutoDL machine, use `/root/miniconda3/bin/python` explicitly;
non-interactive SSH sessions may not have `python` on `PATH`.

## Required Login Setup

Remote SSH endpoint is fixed:

```powershell
ssh -p 19956 root@connect.westb.seetacloud.com
```

Codex cannot safely or reliably pass an interactive SSH password prompt during
automated `ssh` / `scp` tool runs. Native Windows OpenSSH also has no safe
`--password` flag. For Codex to run commands normally, passwordless SSH key
login must be enabled once.

The user should run this one-time setup locally:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\autodl_nla"
type "$env:USERPROFILE\.ssh\autodl_nla.pub" | ssh -p 19956 root@connect.westb.seetacloud.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```

When `ssh-keygen` asks for a passphrase, press Enter twice to leave it empty.
The second command asks for the AutoDL password once. After that, Codex should
use the key explicitly:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "hostname"
```

Optional SSH config alias:

```sshconfig
Host autodl-nla
    HostName connect.westb.seetacloud.com
    Port 19956
    User root
    IdentityFile ~/.ssh/autodl_nla
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

If this alias exists, Codex can use `ssh autodl-nla "..."` and `scp ... autodl-nla:...`.

## Fixed Paths

Local repository:

```text
D:\Projects\nla-from-autodl\natural_language_autoencoders
```

Remote repository:

```text
/root/autodl-tmp/natural_language_autoencoders
```

Remote models:

```text
/root/autodl-tmp/models/Qwen2.5-7B-Instruct
/root/autodl-tmp/models/nla-qwen-av
/root/autodl-tmp/models/nla-qwen-ar
```

Remote result directory:

```text
/root/autodl-tmp
```

Local result directory:

```text
D:\Projects\nla-from-autodl\natural_language_autoencoders\remote_results
```

## Connectivity Check

Codex should run:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "hostname; pwd; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv"
```

If SSH asks for a password, key login is not installed yet and the user must
complete the setup in `Required Login Setup`.

## Sync Local Changes To AutoDL

For the current answer-token AV probe work, sync these files:

```powershell
C:\Windows\System32\OpenSSH\scp.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -P 19956 `
  "D:\Projects\nla-from-autodl\natural_language_autoencoders\demo\answer_probe.py" `
  "root@connect.westb.seetacloud.com:/root/autodl-tmp/natural_language_autoencoders/demo/answer_probe.py"

C:\Windows\System32\OpenSSH\scp.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -P 19956 `
  "D:\Projects\nla-from-autodl\natural_language_autoencoders\README.md" `
  "root@connect.westb.seetacloud.com:/root/autodl-tmp/natural_language_autoencoders/README.md"

C:\Windows\System32\OpenSSH\scp.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -P 19956 `
  "D:\Projects\nla-from-autodl\natural_language_autoencoders\docs\codex_autodl_runbook.md" `
  "root@connect.westb.seetacloud.com:/root/autodl-tmp/natural_language_autoencoders/docs/codex_autodl_runbook.md"
```

Before modifying remote state further, Codex should check:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/natural_language_autoencoders && git status --short"
```

Do not reset or overwrite unrelated remote changes.

## Start Or Check AV Server

Start in a persistent tmux session:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "tmux new-session -d -s nla-av 'cd /root/autodl-tmp/natural_language_autoencoders && export PATH=/root/miniconda3/bin:$PATH && AV_DIR=/root/autodl-tmp/models/nla-qwen-av MEM_FRACTION=0.6 bash demo/launch_av_server.sh > /root/autodl-tmp/nla_av_server.log 2>&1'"
```

Check whether the server is reachable:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "curl -fsS http://localhost:30000/get_model_info | head -c 300"
```

If `/health` is unavailable, inspect the tmux pane:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "tail -n 120 /root/autodl-tmp/nla_av_server.log"
```

## Run Answer Probe

AV-only run:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/natural_language_autoencoders && /root/miniconda3/bin/python demo/answer_probe.py \
  --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --av /root/autodl-tmp/models/nla-qwen-av \
  --prompt 'Explain why the Eiffel Tower is famous in two sentences.' \
  --layer-index 20 \
  --positions 0-12 \
  --answer-temperature 0 \
  --av-temperature 0 \
  --out /root/autodl-tmp/answer_probe_av_only.json"
```

Full context export run. This probes formatted prompt tokens, including system /
role / newline tokens, plus the generated answer. It writes AV explanations to a
JSON file instead of printing them to the terminal:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/natural_language_autoencoders && /root/miniconda3/bin/python demo/answer_probe.py \
  --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --av /root/autodl-tmp/models/nla-qwen-av \
  --ar /root/autodl-tmp/models/nla-qwen-ar \
  --prompt 'Explain why the Eiffel Tower is famous in two sentences.' \
  --layer-index 20 \
  --target all \
  --positions all \
  --quiet \
  --answer-temperature 0 \
  --av-temperature 0 \
  --out /root/autodl-tmp/answer_probe_full_context.json"
```

For custom user prompts, replace only the `--prompt` value. For token selection,
use `--positions all`, `--positions 0,3,5-9`, `--start`, and `--n`. Use
`--target question` for only the raw user prompt, `--target answer` for only the
generated answer, `--target prompt` for only the formatted prompt, and
`--target all` for formatted prompt plus answer.

Interactive VSCode Remote SSH TUI:

```bash
cd /root/autodl-tmp/natural_language_autoencoders
/root/miniconda3/bin/python demo/answer_probe_tui.py
```

The TUI exposes prompt, AR scoring, extraction layer, answer temperature, AV
temperature, max answer tokens, and output path. It exports JSON quietly with
`target=all` and `positions=all`. Layer 20 is the normal Qwen NLA setting;
layer 21 is useful for intentional mismatch probes.

## Fetch Results

```powershell
New-Item -ItemType Directory -Force "D:\Projects\nla-from-autodl\natural_language_autoencoders\remote_results"

C:\Windows\System32\OpenSSH\scp.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -P 19956 `
  "root@connect.westb.seetacloud.com:/root/autodl-tmp/answer_probe_av_only.json" `
  "D:\Projects\nla-from-autodl\natural_language_autoencoders\remote_results\answer_probe_av_only.json"

C:\Windows\System32\OpenSSH\scp.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -P 19956 `
  "root@connect.westb.seetacloud.com:/root/autodl-tmp/answer_probe_with_ar.json" `
  "D:\Projects\nla-from-autodl\natural_language_autoencoders\remote_results\answer_probe_with_ar.json"
```

Preview:

```powershell
Get-Content "D:\Projects\nla-from-autodl\natural_language_autoencoders\remote_results\answer_probe_with_ar.json" -TotalCount 120
```

## Commit Remote Changes If Requested

Only commit when the user asks for it:

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/natural_language_autoencoders && git add README.md demo/answer_probe.py docs/codex_autodl_runbook.md && git commit -m 'Add answer-token AV probe'"
```

## Stop AV Server

```powershell
C:\Windows\System32\OpenSSH\ssh.exe -i "$env:USERPROFILE\.ssh\autodl_nla" -p 19956 root@connect.westb.seetacloud.com "tmux send-keys -t nla-av C-c"
```
