# One-click connect to the AutoDL (seetacloud bjb1) GPU server.
#
#   .\connect.ps1                 # open an interactive shell
#   .\connect.ps1 "nvidia-smi"    # run one command and return
#   .\connect.ps1 "df -h /root/autodl-tmp"
#
# Host, port, user, and key are read from the current `Host autodl` entry in
# ~/.ssh/config so cloned instances do not leave stale ports in this script.

$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"

& $ssh `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=4 `
    autodl @args
