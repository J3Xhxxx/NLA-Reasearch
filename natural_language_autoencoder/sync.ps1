# Push local server scripts to the AutoDL box, or pull results back.
#
#   .\sync.ps1 push      # upload server/*  ->  /root/autodl-tmp/nla_compare/
#   .\sync.ps1 pull      # download /root/autodl-tmp/results/  ->  results\
#
# Uses the current `Host autodl` entry in ~/.ssh/config. No password needed.

param([Parameter(Mandatory = $true)][ValidateSet("push", "pull")][string]$dir)

$scp = "C:\Windows\System32\OpenSSH\scp.exe"
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$target = "autodl"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition

$remoteCompare = "/root/autodl-tmp/nla_compare"
$remoteResults = "/root/autodl-tmp/results"

if ($dir -eq "push") {
    & $ssh $target "mkdir -p $remoteCompare"
    & $scp -r "$here\server\*" "${target}:$remoteCompare/"
    Write-Host "pushed server\* -> $remoteCompare/"
}
else {
    New-Item -ItemType Directory -Force "$here\results" | Out-Null
    & $scp -r "${target}:$remoteResults/*" "$here\results\"
    Write-Host "pulled $remoteResults/* -> results\"
}
