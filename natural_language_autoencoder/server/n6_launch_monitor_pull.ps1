#Requires -Version 5.1
<#
.SYNOPSIS
Safely stages, launches, monitors, and retrieves the frozen N6 remote run.

.DESCRIPTION
Dry-run is the default. Pass -Execute to permit system ssh.exe/scp.exe calls.
The launcher uploads only the frozen preregistration and manifests, their
required sidecars, and code entries named by the frozen N6 code manifest. It
never performs the normal shutdown; the remote supervisor owns that lifecycle.
The optional emergency fallback uses the same literal power-off command as the
supervisor.

For a syntax/shape-only rehearsal when binding inputs do not exist yet, use:
  ./n6_launch_monitor_pull.ps1 -DryRun -SkipLocalValidation
#>
[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $Execute,
    [switch] $SkipLocalValidation,
    [switch] $EmergencyShutdownOnFailure,

    [ValidatePattern('^[A-Za-z0-9_.-]+$')]
    [string] $SshHost = 'autodl',

    [ValidateRange(1, 30)]
    [int] $PollSeconds = 20,

    [ValidateRange(1, 604800)]
    [int] $MaxWaitSeconds = 86400,

    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string] $RemoteRoot = '/root/autodl-tmp',

    [string] $LocalCodeRoot,
    [string] $LocalStagingRoot,
    [string] $LocalNlaInferencePath,

    [string] $PreregPath,
    [string] $CodeManifestPath,
    [string] $N5ModelManifestPath,

    [string] $RemoteSourceCorpusPath,
    [string] $RemoteSourceCorpusManifestPath,
    [string] $RemoteN4ActivationsPath,
    [string] $RemoteN5CohortPlanPath,
    [string] $RemotePileParquetPath,

    [int] $AnalysisTarget = 400,
    [int] $CellSeedQuota = 2,
    [int] $MinCellSize = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$resolvedScriptRoot = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($LocalCodeRoot)) {
    $LocalCodeRoot = $resolvedScriptRoot
}
if ([string]::IsNullOrWhiteSpace($LocalStagingRoot)) {
    $LocalStagingRoot = Join-Path (
        Split-Path -Parent $resolvedScriptRoot
    ) 'results\n6_pull_staging'
}
if ([string]::IsNullOrWhiteSpace($LocalNlaInferencePath)) {
    $LocalNlaInferencePath = Join-Path (
        Split-Path -Parent (Split-Path -Parent $resolvedScriptRoot)
    ) 'nla-from-autodl\natural_language_autoencoders\nla_inference.py'
}

if ($DryRun -and $Execute) {
    throw 'Choose either -DryRun or -Execute, not both.'
}
$script:IsDryRun = -not $Execute
$remoteCodeRoot = "$RemoteRoot/nla_compare"
$remoteNlaRepo = "$RemoteRoot/nla_repo"
$remoteResultsRoot = "$RemoteRoot/results"
$remoteReadyFile = "$remoteResultsRoot/n6_pull_ready_v1.txt"
$remoteStatusFile = "$remoteResultsRoot/n6_supervisor_v1.exit"
$remoteAckFile = "$remoteResultsRoot/n6_pull_ack_v1.txt"
if ([string]::IsNullOrWhiteSpace($RemoteSourceCorpusPath)) {
    $RemoteSourceCorpusPath = "$remoteResultsRoot/n3_corpus_v1.jsonl"
}
if ([string]::IsNullOrWhiteSpace($RemoteSourceCorpusManifestPath)) {
    $RemoteSourceCorpusManifestPath = "$remoteResultsRoot/n3_corpus_v1.json"
}
if ([string]::IsNullOrWhiteSpace($RemoteN4ActivationsPath)) {
    $RemoteN4ActivationsPath = "$RemoteRoot/activations/acts_L32_n3_v1.parquet"
}
if ([string]::IsNullOrWhiteSpace($RemoteN5CohortPlanPath)) {
    $RemoteN5CohortPlanPath = "$remoteResultsRoot/n5_cohort_plan_v2.json"
}
if ([string]::IsNullOrWhiteSpace($RemotePileParquetPath)) {
    $RemotePileParquetPath = (
        "$RemoteRoot/hf/hub/datasets--NeelNanda--pile-10k/blobs/" +
        'a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31'
    )
}

function Write-Plan {
    param([Parameter(Mandatory)][string] $Message)
    Write-Host "[n6-launch] $Message"
}

function Assert-SafeRemotePath {
    param([Parameter(Mandatory)][string] $Path)
    if ($Path -notmatch '^/[A-Za-z0-9_./-]+$' -or $Path.Contains('..')) {
        throw "Unsafe remote path: $Path"
    }
}

function Format-NativeArgument {
    param([Parameter(Mandatory)][string] $Value)
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string] $Program,
        [Parameter(Mandatory)][string[]] $ArgumentList,
        [switch] $AllowFailure
    )
    $display = @($Program) + ($ArgumentList | ForEach-Object { Format-NativeArgument $_ })
    Write-Plan ($display -join ' ')
    if ($script:IsDryRun) {
        return [pscustomobject]@{ ExitCode = 0; Output = @() }
    }

    $lines = @(& $Program @ArgumentList 2>&1 | ForEach-Object { "$_" })
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "$Program failed with exit code $exitCode`: $($lines -join [Environment]::NewLine)"
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $lines }
}

function Resolve-ExistingFile {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $Label
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "$Label path is required."
    }
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.PSIsContainer) {
        throw "$Label is not a regular file: $Path"
    }
    return $item.FullName
}

function Get-Sha256 {
    param([Parameter(Mandatory)][string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-FrozenSidecar {
    param(
        [Parameter(Mandatory)][string] $Path,
        [Parameter(Mandatory)][string] $Label
    )
    $resolved = Resolve-ExistingFile $Path $Label
    $sidecar = Resolve-ExistingFile "$resolved.sha256" "$Label SHA-256 sidecar"
    $lines = @(Get-Content -LiteralPath $sidecar)
    if ($lines.Count -ne 1) {
        throw "$Label sidecar must contain exactly one line: $sidecar"
    }
    $match = [regex]::Match(
        $lines[0],
        '^(?<hash>[0-9A-Fa-f]{64})[ \t]+(?<name>[^ \t]+)$'
    )
    if (-not $match.Success) {
        throw "$Label sidecar must be '<sha256>  <basename>': $sidecar"
    }
    if ($match.Groups['name'].Value -ne [IO.Path]::GetFileName($resolved)) {
        throw "$Label sidecar names the wrong file: $sidecar"
    }
    $actual = Get-Sha256 $resolved
    if ($actual -ne $match.Groups['hash'].Value.ToLowerInvariant()) {
        throw "$Label sidecar hash mismatch: $resolved"
    }
    return [pscustomobject]@{
        Path = $resolved
        Sidecar = $sidecar
        Hash = $actual
    }
}

function Read-CodeManifest {
    param(
        [Parameter(Mandatory)][string] $Manifest,
        [Parameter(Mandatory)][string] $CodeRoot,
        [Parameter(Mandatory)][string] $NlaInferencePath
    )
    $root = (Resolve-Path -LiteralPath $CodeRoot).Path.TrimEnd('\', '/')
    $entries = [Collections.Generic.List[object]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal
    )
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        $lineNumber++
        if ([string]::IsNullOrWhiteSpace($line)) {
            throw "Blank line in N6 code manifest at line $lineNumber."
        }
        $match = [regex]::Match(
            $line,
            '^(?<hash>[0-9A-Fa-f]{64})[ \t]+[*]?(?<name>[A-Za-z0-9][A-Za-z0-9._-]*)$'
        )
        if (-not $match.Success) {
            throw "Malformed or non-basename code entry at line $lineNumber."
        }
        $name = $match.Groups['name'].Value
        if ([IO.Path]::GetExtension($name) -notin @('.py', '.sh')) {
            throw "N6 code manifest may upload only .py/.sh scripts: $name"
        }
        if (-not $seen.Add($name)) {
            throw "Duplicate N6 code manifest entry: $name"
        }
        if ($name -eq 'nla_inference.py') {
            $path = Resolve-ExistingFile $NlaInferencePath "N6 code entry $name"
        } else {
            $path = Resolve-ExistingFile (Join-Path $root $name) "N6 code entry $name"
        }
        $actual = Get-Sha256 $path
        $expected = $match.Groups['hash'].Value.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "N6 code entry hash mismatch: $name"
        }
        $entries.Add([pscustomobject]@{
            Name = $name
            Path = $path
            Hash = $actual
        })
    }
    if ($entries.Count -eq 0) {
        throw 'The N6 code manifest has no entries.'
    }

    $required = @(
        'n6_stage_common.sh',
        'n6_supervisor_template.sh',
        'n6_freeze_cohort_remote.sh',
        'n6_extract_activations_remote.sh',
        'n6_generate_av_remote.sh',
        'n6_freeze_variants_donor_remote.sh',
        'n6_reconstruct_remote.sh',
        'n6_causal_candidate_mass_remote.sh',
        'n6_analyze_remote.sh',
        'n6_independent_audit_remote.sh',
        'n6_common.py',
        'pilot_common.py',
        'nla_inference.py',
        '49_n6_freeze_cohort.py',
        '50_n6_extract_activations.py',
        '51_n6_generate_av.py',
        '52_n6_freeze_variants.py',
        '53_n6_reconstruct.py',
        '54_n6_causal_patch.py',
        '55_n6_analyze.py',
        '56_n6_independent_audit.py'
    )
    $missing = @($required | Where-Object { -not $seen.Contains($_) })
    if ($missing.Count -gt 0) {
        throw "N6 code manifest is missing required scripts: $($missing -join ', ')"
    }
    return @($entries)
}

function Add-FrozenUpload {
    param(
        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [Collections.Generic.List[object]] $List,
        [Parameter(Mandatory)][object] $Frozen,
        [Parameter(Mandatory)][string] $RemoteDirectory
    )
    foreach ($path in @($Frozen.Path, $Frozen.Sidecar)) {
        $name = [IO.Path]::GetFileName($path)
        if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
            throw "Unsafe upload basename: $name"
        }
        $List.Add([pscustomobject]@{
            Local = $path
            Remote = "$RemoteDirectory/$name"
            Hash = Get-Sha256 $path
        })
    }
}

function Test-DownloadedSidecars {
    param([Parameter(Mandatory)][string] $Directory)
    $sidecars = @(Get-ChildItem -LiteralPath $Directory -File -Filter '*.sha256')
    if ($sidecars.Count -eq 0) {
        throw "No SHA-256 sidecars were retrieved into $Directory"
    }
    foreach ($sidecar in $sidecars) {
        $targetPath = $sidecar.FullName.Substring(0, $sidecar.FullName.Length - 7)
        if (-not (Test-Path -LiteralPath $targetPath -PathType Leaf)) {
            throw "Downloaded sidecar has no target: $($sidecar.FullName)"
        }
        [void](Test-FrozenSidecar $targetPath "downloaded artifact")
    }
}

Assert-SafeRemotePath $RemoteRoot
Assert-SafeRemotePath $remoteCodeRoot
Assert-SafeRemotePath $remoteNlaRepo
Assert-SafeRemotePath $remoteResultsRoot
foreach ($remoteLegacyInput in @(
    $RemoteSourceCorpusPath,
    $RemoteSourceCorpusManifestPath,
    $RemoteN4ActivationsPath,
    $RemoteN5CohortPlanPath,
    $RemotePileParquetPath
)) {
    Assert-SafeRemotePath $remoteLegacyInput
}

if ($script:IsDryRun -and $SkipLocalValidation) {
    Write-Plan 'DRY RUN / shape-only: local frozen-file validation is intentionally skipped.'
    [void](Invoke-Native ssh.exe @(
        $SshHost,
        "umask 077; mkdir -p $remoteCodeRoot $remoteNlaRepo $remoteResultsRoot"
    ))
    Write-Plan 'scp.exe <verified frozen prereg/manifests/sidecars/scripts> autodl:<allowlisted destination>'
    [void](Invoke-Native ssh.exe @(
        $SshHost,
        "cd $remoteCodeRoot && nohup env N6_PREREG=<binding-prereg> N6_CODE_MANIFEST=<code-manifest> N6_MODEL_MANIFEST=<model-manifest> N6_SOURCE_CORPUS=$RemoteSourceCorpusPath N6_SOURCE_CORPUS_MANIFEST=$RemoteSourceCorpusManifestPath N6_N4_ACTIVATIONS=$RemoteN4ActivationsPath N6_N5_COHORT_PLAN=$RemoteN5CohortPlanPath N6_PILE_PARQUET=$RemotePileParquetPath N6_ANALYSIS_TARGET=<required> N6_CELL_SEED_QUOTA=<required> N6_MIN_CELL_SIZE=<required> bash $remoteCodeRoot/n6_supervisor_template.sh >$remoteResultsRoot/n6_supervisor_bootstrap_v1.log 2>&1 </dev/null &"
    ))
    [void](Invoke-Native ssh.exe @(
        $SshHost,
        "if [ -f $remoteReadyFile ]; then cat $remoteReadyFile; elif [ -f $remoteStatusFile ]; then printf '__N6_STATUS__='; cat $remoteStatusFile; fi"
    ))
    Write-Plan "Poll interval is $PollSeconds seconds (validated <= 30)."
    Write-Plan "Would list and pull only $remoteResultsRoot/n6_* and $remoteResultsRoot/N6_*."
    Write-Plan 'Would verify every downloaded sidecar and all hashes advertised by the pull-ready file.'
    Write-Plan "Would write the exact advertised token to $remoteResultsRoot/n6_pull_ack_v1.txt."
    Write-Plan 'No normal shutdown is issued by this launcher.'
    return
}

if ($SkipLocalValidation) {
    throw '-SkipLocalValidation is permitted only in dry-run mode.'
}
if ($PreregPath -like '*.DRAFT*') {
    throw "Refusing a draft preregistration: $PreregPath"
}
if ($AnalysisTarget -ne 400) {
    throw '-AnalysisTarget is frozen at 400.'
}
if ($CellSeedQuota -ne 2) {
    throw '-CellSeedQuota is frozen at 2.'
}
if ($MinCellSize -ne 2) {
    throw '-MinCellSize is frozen at 2.'
}

$ssh = (Get-Command ssh.exe -CommandType Application -ErrorAction Stop).Source
$scp = (Get-Command scp.exe -CommandType Application -ErrorAction Stop).Source
$prereg = Test-FrozenSidecar $PreregPath 'binding N6 preregistration'
$codeManifest = Test-FrozenSidecar $CodeManifestPath 'frozen N6 code manifest'
$modelManifest = Test-FrozenSidecar $N5ModelManifestPath 'frozen N5 model manifest'
$codeEntries = Read-CodeManifest (
    $codeManifest.Path
) $LocalCodeRoot $LocalNlaInferencePath

$uploads = [Collections.Generic.List[object]]::new()
Add-FrozenUpload $uploads $prereg $remoteResultsRoot
Add-FrozenUpload $uploads $codeManifest $remoteResultsRoot
Add-FrozenUpload $uploads $modelManifest $remoteResultsRoot
foreach ($entry in $codeEntries) {
    $remoteEntryPath = if ($entry.Name -eq 'nla_inference.py') {
        "$remoteNlaRepo/nla_inference.py"
    } else {
        "$remoteCodeRoot/$($entry.Name)"
    }
    $uploads.Add([pscustomobject]@{
        Local = $entry.Path
        Remote = $remoteEntryPath
        Hash = $entry.Hash
    })
}
$duplicateRemote = @(
    $uploads |
        Group-Object Remote |
        Where-Object Count -gt 1
)
if ($duplicateRemote.Count -gt 0) {
    throw "Upload destinations collide: $(($duplicateRemote.Name) -join ', ')"
}

$remotePrereg = "$remoteResultsRoot/$([IO.Path]::GetFileName($prereg.Path))"
$remoteCodeManifest = "$remoteResultsRoot/$([IO.Path]::GetFileName($codeManifest.Path))"
$remoteModelManifest = "$remoteResultsRoot/$([IO.Path]::GetFileName($modelManifest.Path))"
foreach ($remotePath in @(
    $remotePrereg,
    $remoteCodeManifest,
    $remoteModelManifest
)) {
    Assert-SafeRemotePath $remotePath
}

try {
    [void](Invoke-Native $ssh @(
        $SshHost,
        "umask 077; mkdir -p $remoteCodeRoot $remoteNlaRepo $remoteResultsRoot"
    ))

    foreach ($upload in $uploads) {
        Assert-SafeRemotePath $upload.Remote
        [void](Invoke-Native $ssh @(
            $SshHost,
            "if [ -e $($upload.Remote) ]; then printf '%s  %s\n' $($upload.Hash) $($upload.Remote) | sha256sum --strict --check -; fi"
        ))
        [void](Invoke-Native $scp @(
            '-p',
            $upload.Local,
            "$SshHost`:$($upload.Remote)"
        ))
        [void](Invoke-Native $ssh @(
            $SshHost,
            "printf '%s  %s\n' $($upload.Hash) $($upload.Remote) | sha256sum --strict --check -"
        ))
    }

    $remoteLaunch = (
        "cd $remoteCodeRoot && " +
        'nohup env ' +
        "N6_PREREG=$remotePrereg " +
        "N6_CODE_MANIFEST=$remoteCodeManifest " +
        "N6_MODEL_MANIFEST=$remoteModelManifest " +
        "N6_SOURCE_CORPUS=$RemoteSourceCorpusPath " +
        "N6_SOURCE_CORPUS_MANIFEST=$RemoteSourceCorpusManifestPath " +
        "N6_N4_ACTIVATIONS=$RemoteN4ActivationsPath " +
        "N6_N5_COHORT_PLAN=$RemoteN5CohortPlanPath " +
        "N6_PILE_PARQUET=$RemotePileParquetPath " +
        "N6_ANALYSIS_TARGET=$AnalysisTarget " +
        "N6_CELL_SEED_QUOTA=$CellSeedQuota " +
        "N6_MIN_CELL_SIZE=$MinCellSize " +
        "bash $remoteCodeRoot/n6_supervisor_template.sh " +
        ">$remoteResultsRoot/n6_supervisor_bootstrap_v1.log 2>&1 </dev/null &"
    )
    [void](Invoke-Native $ssh @(
        $SshHost,
        "test ! -e $remoteReadyFile && test ! -e $remoteStatusFile && test ! -e $remoteAckFile"
    ))
    [void](Invoke-Native $ssh @($SshHost, $remoteLaunch))

    if ($script:IsDryRun) {
        Write-Plan "Would poll every $PollSeconds seconds, pull and verify n6_*/N6_*, then write the exact ack token."
        Write-Plan 'No normal shutdown is issued by this launcher.'
        return
    }

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($MaxWaitSeconds)
    $readyLines = $null
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $poll = Invoke-Native $ssh @(
            $SshHost,
            "if [ -f $remoteReadyFile ]; then cat $remoteReadyFile; elif [ -f $remoteStatusFile ]; then printf '__N6_STATUS__='; cat $remoteStatusFile; fi"
        ) -AllowFailure
        if ($poll.ExitCode -eq 0 -and $poll.Output.Count -gt 0) {
            $statusLine = @($poll.Output | Where-Object { $_ -match '^__N6_STATUS__=' })
            $tokenLine = @($poll.Output | Where-Object { $_ -match '^ack_token=' })
            if ($tokenLine.Count -gt 0) {
                $readyLines = @($poll.Output)
                break
            }
            if ($statusLine.Count -gt 0) {
                $remoteExit = ($statusLine[0] -split '=', 2)[1].Trim()
                if ($remoteExit -ne '0') {
                    throw "Remote N6 supervisor failed with exit code $remoteExit."
                }
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if ($null -eq $readyLines) {
        throw "Timed out after $MaxWaitSeconds seconds waiting for N6 pull readiness."
    }

    $ready = @{}
    foreach ($line in $readyLines) {
        if ($line -match '^(?<key>[a-z_]+)=(?<value>.*)$') {
            $ready[$Matches['key']] = $Matches['value']
        }
    }
    foreach ($key in @(
        'ack_token',
        'analysis',
        'analysis_sha256',
        'independent_audit',
        'independent_audit_sha256',
        'resource_report',
        'resource_report_sha256',
        'ack_file'
    )) {
        if (-not $ready.ContainsKey($key)) {
            throw "Pull-ready file is missing $key."
        }
    }
    if ($ready.ack_token -notmatch '^[0-9a-f]{64}$') {
        throw 'Pull-ready ack token is malformed.'
    }
    if ($ready.ack_file -ne "$remoteResultsRoot/n6_pull_ack_v1.txt") {
        throw "Pull-ready ack path is unexpected: $($ready.ack_file)"
    }

    $artifactList = Invoke-Native $ssh @(
        $SshHost,
        "find $remoteResultsRoot -maxdepth 1 -type f \( -name 'n6_*' -o -name 'N6_*' \) -print"
    )
    $remoteArtifacts = @(
        $artifactList.Output |
            Where-Object { $_ -match "^$([regex]::Escape($remoteResultsRoot))/(?:n6_|N6_)[A-Za-z0-9._-]+$" } |
            Sort-Object -Unique
    )
    if ($remoteArtifacts.Count -eq 0) {
        throw 'Remote pull-ready run advertised no n6_*/N6_* files.'
    }

    $stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $localStage = Join-Path $LocalStagingRoot "n6_pull_$stamp"
    [void](New-Item -ItemType Directory -Path $localStage -Force)
    foreach ($remoteArtifact in $remoteArtifacts) {
        [void](Invoke-Native $scp @(
            '-p',
            "$SshHost`:$remoteArtifact",
            $localStage
        ))
    }

    Test-DownloadedSidecars $localStage
    foreach ($advertised in @(
        [pscustomobject]@{ Path = $ready.analysis; Hash = $ready.analysis_sha256 },
        [pscustomobject]@{ Path = $ready.independent_audit; Hash = $ready.independent_audit_sha256 },
        [pscustomobject]@{ Path = $ready.resource_report; Hash = $ready.resource_report_sha256 }
    )) {
        Assert-SafeRemotePath $advertised.Path
        if ($advertised.Hash -notmatch '^[0-9a-f]{64}$') {
            throw "Malformed advertised hash for $($advertised.Path)."
        }
        $localPath = Join-Path $localStage ([IO.Path]::GetFileName($advertised.Path))
        if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
            throw "Advertised artifact was not pulled: $($advertised.Path)"
        }
        if ((Get-Sha256 $localPath) -ne $advertised.Hash) {
            throw "Advertised hash mismatch after pull: $($advertised.Path)"
        }
    }

    $ackCommand = (
        'umask 077; tmp={0}.tmp.$$; ' +
        "printf '%s\n' {1} > `"`$tmp`" && mv `"`$tmp`" {0}"
    ) -f $ready.ack_file, $ready.ack_token
    [void](Invoke-Native $ssh @($SshHost, $ackCommand))
    Write-Plan "Verified pull completed: $localStage"
    Write-Plan 'Exact remote acknowledgement written; the supervisor remains responsible for power-off.'
}
catch {
    if (-not $script:IsDryRun -and $EmergencyShutdownOnFailure) {
        Write-Warning 'Launcher failed; invoking the explicitly requested emergency power-off fallback.'
        [void](Invoke-Native $ssh @(
            $SshHost,
            'sync; /usr/bin/shutdown -h now'
        ) -AllowFailure)
    }
    throw
}
