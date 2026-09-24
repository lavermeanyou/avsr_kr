# Preprocess this PC's part of the dataset (several PCs share the work), then optionally copy the finished part to a
# transfer folder (USB drive or network share) for the main PC. Thin wrapper around `python -m avsr.preprocess
# --shard K/N` and `tools/shards.py pack` (see README.md section 12).
#   .\scripts\preprocess_shard.ps1 -Shard 2 -DataRoot "D:\data\009.<dataset folder>" -Dest E:\avsr_parts
#   .\scripts\preprocess_shard.ps1 -Shard 1 -WorkDir work -DataRoot "..."    # main PC: straight into the work folder
#   .\scripts\preprocess_shard.ps1 -Shard 3 -DataRoot "..." -DryRun           # only list this PC's videos
#   .\scripts\preprocess_shard.ps1 -Shard 2 -NoSplit -DataRoot "..."          # PCs hold DIFFERENT videos: do all here
# -Shard K -NumShards N (default 4): all PCs must hold the same dataset; PC K takes every N-th video (sorted by name).
# -DataRoot: default = the only folder named "009.*" in the Downloads folder.
# -WorkDir: default work_shardK. Re-running resumes (finished videos are skipped).
# -Dest: the part is copied to <Dest>\avsr_partK (files already there with the same size are skipped).
# Relative paths are relative to the project root.
# Other arguments go to avsr.preprocess unchanged, e.g. --workers 10 --decoder cpu.
# Set $env:AVSR_PYTHON to use another python.exe. Exit code: 0 = OK, otherwise the first failing step's code.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][int]$Shard,
    [int]$NumShards = 4,
    [string]$DataRoot = '',
    [string]$WorkDir = '',
    [string]$Dest = '',
    [switch]$NoSplit,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path -LiteralPath $py)) { throw "python.exe not found: $py (run 1_setup.bat, or set `$env:AVSR_PYTHON)" }

function Get-AsciiModelCopy([string]$Model) {
    # MediaPipe cannot open a model file whose path has non-ASCII (e.g. Korean) characters: use a copy in an
    # ASCII-only folder (C:\ProgramData\avsr_kr, C:\Users\Public\avsr_kr or C:\avsr_kr) in that case
    if ($Model -notmatch '[^\x00-\x7F]') { return $Model }
    foreach ($base in @($env:ProgramData, $env:PUBLIC, "$env:SystemDrive\")) {
        if (-not $base -or $base -match '[^\x00-\x7F]') { continue }
        $dir = Join-Path $base 'avsr_kr'
        $copy = Join-Path $dir (Split-Path -Leaf $Model)
        try {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            if (-not (Test-Path -LiteralPath $copy) -or (Get-Item -LiteralPath $copy).Length -ne (Get-Item -LiteralPath $Model).Length) {
                Copy-Item -LiteralPath $Model -Destination $copy -Force
            }
            return $copy
        } catch {
            continue
        }
    }
    return $Model
}

function Clean-PathArg([string]$p) {
    # strip blanks/quotes and a trailing \ or \. (Windows PowerShell 5.1 breaks native arguments like "D:\my dir\")
    $p = $p.Trim().Trim('"').Trim()
    if ($p.Contains('"')) {
        throw "broken path argument: $p`n  A quoted path must not end with a backslash (write ""D:\my dir"", not ""D:\my dir\"")."
    }
    if ($p.EndsWith('\.')) { $p = $p.Substring(0, $p.Length - 1) }
    if ($p -match '^[A-Za-z]:$') { $p += '\' }  # "E:" = the drive root (not the current folder of drive E:)
    if ($p.Length -gt 3) { $p = $p.TrimEnd('\') }
    return $p
}

if ($NoSplit) {
    if ($Shard -lt 1) { throw "-Shard must be 1 or more (it names this PC's part), got $Shard" }
} elseif ($NumShards -lt 1 -or $Shard -lt 1 -or $Shard -gt $NumShards) {
    throw "-Shard must be between 1 and -NumShards ($NumShards), got $Shard"
}
$prevIoEnc = $env:PYTHONIOENCODING
$prevConsoleEnc = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$t0 = Get-Date
$code = 0
try {
    $DataRoot = Clean-PathArg $DataRoot
    if (-not $DataRoot) {
        $downloads = Join-Path $env:USERPROFILE 'Downloads'
        $found = @(Get-ChildItem -LiteralPath $downloads -Directory -Filter '009.*' -ErrorAction SilentlyContinue)
        if ($found.Count -ne 1) {
            $names = ($found | ForEach-Object { $_.FullName }) -join '; '
            throw "Give the dataset folder with -DataRoot: found $($found.Count) folders named 009.* in $downloads $names"
        }
        $DataRoot = $found[0].FullName
        Write-Host "Data folder (from Downloads): $DataRoot"
    }
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { throw "data folder not found: $DataRoot" }
    $WorkDir = Clean-PathArg $WorkDir
    if (-not $WorkDir) { $WorkDir = "work_shard$Shard" }
    $Dest = Clean-PathArg $Dest
    $target = ''
    if ($Dest -and -not $DryRun) {
        # check the transfer folder now, not after the preprocessing (Join-Path would also fail for a missing drive)
        $destFull = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine((Get-Location).Path, $Dest))
        $destRoot = [System.IO.Path]::GetPathRoot($destFull)
        if (-not $destRoot -or -not (Test-Path -LiteralPath $destRoot -PathType Container)) {
            throw "transfer folder not reachable: $Dest (drive or network share '$destRoot' not found). Plug in the USB drive or check the network folder, or leave the transfer folder empty."
        }
        $target = [System.IO.Path]::Combine($destFull, "avsr_part$Shard")
    }

    $ErrorActionPreference = 'Continue'  # stderr lines (logs, progress bars) must not abort, even with 2>&1
    $pre = @('-m', 'avsr.preprocess', '--data-root', $DataRoot, '--work-dir', $WorkDir)
    if ($NoSplit) {
        Write-Host "== PC $Shard : preprocessing ALL videos of $DataRoot into $WorkDir =="
    } else {
        $pre += @('--shard', "$Shard/$NumShards")
        Write-Host "== PC $Shard of $NumShards : preprocessing part $Shard/$NumShards of $DataRoot into $WorkDir =="
    }
    if ($DryRun) { $pre += '--dry-run' }
    if (-not ($Rest | Where-Object { "$_" -like '--model*' })) {
        $model = Join-Path (Get-Location).Path 'assets\face_landmarker.task'
        $useModel = Get-AsciiModelCopy $model
        if ($useModel -ne $model) {
            Write-Host "Note: the project folder path has non-English letters, which MediaPipe cannot open; using a copy of the face model: $useModel"
            $pre += @('--model', $useModel)
        }
    }
    & $py @pre @Rest
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Host "Preprocessing finished with failures (exit code $code, $mins min). Run the same command again to retry them."
    } else {
        Write-Host "Preprocessing done ($mins min)."
    }
    if ($target) {
        Write-Host ""
        Write-Host "== copying the finished videos to $target =="
        & $py 'tools/shards.py' 'pack' '--work-dir' $WorkDir '--dest' $target
        $packCode = $LASTEXITCODE
        if ($code -eq 0) { $code = $packCode }
    } elseif (-not $DryRun) {
        & $py 'tools/shards.py' 'status' '--work-dir' $WorkDir
        Write-Host ""
        Write-Host "The result is in $([System.IO.Path]::GetFullPath([System.IO.Path]::Combine((Get-Location).Path, $WorkDir)))"
        if ($WorkDir -ne 'work') {
            Write-Host "Copy that folder to the main PC and run 3_merge_parts.bat there (give the folder when asked)."
        }
    }
}
finally {
    Pop-Location
    $env:PYTHONIOENCODING = $prevIoEnc
    [Console]::OutputEncoding = $prevConsoleEnc
}
exit $code
