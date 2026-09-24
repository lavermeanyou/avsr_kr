# Main PC: copy the parts preprocessed on other PCs into the work folder, then show its status (per speaker, missing
# files). Thin wrapper around `tools/shards.py merge` (see README.md section 12).
#   .\scripts\merge_shards.ps1 -Src E:\avsr_parts                         # a folder holding avsr_part2, avsr_part3, ...
#   .\scripts\merge_shards.ps1 -Src E:\avsr_part2,F:\avsr_part3,\\PC4\share\avsr_part4
#   .\scripts\merge_shards.ps1 -IncludeLocal                              # the work_shard* folders in the project
#   .\scripts\merge_shards.ps1 -Src E:\avsr_parts -DryRun                 # only show what would be copied
# -Src: part folders (or folders containing them); a value may also list several folders separated by ';'.
# -IncludeLocal: also merge <project>\work_shard* folders. -WorkDir: default work. Relative paths are relative to the
# project root. -Overwrite: replace videos that are already finished in the work folder (default: keep them).
# -Checksum: also compare SHA-1 of every copied file. Source folders are never modified.
# Set $env:AVSR_PYTHON to use another python.exe. Exit code: 0 = OK, 1 = problems (listed), 2 = nothing to merge.
param(
    [string[]]$Src = @(),
    [string]$WorkDir = 'work',
    [switch]$IncludeLocal,
    [switch]$Overwrite,
    [switch]$Checksum,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path -LiteralPath $py)) { throw "python.exe not found: $py (run 1_setup.bat, or set `$env:AVSR_PYTHON)" }
$root = Split-Path -Parent $PSScriptRoot

$prevIoEnc = $env:PYTHONIOENCODING
$prevConsoleEnc = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location -LiteralPath $root
$code = 2
try {
    $folders = New-Object System.Collections.Generic.List[string]
    foreach ($s in $Src) {
        foreach ($part in ($s -split ';')) {
            # strip blanks/quotes and a trailing \ or \. (Windows PowerShell 5.1 breaks native arguments like "D:\my dir\")
            $p = $part.Trim().Trim('"').Trim()
            if ($p.Contains('"')) {
                throw "broken path argument: $p`n  A quoted path must not end with a backslash (write ""D:\my dir"", not ""D:\my dir\"")."
            }
            if ($p.EndsWith('\.')) { $p = $p.Substring(0, $p.Length - 1) }
            if ($p -match '^[A-Za-z]:$') { $p += '\' }  # "E:" = the drive root (not the current folder of drive E:)
            if ($p.Length -gt 3) { $p = $p.TrimEnd('\') }
            if ($p) { $folders.Add($p) }
        }
    }
    if ($IncludeLocal) {
        $workFull = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($root, $WorkDir))
        Get-ChildItem -LiteralPath $root -Directory -Filter 'work_shard*' | Sort-Object Name | ForEach-Object {
            if ((Test-Path -LiteralPath (Join-Path $_.FullName 'manifests')) -and ($_.FullName -ne $workFull)) {
                Write-Host "adding local part folder: $($_.FullName)"
                $folders.Add($_.FullName)
            }
        }
    }
    if ($folders.Count -eq 0) {
        Write-Host 'No part folders to merge: give -Src <folder> (USB drive, network folder) or put work_shard* folders here.'
    } else {
        $ErrorActionPreference = 'Continue'  # stderr lines must not abort, even with 2>&1
        $argv = @('tools/shards.py', 'merge', '--work-dir', $WorkDir)
        foreach ($f in $folders) { $argv += @('--src', $f) }
        if ($Overwrite) { $argv += '--overwrite' }
        if ($Checksum) { $argv += '--checksum' }
        if ($DryRun) { $argv += '--dry-run' }
        & $py @argv
        $code = $LASTEXITCODE
    }
}
finally {
    Pop-Location
    $env:PYTHONIOENCODING = $prevIoEnc
    [Console]::OutputEncoding = $prevConsoleEnc
}
exit $code
