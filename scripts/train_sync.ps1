# Speaker-lip matching (audio-visual sync) model training: thin wrapper around `python -m avsr.sync.train`
# (see README.md section 11).
#   .\scripts\train_sync.ps1                                  # configs/sync.yaml, resumes from last.pt if present
#   .\scripts\train_sync.ps1 --set train.epochs=1 --set data.num_workers=2 --set train.ckpt_dir=work/checkpoints_sync_smoke --limit 256
# --config configs/sync.yaml is added unless another --config is given. All other arguments are passed through
# unchanged; relative paths are resolved against the project root. Quote values containing commas.
# Set $env:AVSR_PYTHON to use another python.exe.
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path -LiteralPath $py)) { throw "python.exe not found: $py (run 1_setup.bat, or set `$env:AVSR_PYTHON)" }
$argv = @($args)
if (-not ($argv | Where-Object { "$_" -like '--config*' })) { $argv = @('--config', 'configs/sync.yaml') + $argv }
$prevIoEnc = $env:PYTHONIOENCODING
$prevConsoleEnc = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $ErrorActionPreference = 'Continue'  # stderr lines (logs, progress bars) must not abort, even with 2>&1
    & $py -m avsr.sync.train @argv
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONIOENCODING = $prevIoEnc
    [Console]::OutputEncoding = $prevConsoleEnc
}
exit $code
