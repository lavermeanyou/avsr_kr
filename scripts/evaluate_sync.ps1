# Speaker-lip matching model evaluation (N-way selection, leakage robustness, sync offset, scene assignment,
# off-screen threshold): thin wrapper around `python -m avsr.sync.evaluate` (see README.md section 11).
#   .\scripts\evaluate_sync.ps1 --split test                  # work/checkpoints_sync/best.pt
#   .\scripts\evaluate_sync.ps1 --ckpt work\checkpoints_sync\epoch_010.pt --split val --max-utts 300
# --ckpt work/checkpoints_sync/best.pt is added unless another --ckpt is given. All other arguments are passed through
# unchanged; relative paths are resolved against the project root. Set $env:AVSR_PYTHON to use another python.exe.
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path -LiteralPath $py)) { throw "python.exe not found: $py (run 1_setup.bat, or set `$env:AVSR_PYTHON)" }
$argv = @($args)
if (-not ($argv | Where-Object { "$_" -like '--ckpt*' })) { $argv = @('--ckpt', 'work/checkpoints_sync/best.pt') + $argv }
$prevIoEnc = $env:PYTHONIOENCODING
$prevConsoleEnc = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $ErrorActionPreference = 'Continue'  # stderr lines (logs, progress bars) must not abort, even with 2>&1
    & $py -m avsr.sync.evaluate @argv
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONIOENCODING = $prevIoEnc
    [Console]::OutputEncoding = $prevConsoleEnc
}
exit $code
