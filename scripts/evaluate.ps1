# Evaluation (CER/WER per condition x SNR sweep -> suggested SNR threshold): thin wrapper around
# `python -m avsr.evaluate` (see README.md).
#   .\scripts\evaluate.ps1 --ckpt work\checkpoints\best.pt --split test
#   .\scripts\evaluate.ps1 --ckpt work\checkpoints\best.pt --split val --max-utts 300 --set "eval.snr_sweep=[10,5,0]"
# All arguments are passed through unchanged. The module runs from the project root, so relative paths are resolved
# against the project root. Quote values containing commas (as above). Set $env:AVSR_PYTHON to use another python.exe.
$ErrorActionPreference = 'Stop'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { 'C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe' }
if (-not (Test-Path -LiteralPath $py)) { throw "python.exe not found: $py (set `$env:AVSR_PYTHON)" }
$prevIoEnc = $env:PYTHONIOENCODING
$prevConsoleEnc = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $ErrorActionPreference = 'Continue'  # stderr lines (logs, progress bars) must not abort, even with 2>&1
    & $py -m avsr.evaluate @args
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONIOENCODING = $prevIoEnc
    [Console]::OutputEncoding = $prevConsoleEnc
}
exit $code
