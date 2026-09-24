# One-time setup of a Windows PC (same Windows image as the main PC, NVIDIA RTX GPU) for avsr_kr.
# Safe to run again: every step checks first and installs only what is missing. Log: <project>\setup.log
#   .\scripts\setup.ps1              # check, install what is missing, verify
#   .\scripts\setup.ps1 -CheckOnly   # only check and verify (installs and changes nothing)
# Steps: 1 Python 3.12 (winget, user scope) - 2 FFmpeg (winget Gyan.FFmpeg) - 3 Python packages from requirements.txt
# (torch/torchaudio/torchvision from the CUDA 12.8 index) + msvc-runtime (without it torch fails on this Windows image
# with "WinError 126 ... c10.dll") - 4 MediaPipe model assets\face_landmarker.task - 5 verification (GPU + torch CUDA
# + bf16, mediapipe face landmarker, ffmpeg/ffprobe incl. NVDEC decoder, free disk space).
# Set $env:AVSR_PYTHON to use a python.exe in another place. Exit code: 0 = everything OK, 1 = problems (listed).
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $Root 'setup.log'
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$TorchIndex = 'https://download.pytorch.org/whl/cu128'
$TorchPins = @('torch==2.11.0', 'torchaudio==2.11.0', 'torchvision==0.26.0')
$AssetUrl = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
$script:Problems = New-Object System.Collections.Generic.List[string]
$script:ProgressBarRe = '^\s*[' + [char]0x2580 + '-' + [char]0x259F + ']'   # winget progress bars (block characters)

function Write-Log([string]$Message, [string]$Color = '') {
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    [System.IO.File]::AppendAllText($LogFile, $line + [Environment]::NewLine, $Utf8NoBom)
    if ($Color) { Write-Host $Message -ForegroundColor $Color } else { Write-Host $Message }
}
function Step([string]$Title) { Write-Log ''; Write-Log "== $Title ==" 'Cyan' }
function Ok([string]$Message) { Write-Log "  [OK] $Message" 'Green' }
function Info([string]$Message) { Write-Log "  $Message" }
function Problem([string]$Message) { Write-Log "  [PROBLEM] $Message" 'Red'; $script:Problems.Add($Message) }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Invoke-Logged([string]$Exe, [string[]]$Arguments) {
    # run a native program, copy its output to the console and setup.log, return its exit code
    Write-Log "  > $Exe $($Arguments -join ' ')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $s = "$_".TrimEnd()
            if ($s -and $s -notmatch '^[\s\-\\|/]*$' -and $s -notmatch $script:ProgressBarRe) { Write-Log "    $s" }
        }
        return $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $prev }
}
function Invoke-PyScript([string]$Code, [string[]]$Arguments) {
    # run a Python snippet from a temp file (avoids Windows PowerShell 5.1 quoting problems); returns output lines
    $tmp = Join-Path $env:TEMP ("avsr_setup_{0}.py" -f [guid]::NewGuid().ToString('N'))
    [System.IO.File]::WriteAllText($tmp, $Code, $Utf8NoBom)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $script:Py $tmp @Arguments 2>&1 | ForEach-Object { "$_" }
        $script:PyExit = $LASTEXITCODE
        return $out
    }
    finally {
        $ErrorActionPreference = $prev
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}
function Get-AsciiModelCopy([string]$Model) {
    # MediaPipe cannot open a model file whose path has non-ASCII (e.g. Korean) characters: copy it to an ASCII-only
    # folder (C:\ProgramData\avsr_kr, C:\Users\Public\avsr_kr or C:\avsr_kr); same helper as preprocess_shard.ps1
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
function Test-Winget {
    if (Get-Command winget -ErrorAction SilentlyContinue) { return $true }
    Problem 'winget not found. Install "App Installer" from the Microsoft Store, then run this again.'
    return $false
}
function Finish {
    Step 'Summary'
    if ($script:Problems.Count -eq 0) {
        Write-Log '  Everything is ready.' 'Green'
        $code = 0
    } else {
        Write-Log "  $($script:Problems.Count) problem(s):" 'Red'
        foreach ($p in $script:Problems) { Write-Log "   - $p" 'Red' }
        if ($CheckOnly) { Write-Log '  Run 1_setup.bat (without check-only) to install what is missing.' 'Yellow' }
        $code = 1
    }
    Write-Log "  Log file: $LogFile"
    exit $code
}

$CheckReqPy = @'
import sys
from importlib import metadata

bad = []
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    spec = line.split(";", 1)[0].strip()
    name, _, want = spec.partition("==")
    name, want = name.strip(), want.strip()
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        bad.append(f"MISSING {name}=={want}")
        continue
    if want and have.split("+")[0] != want:
        bad.append(f"VERSION {name}: {have} installed, requirements.txt wants {want}")
    elif name.lower() == "torch" and "+cu" not in have:
        bad.append(f"CPUTORCH torch {have} is a CPU-only build (need the CUDA 12.8 build)")
for b in bad:
    print(b)
'@

$VerifyPy = @'
import os
import shutil
import subprocess
import sys

root, asset, model = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
failed = 0


def check(name, fn):
    global failed
    try:
        print(f"  [OK] {name}: {fn()}", flush=True)
    except Exception as e:  # report every failure, keep checking the rest
        failed += 1
        print(f"  [PROBLEM] {name}: {type(e).__name__}: {e}", flush=True)


def t_torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) sees no CUDA GPU "
                           f"- check the NVIDIA driver (nvidia-smi)")
    x = torch.randn(256, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all()
    free, total = torch.cuda.mem_get_info()
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)} "
            f"({total / 2**30:.0f} GB), bf16 matmul OK")


def t_audio_vision():
    import torchaudio
    import torchvision
    return f"torchaudio {torchaudio.__version__}, torchvision {torchvision.__version__}"


def t_packages():
    import cv2, jiwer, numpy, scipy, soundfile, tqdm, yaml  # noqa: F401
    return f"numpy {numpy.__version__}, opencv {cv2.__version__}, scipy {scipy.__version__}, soundfile, jiwer, pyyaml, tqdm"


def t_mediapipe():
    import mediapipe
    import numpy as np
    from avsr.video_feats import FaceLandmarker
    if model.isascii():
        with FaceLandmarker(model, running_mode="image") as det:
            pts = det.detect(np.zeros((240, 320, 3), np.uint8))
        how = "" if model == asset else f", using the copy {model}"
        return f"mediapipe {mediapipe.__version__}, face landmarker model loads and runs (blank test image: " \
               f"{'no face, as expected' if pts is None else 'a face?'}){how}"
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    with open(model, "rb") as f:
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=f.read()))
    vision.FaceLandmarker.create_from_options(opts).close()
    return f"mediapipe {mediapipe.__version__}, face landmarker model loads (from memory: MediaPipe cannot open " \
           f"the non-ASCII project path itself)"


def t_ffmpeg():
    out = []
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        if path is None:
            raise RuntimeError(f"{exe} not on PATH")
        ver = subprocess.run([path, "-version"], capture_output=True, text=True, check=True).stdout.split("\n")[0]
        out.append(ver.split(" Copyright")[0])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    out.append("NVDEC h264_cuvid available" if "h264_cuvid" in dec else "no NVDEC decoder (preprocessing uses the CPU)")
    return "; ".join(out)


def t_audio_io():
    from avsr.audio_feats import compute_fbank
    import numpy as np
    fb = compute_fbank(np.zeros(16000, np.int16))
    return f"avsr package imports, fbank {tuple(fb.shape)}"


check("GPU + torch", t_torch)
check("torchaudio/torchvision", t_audio_vision)
check("other packages", t_packages)
check("MediaPipe", t_mediapipe)
check("FFmpeg", t_ffmpeg)
check("project code", t_audio_io)
sys.exit(1 if failed else 0)
'@

Write-Log ''
Write-Log ('=' * 70)
Write-Log "avsr_kr setup$(if ($CheckOnly) { ' (check only)' }) - project: $Root" 'Cyan'
Write-Log "Windows $([Environment]::OSVersion.Version), user $env:USERNAME, computer $env:COMPUTERNAME"

# ---------------------------------------------------------------------------------------------------------------------
Step '1. Python 3.12'
$script:Py = if ($env:AVSR_PYTHON) { $env:AVSR_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (Test-Path -LiteralPath $script:Py) {
    Ok "found $($script:Py)"
} elseif ($CheckOnly) {
    Problem "Python 3.12 not found at $($script:Py)"
    Finish
} elseif (Test-Winget) {
    Info 'installing Python 3.12 for this user with winget (a few minutes) ...'
    $rc = Invoke-Logged 'winget' @('install', '-e', '--id', 'Python.Python.3.12', '--scope', 'user', '--silent',
                                    '--accept-package-agreements', '--accept-source-agreements')
    if (-not (Test-Path -LiteralPath $script:Py)) {
        Problem "Python 3.12 is still not at $($script:Py) after winget (exit code $rc). Install it from python.org (64-bit, 'install for me only') and run this again."
        Finish
    }
    Ok "installed $($script:Py)"
} else {
    Finish
}
$verLines = Invoke-PyScript "import sys, struct; print(sys.version.split()[0], struct.calcsize('P') * 8)" @()
$ver = ($verLines -join ' ').Trim()
if ($ver -match '^3\.12\.\d+ 64$') { Ok "Python $ver-bit" } else { Problem "expected 64-bit Python 3.12, got: $ver"; Finish }

# ---------------------------------------------------------------------------------------------------------------------
Step '2. FFmpeg'
Refresh-Path
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
$fp = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($ff -and $fp) {
    Ok "found $($ff.Source)"
} elseif ($CheckOnly) {
    Problem 'ffmpeg/ffprobe not found on PATH'
} elseif (Test-Winget) {
    Info 'installing FFmpeg with winget ...'
    $rc = Invoke-Logged 'winget' @('install', '-e', '--id', 'Gyan.FFmpeg', '--accept-package-agreements', '--accept-source-agreements')
    Refresh-Path
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        # winget normally puts the bin folder on the user PATH; add it if that did not happen
        $bin = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe') -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($bin) {
            $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
            [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $bin.DirectoryName), 'User')
            Refresh-Path
            Info "added $($bin.DirectoryName) to the user PATH"
        }
    }
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Ok "installed $((Get-Command ffmpeg).Source)" }
    else { Problem "FFmpeg not on PATH after winget (exit code $rc). Open a new window and run this again." }
}

# ---------------------------------------------------------------------------------------------------------------------
Step '3. Python packages (requirements.txt)'
$req = Join-Path $Root 'requirements.txt'
if (-not (Test-Path -LiteralPath $req)) { Problem "requirements.txt not found: $req"; Finish }
$missing = @(Invoke-PyScript $CheckReqPy @($req) | Where-Object { $_ -match '^(MISSING|VERSION|CPUTORCH) ' })
if ($missing.Count -eq 0) {
    Ok 'all packages of requirements.txt are installed with the pinned versions (nothing to install)'
} else {
    foreach ($m in $missing) { Info "needs install: $m" }
    if ($CheckOnly) {
        Problem "$($missing.Count) package(s) missing or with another version (see above)"
    } else {
        $torchNeeded = @($missing | Where-Object { $_ -match '^(MISSING|VERSION) (torch|torchaudio|torchvision)[=:]' -or $_ -match '^CPUTORCH ' })
        if ($torchNeeded.Count -gt 0) {
            Info 'installing torch/torchaudio/torchvision (CUDA 12.8 build, about 3 GB download) ...'
            $torchArgs = @('-m', 'pip', 'install') + $TorchPins + @('--index-url', $TorchIndex)
            if (@($missing | Where-Object { $_ -match '^CPUTORCH ' }).Count -gt 0) { $torchArgs += @('--force-reinstall', '--no-deps') }
            $rc = Invoke-Logged $script:Py $torchArgs
            if ($rc -ne 0) { Problem "pip could not install torch (exit code $rc) - check the internet connection" }
        }
        Info 'installing the other packages ...'
        $rc = Invoke-Logged $script:Py @('-m', 'pip', 'install', '-r', $req, '--extra-index-url', $TorchIndex)
        if ($rc -ne 0) { Problem "pip install -r requirements.txt failed (exit code $rc)" }
        $missing = @(Invoke-PyScript $CheckReqPy @($req) | Where-Object { $_ -match '^(MISSING|VERSION|CPUTORCH) ' })
        if ($missing.Count -eq 0) { Ok 'all packages installed' } else { foreach ($m in $missing) { Problem "still not right: $m" } }
    }
}
# msvc-runtime: the Visual C++ runtime DLLs torch needs on this Windows image (listed in requirements.txt too)
$msvc = Invoke-PyScript "from importlib import metadata; print(metadata.version('msvc-runtime'))" @()
if ($script:PyExit -eq 0) {
    Ok "msvc-runtime $(($msvc -join ' ').Trim())"
} elseif ($CheckOnly) {
    Problem 'msvc-runtime is not installed (torch then fails with WinError 126 c10.dll)'
} else {
    $rc = Invoke-Logged $script:Py @('-m', 'pip', 'install', 'msvc-runtime')
    if ($rc -eq 0) { Ok 'msvc-runtime installed' } else { Problem "pip install msvc-runtime failed (exit code $rc)" }
}

# ---------------------------------------------------------------------------------------------------------------------
Step '4. MediaPipe face landmarker model'
$asset = Join-Path $Root 'assets\face_landmarker.task'
if ((Test-Path -LiteralPath $asset) -and ((Get-Item -LiteralPath $asset).Length -gt 1MB)) {
    Ok "found $asset ($([math]::Round((Get-Item -LiteralPath $asset).Length / 1MB, 1)) MB)"
} elseif ($CheckOnly) {
    Problem "missing $asset"
} else {
    Info "downloading $AssetUrl ..."
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $asset) | Out-Null
        $part = "$asset.part"
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $AssetUrl -OutFile $part -UseBasicParsing
        if ((Get-Item -LiteralPath $part).Length -lt 1MB) { throw 'downloaded file is too small' }
        Move-Item -LiteralPath $part -Destination $asset -Force
        Ok "downloaded $asset"
    } catch {
        Problem "could not download the MediaPipe model: $($_.Exception.Message). Copy assets\face_landmarker.task from the main PC."
    }
}

# ---------------------------------------------------------------------------------------------------------------------
Step '5. Verification'
if (-not $CheckOnly) {
    # files copied from another PC or the internet may be blocked for PowerShell; unblock the project scripts
    Get-ChildItem -LiteralPath (Join-Path $Root 'scripts') -Filter '*.ps1' -File | Unblock-File
    Get-ChildItem -LiteralPath $Root -Filter '*.bat' -File | Unblock-File
}
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& $smi.Source '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader' 2>$null) -join '; '
    Ok "NVIDIA driver: $gpu"
} else {
    Problem 'nvidia-smi not found: install the NVIDIA graphics driver'
}
$model = $asset
if ($Root -match '[^\x00-\x7F]') {
    Write-Log '  [NOTE] The project folder path has non-English letters. MediaPipe (face landmarks) cannot open files' 'Yellow'
    Write-Log '         there. 2_preprocess_part.bat handles this by itself (it uses a copy of the face model in' 'Yellow'
    Write-Log '         C:\ProgramData\avsr_kr). For scripts\preprocess.ps1 add --model <that copy>, for scripts\infer.ps1' 'Yellow'
    Write-Log '         --landmarker <that copy>, or move the avsr_kr folder to a path without Korean letters (C:\avsr_kr).' 'Yellow'
    if (-not $CheckOnly -and (Test-Path -LiteralPath $asset)) { $model = Get-AsciiModelCopy $asset }
}
Refresh-Path
$verify = Invoke-PyScript $VerifyPy @($Root, $asset, $model)
foreach ($line in $verify) {
    if ($line -match '^\s*\[OK\]') { Write-Log $line 'Green' }
    elseif ($line -match '^\s*\[PROBLEM\]') { Write-Log $line 'Red'; $script:Problems.Add($line.Trim()) }
    elseif ($line.Trim()) { Write-Log "    $line" }
}
if ($script:PyExit -ne 0 -and -not ($verify | Where-Object { $_ -match '\[PROBLEM\]' })) {
    Problem "verification script failed (exit code $($script:PyExit))"
}
if (($verify -join ' ') -match 'WinError 126|c10\.dll') {
    Info 'Hint: WinError 126 / c10.dll means the Visual C++ runtime is missing: pip install msvc-runtime'
}
$drive = (Get-Item -LiteralPath $Root).PSDrive
if ($drive -and $drive.Free) {
    $freeGb = [math]::Round($drive.Free / 1GB)
    if ($freeGb -lt 30) { Problem "only $freeGb GB free on drive $($drive.Name): (preprocessing needs about 10 GB per 240 videos incl. temporary files)" }
    else { Ok "free disk space on drive $($drive.Name): $freeGb GB" }
}
Finish
