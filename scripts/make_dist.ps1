# Zip the project code for the other PCs: dist\avsr_kr_code.zip (entries under avsr_kr/). Contains avsr/ tools/
# scripts/ configs/ assets/ tests/ docs/ README.md requirements.txt *.bat - never work*/ (data, checkpoints), dist/,
# __pycache__, *.pyc, *.pt or logs. Unzip it on the other PC (e.g. to the Desktop) and run 1_setup.bat there.
#   .\scripts\make_dist.ps1                    # -> dist\avsr_kr_code.zip
#   .\scripts\make_dist.ps1 -Out D:\avsr_kr_code.zip
param([string]$Out = '')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root 'dist\avsr_kr_code.zip' }
$Out = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($root, $Out))
$includeDirs = @('avsr', 'tools', 'scripts', 'configs', 'assets', 'tests', 'docs')
$includeFiles = @('README.md', 'requirements.txt')
$excludeDirNames = @('__pycache__', '.pytest_cache', '.ipynb_checkpoints')
$excludePatterns = @('*.pyc', '*.pyo', '*.pt', '*.part', '*.log', '*.tmp')

function Test-Excluded([System.IO.FileInfo]$f) {
    foreach ($p in $excludePatterns) { if ($f.Name -like $p) { return $true } }
    $rel = $f.FullName.Substring($root.Length + 1)
    foreach ($seg in $rel.Split('\')) { if ($excludeDirNames -contains $seg) { return $true } }
    return $false
}

$files = New-Object System.Collections.Generic.List[System.IO.FileInfo]
foreach ($d in $includeDirs) {
    $dir = Join-Path $root $d
    if (-not (Test-Path -LiteralPath $dir)) { throw "missing folder: $dir" }
    Get-ChildItem -LiteralPath $dir -Recurse -File | Where-Object { -not (Test-Excluded $_) } | ForEach-Object { $files.Add($_) }
}
foreach ($n in $includeFiles) {
    $f = Get-Item -LiteralPath (Join-Path $root $n)
    $files.Add($f)
}
Get-ChildItem -LiteralPath $root -Filter '*.bat' -File | ForEach-Object { $files.Add($_) }
$sorted = $files | Sort-Object { $_.FullName.Substring($root.Length + 1).ToLowerInvariant() }

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
$part = "$Out.part"
if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
$stream = [System.IO.File]::Open($part, [System.IO.FileMode]::CreateNew)
$zip = New-Object System.IO.Compression.ZipArchive($stream, [System.IO.Compression.ZipArchiveMode]::Create)
$rawBytes = 0
try {
    foreach ($f in $sorted) {
        $entry = 'avsr_kr/' + $f.FullName.Substring($root.Length + 1).Replace('\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entry,
            [System.IO.Compression.CompressionLevel]::Optimal)
        $rawBytes += $f.Length
    }
}
finally {
    $zip.Dispose()
    $stream.Dispose()
}
Move-Item -LiteralPath $part -Destination $Out -Force

# verify: re-open the archive and compare the entry list with the files
$check = [System.IO.Compression.ZipFile]::OpenRead($Out)
try {
    $entries = @($check.Entries | Where-Object { $_.Name })
    $bad = @($entries | Where-Object { $_.FullName -match '(^|/)(work[^/]*|dist|__pycache__)/' -or $_.Name -like '*.pt' })
    if ($entries.Count -ne $sorted.Count) { throw "zip has $($entries.Count) entries, expected $($sorted.Count)" }
    if ($bad.Count) { throw "excluded files ended up in the zip: $($bad[0].FullName)" }
    $byTop = $entries | Group-Object { $_.FullName.Split('/')[1] } | Sort-Object Name
}
finally { $check.Dispose() }
$zipSize = (Get-Item -LiteralPath $Out).Length
Write-Host "created $Out"
Write-Host ("  {0} files, {1:N1} MB packed ({2:N1} MB unpacked)" -f $sorted.Count, ($zipSize / 1MB), ($rawBytes / 1MB))
foreach ($g in $byTop) { Write-Host ("  {0,-22} {1,4} files" -f $g.Name, $g.Count) }
Write-Host 'Copy the zip to the other PC, unzip it (e.g. to the Desktop) and double-click 1_setup.bat in the avsr_kr folder.'
