<#
    One-shot setup for the hero-car matting pipeline.

      .\setup.ps1                  # auto-detect: CUDA wheels if an NVIDIA GPU is present
      .\setup.ps1 -Device cpu      # force CPU wheels
      .\setup.ps1 -Variant base+   # also fetch the Base+ checkpoint

    Creates .venv, installs torch 2.8.0 from the correct index, clones SAM2Matting into
    third_party/, and downloads checkpoints into checkpoints/.
#>
param(
    [ValidateSet("auto", "cuda", "cpu")] [string]$Device = "auto",
    [ValidateSet("tiny", "base+", "both")] [string]$Variant = "tiny",
    [string]$Python = "py -3.12"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# --- pick the torch wheel index -------------------------------------------------------
if ($Device -eq "auto") {
    $hasNvidia = $null -ne (Get-CimInstance Win32_VideoController |
        Where-Object { $_.Name -match "NVIDIA" })
    $Device = if ($hasNvidia) { "cuda" } else { "cpu" }
}
$index = if ($Device -eq "cuda") {
    "https://download.pytorch.org/whl/cu126"
} else {
    "https://download.pytorch.org/whl/cpu"
}
Write-Host "==> target device: $Device  (torch index: $index)" -ForegroundColor Cyan

# --- venv -----------------------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "==> creating .venv" -ForegroundColor Cyan
    Invoke-Expression "$Python -m venv .venv"
}
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "==> installing torch 2.8.0 / torchvision 0.23.0" -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url $index

Write-Host "==> installing pipeline requirements" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

# --- vendored repo --------------------------------------------------------------------
$repo = Join-Path $root "third_party\SAM2Matting"
if (-not (Test-Path $repo)) {
    Write-Host "==> cloning FudanCVL/SAM2Matting" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Join-Path $root "third_party") | Out-Null
    git clone --depth 1 https://github.com/FudanCVL/SAM2Matting.git $repo
} else {
    Write-Host "==> repo already present at third_party\SAM2Matting" -ForegroundColor DarkGray
}

# --- checkpoints ----------------------------------------------------------------------
$ckptDir = Join-Path $root "checkpoints"
New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

$wanted = switch ($Variant) {
    "tiny"  { @("SAM2Matting-SAM2.1Tiny.pt") }
    "base+" { @("SAM2Matting-SAM2.1Base+.pt") }
    "both"  { @("SAM2Matting-SAM2.1Tiny.pt", "SAM2Matting-SAM2.1Base+.pt") }
}

foreach ($name in $wanted) {
    $dest = Join-Path $ckptDir $name
    if (Test-Path $dest) {
        Write-Host "==> $name already downloaded" -ForegroundColor DarkGray
        continue
    }
    $encoded = [uri]::EscapeDataString($name)
    $url = "https://huggingface.co/FudanCVL/SAM2Matting/resolve/main/checkpoints/$encoded"
    Write-Host "==> downloading $name" -ForegroundColor Cyan
    curl.exe -L --fail --progress-bar -o $dest $url
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Verify with:  .\.venv\Scripts\python.exe -c ""import torch; print(torch.__version__, torch.cuda.is_available())"""
Write-Host "Then run:     .\.venv\Scripts\python.exe run_hero_matte.py `"$env:USERPROFILE\Downloads\Original`" -o output"
