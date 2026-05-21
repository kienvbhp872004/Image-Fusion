# Script đóng gói code + data thành 1 file zip để upload Drive.
# Chạy từ Image-Fusion root:  .\remote_train\make_package.ps1

$ErrorActionPreference = "Stop"
$REPO_ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$STAGE     = "$env:TEMP\cddfuse_ag_package"
$ZIP_OUT   = "$REPO_ROOT\remote_train\cddfuse_ag_package.zip"

Write-Host "[1/4] Clean staging area" -ForegroundColor Cyan
if (Test-Path $STAGE) { Remove-Item -Recurse -Force $STAGE }
New-Item -ItemType Directory -Path $STAGE | Out-Null

Write-Host "[2/4] Copy code (chỉ phần cần cho train)" -ForegroundColor Cyan
$copy_items = @(
    "models/MMIF-CDDFuse/net.py",
    "models/MMIF-CDDFuse/train_MIF.py",
    "models/MMIF-CDDFuse/dataprocessing_MIF.py",
    "models/MMIF-CDDFuse/evaluate_cddfuse.py",
    "models/MMIF-CDDFuse/utils",
    "models/MMIF-CDDFuse/variants",
    "models/MMIF-CDDFuse/models/CDDFuse_MIF.pth",
    "metric",
    "remote_train/Dockerfile",
    "remote_train/run_train.sh",
    "remote_train/README.md"
)
foreach ($item in $copy_items) {
    $src = Join-Path $REPO_ROOT $item
    $dst = Join-Path $STAGE $item
    if (Test-Path $src) {
        $dstDir = Split-Path $dst -Parent
        if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
        if ((Get-Item $src).PSIsContainer) {
            Copy-Item -Path $src -Destination $dst -Recurse -Force
        } else {
            Copy-Item -Path $src -Destination $dst -Force
        }
        Write-Host "  OK: $item"
    } else {
        Write-Host "  MISSING: $item" -ForegroundColor Yellow
    }
}

Write-Host "[3/4] Copy dataset Harvard medical" -ForegroundColor Cyan
$ds_src = Join-Path $REPO_ROOT "Havard-Medical-Image-Fusion-Datasets-main"
$ds_dst = Join-Path $STAGE  "Havard-Medical-Image-Fusion-Datasets-main"
if (Test-Path $ds_src) {
    Copy-Item -Path $ds_src -Destination $ds_dst -Recurse -Force
    Write-Host "  OK: Havard-Medical-Image-Fusion-Datasets-main"
} else {
    Write-Host "  MISSING: Havard-Medical-Image-Fusion-Datasets-main" -ForegroundColor Yellow
}

# Cũng copy data/reference cho eval
$ref_src = Join-Path $REPO_ROOT "data/reference"
$ref_dst = Join-Path $STAGE "data/reference"
if (Test-Path $ref_src) {
    New-Item -ItemType Directory -Path (Split-Path $ref_dst -Parent) -Force | Out-Null
    Copy-Item -Path $ref_src -Destination $ref_dst -Recurse -Force
    Write-Host "  OK: data/reference (72 test pairs)"
}

Write-Host "[4/4] Tạo zip" -ForegroundColor Cyan
if (Test-Path $ZIP_OUT) { Remove-Item -Force $ZIP_OUT }
Compress-Archive -Path "$STAGE\*" -DestinationPath $ZIP_OUT -CompressionLevel Optimal

$sizeMB = [math]::Round((Get-Item $ZIP_OUT).Length / 1MB, 2)
Write-Host ""
Write-Host "=== HOÀN TẤT ===" -ForegroundColor Green
Write-Host "Package:  $ZIP_OUT"
Write-Host "Size:     $sizeMB MB"
Write-Host "Staging:  $STAGE (có thể xóa)"
Write-Host ""
Write-Host "Bước tiếp theo:" -ForegroundColor Yellow
Write-Host "  1. Upload $ZIP_OUT lên Google Drive"
Write-Host "  2. Gửi link Drive + remote_train/README.md cho bạn"
