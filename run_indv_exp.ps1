# Sequentially runs grabcut for each color space after hsv_conic, skips rgb

$Repo = Join-Path $HOME "Documents/Github/grab-cut"
Set-Location $Repo

# Activate venv for the session
$VenvActivate = Join-Path $Repo ".gc\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) { . $VenvActivate }

$all = @(
  "opponent",
  "log_chroma",
  "c1c2c3",
)

foreach ($cs in $all) {
  $outDir = ".\$cs"
  if (Test-Path $outDir) {
    Write-Host "Skipping $cs, $outDir already exists"
    continue
  }

  Write-Host "Running $cs"
  python .\grabcut.py `
    --images_dir ../segmentor/server/public/images `
    --anns_dir ..\segmentor\server\public\user_annotations\ `
    --output_dir $outDir `
    --parallel `
    --color_space $cs

  if ($LASTEXITCODE -ne 0) {
    Write-Host "Stopped at $cs, exit $LASTEXITCODE"
    break
  }
  Write-Host "Finished $cs"
}

Write-Host "Done"