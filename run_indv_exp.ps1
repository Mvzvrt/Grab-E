# Sequentially runs grabcut for each color space after hsv_conic, skips rgb

$Repo = Join-Path $HOME "Documents/Github/grab-cut"
Set-Location $Repo

# Activate venv for the session
$VenvActivate = Join-Path $Repo ".gc\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) { . $VenvActivate }

$all = @(
  "rgb",
  "hsv_conic",
  "cielab",
  "c02_scd",
  "c16_scd",
  "oklab",
  "oklch",
  "jzazbz",
  "jzczhz",
  "ictcp_pq",
  "xyz",
  "ycbcr_bt709",
  "srgb_linear",
  "ruderman_lab"
)

$startAfter = "hsv_conic"
$skip = @("rgb","hsv_conic")  # rgb done, hsv_conic currently running
$passed = $false

foreach ($cs in $all) {
  if ($cs -eq $startAfter) { $passed = $true; continue }
  if (-not $passed) { continue }
  if ($skip -contains $cs) { continue }

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