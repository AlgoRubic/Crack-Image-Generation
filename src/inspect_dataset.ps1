param(
    [int]$SamplesPerCategory = 6
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$imageRoot = Join-Path $projectRoot 'data\raw\SDNET2018\images'
$outputRoot = Join-Path $projectRoot 'outputs\plots'
$contactSheetPath = Join-Path $outputRoot 'sdnet2018_contact_sheet.jpg'

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Add-Type -AssemblyName System.Drawing

$categories = @(
    @{ Name = 'bridge_deck_cracked'; Path = 'D\CD' },
    @{ Name = 'bridge_deck_non_cracked'; Path = 'D\UD' },
    @{ Name = 'pavement_cracked'; Path = 'P\CP' },
    @{ Name = 'pavement_non_cracked'; Path = 'P\UP' },
    @{ Name = 'wall_cracked'; Path = 'W\CW' },
    @{ Name = 'wall_non_cracked'; Path = 'W\UW' }
)

$records = @()
$sampleImages = @()

foreach ($category in $categories) {
    $categoryPath = Join-Path $imageRoot $category.Path
    $files = @(Get-ChildItem -LiteralPath $categoryPath -File -Filter '*.jpg' | Sort-Object Name)
if ($files.Count -eq 0) {
        Write-Warning "No JPG files found in $categoryPath"
        continue
    }

    $step = [Math]::Max(1, [Math]::Floor($files.Count / $SamplesPerCategory))
    $selected = @($files | Select-Object -Skip 0 -First $SamplesPerCategory)

    foreach ($file in $selected) {
        $bitmap = [System.Drawing.Image]::FromFile($file.FullName)
        $records += [PSCustomObject]@{
            Category = $category.Name
            File = $file.Name
            Width = $bitmap.Width
            Height = $bitmap.Height
            PixelFormat = $bitmap.PixelFormat.ToString()
        }
        $sampleImages += [PSCustomObject]@{
            Category = $category.Name
            Path = $file.FullName
        }
        $bitmap.Dispose()
    }
}

if ($sampleImages.Count -eq 0) {
    throw "No sample images were found below $imageRoot"
}

Write-Output '--- sampled image properties ---'
$records | Format-Table -AutoSize

Write-Output '--- dimension summary ---'
$records | Group-Object Width,Height | Select-Object Name,Count | Format-Table -AutoSize

$cellWidth = 220
$cellHeight = 180
$columns = 3
$rows = [Math]::Ceiling($sampleImages.Count / $columns)
$sheet = New-Object System.Drawing.Bitmap ($columns * $cellWidth), ($rows * $cellHeight)
$graphics = [System.Drawing.Graphics]::FromImage($sheet)
$graphics.Clear([System.Drawing.Color]::White)
$font = New-Object System.Drawing.Font 'Arial', 9

for ($i = 0; $i -lt $sampleImages.Count; $i++) {
    $item = $sampleImages[$i]
    $x = ($i % $columns) * $cellWidth
    $y = [Math]::Floor($i / $columns) * $cellHeight
    $image = [System.Drawing.Image]::FromFile($item.Path)

    $availableWidth = $cellWidth - 10
    $availableHeight = $cellHeight - 35
    $scale = [Math]::Min($availableWidth / $image.Width, $availableHeight / $image.Height)
    $drawWidth = [int]($image.Width * $scale)
    $drawHeight = [int]($image.Height * $scale)
    $drawX = $x + [int](($cellWidth - $drawWidth) / 2)
    $drawY = $y + 5

    $graphics.DrawImage($image, $drawX, $drawY, $drawWidth, $drawHeight)
    $graphics.DrawString($item.Category, $font, [System.Drawing.Brushes]::Black, $x + 5, $y + $cellHeight - 25)
    $image.Dispose()
}

$sheet.Save($contactSheetPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$graphics.Dispose()
$sheet.Dispose()
$font.Dispose()

Write-Output "Contact sheet written to: $contactSheetPath"
