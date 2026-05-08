param(
  [string]$HomeHtmlPath,
  [string]$DetailHtmlPath
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataJsonPath = Join-Path $RepoRoot "data/draws.json"
$DataJsPath = Join-Path $RepoRoot "data/draws.js"

function Get-HttpText([string]$Url) {
  if ($HomeHtmlPath -and $Url.EndsWith("/ygkj/kjgg/") -and (Test-Path $HomeHtmlPath)) {
    return [string](Get-Content -Path $HomeHtmlPath -Raw -Encoding utf8)
  }
  if ($DetailHtmlPath -and ($Url -match "/c/\d{4}/\d{2}/\d{2}/\d+\.shtml$") -and (Test-Path $DetailHtmlPath)) {
    return [string](Get-Content -Path $DetailHtmlPath -Raw -Encoding utf8)
  }
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
  $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 25 -Headers @{ "User-Agent" = "Mozilla/5.0 (compatible; ssq-data-bot/1.0)" } -ErrorAction Stop
  return [string]$resp.Content
}

function Get-LatestFromCwl() {
  $base = "https://www.cwl.gov.cn"
  $html = Get-HttpText "$base/ygkj/kjgg/"

  $start = $html.IndexOf("block-ssq")
  if ($start -lt 0) { throw "parse_ssq_block_failed" }
  $len = [Math]::Min(30000, $html.Length - $start)
  $slice = $html.Substring($start, $len)
  $stopIdx = $slice.IndexOf("block-kl8")
  if ($stopIdx -gt 0) { $slice = $slice.Substring(0, $stopIdx) }

  $mIssue = [regex]::Match($slice, "(?<issue>\d{7})", [System.Text.RegularExpressions.RegexOptions]::Singleline)
  if (-not $mIssue.Success) { throw "parse_issue_failed" }
  $issue = $mIssue.Groups["issue"].Value

  $redMatches = [regex]::Matches(
    $slice,
    "redCircle\.png[\s\S]*?<[^>]*>(?<n>\d{1,2})\s*<",
    [System.Text.RegularExpressions.RegexOptions]::Singleline
  )
  $blueMatches = [regex]::Matches(
    $slice,
    "blueCircle\.png[\s\S]*?<[^>]*>(?<n>\d{1,2})\s*<",
    [System.Text.RegularExpressions.RegexOptions]::Singleline
  )

  if ($redMatches.Count -lt 6) { throw "parse_reds_failed:$($redMatches.Count)" }
  if ($blueMatches.Count -lt 1) { throw "parse_blue_failed:$($blueMatches.Count)" }

  $reds = @()
  for ($i = 0; $i -lt 6; $i++) { $reds += [int]$redMatches[$i].Groups["n"].Value }
  $blue = [int]$blueMatches[0].Groups["n"].Value

  $mDetail = [regex]::Match(
    $slice,
    'href="(?<u>/c/\d{4}/\d{2}/\d{2}/\d+\.shtml)"',
    [System.Text.RegularExpressions.RegexOptions]::Singleline
  )
  if (-not $mDetail.Success) { throw "parse_detail_failed" }
  $detailUrl = $base + $mDetail.Groups["u"].Value

  $detailHtml = Get-HttpText $detailUrl
  $mDate = [regex]::Match($detailHtml, "(?<d>\d{4}-\d{2}-\d{2})", [System.Text.RegularExpressions.RegexOptions]::Singleline)
  $date = $null
  if ($mDate.Success) { $date = $mDate.Groups["d"].Value }

  return @{
    issue = $issue
    year = [int]$issue.Substring(0, 4)
    date = $date
    reds = ($reds | Sort-Object)
    blue = $blue
    detailUrl = $detailUrl
  }
}

function Assert-Draw([hashtable]$d) {
  if ($d.issue -notmatch "^\d{7}$") { throw "bad_issue:$($d.issue)" }
  if ($d.year -lt 2000 -or $d.year -gt 2100) { throw "bad_year:$($d.year)" }
  if ($d.reds.Count -ne 6) { throw "bad_red_count:$($d.reds.Count)" }
  if (($d.reds | Select-Object -Unique).Count -ne 6) { throw "dup_reds" }
  foreach ($r in $d.reds) { if ($r -lt 1 -or $r -gt 33) { throw "bad_red:$r" } }
  if ($d.blue -lt 1 -or $d.blue -gt 16) { throw "bad_blue:$($d.blue)" }
}

function Write-JsonAtomically([string]$Path, [string]$Content) {
  $tmp = "$Path.tmp"
  Set-Content -Path $tmp -Value $Content -Encoding utf8
  Move-Item -Force -Path $tmp -Destination $Path
}

if (-not (Test-Path $DataJsonPath)) { throw "缺少数据文件：$DataJsonPath" }

$raw = Get-Content -Path $DataJsonPath -Raw -Encoding utf8
$doc = $raw | ConvertFrom-Json
if (-not $doc.draws) { throw "draws.json 缺少 draws 字段" }

$lastIssue = [string]$doc.draws[-1].issue
$latest = Get-LatestFromCwl
Assert-Draw $latest

Write-Output "current=$lastIssue fetched=$($latest.issue) url=$($latest.detailUrl)"

if ([string]$latest.issue -le $lastIssue) {
  Write-Output "no_update"
  exit 0
}

$newDraw = [pscustomobject]@{
  issue = [string]$latest.issue
  year = [int]$latest.year
  date = $latest.date
  reds = @($latest.reds | ForEach-Object { [int]$_ })
  blue = [int]$latest.blue
}

$doc.draws += $newDraw

if (-not $doc.meta) { $doc | Add-Member -NotePropertyName meta -NotePropertyValue ([pscustomobject]@{}) }
$doc.meta.count = $doc.draws.Count
$doc.meta.generatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

$compressed = $doc | ConvertTo-Json -Compress -Depth 20
Write-JsonAtomically -Path $DataJsonPath -Content $compressed
Write-JsonAtomically -Path $DataJsPath -Content ("window.__SSQ_DATA__=" + $compressed)

Write-Output ("updated=" + $newDraw.issue + " total=" + $doc.draws.Count)
