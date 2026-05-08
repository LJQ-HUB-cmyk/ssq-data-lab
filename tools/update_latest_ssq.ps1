param(
  [string]$ApiJsonPath,
  [string]$HomeHtmlPath,
  [string]$DetailHtmlPath
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataJsonPath = Join-Path $RepoRoot "data/draws.json"
$DataJsPath = Join-Path $RepoRoot "data/draws.js"

function Get-HttpText([string]$Url, [hashtable]$ExtraHeaders = $null) {
  if ($ApiJsonPath -and $Url -match "/cwl_admin/.*/findDrawNotice" -and (Test-Path $ApiJsonPath)) {
    return [string](Get-Content -Path $ApiJsonPath -Raw -Encoding utf8)
  }
  if ($HomeHtmlPath -and $Url.EndsWith("/ygkj/kjgg/") -and (Test-Path $HomeHtmlPath)) {
    return [string](Get-Content -Path $HomeHtmlPath -Raw -Encoding utf8)
  }
  if ($DetailHtmlPath -and ($Url -match "/c/\d{4}/\d{2}/\d{2}/\d+\.shtml$") -and (Test-Path $DetailHtmlPath)) {
    return [string](Get-Content -Path $DetailHtmlPath -Raw -Encoding utf8)
  }
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
  $headers = @{
    "User-Agent" = "Mozilla/5.0 (compatible; ssq-data-bot/1.0)"
    "Accept" = "text/html,application/json;q=0.9,*/*;q=0.8"
    "Accept-Language" = "zh-CN,zh;q=0.9,en;q=0.6"
    "Referer" = "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/"
  }
  if ($ExtraHeaders) {
    foreach ($k in $ExtraHeaders.Keys) { $headers[$k] = $ExtraHeaders[$k] }
  }
  if (-not $script:WebSession) { $script:WebSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession }
  $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 25 -Headers $headers -WebSession $script:WebSession -ErrorAction Stop
  return [string]$resp.Content
}

function Get-LatestFromCwlApi() {
  $url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount=1&systemType=PC"

  $jsonText = Get-HttpText $url
  $res = $jsonText | ConvertFrom-Json
  if (-not $res.result -or $res.result.Count -lt 1) { throw "api_empty_result" }
  $it = $res.result[0]

  $issue = [string]$it.code
  $reds = @()
  foreach ($x in ([string]$it.red).Split(",")) {
    if ($x -ne "") { $reds += [int]$x }
  }
  $blue = [int]$it.blue
  $date = $null
  if ($it.date) { $date = [string]$it.date }

  return @{
    issue = $issue
    year = [int]$issue.Substring(0, 4)
    date = $date
    reds = ($reds | Sort-Object)
    blue = $blue
    source = "api"
  }
}

function Get-LatestFromZhcwApi() {
  $ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $url = "https://jc.zhcw.com/port/client_json.php?callback=jQuery1122_1&transactionType=10001001&lotteryId=1&issueCount=1&startIssue=&endIssue=&startDate=&endDate=&type=0&pageNum=1&pageSize=1&tt=0.1&_=$ts"

  $text = Get-HttpText $url @{
    "Accept" = "*/*"
    "Referer" = "https://www.zhcw.com/"
  }
  $m = [regex]::Match($text, "^[^(]*\\((?<j>[\\s\\S]*)\\)\\s*;?\\s*$", [System.Text.RegularExpressions.RegexOptions]::Singleline)
  if (-not $m.Success) { throw "zhcw_jsonp_parse_failed" }
  $res = $m.Groups["j"].Value | ConvertFrom-Json
  if (-not $res.data -or $res.data.Count -lt 1) { throw "zhcw_empty_result" }
  $it = $res.data[0]

  $issue = [string]$it.issue
  $date = $null
  if ($it.openTime) { $date = [string]$it.openTime }

  $reds = @()
  foreach ($x in ([string]$it.frontWinningNum).Split(" ")) {
    if ($x -ne "") { $reds += [int]$x }
  }
  $blue = [int]$it.backWinningNum

  return @{
    issue = $issue
    year = [int]$issue.Substring(0, 4)
    date = $date
    reds = ($reds | Sort-Object)
    blue = $blue
    source = "zhcw"
  }
}

function Get-LatestFromCwlHtml() {
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
    source = "html"
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
$latest = $null
try {
  $latest = Get-LatestFromCwlApi
} catch {
  Write-Output "cwl_failed"
  try {
    $latest = Get-LatestFromZhcwApi
  } catch {
    Write-Output "zhcw_failed"
    $latest = Get-LatestFromCwlHtml
  }
}
Assert-Draw $latest

$u = ""
if ($latest.detailUrl) { $u = $latest.detailUrl }
Write-Output "current=$lastIssue fetched=$($latest.issue) source=$($latest.source) url=$u"

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
