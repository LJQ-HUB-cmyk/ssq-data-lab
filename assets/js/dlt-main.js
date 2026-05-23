// 大乐透主控制器：与 main.js 同构，但全部基于 DLT 配置（5 选 35 + 2 选 12）。

import { $, $$, clamp, pad2, parseNumList } from "./utils.js";
import { loadDltDraws } from "./dlt-data.js";
import { DLT_CONFIG } from "./lottery-config.js";
import {
  zoneFreq, zoneCurrentMiss, topNFromFreq, zoneMissStats, buildZoneTrend,
} from "./lottery-stats.js";
import { renderDltBars } from "./dlt-chart.js";
import { renderDltTrend } from "./dlt-trend-chart.js";
import { generateDltTickets } from "./dlt-generator.js";
import { generateDltAdvanced } from "./dlt-advanced-sampler.js";
import {
  frontOddEvenRatio, frontBigSmallRatio, frontPrimeCompositeRatio,
  frontPath012Ratio, frontZoneRatio, frontAcValue,
  frontConsecutiveGroups, frontMaxSameTail,
  frontSum, frontSpan, backSum, backSpan, backOddEvenRatio,
  groupByRatio,
  FRONT_SIZE, FRONT_PICK, BACK_SIZE, BACK_PICK,
  FRONT_SUM_MIN, FRONT_SUM_MAX, FRONT_SPAN_MIN, FRONT_SPAN_MAX,
  BACK_SUM_MIN, BACK_SUM_MAX,
} from "./dlt-distribution.js";
import { frontChi, backChi, chiSquaredPValue } from "./dlt-chi-square.js";
import {
  dltDanTuoTickets, dltComplexTickets, combinations, dltPriceOf,
} from "./dlt-combinatorics.js";
import {
  buildDltCooccurrenceMatrix, topDltPartners, dltLiftOf, extremeDltPairs,
  FRONT_INDEPENDENT_LIFT_BASELINE,
} from "./dlt-cooccurrence.js";
import { renderDltTimeSeries } from "./dlt-timeseries.js";
import {
  nextDltDrawTime, dltSaleCutoffOf, diffDuration,
  formatChinaTime, nextDltIssueOf,
} from "./dlt-countdown.js";
import {
  setupTabs, setupTheme,
  renderDltLatest, renderDltHeroMeta, renderDltRank, renderDltTable,
  renderDltInsightChips, renderDltTickets, formatDltTicketLine,
  copyToClipboard, renderDltTicketAnalysis, showLoadError,
  showDltDataSourceBanner, setRefreshLoading,
  readDltWinSize, readDltGeneratorConfig, showDltGenError,
  setDltGenDiagnostics, toast, renderDltSamplerDiagnostics,
} from "./dlt-ui.js";

const FRONT_ZONE = DLT_CONFIG.zones[0];
const BACK_ZONE = DLT_CONFIG.zones[1];

const state = {
  meta: {},
  draws: [],
  winSize: 200,
  freqAllFront: null,
  freqAllBack: null,
  freqRecentFront: null,
  freqRecentBack: null,
  missFront: null,
  missBack: null,
  coMatrix: null,
  tableRows: [],
  lastTickets: [],
};

function computeStats() {
  state.freqAllFront = zoneFreq(state.draws, FRONT_ZONE);
  state.freqAllBack = zoneFreq(state.draws, BACK_ZONE);
  const recent = state.draws.slice(-state.winSize);
  state.freqRecentFront = zoneFreq(recent, FRONT_ZONE);
  state.freqRecentBack = zoneFreq(recent, BACK_ZONE);
  state.missFront = zoneCurrentMiss(state.draws, FRONT_ZONE);
  state.missBack = zoneCurrentMiss(state.draws, BACK_ZONE);
  state.coMatrix = null; // lazy
}

function renderOverviewAndInsight() {
  const latest = state.draws[state.draws.length - 1];
  renderDltHeroMeta(state.meta, state.draws);
  renderDltLatest(latest);

  renderDltBars($("#chartFrontAll"), state.freqAllFront, FRONT_SIZE, "front");
  renderDltBars($("#chartBackAll"), state.freqAllBack, BACK_SIZE, "back");
  renderDltRank($("#rankFrontRecent"), topNFromFreq(state.freqRecentFront, 8, FRONT_SIZE), "front");
  renderDltRank($("#rankBackRecent"), topNFromFreq(state.freqRecentBack, 6, BACK_SIZE), "back");

  renderDltBars($("#chartFrontMiss"), state.missFront, FRONT_SIZE, "front-miss", { unit: "期" });
  renderDltBars($("#chartBackMiss"), state.missBack, BACK_SIZE, "back-miss", { unit: "期" });
  renderDltInsightChips({
    freqRecentFront: state.freqRecentFront,
    missFront: state.missFront,
    missBack: state.missBack,
  });

  renderDataTable();
}

function renderTrendPanel() {
  const win = Number($("#trendWindow")?.value || 50);
  const showStats = $("#trendShowStats")?.checked !== false;
  const slice = state.draws.slice(-win);
  const rows = slice.map((d) => ({
    issue: d.issue,
    date: d.date,
    hit: new Set(d.front),
  }));
  const rowsBack = slice.map((d) => ({
    issue: d.issue,
    date: d.date,
    hit: new Set(d.back),
  }));
  const frontStats = showStats ? toIndexed(zoneMissStats(slice, FRONT_ZONE)) : null;
  const backStats = showStats ? toIndexed(zoneMissStats(slice, BACK_ZONE)) : null;
  renderDltTrend($("#trendFront"), rows, {
    size: FRONT_SIZE, kind: "front", stats: frontStats,
    zoneBoundaries: [12, 24],
  });
  renderDltTrend($("#trendBack"), rowsBack, {
    size: BACK_SIZE, kind: "back", stats: backStats,
  });
}

// zoneMissStats 返回从 index 1 起的对象数组；走势图渲染器期望 stats[1..size]
function toIndexed(arr) { return arr; }

function renderRatioList(el, entries, top = 8) {
  if (!el) return;
  el.innerHTML = "";
  const totalCount = entries.reduce((a, b) => a + b[1], 0);
  const max = entries[0]?.[1] || 1;
  const items = entries.slice(0, top);
  for (const [k, v] of items) {
    const pct = ((v / totalCount) * 100).toFixed(1);
    const w = Math.round((v / max) * 100);
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="ratio-row">
        <span class="ratio-key">${k}</span>
        <span class="ratio-bar"><i style="width:0%"></i></span>
        <span class="ratio-val">${v} · ${pct}%</span>
      </div>
    `;
    el.appendChild(li);
    requestAnimationFrame(() => {
      const fill = li.querySelector(".ratio-bar i");
      if (fill) fill.style.width = `${w}%`;
    });
  }
}

function renderDistributionPanel() {
  const draws = state.draws;
  renderRatioList($("#distOddEven"), groupByRatio(draws, (d) => frontOddEvenRatio(d.front)));
  renderRatioList($("#distBigSmall"), groupByRatio(draws, (d) => frontBigSmallRatio(d.front)));
  renderRatioList($("#distPrime"), groupByRatio(draws, (d) => frontPrimeCompositeRatio(d.front)));
  renderRatioList($("#dist012"), groupByRatio(draws, (d) => frontPath012Ratio(d.front)));
  renderRatioList($("#distZone"), groupByRatio(draws, (d) => frontZoneRatio(d.front)));
  const acEntries = groupByRatio(draws, (d) => String(frontAcValue(d.front)))
    .sort((a, b) => Number(a[0]) - Number(b[0]));
  renderRatioList($("#distAC"), acEntries, 8);
  renderRatioList($("#distBackOddEven"), groupByRatio(draws, (d) => backOddEvenRatio(d.back)));

  drawHistogram($("#chartSum"), draws.map((d) => frontSum(d.front)), FRONT_SUM_MIN, FRONT_SUM_MAX);
  drawHistogram($("#chartSpan"), draws.map((d) => frontSpan(d.front)), FRONT_SPAN_MIN, FRONT_SPAN_MAX);
  drawHistogram($("#chartBackSum"), draws.map((d) => backSum(d.back)), BACK_SUM_MIN, BACK_SUM_MAX);

  renderTimeSeriesPanel();
  renderCooccurrencePanel();
}

function renderTimeSeriesPanel() {
  const container = $("#tsChart");
  if (!container) return;
  const kind = $("#tsKind")?.value || "sum";
  const winSel = $("#tsWindow")?.value || "300";
  const ma = clamp(Number($("#tsMA")?.value || 30), 1, 200);
  const len = winSel === "all" ? state.draws.length : Math.min(state.draws.length, Number(winSel));
  const slice = state.draws.slice(-len);
  renderDltTimeSeries(container, slice, kind, { window: ma });
}

function renderCooccurrencePanel() {
  const container = $("#partnerList");
  const extreme = $("#extremePairs");
  if (!container || !extreme) return;
  if (!state.coMatrix) {
    state.coMatrix = buildDltCooccurrenceMatrix(state.draws);
  }
  const m = state.coMatrix;
  const num = clamp(Number($("#partnerNum")?.value || 6), 1, FRONT_SIZE);
  const k = clamp(Number($("#partnerK")?.value || 8), 3, FRONT_SIZE - 1);
  const partners = topDltPartners(m, num, k);
  container.innerHTML = "";
  for (const [n, count] of partners) {
    const lift = dltLiftOf(m, state.freqAllFront, state.draws.length, num, n);
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="ball front" style="width:26px;height:26px;font-size:11px;box-shadow:none">${pad2(n)}</span>
      <span class="muted" style="margin-left:auto; font-family:var(--mono); font-size:12px">
        ×&nbsp;<strong style="color:var(--text)">${count}</strong>
        &nbsp;·&nbsp; lift <strong style="color:${liftColor(lift)}">${lift.toFixed(2)}</strong>
      </span>
    `;
    container.appendChild(li);
  }

  const ex = extremeDltPairs(m, state.freqAllFront, state.draws.length, 8);
  extreme.innerHTML = "";
  for (const p of ex) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="ball front" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(p.a)}</span>
      <span class="ball front" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(p.b)}</span>
      <span class="muted" style="margin-left:auto; font-family:var(--mono); font-size:12px">
        ×&nbsp;${p.count} · lift <strong style="color:${liftColor(p.lift)}">${p.lift.toFixed(2)}</strong>
      </span>
    `;
    extreme.appendChild(li);
  }
}

function liftColor(lift) {
  const dev = Math.abs(lift - FRONT_INDEPENDENT_LIFT_BASELINE);
  if (dev < 0.05) return "var(--dlt-front)";
  if (dev < 0.15) return "var(--gold)";
  return "var(--red-2)";
}

function drawHistogram(container, values, min, max) {
  if (!container) return;
  const buckets = Array(max - min + 1).fill(0);
  for (const v of values) {
    if (v >= min && v <= max) buckets[v - min]++;
  }
  const arr = [0, ...buckets];
  renderDltBars(container, arr, buckets.length, "front", { unit: "期" });
}

function renderScience() {
  const fc = frontChi(state.draws);
  const bc = backChi(state.draws);
  const pF = chiSquaredPValue(fc.chi, fc.df);
  const pB = chiSquaredPValue(bc.chi, bc.df);

  $("#chiFront").textContent = fc.chi.toFixed(2);
  $("#pFront").textContent = formatP(pF);
  $("#rejectFront").innerHTML = verdictChip(pF);
  $("#chiBack").textContent = bc.chi.toFixed(2);
  $("#pBack").textContent = formatP(pB);
  $("#rejectBack").innerHTML = verdictChip(pB);

  const fRej = pF < 0.05, bRej = pB < 0.05;
  let verdict;
  if (!fRej && !bRej) {
    verdict = `前区 p=${formatP(pF)}、后区 p=${formatP(pB)}，两者都 <strong>显著大于 0.05</strong>。
      统计上没有证据拒绝"均匀分布"假设——换句话说，
      <strong>历史数据与"完全随机摇出"完全一致</strong>。
      "冷号热号"只是随机波动的自然产物，不构成下期的预测依据。`;
  } else if (fRej && !bRej) {
    verdict = `前区 p=${formatP(pF)} < 0.05，观察到显著偏差；后区 p=${formatP(pB)} 未显著。
      前区偏差可能来自摇奖设备物理不对称、样本量、或碰巧的极端情况。
      <strong>显著性 ≠ 可预测性</strong>，它只告诉你"均匀假设在此样本下成立/不成立"。`;
  } else {
    verdict = `前区 p=${formatP(pF)}、后区 p=${formatP(pB)}，存在显著偏差。
      这不代表"冷号会补"，更可能的解释是：摇奖装置的物理随机性不完美。
      即便如此，<strong>下一期仍然是独立同分布的随机抽取</strong>。`;
  }
  $("#scienceVerdict").innerHTML = verdict + "<br/><br/>"
    + `<em class="muted">样本：${state.draws.length} 期（${state.draws[0].issue} – ${state.draws[state.draws.length - 1].issue}）</em>`;
}

function formatP(p) {
  if (p < 0.0001) return "< 0.0001";
  return p.toFixed(4);
}

function verdictChip(p) {
  if (p < 0.05) return `<span class="chip chip-warn">拒绝（p&lt;0.05）</span>`;
  return `<span class="chip chip-ok">不拒绝（数据与均匀一致）</span>`;
}

function readDataLimit() {
  const value = $("#dataLimit")?.value || "50";
  return value === "all" ? Infinity : Number(value);
}

function limitDataRows(rows) {
  const limit = readDataLimit();
  const ordered = rows.slice().reverse();
  return Number.isFinite(limit) ? ordered.slice(0, limit) : ordered;
}

function filterDataRows(query) {
  const q = query.trim();
  if (!q) return state.draws;
  const qLower = q.toLowerCase();
  // 后区匹配："后 02"、"back 02"、"b 02"
  const backMatch = q.match(/(?:后|back|b)\s*0?(\d{1,2})/i);
  const back = backMatch ? Number(backMatch[1]) : null;
  const frontNums = backMatch ? [] : parseNumList(q, 1, FRONT_SIZE);

  return state.draws.filter((d) => {
    const textHit = String(d.issue).includes(q) || String(d.date || "").toLowerCase().includes(qLower);
    const frontHit = frontNums.length > 0 && frontNums.every((n) => d.front.includes(n));
    const backHit = back != null && d.back.includes(back);
    return textHit || frontHit || backHit;
  });
}

function renderDataTable() {
  const q = ($("#qIssue")?.value || "").trim();
  const rows = limitDataRows(filterDataRows(q));
  state.tableRows = rows;
  const scope = readDataLimit() === Infinity ? "全部" : `最近 ${readDataLimit()} 期`;
  const prefix = q ? `搜索 "${q}"` : scope;
  renderDltTable(rows, `${prefix}：显示 ${rows.length} 条；总数据 ${state.draws.length} 期（倒序）。`);
}

function exportCurrentCsv() {
  const rows = state.tableRows;
  const lines = [["issue", "date", "front", "back"].join(",")];
  for (const d of rows) {
    lines.push([
      d.issue, d.date || "",
      d.front.map(pad2).join(" "),
      d.back.map(pad2).join(" "),
    ].map((v) => `"${String(v).replaceAll('"', '""')}"`).join(","));
  }
  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `dlt-data-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function analyseManualTicket() {
  const front = parseNumList($("#manualFront")?.value, 1, FRONT_SIZE).sort((a, b) => a - b);
  const back = parseNumList($("#manualBack")?.value, 1, BACK_SIZE).sort((a, b) => a - b);
  if (front.length !== FRONT_PICK) {
    renderDltTicketAnalysis({ error: `前区需要 ${FRONT_PICK} 个不重复号码（当前 ${front.length} 个）` });
    return;
  }
  if (back.length !== BACK_PICK) {
    renderDltTicketAnalysis({ error: `后区需要 ${BACK_PICK} 个不重复号码（当前 ${back.length} 个）` });
    return;
  }

  const latest = state.draws[state.draws.length - 1];
  const fkey = front.join(",");
  const bkey = back.join(",");
  const historyHits = state.draws.filter((d) =>
    d.front.join(",") === fkey && d.back.join(",") === bkey);
  renderDltTicketAnalysis({
    front, back,
    sum: frontSum(front),
    span: frontSpan(front),
    oddEven: frontOddEvenRatio(front),
    bigSmall: frontBigSmallRatio(front),
    primeComposite: frontPrimeCompositeRatio(front),
    path012: frontPath012Ratio(front),
    zone: frontZoneRatio(front),
    ac: frontAcValue(front),
    consecutiveGroups: frontConsecutiveGroups(front),
    maxSameTail: frontMaxSameTail(front),
    backSum: backSum(back),
    backOddEven: backOddEvenRatio(back),
    repeatFront: latest ? front.filter((n) => latest.front.includes(n)) : [],
    repeatBack: latest ? back.filter((n) => latest.back.includes(n)) : [],
    historyHits,
  });
}

function renderTools() {
  $("#btnCalcDanTuo")?.addEventListener("click", () => {
    try {
      const n = dltDanTuoTickets({
        danFront: Number($("#danFront")?.value || 0),
        tuoFront: Number($("#tuoFront")?.value || 6),
        danBack: Number($("#danBack")?.value || 0),
        tuoBack: Number($("#tuoBack")?.value || 3),
      });
      $("#danTuoResult").innerHTML =
        `<div><strong class="mono big-num">${n.toLocaleString()}</strong> 注</div>
         <div class="muted">合计 <strong class="mono">${dltPriceOf(n).toLocaleString()}</strong> 元</div>`;
    } catch (e) {
      $("#danTuoResult").innerHTML = `<span class="chip chip-warn">${e.message}</span>`;
    }
  });
  $("#btnCalcComplex")?.addEventListener("click", () => {
    try {
      const front = Number($("#complexFront")?.value || 6);
      const back = Number($("#complexBack")?.value || 3);
      const n = dltComplexTickets(front, back);
      $("#complexResult").innerHTML =
        `<div><strong class="mono big-num">${n.toLocaleString()}</strong> 注</div>
         <div class="muted">合计 <strong class="mono">${dltPriceOf(n).toLocaleString()}</strong> 元</div>`;
    } catch (e) {
      $("#complexResult").innerHTML = `<span class="chip chip-warn">${e.message}</span>`;
    }
  });
  $("#btnAnalyseTicket")?.addEventListener("click", analyseManualTicket);
  $("#manualFront")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") analyseManualTicket();
  });
  $("#manualBack")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") analyseManualTicket();
  });

  const ref = $("#refTable");
  if (ref) {
    ref.innerHTML = "";
    // 单后区 (2) 复式：选 N 个前区
    for (const n of [5, 6, 7, 8, 9, 10, 12, 15]) {
      const tickets = combinations(n, 5) * combinations(2, 2);
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="mono">${n}</td><td class="mono">${tickets.toLocaleString()}</td><td class="mono">${dltPriceOf(tickets).toLocaleString()}</td>`;
      ref.appendChild(tr);
    }
  }
}

function onGenerate() {
  const recent = state.draws.slice(-state.winSize);
  const freqFront = zoneFreq(recent, FRONT_ZONE);
  const freqBack = zoneFreq(recent, BACK_ZONE);
  const cfg = readDltGeneratorConfig();
  const lastDraw = state.draws[state.draws.length - 1];
  const avoidLastFront = cfg.avoidLast && lastDraw ? lastDraw.front : [];
  const avoidLastBack = cfg.avoidLast && lastDraw ? lastDraw.back : [];
  const engine = $("#engine")?.value || "bayes-dpp";
  const seed = $("#seedInput")?.value?.trim() || null;

  try {
    if (engine === "legacy") {
      runLegacyGenerator({ freqFront, freqBack, cfg, avoidLastFront, avoidLastBack });
    } else {
      runAdvancedGenerator({
        freqFront, freqBack,
        totalDraws: recent.length,
        cfg, avoidLastFront, avoidLastBack, engine, seed,
      });
    }
  } catch (e) {
    state.lastTickets = [];
    const btn = $("#btnCopyAll");
    if (btn) btn.disabled = true;
    showDltGenError(e.message || String(e));
    setDltGenDiagnostics("");
    renderDltSamplerDiagnostics(null);
  }
}

function runLegacyGenerator({ freqFront, freqBack, cfg, avoidLastFront, avoidLastBack }) {
  const result = generateDltTickets({
    freqFront, freqBack,
    strategyFront: cfg.strategyFront,
    strategyBack: cfg.strategyBack,
    alpha: cfg.alpha,
    constraints: cfg.constraints,
    count: cfg.count,
    optimize: cfg.optimize,
    includeFront: cfg.includeFront,
    excludeFront: cfg.excludeFront,
    avoidLastFront,
    includeBack: cfg.includeBack,
    excludeBack: cfg.excludeBack,
    avoidLastBack,
  });
  state.lastTickets = result.tickets;
  renderDltTickets(result.tickets, { tries: result.tries, failureReasons: result.failureReasons });
  $("#btnCopyAll").disabled = result.tickets.length === 0;
  const parsed = [
    cfg.includeFront.length ? `前胆 ${cfg.includeFront.length}` : "",
    cfg.includeBack.length ? `后胆 ${cfg.includeBack.length}` : "",
    cfg.excludeFront.length ? `排前 ${cfg.excludeFront.length}` : "",
    cfg.excludeBack.length ? `排后 ${cfg.excludeBack.length}` : "",
    cfg.optimize === "diverse" ? "低撞号/分散覆盖已启用" : "",
  ].filter(Boolean).join(" · ");
  setDltGenDiagnostics(
    `经典引擎 · 已生成 ${result.tickets.length}/${cfg.count} 注 · ${dltPriceOf(result.tickets.length)} 元 · 尝试 ${result.tries} 次` +
    (parsed ? ` · ${parsed}` : "") +
    (avoidLastFront.length ? ` · 已避开上一期 ${avoidLastFront.length} 个前区` : "")
  );
  renderDltSamplerDiagnostics(null);
}

function runAdvancedGenerator({
  freqFront, freqBack, totalDraws, cfg, avoidLastFront, avoidLastBack, engine, seed,
}) {
  const result = generateDltAdvanced({
    freqFront, freqBack, totalDraws,
    method: engine,
    count: cfg.count,
    constraints: cfg.constraints,
    includeFront: cfg.includeFront,
    excludeFront: cfg.excludeFront,
    avoidLastFront,
    includeBack: cfg.includeBack,
    excludeBack: cfg.excludeBack,
    avoidLastBack,
    seed,
  });
  state.lastTickets = result.tickets;
  renderDltTickets(result.tickets, {
    tries: result.diagnostics.tries || 0,
    failureReasons: result.diagnostics.failureReasons || {},
  });
  $("#btnCopyAll").disabled = result.tickets.length === 0;
  setDltGenDiagnostics(
    `${result.diagnostics.samplerLabel || engine} · 已生成 ${result.tickets.length}/${cfg.count} 注 · ${dltPriceOf(result.tickets.length)} 元`
  );
  renderDltSamplerDiagnostics(result.diagnostics, DLT_CONFIG.jackpotProbability);
}

async function onCopyAll() {
  if (!state.lastTickets.length) return;
  await copyToClipboard(state.lastTickets.map(formatDltTicketLine).join("\n"));
  const btn = $("#btnCopyAll");
  const original = btn.textContent;
  btn.textContent = "已复制";
  setTimeout(() => (btn.textContent = original), 1200);
  toast(`已复制 ${state.lastTickets.length} 注到剪贴板`);
}

function onSearch() { renderDataTable(); }

async function onRefresh() {
  setRefreshLoading(true);
  try {
    const { meta, draws } = await loadDltDraws({ noCache: true });
    state.meta = meta;
    state.draws = draws;
    state.winSize = readDltWinSize();
    computeStats();
    renderAll();
    toast(`已刷新：${draws.length} 期`);
  } catch (err) {
    toast(`刷新失败：${err.message || err}`);
  } finally {
    setRefreshLoading(false);
  }
}

function renderAll() {
  renderOverviewAndInsight();
  renderTrendPanel();
  renderDistributionPanel();
  renderScience();
}

function bindInteractions() {
  $("#btnApplyWin")?.addEventListener("click", () => {
    state.winSize = clamp(Number($("#winSize")?.value || 200), 20, 1000);
    computeStats();
    renderAll();
  });
  $("#trendWindow")?.addEventListener("change", renderTrendPanel);
  $("#trendShowStats")?.addEventListener("change", renderTrendPanel);
  $("#tsKind")?.addEventListener("change", renderTimeSeriesPanel);
  $("#tsWindow")?.addEventListener("change", renderTimeSeriesPanel);
  $("#tsMA")?.addEventListener("input", renderTimeSeriesPanel);
  $("#btnPartner")?.addEventListener("click", renderCooccurrencePanel);
  $("#partnerNum")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") renderCooccurrencePanel();
  });
  $("#btnGen")?.addEventListener("click", onGenerate);
  $("#btnCopyAll")?.addEventListener("click", onCopyAll);
  $("#btnSearch")?.addEventListener("click", onSearch);
  $("#qIssue")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onSearch();
  });
  $("#dataLimit")?.addEventListener("change", renderDataTable);
  $("#btnClear")?.addEventListener("click", () => {
    $("#qIssue").value = "";
    renderDataTable();
  });
  $("#btnExportCsv")?.addEventListener("click", () => {
    exportCurrentCsv();
    toast("已导出 CSV");
  });
  $("#btnRefresh")?.addEventListener("click", onRefresh);
  document.addEventListener("ssq:theme", () => {
    renderAll();
  });

  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      requestAnimationFrame(() => {
        const name = t.dataset.tab;
        if (name === "overview") {
          renderDltBars($("#chartFrontAll"), state.freqAllFront, FRONT_SIZE, "front");
          renderDltBars($("#chartBackAll"), state.freqAllBack, BACK_SIZE, "back");
        } else if (name === "insight") {
          renderDltBars($("#chartFrontMiss"), state.missFront, FRONT_SIZE, "front-miss", { unit: "期" });
          renderDltBars($("#chartBackMiss"), state.missBack, BACK_SIZE, "back-miss", { unit: "期" });
        } else if (name === "distribution") {
          renderDistributionPanel();
        } else if (name === "trend") {
          renderTrendPanel();
        }
      });
    });
  });

  window.addEventListener("resize", debounceResize);
  renderTools();
}

let resizeTimer = null;
function debounceResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    renderDltBars($("#chartFrontAll"), state.freqAllFront, FRONT_SIZE, "front");
    renderDltBars($("#chartBackAll"), state.freqAllBack, BACK_SIZE, "back");
    renderDltBars($("#chartFrontMiss"), state.missFront, FRONT_SIZE, "front-miss", { unit: "期" });
    renderDltBars($("#chartBackMiss"), state.missBack, BACK_SIZE, "back-miss", { unit: "期" });
  }, 160);
}

async function main() {
  document.body.dataset.lottery = "dlt";
  setupTabs();
  setupTheme();
  try {
    const { meta, draws, source, fetchError } = await loadDltDraws();
    if (!draws.length) throw new Error("数据为空");
    state.meta = meta;
    state.draws = draws;
    state.winSize = readDltWinSize();
    computeStats();
    renderAll();
    bindInteractions();
    showDltDataSourceBanner(source, fetchError);
    startCountdown();
  } catch (err) {
    showLoadError(String(err.message || err));
  }
}

let countdownTimer = null;
function startCountdown() {
  const tick = () => {
    const now = new Date();
    const target = nextDltDrawTime(now);
    if (!target) return;
    const diff = diffDuration(target, now);
    const setText = (id, v) => {
      const el = $(id);
      if (el) el.textContent = String(v).padStart(2, "0");
    };
    setText("#cdDays", diff.days);
    setText("#cdHours", diff.hours);
    setText("#cdMinutes", diff.minutes);
    setText("#cdSeconds", diff.seconds);

    const cutoff = dltSaleCutoffOf(target);
    const beforeCutoff = now.getTime() < cutoff.getTime();
    const latest = state.draws[state.draws.length - 1];
    const nextIssue = latest ? nextDltIssueOf(latest.issue, target) : null;
    const labelEl = $("#countdownLabel");
    if (labelEl) labelEl.textContent = nextIssue ? `第 ${nextIssue} 期` : "下期";

    const metaEl = $("#countdownMeta");
    if (metaEl) {
      metaEl.innerHTML = `开奖时间：<strong class="mono">${formatChinaTime(target)}</strong> · ${
        beforeCutoff
          ? `投注截止：<strong class="mono">${formatChinaTime(cutoff).slice(11)}</strong>`
          : `<span style="color:var(--gold)">本期投注已截止，等待开奖</span>`
      }`;
    }
  };
  tick();
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(tick, 1000);
}

window.addEventListener("DOMContentLoaded", main);
