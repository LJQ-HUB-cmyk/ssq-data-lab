import { $, $$, clamp, pad2 } from "./utils.js";
import { loadDraws } from "./data.js";
import {
  freqFromDraws,
  missCounts,
  topN,
  sumOf,
  spanOf,
  RED_MAX,
  BLUE_MAX,
} from "./stats.js";
import { renderBars } from "./chart.js";
import { renderTrend } from "./trend-chart.js";
import { generateTickets } from "./generator.js";
import {
  oddEvenRatio,
  bigSmallRatio,
  primeCompositeRatio,
  path012Ratio,
  zoneRatio,
  acValue,
  groupBy,
  histogram,
} from "./distribution.js";
import { redChi, blueChi, chiSquaredPValue } from "./chi-square.js";
import { danTuoTickets, complexTickets, combinations, priceOf } from "./combinatorics.js";
import { buildTrendMatrix } from "./trend.js";
import {
  setupTabs,
  renderLatest,
  renderHeroMeta,
  renderRank,
  renderTable,
  renderInsightChips,
  renderTickets,
  showLoadError,
  showDataSourceBanner,
  setRefreshLoading,
  readWinSize,
  readGeneratorConfig,
  showGenError,
  setGenDiagnostics,
} from "./ui.js";

const state = {
  meta: {},
  draws: [],
  winSize: 200,
  freqAllRed: null,
  freqAllBlue: null,
  freqRecentRed: null,
  freqRecentBlue: null,
  missRed: null,
  missBlue: null,
};

function computeStats() {
  state.freqAllRed = freqFromDraws(state.draws, "reds", RED_MAX);
  state.freqAllBlue = freqFromDraws(state.draws, "blue", BLUE_MAX);
  const recent = state.draws.slice(-state.winSize);
  state.freqRecentRed = freqFromDraws(recent, "reds", RED_MAX);
  state.freqRecentBlue = freqFromDraws(recent, "blue", BLUE_MAX);
  state.missRed = missCounts(state.draws, "reds", RED_MAX);
  state.missBlue = missCounts(state.draws, "blue", BLUE_MAX);
}

function renderOverviewAndInsight() {
  const latest = state.draws[state.draws.length - 1];
  renderHeroMeta(state.meta, state.draws);
  renderLatest(latest);

  renderBars($("#chartRedAll"), state.freqAllRed, RED_MAX, "red");
  renderBars($("#chartBlueAll"), state.freqAllBlue, BLUE_MAX, "blue");
  renderRank($("#rankRedRecent"), topN(state.freqRecentRed, 8, RED_MAX));
  renderRank($("#rankBlueRecent"), topN(state.freqRecentBlue, 6, BLUE_MAX));

  renderBars($("#chartRedMiss"), state.missRed, RED_MAX, "miss", { unit: "期" });
  renderBars($("#chartBlueMiss"), state.missBlue, BLUE_MAX, "miss", { unit: "期" });
  renderInsightChips(state);

  renderTable(state.draws.slice(-50).reverse(), `共 ${state.draws.length} 期；显示最近 50 期（倒序）。`);
}

function renderTrendPanel() {
  const win = Number($("#trendWindow").value || 50);
  const rows = buildTrendMatrix(state.draws, win);
  renderTrend($("#trendRed"), rows, { size: RED_MAX, kind: "red" });
  renderTrend($("#trendBlue"), rows, { size: BLUE_MAX, kind: "blue" });
}

function renderRatioList(el, entries, top = 8) {
  el.innerHTML = "";
  const totalCount = entries.reduce((a, b) => a + b[1], 0);
  const max = entries[0]?.[1] || 1;
  for (const [k, v] of entries.slice(0, top)) {
    const pct = ((v / totalCount) * 100).toFixed(1);
    const w = Math.round((v / max) * 100);
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="ratio-row">
        <span class="ratio-key mono">${k}</span>
        <span class="ratio-bar"><i style="width:${w}%"></i></span>
        <span class="ratio-val mono">${v} · ${pct}%</span>
      </div>
    `;
    el.appendChild(li);
  }
}

function renderDistributionPanel() {
  const draws = state.draws;
  renderRatioList($("#distOddEven"), groupBy(draws, (r) => oddEvenRatio(r)));
  renderRatioList($("#distBigSmall"), groupBy(draws, (r) => bigSmallRatio(r)));
  renderRatioList($("#distPrime"), groupBy(draws, (r) => primeCompositeRatio(r)));
  renderRatioList($("#dist012"), groupBy(draws, (r) => path012Ratio(r)));
  renderRatioList($("#distZone"), groupBy(draws, (r) => zoneRatio(r)));
  const acEntries = groupBy(draws, (r) => String(acValue(r)))
    .sort((a, b) => Number(a[0]) - Number(b[0]));
  renderRatioList($("#distAC"), acEntries, 11);

  // 和值分布直方图：用现有 renderBars 需要定长数组，转成 [0..N] 形式
  const sums = draws.map((d) => sumOf(d.reds));
  const spans = draws.map((d) => spanOf(d.reds));
  drawHistogram($("#chartSum"), sums, 21, 183);
  drawHistogram($("#chartSpan"), spans, 5, 32);
}

function drawHistogram(container, values, min, max) {
  const buckets = Array(max - min + 1).fill(0);
  for (const v of values) {
    if (v >= min && v <= max) buckets[v - min]++;
  }
  // 用一个"伪 1-indexed"数组喂给 renderBars（它忽略 index 0）
  const arr = [0, ...buckets];
  renderBars(container, arr, buckets.length, "red", { unit: "期" });
}

function renderScience() {
  const rc = redChi(state.draws);
  const bc = blueChi(state.draws);
  const pR = chiSquaredPValue(rc.chi, rc.df);
  const pB = chiSquaredPValue(bc.chi, bc.df);

  $("#chiRed").textContent = rc.chi.toFixed(2);
  $("#pRed").textContent = formatP(pR);
  $("#rejectRed").innerHTML = verdictChip(pR);
  $("#chiBlue").textContent = bc.chi.toFixed(2);
  $("#pBlue").textContent = formatP(pB);
  $("#rejectBlue").innerHTML = verdictChip(pB);

  const rRej = pR < 0.05, bRej = pB < 0.05;
  let verdict;
  if (!rRej && !bRej) {
    verdict = `红球 p=${formatP(pR)}、蓝球 p=${formatP(pB)}，两者都 <strong>显著大于 0.05</strong>。
      统计上没有证据拒绝"均匀分布"假设——换句话说，
      <strong>历史数据与"完全随机摇出"完全一致</strong>。
      "冷号热号"只是随机波动的自然产物，不构成下期的预测依据。`;
  } else if (rRej && !bRej) {
    verdict = `红球 p=${formatP(pR)} < 0.05，观察到显著偏差；蓝球 p=${formatP(pB)} 未显著。
      红球偏差可能来自摇奖设备物理不对称、样本量、或碰巧的极端情况。
      <strong>显著性 ≠ 可预测性</strong>，它只告诉你"均匀假设在此样本下成立/不成立"。`;
  } else {
    verdict = `红球 p=${formatP(pR)}、蓝球 p=${formatP(pB)}，存在显著偏差。
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
  if (p < 0.05) {
    return `<span class="chip chip-warn">拒绝（p&lt;0.05）</span>`;
  }
  return `<span class="chip chip-ok">不拒绝（数据与均匀一致）</span>`;
}

function renderTools() {
  $("#btnCalcDanTuo").addEventListener("click", () => {
    try {
      const n = danTuoTickets({
        danCount: Number($("#danCount").value),
        tuoCount: Number($("#tuoCount").value),
        blueCount: Number($("#blueCount").value),
      });
      $("#danTuoResult").innerHTML =
        `<div><strong class="mono big-num">${n.toLocaleString()}</strong> 注</div>
         <div class="muted">合计 <strong class="mono">${priceOf(n).toLocaleString()}</strong> 元</div>`;
    } catch (e) {
      $("#danTuoResult").innerHTML = `<span class="chip chip-warn">${e.message}</span>`;
    }
  });
  $("#btnCalcComplex").addEventListener("click", () => {
    try {
      const red = Number($("#complexRed").value);
      const blue = Number($("#complexBlue").value);
      const n = complexTickets(red, blue);
      $("#complexResult").innerHTML =
        `<div><strong class="mono big-num">${n.toLocaleString()}</strong> 注</div>
         <div class="muted">合计 <strong class="mono">${priceOf(n).toLocaleString()}</strong> 元</div>`;
    } catch (e) {
      $("#complexResult").innerHTML = `<span class="chip chip-warn">${e.message}</span>`;
    }
  });

  const ref = $("#refTable");
  ref.innerHTML = "";
  for (const r of [6, 7, 8, 9, 10, 12, 15, 20]) {
    const n = combinations(r, 6);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="mono">${r}</td><td class="mono">${n.toLocaleString()}</td><td class="mono">${priceOf(n).toLocaleString()}</td>`;
    ref.appendChild(tr);
  }
}

function onGenerate() {
  const recent = state.draws.slice(-state.winSize);
  const freqR = freqFromDraws(recent, "reds", RED_MAX);
  const freqB = freqFromDraws(recent, "blue", BLUE_MAX);
  const cfg = readGeneratorConfig();
  const lastDraw = state.draws[state.draws.length - 1];
  const avoidLast = cfg.avoidLast && lastDraw ? lastDraw.reds : [];
  try {
    const result = generateTickets({
      freqR, freqB,
      strategyRed: cfg.strategyRed,
      strategyBlue: cfg.strategyBlue,
      alpha: cfg.alpha,
      constraints: cfg.constraints,
      count: cfg.count,
      includeRed: cfg.includeRed,
      excludeRed: cfg.excludeRed,
      excludeBlue: cfg.excludeBlue,
      avoidLast,
    });
    renderTickets(result.tickets, { tries: result.tries, failureReasons: result.failureReasons });
    setGenDiagnostics(
      `已生成 ${result.tickets.length}/${cfg.count} 注 · 尝试 ${result.tries} 次` +
      (avoidLast.length ? ` · 已避开上一期 ${avoidLast.length} 个红球` : "")
    );
  } catch (e) {
    showGenError(e.message || String(e));
    setGenDiagnostics("");
  }
}

function onSearch() {
  const q = ($("#qIssue").value || "").trim();
  if (!q) {
    renderTable(state.draws.slice(-50).reverse(), `共 ${state.draws.length} 期；显示最近 50 期（倒序）。`);
    return;
  }
  const hit = state.draws.filter((d) => d.issue.includes(q)).slice(-120).reverse();
  renderTable(hit, `搜索 "${q}" ：命中 ${hit.length} 条（最多展示 120 条）。`);
}

async function onRefresh() {
  setRefreshLoading(true);
  try {
    const { meta, draws } = await loadDraws({ noCache: true });
    state.meta = meta;
    state.draws = draws;
    state.winSize = readWinSize();
    computeStats();
    renderAll();
  } catch (err) {
    alert(`刷新失败：${err.message || err}`);
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
  $("#btnApplyWin").addEventListener("click", () => {
    state.winSize = clamp(Number($("#winSize").value || 200), 20, 1000);
    computeStats();
    renderAll();
  });
  $("#trendWindow").addEventListener("change", renderTrendPanel);
  $("#btnGen").addEventListener("click", onGenerate);
  $("#btnSearch").addEventListener("click", onSearch);
  $("#qIssue").addEventListener("keydown", (e) => {
    if (e.key === "Enter") onSearch();
  });
  $("#btnClear").addEventListener("click", () => {
    $("#qIssue").value = "";
    renderTable(state.draws.slice(-50).reverse(), `共 ${state.draws.length} 期；显示最近 50 期（倒序）。`);
  });
  $("#btnRefresh").addEventListener("click", onRefresh);
  window.addEventListener("resize", debounceResize);
  renderTools();
}

let resizeTimer = null;
function debounceResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    renderBars($("#chartRedAll"), state.freqAllRed, RED_MAX, "red");
    renderBars($("#chartBlueAll"), state.freqAllBlue, BLUE_MAX, "blue");
    renderBars($("#chartRedMiss"), state.missRed, RED_MAX, "miss", { unit: "期" });
    renderBars($("#chartBlueMiss"), state.missBlue, BLUE_MAX, "miss", { unit: "期" });
  }, 160);
}

async function main() {
  setupTabs();
  try {
    const { meta, draws, source, fetchError } = await loadDraws();
    if (!draws.length) throw new Error("数据为空");
    state.meta = meta;
    state.draws = draws;
    state.winSize = readWinSize();
    computeStats();
    renderAll();
    bindInteractions();
    showDataSourceBanner(source, fetchError);
  } catch (err) {
    showLoadError(String(err.message || err));
  }
}

window.addEventListener("DOMContentLoaded", main);
