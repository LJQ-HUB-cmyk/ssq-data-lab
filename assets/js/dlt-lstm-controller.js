// 大乐透 LSTM 面板控制器（结构同 lstm-controller.js）

import { $, pad2 } from "./utils.js";
import { toast, copyToClipboard } from "./ui.js";
import {
  createDltModel, forwardDltModel, encodeDltSequence,
  topKFront, topKBack,
  serializeDltModel, deserializeDltModel,
  FRONT_DIM, BACK_DIM, FRONT_PICK, BACK_PICK,
} from "./dlt-nn-model.js";
import { trainDltModel, buildDltSamples } from "./dlt-nn-trainer.js";
import {
  backtestDltModel, backtestDltUniformBaseline,
  backtestDltFreqBaseline, backtestDltBayesBaseline,
  DLT_RANDOM_BASELINE,
} from "./dlt-nn-backtest.js";
import {
  bootstrapCI, pairedBootstrap,
  metricAvgHit6, metricBlueAcc,
  reliabilityDiagram,
} from "./nn-statistics.js";
import { createRng } from "./rng.js";

const STORAGE_KEY = "dlt-lstm-model-v1";

const state = {
  draws: [],
  model: null,
  history: null,
  trainSamples: null,
  valSamples: null,
  seqLen: 15,
  shouldStop: false,
  isTraining: false,
};

export function setupDltLstmController(allDraws) {
  state.draws = allDraws;
  $("#btnDltLstmTrain")?.addEventListener("click", onTrain);
  $("#btnDltLstmStop")?.addEventListener("click", () => {
    state.shouldStop = true;
    setStatus("stopping…", "warn");
  });
  $("#btnDltLstmPredict")?.addEventListener("click", onPredict);
  $("#btnDltLstmBacktest")?.addEventListener("click", onBacktest);
  $("#btnDltLstmSave")?.addEventListener("click", onSave);
  $("#btnDltLstmLoad")?.addEventListener("click", onLoad);
  tryAutoLoad();
}

export function updateDltLstmDraws(draws) {
  state.draws = draws;
}

async function onTrain() {
  if (state.isTraining) return;
  state.isTraining = true;
  state.shouldStop = false;
  setControls(true);

  try {
    const seqLen = clampInt("#dltLstmSeqLen", 5, 50, 15);
    const hidden = clampInt("#dltLstmHidden", 16, 256, 64);
    const numLayers = clampInt("#dltLstmLayers", 1, 4, 2);
    const split = clampNum("#dltLstmSplit", 0.5, 0.95, 0.85);
    const lr = clampNum("#dltLstmLr", 1e-4, 0.1, 0.003);
    const epochs = clampInt("#dltLstmEpochs", 1, 100, 20);
    const batchSize = clampInt("#dltLstmBatch", 4, 128, 32);
    const dropoutInput = clampNum("#dltLstmDropIn", 0, 0.5, 0.1);
    const dropoutHidden = clampNum("#dltLstmDropHidden", 0, 0.5, 0.2);
    const dropoutOutput = clampNum("#dltLstmDropOut", 0, 0.5, 0.2);
    const seedStr = $("#dltLstmSeed")?.value?.trim() || `dlt-train-${Date.now()}`;
    state.seqLen = seqLen;

    setStatus("准备样本…");
    const samples = buildDltSamples(state.draws, seqLen);
    if (samples.length < 100) throw new Error(`数据太少，至少需要 ${100 + seqLen} 期`);
    const splitIdx = Math.floor(samples.length * split);
    state.trainSamples = samples.slice(0, splitIdx);
    state.valSamples = samples.slice(splitIdx);

    const arch = `H=${hidden} · L=${numLayers} · drop[in/h/out]=${dropoutInput}/${dropoutHidden}/${dropoutOutput}`;
    setStatus(`训练：${state.trainSamples.length} train / ${state.valSamples.length} val · T=${seqLen} · ${arch}`);
    initCurves();
    resetSeries();

    const memberRng = createRng(seedStr).next;
    state.model = createDltModel({
      hiddenDim: hidden, numLayers,
      dropoutInput, dropoutHidden, dropoutOutput,
      rng: memberRng,
    });
    const t0 = Date.now();
    const result = await trainDltModel(state.model, state.trainSamples, state.valSamples, {
      epochs, batchSize, lr,
      gradClip: 5, patience: 6, weightDecay: 1e-5,
      rng: memberRng,
      onBatch: (b) => {
        if (b.totalBatches) setProgress(b.batch / b.totalBatches);
        if (b.nan) setStatus(`epoch ${b.epoch + 1} batch ${b.batch}: NaN 跳过`, "warn");
      },
      onEpoch: (e) => {
        appendCurve(e);
        setStatus(`epoch ${e.epoch + 1}/${e.totalEpochs} · train ${e.trainLoss.toFixed(4)} · val ${e.valLoss.toFixed(4)} · 前 hit@5 ${e.valFrontHit5.toFixed(3)} · 后 hit@2 ${e.valBackHit2.toFixed(3)}`);
        setProgress((e.epoch + 1) / e.totalEpochs);
      },
      shouldStop: () => state.shouldStop,
    });
    state.history = result.history;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const bestLoss = state.history ? Math.min(...state.history.valLoss).toFixed(4) : "—";
    setStatus(`训练完成 · ${elapsed}s · best val ${bestLoss}`, "ok");
    setProgress(1);

    if (state.history) renderFinalMetrics(state.history);

    $("#btnDltLstmPredict").disabled = false;
    $("#btnDltLstmBacktest").disabled = false;
    $("#btnDltLstmSave").disabled = false;
  } catch (err) {
    setStatus(`错误：${err.message || err}`, "bad");
    console.error(err);
  } finally {
    state.isTraining = false;
    setControls(false);
  }
}

function onPredict() {
  if (!state.model) return;
  const window = state.draws.slice(-state.seqLen);
  const seq = encodeDltSequence(window);
  const fwd = forwardDltModel(state.model, seq, { training: false });
  const top5 = topKFront(fwd.fProbs, FRONT_PICK);
  const top2 = topKBack(fwd.bProbs, BACK_PICK);

  const card = $("#dltLstmPredictionCard");
  const body = $("#dltLstmPredictionBody");
  card.style.display = "";

  const fBars = [];
  let fMax = 0;
  for (let i = 0; i < FRONT_DIM; i++) fMax = Math.max(fMax, fwd.fProbs.data[i]);
  for (let i = 0; i < FRONT_DIM; i++) {
    const p = fwd.fProbs.data[i];
    const w = (p / Math.max(1e-9, fMax)) * 100;
    const isPicked = top5.some(([n]) => n === i + 1);
    fBars.push(`
      <div class="prob-row">
        <span class="ball front ${isPicked ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(i + 1)}</span>
        <span class="prob-bar"><i style="width:${w.toFixed(1)}%"></i></span>
        <span class="mono prob-val">${(p * 100).toFixed(1)}%</span>
      </div>
    `);
  }
  const bRanked = [];
  for (let i = 0; i < BACK_DIM; i++) bRanked.push([i + 1, fwd.bProbs.data[i]]);
  bRanked.sort((a, b) => b[1] - a[1]);
  const bBars = bRanked.map(([n, p]) => {
    const isPicked = top2.some(([m]) => m === n);
    return `
      <div class="prob-row">
        <span class="ball back ${isPicked ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(n)}</span>
        <span class="prob-bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span>
        <span class="mono prob-val">${(p * 100).toFixed(1)}%</span>
      </div>
    `;
  }).join("");

  body.innerHTML = `
    <div class="prediction-pick">
      <div class="prediction-label">Top-5 前 + Top-2 后</div>
      <div class="balls">
        ${top5.map(([n]) => `<span class="ball front">${pad2(n)}</span>`).join("")}
        ${top2.map(([n], idx) => `<span class="ball back${idx === 0 ? " plus" : ""}">${pad2(n)}</span>`).join("")}
      </div>
    </div>
    <div class="prediction-cols">
      <div>
        <div class="card-title">前区 35 路概率</div>
        ${fBars.join("")}
      </div>
      <div>
        <div class="card-title">后区 12 路概率</div>
        ${bBars}
      </div>
    </div>
    <div class="callout" style="margin-top:14px">
      <div class="callout-title">⚠️ 模型局限</div>
      <div class="callout-body">
        本预测器在 walk-forward 回测里前区 Top-5 命中数与<strong>均匀基线 ${DLT_RANDOM_BASELINE.frontHit5.toFixed(3)}</strong>统计上不可区分；
        后区 Top-2 命中数与<strong>${DLT_RANDOM_BASELINE.backHit2.toFixed(3)}</strong>统计上不可区分。<br/>
        <strong>这不是模型差，是大乐透 i.i.d. 抽奖装置没有可学习的时间规律。</strong>
        点下方"回测"亲眼看证据。
      </div>
    </div>
  `;
  toast("已生成预测");
}

async function onBacktest() {
  if (!state.model) return;
  setStatus("回测中…");
  await pause();
  try {
    const seqLen = state.seqLen;
    const valTargets = state.valSamples.map((s) => s.raw.target);
    const valIssues = new Set(valTargets.map((d) => d.issue));
    const splitIdx = state.draws.findIndex((d) => valIssues.has(d.issue));
    if (splitIdx < seqLen) throw new Error("回测窗口不足");
    const trainTail = state.draws.slice(splitIdx - seqLen, splitIdx);
    const testDraws = state.draws.slice(splitIdx);

    const lstmRes = backtestDltModel(state.model, trainTail, testDraws, seqLen);
    const freqRes = backtestDltFreqBaseline(state.draws.slice(0, splitIdx), testDraws);
    const bayesRes = backtestDltBayesBaseline(state.draws.slice(0, splitIdx), testDraws);
    const uniformRes = backtestDltUniformBaseline(testDraws, 80, "uniform-dlt");

    const card = $("#dltLstmBacktestCard");
    const body = $("#dltLstmBacktestBody");
    card.style.display = "";
    body.innerHTML = renderBacktestTable(lstmRes, freqRes, bayesRes, uniformRes, testDraws.length);
    setStatus(`回测完成：${testDraws.length} 期`, "ok");
  } catch (err) {
    setStatus(`回测失败：${err.message || err}`, "bad");
    console.error(err);
  }
}

function renderBacktestTable(lstm, freq, bayes, uniform, n) {
  const fmt = (v, d = 4) => (v == null ? "—" : v.toFixed(d));
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);

  // metricAvgHit6 取的是 record.redHit6——我们把 fHit5 映射到 redHit6 字段
  const lstmCi = bootstrapCI(lstm.records, metricAvgHit6, { B: 500, seed: "dlt-bt-lstm" });
  const uniCi = bootstrapCI(uniform.records, metricAvgHit6, { B: 500, seed: "dlt-bt-uni" });
  const paired = pairedBootstrap(lstm.records, uniform.records, metricAvgHit6, { B: 500, seed: "dlt-bt-paired" });

  const formatCI = (mean, lo, hi) => `${mean.toFixed(3)} <span class="muted">[${lo.toFixed(3)}, ${hi.toFixed(3)}]</span>`;

  const rows = [];
  rows.push({
    label: "LSTM（前区 5 选 35）",
    tag: "primary",
    f5: formatCI(lstm.summary.avgFrontHit5, lstmCi.lower, lstmCi.upper),
    f7: lstm.summary.avgFrontHit7.toFixed(3),
    b2: lstm.summary.avgBackHit2.toFixed(3),
    brier: lstm.summary.avgBrier,
    ll: lstm.summary.avgFrontLL + lstm.summary.avgBackLL,
  });
  rows.push({
    label: "贝叶斯后验 baseline",
    f5: bayes.summary.avgFrontHit5.toFixed(3),
    f7: bayes.summary.avgFrontHit7.toFixed(3),
    b2: bayes.summary.avgBackHit2.toFixed(3),
  });
  rows.push({
    label: "频率 baseline",
    f5: freq.summary.avgFrontHit5.toFixed(3),
    f7: freq.summary.avgFrontHit7.toFixed(3),
    b2: freq.summary.avgBackHit2.toFixed(3),
  });
  rows.push({
    label: "均匀随机 baseline (80×MC)",
    f5: formatCI(uniform.summary.avgFrontHit5, uniCi.lower, uniCi.upper),
    f7: uniform.summary.avgFrontHit7.toFixed(3),
    b2: uniform.summary.avgBackHit2.toFixed(3),
  });
  rows.push({
    label: "理论期望（任意预测器）",
    tag: "theory",
    f5: DLT_RANDOM_BASELINE.frontHit5.toFixed(3),
    f7: DLT_RANDOM_BASELINE.frontHit7.toFixed(3),
    b2: DLT_RANDOM_BASELINE.backHit2.toFixed(3),
  });

  const tableRows = rows.map((r) => `
    <tr class="${r.tag === "primary" ? "row-primary" : ""}${r.tag === "theory" ? " row-theory" : ""}">
      <td>${r.label}</td>
      <td class="mono">${r.f5}</td>
      <td class="mono">${r.f7}</td>
      <td class="mono">${r.b2}</td>
      <td class="mono">${r.brier != null ? fmt(r.brier) : "—"}</td>
      <td class="mono">${r.ll != null ? fmt(r.ll, 3) : "—"}</td>
    </tr>
  `).join("");

  const ciIncludesZero = paired.lower <= 0 && paired.upper >= 0;
  const verdict = ciIncludesZero
    ? `LSTM 与均匀随机的前区 hit@5 差值 95% CI = [${paired.lower.toFixed(3)}, ${paired.upper.toFixed(3)}] <strong>包含 0</strong>。差异不显著——这正是预期：大乐透前后区都是 i.i.d. 摇号。`
    : `差值 95% CI = [${paired.lower.toFixed(3)}, ${paired.upper.toFixed(3)}] <strong>不含 0</strong>。但要警惕：(1) ${n} 期的局部偏差不一定泛化；(2) 即便差异真实，也不构成可重复的预测能力。`;

  const reliab = reliabilityDiagram(lstm.records, { bins: 10 });

  return `
    <div class="bt-table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th>方法</th>
            <th>前 hit@5 [95% CI]</th>
            <th>前 hit@7</th>
            <th>后 hit@2</th>
            <th>Brier</th>
            <th>NLL</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
    <div class="callout" style="margin-top:14px">
      <div class="callout-title">配对 Bootstrap 显著性检验（B=500）</div>
      <div class="callout-body">${verdict}</div>
    </div>
    <div class="card" style="margin-top:14px; padding: var(--space-4)">
      <div class="card-title">校准曲线 · Reliability Diagram <span class="card-num">ECE = ${reliab.ece.toFixed(4)}</span></div>
      ${renderReliab(reliab)}
      <div class="hint">点越接近对角线 y=x 越好。完美校准 ECE = 0。</div>
    </div>
  `;
}

function renderReliab(reliab) {
  const W = 380, H = 220;
  const padL = 36, padR = 12, padT = 12, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const dots = reliab.points.map((p) => {
    if (p.observedFreq == null) return "";
    const cx = padL + p.avgPred * innerW;
    const cy = padT + (1 - p.observedFreq) * innerH;
    const r = 2 + Math.min(8, Math.sqrt(p.count) * 0.5);
    return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="var(--dlt-front)" opacity="0.78"/>`;
  }).join("");
  const refLine = `<line x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT}" stroke="rgba(255,255,255,.35)" stroke-dasharray="3 4"/>`;
  const xAxis = `<line x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const yAxis = `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) => {
    const x = padL + v * innerW;
    const y = padT + (1 - v) * innerH;
    return `
      <text x="${x}" y="${padT + innerH + 14}" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.5)" font-family="JetBrains Mono, monospace">${v.toFixed(2)}</text>
      <text x="${padL - 4}" y="${y + 3}" text-anchor="end" font-size="9" fill="rgba(255,255,255,.5)" font-family="JetBrains Mono, monospace">${v.toFixed(2)}</text>
    `;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">${xAxis}${yAxis}${refLine}${ticks}${dots}</svg>`;
}

/* ============================================================
 * 持久化
 * ============================================================ */
function onSave() {
  if (!state.model) return;
  try {
    const payload = {
      model: serializeDltModel(state.model),
      seqLen: state.seqLen,
      history: state.history,
      savedAt: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    toast("DLT 模型已保存");
  } catch (e) {
    toast(`保存失败：${e.message}`);
  }
}

function tryAutoLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    state.model = deserializeDltModel(payload.model);
    state.seqLen = payload.seqLen || 15;
    state.history = payload.history || null;
    setStatus(`已自动加载 DLT 模型（${payload.savedAt?.slice(0, 19) || ""}）`, "ok");
    if (state.history) renderFinalMetrics(state.history);
    $("#btnDltLstmPredict").disabled = false;
  } catch {}
}

function onLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) { toast("没有找到已保存的模型"); return; }
    const payload = JSON.parse(raw);
    state.model = deserializeDltModel(payload.model);
    state.seqLen = payload.seqLen || 15;
    state.history = payload.history || null;
    if (state.history) renderFinalMetrics(state.history);
    $("#btnDltLstmPredict").disabled = false;
    $("#btnDltLstmBacktest").disabled = !state.valSamples;
    $("#btnDltLstmSave").disabled = false;
    toast("已加载");
  } catch (e) {
    toast(`加载失败：${e.message}`);
  }
}

/* ============================================================
 * UI helpers
 * ============================================================ */
function clampInt(sel, lo, hi, def) {
  const n = parseInt($(sel)?.value, 10);
  if (!Number.isFinite(n)) return def;
  return Math.min(hi, Math.max(lo, n));
}
function clampNum(sel, lo, hi, def) {
  const n = parseFloat($(sel)?.value);
  if (!Number.isFinite(n)) return def;
  return Math.min(hi, Math.max(lo, n));
}
function setStatus(text, kind = "") {
  const el = $("#dltLstmStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `lstm-status ${kind ? `is-${kind}` : ""}`;
}
function setProgress(ratio) {
  const bar = $("#dltLstmProgressBar");
  if (!bar) return;
  bar.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
}
function setControls(training) {
  $("#btnDltLstmTrain").disabled = training;
  $("#btnDltLstmStop").disabled = !training;
  if (training) {
    $("#btnDltLstmPredict").disabled = true;
    $("#btnDltLstmBacktest").disabled = true;
    $("#btnDltLstmSave").disabled = true;
  }
}
function initCurves() {
  const el = $("#dltLstmCurves");
  if (!el) return;
  el.innerHTML = `
    <div class="curve-wrap" id="dltLstmLossCurve" data-label="Loss"></div>
    <div class="curve-wrap" id="dltLstmHitCurve" data-label="Front Hit@5"></div>
  `;
}
const liveSeries = { trainLoss: [], valLoss: [], hit5: [], hit2: [] };
function resetSeries() {
  liveSeries.trainLoss = []; liveSeries.valLoss = [];
  liveSeries.hit5 = []; liveSeries.hit2 = [];
}
function appendCurve(e) {
  liveSeries.trainLoss.push(e.trainLoss);
  liveSeries.valLoss.push(e.valLoss);
  liveSeries.hit5.push(e.valFrontHit5);
  liveSeries.hit2.push(e.valBackHit2);
  drawSpark("#dltLstmLossCurve", [
    { label: "train", series: liveSeries.trainLoss, color: "var(--dlt-front)" },
    { label: "val", series: liveSeries.valLoss, color: "var(--dlt-back)" },
  ]);
  drawSpark("#dltLstmHitCurve", [
    { label: "val hit@5", series: liveSeries.hit5, color: "var(--acid)" },
  ], { ref: DLT_RANDOM_BASELINE.frontHit5, refLabel: `随机基线 ${DLT_RANDOM_BASELINE.frontHit5.toFixed(3)}` });
}
function drawSpark(sel, series, extra = {}) {
  const el = document.querySelector(sel);
  if (!el) return;
  const W = el.clientWidth || 400;
  const H = 80;
  const padL = 32, padR = 8, padT = 8, padB = 14;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const all = series.flatMap((s) => s.series).concat(extra.ref != null ? [extra.ref] : []);
  if (all.length === 0) return;
  const minV = Math.min(...all);
  const maxV = Math.max(...all);
  const range = maxV - minV || 1;
  const yScale = (v) => padT + innerH - ((v - minV) / range) * innerH;
  const lines = series.map((s) => {
    if (s.series.length < 1) return "";
    const dx = innerW / Math.max(1, s.series.length - 1);
    const pts = s.series.map((v, i) => `${(padL + i * dx).toFixed(2)},${yScale(v).toFixed(2)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.6" stroke-linecap="round"/>`;
  }).join("");
  const refLine = extra.ref != null
    ? `<line x1="${padL}" x2="${W - padR}" y1="${yScale(extra.ref)}" y2="${yScale(extra.ref)}" stroke="rgba(255,255,255,.35)" stroke-dasharray="2 4"/>
       <text x="${padL + 4}" y="${yScale(extra.ref) - 4}" font-size="9" fill="rgba(255,255,255,.55)" font-family="JetBrains Mono, monospace">${extra.refLabel}</text>`
    : "";
  el.innerHTML = `
    <div class="curve-label">${el.dataset.label}</div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
      ${refLine}
      ${lines}
    </svg>
    <div class="curve-legend">${series.map((s) => `<span style="color:${s.color}">— ${s.label}</span>`).join("&nbsp;&nbsp;")}</div>
  `;
}
function renderFinalMetrics(history) {
  if (!history || !history.epochs.length) return;
  const last = history.epochs.length - 1;
  const items = [
    ["最佳验证损失", Math.min(...history.valLoss).toFixed(4)],
    ["末次训练损失", history.trainLoss[last]?.toFixed(4) ?? "—"],
    ["末次验证损失", history.valLoss[last]?.toFixed(4) ?? "—"],
    ["验证 前区 Hit@5", `${history.valFrontHit5[last]?.toFixed(3)}（基线 ${DLT_RANDOM_BASELINE.frontHit5.toFixed(3)}）`],
    ["验证 后区 Hit@2", `${history.valBackHit2[last]?.toFixed(3)}（基线 ${DLT_RANDOM_BASELINE.backHit2.toFixed(3)}）`],
    ["训练 epoch 数", String(history.epochs.length)],
  ];
  const el = $("#dltLstmMetrics");
  if (!el) return;
  el.innerHTML = items.map(([k, v]) => `
    <div class="diag-line"><span>${k}</span><strong class="mono">${v}</strong></div>
  `).join("");
}
function pause(ms = 0) { return new Promise((r) => setTimeout(r, ms)); }
