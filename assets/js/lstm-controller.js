// LSTM 面板的 UI 控制器
//
// 职责：从 #panel-lstm 表单读参数 → 调用 trainer/backtest → 把结果渲染回页面

import { $, pad2 } from "./utils.js";
import { toast, copyToClipboard } from "./ui.js";
import {
  createModel, forwardModel, encodeSequence,
  topKRed, argMaxBlue,
  serializeModel, deserializeModel,
  RED_DIM, BLUE_DIM,
} from "./nn-ssq-model.js";
import { trainModel, buildSamples } from "./nn-trainer.js";
import {
  backtestModel, backtestFreqBaseline, backtestBayesBaseline, backtestUniformBaseline,
  RANDOM_BASELINE,
} from "./nn-backtest.js";
import { createRng } from "./rng.js";

const STORAGE_KEY = "ssq-lstm-model-v1";

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

export function setupLstmController(allDraws) {
  state.draws = allDraws;

  $("#btnLstmTrain")?.addEventListener("click", onTrain);
  $("#btnLstmStop")?.addEventListener("click", () => {
    state.shouldStop = true;
    setStatus("stopping…", "warn");
  });
  $("#btnLstmPredict")?.addEventListener("click", onPredict);
  $("#btnLstmBacktest")?.addEventListener("click", onBacktest);
  $("#btnLstmSave")?.addEventListener("click", onSave);
  $("#btnLstmLoad")?.addEventListener("click", onLoad);

  // 启动时尝试加载已保存的模型
  tryAutoLoadModel();
}

export function updateLstmDraws(draws) {
  state.draws = draws;
}

/* ============================================================
 * 训练
 * ============================================================ */
async function onTrain() {
  if (state.isTraining) return;
  state.isTraining = true;
  state.shouldStop = false;
  setControlsDuringTraining(true);

  try {
    const seqLen = clampInt("#lstmSeqLen", 5, 50, 15);
    const hidden = clampInt("#lstmHidden", 16, 256, 64);
    const split = clampNum("#lstmSplit", 0.5, 0.95, 0.85);
    const lr = clampNum("#lstmLr", 1e-4, 0.1, 0.003);
    const epochs = clampInt("#lstmEpochs", 1, 100, 20);
    const batchSize = clampInt("#lstmBatch", 4, 128, 32);
    const seedStr = $("#lstmSeed")?.value?.trim() || `train-${Date.now()}`;
    const rng = createRng(seedStr).next;
    state.seqLen = seqLen;

    setStatus("准备样本…");
    const samples = buildSamples(state.draws, seqLen);
    if (samples.length < 100) throw new Error(`数据太少，至少需要 ${100 + seqLen} 期`);
    const splitIdx = Math.floor(samples.length * split);
    state.trainSamples = samples.slice(0, splitIdx);
    state.valSamples = samples.slice(splitIdx);

    setStatus(`训练中：${state.trainSamples.length} 训练 / ${state.valSamples.length} 验证 · H=${hidden} · T=${seqLen}`);
    state.model = createModel({ hiddenDim: hidden, rng });
    state.history = null;
    initCurves();

    const t0 = Date.now();
    const result = await trainModel(state.model, state.trainSamples, state.valSamples, {
      epochs, batchSize, lr,
      gradClip: 5,
      patience: 6,
      weightDecay: 1e-5,
      rng,
      onBatch: (b) => {
        if (b.totalBatches) setProgress(b.batch / b.totalBatches);
        if (b.nan) setStatus(`epoch ${b.epoch + 1} batch ${b.batch}: NaN 跳过`, "warn");
      },
      onEpoch: (e) => {
        appendCurve(e);
        setStatus(`epoch ${e.epoch + 1}/${e.totalEpochs} · train ${e.trainLoss.toFixed(4)} · val ${e.valLoss.toFixed(4)} · 红 hit@6 ${e.valRedHit6.toFixed(3)} · 蓝 acc ${(e.valBlueAcc * 100).toFixed(1)}%`);
        setProgress((e.epoch + 1) / e.totalEpochs);
      },
      shouldStop: () => state.shouldStop,
    });
    state.history = result.history;
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    setStatus(`训练完成 · ${elapsed}s · best val ${result.bestValLoss.toFixed(4)}`, "ok");
    setProgress(1);

    renderFinalMetrics(result.history);

    $("#btnLstmPredict").disabled = false;
    $("#btnLstmBacktest").disabled = false;
    $("#btnLstmSave").disabled = false;
  } catch (err) {
    setStatus(`错误：${err.message || err}`, "bad");
    console.error(err);
  } finally {
    state.isTraining = false;
    setControlsDuringTraining(false);
  }
}

/* ============================================================
 * 预测下一期
 * ============================================================ */
function onPredict() {
  if (!state.model) return;
  const window = state.draws.slice(-state.seqLen);
  const seq = encodeSequence(window);
  const fwd = forwardModel(state.model, seq);
  const top6 = topKRed(fwd.redProbs, 6);
  const top10 = topKRed(fwd.redProbs, 10);
  const blueArg = argMaxBlue(fwd.blueProbs);
  const blueRanked = [];
  for (let i = 0; i < BLUE_DIM; i++) blueRanked.push([i + 1, fwd.blueProbs.data[i]]);
  blueRanked.sort((a, b) => b[1] - a[1]);

  const card = $("#lstmPredictionCard");
  const body = $("#lstmPredictionBody");
  card.style.display = "";

  // 红球 33 个号码概率热度条
  const redBars = [];
  let redMax = 0;
  for (let i = 0; i < RED_DIM; i++) redMax = Math.max(redMax, fwd.redProbs.data[i]);
  for (let i = 0; i < RED_DIM; i++) {
    const p = fwd.redProbs.data[i];
    const w = (p / Math.max(1e-9, redMax)) * 100;
    const isPicked = top6.some(([n]) => n === i + 1);
    redBars.push(`
      <div class="prob-row">
        <span class="ball red ${isPicked ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(i + 1)}</span>
        <span class="prob-bar"><i style="width:${w.toFixed(1)}%"></i></span>
        <span class="mono prob-val">${(p * 100).toFixed(1)}%</span>
      </div>
    `);
  }
  // 蓝球
  const blueBars = blueRanked.map(([n, p]) => `
    <div class="prob-row">
      <span class="ball blue ${n === blueArg.num ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(n)}</span>
      <span class="prob-bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span>
      <span class="mono prob-val">${(p * 100).toFixed(1)}%</span>
    </div>
  `).join("");

  body.innerHTML = `
    <div class="prediction-pick">
      <div class="prediction-label">Top-6 红 + 蓝</div>
      <div class="balls">
        ${top6.map(([n]) => `<span class="ball red">${pad2(n)}</span>`).join("")}
        <span class="ball blue plus">${pad2(blueArg.num)}</span>
      </div>
    </div>
    <div class="prediction-cols">
      <div>
        <div class="card-title">红球 33 路概率</div>
        ${redBars.join("")}
      </div>
      <div>
        <div class="card-title">蓝球 16 路概率</div>
        ${blueBars}
      </div>
    </div>
    <div class="callout" style="margin-top:14px">
      <div class="callout-title">⚠️ 模型局限性</div>
      <div class="callout-body">
        本预测器在 walk-forward 回测里的红球 Top-6 命中数与<strong>均匀随机基线（≈ 1.09 / 期）</strong>统计上不可区分；
        蓝球 Top-1 准确率与<strong>1/16 ≈ 6.25%</strong> 的随机基线统计上不可区分。<br/>
        <strong>这不是因为模型差，而是因为彩票本身没有可学习的时间规律。</strong>
        点击下方「Walk-forward 回测」可亲眼验证。预测号码<strong>不提高</strong>实际中奖概率，仅供学习与娱乐。
      </div>
    </div>
  `;
  toast("已生成预测");
}

/* ============================================================
 * Walk-forward 回测
 * ============================================================ */
async function onBacktest() {
  if (!state.model) return;
  setStatus("回测中…");
  await pause();
  try {
    // 用最后 valSamples 对应的真实期数做回测
    const seqLen = state.seqLen;
    // 找到 train/val 的分界 issue
    const valTargets = state.valSamples.map((s) => s.raw.target);
    const valIssues = new Set(valTargets.map((d) => d.issue));
    const splitIdx = state.draws.findIndex((d) => valIssues.has(d.issue));
    if (splitIdx < seqLen) throw new Error("回测窗口不足");
    const trainTail = state.draws.slice(splitIdx - seqLen, splitIdx);
    const testDraws = state.draws.slice(splitIdx);

    const lstmRes = backtestModel(state.model, trainTail, testDraws, seqLen);
    const freqRes = backtestFreqBaseline(state.draws.slice(0, splitIdx), testDraws);
    const bayesRes = backtestBayesBaseline(state.draws.slice(0, splitIdx), testDraws);
    const uniformRes = backtestUniformBaseline(testDraws, 100, "uniform-baseline");

    const card = $("#lstmBacktestCard");
    const body = $("#lstmBacktestBody");
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

  const rows = [
    {
      label: "LSTM（你的模型）",
      tag: "primary",
      redHit6: lstm.summary.avgRedHit6,
      redHit8: lstm.summary.avgRedHit8,
      blueAcc: lstm.summary.blueAccuracy,
      brier: lstm.summary.avgBrier,
      ll: lstm.summary.avgRedLL + lstm.summary.avgBlueLL,
    },
    {
      label: "贝叶斯后验 baseline",
      redHit6: bayes.summary.avgRedHit6,
      redHit8: bayes.summary.avgRedHit8,
      blueAcc: bayes.summary.blueAccuracy,
    },
    {
      label: "频率 baseline",
      redHit6: freq.summary.avgRedHit6,
      redHit8: freq.summary.avgRedHit8,
      blueAcc: freq.summary.blueAccuracy,
    },
    {
      label: "均匀随机 baseline (100次蒙特卡洛)",
      redHit6: uniform.summary.avgRedHit6,
      redHit8: uniform.summary.avgRedHit8,
      blueAcc: uniform.summary.blueAccuracy,
    },
    {
      label: "理论期望（任意预测器渐近）",
      tag: "theory",
      redHit6: RANDOM_BASELINE.redHit6,
      redHit8: RANDOM_BASELINE.redHit8,
      blueAcc: RANDOM_BASELINE.blueAcc,
    },
  ];

  const tableRows = rows.map((r) => `
    <tr class="${r.tag === "primary" ? "row-primary" : ""}${r.tag === "theory" ? " row-theory" : ""}">
      <td>${r.label}</td>
      <td class="mono">${fmt(r.redHit6, 3)}</td>
      <td class="mono">${fmt(r.redHit8, 3)}</td>
      <td class="mono">${pct(r.blueAcc)}</td>
      <td class="mono">${r.brier != null ? fmt(r.brier) : "—"}</td>
      <td class="mono">${r.ll != null ? fmt(r.ll, 3) : "—"}</td>
    </tr>
  `).join("");

  // 显著性提示：检查 LSTM 是否在 95% CI 内显著优于均匀
  const lstmHit6 = lstm.summary.avgRedHit6;
  const uniHit6 = uniform.summary.avgRedHit6;
  // 红球 hit@6 ~ 二项 sum；近似 std = sqrt(6 * 6/33 * 27/33) ≈ 1.0
  const sigmaPerSample = Math.sqrt(6 * (6 / 33) * (27 / 33));
  const seMean = sigmaPerSample / Math.sqrt(n);
  const z = (lstmHit6 - uniHit6) / seMean;
  const pVal = approxNormalP(Math.abs(z));

  const verdict = pVal > 0.05
    ? `<strong>p = ${pVal.toFixed(3)} &gt; 0.05</strong>，与均匀随机 <strong>统计上不可区分</strong>。这正是预期：彩票是 i.i.d. 随机抽取，没有可学习的时间规律。`
    : `<strong>p = ${pVal.toFixed(3)} &lt; 0.05</strong>，差异显著。但要警惕：(1) 测试样本仅 ${n} 期，局部偏差可能；(2) 即便差异真实存在，也可能源于摇奖设备物理偏差，不构成可预测性。建议增大测试集复测。`;

  return `
    <div class="bt-table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th>方法</th>
            <th>红 Top-6 命中数</th>
            <th>红 Top-8 命中数</th>
            <th>蓝 Top-1 准确率</th>
            <th>Brier</th>
            <th>NLL</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>
    <div class="callout" style="margin-top:12px">
      <div class="callout-title">显著性检验</div>
      <div class="callout-body">
        n = ${n} 期。LSTM 红球 Top-6 期望命中 ${lstmHit6.toFixed(3)} vs 均匀基线 ${uniHit6.toFixed(3)}（z = ${z.toFixed(2)}）。<br/>
        ${verdict}
      </div>
    </div>
  `;
}

function approxNormalP(z) {
  // 双侧 p 值的近似（Abramowitz & Stegun 26.2.17）
  const t = 1 / (1 + 0.2316419 * z);
  const d = 0.39894228 * Math.exp(-0.5 * z * z);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return 2 * p;
}

/* ============================================================
 * 持久化
 * ============================================================ */
function onSave() {
  if (!state.model) return;
  try {
    const payload = {
      model: serializeModel(state.model),
      seqLen: state.seqLen,
      history: state.history,
      savedAt: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    toast("模型已保存到浏览器 localStorage");
  } catch (e) {
    toast(`保存失败：${e.message}（可能超出 localStorage 配额）`);
  }
}

function tryAutoLoadModel() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    state.model = deserializeModel(payload.model);
    state.seqLen = payload.seqLen || 15;
    state.history = payload.history || null;
    setStatus(`已自动加载保存的模型（${payload.savedAt?.slice(0, 19) || ""}），可直接预测`, "ok");
    if (state.history) renderFinalMetrics(state.history);
    $("#btnLstmPredict").disabled = false;
  } catch (e) {
    // 损坏的 payload 直接忽略
  }
}

function onLoad() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      toast("没有找到已保存的模型");
      return;
    }
    const payload = JSON.parse(raw);
    state.model = deserializeModel(payload.model);
    state.seqLen = payload.seqLen || 15;
    state.history = payload.history || null;
    if (state.history) renderFinalMetrics(state.history);
    $("#btnLstmPredict").disabled = false;
    $("#btnLstmBacktest").disabled = !state.valSamples;
    toast("已加载模型");
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
  const el = $("#lstmStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `lstm-status ${kind ? `is-${kind}` : ""}`;
}

function setProgress(ratio) {
  const bar = $("#lstmProgressBar");
  if (!bar) return;
  bar.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
}

function setControlsDuringTraining(training) {
  $("#btnLstmTrain").disabled = training;
  $("#btnLstmStop").disabled = !training;
  if (training) {
    $("#btnLstmPredict").disabled = true;
    $("#btnLstmBacktest").disabled = true;
    $("#btnLstmSave").disabled = true;
  }
}

function initCurves() {
  const el = $("#lstmCurves");
  if (!el) return;
  el.innerHTML = `
    <div class="curve-wrap" id="lstmLossCurve" data-label="Loss"></div>
    <div class="curve-wrap" id="lstmHitCurve" data-label="Red Top-6 Hit"></div>
  `;
}

const liveSeries = {
  trainLoss: [], valLoss: [], hit6: [], blueAcc: [],
};

function appendCurve(epochState) {
  liveSeries.trainLoss.push(epochState.trainLoss);
  liveSeries.valLoss.push(epochState.valLoss);
  liveSeries.hit6.push(epochState.valRedHit6);
  liveSeries.blueAcc.push(epochState.valBlueAcc);

  drawSpark("#lstmLossCurve", [
    { label: "train", series: liveSeries.trainLoss, color: "var(--blue)" },
    { label: "val", series: liveSeries.valLoss, color: "var(--red)" },
  ], "min");
  drawSpark("#lstmHitCurve", [
    { label: "val hit@6", series: liveSeries.hit6, color: "var(--acid)" },
  ], "max", { ref: 6 * 6 / 33, refLabel: "随机基线 1.09" });
}

function drawSpark(sel, series, opt, extra = {}) {
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
    ["验证 红球 Hit@6", `${history.valRedHit6[last]?.toFixed(3)}（基线 ${(6*6/33).toFixed(3)}）`],
    ["验证 蓝球 Top-1", `${(history.valBlueAcc[last] * 100).toFixed(2)}%（基线 6.25%）`],
    ["训练 epoch 数", String(history.epochs.length)],
  ];
  const el = $("#lstmMetrics");
  if (!el) return;
  el.innerHTML = items.map(([k, v]) => `
    <div class="diag-line"><span>${k}</span><strong class="mono">${v}</strong></div>
  `).join("");
}

function pause(ms = 0) {
  return new Promise((r) => setTimeout(r, ms));
}
