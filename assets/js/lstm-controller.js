// LSTM 面板的 UI 控制器
//
// 职责：从 #panel-lstm 表单读参数 → 调用 trainer/backtest/ensemble → 渲染回页面

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
import {
  bootstrapCI, pairedBootstrap,
  metricAvgHit6, metricBlueAcc,
  reliabilityDiagram,
} from "./nn-statistics.js";
import { trainEnsemble, ensembleForward } from "./nn-ensemble.js";
import { createRng } from "./rng.js";

const STORAGE_KEY = "ssq-lstm-model-v2";

const state = {
  draws: [],
  model: null,        // 单模型
  ensemble: null,     // K 个模型 [{model}, ...]
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
    const numLayers = clampInt("#lstmLayers", 1, 4, 2);
    const split = clampNum("#lstmSplit", 0.5, 0.95, 0.85);
    const lr = clampNum("#lstmLr", 1e-4, 0.1, 0.003);
    const epochs = clampInt("#lstmEpochs", 1, 100, 20);
    const batchSize = clampInt("#lstmBatch", 4, 128, 32);
    const dropoutInput = clampNum("#lstmDropIn", 0, 0.5, 0.1);
    const dropoutHidden = clampNum("#lstmDropHidden", 0, 0.5, 0.2);
    const dropoutOutput = clampNum("#lstmDropOut", 0, 0.5, 0.2);
    const ensembleK = clampInt("#lstmEnsembleK", 1, 8, 1);
    const seedStr = $("#lstmSeed")?.value?.trim() || `train-${Date.now()}`;
    state.seqLen = seqLen;

    setStatus("准备样本…");
    const samples = buildSamples(state.draws, seqLen);
    if (samples.length < 100) throw new Error(`数据太少，至少需要 ${100 + seqLen} 期`);
    const splitIdx = Math.floor(samples.length * split);
    state.trainSamples = samples.slice(0, splitIdx);
    state.valSamples = samples.slice(splitIdx);

    const arch = `H=${hidden} · L=${numLayers} · drop[in/h/out]=${dropoutInput}/${dropoutHidden}/${dropoutOutput} · K=${ensembleK}`;
    setStatus(`训练：${state.trainSamples.length} train / ${state.valSamples.length} val · T=${seqLen} · ${arch}`);
    state.history = null;
    state.ensemble = null;
    state.model = null;
    initCurves();
    resetLiveSeries();

    const t0 = Date.now();
    const baseModelOpts = {
      hiddenDim: hidden, numLayers,
      dropoutInput, dropoutHidden, dropoutOutput,
    };
    const baseTrainOpts = {
      epochs, batchSize, lr,
      gradClip: 5,
      patience: 6,
      weightDecay: 1e-5,
    };

    if (ensembleK > 1) {
      const result = await trainEnsemble(state.trainSamples, state.valSamples, {
        K: ensembleK,
        seedBase: seedStr,
        modelOpts: baseModelOpts,
        trainOpts: {
          ...baseTrainOpts,
          onBatch: (b) => {
            if (b.totalBatches) setProgress((b.member / b.totalMembers) + (b.batch / b.totalBatches / b.totalMembers));
          },
          onEpoch: (e) => {
            appendCurve(e, e.member);
            setStatus(`成员 ${e.member + 1}/${e.totalMembers} · epoch ${e.epoch + 1}/${e.totalEpochs} · train ${e.trainLoss.toFixed(4)} · val ${e.valLoss.toFixed(4)} · 红 hit@6 ${e.valRedHit6.toFixed(3)} · 蓝 ${(e.valBlueAcc * 100).toFixed(1)}%`);
          },
        },
        shouldStop: () => state.shouldStop,
      });
      state.ensemble = { members: result.members, histories: result.histories };
      state.history = result.histories[result.histories.length - 1];
    } else {
      const memberRng = createRng(seedStr).next;
      state.model = createModel({ ...baseModelOpts, rng: memberRng });
      const result = await trainModel(state.model, state.trainSamples, state.valSamples, {
        ...baseTrainOpts,
        rng: memberRng,
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
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const bestLoss = state.history ? Math.min(...state.history.valLoss).toFixed(4) : "—";
    setStatus(`训练完成 · ${elapsed}s · best val ${bestLoss}${state.ensemble ? ` · ${ensembleK} 模型集成` : ""}`, "ok");
    setProgress(1);

    if (state.history) renderFinalMetrics(state.history);

    $("#btnLstmPredict").disabled = false;
    $("#btnLstmBacktest").disabled = false;
    $("#btnLstmSave").disabled = !state.model; // ensemble 暂不支持保存
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
  if (!state.model && !state.ensemble) return;
  const window = state.draws.slice(-state.seqLen);
  const seq = encodeSequence(window);

  let redProbs, blueProbs, redStd = null, blueStd = null;
  if (state.ensemble) {
    const out = ensembleForward(state.ensemble.members, seq);
    redProbs = out.redProbs;
    blueProbs = out.blueProbs;
    redStd = out.redStd;
    blueStd = out.blueStd;
  } else {
    const fwd = forwardModel(state.model, seq, { training: false });
    redProbs = fwd.redProbs;
    blueProbs = fwd.blueProbs;
  }

  const top6 = topKRed(redProbs, 6);
  const blueArg = argMaxBlue(blueProbs);
  const blueRanked = [];
  for (let i = 0; i < BLUE_DIM; i++) blueRanked.push([i + 1, blueProbs.data[i]]);
  blueRanked.sort((a, b) => b[1] - a[1]);

  const card = $("#lstmPredictionCard");
  const body = $("#lstmPredictionBody");
  card.style.display = "";

  // 红球 33 个号码概率热度条
  const redBars = [];
  let redMax = 0;
  for (let i = 0; i < RED_DIM; i++) redMax = Math.max(redMax, redProbs.data[i]);
  for (let i = 0; i < RED_DIM; i++) {
    const p = redProbs.data[i];
    const w = (p / Math.max(1e-9, redMax)) * 100;
    const isPicked = top6.some(([n]) => n === i + 1);
    const stdInfo = redStd
      ? ` <span class="muted fine">±${(redStd.data[i] * 100).toFixed(1)}%</span>`
      : "";
    redBars.push(`
      <div class="prob-row">
        <span class="ball red ${isPicked ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(i + 1)}</span>
        <span class="prob-bar"><i style="width:${w.toFixed(1)}%"></i></span>
        <span class="mono prob-val">${(p * 100).toFixed(1)}%${stdInfo}</span>
      </div>
    `);
  }
  // 蓝球
  const blueBars = blueRanked.map(([n, p]) => {
    const stdInfo = blueStd
      ? ` <span class="muted fine">±${(blueStd.data[n - 1] * 100).toFixed(1)}%</span>`
      : "";
    return `
      <div class="prob-row">
        <span class="ball blue ${n === blueArg.num ? "" : "muted-ball"}" style="width:24px;height:24px;font-size:10px;box-shadow:none">${pad2(n)}</span>
        <span class="prob-bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span>
        <span class="mono prob-val">${(p * 100).toFixed(1)}%${stdInfo}</span>
      </div>
    `;
  }).join("");

  const ensembleBadge = state.ensemble
    ? `<span class="chip chip-ok" style="margin-left:8px">${state.ensemble.members.length} 模型集成</span>`
    : "";

  body.innerHTML = `
    <div class="prediction-pick">
      <div class="prediction-label">Top-6 红 + 蓝${ensembleBadge}</div>
      <div class="balls">
        ${top6.map(([n]) => `<span class="ball red">${pad2(n)}</span>`).join("")}
        <span class="ball blue plus">${pad2(blueArg.num)}</span>
      </div>
    </div>
    <div class="prediction-cols">
      <div>
        <div class="card-title">红球 33 路概率${redStd ? "（± 集成 std）" : ""}</div>
        ${redBars.join("")}
      </div>
      <div>
        <div class="card-title">蓝球 16 路概率${blueStd ? "（± 集成 std）" : ""}</div>
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
  const sourceModel = state.model || (state.ensemble ? state.ensemble.members[0] : null);
  if (!sourceModel) return;
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

    // LSTM（单模型 or ensemble 第一个）；如果有 ensemble，再额外算一个 ensemble backtest
    const lstmRes = backtestModel(sourceModel, trainTail, testDraws, seqLen);
    let ensembleRes = null;
    if (state.ensemble && state.ensemble.members.length > 1) {
      ensembleRes = backtestEnsemble(state.ensemble.members, trainTail, testDraws, seqLen);
    }

    const freqRes = backtestFreqBaseline(state.draws.slice(0, splitIdx), testDraws);
    const bayesRes = backtestBayesBaseline(state.draws.slice(0, splitIdx), testDraws);
    const uniformRes = backtestUniformBaseline(testDraws, 100, "uniform-baseline");

    const card = $("#lstmBacktestCard");
    const body = $("#lstmBacktestBody");
    card.style.display = "";
    body.innerHTML = renderBacktestTable(lstmRes, ensembleRes, freqRes, bayesRes, uniformRes, testDraws.length);
    setStatus(`回测完成：${testDraws.length} 期`, "ok");
  } catch (err) {
    setStatus(`回测失败：${err.message || err}`, "bad");
    console.error(err);
  }
}

function backtestEnsemble(members, trainTail, testDraws, seqLen) {
  let history = trainTail.slice(-seqLen);
  const records = [];
  for (const target of testDraws) {
    const window = history.slice(-seqLen);
    const seq = encodeSequence(window);
    const out = ensembleForward(members, seq);
    const top6 = topKRed(out.redProbs, 6).map(([n]) => n);
    const top8 = topKRed(out.redProbs, 8).map(([n]) => n);
    const blueArg = argMaxBlue(out.blueProbs);
    const redHit6 = top6.filter((n) => target.reds.includes(n)).length;
    const redHit8 = top8.filter((n) => target.reds.includes(n)).length;
    let brier = 0;
    for (let i = 0; i < RED_DIM; i++) {
      const p = out.redProbs.data[i];
      const y = target.reds.includes(i + 1) ? 1 : 0;
      brier += (p - y) ** 2;
    }
    brier /= RED_DIM;
    let redLL = 0;
    for (let i = 0; i < RED_DIM; i++) {
      const p = Math.max(1e-12, Math.min(1 - 1e-12, out.redProbs.data[i]));
      const y = target.reds.includes(i + 1) ? 1 : 0;
      redLL -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    }
    redLL /= RED_DIM;
    let blueLL = 0;
    for (let i = 0; i < BLUE_DIM; i++) {
      const p = Math.max(1e-12, out.blueProbs.data[i]);
      const y = (target.blue === i + 1) ? 1 : 0;
      blueLL -= y * Math.log(p);
    }
    records.push({
      issue: target.issue, realReds: target.reds, realBlue: target.blue,
      predTop6: top6, predBlue: blueArg.num, predBlueProb: blueArg.prob,
      redHit6, redHit8,
      blueHit: blueArg.num === target.blue,
      brier, redLL, blueLL,
      redProbs: Array.from(out.redProbs.data),
      blueProbs: Array.from(out.blueProbs.data),
    });
    history.push(target);
  }
  const summary = {
    n: records.length,
    avgRedHit6: records.reduce((s, r) => s + r.redHit6, 0) / records.length,
    avgRedHit8: records.reduce((s, r) => s + r.redHit8, 0) / records.length,
    blueAccuracy: records.reduce((s, r) => s + (r.blueHit ? 1 : 0), 0) / records.length,
    avgBrier: records.reduce((s, r) => s + r.brier, 0) / records.length,
    avgRedLL: records.reduce((s, r) => s + r.redLL, 0) / records.length,
    avgBlueLL: records.reduce((s, r) => s + r.blueLL, 0) / records.length,
  };
  return { records, summary };
}

function renderBacktestTable(lstm, ensemble, freq, bayes, uniform, n) {
  const fmt = (v, d = 4) => (v == null ? "—" : v.toFixed(d));
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(2)}%`);

  // 关键策略的 95% bootstrap CI
  const lstmCi = bootstrapCI(lstm.records, metricAvgHit6, { B: 500, seed: "bt-lstm" });
  const lstmBlueCi = bootstrapCI(lstm.records, metricBlueAcc, { B: 500, seed: "bt-lstm-b" });
  const uniCi = bootstrapCI(uniform.records, metricAvgHit6, { B: 500, seed: "bt-uni" });

  // Paired bootstrap：LSTM vs Uniform 红球差异
  const paired = pairedBootstrap(lstm.records, uniform.records, metricAvgHit6, { B: 500, seed: "bt-paired" });

  const formatCI = (mean, lo, hi) => `${mean.toFixed(3)} <span class="muted">[${lo.toFixed(3)}, ${hi.toFixed(3)}]</span>`;
  const formatPctCI = (mean, lo, hi) => `${(mean*100).toFixed(2)}% <span class="muted">[${(lo*100).toFixed(2)}, ${(hi*100).toFixed(2)}]</span>`;

  const rows = [];
  rows.push({
    label: "LSTM（单模型）",
    tag: "primary",
    redHit6Cell: formatCI(lstm.summary.avgRedHit6, lstmCi.lower, lstmCi.upper),
    redHit8: lstm.summary.avgRedHit8,
    blueAccCell: formatPctCI(lstm.summary.blueAccuracy, lstmBlueCi.lower, lstmBlueCi.upper),
    brier: lstm.summary.avgBrier,
    ll: lstm.summary.avgRedLL + lstm.summary.avgBlueLL,
  });
  if (ensemble) {
    const ensCi = bootstrapCI(ensemble.records, metricAvgHit6, { B: 500, seed: "bt-ens" });
    const ensBlueCi = bootstrapCI(ensemble.records, metricBlueAcc, { B: 500, seed: "bt-ens-b" });
    rows.push({
      label: `LSTM Ensemble (K=${state.ensemble.members.length})`,
      tag: "primary",
      redHit6Cell: formatCI(ensemble.summary.avgRedHit6, ensCi.lower, ensCi.upper),
      redHit8: ensemble.summary.avgRedHit8,
      blueAccCell: formatPctCI(ensemble.summary.blueAccuracy, ensBlueCi.lower, ensBlueCi.upper),
      brier: ensemble.summary.avgBrier,
      ll: ensemble.summary.avgRedLL + ensemble.summary.avgBlueLL,
    });
  }
  rows.push({
    label: "贝叶斯后验 baseline",
    redHit6Cell: bayes.summary.avgRedHit6.toFixed(3),
    redHit8: bayes.summary.avgRedHit8,
    blueAccCell: pct(bayes.summary.blueAccuracy),
  });
  rows.push({
    label: "频率 baseline",
    redHit6Cell: freq.summary.avgRedHit6.toFixed(3),
    redHit8: freq.summary.avgRedHit8,
    blueAccCell: pct(freq.summary.blueAccuracy),
  });
  rows.push({
    label: "均匀随机 baseline (100×MC)",
    redHit6Cell: formatCI(uniform.summary.avgRedHit6, uniCi.lower, uniCi.upper),
    redHit8: uniform.summary.avgRedHit8,
    blueAccCell: pct(uniform.summary.blueAccuracy),
  });
  rows.push({
    label: "理论期望（任意预测器渐近）",
    tag: "theory",
    redHit6Cell: RANDOM_BASELINE.redHit6.toFixed(3),
    redHit8: RANDOM_BASELINE.redHit8,
    blueAccCell: pct(RANDOM_BASELINE.blueAcc),
  });

  const tableRows = rows.map((r) => `
    <tr class="${r.tag === "primary" ? "row-primary" : ""}${r.tag === "theory" ? " row-theory" : ""}">
      <td>${r.label}</td>
      <td class="mono">${r.redHit6Cell}</td>
      <td class="mono">${typeof r.redHit8 === "number" ? r.redHit8.toFixed(3) : (r.redHit8 ?? "—")}</td>
      <td class="mono">${r.blueAccCell}</td>
      <td class="mono">${r.brier != null ? fmt(r.brier) : "—"}</td>
      <td class="mono">${r.ll != null ? fmt(r.ll, 3) : "—"}</td>
    </tr>
  `).join("");

  // 主结论：paired bootstrap 95% CI 是否包含 0
  const ciIncludesZero = paired.lower <= 0 && paired.upper >= 0;
  const verdict = ciIncludesZero
    ? `LSTM 与均匀随机的 hit@6 差值 95% CI = [${paired.lower.toFixed(3)}, ${paired.upper.toFixed(3)}] <strong>包含 0</strong>，差异在统计上不显著——这正是预期：彩票是 i.i.d. 随机抽取，没有可学习的时间规律。`
    : `LSTM 与均匀随机的 hit@6 差值 95% CI = [${paired.lower.toFixed(3)}, ${paired.upper.toFixed(3)}] <strong>不含 0</strong>。但要警惕：(1) ${n} 期的局部偏差可能是数据集偏差，不一定泛化；(2) 即便差异真实，也可能源于硬件物理偏差或 multi-seed 选最优结果，不构成可重复的可预测性。`;

  // Reliability diagram for LSTM
  const reliab = reliabilityDiagram(lstm.records, { bins: 10 });
  const reliabSvg = renderReliabilityDiagram(reliab);

  return `
    <div class="bt-table-wrap">
      <table class="table">
        <thead>
          <tr>
            <th>方法</th>
            <th>红 hit@6 [95% CI]</th>
            <th>红 hit@8</th>
            <th>蓝 Top-1 准确率 [95% CI]</th>
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
      ${reliabSvg}
      <div class="hint">点越接近对角线 y=x 越好——表示 "概率 20% 的号码大约真有 20% 的命中率"。完美校准 ECE = 0。</div>
    </div>
  `;
}

function renderReliabilityDiagram(reliab) {
  const W = 380, H = 220;
  const padL = 36, padR = 12, padT = 12, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const dots = reliab.points.map((p) => {
    if (p.observedFreq == null) return "";
    const cx = padL + p.avgPred * innerW;
    const cy = padT + (1 - p.observedFreq) * innerH;
    const r = 2 + Math.min(8, Math.sqrt(p.count) * 0.5);
    return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" fill="var(--blue)" opacity="0.78"/>`;
  }).join("");
  const refLine = `<line x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT}" stroke="rgba(255,255,255,.35)" stroke-dasharray="3 4"/>`;
  const xAxis = `<line x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const yAxis = `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const xLabel = `<text x="${padL + innerW / 2}" y="${H - 6}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="rgba(255,255,255,.6)">预测概率（avg per bucket）</text>`;
  const yLabel = `<text x="${padL - 28}" y="${padT + innerH / 2}" text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" fill="rgba(255,255,255,.6)" transform="rotate(-90 ${padL - 28} ${padT + innerH / 2})">观察到的命中率</text>`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) => {
    const x = padL + v * innerW;
    const y = padT + (1 - v) * innerH;
    return `
      <text x="${x}" y="${padT + innerH + 14}" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.5)" font-family="JetBrains Mono, monospace">${v.toFixed(2)}</text>
      <text x="${padL - 4}" y="${y + 3}" text-anchor="end" font-size="9" fill="rgba(255,255,255,.5)" font-family="JetBrains Mono, monospace">${v.toFixed(2)}</text>
    `;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
    ${xAxis}${yAxis}${refLine}${ticks}${xLabel}${yLabel}${dots}
  </svg>`;
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

function resetLiveSeries() {
  liveSeries.trainLoss = [];
  liveSeries.valLoss = [];
  liveSeries.hit6 = [];
  liveSeries.blueAcc = [];
}

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
