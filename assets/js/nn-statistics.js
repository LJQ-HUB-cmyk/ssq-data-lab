// Backtest 的统计严谨性辅助
//
// 核心：把"LSTM hit@6 = 1.18，均匀基线 = 1.09"这种点值差，
// 升级成"LSTM 95% CI = [0.95, 1.42]，与基线 1.09 重叠"——一个工程师能立刻读懂的判断。
//
// 1. Bootstrap CI：样本重采样 B 次，得到 metric 的经验分布
// 2. 配对 z 检验：两组在同一期上的差值
// 3. Reliability diagram：把预测概率分桶，看每桶里"真实命中率 vs 平均预测概率"
//    完美校准 → 点在对角线 y=x 上

import { createRng } from "./rng.js";

/**
 * Bootstrap 重采样：从 records 中有放回抽样 B 次，对每次重采样计算 metric，
 * 返回 B 个 metric 值的 (lower, upper) 分位数。
 */
export function bootstrapCI(records, metricFn, { B = 500, level = 0.95, seed = "bs" } = {}) {
  const n = records.length;
  if (n === 0) return { mean: 0, lower: 0, upper: 0, samples: [] };
  const rng = createRng(seed).next;
  const samples = new Array(B);
  for (let b = 0; b < B; b++) {
    const resample = new Array(n);
    for (let i = 0; i < n; i++) resample[i] = records[Math.floor(rng() * n)];
    samples[b] = metricFn(resample);
  }
  samples.sort((a, b) => a - b);
  const alpha = (1 - level) / 2;
  const loIdx = Math.max(0, Math.floor(alpha * B));
  const hiIdx = Math.min(B - 1, Math.ceil((1 - alpha) * B) - 1);
  const mean = samples.reduce((s, x) => s + x, 0) / B;
  return {
    mean,
    lower: samples[loIdx],
    upper: samples[hiIdx],
    samples,
  };
}

/** 给定两组配对样本（按相同期数），算配对差值的均值与 95% CI。 */
export function pairedBootstrap(recordsA, recordsB, metricFn, { B = 500, seed = "pbs" } = {}) {
  const n = Math.min(recordsA.length, recordsB.length);
  if (n === 0) return { mean: 0, lower: 0, upper: 0 };
  const rng = createRng(seed).next;
  const samples = new Array(B);
  for (let b = 0; b < B; b++) {
    const a = new Array(n), bb = new Array(n);
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng() * n);
      a[i] = recordsA[idx];
      bb[i] = recordsB[idx];
    }
    samples[b] = metricFn(a) - metricFn(bb);
  }
  samples.sort((x, y) => x - y);
  const lo = samples[Math.floor(0.025 * B)];
  const hi = samples[Math.ceil(0.975 * B) - 1];
  const mean = samples.reduce((s, x) => s + x, 0) / B;
  return { mean, lower: lo, upper: hi, samples };
}

/** Metric: average red Top-6 hit. */
export const metricAvgHit6 = (records) => {
  if (records.length === 0) return 0;
  let s = 0;
  for (const r of records) s += r.redHit6;
  return s / records.length;
};

/** Metric: blue accuracy. */
export const metricBlueAcc = (records) => {
  if (records.length === 0) return 0;
  let s = 0;
  for (const r of records) s += (typeof r.blueHit === "boolean" ? (r.blueHit ? 1 : 0) : r.blueHit);
  return s / records.length;
};

/**
 * Reliability diagram：把预测概率分到 [0, 1/N, 2/N, ...] 桶，
 * 每桶里计算 (avgPredProb, observedHitRate)。
 *
 * 完美校准的模型：avgPredProb ≈ observedHitRate。
 *
 * 输入：records，每条 record 必须含 redProbs[33] 与 realReds[6]
 *      （从 backtestModel 的输出来）
 *
 * 把全部 records × 33 个号码展开成 (prob, hit ∈ {0,1}) 大集合，再分桶。
 */
export function reliabilityDiagram(records, { bins = 10, redIndices = null } = {}) {
  const buckets = Array.from({ length: bins }, () => ({ sumP: 0, sumY: 0, count: 0 }));
  const idxList = redIndices || Array.from({ length: 33 }, (_, i) => i);
  for (const r of records) {
    if (!r.redProbs) continue;
    for (const i of idxList) {
      const p = r.redProbs[i];
      const y = r.realReds.includes(i + 1) ? 1 : 0;
      let b = Math.floor(p * bins);
      if (b >= bins) b = bins - 1;
      if (b < 0) b = 0;
      buckets[b].sumP += p;
      buckets[b].sumY += y;
      buckets[b].count += 1;
    }
  }
  const points = buckets.map((b, i) => ({
    binStart: i / bins,
    binEnd: (i + 1) / bins,
    avgPred: b.count > 0 ? b.sumP / b.count : (i + 0.5) / bins,
    observedFreq: b.count > 0 ? b.sumY / b.count : null,
    count: b.count,
  }));
  // Expected Calibration Error (ECE)
  const total = buckets.reduce((s, b) => s + b.count, 0);
  let ece = 0;
  for (const b of buckets) {
    if (b.count === 0) continue;
    const avgP = b.sumP / b.count;
    const obs = b.sumY / b.count;
    ece += (b.count / total) * Math.abs(avgP - obs);
  }
  return { points, ece };
}

/** 二项均匀基线对应的 hit@K：K * 6 / 33。 */
export function uniformBaselineHitK(K) {
  return K * 6 / 33;
}

/** 计算两组样本的 Brier score 差异 + paired bootstrap CI。 */
export const metricAvgBrier = (records) => {
  if (records.length === 0) return 0;
  let s = 0; let n = 0;
  for (const r of records) if (r.brier != null) { s += r.brier; n++; }
  return n === 0 ? 0 : s / n;
};
