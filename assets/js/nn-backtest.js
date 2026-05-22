// Walk-forward backtest（行业标准）
//
// 思路：
//   把测试集划分为 K 个连续段；模型用「截至段开始」的所有数据训练，然后预测段内每一期。
//   这模拟"真实部署"——预测每一期时，模型只能看过去。简化版：固定训练集训练一次模型，
//   再在测试集上一次性 forward（窗口滑动）。
//
// 评估指标：
//   - 红球 Top-6 命中数：模型选 6 个最高概率红球，与真实 6 个红球求交集大小
//   - 蓝球 Top-1 准确率：argmax 是否等于真实蓝球
//   - 红球 Top-K 命中率分布：K=6/8/10/12 等
//   - Brier score（红球 multi-label 0/1 概率校准）
//   - Log-likelihood
//
// Baseline：
//   - 均匀随机：每期红球随机 6 个，蓝球均匀
//   - 频率：用过去 N 期频率排名 Top-6 红 + 频率最高蓝
//   - Beta 后验：用 bayes.js 的后验均值 Top-6
//
// 关键结论（理论）：
//   - 红球 Top-6 期望命中（任意预测策略）= 6 × 6/33 = 1.0909...
//   - 蓝球 Top-1 准确率（任意预测策略）= 1/16 = 0.0625
//   即使一个"好"模型，也无法系统性高于这两个 baseline——这就是这个 backtester 要客观证明的。

import {
  encodeSequence, encodeTarget,
  forwardModel, topKRed, argMaxBlue,
  RED_DIM, BLUE_DIM,
} from "./nn-ssq-model.js";
import { freqFromDraws } from "./stats.js";
import { posteriorMeanArray, RED_PRIOR, BLUE_PRIOR } from "./bayes.js";
import { createRng } from "./rng.js";

/**
 * 在 testDraws 上做 backtest（不更新模型权重）。
 * @param model 已训练的 LSTM 模型
 * @param trainTail 训练集最后 seqLen-1 期（用作初始上下文）
 * @param testDraws 测试集（按时间升序）
 * @param seqLen 序列长度
 */
export function backtestModel(model, trainTail, testDraws, seqLen) {
  // 维护 rolling 窗口
  let history = trainTail.slice(-seqLen);
  const records = [];
  for (const target of testDraws) {
    const window = history.slice(-seqLen);
    const seq = encodeSequence(window);
    const fwd = forwardModel(model, seq, { training: false });

    const top6Red = topKRed(fwd.redProbs, 6).map(([n]) => n);
    const realReds = target.reds;
    const redHit6 = top6Red.filter((n) => realReds.includes(n)).length;
    const top8 = topKRed(fwd.redProbs, 8).map(([n]) => n);
    const redHit8 = top8.filter((n) => realReds.includes(n)).length;

    const blueArg = argMaxBlue(fwd.blueProbs);
    const blueHit = blueArg.num === target.blue;

    // Brier 红球
    let brier = 0;
    for (let i = 0; i < RED_DIM; i++) {
      const p = fwd.redProbs.data[i];
      const y = realReds.includes(i + 1) ? 1 : 0;
      brier += (p - y) ** 2;
    }
    brier /= RED_DIM;

    // 红球 logloss
    let redLL = 0;
    for (let i = 0; i < RED_DIM; i++) {
      const p = Math.max(1e-12, Math.min(1 - 1e-12, fwd.redProbs.data[i]));
      const y = realReds.includes(i + 1) ? 1 : 0;
      redLL -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    }
    redLL /= RED_DIM;
    // 蓝球 logloss
    let blueLL = 0;
    for (let i = 0; i < BLUE_DIM; i++) {
      const p = Math.max(1e-12, fwd.blueProbs.data[i]);
      const y = (target.blue === i + 1) ? 1 : 0;
      blueLL -= y * Math.log(p);
    }

    records.push({
      issue: target.issue,
      realReds, realBlue: target.blue,
      predTop6: top6Red,
      predBlue: blueArg.num, predBlueProb: blueArg.prob,
      redHit6, redHit8,
      blueHit,
      brier, redLL, blueLL,
      redProbs: Array.from(fwd.redProbs.data),
      blueProbs: Array.from(fwd.blueProbs.data),
    });

    // 推进窗口
    history.push(target);
  }
  return summarize(records);
}

/** Baseline：每期独立按"过去全部数据的频率"做 top-K 选择。 */
export function backtestFreqBaseline(allDrawsBeforeTest, testDraws) {
  const records = [];
  let history = allDrawsBeforeTest.slice();
  for (const target of testDraws) {
    const fr = freqFromDraws(history, "reds", RED_DIM);
    const fb = freqFromDraws(history, "blue", BLUE_DIM);
    const redRanked = [];
    for (let i = 1; i <= RED_DIM; i++) redRanked.push([i, fr[i]]);
    redRanked.sort((a, b) => b[1] - a[1]);
    const top6 = redRanked.slice(0, 6).map(([n]) => n);
    let bestBlue = 1, bestBlueF = -1;
    for (let i = 1; i <= BLUE_DIM; i++) if (fb[i] > bestBlueF) { bestBlueF = fb[i]; bestBlue = i; }

    records.push(buildBaselineRecord(target, top6, bestBlue, redRanked.slice(0, 8).map(([n]) => n)));
    history.push(target);
  }
  return summarize(records);
}

/** Baseline：贝叶斯后验均值 top-K。 */
export function backtestBayesBaseline(allDrawsBeforeTest, testDraws) {
  const records = [];
  let history = allDrawsBeforeTest.slice();
  for (const target of testDraws) {
    const fr = freqFromDraws(history, "reds", RED_DIM);
    const fb = freqFromDraws(history, "blue", BLUE_DIM);
    const meanR = posteriorMeanArray(fr, history.length, RED_PRIOR);
    const meanB = posteriorMeanArray(fb, history.length, BLUE_PRIOR);
    const ranked = [];
    for (let i = 1; i <= RED_DIM; i++) ranked.push([i, meanR[i]]);
    ranked.sort((a, b) => b[1] - a[1]);
    const top6 = ranked.slice(0, 6).map(([n]) => n);
    let bestBlue = 1, bestBlueP = -1;
    for (let i = 1; i <= BLUE_DIM; i++) if (meanB[i] > bestBlueP) { bestBlueP = meanB[i]; bestBlue = i; }

    records.push(buildBaselineRecord(target, top6, bestBlue, ranked.slice(0, 8).map(([n]) => n)));
    history.push(target);
  }
  return summarize(records);
}

/** Baseline：均匀随机（蒙特卡洛 R 次取平均）。 */
export function backtestUniformBaseline(testDraws, runs = 50, seed = "uniform") {
  const records = [];
  // 每期跑 runs 次随机预测，求平均命中
  const rng = createRng(seed).next;
  for (const target of testDraws) {
    let acc = 0, accBlue = 0, acc8 = 0;
    for (let r = 0; r < runs; r++) {
      const reds = sampleRandomReds(rng);
      const blue = 1 + Math.floor(rng() * 16);
      acc += reds.filter((n) => target.reds.includes(n)).length;
      const reds8 = sampleRandomKReds(rng, 8);
      acc8 += reds8.filter((n) => target.reds.includes(n)).length;
      if (blue === target.blue) accBlue++;
    }
    records.push({
      issue: target.issue,
      realReds: target.reds, realBlue: target.blue,
      predTop6: [], predBlue: 0, predBlueProb: 1 / 16,
      redHit6: acc / runs, redHit8: acc8 / runs,
      blueHit: accBlue / runs,
      brier: 0, redLL: 0, blueLL: 0,
    });
  }
  return summarize(records);
}

function sampleRandomReds(rng) {
  return sampleRandomKReds(rng, 6);
}
function sampleRandomKReds(rng, k) {
  const pool = [];
  for (let i = 1; i <= RED_DIM; i++) pool.push(i);
  const out = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.floor(rng() * pool.length);
    out.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return out;
}

function buildBaselineRecord(target, top6, bestBlue, top8) {
  const redHit6 = top6.filter((n) => target.reds.includes(n)).length;
  const redHit8 = top8.filter((n) => target.reds.includes(n)).length;
  return {
    issue: target.issue,
    realReds: target.reds, realBlue: target.blue,
    predTop6: top6, predBlue: bestBlue, predBlueProb: 1 / 16,
    redHit6, redHit8,
    blueHit: bestBlue === target.blue,
    brier: 0, redLL: 0, blueLL: 0,
  };
}

function summarize(records) {
  const n = records.length;
  if (n === 0) return { records, summary: null };
  const sum = (k) => records.reduce((s, r) => s + (typeof r[k] === "boolean" ? (r[k] ? 1 : 0) : r[k]), 0);
  return {
    records,
    summary: {
      n,
      avgRedHit6: sum("redHit6") / n,
      avgRedHit8: sum("redHit8") / n,
      blueAccuracy: sum("blueHit") / n,
      avgBrier: sum("brier") / n,
      avgRedLL: sum("redLL") / n,
      avgBlueLL: sum("blueLL") / n,
    },
  };
}

/** 期望基线（理论值，随机预测器的渐近表现）。 */
export const RANDOM_BASELINE = {
  redHit6: 6 * 6 / 33,    // ≈ 1.0909
  redHit8: 6 * 8 / 33,    // ≈ 1.4545
  blueAcc: 1 / 16,        // 0.0625
};
