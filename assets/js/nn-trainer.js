// SSQ LSTM 训练器
//
// 训练流程：
//   1. 从历史 N 期数据，构造 (sequence, target) 样本：用 [t-T, t-1] 预测 t
//      训练集占前 splitRatio 比例，验证集占后 (1-splitRatio)
//   2. 每个 epoch 把训练集随机打乱，分 mini-batch，对每批样本累加梯度后做 Adam 更新
//   3. 每个 epoch 在验证集上评估 loss + 命中率
//   4. 早停：若 patience 个 epoch 验证 loss 未改善则停止
//
// 全部在主线程同步运行；训练时 yield 回 UI（每个 batch 后 await 一个 microtask 让浏览器更新进度）

import {
  encodeSequence, encodeTarget,
  lossAndGrads, flattenParams, flattenGrads,
  RED_DIM, BLUE_DIM,
  forwardModel, topKRed, argMaxBlue,
} from "./nn-ssq-model.js";
import { clipGradGlobal, makeMat, hasNaN } from "./nn-math.js";
import { createAdam } from "./nn-optim.js";

/** 从历史 draws 构造样本数组。 */
export function buildSamples(draws, seqLen) {
  const samples = [];
  for (let t = seqLen; t < draws.length; t++) {
    const window = draws.slice(t - seqLen, t);
    const target = draws[t];
    samples.push({
      issue: target.issue,
      sequence: encodeSequence(window),
      target: encodeTarget(target),
      raw: { window, target },
    });
  }
  return samples;
}

export function shuffleInPlace(arr, rng = Math.random) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/** 红球 hit count + 蓝球准确率 + 头部 K=6 命中数等指标。 */
export function evaluate(model, samples) {
  let totalLoss = 0;
  let totalRed = 0;
  let totalBlue = 0;
  let totalRedHit6 = 0;
  let totalBlueHit = 0;
  let n = 0;
  for (const s of samples) {
    // 推理模式：禁用 dropout
    const fwd = forwardModel(model, s.sequence, { training: false });
    // 复用 lossAndGrads 计算 loss 略重；这里只算 loss
    let redLoss = 0;
    for (let i = 0; i < RED_DIM; i++) {
      const p = Math.max(1e-12, Math.min(1 - 1e-12, fwd.redProbs.data[i]));
      const t = s.target.red.data[i];
      redLoss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
    }
    redLoss /= RED_DIM;
    let blueLoss = 0;
    for (let i = 0; i < BLUE_DIM; i++) {
      if (s.target.blue.data[i] > 0) {
        blueLoss -= s.target.blue.data[i] * Math.log(Math.max(1e-12, fwd.blueProbs.data[i]));
      }
    }
    totalLoss += redLoss + 6 * blueLoss;
    totalRed += redLoss;
    totalBlue += blueLoss;

    // 命中数：top-6 红球与真实 6 个红球的重合
    const top6 = topKRed(fwd.redProbs, 6).map(([n]) => n);
    const realReds = [];
    for (let i = 0; i < RED_DIM; i++) if (s.target.red.data[i] > 0) realReds.push(i + 1);
    let hit = 0;
    for (const x of top6) if (realReds.includes(x)) hit++;
    totalRedHit6 += hit;

    const blueArg = argMaxBlue(fwd.blueProbs);
    let realBlue = 1;
    for (let i = 0; i < BLUE_DIM; i++) if (s.target.blue.data[i] > 0) realBlue = i + 1;
    if (blueArg.num === realBlue) totalBlueHit++;

    n++;
  }
  return {
    avgLoss: totalLoss / Math.max(1, n),
    avgRedLoss: totalRed / Math.max(1, n),
    avgBlueLoss: totalBlue / Math.max(1, n),
    avgRedHit6: totalRedHit6 / Math.max(1, n),  // 期望均匀随机 ≈ 6×6/33 = 1.09
    blueAccuracy: totalBlueHit / Math.max(1, n), // 期望均匀随机 = 1/16 = 0.0625
    samples: n,
  };
}

/**
 * 训练主循环。
 * @param model
 * @param trainSamples / valSamples
 * @param opts {
 *   epochs, batchSize, lr, lrDecay, weightDecay,
 *   gradClip, patience, rng,
 *   onEpoch(state)  // state 含每 epoch 的 loss、命中率
 *   onBatch(state)  // state 含 epoch、batch、avgLoss
 *   shouldStop()    // 用户可中断
 * }
 */
export async function trainModel(model, trainSamples, valSamples, opts = {}) {
  const {
    epochs = 30,
    batchSize = 32,
    lr = 1e-3,
    lrDecay = 0,
    weightDecay = 1e-5,
    gradClip = 5,
    patience = 5,
    rng = Math.random,
    onEpoch,
    onBatch,
    shouldStop,
  } = opts;

  const params = flattenParams(model);
  const adam = createAdam(params, { lr, weightDecay });
  const history = {
    epochs: [],
    lr: [],
    trainLoss: [],
    valLoss: [],
    valRedHit6: [],
    valBlueAcc: [],
    gradNorms: [],
  };

  let bestValLoss = Infinity;
  let bestParams = null;
  let stale = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (shouldStop && shouldStop()) break;
    const epochLr = lr * Math.pow(1 - lrDecay, epoch);
    adam.setLr(epochLr);

    // shuffle
    const order = trainSamples.slice();
    shuffleInPlace(order, rng);

    let epochLoss = 0;
    let epochCount = 0;
    let runningGradNorm = 0;
    let batchIdx = 0;

    for (let i = 0; i < order.length; i += batchSize) {
      if (shouldStop && shouldStop()) break;
      const batch = order.slice(i, i + batchSize);

      // 累加梯度（mini-batch）
      const accum = makeAccumGrads(model);
      let batchLoss = 0;
      for (const sample of batch) {
        const { loss, grads } = lossAndGrads(model, sample.sequence, sample.target, { rng });
        batchLoss += loss;
        addInto(accum, flattenGrads(grads));
      }
      // 求平均
      scaleGrads(accum, 1 / batch.length);
      const norm = clipGradGlobal(Object.values(accum), gradClip);
      runningGradNorm = 0.9 * runningGradNorm + 0.1 * norm;

      // NaN 守门
      let nanFound = false;
      for (const v of Object.values(accum)) if (hasNaN(v)) { nanFound = true; break; }
      if (nanFound) {
        // 跳过这个 batch（不更新参数）
        if (onBatch) onBatch({ epoch, batch: batchIdx, loss: batchLoss / batch.length, gradNorm: norm, nan: true });
        batchIdx++;
        continue;
      }

      adam.step(accum);
      epochLoss += batchLoss;
      epochCount += batch.length;
      batchIdx++;

      if (onBatch && batchIdx % 4 === 0) {
        onBatch({
          epoch,
          batch: batchIdx,
          totalBatches: Math.ceil(order.length / batchSize),
          loss: batchLoss / batch.length,
          gradNorm: norm,
        });
        await pause();
      }
    }

    // 验证
    const valStats = evaluate(model, valSamples);
    const trainAvg = epochLoss / Math.max(1, epochCount);

    history.epochs.push(epoch);
    history.lr.push(epochLr);
    history.trainLoss.push(trainAvg);
    history.valLoss.push(valStats.avgLoss);
    history.valRedHit6.push(valStats.avgRedHit6);
    history.valBlueAcc.push(valStats.blueAccuracy);
    history.gradNorms.push(runningGradNorm);

    if (onEpoch) {
      onEpoch({
        epoch, totalEpochs: epochs,
        lr: epochLr,
        trainLoss: trainAvg,
        valLoss: valStats.avgLoss,
        valRedHit6: valStats.avgRedHit6,
        valBlueAcc: valStats.blueAccuracy,
        gradNorm: runningGradNorm,
      });
      await pause();
    }

    if (valStats.avgLoss + 1e-6 < bestValLoss) {
      bestValLoss = valStats.avgLoss;
      bestParams = snapshotParams(model);
      stale = 0;
    } else {
      stale++;
      if (stale >= patience) break;
    }
  }

  // 还原到最佳
  if (bestParams) restoreParams(model, bestParams);
  return { model, history, bestValLoss };
}

function makeAccumGrads(model) {
  const params = flattenParams(model);
  const out = {};
  for (const k of Object.keys(params)) out[k] = makeMat(params[k].rows, params[k].cols);
  return out;
}

function addInto(accum, grads) {
  for (const k of Object.keys(grads)) {
    const a = accum[k].data;
    const b = grads[k].data;
    for (let i = 0; i < a.length; i++) a[i] += b[i];
  }
}

function scaleGrads(grads, s) {
  for (const k of Object.keys(grads)) {
    const d = grads[k].data;
    for (let i = 0; i < d.length; i++) d[i] *= s;
  }
}

function snapshotParams(model) {
  const p = flattenParams(model);
  const snap = {};
  for (const k of Object.keys(p)) snap[k] = new Float32Array(p[k].data);
  return snap;
}

function restoreParams(model, snap) {
  const p = flattenParams(model);
  for (const k of Object.keys(snap)) p[k].data.set(snap[k]);
}

function pause() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
