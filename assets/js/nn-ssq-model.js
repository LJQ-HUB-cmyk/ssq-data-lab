// SSQ LSTM 预测模型
//
// 架构：
//   输入: 序列长度 T，每步特征 49 维
//        [redMultiHot(33) | blueOneHot(16)]
//
//   层:
//     LSTM(49 → H)     —— 单层 LSTM，H 通常 64~128
//     Dense Red Head:   h → 33 logits → sigmoid（每个红球独立 0/1，multi-label BCE）
//     Dense Blue Head:  h → 16 logits → softmax（蓝球唯一，cross-entropy）
//
// 损失 = BCE(redLogits, redTarget) + BCE_BLUE_WEIGHT * CE(blueLogits, blueTarget)
//   红球 multi-label 的 BCE 平均到 33 维；蓝球 CE 默认权重 6（与红球 6 个号对齐）。
//
// 推理：取最后一步 h_T，分别得到 red 概率向量和 blue 概率分布。
//   红球预测 = argTopK(red_probs, 6)
//   蓝球预测 = argMax(blue_probs)

import {
  makeMat, zero,
  matmul, transpose, add,
  sigmoid, sigmoidBCEBackward, bceLoss,
  softmax, softmaxCEBackward, crossEntropy,
  xavierInit, matmulAdd,
} from "./nn-math.js";
import {
  createLSTM, lstmForward, lstmBackward,
  serializeLSTM, deserializeLSTM,
} from "./nn-lstm.js";

export const RED_DIM = 33;
export const BLUE_DIM = 16;
export const FEATURE_DIM = RED_DIM + BLUE_DIM;
export const BLUE_LOSS_WEIGHT = 6;

/**
 * 把一期开奖编码为 49 维向量：
 *   [pos 0..32]: 红球 multi-hot；[pos 33..48]: 蓝球 one-hot
 */
export function encodeDraw(draw) {
  const v = new Float32Array(FEATURE_DIM);
  for (const r of draw.reds) v[r - 1] = 1;
  v[RED_DIM + draw.blue - 1] = 1;
  return { rows: FEATURE_DIM, cols: 1, data: v };
}

/** 把多期序列编码：返回 [matrix per step]。 */
export function encodeSequence(draws) {
  return draws.map(encodeDraw);
}

/** 取一期作为 target：red multi-hot (33×1) + blue one-hot (16×1) 单独返回。 */
export function encodeTarget(draw) {
  const red = makeMat(RED_DIM, 1);
  for (const r of draw.reds) red.data[r - 1] = 1;
  const blue = makeMat(BLUE_DIM, 1);
  blue.data[draw.blue - 1] = 1;
  return { red, blue };
}

/** 创建模型。 */
export function createModel({ hiddenDim = 64, rng = Math.random } = {}) {
  const lstm = createLSTM(FEATURE_DIM, hiddenDim, rng);
  const redHead = {
    W: xavierInit(RED_DIM, hiddenDim, rng),
    b: makeMat(RED_DIM, 1),
  };
  const blueHead = {
    W: xavierInit(BLUE_DIM, hiddenDim, rng),
    b: makeMat(BLUE_DIM, 1),
  };
  return {
    hiddenDim,
    lstm,
    redHead,
    blueHead,
  };
}

/**
 * 一次完整前向：输入 T 期序列，使用最后一步的 h_T 做预测。
 * 返回前向缓存供 BPTT。
 */
export function forwardModel(model, sequence) {
  const { hs, cs, caches, hLast } = lstmForward(model.lstm, sequence);
  const redLogits = add(matmul(model.redHead.W, hLast), model.redHead.b);
  const redProbs = sigmoid(redLogits);
  const blueLogits = add(matmul(model.blueHead.W, hLast), model.blueHead.b);
  const blueProbs = softmax(blueLogits);
  return {
    hs, cs, caches, hLast,
    redLogits, redProbs,
    blueLogits, blueProbs,
  };
}

/**
 * 计算 loss + 反向传播一次。返回 { loss, redLoss, blueLoss, grads }。
 *
 * grads 字典 keys:
 *   lstm.dW, lstm.dU, lstm.db,
 *   redHead.dW, redHead.db, blueHead.dW, blueHead.db
 */
export function lossAndGrads(model, sequence, target) {
  const fwd = forwardModel(model, sequence);
  const T = sequence.length;
  const H = model.hiddenDim;

  const redLoss = bceLoss(fwd.redProbs, target.red) / RED_DIM;
  const blueLoss = crossEntropy(fwd.blueProbs, target.blue);
  const totalLoss = redLoss + BLUE_LOSS_WEIGHT * blueLoss;

  // 输出层反向
  // dRedLogits = (sigmoid(redLogits) - redTarget) / RED_DIM
  const dRedLogits = sigmoidBCEBackward(fwd.redProbs, target.red);
  for (let i = 0; i < dRedLogits.data.length; i++) dRedLogits.data[i] /= RED_DIM;

  const dBlueLogits = softmaxCEBackward(fwd.blueProbs, target.blue);
  for (let i = 0; i < dBlueLogits.data.length; i++) dBlueLogits.data[i] *= BLUE_LOSS_WEIGHT;

  // dRedHead.W = dRedLogits · h_T^T
  const hLastT = transpose(fwd.hLast);
  const grads = {
    redHead: {
      dW: matmul(dRedLogits, hLastT),
      db: makeMat(RED_DIM, 1),
    },
    blueHead: {
      dW: matmul(dBlueLogits, hLastT),
      db: makeMat(BLUE_DIM, 1),
    },
  };
  for (let i = 0; i < dRedLogits.data.length; i++) grads.redHead.db.data[i] = dRedLogits.data[i];
  for (let i = 0; i < dBlueLogits.data.length; i++) grads.blueHead.db.data[i] = dBlueLogits.data[i];

  // 上传到 hLast：dh_T = redHead.W^T · dRedLogits + blueHead.W^T · dBlueLogits
  const dhFromRed = matmul(transpose(model.redHead.W), dRedLogits);
  const dhFromBlue = matmul(transpose(model.blueHead.W), dBlueLogits);
  const dhLast = add(dhFromRed, dhFromBlue);

  // 把 dhLast 放到 dhFromAbove[T-1]，其他时间步为 0（只在最后一步监督）
  const dhFromAbove = new Array(T);
  for (let t = 0; t < T - 1; t++) dhFromAbove[t] = makeMat(H, 1);
  dhFromAbove[T - 1] = dhLast;

  const { grads: lstmGrads } = lstmBackward(model.lstm, fwd.caches, dhFromAbove);
  grads.lstm = { dW: lstmGrads.dW, dU: lstmGrads.dU, db: lstmGrads.db };

  return { loss: totalLoss, redLoss, blueLoss, grads, fwd };
}

/** 把 grads / params 摊平为 Adam 期望的字典形式。 */
export function flattenParams(model) {
  return {
    "lstm.W": model.lstm.params.W,
    "lstm.U": model.lstm.params.U,
    "lstm.b": model.lstm.params.b,
    "redHead.W": model.redHead.W,
    "redHead.b": model.redHead.b,
    "blueHead.W": model.blueHead.W,
    "blueHead.b": model.blueHead.b,
  };
}

export function flattenGrads(g) {
  return {
    "lstm.W": g.lstm.dW,
    "lstm.U": g.lstm.dU,
    "lstm.b": g.lstm.db,
    "redHead.W": g.redHead.dW,
    "redHead.b": g.redHead.db,
    "blueHead.W": g.blueHead.dW,
    "blueHead.b": g.blueHead.db,
  };
}

/** 序列化（用于训练后保存到 localStorage 或下载）。 */
export function serializeModel(model) {
  const flat = (m) => ({ rows: m.rows, cols: m.cols, data: Array.from(m.data) });
  return {
    type: "ssq-lstm-v1",
    hiddenDim: model.hiddenDim,
    lstm: serializeLSTM(model.lstm),
    redHead: { W: flat(model.redHead.W), b: flat(model.redHead.b) },
    blueHead: { W: flat(model.blueHead.W), b: flat(model.blueHead.b) },
  };
}

export function deserializeModel(obj) {
  const inflate = (m) => ({ rows: m.rows, cols: m.cols, data: new Float32Array(m.data) });
  return {
    hiddenDim: obj.hiddenDim,
    lstm: deserializeLSTM(obj.lstm),
    redHead: { W: inflate(obj.redHead.W), b: inflate(obj.redHead.b) },
    blueHead: { W: inflate(obj.blueHead.W), b: inflate(obj.blueHead.b) },
  };
}

/** 给定 redProbs (33×1)，返回前 K 个号码（1-indexed）。 */
export function topKRed(redProbs, k = 6) {
  const arr = [];
  for (let i = 0; i < RED_DIM; i++) arr.push([i + 1, redProbs.data[i]]);
  arr.sort((a, b) => b[1] - a[1]);
  return arr.slice(0, k);
}

/** 给定 blueProbs (16×1)，返回最高概率号码（1-indexed）。 */
export function argMaxBlue(blueProbs) {
  let best = 1, bestP = blueProbs.data[0];
  for (let i = 1; i < BLUE_DIM; i++) {
    if (blueProbs.data[i] > bestP) {
      bestP = blueProbs.data[i];
      best = i + 1;
    }
  }
  return { num: best, prob: bestP };
}
