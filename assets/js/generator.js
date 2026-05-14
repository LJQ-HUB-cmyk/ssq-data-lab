import { passesConstraints, analyseConstraintFailures } from "./stats.js";

export function makeWeightsFromFreq(freq, strategy, alpha) {
  const size = freq.length - 1;
  const maxF = Math.max(...freq.slice(1));
  const w = [];
  for (let n = 1; n <= size; n++) {
    const f = freq[n];
    let base = 1;
    if (strategy === "hot") base = f + 1;
    else if (strategy === "cold") base = maxF - f + 1;
    w.push(Math.pow(base, alpha));
  }
  return w;
}

// 真正的"混合"：几何平均。hot 极端高 / cold 极端高的号码都会被抑制，
// 让中等频次的号码有机会被采到，更符合"混合"直觉。
export function makeMixedWeights(freq, alpha) {
  const wHot = makeWeightsFromFreq(freq, "hot", alpha);
  const wCold = makeWeightsFromFreq(freq, "cold", alpha);
  return wHot.map((h, i) => Math.sqrt(h * wCold[i]));
}

export function weightedPickOne(items, weights, rand = Math.random) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) return items[Math.floor(rand() * items.length)];
  let r = rand() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

export function weightedSampleWithoutReplacement(items, weights, k, rand = Math.random) {
  if (k > items.length) throw new Error(`cannot sample ${k} from ${items.length}`);
  const pool = items.slice();
  const pw = weights.slice();
  const out = [];
  for (let i = 0; i < k; i++) {
    const picked = weightedPickOne(pool, pw, rand);
    out.push(picked);
    const idx = pool.indexOf(picked);
    pool.splice(idx, 1);
    pw.splice(idx, 1);
  }
  return out;
}

function buildRedPool({ include = [], exclude = [], avoidLast = [] }) {
  const includeSet = new Set(include);
  const excludeSet = new Set([...exclude, ...avoidLast]);
  // include 与 exclude 冲突时以 include 优先
  for (const n of includeSet) excludeSet.delete(n);
  const pool = [];
  for (let n = 1; n <= 33; n++) {
    if (includeSet.has(n)) continue;
    if (excludeSet.has(n)) continue;
    pool.push(n);
  }
  return { pool, includeList: [...includeSet].sort((a, b) => a - b) };
}

function subsetWeights(sourceWeights, pool) {
  // sourceWeights[0..32] 对应号码 1..33；pool 是实际候选号码列表
  return pool.map((n) => sourceWeights[n - 1]);
}

export function generateTickets({
  freqR,
  freqB,
  strategyRed,
  strategyBlue,
  alpha,
  constraints,
  count,
  includeRed = [],
  excludeRed = [],
  avoidLast = [],
  excludeBlue = [],
  maxTry = 2000,
  rand = Math.random,
}) {
  const { pool, includeList } = buildRedPool({ include: includeRed, exclude: excludeRed, avoidLast });
  if (includeList.length > 6) {
    throw new Error(`胆码不能超过 6 个（当前 ${includeList.length}）`);
  }
  const needToPick = 6 - includeList.length;
  if (pool.length < needToPick) {
    throw new Error(`排除过多：还需抽 ${needToPick} 个但只剩 ${pool.length} 个候选`);
  }

  const blueItems = [];
  for (let n = 1; n <= 16; n++) if (!excludeBlue.includes(n)) blueItems.push(n);
  if (blueItems.length === 0) throw new Error("所有蓝球都被排除了");

  const tickets = [];
  const failureReasons = Object.create(null);
  let tries = 0;

  const sourceW = strategyRed === "mix"
    ? makeMixedWeights(freqR, alpha)
    : makeWeightsFromFreq(freqR, strategyRed, alpha);
  const wBSource = makeWeightsFromFreq(freqB, strategyBlue, alpha);
  const wB = blueItems.map((n) => wBSource[n - 1]);

  while (tickets.length < count && tries < maxTry) {
    tries++;
    const picked = needToPick === 0
      ? []
      : weightedSampleWithoutReplacement(pool, subsetWeights(sourceW, pool), needToPick, rand);
    const reds = [...includeList, ...picked].sort((a, b) => a - b);
    const blue = weightedPickOne(blueItems, wB, rand);

    if (!passesConstraints(reds, constraints)) {
      for (const reason of analyseConstraintFailures(reds, constraints)) {
        failureReasons[reason] = (failureReasons[reason] || 0) + 1;
      }
      continue;
    }
    const key = `${reds.join(",")}|${blue}`;
    if (tickets.some((t) => t.key === key)) continue;
    tickets.push({ key, reds, blue });
  }

  return { tickets, tries, failureReasons };
}
