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

export function generateTickets({
  freqR,
  freqB,
  strategyRed,
  strategyBlue,
  alpha,
  constraints,
  count,
  maxTry = 2000,
  rand = Math.random,
}) {
  const itemsR = Array.from({ length: 33 }, (_, i) => i + 1);
  const itemsB = Array.from({ length: 16 }, (_, i) => i + 1);
  const tickets = [];
  const failureReasons = Object.create(null);
  let tries = 0;

  while (tickets.length < count && tries < maxTry) {
    tries++;
    let reds;
    if (strategyRed === "mix") {
      const wHot = makeWeightsFromFreq(freqR, "hot", alpha);
      const wCold = makeWeightsFromFreq(freqR, "cold", alpha);
      const w = wHot.map((x, i) => (x + wCold[i]) / 2);
      reds = weightedSampleWithoutReplacement(itemsR, w, 6, rand).sort((a, b) => a - b);
    } else {
      const w = makeWeightsFromFreq(freqR, strategyRed, alpha);
      reds = weightedSampleWithoutReplacement(itemsR, w, 6, rand).sort((a, b) => a - b);
    }
    const wB = makeWeightsFromFreq(freqB, strategyBlue, alpha);
    const blue = weightedPickOne(itemsB, wB, rand);

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
