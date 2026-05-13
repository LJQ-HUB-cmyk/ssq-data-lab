import { test } from "node:test";
import assert from "node:assert/strict";

import {
  makeWeightsFromFreq,
  weightedPickOne,
  weightedSampleWithoutReplacement,
  generateTickets,
} from "../assets/js/generator.js";

const RED_SIZE = 33;
const BLUE_SIZE = 16;

function seededRand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function makeFreq(size, fill = 1) {
  const f = Array(size + 1).fill(fill);
  return f;
}

test("makeWeightsFromFreq hot favours higher-freq numbers", () => {
  const freq = makeFreq(RED_SIZE, 0);
  freq[1] = 10;
  freq[2] = 0;
  const w = makeWeightsFromFreq(freq, "hot", 1);
  assert.ok(w[0] > w[1]); // index 0 corresponds to number 1
});

test("makeWeightsFromFreq cold favours lower-freq numbers", () => {
  const freq = makeFreq(RED_SIZE, 0);
  freq[1] = 10;
  freq[2] = 0;
  const w = makeWeightsFromFreq(freq, "cold", 1);
  assert.ok(w[1] > w[0]);
});

test("uniform strategy gives equal weights", () => {
  const freq = makeFreq(RED_SIZE, 5);
  const w = makeWeightsFromFreq(freq, "uniform", 1);
  assert.ok(w.every((x) => x === w[0]));
});

test("weightedPickOne returns an item from items list", () => {
  const items = [10, 20, 30];
  const weights = [0, 1, 0];
  const rand = seededRand(42);
  assert.equal(weightedPickOne(items, weights, rand), 20);
});

test("weightedSampleWithoutReplacement returns k unique items", () => {
  const items = Array.from({ length: RED_SIZE }, (_, i) => i + 1);
  const weights = items.map(() => 1);
  const rand = seededRand(7);
  const picks = weightedSampleWithoutReplacement(items, weights, 6, rand);
  assert.equal(picks.length, 6);
  assert.equal(new Set(picks).size, 6);
  for (const p of picks) assert.ok(p >= 1 && p <= RED_SIZE);
});

test("generateTickets produces requested count under loose constraints", () => {
  const freqR = makeFreq(RED_SIZE, 1);
  const freqB = makeFreq(BLUE_SIZE, 1);
  const rand = seededRand(123);
  const { tickets } = generateTickets({
    freqR,
    freqB,
    strategyRed: "uniform",
    strategyBlue: "uniform",
    alpha: 1,
    constraints: { sum: false, odd: false, span: false, zone: false },
    count: 5,
    rand,
  });
  assert.equal(tickets.length, 5);
  for (const t of tickets) {
    assert.equal(t.reds.length, 6);
    assert.equal(new Set(t.reds).size, 6);
    assert.ok(t.blue >= 1 && t.blue <= BLUE_SIZE);
    for (let i = 1; i < t.reds.length; i++) assert.ok(t.reds[i] > t.reds[i - 1]); // sorted asc
  }
  const keys = new Set(tickets.map((t) => t.key));
  assert.equal(keys.size, tickets.length); // deduped
});

test("generateTickets reports failures when constraints are unsatisfiable", () => {
  // Force impossible span on a tiny pool: make weights concentrate on low numbers only
  const freqR = Array(RED_SIZE + 1).fill(0);
  for (let i = 1; i <= 6; i++) freqR[i] = 100; // only small reds have weight
  const freqB = makeFreq(BLUE_SIZE, 1);
  const rand = seededRand(1);
  const { tickets, tries, failureReasons } = generateTickets({
    freqR,
    freqB,
    strategyRed: "hot",
    strategyBlue: "uniform",
    alpha: 2,
    constraints: { span: true, sum: false, odd: false, zone: false },
    count: 3,
    maxTry: 200,
    rand,
  });
  assert.ok(tries > 0);
  assert.ok(tickets.length <= 3);
  if (tickets.length === 0) {
    assert.ok(Object.keys(failureReasons).length > 0);
  }
});
