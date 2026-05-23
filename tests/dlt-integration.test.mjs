import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

function read(rel) {
  return readFileSync(join(ROOT, rel), "utf8");
}

test("root page exposes a first-class DLT entry point", () => {
  const html = read("index.html");
  assert.match(html, /class="lottery-switcher"/);
  assert.match(html, /href="\.\/dlt\.html"/);
  assert.match(html, /大乐透/);
  assert.match(html, /assets\/dlt-styles\.css/);
});

test("DLT backtest module is available to the PWA cache", () => {
  const sw = read("sw.js");

  assert.match(sw, /assets\/js\/dlt-backtest\.js/);
});

test("DLT page is wired to DLT data, styles, and controller", () => {
  const page = join(ROOT, "dlt.html");
  assert.ok(existsSync(page), "dlt.html should exist");
  const html = read("dlt.html");

  assert.match(html, /<title>大乐透数据实验室 · DLT Data Lab<\/title>/);
  assert.match(html, /data\/dlt-draws\.js/);
  assert.match(html, /assets\/js\/dlt-main\.js/);
  assert.match(html, /assets\/dlt-styles\.css/);
  assert.match(html, /data-lottery="dlt"/);
  assert.doesNotMatch(html, /data-tab="lstm"/);
});

test("DLT page contains the expected analytical workbench DOM hooks", () => {
  const html = read("dlt.html");
  const requiredIds = [
    "chartFrontAll",
    "chartBackAll",
    "trendFront",
    "trendBack",
    "distBackOddEven",
    "chartBackSum",
    "strategyFront",
    "strategyBack",
    "includeFront",
    "includeBack",
    "manualFront",
    "manualBack",
    "backtestRounds",
    "backtestMethod",
    "backtestSummary",
    "backtestMatrix",
    "chiFront",
    "chiBack",
  ];

  for (const id of requiredIds) {
    assert.match(html, new RegExp(`id="${id}"`), `${id} should exist`);
  }
  assert.match(html, /1\/21,425,712/);
  assert.match(html, /前区 5 个 \+ 后区 2 个/);
});
