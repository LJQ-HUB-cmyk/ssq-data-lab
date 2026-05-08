/* global window, document */

const DATA_URL = "./data/draws.json";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function pad2(n) {
  return String(n).padStart(2, "0");
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function sum(arr) {
  return arr.reduce((x, y) => x + y, 0);
}

function span(reds) {
  return Math.max(...reds) - Math.min(...reds);
}

function oddCount(reds) {
  return reds.filter((x) => x % 2 === 1).length;
}

function zoneIndex(n) {
  if (n <= 11) return 0;
  if (n <= 22) return 1;
  return 2;
}

function zoneCounts(reds) {
  const z = [0, 0, 0];
  for (const r of reds) z[zoneIndex(r)]++;
  return z;
}

function makeBall(n, color) {
  const s = document.createElement("span");
  s.className = `ball ${color}`;
  s.textContent = pad2(n);
  return s;
}

function freqFromDraws(draws, field, size) {
  const f = Array(size + 1).fill(0);
  for (const d of draws) {
    if (field === "reds") {
      for (const r of d.reds) f[r] += 1;
    } else if (field === "blue") {
      f[d.blue] += 1;
    }
  }
  return f;
}

function missCounts(draws, field, size) {
  // “遗漏”：距离最后一次出现的期数；0 表示最后一期就出现
  const lastSeen = Array(size + 1).fill(null);
  const lastIndex = draws.length - 1;
  for (let i = lastIndex; i >= 0; i--) {
    const d = draws[i];
    if (field === "reds") {
      for (const r of d.reds) {
        if (lastSeen[r] == null) lastSeen[r] = i;
      }
    } else if (field === "blue") {
      const b = d.blue;
      if (lastSeen[b] == null) lastSeen[b] = i;
    }
  }
  const miss = Array(size + 1).fill(0);
  for (let n = 1; n <= size; n++) {
    miss[n] = lastSeen[n] == null ? draws.length : lastIndex - lastSeen[n];
  }
  return miss;
}

function topN(freqArr, n, size) {
  const pairs = [];
  for (let i = 1; i <= size; i++) pairs.push([i, freqArr[i]]);
  pairs.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return pairs.slice(0, n);
}

function renderRank(el, pairs) {
  el.innerHTML = "";
  for (const [num, val] of pairs) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="mono">${pad2(num)}</span> <span class="muted">×</span> <span class="mono">${val}</span>`;
    el.appendChild(li);
  }
}

function renderBars(el, values, size, mode = "red") {
  el.innerHTML = "";
  const maxV = Math.max(...values.slice(1));
  for (let i = 1; i <= size; i++) {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.dataset.label = pad2(i);
    const inner = document.createElement("i");
    const h = maxV === 0 ? 0 : Math.round((values[i] / maxV) * 100);
    inner.style.height = `${clamp(h, 0, 100)}%`;
    if (mode === "miss") {
      // 遗漏：用“酸绿”强调高遗漏
      inner.style.background =
        "linear-gradient(180deg, rgba(185,255,90,.95), rgba(185,255,90,.25))";
    }
    bar.title = `${pad2(i)}：${values[i]}`;
    bar.appendChild(inner);
    el.appendChild(bar);
  }
}

function weightedPickOne(items, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) return items[Math.floor(Math.random() * items.length)];
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

function weightedSampleWithoutReplacement(items, weights, k) {
  const poolItems = items.slice();
  const poolWeights = weights.slice();
  const out = [];
  for (let i = 0; i < k; i++) {
    const picked = weightedPickOne(poolItems, poolWeights);
    out.push(picked);
    const idx = poolItems.indexOf(picked);
    poolItems.splice(idx, 1);
    poolWeights.splice(idx, 1);
  }
  return out;
}

function makeWeightsFromFreq(freq, strategy, alpha01) {
  // alpha01: 0..2（越大越偏）
  const size = freq.length - 1;
  const maxF = Math.max(...freq.slice(1));
  const w = [];
  for (let n = 1; n <= size; n++) {
    const f = freq[n];
    let base = 1;
    if (strategy === "uniform") base = 1;
    if (strategy === "hot") base = f + 1;
    if (strategy === "cold") base = (maxF - f) + 1;
    w.push(Math.pow(base, alpha01));
  }
  return w;
}

function passesConstraints(reds, c) {
  const s = sum(reds);
  const oc = oddCount(reds);
  const sp = span(reds);
  const z = zoneCounts(reds);

  if (c.sum && (s < 70 || s > 150)) return false;
  if (c.odd && (oc < 2 || oc > 4)) return false;
  if (c.span && sp < 18) return false;
  if (c.zone && Math.max(...z) > 4) return false;
  return true;
}

function ticketLabel(reds, blue) {
  const s = sum(reds);
  const oc = oddCount(reds);
  const sp = span(reds);
  const z = zoneCounts(reds).join(":");
  return `和值 ${s} · 奇数 ${oc} · 跨度 ${sp} · 三区 ${z}`;
}

function renderLatest(draw) {
  $("#latestIssue").textContent = draw.issue;
  $("#latestDate").textContent = draw.date || "（无）";

  const wrap = $("#latestBalls");
  wrap.innerHTML = "";
  for (const r of draw.reds) wrap.appendChild(makeBall(r, "red"));
  wrap.appendChild(makeBall(draw.blue, "blue"));

  $("#mLatest").textContent = draw.issue;
}

function renderHeroMeta(meta, draws) {
  $("#mCount").textContent = String(meta.count || draws.length);
  $("#mRange").textContent = `${draws[0].issue} – ${draws[draws.length - 1].issue}`;
}

function renderTable(draws, note) {
  const tbody = $("#tbody");
  tbody.innerHTML = "";
  for (const d of draws) {
    const tr = document.createElement("tr");
    const reds = d.reds.map((x) => `<span class="mono" style="color:rgba(255,255,255,.9)">${pad2(x)}</span>`).join(" ");
    tr.innerHTML = `
      <td class="mono">${d.issue}</td>
      <td class="mono">${d.date || ""}</td>
      <td>${reds}</td>
      <td><span class="mono" style="color:rgba(138,209,255,.95)">${pad2(d.blue)}</span></td>
    `;
    tbody.appendChild(tr);
  }
  $("#dataFootnote").textContent = note || "";
}

function renderInsightChips(freqRecentRed, freqRecentBlue, missRed, missBlue) {
  const chips = $("#insightChips");
  chips.innerHTML = "";

  const hotR = topN(freqRecentRed, 3, 33).map(([n, v]) => `${pad2(n)}×${v}`).join(" / ");
  const coldR = topN(freqRecentRed.map((x) => -x), 3, 33).map(([n]) => pad2(n)).join(" / "); // 最低频

  const maxMissR = topN(missRed, 3, 33).map(([n, v]) => `${pad2(n)}·${v}`).join(" / ");
  const maxMissB = topN(missBlue, 2, 16).map(([n, v]) => `${pad2(n)}·${v}`).join(" / ");

  const list = [
    { k: "近期红热", v: hotR },
    { k: "近期红冷", v: coldR },
    { k: "红高遗漏", v: maxMissR },
    { k: "蓝高遗漏", v: maxMissB },
  ];
  for (const it of list) {
    const c = document.createElement("div");
    c.className = "chip";
    c.innerHTML = `${it.k} <strong>${it.v}</strong>`;
    chips.appendChild(c);
  }
}

function setupTabs() {
  const tabs = $$(".tab");
  const panels = $$(".panel");
  function activate(name) {
    for (const t of tabs) t.classList.toggle("is-active", t.dataset.tab === name);
    for (const p of panels) p.classList.toggle("is-active", p.dataset.panel === name);
  }
  for (const t of tabs) {
    t.addEventListener("click", () => activate(t.dataset.tab));
  }
}

async function loadData(noCache = false) {
  const url = noCache ? `${DATA_URL}?t=${Date.now()}` : DATA_URL;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`加载数据失败：${res.status}`);
  return res.json();
}

function loadEmbeddedData() {
  // draws.js 会注入 window.__SSQ_DATA__，用于 file:// 直接打开时绕过 fetch 限制
  if (window && window.__SSQ_DATA__ && window.__SSQ_DATA__.draws) return window.__SSQ_DATA__;
  return null;
}

function renderAll(state) {
  const { meta, draws, winSize } = state;
  const latest = draws[draws.length - 1];
  renderHeroMeta(meta, draws);
  renderLatest(latest);

  // 全量频次
  state.freqAllRed = freqFromDraws(draws, "reds", 33);
  state.freqAllBlue = freqFromDraws(draws, "blue", 16);
  renderBars($("#chartRedAll"), state.freqAllRed, 33, "red");
  renderBars($("#chartBlueAll"), state.freqAllBlue, 16, "blue");

  // 近期频次
  const recent = draws.slice(-winSize);
  state.freqRecentRed = freqFromDraws(recent, "reds", 33);
  state.freqRecentBlue = freqFromDraws(recent, "blue", 16);
  renderRank($("#rankRedRecent"), topN(state.freqRecentRed, 8, 33));
  renderRank($("#rankBlueRecent"), topN(state.freqRecentBlue, 6, 16));

  // 遗漏
  state.missRed = missCounts(draws, "reds", 33);
  state.missBlue = missCounts(draws, "blue", 16);
  renderBars($("#chartRedMiss"), state.missRed, 33, "miss");
  renderBars($("#chartBlueMiss"), state.missBlue, 16, "miss");
  renderInsightChips(state.freqRecentRed, state.freqRecentBlue, state.missRed, state.missBlue);

  // 数据表（默认最近 50）
  renderTable(draws.slice(-50).reverse(), `共 ${draws.length} 期；显示最近 50 期（倒序）。`);
}

function generateTickets(state) {
  const { draws, winSize } = state;
  const recent = draws.slice(-winSize);
  const freqR = freqFromDraws(recent, "reds", 33);
  const freqB = freqFromDraws(recent, "blue", 16);

  const strategyRed = $("#strategyRed").value;
  const strategyBlue = $("#strategyBlue").value;
  const alpha = Number($("#alpha").value || 0);
  const alpha01 = clamp(alpha / 100, 0, 2);

  const itemsR = Array.from({ length: 33 }, (_, i) => i + 1);
  const itemsB = Array.from({ length: 16 }, (_, i) => i + 1);

  const genN = clamp(Number($("#genN").value || 1), 1, 20);

  const c = {
    sum: $("#cSum").checked,
    odd: $("#cOdd").checked,
    span: $("#cSpan").checked,
    zone: $("#cNo4SameZone").checked,
  };

  const wrap = $("#results");
  wrap.innerHTML = "";

  const tickets = [];
  const maxTry = 2000;
  let tries = 0;

  while (tickets.length < genN && tries < maxTry) {
    tries++;

    let reds;
    if (strategyRed === "mix") {
      // 混合：一半热权重 + 一半冷权重
      const wHot = makeWeightsFromFreq(freqR, "hot", alpha01);
      const wCold = makeWeightsFromFreq(freqR, "cold", alpha01);
      const w = wHot.map((x, i) => (x + wCold[i]) / 2);
      reds = weightedSampleWithoutReplacement(itemsR, w, 6).sort((a, b) => a - b);
    } else {
      const w = makeWeightsFromFreq(freqR, strategyRed, alpha01);
      reds = weightedSampleWithoutReplacement(itemsR, w, 6).sort((a, b) => a - b);
    }

    const wB = makeWeightsFromFreq(freqB, strategyBlue, alpha01);
    const blue = weightedPickOne(itemsB, wB);

    if (!passesConstraints(reds, c)) continue;

    // 去重：避免同一注重复生成
    const key = `${reds.join(",")}|${blue}`;
    if (tickets.some((t) => t.key === key)) continue;
    tickets.push({ key, reds, blue });
  }

  if (tickets.length === 0) {
    const note = document.createElement("div");
    note.className = "fine muted";
    note.textContent = "没有生成成功：可能是约束过多或窗口过小。请放宽约束后再试。";
    wrap.appendChild(note);
    return;
  }

  tickets.forEach((t, idx) => {
    const row = document.createElement("div");
    row.className = "ticket";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `#${idx + 1} · ${ticketLabel(t.reds, t.blue)}`;

    const nums = document.createElement("div");
    nums.className = "nums";
    for (const r of t.reds) nums.appendChild(makeBall(r, "red"));
    nums.appendChild(makeBall(t.blue, "blue"));

    row.appendChild(meta);
    row.appendChild(nums);
    wrap.appendChild(row);
  });
}

function setupInteractions(state) {
  $("#btnApplyWin").addEventListener("click", () => {
    const v = clamp(Number($("#winSize").value || 200), 20, 1000);
    state.winSize = v;
    renderAll(state);
  });

  $("#btnGen").addEventListener("click", () => generateTickets(state));

  $("#btnSearch").addEventListener("click", () => {
    const q = ($("#qIssue").value || "").trim();
    if (!q) {
      renderTable(state.draws.slice(-50).reverse(), `共 ${state.draws.length} 期；显示最近 50 期（倒序）。`);
      return;
    }
    const hit = state.draws.filter((d) => d.issue.includes(q)).slice(-120).reverse();
    renderTable(hit, `搜索 “${q}” ：命中 ${hit.length} 条（最多展示 120 条）。`);
  });

  $("#btnClear").addEventListener("click", () => {
    $("#qIssue").value = "";
    renderTable(state.draws.slice(-50).reverse(), `共 ${state.draws.length} 期；显示最近 50 期（倒序）。`);
  });

  $("#btnRefresh").addEventListener("click", async () => {
    try {
      const json = await loadData(true);
      state.meta = json.meta || {};
      state.draws = json.draws || [];
      renderAll(state);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(String(e.message || e));
    }
  });
}

async function main() {
  setupTabs();
  const state = { meta: {}, draws: [], winSize: 200 };

  try {
    let json = null;
    try {
      json = await loadData(false);
    } catch (e) {
      // fetch 失败时，尝试使用内置数据（例如用户直接双击 index.html 用 file:// 打开）
      const embedded = loadEmbeddedData();
      if (embedded) json = embedded;
      else throw e;
    }
    state.meta = json.meta || {};
    state.draws = (json.draws || []).filter((d) => d && d.reds && d.blue);
    state.winSize = clamp(Number($("#winSize").value || 200), 20, 1000);
    renderAll(state);
    setupInteractions(state);
  } catch (e) {
    const shell = document.querySelector(".shell");
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-title">加载失败</div>
      <div class="fine">无法加载 <span class="mono">${DATA_URL}</span>。建议用本地 HTTP 方式打开（例如 python -m http.server）。</div>
      <div class="fine muted" style="margin-top:8px">${String(e.message || e)}</div>
    `;
    shell.prepend(card);
  }
}

window.addEventListener("DOMContentLoaded", main);
