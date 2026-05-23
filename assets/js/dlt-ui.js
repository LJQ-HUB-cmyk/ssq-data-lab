// 大乐透 UI 渲染层
//
// 与 ui.js 中 SSQ 版本一一对应，只处理 DLT 特有的 front/back 字段。
// 通用工具（setupTabs / setupTheme / copyToClipboard / toast / setRefreshLoading / showLoadError）
// 直接 re-export 自 ui.js，避免重复代码。

import { $, $$, pad2, makeBall, createEl, clamp, parseNumList } from "./utils.js";
import {
  setupTabs, setupTheme, copyToClipboard, toast,
  setRefreshLoading, showLoadError,
} from "./ui.js";
import { topNFromFreq, bottomNFromFreq } from "./lottery-stats.js";
import {
  frontSum, frontSpan, frontOddCount, frontAcValue,
  frontOddEvenRatio, frontBigSmallRatio, frontPrimeCompositeRatio,
  frontPath012Ratio, frontZoneRatio,
  frontConsecutiveGroups, frontMaxSameTail,
  backSum, backSpan, backOddEvenRatio,
  FRONT_SIZE, FRONT_PICK, BACK_SIZE, BACK_PICK,
} from "./dlt-distribution.js";

// 通用工具直接 re-export
export {
  setupTabs, setupTheme, copyToClipboard, toast,
  setRefreshLoading, showLoadError,
};

/* =========================================================
   Hero / Latest
   ========================================================= */

export function renderDltLatest(draw) {
  $("#latestIssue").textContent = draw.issue;
  $("#latestDate").textContent = draw.date || "（无）";
  const wrap = $("#latestBalls");
  wrap.innerHTML = "";
  for (const r of draw.front) wrap.appendChild(makeBall(r, "front"));
  draw.back.forEach((b, i) => {
    const ball = makeBall(b, "back");
    if (i === 0) ball.classList.add("plus");
    wrap.appendChild(ball);
  });
  $("#mLatest").textContent = draw.issue;
}

export function renderDltHeroMeta(meta, draws) {
  const n = Number(meta.count || draws.length) || draws.length;
  $("#mCount").textContent = n.toLocaleString();
  $("#mRange").textContent = `${draws[0].issue} – ${draws[draws.length - 1].issue}`;
}

/* =========================================================
   Rank list (前区绿球 / 后区紫球)
   ========================================================= */

export function renderDltRank(el, pairs, color = "front") {
  el.innerHTML = "";
  pairs.forEach(([num, val], i) => {
    const li = createEl("li", {
      html: `
        <span class="mono" style="color:var(--muted-3); width:18px; text-align:right">${i + 1}.</span>
        <span class="ball ${color}" style="width:26px; height:26px; font-size:11px; box-shadow:none">${pad2(num)}</span>
        <span class="muted" style="margin-left:auto; font-family:var(--mono); font-size:12px">×&nbsp;<strong style="color:var(--text)">${val}</strong></span>
      `,
    });
    el.appendChild(li);
  });
}

/* =========================================================
   Data table
   ========================================================= */

export function renderDltTable(rows, note) {
  const tbody = $("#tbody");
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.appendChild(createEl("tr", {
      html: `<td colspan="4" class="muted" style="text-align:center; padding:24px">没有匹配的数据。</td>`,
    }));
  }
  for (const d of rows) {
    const front = d.front.map((x) => `<span class="front-num">${pad2(x)}</span>`).join("&nbsp;&nbsp;");
    const back = d.back.map((x) => `<span class="back-num">${pad2(x)}</span>`).join("&nbsp;");
    const tr = createEl("tr", {
      html: `
        <td class="mono">${d.issue}</td>
        <td class="mono muted">${d.date || ""}</td>
        <td>${front}</td>
        <td>${back}</td>
      `,
    });
    tbody.appendChild(tr);
  }
  $("#dataFootnote").textContent = note || "";
}

/* =========================================================
   Insight chips
   ========================================================= */

function missTopFromFreq(miss, n, size) {
  const pairs = [];
  for (let i = 1; i <= size; i++) pairs.push([i, miss[i]]);
  pairs.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return pairs.slice(0, n);
}

export function renderDltInsightChips({ freqRecentFront, missFront, missBack }) {
  const chips = $("#insightChips");
  if (!chips) return;
  chips.innerHTML = "";
  const items = [
    { k: "近期前热", v: topNFromFreq(freqRecentFront, 3, FRONT_SIZE).map(([n, v]) => `${pad2(n)}·${v}`).join(" / "), kind: "warn" },
    { k: "近期前冷", v: bottomNFromFreq(freqRecentFront, 3, FRONT_SIZE).map(([n, v]) => `${pad2(n)}·${v}`).join(" / "), kind: "" },
    { k: "前高遗漏", v: missTopFromFreq(missFront, 3, FRONT_SIZE).map(([n, v]) => `${pad2(n)}·${v}`).join(" / "), kind: "" },
    { k: "后高遗漏", v: missTopFromFreq(missBack, 2, BACK_SIZE).map(([n, v]) => `${pad2(n)}·${v}`).join(" / "), kind: "" },
  ];
  for (const it of items) {
    const cls = it.kind ? `chip chip-${it.kind}` : "chip";
    chips.appendChild(createEl("div", { cls, html: `${it.k} <strong>${it.v}</strong>` }));
  }
}

/* =========================================================
   Tickets
   ========================================================= */

export function ticketLabel(front, back) {
  const z = [0, 0, 0];
  for (const r of front) z[r <= 12 ? 0 : r <= 24 ? 1 : 2]++;
  return `和值 ${frontSum(front)} · 奇 ${frontOddCount(front)} · 跨度 ${frontSpan(front)} · 三区 ${z.join(":")} · 后 ${back.map(pad2).join("·")}`;
}

export function formatDltTicketLine(ticket) {
  return `${ticket.front.map(pad2).join(" ")} + ${ticket.back.map(pad2).join(" ")}`;
}

export function renderDltTickets(tickets, diagnostics) {
  const wrap = $("#results");
  wrap.innerHTML = "";
  if (tickets.length === 0) {
    const lines = ["没有生成成功：约束过严或窗口过小。"];
    if (diagnostics && diagnostics.failureReasons) {
      const entries = Object.entries(diagnostics.failureReasons).sort((a, b) => b[1] - a[1]);
      if (entries.length) {
        lines.push(`尝试 ${diagnostics.tries} 次，失败原因 Top：`);
        for (const [reason, cnt] of entries.slice(0, 3)) lines.push(`· ${reason}（${cnt}）`);
      }
    }
    wrap.appendChild(createEl("div", { cls: "fine muted", html: lines.join("<br/>") }));
    return;
  }
  tickets.forEach((t, idx) => {
    const meta = createEl("div", {
      cls: "meta",
      html: `<strong style="color:var(--text); font-weight:600">#${pad2(idx + 1)}</strong> · ${ticketLabel(t.front, t.back)}`,
    });
    const nums = createEl("div", { cls: "nums" });
    for (const r of t.front) nums.appendChild(makeBall(r, "front"));
    t.back.forEach((b, i) => {
      const ball = makeBall(b, "back");
      if (i === 0) ball.classList.add("plus");
      nums.appendChild(ball);
    });
    const copyBtn = createEl("button", {
      cls: "btn ghost btn-copy",
      text: "复制",
      attrs: { type: "button", "aria-label": `复制第 ${idx + 1} 注` },
    });
    copyBtn.addEventListener("click", async () => {
      await copyToClipboard(formatDltTicketLine(t));
      const original = copyBtn.textContent;
      copyBtn.textContent = "已复制";
      copyBtn.disabled = true;
      setTimeout(() => {
        copyBtn.textContent = original;
        copyBtn.disabled = false;
      }, 1200);
    });
    const right = createEl("div", { cls: "ticket-right" }, [nums, copyBtn]);
    const row = createEl("div", { cls: "ticket" }, [meta, right]);
    wrap.appendChild(row);
  });
}

/* =========================================================
   Banners
   ========================================================= */

export function showDltDataSourceBanner(source, fetchError) {
  if (source !== "embedded") return;
  const shell = $(".shell");
  if (!shell || shell.querySelector(".banner-embedded")) return;
  const banner = createEl("div", {
    cls: "card banner-embedded",
    html: `
      <div class="card-title">提示：正在使用内置数据</div>
      <div class="fine">当前通过 <span class="mono">window.__DLT_DATA__</span> 兜底加载。<br/>
        原因：<span class="mono">${(fetchError && fetchError.message) || "无法 fetch dlt-draws.json"}</span>。
        建议用 <span class="mono">python -m http.server 8000</span> 打开以获得最新数据。</div>
    `,
  });
  shell.prepend(banner);
}

/* =========================================================
   Generator config readers
   ========================================================= */

export function readDltWinSize() {
  return clamp(Number($("#winSize")?.value || 200), 20, 1000);
}

export function readDltGeneratorConfig() {
  return {
    strategyFront: $("#strategyFront")?.value || "mix",
    strategyBack: $("#strategyBack")?.value || "uniform",
    optimize: $("#optimizeMode")?.value || "none",
    alpha: clamp(Number($("#alpha")?.value || 0) / 100, 0, 2),
    count: clamp(Number($("#genN")?.value || 1), 1, 20),
    constraints: {
      sum: $("#cSum")?.checked || false,
      odd: $("#cOdd")?.checked || false,
      span: $("#cSpan")?.checked || false,
      zone: $("#cNo4SameZone")?.checked || false,
      ac: $("#cAC")?.checked || false,
      noConsec: $("#cNoConsec")?.checked || false,
    },
    includeFront: parseNumList($("#includeFront")?.value, 1, FRONT_SIZE),
    excludeFront: parseNumList($("#excludeFront")?.value, 1, FRONT_SIZE),
    includeBack: parseNumList($("#includeBack")?.value, 1, BACK_SIZE),
    excludeBack: parseNumList($("#excludeBack")?.value, 1, BACK_SIZE),
    avoidLast: $("#cAvoidLast")?.checked || false,
  };
}

export function showDltGenError(message) {
  const wrap = $("#results");
  if (!wrap) return;
  wrap.innerHTML = "";
  wrap.appendChild(createEl("div", {
    cls: "fine",
    html: `<span class="chip chip-warn">${message}</span>`,
  }));
}

export function setDltGenDiagnostics(text) {
  const el = $("#genDiag");
  if (!el) return;
  el.textContent = text || "";
}

/* =========================================================
   Manual ticket analysis
   ========================================================= */

export function renderDltTicketAnalysis(result) {
  const el = $("#ticketAnalysis");
  if (!el) return;
  if (result.error) {
    el.innerHTML = `<span class="chip chip-warn">${result.error}</span>`;
    return;
  }

  const metrics = [
    ["前区和值", result.sum],
    ["前区跨度", result.span],
    ["前区奇偶", result.oddEven],
    ["前区大小", result.bigSmall],
    ["前区质合", result.primeComposite],
    ["前区 012 路", result.path012],
    ["前区三区", result.zone],
    ["前区 AC 值", result.ac],
    ["前区连号组", result.consecutiveGroups],
    ["前区最大同尾", result.maxSameTail],
    ["后区和值", result.backSum],
    ["后区奇偶", result.backOddEven],
  ];
  const hitText = result.historyHits.length
    ? result.historyHits.map((d) => `${d.issue}${d.date ? ` · ${d.date}` : ""}`).join(" / ")
    : "历史上从未完整出现过";

  el.innerHTML = `
    <div class="balls" style="margin-bottom:12px">
      ${result.front.map((r) => `<span class="ball front" style="width:32px;height:32px;font-size:12px">${pad2(r)}</span>`).join("")}
      ${result.back.map((b, i) => `<span class="ball back${i === 0 ? " plus" : ""}" style="width:32px;height:32px;font-size:12px">${pad2(b)}</span>`).join("")}
    </div>
    <div class="analysis-grid">
      ${metrics.map(([k, v]) => `<div class="metric-line"><span>${k}</span><strong>${v}</strong></div>`).join("")}
    </div>
    <div class="callout">
      <div class="callout-title">历史对照</div>
      <div class="callout-body">
        与最新期前区重复 <strong>${result.repeatFront.length}</strong> 个${result.repeatFront.length ? `（${result.repeatFront.map(pad2).join(" ")}）` : ""}；后区重复 <strong>${result.repeatBack.length}</strong> 个${result.repeatBack.length ? `（${result.repeatBack.map(pad2).join(" ")}）` : ""}。<br/>
        ${hitText}。历史未出现不代表更可能出现。
      </div>
    </div>
  `;
}

/* =========================================================
   Sampler diagnostics
   ========================================================= */

function escape(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export function renderDltSamplerDiagnostics(diag, jackpotProb = 1 / 21425712) {
  const el = $("#samplerDiag");
  if (!el) return;
  if (!diag) {
    el.style.display = "none";
    el.innerHTML = "";
    return;
  }
  el.style.display = "";

  const score = diag.qualityScore;
  const scoreColor = score >= 80 ? "var(--dlt-front)" : score >= 60 ? "var(--gold)" : "var(--red-2)";
  const items = [];
  items.push(`<div class="diag-line"><span>采样器</span><strong class="mono">${escape(diag.samplerLabel || diag.method)}</strong></div>`);
  items.push(`<div class="diag-line"><span>种子（可复制重现）</span><strong class="mono" title="点击复制" id="diagSeed">${escape(diag.seed)}</strong></div>`);
  items.push(`<div class="diag-line"><span>前区候选池</span><strong class="mono">${diag.poolSize ?? "—"}${diag.pinned?.length ? ` · 胆码 ${diag.pinned.length}` : ""}</strong></div>`);
  items.push(`<div class="diag-line"><span>后区候选池</span><strong class="mono">${diag.poolBackSize ?? "—"}${diag.pinnedBack?.length ? ` · 胆码 ${diag.pinnedBack.length}` : ""}</strong></div>`);
  items.push(`<div class="diag-line"><span>分布质量分（vs 后验）</span><strong class="mono" style="color:${scoreColor}">${score}/100</strong></div>`);
  items.push(`<div class="diag-line"><span>JS 距离</span><strong class="mono">${diag.jsDistance.toFixed(4)}</strong></div>`);
  items.push(`<div class="diag-line"><span>Wasserstein-1</span><strong class="mono">${diag.wasserstein.toFixed(3)}</strong></div>`);
  if (diag.acceptRate != null) {
    const ar = diag.acceptRate;
    const arColor = ar >= 0.2 && ar <= 0.5 ? "var(--dlt-front)" : "var(--gold)";
    items.push(`<div class="diag-line"><span>MCMC 接受率</span><strong class="mono" style="color:${arColor}">${(ar * 100).toFixed(1)}%</strong> <span class="muted fine">理想区间 20%–50%</span></div>`);
  }
  if (diag.ess != null) {
    items.push(`<div class="diag-line"><span>有效样本数 ESS</span><strong class="mono">${Math.round(diag.ess)}</strong> · τ_int <span class="mono">${diag.tauInt?.toFixed(2)}</span></div>`);
  }
  if (diag.rHat != null && Number.isFinite(diag.rHat)) {
    const rOk = diag.rHat < 1.1;
    const rColor = rOk ? "var(--dlt-front)" : diag.rHat < 1.2 ? "var(--gold)" : "var(--red-2)";
    items.push(`<div class="diag-line"><span>Gelman-Rubin R̂</span><strong class="mono" style="color:${rColor}">${diag.rHat.toFixed(3)}</strong> <span class="muted fine">&lt; 1.1 为收敛</span></div>`);
  }
  if (diag.tries) {
    items.push(`<div class="diag-line"><span>采样尝试</span><strong class="mono">${diag.tries}</strong></div>`);
  }

  el.innerHTML = `
    <div class="diag-head">
      <span class="diag-tag">采样诊断</span>
      <span class="muted fine">所有指标在浏览器本地实时计算</span>
    </div>
    <div class="diag-grid">${items.join("")}</div>
    <div class="callout" style="margin-top:12px">
      <div class="callout-title">如何理解</div>
      <div class="callout-body">
        <strong>质量分</strong> 衡量采样输出的频率分布与贝叶斯后验的接近程度。<br/>
        <strong>它和"中奖概率无关"</strong>。在独立同分布的彩票模型下，任何采样器的中奖期望都等于均匀随机：${jackpotProb.toExponential(2)}（一等奖 1/${(1/jackpotProb).toLocaleString()}）。
      </div>
    </div>
  `;

  const seedEl = $("#diagSeed");
  if (seedEl) {
    seedEl.style.cursor = "copy";
    seedEl.addEventListener("click", async () => {
      await copyToClipboard(diag.seed);
      toast("种子已复制：粘到「种子」输入框可重现该结果");
    });
  }
}
