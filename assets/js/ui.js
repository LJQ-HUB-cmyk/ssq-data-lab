import { $, $$, pad2, makeBall, createEl, clamp, parseNumList } from "./utils.js";
import { topN, bottomN, sumOf, oddCountOf, spanOf, zoneCounts } from "./stats.js";

export function setupTabs(onChange) {
  const tabs = $$(".tab");
  const panels = $$(".panel");
  const activate = (name) => {
    for (const t of tabs) {
      const active = t.dataset.tab === name;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
      t.tabIndex = active ? 0 : -1;
    }
    for (const p of panels) p.classList.toggle("is-active", p.dataset.panel === name);
    if (onChange) onChange(name);
  };
  tabs.forEach((t, idx) => {
    t.addEventListener("click", () => activate(t.dataset.tab));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const dir = e.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(idx + dir + tabs.length) % tabs.length];
        next.focus();
        activate(next.dataset.tab);
      }
    });
  });
  return activate;
}

export function renderLatest(draw) {
  $("#latestIssue").textContent = draw.issue;
  $("#latestDate").textContent = draw.date || "（无）";
  const wrap = $("#latestBalls");
  wrap.innerHTML = "";
  for (const r of draw.reds) wrap.appendChild(makeBall(r, "red"));
  wrap.appendChild(makeBall(draw.blue, "blue"));
  $("#mLatest").textContent = draw.issue;
}

export function renderHeroMeta(meta, draws) {
  $("#mCount").textContent = String(meta.count || draws.length);
  $("#mRange").textContent = `${draws[0].issue} – ${draws[draws.length - 1].issue}`;
}

export function renderRank(el, pairs) {
  el.innerHTML = "";
  for (const [num, val] of pairs) {
    const li = createEl("li", {
      html: `<span class="mono">${pad2(num)}</span> <span class="muted">×</span> <span class="mono">${val}</span>`,
    });
    el.appendChild(li);
  }
}

export function renderTable(rows, note) {
  const tbody = $("#tbody");
  tbody.innerHTML = "";
  for (const d of rows) {
    const reds = d.reds.map((x) => `<span class="mono" style="color:rgba(255,255,255,.9)">${pad2(x)}</span>`).join(" ");
    const tr = createEl("tr", {
      html: `
        <td class="mono">${d.issue}</td>
        <td class="mono">${d.date || ""}</td>
        <td>${reds}</td>
        <td><span class="mono" style="color:rgba(138,209,255,.95)">${pad2(d.blue)}</span></td>
      `,
    });
    tbody.appendChild(tr);
  }
  $("#dataFootnote").textContent = note || "";
}

export function renderInsightChips({ freqRecentRed, missRed, missBlue }) {
  const chips = $("#insightChips");
  chips.innerHTML = "";
  const hotR = topN(freqRecentRed, 3, 33).map(([n, v]) => `${pad2(n)}×${v}`).join(" / ");
  const coldR = bottomN(freqRecentRed, 3, 33).map(([n, v]) => `${pad2(n)}×${v}`).join(" / ");
  const maxMissR = topN(missRed, 3, 33).map(([n, v]) => `${pad2(n)}·${v}`).join(" / ");
  const maxMissB = topN(missBlue, 2, 16).map(([n, v]) => `${pad2(n)}·${v}`).join(" / ");
  const list = [
    { k: "近期红热", v: hotR },
    { k: "近期红冷", v: coldR },
    { k: "红高遗漏", v: maxMissR },
    { k: "蓝高遗漏", v: maxMissB },
  ];
  for (const it of list) {
    chips.appendChild(createEl("div", { cls: "chip", html: `${it.k} <strong>${it.v}</strong>` }));
  }
}

export function ticketLabel(reds) {
  const z = zoneCounts(reds).join(":");
  return `和值 ${sumOf(reds)} · 奇数 ${oddCountOf(reds)} · 跨度 ${spanOf(reds)} · 三区 ${z}`;
}

function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve) => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch {}
    document.body.removeChild(ta);
    resolve();
  });
}

export function renderTickets(tickets, diagnostics) {
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
    const meta = createEl("div", { cls: "meta", text: `#${idx + 1} · ${ticketLabel(t.reds)}` });
    const nums = createEl("div", { cls: "nums" });
    for (const r of t.reds) nums.appendChild(makeBall(r, "red"));
    nums.appendChild(makeBall(t.blue, "blue"));

    const copyBtn = createEl("button", { cls: "btn ghost btn-copy", text: "复制", attrs: { type: "button" } });
    copyBtn.addEventListener("click", async () => {
      const line = `${t.reds.map(pad2).join(" ")} + ${pad2(t.blue)}`;
      await copyToClipboard(line);
      const original = copyBtn.textContent;
      copyBtn.textContent = "已复制";
      setTimeout(() => (copyBtn.textContent = original), 1200);
    });

    const right = createEl("div", { cls: "ticket-right" }, [nums, copyBtn]);
    const row = createEl("div", { cls: "ticket" }, [meta, right]);
    wrap.appendChild(row);
  });
}

export function showLoadError(message) {
  const shell = $(".shell");
  const card = createEl("div", {
    cls: "card",
    html: `
      <div class="card-title">加载失败</div>
      <div class="fine">无法加载数据。建议用本地 HTTP 方式打开，例如：</div>
      <pre class="mono fine" style="margin:8px 0 0;white-space:pre-wrap">python -m http.server 8000</pre>
      <div class="fine muted" style="margin-top:8px">${message}</div>
    `,
  });
  shell.prepend(card);
}

export function showDataSourceBanner(source, fetchError) {
  if (source !== "embedded") return;
  const shell = $(".shell");
  const existing = shell.querySelector(".banner-embedded");
  if (existing) return;
  const banner = createEl("div", {
    cls: "card banner-embedded",
    html: `
      <div class="card-title">提示：正在使用内置数据</div>
      <div class="fine">当前通过 <span class="mono">window.__SSQ_DATA__</span> 兜底加载。<br/>
        原因：<span class="mono">${(fetchError && fetchError.message) || "无法 fetch draws.json"}</span>。
        建议用 <span class="mono">python -m http.server 8000</span> 打开以获得实时数据。</div>
    `,
  });
  shell.prepend(banner);
}

export function setRefreshLoading(loading) {
  const btn = $("#btnRefresh");
  if (!btn) return;
  btn.disabled = loading;
  btn.classList.toggle("is-loading", loading);
  btn.textContent = loading ? "加载中…" : "刷新";
}

export function readWinSize() {
  return clamp(Number($("#winSize").value || 200), 20, 1000);
}

export function readGeneratorConfig() {
  return {
    strategyRed: $("#strategyRed").value,
    strategyBlue: $("#strategyBlue").value,
    alpha: clamp(Number($("#alpha").value || 0) / 100, 0, 2),
    count: clamp(Number($("#genN").value || 1), 1, 20),
    constraints: {
      sum: $("#cSum").checked,
      odd: $("#cOdd").checked,
      span: $("#cSpan").checked,
      zone: $("#cNo4SameZone").checked,
      ac: $("#cAC")?.checked || false,
      noConsec: $("#cNoConsec")?.checked || false,
    },
    includeRed: parseNumList($("#includeRed")?.value, 1, 33),
    excludeRed: parseNumList($("#excludeRed")?.value, 1, 33),
    excludeBlue: parseNumList($("#excludeBlue")?.value, 1, 16),
    avoidLast: $("#cAvoidLast")?.checked || false,
  };
}

export function showGenError(message) {
  const wrap = $("#results");
  wrap.innerHTML = "";
  wrap.appendChild(createEl("div", {
    cls: "fine",
    html: `<span class="chip chip-warn">${message}</span>`,
  }));
}

export function setGenDiagnostics(text) {
  const el = $("#genDiag");
  if (!el) return;
  el.textContent = text || "";
}
