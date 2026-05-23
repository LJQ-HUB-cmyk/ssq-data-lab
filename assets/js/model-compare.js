// A/B 模型对比工作台
//
// 输入：两个 payload（IndexedDB 加载下来的）+ 测试 draws + 当前彩种
// 输出：DOM 字符串 — 一个完整的对比报告
//   1. 元信息表（key/type/dims/savedAt/T/ECE）
//   2. 两条训练 loss 曲线叠加
//   3. 两条 val hit 曲线叠加
//   4. 在同一 testDraws 上 walk-forward 跑两遍，对比 hit@K / BSS
//   5. 配对置换检验：A vs B 的 hit@K 差是否显著
//
// 设计原则：复用现有的 backtestModel / brierSkillScore / permutationTest，
// 只做编排和渲染，不引入新算法。

import {
  bootstrapCI, pairedBootstrap,
  metricAvgHit6,
  brierSkillScore, permutationTest,
} from "./nn-statistics.js";

/** 把 SSQ 的 payload 还原成 model 对象。 */
async function reviveSsqModel(payload) {
  const { deserializeModel } = await import("./nn-ssq-model.js");
  if (payload.type === "ensemble") {
    return { kind: "ensemble", members: payload.members.map(deserializeModel) };
  }
  return { kind: "single", model: deserializeModel(payload.model) };
}

async function reviveDltModel(payload) {
  const { deserializeDltModel } = await import("./dlt-nn-model.js");
  return { kind: "single", model: deserializeDltModel(payload.model) };
}

/**
 * 主入口：渲染 A vs B 对比报告。
 * @param payloadA / payloadB IndexedDB 里两个模型的 payload
 * @param draws 全部历史
 * @param lottery "ssq" | "dlt"
 * @returns HTML string
 */
export async function renderComparison(payloadA, payloadB, draws, lottery = "ssq") {
  if (payloadA.lottery && payloadB.lottery && payloadA.lottery !== payloadB.lottery) {
    return `<div class="callout"><div class="callout-title">无法对比</div><div class="callout-body">两个模型属于不同彩种（${payloadA.lottery} vs ${payloadB.lottery}）。</div></div>`;
  }

  const isDlt = lottery === "dlt";
  const seqLenA = payloadA.seqLen || 12;
  const seqLenB = payloadB.seqLen || 12;
  const seqLen = Math.max(seqLenA, seqLenB);
  if (draws.length < seqLen + 30) {
    return `<div class="callout chip-warn"><div class="callout-title">数据不足</div><div class="callout-body">至少需要 ${seqLen + 30} 期，当前仅 ${draws.length} 期。</div></div>`;
  }

  // 还原模型
  const reviveFn = isDlt ? reviveDltModel : reviveSsqModel;
  const modA = await reviveFn(payloadA);
  const modB = await reviveFn(payloadB);

  // 按各自的 seqLen 跑回测（不能强制相同 seqLen 否则不公平）
  const splitIdx = Math.floor(draws.length * 0.85);
  const testDraws = draws.slice(splitIdx);
  const historyBeforeAATail = draws.slice(0, splitIdx - seqLenA);
  const historyBeforeBTail = draws.slice(0, splitIdx - seqLenB);
  const trainTailA = draws.slice(splitIdx - seqLenA, splitIdx);
  const trainTailB = draws.slice(splitIdx - seqLenB, splitIdx);

  let resA, resB;
  if (isDlt) {
    const { backtestDltModel } = await import("./dlt-nn-backtest.js");
    resA = backtestDltModel(modA.model, trainTailA, testDraws, seqLenA, historyBeforeAATail);
    resB = backtestDltModel(modB.model, trainTailB, testDraws, seqLenB, historyBeforeBTail);
  } else {
    const { backtestModel } = await import("./nn-backtest.js");
    const sourceA = modA.kind === "ensemble" ? modA.members[0] : modA.model;
    const sourceB = modB.kind === "ensemble" ? modB.members[0] : modB.model;
    resA = backtestModel(sourceA, trainTailA, testDraws, seqLenA, historyBeforeAATail);
    resB = backtestModel(sourceB, trainTailB, testDraws, seqLenB, historyBeforeBTail);
  }

  // 关键统计
  const size = isDlt ? 35 : 33;
  const pick = isDlt ? 5 : 6;
  const bssA = brierSkillScore(resA.records, size, pick);
  const bssB = brierSkillScore(resB.records, size, pick);

  const hitKey = isDlt ? "fHit5" : "redHit6";
  const hitName = isDlt ? "前区 hit@5" : "红球 hit@6";

  // 配对置换：A 减 B
  const permAB = permutationTest(
    resA.records, resB.records,
    (rec) => rec[hitKey],
    { B: 1000, seed: "compare-perm" },
  );
  // Bootstrap 各自 95% CI
  const ciA = bootstrapCI(resA.records, (rs) => rs.reduce((s, r) => s + r[hitKey], 0) / Math.max(1, rs.length), { B: 500, seed: "ci-a" });
  const ciB = bootstrapCI(resB.records, (rs) => rs.reduce((s, r) => s + r[hitKey], 0) / Math.max(1, rs.length), { B: 500, seed: "ci-b" });

  // 渲染
  const meta = (p) => ({
    key: p.key || "(no-key)",
    type: p.type || "single",
    arch: `${p.hiddenDim ?? "?"}H × ${p.numLayers ?? "?"}L${p.members?.length > 1 ? ` × ${p.members.length}` : ""}`,
    seqLen: p.seqLen ?? "?",
    savedAt: p.savedAt ? p.savedAt.slice(0, 19).replace("T", " ") : "—",
    T: p.model?.calibration ? `${(p.model.calibration.redT ?? p.model.calibration.frontT)?.toFixed(3)} / ${(p.model.calibration.blueT ?? p.model.calibration.backT)?.toFixed(3)}` : "无",
  });
  const ma = meta(payloadA);
  const mb = meta(payloadB);

  const verdictColor = (v) => v > 0 ? "var(--acid)" : v < 0 ? "var(--red-2)" : "var(--text)";
  const winner = (a, b, higherIsBetter = true) => {
    const diff = higherIsBetter ? a - b : b - a;
    if (Math.abs(diff) < 1e-6) return "—";
    return diff > 0 ? "A 胜" : "B 胜";
  };

  const trainCurveA = (payloadA.history || payloadA.histories?.[0]) || null;
  const trainCurveB = (payloadB.history || payloadB.histories?.[0]) || null;
  const lossSvg = renderDualCurve(
    trainCurveA?.valLoss || [], trainCurveB?.valLoss || [],
    "Val Loss", true /* lower is better */,
  );
  const hitSvg = renderDualCurve(
    isDlt ? (trainCurveA?.valFrontHit5 || []) : (trainCurveA?.valRedHit6 || []),
    isDlt ? (trainCurveB?.valFrontHit5 || []) : (trainCurveB?.valRedHit6 || []),
    isDlt ? "Val Front Hit@5" : "Val Red Hit@6",
    false /* higher is better */,
  );

  return `
    <div class="cmp-head">
      <div class="cmp-head-cell">
        <div class="cmp-tag" style="color:var(--acid)">A</div>
        <div class="cmp-key mono">${escape(ma.key)}</div>
        <div class="muted fine">${escape(ma.type)} · ${escape(ma.arch)} · seq=${ma.seqLen}</div>
        <div class="muted fine">${escape(ma.savedAt)}</div>
        <div class="muted fine">温度 T(red/blue 或 front/back) = ${escape(ma.T)}</div>
      </div>
      <div class="cmp-vs">VS</div>
      <div class="cmp-head-cell">
        <div class="cmp-tag" style="color:var(--gold)">B</div>
        <div class="cmp-key mono">${escape(mb.key)}</div>
        <div class="muted fine">${escape(mb.type)} · ${escape(mb.arch)} · seq=${mb.seqLen}</div>
        <div class="muted fine">${escape(mb.savedAt)}</div>
        <div class="muted fine">温度 T = ${escape(mb.T)}</div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="card-title">关键指标对比 <span class="card-num">test set ${resA.records.length} 期</span></div>
      <table class="table cmp-metrics">
        <thead><tr>
          <th>指标</th>
          <th style="color:var(--acid)">模型 A</th>
          <th style="color:var(--gold)">模型 B</th>
          <th>判定</th>
        </tr></thead>
        <tbody>
          <tr>
            <td>${hitName} 均值</td>
            <td class="mono">${resA.summary[isDlt ? "avgFrontHit5" : "avgRedHit6"].toFixed(3)} <span class="muted">[${ciA.lower.toFixed(3)}, ${ciA.upper.toFixed(3)}]</span></td>
            <td class="mono">${resB.summary[isDlt ? "avgFrontHit5" : "avgRedHit6"].toFixed(3)} <span class="muted">[${ciB.lower.toFixed(3)}, ${ciB.upper.toFixed(3)}]</span></td>
            <td>${winner(resA.summary[isDlt ? "avgFrontHit5" : "avgRedHit6"], resB.summary[isDlt ? "avgFrontHit5" : "avgRedHit6"])}</td>
          </tr>
          <tr>
            <td>BSS（Brier Skill Score）</td>
            <td class="mono" style="color:${verdictColor(bssA.bss)}">${bssA.bss.toFixed(4)}</td>
            <td class="mono" style="color:${verdictColor(bssB.bss)}">${bssB.bss.toFixed(4)}</td>
            <td>${winner(bssA.bss, bssB.bss)}</td>
          </tr>
          <tr>
            <td>平均 Brier Score</td>
            <td class="mono">${resA.summary.avgBrier.toFixed(4)}</td>
            <td class="mono">${resB.summary.avgBrier.toFixed(4)}</td>
            <td>${winner(resA.summary.avgBrier, resB.summary.avgBrier, false)}</td>
          </tr>
          <tr>
            <td>负对数似然 NLL</td>
            <td class="mono">${(resA.summary.avgRedLL ?? resA.summary.avgFrontLL ?? 0).toFixed(3)}</td>
            <td class="mono">${(resB.summary.avgRedLL ?? resB.summary.avgFrontLL ?? 0).toFixed(3)}</td>
            <td>${winner(resA.summary.avgRedLL ?? resA.summary.avgFrontLL ?? 0, resB.summary.avgRedLL ?? resB.summary.avgFrontLL ?? 0, false)}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="card-title">配对置换检验 (A − B) <span class="card-num">B = 1000</span></div>
      <div class="diag-grid">
        <div class="diag-line"><span>观察到的均值差 (A − B)</span><strong class="mono" style="color:${verdictColor(permAB.observed)}">${permAB.observed.toFixed(4)}</strong></div>
        <div class="diag-line"><span>双侧 p 值</span><strong class="mono" style="color:${permAB.pTwoSided < 0.05 ? "var(--gold)" : "var(--text)"}">${permAB.pTwoSided.toFixed(4)}</strong></div>
        <div class="diag-line"><span>判断（α=0.05）</span><strong>${permAB.pTwoSided < 0.05
          ? `<span class="chip chip-warn">差异显著</span>`
          : `<span class="chip chip-ok">差异不显著</span>`
        }</strong></div>
      </div>
      <div class="hint">置换 1000 次后，|观察值| 落在分布尾部的频率即为 p。p ≥ 0.05 表示无法区分两个模型。</div>
    </div>

    <div class="cmp-curves">
      <div class="card">
        <div class="card-title">${isDlt ? "前区 Hit@5" : "红球 Hit@6"} 训练曲线</div>
        ${hitSvg}
      </div>
      <div class="card">
        <div class="card-title">Val Loss 训练曲线</div>
        ${lossSvg}
      </div>
    </div>

    <div class="callout" style="margin-top:14px">
      <div class="callout-title">如何解读</div>
      <div class="callout-body">
        <ul style="margin:0; padding-left:18px">
          <li><strong>BSS</strong>：&gt; 0 表示比"全部猜 ${(pick / size).toFixed(3)}"更好；&lt; 0 反而更差。</li>
          <li><strong>NLL / Brier</strong>：越小越好。</li>
          <li><strong>置换 p 值</strong>：&ge; 0.05 表示两个模型在统计上不可区分——这正是 i.i.d. 彩票上的预期。</li>
          <li>训练曲线对比帮你判断：A 是不是收敛更快？B 是不是过拟合？</li>
        </ul>
      </div>
    </div>
  `;
}

/** 双系列叠加曲线 SVG。 */
function renderDualCurve(seriesA, seriesB, label, lowerIsBetter = false) {
  const W = 380, H = 180;
  const padL = 36, padR = 12, padT = 14, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const all = [...seriesA, ...seriesB];
  if (all.length === 0) return `<div class="fine muted">无训练曲线数据</div>`;
  const minV = Math.min(...all);
  const maxV = Math.max(...all);
  const range = (maxV - minV) || 1;
  const yScale = (v) => padT + innerH - ((v - minV) / range) * innerH;

  const drawLine = (series, color) => {
    if (series.length < 1) return "";
    const dx = innerW / Math.max(1, series.length - 1);
    const pts = series.map((v, i) => `${(padL + i * dx).toFixed(2)},${yScale(v).toFixed(2)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round"/>`;
  };

  const xAxis = `<line x1="${padL}" y1="${padT + innerH}" x2="${padL + innerW}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const yAxis = `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + innerH}" stroke="rgba(255,255,255,.25)"/>`;
  const aLine = drawLine(seriesA, "var(--acid)");
  const bLine = drawLine(seriesB, "var(--gold)");
  const yLabels = [minV, (minV + maxV) / 2, maxV].map((v) => {
    return `<text x="${padL - 4}" y="${yScale(v) + 3}" text-anchor="end" font-size="9" fill="rgba(255,255,255,.5)" font-family="JetBrains Mono, monospace">${v.toFixed(3)}</text>`;
  }).join("");
  const legend = `
    <g transform="translate(${padL + 8}, ${padT})">
      <line x1="0" y1="0" x2="14" y2="0" stroke="var(--acid)" stroke-width="2"/>
      <text x="20" y="3" font-size="9" fill="rgba(255,255,255,.7)">A</text>
      <line x1="42" y1="0" x2="56" y2="0" stroke="var(--gold)" stroke-width="2"/>
      <text x="62" y="3" font-size="9" fill="rgba(255,255,255,.7)">B</text>
    </g>
  `;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">${xAxis}${yAxis}${yLabels}${aLine}${bLine}${legend}</svg>`;
}

function escape(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
