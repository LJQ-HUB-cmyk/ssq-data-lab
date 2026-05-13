import { pad2 } from "./utils.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    node.setAttribute(k, String(v));
  }
  for (const c of children) if (c) node.appendChild(c);
  return node;
}

// 基本走势图：横轴=期号（时间从左到右），纵轴=号码（1..size）
// 命中用实心圆点，未命中用淡网格点；点上方文字显示数字
export function renderTrend(container, draws, { size, kind = "red" } = {}) {
  container.innerHTML = "";
  const cellW = 22;
  const cellH = 18;
  const padLeft = 80;
  const padTop = 28;
  const padRight = 16;
  const padBottom = 12;
  const W = padLeft + cellW * size + padRight;
  const H = padTop + cellH * draws.length + padBottom;

  const svg = el("svg", {
    class: "trend-svg",
    viewBox: `0 0 ${W} ${H}`,
    width: W,
    height: H,
  });

  const color = kind === "blue" ? "rgba(58,163,255,.95)" : "rgba(255,59,59,.95)";
  const colorFade = kind === "blue" ? "rgba(138,209,255,.5)" : "rgba(255,107,107,.5)";

  // 顶部号码刻度
  for (let n = 1; n <= size; n++) {
    const x = padLeft + (n - 0.5) * cellW;
    const t = el("text", {
      x, y: padTop - 12, "text-anchor": "middle",
      "font-size": "10",
      "font-family": "ui-monospace, Menlo, monospace",
      fill: "rgba(255,255,255,.62)",
    });
    t.textContent = pad2(n);
    svg.appendChild(t);
  }

  // 斑马纹背景
  draws.forEach((d, rowIdx) => {
    const y = padTop + rowIdx * cellH;
    if (rowIdx % 2 === 0) {
      svg.appendChild(el("rect", {
        x: 0, y, width: W, height: cellH,
        fill: "rgba(255,255,255,.015)",
      }));
    }

    const label = el("text", {
      x: padLeft - 8, y: y + cellH / 2 + 3,
      "text-anchor": "end",
      "font-size": "10",
      "font-family": "ui-monospace, Menlo, monospace",
      fill: "rgba(255,255,255,.55)",
    });
    label.textContent = d.issue;
    svg.appendChild(label);

    for (let n = 1; n <= size; n++) {
      const cx = padLeft + (n - 0.5) * cellW;
      const cy = y + cellH / 2;
      const hit = kind === "blue" ? d.blue === n : d.reds.has(n);
      if (hit) {
        svg.appendChild(el("circle", {
          cx, cy, r: 7, fill: color,
          stroke: "rgba(255,255,255,.2)",
          "stroke-width": 0.8,
        }));
        const t = el("text", {
          x: cx, y: cy + 3.2,
          "text-anchor": "middle",
          "font-size": "9",
          "font-family": "ui-monospace, monospace",
          fill: kind === "blue" ? "#051c2b" : "#2b0505",
          "font-weight": "700",
        });
        t.textContent = pad2(n);
        svg.appendChild(t);
      } else {
        svg.appendChild(el("circle", {
          cx, cy, r: 1.6, fill: colorFade, opacity: 0.25,
        }));
      }
    }
  });

  container.appendChild(svg);
}
