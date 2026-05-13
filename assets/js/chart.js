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

function ensureTooltip(container) {
  let tip = container.querySelector(".chart-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "chart-tip";
    tip.setAttribute("role", "tooltip");
    container.appendChild(tip);
  }
  return tip;
}

function palette(mode) {
  if (mode === "blue") {
    return { fillTop: "rgba(138,209,255,.95)", fillBot: "rgba(58,163,255,.35)", stroke: "rgba(58,163,255,.55)", glow: "rgba(58,163,255,.45)" };
  }
  if (mode === "miss") {
    return { fillTop: "rgba(185,255,90,.95)", fillBot: "rgba(185,255,90,.20)", stroke: "rgba(185,255,90,.55)", glow: "rgba(185,255,90,.45)" };
  }
  return { fillTop: "rgba(255,107,107,.95)", fillBot: "rgba(255,59,59,.35)", stroke: "rgba(255,59,59,.60)", glow: "rgba(255,59,59,.45)" };
}

export function renderBars(container, values, size, mode = "red", { unit = "次" } = {}) {
  container.innerHTML = "";
  container.style.position = "relative";
  const tip = ensureTooltip(container);

  const W = container.clientWidth || 520;
  const H = container.clientHeight || 180;
  const padX = 14;
  const padTop = 10;
  const padBot = 22;
  const innerW = W - padX * 2;
  const innerH = H - padTop - padBot;
  const gap = 3;
  const barW = Math.max(4, (innerW - gap * (size - 1)) / size);

  const vs = values.slice(1, size + 1);
  const maxV = Math.max(1, ...vs);
  const colors = palette(mode);

  const svg = el("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${W} ${H}`,
    width: "100%",
    height: "100%",
    preserveAspectRatio: "none",
  });

  const defs = el("defs");
  const gradId = `grad-${mode}-${Math.random().toString(36).slice(2, 7)}`;
  const grad = el("linearGradient", { id: gradId, x1: "0", x2: "0", y1: "0", y2: "1" });
  grad.appendChild(el("stop", { offset: "0%", "stop-color": colors.fillTop }));
  grad.appendChild(el("stop", { offset: "100%", "stop-color": colors.fillBot }));
  defs.appendChild(grad);
  svg.appendChild(defs);

  for (let t = 1; t <= 3; t++) {
    const y = padTop + innerH - (innerH * t) / 4;
    svg.appendChild(el("line", {
      x1: padX, x2: W - padX, y1: y, y2: y,
      stroke: "rgba(255,255,255,.06)", "stroke-dasharray": "3 4",
    }));
  }

  for (let i = 0; i < size; i++) {
    const v = vs[i];
    const h = maxV === 0 ? 0 : Math.round((v / maxV) * innerH);
    const x = padX + i * (barW + gap);
    const y = padTop + innerH - h;

    const bar = el("rect", {
      x, y, width: barW, height: Math.max(1, h),
      rx: 2, ry: 2,
      fill: `url(#${gradId})`,
      stroke: colors.stroke, "stroke-width": 0.8,
      "data-idx": i + 1, "data-val": v,
      class: "chart-bar",
    });
    svg.appendChild(bar);

    if ((i + 1) % 5 === 0 || i === 0 || i === size - 1) {
      const label = el("text", {
        x: x + barW / 2,
        y: H - 6,
        "text-anchor": "middle",
        "font-size": "10",
        "font-family": "ui-monospace, Menlo, Monaco, Consolas, monospace",
        fill: "rgba(255,255,255,.46)",
      });
      label.textContent = pad2(i + 1);
      svg.appendChild(label);
    }
  }

  container.appendChild(svg);

  svg.addEventListener("mousemove", (ev) => {
    const target = ev.target;
    if (!(target instanceof SVGRectElement)) return;
    const idx = target.getAttribute("data-idx");
    const val = target.getAttribute("data-val");
    if (!idx) return;
    tip.textContent = `${pad2(Number(idx))} · ${val} ${unit}`;
    tip.style.opacity = "1";
    const rect = container.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    tip.style.left = `${x + 12}px`;
    tip.style.top = `${y - 28}px`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.style.opacity = "0";
  });
}
