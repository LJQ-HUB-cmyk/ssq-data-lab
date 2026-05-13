export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export const pad2 = (n) => String(n).padStart(2, "0");

export const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

export const sum = (arr) => arr.reduce((a, b) => a + b, 0);

export function createEl(tag, { cls, text, html, attrs } = {}, children = []) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  if (html != null) el.innerHTML = html;
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "dataset") {
        for (const [dk, dv] of Object.entries(v)) el.dataset[dk] = dv;
      } else {
        el.setAttribute(k, v);
      }
    }
  }
  for (const c of children) if (c) el.appendChild(c);
  return el;
}

export function makeBall(n, color) {
  return createEl("span", { cls: `ball ${color}`, text: pad2(n) });
}

export function debounce(fn, ms = 180) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
