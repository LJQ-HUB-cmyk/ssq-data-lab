// 模型管理器对话框：
//   列出 IndexedDB 里所有模型 → 切换 / 删除 / 下载 / 重命名
//
// 通过简单的 <dialog> 元素弹出，零依赖。
// 设计：键盘 ESC 关闭、点击 backdrop 关闭、复用现有 .btn 样式。

import * as modelStorage from "./model-storage.js";

let _dialogEl = null;

/** 打开模型管理器。lottery="ssq"|"dlt"，过滤只显示该彩种的模型。 */
export async function openModelManager({
  lottery,
  currentKey,
  onSwitch,        // (newKey, payload) => void
  onClose,
}) {
  if (!_dialogEl) {
    _dialogEl = createDialog();
    document.body.appendChild(_dialogEl);
  }

  await refreshList(lottery, currentKey, onSwitch);

  if (typeof _dialogEl.showModal === "function") _dialogEl.showModal();
  else _dialogEl.setAttribute("open", "");

  const handleClose = () => {
    onClose?.();
    _dialogEl.removeEventListener("close", handleClose);
  };
  _dialogEl.addEventListener("close", handleClose);
}

function createDialog() {
  const dlg = document.createElement("dialog");
  dlg.className = "model-manager-dialog";
  dlg.innerHTML = `
    <div class="mm-head">
      <strong>模型管理</strong>
      <span class="muted fine" id="mmQuota">—</span>
      <button class="btn ghost btn-sm" id="mmClose" type="button">关闭</button>
    </div>
    <div class="mm-actions">
      <input type="text" id="mmRenameKey" placeholder="新模型 key（如 my-best-run-2026-05）" />
      <button class="btn ghost btn-sm" id="mmDuplicateCurrent" type="button">复制当前为新 key</button>
    </div>
    <div id="mmList" class="mm-list">加载中…</div>
    <div class="mm-foot fine muted">
      存储在 IndexedDB（不会被清缓存抹掉）。删除前请先「下载备份」以防误删。
    </div>
  `;
  // 点击 backdrop 关闭
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) dlg.close();
  });
  dlg.querySelector("#mmClose").addEventListener("click", () => dlg.close());
  return dlg;
}

async function refreshList(lottery, currentKey, onSwitch) {
  const listEl = _dialogEl.querySelector("#mmList");
  const quotaEl = _dialogEl.querySelector("#mmQuota");
  listEl.textContent = "加载中…";

  try {
    const items = await modelStorage.list();
    const filtered = lottery
      ? items.filter((it) => !it.lottery || it.lottery === lottery)
      : items;

    if (filtered.length === 0) {
      listEl.innerHTML = `<div class="fine muted" style="padding:14px">暂无保存的模型。训练完成后点「保存到本地」会出现在这里。</div>`;
    } else {
      listEl.innerHTML = filtered.map((it) => {
        const date = it.savedAt ? it.savedAt.slice(0, 19).replace("T", " ") : "—";
        const arch = `${it.hiddenDim ?? "?"}H × ${it.numLayers ?? "?"}L${it.memberCount > 1 ? ` × ${it.memberCount} ensemble` : ""}`;
        const isCurrent = it.key === currentKey;
        return `
          <div class="mm-item ${isCurrent ? "is-current" : ""}" data-key="${escapeAttr(it.key)}">
            <div class="mm-meta">
              <strong class="mm-key">${escapeHtml(it.key)}${isCurrent ? `<span class="chip chip-ok" style="margin-left:8px">当前</span>` : ""}</strong>
              <span class="muted fine">${escapeHtml(it.type || "—")} · ${arch} · ${date}</span>
            </div>
            <div class="mm-row-actions">
              <button class="btn ghost btn-sm" data-act="switch">切换到此</button>
              <button class="btn ghost btn-sm" data-act="download">下载</button>
              <button class="btn ghost btn-sm" data-act="delete" style="color:var(--red-2)">删除</button>
            </div>
          </div>
        `;
      }).join("");

      // 绑定按钮
      listEl.querySelectorAll(".mm-item").forEach((row) => {
        const key = row.dataset.key;
        row.querySelector("[data-act='switch']").addEventListener("click", async () => {
          const payload = await modelStorage.load(key);
          if (payload && onSwitch) onSwitch(key, payload);
          _dialogEl.close();
        });
        row.querySelector("[data-act='download']").addEventListener("click", async () => {
          const payload = await modelStorage.load(key);
          if (payload) modelStorage.exportToFile(payload, `${key}.lottery.json`);
        });
        row.querySelector("[data-act='delete']").addEventListener("click", async () => {
          if (!confirm(`确定删除模型「${key}」？`)) return;
          await modelStorage.remove(key);
          await refreshList(lottery, currentKey, onSwitch);
        });
      });
    }

    // 配额
    const q = await modelStorage.getQuota();
    if (q) {
      const usedMB = (q.usage / 1024 / 1024).toFixed(1);
      const quotaMB = (q.quota / 1024 / 1024).toFixed(0);
      quotaEl.textContent = `已用 ${usedMB} / ${quotaMB} MB（${q.usagePercent.toFixed(2)}%）`;
    }

    // 复制当前 key
    _dialogEl.querySelector("#mmDuplicateCurrent").onclick = async () => {
      const newKey = (_dialogEl.querySelector("#mmRenameKey").value || "").trim();
      if (!newKey) {
        alert("请输入新 key");
        return;
      }
      if (!currentKey) {
        alert("没有当前模型可复制");
        return;
      }
      const payload = await modelStorage.load(currentKey);
      if (!payload) {
        alert("当前 key 没有数据");
        return;
      }
      await modelStorage.save(newKey, payload);
      _dialogEl.querySelector("#mmRenameKey").value = "";
      await refreshList(lottery, currentKey, onSwitch);
    };
  } catch (e) {
    listEl.innerHTML = `<div class="fine" style="color:var(--red-2);padding:14px">读取失败：${escapeHtml(e.message || String(e))}</div>`;
  }
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", "&#39;");
}
