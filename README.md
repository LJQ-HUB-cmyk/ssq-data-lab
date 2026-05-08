# 双色球数据实验室（静态站点）

本项目基于你上传的双色球历史数据解析生成 `data/draws.json`，并提供：
- 全量/近期频次可视化
- 冷热与遗漏统计
- “加权随机”号码生成器（娱乐用途，含免责声明）

## 如何运行（本地预览）

在项目根目录启动一个静态服务器即可：

```bash
python3 -m http.server 8000
```

然后用浏览器打开：

```
http://localhost:8000/
```

> 说明：直接用 `file://` 打开时，浏览器可能会阻止 `fetch('./data/draws.json')`，因此建议使用本地 HTTP 服务。
>
> 另外：本项目也内置了 `data/draws.js`（window.__SSQ_DATA__），即使直接双击打开 `index.html`，也能正常显示数据。

## 更新数据（可选）

### 方式A：重新生成全量历史（你提供 txt/xlsx）

如果你后续拿到新的历史数据 txt/xlsx，可以用自带脚本重新生成 JSON：

```bash
python3 tools/parse_ssq.py 你的历史数据.txt data/draws.json
```

### 方式B：自动补齐最新一期（GitHub Actions + 中国福彩网抓取）

仓库内提供 `tools/update_latest_ssq.ps1`，用于从中国福彩网开奖数据接口获取最新一期，并在发现新期号时自动更新：

- `data/draws.json`
- `data/draws.js`

配套的 GitHub Actions 工作流：`.github/workflows/update-data.yml`

- 定时：每天 09:00（北京时间）运行一次
- 手动：支持在 GitHub Actions 页面 `Run workflow` 立即运行
- 行为：无新数据则不提交；有新数据则自动提交并推送

本地手动运行（需要 PowerShell）：

```powershell
.\tools\update_latest_ssq.ps1
```

## 部署上线（GitHub Pages）

这是纯静态站点，发布到 GitHub Pages 的常见方式：

1. 在 GitHub 仓库设置里开启 Pages（Source 选择默认分支的根目录）
2. 访问生成的 Pages 地址即可
