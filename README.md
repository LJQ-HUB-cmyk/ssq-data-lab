# 双色球数据实验室 · SSQ Data Lab

[English](./README.en.md) · **中文**

一个**诚实**的双色球历史数据分析站点：用 3400+ 期真实开奖数据做可视化、分布分析、卡方检验与娱乐性号码推荐。**不预测、不承诺中奖、不引导购彩**。

> 彩票是独立同分布的随机事件。历史频率无法预测未来——这个项目就是要用数据本身证明这一点。

## 特性

- **基本走势图**：近 30/50/100 期的红球、蓝球命中点阵
- **冷热 / 遗漏分析**：频次柱状图 + 遗漏期数
- **分布分析**：奇偶比、大小比、质合比、012 路、三区比、AC 值、和值、跨度的历史分布
- **卡方拟合优度检验**：用统计方法验证"均匀分布"假设，p 值实时计算（差异化功能）
- **加权随机生成器**：热/冷/混合/均匀策略 × 和值/奇偶/跨度/分区约束
- **胆拖 / 复式注数试算**：C(n, k) 实时计算
- **数据源**：500.com 历史接口，每周一/三/五自动拉取最近 30 期合并

**技术栈**：原生 ES modules + SVG，零构建；Python stdlib 抓取；Node.js / unittest 测试。**无任何 runtime 依赖**。

## Live Demo

<!-- TODO: GitHub Pages / Cloudflare Pages 上线后替换 -->

- GitHub Pages：`https://<username>.github.io/ssq-data-lab/`
- 国内镜像（Cloudflare Pages）：`https://ssq-data-lab.pages.dev/`

> 还没部署？见 [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)，配好 CI 后 push 即发布。

## 快速开始

```bash
# 克隆
git clone https://github.com/<username>/ssq-data-lab
cd ssq-data-lab

# 起服务器
npm run serve
# → http://localhost:5173/

# 跑测试
npm test           # 前端单测（61 个）
npm run test:py    # 抓取脚本（15 个）

# 抓最新数据
npm run update-data
```

> 要求：Node.js ≥ 18、Python ≥ 3.10。

## 目录结构

```
ssq-data-lab/
├── index.html                  单页入口，8 个 Tab
├── assets/
│   ├── styles.css
│   └── js/
│       ├── main.js             生命周期 + 事件
│       ├── data.js             fetch + window.__SSQ_DATA__ 兜底
│       ├── stats.js            频次 / 遗漏 / TopN
│       ├── distribution.js     分布分析（奇偶/大小/012路/AC 值...）
│       ├── chi-square.js       卡方检验 + 不完全伽马 p 值
│       ├── combinatorics.js    胆拖 / 复式 / C(n,k)
│       ├── trend.js            走势矩阵
│       ├── trend-chart.js      走势点阵 SVG
│       ├── chart.js            频次柱状图 SVG
│       ├── generator.js        加权采样
│       ├── ui.js               DOM 渲染
│       └── utils.js            $ / pad2 / clamp / ...
├── data/
│   ├── draws.json              3450+ 期主数据
│   └── draws.js                window.__SSQ_DATA__ 等价副本
├── tools/
│   ├── parse_ssq.py            txt / xlsx → draws.json
│   ├── update_ssq.py           500.com 抓取 + 合并
│   └── fixtures/               离线测试 HTML
├── tests/
│   ├── stats.test.mjs
│   ├── generator.test.mjs
│   ├── distribution.test.mjs
│   ├── chi-square.test.mjs
│   ├── combinatorics.test.mjs
│   └── test_update_ssq.py
├── docs/
│   └── DEPLOYMENT.md
└── .github/
    ├── workflows/
    │   ├── update-data.yml     数据自动更新
    │   └── pages.yml           GitHub Pages 部署
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

## 开发

```bash
# 前端测试
npm test

# 抓取脚本测试
npm run test:py

# 手动抓一次
python tools/update_ssq.py --count 30

# 离线跑（不发网络请求）
python tools/update_ssq.py --fixture tools/fixtures/500_history.html

# 重新解析历史 txt/xlsx
python tools/parse_ssq.py 历史.txt data/draws.json
```

## 部署

见 [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)。推荐 **GitHub Pages + Cloudflare Pages 双通道**（覆盖海外 + 国内）。

## 贡献

欢迎 PR。先读 [CONTRIBUTING.md](./CONTRIBUTING.md)。

**有些东西永远不会被合入**：
- "AI/LSTM/神经网络预测下一期" —— 这是伪命题，项目底线
- 任何引导未成年人购彩的功能
- 封闭式爬虫（数据源必须可替换）

## 免责声明

- 彩票为随机事件，历史统计**无法**预测未来
- 本工具仅输出"加权随机"建议，**不提高**中奖概率
- 理性消费、量力而行；**未成年人禁止参与**
- 本项目不从购彩行为中获取任何直接或间接收益

## License

[MIT](./LICENSE)
