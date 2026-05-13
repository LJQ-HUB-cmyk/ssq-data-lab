# SSQ Data Lab · 双色球数据实验室

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen?style=flat-square)](https://wanghao137.github.io/ssq-data-lab/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-61%20passed-success?style=flat-square)](./tests)
[![No build step](https://img.shields.io/badge/build-zero--config-informational?style=flat-square)](./index.html)

**English** · [中文](./README.md)

An **honest** data lab for Union Lotto (SSQ, 双色球): 3,400+ historical draws analysed with visualisations, distribution stats, chi-square goodness-of-fit testing, and a recreational number generator. **No prediction. No promises. Not a gambling tool.**

> Lottery draws are i.i.d. random events. Historical frequencies cannot predict the future. This project exists to *demonstrate* that point with data.

## Live Demo

- **GitHub Pages**: <https://wanghao137.github.io/ssq-data-lab/>
- **Cloudflare Pages (China-friendly mirror)**: coming soon — see [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)

## Features

- **Trend charts** — dot-matrix for the last 30/50/100 draws (red + blue)
- **Hot/cold & miss analysis** — frequency bars + draws-since-last-seen
- **Distribution analysis** — odd/even, high/low, prime/composite, 012-paths, zone ratios, AC value, sum, span
- **Chi-square goodness-of-fit test** — live p-values on the null hypothesis that draws are uniformly random (the differentiator)
- **Weighted random generator** — hot/cold/mixed/uniform × sum/odd/span/zone constraints
- **Dan-Tuo / complex ticket calculator** — live C(n, k)
- **Data pipeline** — Python stdlib scraper from 500.com, auto-updated Mon/Wed/Fri via GitHub Actions

**Stack**: vanilla ES modules + SVG, **zero build**; Python stdlib for scraping; `node --test` / `unittest` for tests. **No runtime dependencies.**

See [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) to deploy your own.

## Quick Start

```bash
git clone https://github.com/wanghao137/ssq-data-lab
cd ssq-data-lab

npm run serve           # http://localhost:5173/
npm test                # 61 front-end tests
npm run test:py         # 15 scraper tests
npm run update-data     # pull latest draws from 500.com
```

Requires Node.js ≥ 18 and Python ≥ 3.10.

## Why This Project Is Different

Every "lottery analyser" on GitHub falls into one of two buckets:

1. **Trend-chart toys** — copies of the `datachart.500.com` layout; no statistical rigor.
2. **"AI prediction" projects** — LSTMs predicting lottery draws. A statistical impossibility sold as cleverness.

This project is deliberately neither. It **uses real statistics to show there is nothing to predict** — the chi-square panel computes p-values in real time and tells you whether you can reject the null hypothesis that draws are uniform. On 3,400 draws the answer is: **blue balls are indistinguishable from uniform, red balls have marginal deviation consistent with hardware noise, neither is predictive.**

That's the message. The generator is explicitly framed as "weighted random dice-roll for fun" — it does not lower the odds.

## Project Structure

See [README 中文版](./README.md#目录结构) for a file-tree map.

## Contributing

PRs welcome — read [CONTRIBUTING.md](./CONTRIBUTING.md) first.

**Off-limits forever**:
- "AI predicts next draw" (not because we haven't tried — because it's mathematically impossible to predict i.i.d. draws)
- Features that facilitate gambling by minors
- Closed-source scrapers without a replaceable source abstraction

## Disclaimer

- Lottery draws are random. Historical frequencies do not predict the future.
- This tool produces **randomly-weighted suggestions** that do **not** improve your odds.
- Play responsibly. Not for minors. Not legal advice. Not financial advice.
- The authors do not profit from any purchase you make based on this tool's output.

## License

[MIT](./LICENSE)
