// Lottery Data Lab · service worker
//
// 设计目标：
//   1. 离线可用：缓存 app shell（HTML/CSS/JS）
//   2. 数据保鲜：data/draws.json / data/dlt-draws.json 走 stale-while-revalidate
//   3. 不缓存第三方：仅同源资源
//
// 关键陷阱（v9 修复）：
//   ── 不能把"被重定向过的 Response"放进 cache。──
//   按 Fetch 规范，SW 把 redirected==true 的响应返回给浏览器会触发 TypeError，
//   表现为 ERR_FAILED。这里的踩坑场景：
//     APP_SHELL 里写了 "./dlt.html"，但 Cloudflare Pages 把 /dlt.html 308 →
//     /dlt，于是 cache.add("/dlt.html") 缓存到的是 redirected=true 的响应。
//     第二次访问 dlt.html 时 SW 命中缓存返回，浏览器拒绝并抛 ERR_FAILED。
//
//   修复：
//     a) 预缓存阶段用 `redirect: "follow"` + 重建 Response（去掉 redirected 标记）
//     b) 运行时对 navigation 请求用 network-first（永远拿到正确的最终 URL 文档）
//     c) 命中缓存时检查 res.redirected，如发现就丢弃并回源

const CACHE_VERSION = "ssq-lab-v9";
const APP_SHELL = [
  "./",
  "./index.html",
  "./dlt.html",
  "./assets/styles.css",
  "./assets/dlt-styles.css",
  "./assets/js/main.js",
  "./assets/js/dlt-main.js",
  "./assets/js/data.js",
  "./assets/js/dlt-data.js",
  "./assets/js/utils.js",
  "./assets/js/lottery-config.js",
  "./assets/js/lottery-stats.js",
  "./assets/js/stats.js",
  "./assets/js/distribution.js",
  "./assets/js/dlt-distribution.js",
  "./assets/js/chi-square.js",
  "./assets/js/dlt-chi-square.js",
  "./assets/js/combinatorics.js",
  "./assets/js/dlt-combinatorics.js",
  "./assets/js/generator.js",
  "./assets/js/dlt-generator.js",
  "./assets/js/chart.js",
  "./assets/js/dlt-chart.js",
  "./assets/js/trend.js",
  "./assets/js/trend-chart.js",
  "./assets/js/dlt-trend-chart.js",
  "./assets/js/ui.js",
  "./assets/js/dlt-ui.js",
  "./assets/js/miss-stats.js",
  "./assets/js/cooccurrence.js",
  "./assets/js/dlt-cooccurrence.js",
  "./assets/js/timeseries.js",
  "./assets/js/dlt-timeseries.js",
  "./assets/js/countdown.js",
  "./assets/js/dlt-countdown.js",
  "./assets/js/rng.js",
  "./assets/js/bayes.js",
  "./assets/js/dpp.js",
  "./assets/js/mcmc.js",
  "./assets/js/distance.js",
  "./assets/js/advanced-sampler.js",
  "./assets/js/dlt-advanced-sampler.js",
  "./assets/js/dlt-backtest.js",
  "./assets/js/dlt-prize.js",
  "./assets/js/dlt-independence.js",
  "./assets/js/dlt-explainer.js",
  "./assets/js/dlt-chase.js",
  "./assets/js/dlt-nn-model.js",
  "./assets/js/dlt-nn-trainer.js",
  "./assets/js/dlt-nn-backtest.js",
  "./assets/js/dlt-lstm-controller.js",
  "./assets/js/nn-math.js",
  "./assets/js/nn-optim.js",
  "./assets/js/nn-lstm.js",
  "./assets/js/nn-stack.js",
  "./assets/js/nn-ssq-model.js",
  "./assets/js/nn-trainer.js",
  "./assets/js/nn-backtest.js",
  "./assets/js/nn-statistics.js",
  "./assets/js/nn-ensemble.js",
  "./assets/js/lstm-controller.js",
  "./manifest.webmanifest",
];

/** 取一个 URL 并返回去掉 redirected 标记的 Response（如果发生过 redirect）。 */
async function fetchAndStrip(url) {
  const res = await fetch(url, { redirect: "follow", credentials: "same-origin" });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  if (!res.redirected) return res;
  // 重建一个干净的 Response，cache 里不会再出现 redirected: true 的条目
  const body = await res.blob();
  return new Response(body, {
    status: res.status,
    statusText: res.statusText,
    headers: res.headers,
  });
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_VERSION);
      // 不能用 cache.addAll：它会把 redirected 的响应直接放进去
      await Promise.all(
        APP_SHELL.map(async (url) => {
          try {
            const res = await fetchAndStrip(url);
            await cache.put(url, res);
          } catch (e) {
            // 单个文件失败不阻塞 SW 安装
            console.warn("[sw] precache failed:", url, e.message);
          }
        })
      );
      // 立刻接管，避免用户刷新时还在用旧 SW
      self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;

  // 1) 数据接口：stale-while-revalidate
  if (url.pathname.endsWith("/data/draws.json") || url.pathname.endsWith("/data/dlt-draws.json")) {
    event.respondWith(staleWhileRevalidate(req));
    return;
  }

  // 2) HTML 文档（navigation）：network-first
  //    避免被 308 重定向污染过的旧缓存卡住后续访问
  if (req.mode === "navigate" || (req.destination === "document")) {
    event.respondWith(networkFirstNavigation(req));
    return;
  }

  // 3) 其他静态资源：cache-first
  event.respondWith(cacheFirst(req));
});

/**
 * Navigation 请求：优先走网络。一旦网络成功，把剥离 redirected 后的副本放进缓存。
 * 网络失败时再回退到缓存（也再做一次 redirected 防御）。
 */
async function networkFirstNavigation(req) {
  const cache = await caches.open(CACHE_VERSION);
  try {
    const res = await fetch(req, { redirect: "follow" });
    // 把"最终 URL（重定向后）"和"原始请求 URL"都缓存上一份干净副本
    if (res.ok) {
      const cleaned = res.redirected
        ? new Response(await res.clone().blob(), {
            status: res.status,
            statusText: res.statusText,
            headers: res.headers,
          })
        : res.clone();
      // 用原始 request 作为 key，而不是重定向后的 URL，否则下次同 URL 还是命不中
      cache.put(req, cleaned).catch(() => {});
    }
    return res;
  } catch (err) {
    const cached = await cache.match(req);
    if (cached && !cached.redirected) return cached;
    // 兜底：返回首页
    const home = await cache.match("./index.html");
    if (home && !home.redirected) return home;
    return new Response("offline", { status: 503, headers: { "Content-Type": "text/plain;charset=utf-8" } });
  }
}

async function cacheFirst(req) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(req);
  if (cached && !cached.redirected) return cached;
  // 缓存里是被污染的（redirected=true）→ 删掉重新拉
  if (cached && cached.redirected) {
    await cache.delete(req);
  }
  try {
    const res = await fetch(req, { redirect: "follow" });
    if (res.ok) {
      const toCache = res.redirected
        ? new Response(await res.clone().blob(), {
            status: res.status,
            statusText: res.statusText,
            headers: res.headers,
          })
        : res.clone();
      cache.put(req, toCache).catch(() => {});
    }
    return res;
  } catch (err) {
    if (req.mode === "navigate") {
      const home = await cache.match("./index.html");
      if (home && !home.redirected) return home;
    }
    throw err;
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(req);
  const network = fetch(req, { redirect: "follow" })
    .then(async (res) => {
      if (res.ok) {
        const toCache = res.redirected
          ? new Response(await res.clone().blob(), {
              status: res.status,
              statusText: res.statusText,
              headers: res.headers,
            })
          : res.clone();
        cache.put(req, toCache).catch(() => {});
      }
      return res;
    })
    .catch(() => null);
  if (cached && !cached.redirected) return cached;
  return (await network) || new Response("offline", { status: 503 });
}
