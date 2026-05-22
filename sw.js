// SSQ Data Lab · 极简 service worker
//
// 设计目标：
//   1. 离线可用：缓存 app shell（HTML/CSS/JS）
//   2. 数据保鲜：data/draws.json 走 stale-while-revalidate，离线时回退缓存
//   3. 不缓存第三方：仅同源资源
//
// 升级策略：
//   - 改 CACHE_VERSION 即可使旧缓存失效
//   - 不主动跳过 waiting，避免在用户刷新时切版本

const CACHE_VERSION = "ssq-lab-v2";
const APP_SHELL = [
  "./",
  "./index.html",
  "./assets/styles.css",
  "./assets/js/main.js",
  "./assets/js/data.js",
  "./assets/js/utils.js",
  "./assets/js/stats.js",
  "./assets/js/distribution.js",
  "./assets/js/chi-square.js",
  "./assets/js/combinatorics.js",
  "./assets/js/generator.js",
  "./assets/js/chart.js",
  "./assets/js/trend.js",
  "./assets/js/trend-chart.js",
  "./assets/js/ui.js",
  "./assets/js/miss-stats.js",
  "./assets/js/cooccurrence.js",
  "./assets/js/timeseries.js",
  "./assets/js/countdown.js",
  "./assets/js/rng.js",
  "./assets/js/bayes.js",
  "./assets/js/dpp.js",
  "./assets/js/mcmc.js",
  "./assets/js/distance.js",
  "./assets/js/advanced-sampler.js",
  "./manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(APP_SHELL))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  // 仅处理同源 GET
  if (req.method !== "GET" || url.origin !== self.location.origin) return;

  // 数据走 stale-while-revalidate
  if (url.pathname.endsWith("/data/draws.json")) {
    event.respondWith(staleWhileRevalidate(req));
    return;
  }
  // 其他静态资源走 cache-first
  event.respondWith(cacheFirst(req));
});

async function cacheFirst(req) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(req);
  if (cached) return cached;
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (err) {
    // 离线兜底：返回 index.html
    if (req.mode === "navigate") {
      return cache.match("./index.html");
    }
    throw err;
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(req);
  const network = fetch(req)
    .then((res) => {
      if (res.ok) cache.put(req, res.clone());
      return res;
    })
    .catch(() => null);
  return cached || (await network) || new Response("offline", { status: 503 });
}
