// Service worker.
//
// Strategy matters here: the app shell (HTML/JS) is fetched network-first so a
// deploy reaches users immediately, falling back to cache when offline. Large
// immutable assets (the WASM runtime, the tiny model's weights) are cache-first
// because they never change without a filename change and re-downloading them
// would be wasteful. GGUF models are managed by wllama itself (origin-private
// filesystem), not by this cache.
const CACHE = "pocketgpt-v4";

const SHELL = [
  ".", "index.html", "pocketgpt.js", "manifest.webmanifest",
  "shakespeare.html", "shakespeare.js", "tinygpt.js",
];
const HEAVY = [
  "apple-touch-icon.png",
  "vendor/wllama/index.js", "vendor/wllama/wllama.wasm",
  "model/config.json", "model/vocab.json", "model/manifest.json", "model/weights.bin",
];

const isHeavy = (url) =>
  /\.(wasm|bin|png)$/.test(url.pathname) || url.pathname.includes("/model/");

self.addEventListener("install", (e) => {
  // individual failures must not abort the whole install
  e.waitUntil(caches.open(CACHE).then((c) =>
    Promise.all([...SHELL, ...HEAVY].map((a) => c.add(a).catch(() => {})))
  ));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // model downloads and any other cross-origin traffic go straight to network
  if (url.origin !== self.location.origin) return;
  if (e.request.method !== "GET") return;

  if (isHeavy(url)) {
    e.respondWith(
      caches.match(e.request, { ignoreSearch: true })
        .then((hit) => hit ?? fetch(e.request))
    );
    return;
  }

  // app shell: network first, cache as offline fallback
  e.respondWith((async () => {
    try {
      const fresh = await fetch(e.request, { cache: "no-cache" });
      if (fresh && fresh.ok) {
        const copy = fresh.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
      }
      return fresh;
    } catch {
      const hit = await caches.match(e.request, { ignoreSearch: true });
      if (hit) return hit;
      throw new Error("offline and not cached");
    }
  })());
});
