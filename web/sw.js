// Cache-first service worker. The app shell and the tiny Shakespeare model are
// precached so everything works offline; the wllama WASM runtime is cached on
// first use. GGUF models are managed by wllama itself (origin-private
// filesystem), not by this cache.
const CACHE = "pocketgpt-v3";
const ASSETS = [
  ".", "index.html", "pocketgpt.js", "manifest.webmanifest", "apple-touch-icon.png",
  "shakespeare.html", "shakespeare.js", "tinygpt.js",
  "vendor/wllama/index.js", "vendor/wllama/wllama.wasm",
  "model/config.json", "model/vocab.json", "model/manifest.json", "model/weights.bin",
];

self.addEventListener("install", (e) => {
  // individual failures must not abort the whole install
  e.waitUntil(caches.open(CACHE).then((c) =>
    Promise.all(ASSETS.map((a) => c.add(a).catch(() => {})))
  ));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // only serve same-origin assets from cache; model downloads and any other
  // cross-origin traffic go straight to the network
  if (url.origin !== self.location.origin) return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) => hit ?? fetch(e.request))
  );
});
