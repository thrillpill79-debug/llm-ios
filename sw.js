// Service worker.
//
// Strategy matters here: the app shell (HTML/JS) is fetched network-first so a
// deploy reaches users immediately, falling back to cache when offline. Large
// immutable assets (the WASM runtime, the tiny model's weights) are cache-first
// because they never change without a filename change and re-downloading them
// would be wasteful. GGUF models are managed by wllama itself (origin-private
// filesystem), not by this cache.
const CACHE = "pocketgpt-v4";

// Turbo mode: GitHub Pages cannot send COOP/COEP, so when registered as
// "sw.js?coi=1" we add those headers to our own responses. That makes the page
// cross-origin isolated, which is what enables SharedArrayBuffer and therefore
// multi-threaded inference. Cross-origin model downloads are unaffected: they
// are CORS requests, which remain allowed under require-corp.
const COI = new URL(self.location).searchParams.get("coi") === "1";

function withIsolation(response) {
  if (!COI || !response || response.type === "opaque") return response;
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

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
        .then(withIsolation)
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
      return withIsolation(fresh);
    } catch {
      const hit = await caches.match(e.request, { ignoreSearch: true });
      if (hit) return withIsolation(hit);
      throw new Error("offline and not cached");
    }
  })());
});
