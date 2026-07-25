// Cache-first service worker: after the first visit the app and the model
// work fully offline (weights are 3.3 MB, cached once).
const CACHE = "tinyllm-v2"; // bumped: chat UI
const ASSETS = [
  ".", "index.html", "app.js", "tinygpt.js", "manifest.webmanifest",
  "apple-touch-icon.png",
  "model/config.json", "model/vocab.json", "model/manifest.json",
  "model/weights.bin",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
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
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(
      (hit) => hit ?? fetch(e.request)
    )
  );
});
