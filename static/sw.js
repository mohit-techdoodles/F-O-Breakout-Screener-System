const CACHE_NAME = 'fo-screener-pwa-v1';
const urlsToCache = [
  '/',
  '/app/static/manifest.json',
  '/app/static/icon-192.png',
  '/app/static/icon-512.png',
  '/app/static/apple-touch-icon.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache).catch(err => console.log('PWA cache prefetch note:', err));
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  // Network first for real-time market data, fallback to cache for shell assets
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
