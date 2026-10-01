// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The service worker caches the application shell and nothing else (docs/SPEC.md
// section 4). Written by hand rather than generated, because the important part
// is a list of things it must *not* do, and test 19 checks exactly that:
//
//   * /api/** and /ws are never intercepted. The state lives on the server, and
//     a stale answer would show the user a reality that is not there. Two reads
//     of the same endpoint, with the server changed in between, must differ;
//   * the proxy images are never cached here either. They are megabytes each,
//     the server already marks them immutable, and the browser's own HTTP cache
//     does the job without a quota of ours to manage;
//   * only same-origin GET navigations and shell assets go through the cache.
//
// When the server is down, a navigation gets the offline page of section 4
// instead of the browser's network error -- the one moment where answering from
// the cache tells the truth: the program is not running.
/// <reference lib="webworker" />
import offlineDocument from './offline.html?raw'

// `self.__WB_MANIFEST` is replaced at build time with the list of shell files,
// each with a revision. The literal spelling matters: workbox finds the
// injection point by searching the built worker for it.
declare const self: ServiceWorkerGlobalScope & {
  __WB_MANIFEST: Array<{ url: string; revision: string | null }>
}

const SHELL_CACHE = 'ape-shell-v1'
const SHELL_FILES = self.__WB_MANIFEST

/** Paths the worker keeps its hands off entirely (section 4). */
function isLiveData(url: URL): boolean {
  return (
    url.pathname.startsWith('/api/') ||
    url.pathname === '/api' ||
    url.pathname === '/ws' ||
    url.pathname.startsWith('/ws/')
  )
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE)
      await cache.addAll(SHELL_FILES.map((entry) => entry.url))
      await self.skipWaiting()
    })(),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys()
      await Promise.all(names.filter((name) => name !== SHELL_CACHE).map((n) => caches.delete(n)))
      await self.clients.claim()
    })(),
  )
})

self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') void self.skipWaiting()
})

self.addEventListener('fetch', (event) => {
  const request = event.request
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return

  // Not ours to answer. Returning without calling respondWith leaves the
  // request to the network exactly as if no worker existed.
  if (isLiveData(url)) return

  // A document request: the shell if we have it, the offline page if the server
  // is gone. The router inside the shell resolves the path (section 21.2).
  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          return await fetch(request)
        } catch {
          const cached = await caches.match('/index.html')
          if (cached) return cached
          return new Response(offlineDocument, {
            status: 503,
            headers: { 'Content-Type': 'text/html; charset=utf-8' },
          })
        }
      })(),
    )
    return
  }

  // Shell assets are content-hashed, so a hit is always the right bytes.
  event.respondWith(
    (async () => {
      const cached = await caches.match(request)
      if (cached) return cached
      const response = await fetch(request)
      if (response.ok && response.type === 'basic' && isShellAsset(url)) {
        const cache = await caches.open(SHELL_CACHE)
        cache.put(request, response.clone())
      }
      return response
    })(),
  )
})

/** Hashed build output and icons: everything the shell is made of. */
function isShellAsset(url: URL): boolean {
  return (
    url.pathname.startsWith('/assets/') ||
    url.pathname.startsWith('/icons/') ||
    url.pathname.endsWith('.webmanifest')
  )
}
