// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The build writes straight into the Python package (docs/SPEC.md section 4): in
// production there is no second process, uvicorn serves these files from the
// same port that answers /api. In development `vite dev` proxies /api and /ws
// to the server, so the two setups differ in who serves the HTML and in nothing
// else.
//
// The service worker is built from src/pwa/sw.ts with `injectManifest`, not
// generated: section 4 forbids caching /api, /ws and the proxy images, and
// "never intercept" is a rule about what the worker does *not* do -- easier to
// write by hand, and easier for test 19 to read.
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { writeFileSync } from 'node:fs'
import { VitePWA } from 'vite-plugin-pwa'
import { manifest } from './src/pwa/manifest'

// The build directory is a package directory that git keeps empty (§18.3 puts
// the built interface in release tags, not in the repository). Emptying it on
// every build would delete the placeholder that keeps it in the tree.
const keepDirectory = {
  name: 'ape-keep-static-dir',
  closeBundle() {
    writeFileSync(fileURLToPath(new URL('../backend/ape/static/.gitkeep', import.meta.url)), '')
  },
}

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src/pwa',
      filename: 'sw.ts',
      registerType: 'autoUpdate',
      injectRegister: 'script-defer',
      manifest,
      injectManifest: {
        // The shell, and only the shell. Photographs are never precached: they
        // are megabytes each and they live on the server (section 4).
        globPatterns: ['**/*.{js,css,html,woff2,png,svg,webmanifest}'],
        globIgnores: ['**/node_modules/**', 'offline.html'],
      },
      devOptions: { enabled: false },
    }),
    keepDirectory,
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8787', changeOrigin: false },
      '/ws': { target: 'ws://127.0.0.1:8787', ws: true },
    },
  },
  build: {
    outDir: '../backend/ape/static',
    emptyOutDir: true,
    sourcemap: false,
  },
})
