// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The manifest is what makes the window installable (docs/SPEC.md section 4 and
// 21.2), and test 19 checks it against the installability criteria: icons at
// 192 and 512, `display: standalone`, and a `start_url` inside `scope`.
//
// `id` is fixed and absolute on purpose. It is what the browser uses to decide
// whether an install is *this* app or a new one; deriving it from the URL would
// make a run on --port 9000 into a second, different application in the
// browser's eyes.
import type { ManifestOptions } from 'vite-plugin-pwa'

export const manifest: Partial<ManifestOptions> = {
  id: '/?app=autophotoedit',
  name: 'autoPhotoEdit',
  short_name: 'autoPhotoEdit',
  description:
    'Post-produzione automatica di file RAW Sony, in locale e non distruttiva.',
  lang: 'it',
  dir: 'ltr',
  start_url: '/',
  scope: '/',
  display: 'standalone',
  display_override: ['window-controls-overlay', 'standalone'],
  orientation: 'landscape',
  // Neutral greys: section 10 forbids saturated colour anywhere near the photos,
  // and the install splash screen is as near as it gets.
  theme_color: '#111111',
  background_color: '#111111',
  categories: ['photo', 'graphics', 'productivity'],
  icons: [
    { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
    { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
    {
      src: '/icons/icon-512-maskable.png',
      sizes: '512x512',
      type: 'image/png',
      purpose: 'maskable',
    },
    { src: '/icons/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
  ],
}
