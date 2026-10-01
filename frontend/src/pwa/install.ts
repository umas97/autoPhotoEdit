// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Installa come app" (docs/SPEC.md section 21.2). The button exists only when the
// browser has offered us the prompt and we are not already running as an
// installed app: offering to install something that is installed is how an
// application ends up with two entries in the menu, and section 21.2 forbids
// the app from ever creating a second one.
export type InstallState = {
  /** The browser offered an install prompt and we are holding it. */
  available: boolean
  /** Already running in its own window: nothing to install, nothing to say. */
  standalone: boolean
}

type InstallPromptEvent = Event & {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

let deferred: InstallPromptEvent | null = null
const listeners = new Set<(state: InstallState) => void>()

export function isStandalone(): boolean {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    window.matchMedia('(display-mode: window-controls-overlay)').matches ||
    // iOS Safari, and the flag Chromium sets for an --app window.
    (window.navigator as { standalone?: boolean }).standalone === true
  )
}

function state(): InstallState {
  return { available: deferred !== null && !isStandalone(), standalone: isStandalone() }
}

function publish(): void {
  const current = state()
  listeners.forEach((listener) => listener(current))
}

window.addEventListener('beforeinstallprompt', (event) => {
  // Keeping the event is what lets the button appear where it belongs instead
  // of in the browser's own chrome -- which, in the dedicated window, is not
  // visible at all.
  event.preventDefault()
  deferred = event as InstallPromptEvent
  publish()
})

window.addEventListener('appinstalled', () => {
  deferred = null
  publish()
})

export function subscribeInstall(listener: (state: InstallState) => void): () => void {
  listeners.add(listener)
  listener(state())
  return () => listeners.delete(listener)
}

export function installState(): InstallState {
  return state()
}

/** Show the browser's install prompt. Resolves to true if the user accepted. */
export async function promptInstall(): Promise<boolean> {
  if (!deferred) return false
  await deferred.prompt()
  const choice = await deferred.userChoice
  deferred = null
  publish()
  return choice.outcome === 'accepted'
}
