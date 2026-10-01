// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Messages that come and go in a corner: what failed, in the server's own
// sentence (ApiError carries it). Every mutation that has no error handling of
// its own lands here (main.tsx), so a failed save is never silent.
import { create } from 'zustand'

export interface Toast {
  id: number
  message: string
  tone: 'error' | 'info'
}

interface ToastState {
  toasts: Toast[]
  push: (message: string, tone?: Toast['tone']) => void
  dismiss: (id: number) => void
}

let counter = 0

/** How long a message stays; an error a little longer, it may need reading twice. */
const LIFETIME_MS = { info: 4_000, error: 8_000 }

export const useToasts = create<ToastState>()((set, get) => ({
  toasts: [],
  push: (message, tone = 'error') => {
    // The same failure repeated (a slider sending one bad value per frame) is one message.
    if (get().toasts.some((toast) => toast.message === message)) return
    counter += 1
    const id = counter
    set({ toasts: [...get().toasts.slice(-3), { id, message, tone }] })
    window.setTimeout(() => get().dismiss(id), LIFETIME_MS[tone])
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((toast) => toast.id !== id) }),
}))

export function toast(message: string, tone: Toast['tone'] = 'error'): void {
  useToasts.getState().push(message, tone)
}
