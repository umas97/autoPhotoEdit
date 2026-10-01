// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The 150 ms of section 10, from the browser's side.
//
// Three rules, and they are the whole hook:
//
//   1. **one request in flight, ever.** A dragged slider produces new parameter
//      states faster than any renderer answers. Queueing them means the picture
//      lags further behind the handle the longer the drag lasts; aborting the
//      previous one means it is always at most one render behind. The last
//      state is the only one anybody wants to see.
//   2. **the size follows the gesture.** 1024 px while the handle is down,
//      2048 px when it is released -- decided by the server, which decodes at
//      the size asked for rather than resizing at the end (see deps.py).
//   3. **the previous frame stays up** until the next one has decoded. Blanking
//      the image between renders makes a smooth drag look like a flicker.
//
// The blob URLs are revoked as they are replaced. Without that, a minute of
// dragging leaks a few hundred megabytes of JPEG into the tab.
import { useCallback, useEffect, useRef, useState } from 'react'
import { renderPreview } from './api'
import type { EditParams } from './types'

export const DRAG_EDGE = 1024
export const IDLE_EDGE = 2048

export interface PreviewState {
  /** Object URL of the current render, or null before the first one. */
  url: string | null
  rendering: boolean
  /** Milliseconds the last render took, round trip included. */
  elapsedMs: number | null
  edge: number
  error: string | null
}

export interface PreviewOptions {
  /** "Mostra rimozioni": false renders without them, parameters untouched. */
  retouch?: boolean
  /**
   * Bumped when something the server resolves changed under the same
   * parameters -- an eraser's fill arrived (components/retouch/useRetouch.ts).
   */
  revision?: number
}

export function usePreview(
  photoId: number | null,
  params: EditParams | null,
  dragging: boolean,
  options: PreviewOptions = {},
): PreviewState {
  const [state, setState] = useState<PreviewState>({
    url: null,
    rendering: false,
    elapsedMs: null,
    edge: IDLE_EDGE,
    error: null,
  })

  const controller = useRef<AbortController | null>(null)
  const urlRef = useRef<string | null>(null)
  const lastKey = useRef<string>('')

  const run = useCallback(
    async (id: number, body: EditParams, edge: number, retouch: boolean) => {
      controller.current?.abort()
      const abort = new AbortController()
      controller.current = abort
      const started = performance.now()
      setState((previous) => ({ ...previous, rendering: true, error: null }))
      try {
        const blob = await renderPreview(id, body, edge, abort.signal, retouch)
        const url = URL.createObjectURL(blob)
        if (urlRef.current) URL.revokeObjectURL(urlRef.current)
        urlRef.current = url
        setState({
          url,
          rendering: false,
          elapsedMs: Math.round(performance.now() - started),
          edge,
          error: null,
        })
      } catch (error) {
        if ((error as Error).name === 'AbortError') return
        setState((previous) => ({
          ...previous,
          rendering: false,
          error: (error as Error).message,
        }))
      }
    },
    [],
  )

  const retouch = options.retouch ?? true
  const revision = options.revision ?? 0
  useEffect(() => {
    if (photoId === null || params === null) return
    const edge = dragging ? DRAG_EDGE : IDLE_EDGE
    const key = `${photoId}:${edge}:${retouch}:${revision}:${JSON.stringify(params)}`
    // Re-rendering identical parameters at the same size would cost a second of
    // CPU to produce the bytes already on screen.
    if (key === lastKey.current) return
    lastKey.current = key
    void run(photoId, params, edge, retouch)
  }, [photoId, params, dragging, retouch, revision, run])

  useEffect(
    () => () => {
      controller.current?.abort()
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    },
    [],
  )

  return state
}
