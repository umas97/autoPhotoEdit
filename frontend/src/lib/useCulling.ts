// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The culling screen's connection to the server.
//
// Section 14 asks that changing mode or slider update the selection "without
// perceptible reanalysis (< 100 ms)". The server's half of that is a pure
// recomputation over stored scores that answers with only the photos whose
// outcome changed (backend/ape/culling/select.py). This hook's half is keeping
// that partial answer meaningful.
//
// **Writes are serialised, never cancelled.** The server diffs against what it
// has stored, not against what the screen shows, and cancelling a request in
// the browser does not stop the server from applying it: an aborted request's
// changes would be written and never shown, and the next answer would not
// repeat them. So one write is in flight at a time, every answer is applied in
// order, and settings that arrive while one is in flight are merged and sent
// as one when it returns -- a dragged slider still costs one round trip at a
// time, and the screen is never more than one step behind the hand.
//
// **A poll never overwrites a write.** While photos are being analysed the
// whole view is refetched every two seconds. A refetch that started before a
// write committed would put the old outcome back on screen, so each fetch
// carries the write generation it started under and is dropped if a write has
// begun since.
//
// **The count is the check.** After every write the bottom-bar count from the
// server is compared with the cards; if they disagree -- another window, a
// lost answer -- the whole view is fetched again.
//
// Undo is a stack of batches, each holding what its photos were before. A
// batch because the gestures of section 7.6 come in pairs -- a new proposed
// frame for a burst keeps one photo and releases another -- and one Ctrl+Z
// must undo the pair.
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ApiError, updateCullSettings } from './api'
import { applySelection, currentDecision, fromView, type CullState } from './culling'
import type { CullSelection, CullSettings, UserDecision } from './cullTypes'

type Batch = Array<{ photo_id: number; decision: UserDecision }>
type SettingsPatch = Partial<Omit<CullSettings, 'threshold'>>

function consistent(state: CullState): boolean {
  let selected = 0
  for (const photo of state.photos.values()) if (!photo.culled) selected += 1
  return selected === state.view.summary.selected
}

export function useCulling(projectId: number) {
  const queryClient = useQueryClient()
  const generation = useRef(0)
  const view = useQuery({
    queryKey: ['culling', projectId],
    queryFn: async () => {
      const started = generation.current
      return { view: await api.culling(projectId), generation: started }
    },
    // While photos are still being analysed the grid fills in as they finish;
    // once nothing is pending, nothing changes without the user, so stop.
    refetchInterval: (query) =>
      (query.state.data?.view.summary.pending ?? 0) > 0 ? 2_000 : false,
    refetchOnWindowFocus: false,
  })

  const [state, setState] = useState<CullState | null>(null)
  const stateRef = useRef<CullState | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const undoStack = useRef<Batch[]>([])
  const [undoDepth, setUndoDepth] = useState(0)
  const chain = useRef<Promise<void>>(Promise.resolve())
  const writing = useRef(0)
  const pendingPatch = useRef<SettingsPatch | null>(null)

  const commit = useCallback((next: CullState | null) => {
    stateRef.current = next
    setState(next)
  }, [])

  useEffect(() => {
    if (!view.data) return
    if (view.data.generation !== generation.current || writing.current > 0) return
    commit(fromView(view.data.view))
  }, [commit, view.data])

  const refresh = useCallback(
    () => queryClient.invalidateQueries({ queryKey: ['culling', projectId] }),
    [projectId, queryClient],
  )

  /** Run one write after the ones before it, with polls held off meanwhile. */
  const write = useCallback(
    (task: () => Promise<CullSelection | null>) => {
      writing.current += 1
      generation.current += 1
      setBusy(true)
      void queryClient.cancelQueries({ queryKey: ['culling', projectId] })
      const run = async () => {
        try {
          const selection = await task()
          if (selection && stateRef.current) {
            commit(applySelection(stateRef.current, selection))
          }
          setError(null)
        } catch (caught) {
          setError(caught instanceof ApiError ? caught.message : String(caught))
        } finally {
          writing.current -= 1
          if (writing.current === 0) {
            setBusy(false)
            const current = stateRef.current
            if (current && (!consistent(current) || current.view.summary.pending > 0)) {
              void refresh()
            }
          }
        }
      }
      chain.current = chain.current.then(run)
      return chain.current
    },
    [commit, projectId, queryClient, refresh],
  )

  const updateSettings = useCallback(
    (patch: SettingsPatch) => {
      // Merged into whatever is waiting; only the first caller queues a send.
      const queued = pendingPatch.current !== null
      pendingPatch.current = { ...(pendingPatch.current ?? {}), ...patch }
      if (queued) return Promise.resolve()
      return write(async () => {
        const body = pendingPatch.current
        pendingPatch.current = null
        if (!body) return null
        const started = performance.now()
        const selection = await updateCullSettings(projectId, body)
        setLatencyMs(Math.round(performance.now() - started))
        return selection
      })
    },
    [projectId, write],
  )

  /**
   * Record decisions. ``batch`` may be a function of the state at the moment
   * the write actually goes out, which is what a gesture that depends on the
   * current proposed frame of a burst needs when several are queued.
   */
  const decide = useCallback(
    (batch: Batch | ((current: CullState) => Batch)) =>
      write(async () => {
        const current = stateRef.current
        if (!current) return null
        const items = typeof batch === 'function' ? batch(current) : batch
        if (items.length === 0) return null
        const before = items.map((item) => ({
          photo_id: item.photo_id,
          decision: currentDecision(current.photos.get(item.photo_id)!),
        }))
        const selection = await api.cullDecisions(projectId, items)
        undoStack.current = [...undoStack.current.slice(-99), before]
        setUndoDepth(undoStack.current.length)
        return selection
      }),
    [projectId, write],
  )

  const undo = useCallback(
    () =>
      write(async () => {
        const batch = undoStack.current.pop()
        setUndoDepth(undoStack.current.length)
        return batch ? api.cullDecisions(projectId, batch) : null
      }),
    [projectId, write],
  )

  return {
    state,
    loading: view.isLoading,
    loadError: view.error ? (view.error as ApiError).message : null,
    error,
    busy,
    latencyMs,
    canUndo: undoDepth > 0,
    updateSettings,
    decide,
    undo,
    refresh,
  }
}
