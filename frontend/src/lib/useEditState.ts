// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The edit under the user's hand, in the viewer and in the review: the
// parameters on screen, whether a slider is being dragged (the preview renders
// small meanwhile, section 10), the gestures Ctrl+Z steps back through -- and
// Ctrl+Shift+Z forward again -- and the saving, which happens by itself
// (lib/autosave.ts: one version per gesture, two seconds of coalescence).
//
// One undo entry per *gesture*: a slider or a handle records the state from
// before the gesture when it is released, not at every frame of the drag. A new
// gesture after some undos forgets the steps undone, as every editor does. The
// undo stack lives in memory (the user's choice, 2026-09-29): after a restart
// the way back is the version strip, whose every entry the autosave wrote.
//
// **Where the edit started** (`base`): the version of each photo as it was when
// this screen first showed it, or when something other than this editor last
// changed it. The review measures "the correction of the representative" from
// there -- the current version is, by the time the user presses "Applica alla
// scena", already the correction.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AutoSaver, changedPaths, type SaveState } from './autosave'
import { neutralParams, sameParams } from './params'
import { toast } from './toast'
import type { EditParams, PhotoDetail } from './types'
import { t } from '../i18n/it'

const DEPTH = 50

export interface EditBase {
  versionId: number | null
  params: EditParams
}

export function useEditState(detail: PhotoDetail | undefined, photoId: number | null) {
  const queryClient = useQueryClient()
  const [params, setParamsState] = useState<EditParams | null>(null)
  const [dragging, setDragging] = useState(false)
  const [saveState, setSaveState] = useState<SaveState>('saved')
  const [, setRevision] = useState(0)
  const current = useRef<EditParams | null>(null)
  const history = useRef<EditParams[]>([])
  const future = useRef<EditParams[]>([])
  const gestureStart = useRef<EditParams | null>(null)
  const dragTimer = useRef<number | undefined>(undefined)
  /** The photo the parameters on screen belong to, and its version we know. */
  const loaded = useRef<{ photoId: number; versionId: number | null } | null>(null)
  const bases = useRef(new Map<number, EditBase>())

  const show = useCallback((next: EditParams | null) => {
    current.current = next
    setParamsState(next)
  }, [])

  const saver = useMemo(
    () =>
      new AutoSaver({
        saved: (updated) => {
          if (loaded.current?.photoId === updated.id) loaded.current.versionId = updated.current_version_id
          // A read of the photo still on its way would put the old version back.
          void queryClient.cancelQueries({ queryKey: ['photo', updated.id] })
          queryClient.setQueryData(['photo', updated.id], updated)
          for (const queryKey of [['photoReview', updated.id], ['photoStyle', updated.id], ['review']]) {
            void queryClient.invalidateQueries({ queryKey })
          }
        },
        state: setSaveState,
        failed: (message) => toast(t('editing.saveFailed', { message })),
      }),
    [queryClient],
  )

  useEffect(() => {
    if (!detail || detail.id !== photoId) return
    const incoming = detail.params ?? neutralParams()
    const here = loaded.current
    const same = here?.photoId === detail.id
    if (same) {
      const known = here.versionId
      if (detail.current_version_id === known) return
      // Versions only grow: a smaller one is a read older than our last save.
      if (known !== null && detail.current_version_id !== null && detail.current_version_id < known) return
      // Our own save, seen by another request before its answer came back.
      if (saver.isWriting(detail.id)) return
      if (current.current && sameParams(incoming, current.current)) {
        here.versionId = detail.current_version_id
        return
      }
      // Changed by something else -- a restore, a review action, a crop: it wins.
      saver.discard(detail.id)
    }
    loaded.current = { photoId: detail.id, versionId: detail.current_version_id }
    saver.known(detail.id, incoming)
    if (same || !bases.current.has(detail.id)) {
      bases.current.set(detail.id, { versionId: detail.current_version_id, params: incoming })
    }
    show(incoming)
    history.current = []
    future.current = []
    gestureStart.current = null
  }, [detail, photoId, saver, show])

  // Leaving a photo or the screen writes what is waiting; so does closing the window.
  useEffect(() => () => void saver.flush(), [photoId, saver])
  useEffect(() => {
    const leave = () => void saver.flush()
    window.addEventListener('pagehide', leave)
    return () => window.removeEventListener('pagehide', leave)
  }, [saver])

  const record = useCallback(
    (before: EditParams, next: EditParams) => {
      const photo = loaded.current?.photoId
      if (photo !== undefined) saver.record(photo, next, changedPaths(before, next).join(' '))
    },
    [saver],
  )

  const onChange = useCallback(
    (next: EditParams, commit: boolean) => {
      if (!commit) {
        if (gestureStart.current === null) gestureStart.current = current.current
        show(next)
        setDragging(true)
        return
      }
      const before = gestureStart.current ?? current.current
      gestureStart.current = null
      show(next)
      window.clearTimeout(dragTimer.current)
      // A hair after the gesture: the commit and the last move arrive in the
      // same frame, and flipping the size instantly would start the 2048 px
      // render while the 1024 px one is still the thing on screen.
      dragTimer.current = window.setTimeout(() => setDragging(false), 30)
      if (!before || sameParams(before, next)) return
      history.current = [...history.current.slice(-(DEPTH - 1)), before]
      future.current = []
      record(before, next)
    },
    [record, show],
  )

  /** Step back one gesture; false when there is none left. */
  const stepBack = useCallback(() => {
    const previous = history.current.pop()
    if (!previous) return false
    const now = current.current
    if (now) future.current = [...future.current, now]
    show(previous)
    setDragging(false)
    if (now) record(now, previous)
    return true
  }, [record, show])

  /** Redo the last gesture undone; false when there is none. */
  const stepForward = useCallback(() => {
    const next = future.current.pop()
    if (!next) return false
    const now = current.current
    if (now) history.current = [...history.current, now]
    show(next)
    setDragging(false)
    if (now) record(now, next)
    return true
  }, [record, show])

  /** Write what is waiting now, before an action that reads the saved state. */
  const flush = useCallback(() => saver.flush(), [saver])

  /**
   * The gestures so far became an action (a review decision): Ctrl+Z no longer
   * steps through them, and every edit starts again from here.
   */
  const settle = useCallback(() => {
    history.current = []
    future.current = []
    bases.current.clear()
    const here = loaded.current
    if (here && current.current) {
      bases.current.set(here.photoId, { versionId: here.versionId, params: current.current })
    }
    setRevision((n) => n + 1)
  }, [])

  const base = photoId !== null ? (bases.current.get(photoId) ?? null) : null
  /** The parameters on screen differ from where the edit started. */
  const edited = params !== null && base !== null && !sameParams(params, base.params)
  return { params, dragging, onChange, stepBack, stepForward, saveState, flush, base, edited, settle }
}
