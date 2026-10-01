// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The editor saves by itself (section 23.1; the user's choice of 2026-09-29:
// saving is automatic, Ctrl+Z stays on the gestures of the session). Every
// gesture becomes an EditVersion, but "the same slider moved again within two
// seconds" is one version, not twenty: a gesture waits two seconds before it
// is written, and a gesture on the same control inside that window replaces it
// and restarts the wait. A gesture on anything else, a change of photo, leaving
// the page or an action that needs the saved state (review, crop, restore)
// writes what is waiting at once.
//
// "The same control" is read from the parameters themselves -- the paths a
// gesture changed -- rather than threaded through every slider: two moves of
// the exposure change the same path, a move of the exposure and one of the
// contrast do not.
//
// Saves are written one after the other, never in parallel, so the server's
// "current version" is always the last gesture that reached it.
import { api, ApiError } from './api'
import type { EditParams, PhotoDetail } from './types'

/** Section 23.1: consecutive moves of one slider within this long are one version. */
export const COALESCE_MS = 2_000

export type SaveState = 'saved' | 'pending' | 'saving' | 'error'

/** The leaf paths where two parameter documents differ, like `exposure.ev`. */
export function changedPaths(a: unknown, b: unknown, prefix = ''): string[] {
  if (a === b) return []
  if (
    a !== null &&
    b !== null &&
    typeof a === 'object' &&
    typeof b === 'object' &&
    Array.isArray(a) === Array.isArray(b)
  ) {
    // A mask added or removed is a change of the list, not of its entries.
    if (Array.isArray(a) && a.length !== (b as unknown[]).length) return [prefix]
    const left = a as Record<string, unknown>
    const right = b as Record<string, unknown>
    const keys = new Set([...Object.keys(left), ...Object.keys(right)])
    return [...keys].flatMap((key) =>
      changedPaths(left[key], right[key], prefix ? `${prefix}.${key}` : key),
    )
  }
  return [prefix]
}

interface Pending {
  photoId: number
  params: EditParams
  /** What the gesture moved: gestures with the same key coalesce. */
  key: string
  timer?: number
}

export interface SaverEvents {
  saved: (detail: PhotoDetail) => void
  state: (state: SaveState) => void
  failed: (message: string) => void
}

export class AutoSaver {
  private pending: Pending | null = null
  private chain: Promise<void> = Promise.resolve()
  private writing = new Map<number, number>()
  private failing = false
  /** What the server holds for each photo, as JSON: an equal save is skipped. */
  private stored = new Map<number, string>()

  constructor(private readonly events: SaverEvents) {}

  /** The server holds `params` for this photo (loaded, or changed elsewhere). */
  known(photoId: number, params: EditParams): void {
    this.stored.set(photoId, JSON.stringify(params))
  }

  /** A gesture ended on `params`; `key` names what it moved. */
  record(photoId: number, params: EditParams, key: string): void {
    const waiting = this.pending
    if (waiting && (waiting.photoId !== photoId || waiting.key !== key)) void this.flush()
    if (this.pending) {
      window.clearTimeout(this.pending.timer)
      this.pending.params = params
    } else {
      this.pending = { photoId, params, key }
    }
    this.pending.timer = window.setTimeout(() => void this.flush(), COALESCE_MS)
    this.emit()
  }

  /** Forget what waits for this photo: something else changed it, and wins. */
  discard(photoId: number): void {
    if (this.pending?.photoId !== photoId) return
    window.clearTimeout(this.pending.timer)
    this.pending = null
    this.emit()
  }

  /** A save of this photo is on its way to the server. */
  isWriting(photoId: number): boolean {
    return (this.writing.get(photoId) ?? 0) > 0
  }

  /** Write what is waiting now; resolves when every save so far has an answer. */
  flush(): Promise<void> {
    const waiting = this.pending
    if (waiting) {
      window.clearTimeout(waiting.timer)
      this.pending = null
      this.chain = this.chain.then(() => this.write(waiting))
    }
    return this.chain
  }

  private async write(entry: Pending): Promise<void> {
    const json = JSON.stringify(entry.params)
    if (this.stored.get(entry.photoId) === json) {
      // An undo back to the saved state, or a retry already answered.
      this.failing = false
      this.emit()
      return
    }
    this.writing.set(entry.photoId, (this.writing.get(entry.photoId) ?? 0) + 1)
    this.emit()
    try {
      // keepalive: the last gesture before the window closes still arrives.
      const detail = await api.saveParams(entry.photoId, entry.params, 'user_edited', true)
      this.stored.set(entry.photoId, json)
      this.failing = false
      this.events.saved(detail)
    } catch (error) {
      this.failing = true
      // Kept for the next flush -- "Riprova", or the next gesture -- unless a
      // newer gesture is already waiting, which carries everything this one did.
      if (!this.pending) this.pending = { photoId: entry.photoId, params: entry.params, key: '' }
      this.events.failed(error instanceof ApiError ? error.message : String(error))
    } finally {
      this.writing.set(entry.photoId, (this.writing.get(entry.photoId) ?? 1) - 1)
      this.emit()
    }
  }

  private emit(): void {
    const writing = [...this.writing.values()].some((count) => count > 0)
    this.events.state(
      writing ? 'saving' : this.failing ? 'error' : this.pending ? 'pending' : 'saved',
    )
  }
}
