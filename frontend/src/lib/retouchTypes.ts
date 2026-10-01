// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The removals of an edit, as backend/ape/pipeline/retouch_params.py declares
// them. Coordinates in the frame (as the masks'), radii in long edges.

export type RetouchKind = 'heal' | 'erase'

interface ItemBase {
  /** Stable across edits: what the panel selects, and what a fill is matched to. */
  id: string
  visible: boolean
  opacity: number
}

export interface HealItem extends ItemBase {
  kind: 'heal'
  cx: number
  cy: number
  radius: number
  sx: number
  sy: number
  /** False once the source was dragged by hand. */
  source_auto: boolean
  /** Fraction of the radius the copy fades over. */
  feather: number
}

export type EraseEngine = 'classic' | 'ml'

export interface EraseItem extends ItemBase {
  kind: 'erase'
  /** The painted area, a raster of the masks folder. */
  area: string
  expand: number
  /** Fraction of the area's radius the fill fades over. */
  feather: number
  engine: EraseEngine
  seed: number
  /** The fill the worker made; the server resolves it, the editor never sets it. */
  fill: string | null
}

export type RetouchItem = HealItem | EraseItem

/** The state of one removal (backend/ape/retouch/service.py). */
export type RetouchState = 'ready' | 'computing' | 'stale' | 'error'

export interface RetouchItemState {
  id: string
  kind: RetouchKind
  state: RetouchState
  error: string | null
}

export interface RetouchStates {
  items: RetouchItemState[]
  ml: { available: boolean; reason: string | null }
}
