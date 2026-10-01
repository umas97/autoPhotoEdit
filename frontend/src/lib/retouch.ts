// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The removals of an edit as data: how one starts, how one changes, and the
// controls the Rimozione panel draws -- in the shape of lib/masks.ts, so that
// tests/test_frontend_contract.py checks every range against HealItem and
// EraseItem (backend/ape/pipeline/retouch_params.py).
//
// A removal is its photo's own: nothing here reads the style, and no style,
// scene or preset writes a removal (docs/SPEC_rimozione.md R6).
import type { Control } from './controls'
import type { EraseItem, HealItem, RetouchItem } from './retouchTypes'
import type { EditParams } from './types'

/** The spot's size, in long edges: a speck of dust at 1:1 to a patch of cheek. */
export const HEAL_RADIUS = { min: 0.001, max: 0.15, start: 0.012 }

/**
 * Controls of a removal. The first segment of the path names the model it is
 * checked against: `heal` HealItem, `erase` EraseItem.
 */
export const RETOUCH_CONTROLS: Record<string, Control> = {
  healRadius: {
    path: 'heal.radius', label: 'retouch.size', min: 0.001, max: 0.15, step: 0.0005,
    neutral: 0.01, digits: 3,
  },
  healFeather: {
    path: 'heal.feather', label: 'retouch.feather', min: 0, max: 1, step: 0.01,
    neutral: 0.5, digits: 2,
  },
  healOpacity: {
    path: 'heal.opacity', label: 'retouch.opacity', min: 0, max: 1, step: 0.01,
    neutral: 1, digits: 2,
  },
  eraseExpand: {
    path: 'erase.expand', label: 'retouch.expand', min: 0, max: 0.05, step: 0.0005,
    neutral: 0.002, digits: 3,
  },
  eraseFeather: {
    path: 'erase.feather', label: 'retouch.feather', min: 0, max: 1, step: 0.01,
    neutral: 0.25, digits: 2,
  },
  eraseOpacity: {
    path: 'erase.opacity', label: 'retouch.opacity', min: 0, max: 1, step: 0.01,
    neutral: 1, digits: 2,
  },
}

/** An id no removal of this photo uses: short, and stable across edits. */
export function newId(items: RetouchItem[], prefix: 'h' | 'e'): string {
  const taken = new Set(items.map((item) => item.id))
  for (;;) {
    const candidate = `${prefix}${Math.random().toString(36).slice(2, 10)}`
    if (!taken.has(candidate)) return candidate
  }
}

export function newHeal(
  params: EditParams,
  at: { cx: number; cy: number; radius: number },
  source: { sx: number; sy: number },
): HealItem {
  return {
    kind: 'heal',
    id: newId(params.retouch, 'h'),
    visible: true,
    opacity: 1,
    cx: at.cx,
    cy: at.cy,
    radius: at.radius,
    sx: source.sx,
    sy: source.sy,
    source_auto: true,
    feather: 0.5,
  }
}

export function newErase(params: EditParams, area: string, expand = 0.002): EraseItem {
  return {
    kind: 'erase',
    id: newId(params.retouch, 'e'),
    visible: true,
    opacity: 1,
    area,
    expand,
    feather: 0.25,
    engine: 'classic',
    seed: 0,
    fill: null,
  }
}

export function withItem(params: EditParams, next: RetouchItem): EditParams {
  return { ...params, retouch: params.retouch.map((item) => (item.id === next.id ? next : item)) }
}

export function withAddedItem(params: EditParams, item: RetouchItem): EditParams {
  return { ...params, retouch: [...params.retouch, item] }
}

export function withoutItem(params: EditParams, id: string): EditParams {
  return { ...params, retouch: params.retouch.filter((item) => item.id !== id) }
}

/** The removal with this id, if the parameters still have it (an undo may take it). */
export function itemById(params: EditParams | null, id: string | null): RetouchItem | undefined {
  if (!params || id === null) return undefined
  return params.retouch.find((item) => item.id === id)
}

/** What makes a fill: when these change, a new one is computed. */
export function fillInputs(params: EditParams): string {
  return JSON.stringify(
    params.retouch.map((item) =>
      item.kind === 'erase'
        ? [item.id, item.area, item.expand, item.feather, item.engine, item.seed, item.visible]
        : [item.id, item.cx, item.cy, item.radius, item.sx, item.sy, item.feather,
           item.opacity, item.visible],
    ),
  ) + JSON.stringify(params.geometry.lens_correction)
}
