// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A parameter document on this side of the wire: the neutral one, reading and
// writing a dotted path, and comparing two. The controls that edit it are
// described in controls.ts.
import type { EditParams, HSLBandName } from './types'

/** The neutral development: what `neutral_params()` produces on the server. */
export function neutralParams(): EditParams {
  return {
    params_version: 3,
    white_balance: { mode: 'as_shot', temperature_k: 5500, tint: 0 },
    exposure: { ev: 0 },
    highlight_recovery: { strength: 0.7, threshold: 0.96 },
    noise: { luminance: 0, chrominance: 0.25, radius: 0.004 },
    tone: {
      black_point_ev: -8,
      white_point_ev: 4.5,
      contrast: 1.2,
      pivot: 0,
      toe: 1.2,
      shoulder: 1.5,
      chroma_preservation: 0.4,
    },
    tone_shaping: { shadows: 0, highlights: 0, whites: 0, blacks: 0 },
    tone_curve: { highlights: 0, lights: 0, darks: 0, shadows: 0, points: [] },
    color: {
      saturation: 0,
      vibrance: 0,
      hsl: {},
      split_toning: {
        shadow_hue: 0,
        shadow_saturation: 0,
        highlight_hue: 0,
        highlight_saturation: 0,
        balance: 0,
      },
    },
    local_contrast: { clarity: 0, radius: 0.02, shadows: 0, highlights: 0 },
    sharpen: { amount: 0, radius: 0.0008, threshold: 0.01 },
    geometry: { lens_correction: true, rotation_deg: 0, crop: null },
    masks: [],
    retouch: [],
  }
}

type Indexable = Record<string, unknown>

/** Read a dotted path out of a parameter document. */
export function readPath(params: EditParams, path: string): number {
  let node: unknown = params
  for (const key of path.split('.')) {
    node = (node as Indexable)?.[key]
  }
  return typeof node === 'number' ? node : 0
}

/**
 * A copy of `params` with one dotted path replaced.
 *
 * Copying rather than mutating is what lets React see the change and what keeps
 * an undo stack honest: every state in the stack is a document nobody else
 * holds a reference into.
 */
export function withPath(params: EditParams, path: string, value: number): EditParams {
  const keys = path.split('.')
  const next = structuredClone(params) as unknown as Indexable
  let node = next
  for (const key of keys.slice(0, -1)) {
    node[key] = { ...((node[key] ?? {}) as Indexable) }
    node = node[key] as Indexable
  }
  node[keys[keys.length - 1]] = value
  return next as unknown as EditParams
}

/** Read one HSL band, which is absent from the document until it is touched. */
export function readBand(params: EditParams, band: HSLBandName, field: keyof HSLBandValues): number {
  return params.color.hsl[band]?.[field] ?? 0
}

export type HSLBandValues = { hue: number; saturation: number; luminance: number }

export function withBand(
  params: EditParams,
  band: HSLBandName,
  field: keyof HSLBandValues,
  value: number,
): EditParams {
  const next = structuredClone(params)
  const current = next.color.hsl[band] ?? { hue: 0, saturation: 0, luminance: 0 }
  next.color.hsl[band] = { ...current, [field]: value }
  return next
}

/**
 * Are these the parameters we started from? Drives the "unsaved" marker.
 *
 * An eraser's `fill` is not part of the edit: the server resolves it whenever
 * it uses the parameters (backend/ape/retouch/fills.py), so two documents
 * that differ only there are the same gesture.
 */
export function sameParams(a: EditParams, b: EditParams): boolean {
  return JSON.stringify(comparable(a)) === JSON.stringify(comparable(b))
}

function comparable(params: EditParams): EditParams {
  const retouch = params.retouch ?? []
  if (!retouch.some((item) => item.kind === 'erase' && item.fill !== null)) return params
  return {
    ...params,
    retouch: retouch.map((item) => (item.kind === 'erase' ? { ...item, fill: null } : item)),
  }
}
