// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The masks of an edit as data: how a new one starts, how one is changed, and
// the controls the masks panel draws -- in the shape of lib/params.ts, so that
// tests/test_frontend_contract.py checks every range against MaskParams and
// the definitions of backend/ape/pipeline/mask_defs.py.
//
// A mask carries only what section 6.3 lets it carry: exposure, the tone
// shaping, saturation and vibrance, and the local contrast. Geometry and the
// tone mapping stay global.
import { t, type StringKey } from '../i18n/it'
import { apply, viewToFrame, type FrameSize } from './geometry'
import type { CombineStep, MaskDefinition, MaskKind, MaskParams } from './maskTypes'
import type { Control, Group } from './controls'
import { neutralParams } from './params'
import type { EditParams } from './types'

const unit = (path: string, label: StringKey, neutral = 0): Control => ({
  path,
  label,
  min: -1,
  max: 1,
  step: 0.01,
  neutral,
  digits: 2,
})

/** The adjustments of a mask; paths are into `MaskParams`. */
export const MASK_GROUPS: Group[] = [
  {
    key: 'mask.light',
    label: 'masks.group.light',
    open: true,
    controls: [
      {
        path: 'exposure.ev',
        label: 'params.exposure.ev',
        min: -4,
        max: 4,
        step: 0.01,
        neutral: 0,
        digits: 2,
        unit: ' EV',
      },
      unit('tone_shaping.highlights', 'params.tone_shaping.highlights'),
      unit('tone_shaping.shadows', 'params.tone_shaping.shadows'),
      unit('tone_shaping.whites', 'params.tone_shaping.whites'),
      unit('tone_shaping.blacks', 'params.tone_shaping.blacks'),
    ],
  },
  {
    key: 'mask.color',
    label: 'masks.group.color',
    open: true,
    controls: [
      unit('color.saturation', 'params.color.saturation'),
      unit('color.vibrance', 'params.color.vibrance'),
    ],
  },
  {
    key: 'mask.detail',
    label: 'masks.group.detail',
    open: true,
    controls: [
      unit('local_contrast.clarity', 'params.local_contrast.clarity'),
      unit('local_contrast.highlights', 'params.local_contrast.highlights'),
      unit('local_contrast.shadows', 'params.local_contrast.shadows'),
    ],
  },
]

/**
 * Controls of the selection itself. The first segment of the path names the
 * model it is checked against: `mask` MaskParams, `radial` RadialDef, `range`
 * ValueRange, `hue` HueRange.
 */
export const SELECTION_CONTROLS: Record<string, Control> = {
  opacity: {
    path: 'mask.opacity',
    label: 'masks.opacity',
    min: 0,
    max: 1,
    step: 0.01,
    neutral: 1,
    digits: 2,
  },
  feather: {
    path: 'radial.feather',
    label: 'masks.feather',
    min: 0,
    max: 1,
    step: 0.01,
    neutral: 0.5,
    digits: 2,
  },
  low: { path: 'range.low', label: 'masks.range.low', min: 0, max: 1, step: 0.01, neutral: 0, digits: 2 },
  high: { path: 'range.high', label: 'masks.range.high', min: 0, max: 1, step: 0.01, neutral: 1, digits: 2 },
  rangeFeather: {
    path: 'range.feather',
    label: 'masks.feather',
    min: 0,
    max: 0.5,
    step: 0.01,
    neutral: 0.1,
    digits: 2,
  },
  center: { path: 'hue.center', label: 'masks.hue.center', min: 0, max: 360, step: 1, neutral: 0, digits: 0, unit: '°' },
  width: { path: 'hue.width', label: 'masks.hue.width', min: 0, max: 360, step: 1, neutral: 60, digits: 0, unit: '°' },
  hueFeather: {
    path: 'hue.feather',
    label: 'masks.feather',
    min: 0,
    max: 90,
    step: 1,
    neutral: 20,
    digits: 0,
    unit: '°',
  },
}

export const ADDABLE: MaskKind[] = ['linear', 'radial', 'brush', 'parametric', 'segment']

/** A mask with every adjustment at zero: it changes nothing until a slider moves. */
function blank(kind: MaskKind, name: string, definition: MaskDefinition): MaskParams {
  const neutral = neutralParams()
  return {
    kind,
    name,
    invert: false,
    opacity: 1,
    definition,
    exposure: neutral.exposure,
    tone_shaping: neutral.tone_shaping,
    color: neutral.color,
    local_contrast: neutral.local_contrast,
  }
}

/** "Radiale 2": the first number no mask of that kind uses yet. */
export function nextName(kind: MaskKind, masks: MaskParams[]): string {
  const base = t(`masks.kind.${kind}` as StringKey)
  const taken = new Set(masks.map((mask) => mask.name))
  let n = 1
  while (taken.has(`${base} ${n}`)) n += 1
  return `${base} ${n}`
}

/**
 * A new mask of `kind`, placed in the part of the frame the preview shows:
 * with a crop, the middle of the frame may be outside it.
 *
 * `raster` is required for a brush -- an empty canvas already uploaded -- and
 * for a subject, whose selection the segmentation produced.
 */
export function newMask(
  kind: MaskKind,
  params: EditParams,
  frame: FrameSize | null,
  raster?: string,
): MaskParams {
  const name = nextName(kind, params.masks)
  const toFrame = frame ? viewToFrame(frame, params.geometry) : null
  const at = (a: number, b: number): [number, number] => (toFrame ? apply(toFrame, a, b) : [a, b])
  if (kind === 'linear') {
    const [x0, y0] = at(0.5, 0)
    const [x1, y1] = at(0.5, 0.45)
    return blank(kind, name, { x0, y0, x1, y1 })
  }
  if (kind === 'radial') {
    const [cx, cy] = at(0.5, 0.5)
    // A quarter of the long edge of what is on screen, in frame long-edge units.
    const shown = params.geometry.crop
      ? Math.max(params.geometry.crop.width, params.geometry.crop.height)
      : 1
    const radius = 0.22 * shown
    return blank(kind, name, { cx, cy, rx: radius, ry: radius * 0.8, angle: 0, feather: 0.5 })
  }
  if (kind === 'brush') return blank(kind, name, { raster })
  if (kind === 'segment') return blank(kind, name, { subject: 'sky', raster: raster ?? null })
  return blank(kind, name, { luminance: { low: 0.7, high: 1, feather: 0.1 } })
}

export function withMask(params: EditParams, index: number, mask: MaskParams): EditParams {
  const next = structuredClone(params)
  next.masks[index] = mask
  return next
}

export function withoutMask(params: EditParams, index: number): EditParams {
  const next = structuredClone(params)
  next.masks.splice(index, 1)
  return next
}

export function withAddedMask(params: EditParams, mask: MaskParams): EditParams {
  const next = structuredClone(params)
  next.masks.push(mask)
  return next
}

type Indexable = Record<string, unknown>

export function readMaskPath(mask: MaskParams, path: string): number {
  let node: unknown = mask
  for (const key of path.split('.')) node = (node as Indexable)?.[key]
  return typeof node === 'number' ? node : 0
}

export function withMaskPath(mask: MaskParams, path: string, value: number): MaskParams {
  const keys = path.split('.')
  const next = structuredClone(mask) as unknown as Indexable
  let node = next
  for (const key of keys.slice(0, -1)) {
    node[key] = { ...((node[key] ?? {}) as Indexable) }
    node = node[key] as Indexable
  }
  node[keys[keys.length - 1]] = value
  return next as unknown as MaskParams
}

export function withDefinition(mask: MaskParams, patch: MaskDefinition): MaskParams {
  return { ...structuredClone(mask), definition: { ...mask.definition, ...patch } }
}

/** Does the mask adjust anything? A mask that does not is drawn but inert. */
export function adjusts(mask: MaskParams): boolean {
  const neutral = blank(mask.kind, '', {})
  return (['exposure', 'tone_shaping', 'color', 'local_contrast'] as const).some(
    (section) => JSON.stringify(mask[section]) !== JSON.stringify(neutral[section]),
  )
}

/** The refinements the panel offers on a shape: "only these tones", "only this colour". */
export type Limit = 'luminance' | 'hue'

export function limitStep(mask: MaskParams, limit: Limit): number {
  return (mask.definition.combine ?? []).findIndex(
    (step) => step.kind === 'parametric' && step.op === 'intersect' && limit in step.definition,
  )
}

export function withLimit(mask: MaskParams, limit: Limit, on: boolean): MaskParams {
  const combine = [...(mask.definition.combine ?? [])]
  const at = limitStep(mask, limit)
  if (!on) {
    if (at >= 0) combine.splice(at, 1)
  } else if (at < 0) {
    const step: CombineStep = {
      op: 'intersect',
      kind: 'parametric',
      invert: false,
      definition:
        limit === 'luminance'
          ? { luminance: { low: 0.5, high: 1, feather: 0.1 } }
          : { hue: { center: 220, width: 60, feather: 20 } },
    }
    combine.push(step)
  }
  return withDefinition(mask, { combine })
}

export function withLimitRange(
  mask: MaskParams,
  limit: Limit,
  field: string,
  value: number,
): MaskParams {
  const at = limitStep(mask, limit)
  if (at < 0) return mask
  const combine = structuredClone(mask.definition.combine ?? [])
  const range = { ...(combine[at].definition[limit] as unknown as Indexable), [field]: value }
  combine[at] = { ...combine[at], definition: { ...combine[at].definition, [limit]: range } }
  return withDefinition(mask, { combine })
}

/** Keeps `low <= high`, which the server refuses otherwise. */
export function orderedRange(
  range: { low: number; high: number; feather: number },
  field: string,
  value: number,
): { low: number; high: number; feather: number } {
  const next = { ...range, [field]: value }
  if (field === 'low' && next.low > next.high) next.high = next.low
  if (field === 'high' && next.high < next.low) next.low = next.high
  return next
}

/** Where a frame point in long-edge units lands in normalised frame coordinates. */
export function longToFrame(frame: FrameSize, x: number, y: number): [number, number] {
  const long = Math.max(frame.width, frame.height)
  return [(x * long) / frame.width, (y * long) / frame.height]
}

export function frameToLong(frame: FrameSize, x: number, y: number): [number, number] {
  const long = Math.max(frame.width, frame.height)
  return [(x * frame.width) / long, (y * frame.height) / long]
}

