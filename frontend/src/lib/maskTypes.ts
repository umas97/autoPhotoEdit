// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The masks of an edit, as backend/ape/pipeline/mask_defs.py and
// MaskParams in params.py define them. Coordinates are in the *frame*: the
// lens-corrected picture before straightening and crop, normalised to 0..1
// along each side; radii are fractions of the frame's long edge; angles are
// counter-clockwise degrees. lib/geometry.ts maps them to the screen.
import type { EditParams } from './types'

export type MaskKind = 'parametric' | 'linear' | 'radial' | 'brush' | 'segment'
export type SegmentSubject = 'sky' | 'person' | 'skin'
export type CombineOp = 'add' | 'intersect' | 'subtract'

export interface ValueRange {
  low: number
  high: number
  feather: number
}

export interface HueRange {
  center: number
  width: number
  feather: number
}

export interface CombineStep {
  op: CombineOp
  kind: MaskKind
  invert: boolean
  definition: MaskDefinition
}

interface Shape {
  combine?: CombineStep[]
}

export interface LinearDef extends Shape {
  x0: number
  y0: number
  x1: number
  y1: number
}

export interface RadialDef extends Shape {
  cx: number
  cy: number
  rx: number
  ry: number
  angle: number
  feather: number
}

export interface RasterDef extends Shape {
  raster: string
}

export interface ParametricDef extends Shape {
  luminance?: ValueRange | null
  saturation?: ValueRange | null
  hue?: HueRange | null
}

export interface SegmentDef extends Shape {
  subject: SegmentSubject
  raster?: string | null
}

/**
 * Any kind's definition, every field optional: what the server validates per
 * kind (mask_defs.parse), the interface reads as one bag.
 */
export interface MaskDefinition
  extends Partial<Omit<LinearDef & RadialDef & ParametricDef, 'combine'>> {
  raster?: string | null
  subject?: SegmentSubject
  combine?: CombineStep[]
}

export interface MaskParams {
  kind: MaskKind
  name: string
  invert: boolean
  opacity: number
  definition: MaskDefinition
  exposure: EditParams['exposure']
  tone_shaping: EditParams['tone_shaping']
  color: EditParams['color']
  local_contrast: EditParams['local_contrast']
}
