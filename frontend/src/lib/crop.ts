// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The arithmetic of the hand-drawn crop, kept apart from the component that
// draws it so that it can be read on its own.
//
// A `CropRect` is normalised to the straightened frame, whose sides are
// `aspect` (width / height) in proportion: the same frame as the proxy, since
// the straightening keeps the ratio (backend/ape/pipeline/geometry.py). A
// rectangle of normalised size (w, h) is therefore `w * aspect / h` wide in
// pixels for each pixel of height, and `w === h` is the frame's own ratio
// exactly -- 3:2 or 2:3 for a single shot, whatever the stitching gave a
// panorama. That is the default: any other ratio is a choice the user makes.
import type { CropRect } from './analysisTypes'

/** The ratios offered. `original` is the frame's own; `free` is no ratio. */
export type CropAspect = 'original' | 'free' | '1:1' | '4:5' | '16:9'

export const CROP_ASPECTS: CropAspect[] = ['original', 'free', '1:1', '4:5', '16:9']

/** Which corner or side a handle moves. `move` drags the whole rectangle. */
export type CropHandle = 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'w' | 'e' | 'move'

export const FULL_FRAME: CropRect = { x: 0, y: 0, width: 1, height: 1 }

/** Smallest side of a crop, as a fraction of the frame's side. */
const MIN_SIDE = 0.05

/** Two ratios this close are the same ratio (a rounding of the server's). */
const SAME_RATIO = 0.01

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

/**
 * Pixel ratio (width / height) a choice asks for, or null for a free crop.
 *
 * Fixed ratios are oriented like the frame -- "4:5" on a landscape frame is
 * 5:4 -- and `flipped` turns them the other way: 2:3 on a 3:2 frame.
 */
export function targetRatio(choice: CropAspect, flipped: boolean, aspect: number): number | null {
  if (choice === 'free') return null
  let ratio = aspect
  if (choice !== 'original') {
    const [a, b] = choice.split(':').map(Number)
    ratio = a / b
    if (ratio > 1 !== aspect > 1) ratio = 1 / ratio
  }
  return flipped ? 1 / ratio : ratio
}

/** The pixel ratio of a rectangle on a frame of `aspect`. */
export function rectRatio(rect: CropRect, aspect: number): number {
  return (rect.width * aspect) / rect.height
}

/** The choice that reproduces `rect`'s ratio, so a crop opens as it was drawn. */
export function detectAspect(
  rect: CropRect | null,
  aspect: number,
): { choice: CropAspect; flipped: boolean } {
  if (!rect) return { choice: 'original', flipped: false }
  const ratio = rectRatio(rect, aspect)
  for (const choice of CROP_ASPECTS) {
    for (const flipped of [false, true]) {
      const target = targetRatio(choice, flipped, aspect)
      if (target !== null && Math.abs(ratio / target - 1) < SAME_RATIO) return { choice, flipped }
    }
  }
  return { choice: 'free', flipped: false }
}

/**
 * `rect` brought to `ratio`, around its own centre and with about its area,
 * shrunk to fit the frame if it has to be.
 */
export function fitRatio(rect: CropRect, ratio: number | null, aspect: number): CropRect {
  if (ratio === null) return rect
  // In units of the frame's height, where the frame is `aspect` x 1.
  const area = rect.width * aspect * rect.height
  let w = Math.sqrt(area * ratio)
  let h = Math.sqrt(area / ratio)
  const shrink = Math.min(1, aspect / w, 1 / h)
  w *= shrink
  h *= shrink
  const width = w / aspect
  const height = h
  const cx = rect.x + rect.width / 2
  const cy = rect.y + rect.height / 2
  return {
    x: clamp(cx - width / 2, 0, 1 - width),
    y: clamp(cy - height / 2, 0, 1 - height),
    width,
    height,
  }
}

/** The largest rectangle of `ratio` on the frame, centred. */
export function largest(ratio: number | null, aspect: number): CropRect {
  if (ratio === null) return FULL_FRAME
  // As wide as the frame, unless that is taller than it.
  let w = aspect
  let h = aspect / ratio
  if (h > 1) {
    h = 1
    w = ratio
  }
  const width = Math.min(1, w / aspect)
  const height = Math.min(1, h)
  return { x: (1 - width) / 2, y: (1 - height) / 2, width, height }
}

/**
 * `start` with `handle` dragged by (dx, dy), in normalised frame units.
 *
 * With a ratio, a corner moves along it and the opposite corner stays put;
 * the sides are only offered without one, since moving a single side is
 * exactly what changes the ratio.
 */
export function dragRect(
  start: CropRect,
  handle: CropHandle,
  dx: number,
  dy: number,
  ratio: number | null,
  aspect: number,
): CropRect {
  if (handle === 'move') {
    return {
      ...start,
      x: clamp(start.x + dx, 0, 1 - start.width),
      y: clamp(start.y + dy, 0, 1 - start.height),
    }
  }
  let left = start.x
  let top = start.y
  let right = start.x + start.width
  let bottom = start.y + start.height
  const west = handle.includes('w')
  const east = handle.includes('e')
  const north = handle.includes('n')
  const south = handle.includes('s')

  if (ratio === null) {
    if (west) left = clamp(left + dx, 0, right - MIN_SIDE)
    if (east) right = clamp(right + dx, left + MIN_SIDE, 1)
    if (north) top = clamp(top + dy, 0, bottom - MIN_SIDE)
    if (south) bottom = clamp(bottom + dy, top + MIN_SIDE, 1)
    return { x: left, y: top, width: right - left, height: bottom - top }
  }
  if (!(west || east) || !(north || south)) return start

  // The corner opposite the handle is the anchor; sizes in units of the
  // frame's height, where the ratio is a ratio of lengths.
  const anchorX = west ? right : left
  const anchorY = north ? bottom : top
  const pointerX = (west ? left : right) + dx
  const pointerY = (north ? top : bottom) + dy
  const reachX = Math.max(0, west ? anchorX - pointerX : pointerX - anchorX) * aspect
  const reachY = Math.max(0, north ? anchorY - pointerY : pointerY - anchorY)
  const roomX = (west ? anchorX : 1 - anchorX) * aspect
  const roomY = north ? anchorY : 1 - anchorY
  const smallest = Math.max(MIN_SIDE * aspect, MIN_SIDE * ratio)
  let w = Math.max(reachX, reachY * ratio, smallest)
  w = Math.min(w, roomX, roomY * ratio)
  const width = w / aspect
  const height = w / ratio
  return {
    x: west ? anchorX - width : anchorX,
    y: north ? anchorY - height : anchorY,
    width,
    height,
  }
}

/** True when `rect` keeps the whole frame: no crop at all. */
export function isFullFrame(rect: CropRect): boolean {
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-4
  return near(rect.x, 0) && near(rect.y, 0) && near(rect.width, 1) && near(rect.height, 1)
}

/** "3:2", "16:9", "2.84:1" -- a pixel ratio as a person reads it. */
export function ratioLabel(ratio: number): string {
  const wide = ratio >= 1 ? ratio : 1 / ratio
  for (let short = 1; short <= 9; short++) {
    const long = wide * short
    // Close enough to a ratio of small whole numbers to be one -- not so
    // close that a panorama's 2.84 reads as 17:6.
    if (Math.abs(long / short - Math.round(long) / short) < 0.006) {
      const pair = [Math.round(long), short]
      return (ratio >= 1 ? pair : pair.reverse()).join(':')
    }
  }
  const value = wide.toFixed(2).replace(/\.?0+$/, '')
  return ratio >= 1 ? `${value}:1` : `1:${value}`
}
