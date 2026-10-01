// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Where a point of the frame is on the screen, and back.
//
// Masks are drawn in the *frame* -- the lens-corrected picture before
// straightening and crop -- so that a gradient stays on the sky it was drawn on
// when the rotation or the crop change afterwards. The preview shows the frame
// after both. This file is the map between the two, and it is the same
// arithmetic as backend/ape/pipeline/geometry.py `apply`, written as an affine
// map of normalised coordinates:
//
//   view (a, b) in 0..1 over the preview
//     -> straightened frame  X = crop.x + crop.width * a
//     -> centred pixels      d = k * (W (X - 1/2), H (Y - 1/2))
//     -> rotated back        p = [[c, -s], [s, c]] d
//     -> frame               f = p / (W, H) + 1/2
//
// with k the inscribed scale of the rotation. In pixels it is a similarity --
// a rotation and one scale -- so perpendiculars stay perpendicular and circles
// stay circles on screen, which the overlays rely on.
import { useEffect, useState, type RefObject } from 'react'
import type { EditParams } from './types'
import type { CropRect } from './analysisTypes'

/** CSS's order: x' = a x + c y + e, y' = b x + d y + f. */
export interface Affine {
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
}

export interface Box {
  left: number
  top: number
  width: number
  height: number
}

export interface FrameSize {
  width: number
  height: number
}

/** Below this a rotation is none, as `_EPSILON_DEG` on the server. */
const EPSILON_DEG = 1e-3

export function apply(m: Affine, x: number, y: number): [number, number] {
  return [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]
}

/** `outer` after `inner`. */
export function compose(outer: Affine, inner: Affine): Affine {
  return {
    a: outer.a * inner.a + outer.c * inner.b,
    b: outer.b * inner.a + outer.d * inner.b,
    c: outer.a * inner.c + outer.c * inner.d,
    d: outer.b * inner.c + outer.d * inner.d,
    e: outer.a * inner.e + outer.c * inner.f + outer.e,
    f: outer.b * inner.e + outer.d * inner.f + outer.f,
  }
}

export function invert(m: Affine): Affine {
  const det = m.a * m.d - m.b * m.c
  return {
    a: m.d / det,
    b: -m.b / det,
    c: -m.c / det,
    d: m.a / det,
    e: (m.c * m.f - m.d * m.e) / det,
    f: (m.b * m.e - m.a * m.f) / det,
  }
}

export function scale(sx: number, sy: number, tx = 0, ty = 0): Affine {
  return { a: sx, b: 0, c: 0, d: sy, e: tx, f: ty }
}

export function cssMatrix(m: Affine): string {
  return `matrix(${m.a}, ${m.b}, ${m.c}, ${m.d}, ${m.e}, ${m.f})`
}

/** `geometry.inscribed_scale`: how much of each side survives the rotation. */
export function inscribedScale(width: number, height: number, rotationDeg: number): number {
  const theta = (Math.abs(rotationDeg) * Math.PI) / 180
  if (Math.abs(rotationDeg) < EPSILON_DEG) return 1
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return Math.min(width / (width * c + height * s), height / (width * s + height * c))
}

/** Normalised preview coordinates -> normalised frame coordinates. */
export function viewToFrame(frame: FrameSize, geometry: EditParams['geometry']): Affine {
  const { width: W, height: H } = frame
  const rotation = Math.abs(geometry.rotation_deg) < EPSILON_DEG ? 0 : geometry.rotation_deg
  const k = inscribedScale(W, H, rotation)
  const crop: CropRect = geometry.crop ?? { x: 0, y: 0, width: 1, height: 1 }
  const theta = (rotation * Math.PI) / 180
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  // view -> centred pixels of the straightened frame
  const toPixels: Affine = {
    a: W * k * crop.width,
    b: 0,
    c: 0,
    d: H * k * crop.height,
    e: W * k * (crop.x - 0.5),
    f: H * k * (crop.y - 0.5),
  }
  const rotate: Affine = { a: c, b: s, c: -s, d: c, e: 0, f: 0 }
  const toFrame = scale(1 / W, 1 / H, 0.5, 0.5)
  return compose(toFrame, compose(rotate, toPixels))
}

/**
 * Frame coordinates in units of the long edge -> container pixels.
 *
 * The unit the masks' radii are written in: in it a circle is a circle
 * whatever the frame's aspect. `box` is where the preview sits on screen.
 */
export function frameLongToScreen(
  frame: FrameSize,
  geometry: EditParams['geometry'],
  box: Box,
): Affine {
  const long = Math.max(frame.width, frame.height)
  const longToFrame = scale(long / frame.width, long / frame.height)
  const frameToView = invert(viewToFrame(frame, geometry))
  const viewToScreen = scale(box.width, box.height, box.left, box.top)
  return compose(viewToScreen, compose(frameToView, longToFrame))
}

/** Where an `object-contain` image actually sits inside its element. */
export function displayedBox(image: HTMLImageElement): Box | null {
  const { naturalWidth, naturalHeight, clientWidth, clientHeight, offsetLeft, offsetTop } = image
  if (!naturalWidth || !naturalHeight || !clientWidth || !clientHeight) return null
  const factor = Math.min(clientWidth / naturalWidth, clientHeight / naturalHeight)
  const width = naturalWidth * factor
  const height = naturalHeight * factor
  return {
    left: offsetLeft + (clientWidth - width) / 2,
    top: offsetTop + (clientHeight - height) / 2,
    width,
    height,
  }
}

/** `displayedBox`, kept current as the image loads and its element resizes. */
export function useDisplayedBox(
  imageRef: RefObject<HTMLImageElement | null>,
  key: unknown,
): Box | null {
  const [box, setBox] = useState<Box | null>(null)
  useEffect(() => {
    const image = imageRef.current
    if (!image) return
    const update = () =>
      setBox((previous) => {
        const next = displayedBox(image)
        const same =
          previous !== null &&
          next !== null &&
          previous.left === next.left &&
          previous.top === next.top &&
          previous.width === next.width &&
          previous.height === next.height
        return same ? previous : next
      })
    update()
    image.addEventListener('load', update)
    const observer = new ResizeObserver(update)
    observer.observe(image)
    return () => {
      image.removeEventListener('load', update)
      observer.disconnect()
    }
  }, [imageRef, key])
  return box
}

/**
 * The frame's size, from the neutral proxy: lens-corrected, upright, not
 * straightened nor cropped -- the frame itself, at a smaller scale. Only its
 * aspect matters.
 */
export function useFrameSize(proxyUrl: string | null): FrameSize | null {
  const [size, setSize] = useState<FrameSize | null>(null)
  useEffect(() => {
    setSize(null)
    if (!proxyUrl) return
    const image = new Image()
    let live = true
    image.onload = () => {
      if (live) setSize({ width: image.naturalWidth, height: image.naturalHeight })
    }
    image.src = proxyUrl
    return () => {
      live = false
    }
  }, [proxyUrl])
  return size
}
