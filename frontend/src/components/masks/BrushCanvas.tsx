// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Painting a mask.
//
// The canvas lives in the *frame*, like every mask: RASTER_EDGE pixels on the
// long edge, at the frame's aspect, and carried onto the straightened, cropped
// preview by a CSS transform built from lib/geometry.ts. Painting therefore
// lands where the pipeline will read it, whatever the rotation, and a crop
// changed later does not move what was painted.
//
// Each stroke ends in an upload: the canvas as a PNG whose alpha is the
// selection (backend/ape/masks_store.py keeps the alpha), named by its content.
// The owner then names the new raster -- a brush mask's definition, a magic
// eraser's area -- as one step of the undo history. Undoing brings back the
// previous name, and the canvas reloads it; no name is an empty canvas.
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import {
  apply,
  compose,
  cssMatrix,
  frameLongToScreen,
  invert,
  scale,
  type Box,
  type FrameSize,
} from '../../lib/geometry'
import { rasterUrl, uploadRaster } from '../../lib/masksApi'
import type { EditParams } from '../../lib/types'
import type { BrushSettings } from './useMaskEditor'

/** Long edge of a painted raster: finer than a mouse paints, a few hundred KB as PNG. */
export const RASTER_EDGE = 2048
const PAINT = [255, 48, 48] as const

export function rasterSize(frame: FrameSize): { width: number; height: number } {
  const long = Math.max(frame.width, frame.height)
  return {
    width: Math.max(1, Math.round((RASTER_EDGE * frame.width) / long)),
    height: Math.max(1, Math.round((RASTER_EDGE * frame.height) / long)),
  }
}

function toPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG'))), 'image/png'),
  )
}

/** An empty raster, uploaded: what a new brush mask starts from. */
export async function uploadEmptyRaster(frame: FrameSize): Promise<string> {
  const { width, height } = rasterSize(frame)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return uploadRaster(await toPng(canvas))
}

export function BrushCanvas({
  raster,
  frame,
  geometry,
  box,
  brush,
  busy,
  setBusy,
  onPainted,
  onError,
  readOnly = false,
  pulse = false,
  faint = false,
}: {
  /** The raster on the canvas, or null to start empty. */
  raster: string | null
  frame: FrameSize
  geometry: EditParams['geometry']
  box: Box
  brush: BrushSettings
  busy: boolean
  setBusy: (busy: boolean) => void
  /** A stroke ended and was uploaded under this name. */
  onPainted: (name: string) => void
  onError: (message: string) => void
  /** Shown, not painted on: an eraser's area while its fill is computed. */
  readOnly?: boolean
  /** Breathe: something is being computed for this area. */
  pulse?: boolean
  /** Barely there unless a stroke is under way: the result is what matters. */
  faint?: boolean
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const layer = useRef<HTMLDivElement>(null)
  /** The raster the canvas holds now: painted here, or loaded. */
  const shown = useRef<string | null>(null)
  const last = useRef<[number, number] | null>(null)
  const erasing = useRef(false)
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const [painting, setPainting] = useState(false)

  const { width, height } = rasterSize(frame)
  // canvas pixels -> frame long-edge units -> screen
  const long = Math.max(frame.width, frame.height)
  const canvasToLong = scale(frame.width / (long * width), frame.height / (long * height))
  const toScreen = compose(frameLongToScreen(frame, geometry, box), canvasToLong)
  const toCanvas = invert(toScreen)
  const pxPerLong = width / (frame.width / long)
  const zoom = Math.sqrt(Math.abs(toScreen.a * toScreen.d - toScreen.b * toScreen.c))

  // Load the raster named, unless it is the one just painted; none, empty.
  useEffect(() => {
    const element = canvas.current
    if (!element || raster === shown.current) return
    if (!raster) {
      element.getContext('2d')!.clearRect(0, 0, width, height)
      shown.current = null
      return
    }
    let live = true
    const image = new Image()
    image.onload = () => {
      if (!live) return
      const context = element.getContext('2d')!
      const scratch = document.createElement('canvas')
      scratch.width = width
      scratch.height = height
      const reading = scratch.getContext('2d')!
      reading.drawImage(image, 0, 0, width, height)
      // Stored as one grey channel: the grey is the alpha of the paint.
      const pixels = reading.getImageData(0, 0, width, height)
      const data = pixels.data
      for (let i = 0; i < data.length; i += 4) {
        const value = data[i]
        data[i] = PAINT[0]
        data[i + 1] = PAINT[1]
        data[i + 2] = PAINT[2]
        data[i + 3] = value
      }
      context.clearRect(0, 0, width, height)
      context.putImageData(pixels, 0, 0)
      shown.current = raster
    }
    image.onerror = () => onError(`maschera ${raster.slice(0, 8)} non leggibile`)
    image.src = rasterUrl(raster)
    return () => {
      live = false
    }
  }, [raster, width, height, onError])

  const at = (event: ReactPointerEvent): [number, number] => {
    const rect = layer.current!.getBoundingClientRect()
    return apply(toCanvas, event.clientX - rect.left, event.clientY - rect.top)
  }

  const dab = (context: CanvasRenderingContext2D, x: number, y: number) => {
    const radius = Math.max(1, brush.size * pxPerLong)
    const hard = Math.min(0.98, Math.max(0, 1 - brush.softness))
    const gradient = context.createRadialGradient(x, y, radius * hard, x, y, radius)
    const colour = `${PAINT[0]}, ${PAINT[1]}, ${PAINT[2]}`
    gradient.addColorStop(0, `rgba(${colour}, ${brush.flow})`)
    gradient.addColorStop(1, `rgba(${colour}, 0)`)
    context.fillStyle = gradient
    context.beginPath()
    context.arc(x, y, radius, 0, Math.PI * 2)
    context.fill()
  }

  const strokeTo = (point: [number, number]) => {
    const context = canvas.current!.getContext('2d')!
    context.globalCompositeOperation = erasing.current ? 'destination-out' : 'source-over'
    const from = last.current ?? point
    const spacing = Math.max(1, brush.size * pxPerLong * 0.25)
    const distance = Math.hypot(point[0] - from[0], point[1] - from[1])
    const steps = Math.max(1, Math.ceil(distance / spacing))
    for (let i = last.current ? 1 : 0; i <= steps; i += 1) {
      const k = i / steps
      dab(context, from[0] + (point[0] - from[0]) * k, from[1] + (point[1] - from[1]) * k)
    }
    last.current = point
  }

  const down = (event: ReactPointerEvent) => {
    if (busy || event.button !== 0) return
    event.preventDefault()
    layer.current!.setPointerCapture(event.pointerId)
    erasing.current = brush.erase !== event.altKey
    last.current = null
    setPainting(true)
    strokeTo(at(event))
  }

  const moveCursor = (event: ReactPointerEvent) => {
    const rect = layer.current!.getBoundingClientRect()
    setCursor([event.clientX - rect.left, event.clientY - rect.top])
    if (last.current) strokeTo(at(event))
  }

  const up = async () => {
    if (!last.current) return
    last.current = null
    setPainting(false)
    setBusy(true)
    try {
      const name = await uploadRaster(await toPng(canvas.current!))
      shown.current = name
      onPainted(name)
    } catch (error) {
      onError((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const clip = `inset(${box.top}px calc(100% - ${box.left + box.width}px) calc(100% - ${box.top + box.height}px) ${box.left}px)`
  const handlers = readOnly
    ? {}
    : {
        onPointerDown: down,
        onPointerMove: moveCursor,
        onPointerUp: () => void up(),
        onPointerCancel: () => void up(),
        onPointerLeave: () => setCursor(null),
      }
  return (
    <div
      ref={layer}
      className={readOnly ? 'pointer-events-none absolute inset-0' : 'absolute inset-0 touch-none'}
      style={{ clipPath: clip, cursor: readOnly ? undefined : 'none' }}
      {...handlers}
    >
      <canvas
        ref={canvas}
        width={width}
        height={height}
        className={`pointer-events-none absolute left-0 top-0 ${faint && !painting ? 'opacity-15' : 'opacity-45'}${pulse ? ' animate-pulse' : ''}`}
        style={{
          width,
          height,
          transformOrigin: '0 0',
          transform: cssMatrix(toScreen),
        }}
      />
      {cursor ? (
        <svg className="pointer-events-none absolute inset-0 h-full w-full">
          <circle cx={cursor[0]} cy={cursor[1]} r={Math.max(2, brush.size * zoom * pxPerLong)}
            fill="none" className="stroke-ink-50" strokeWidth={1.25}
            strokeDasharray={brush.erase ? '4 3' : undefined}
            style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }} />
        </svg>
      ) : null}
    </div>
  )
}
