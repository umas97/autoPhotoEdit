// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The handles of a linear or radial mask, over the photo.
//
// Drawn in screen pixels from the mask's frame coordinates (lib/geometry.ts),
// so they sit on the straightened, cropped preview exactly where the
// pipeline evaluates the mask. A drag reports like a slider: continuously
// while the pointer moves, once more when it is released -- one undo step per
// gesture.
//
// Linear: the solid line is where the effect is full, the dashed one where it
// has faded to nothing; drag either end, or the middle to move both. Radial:
// the outer ellipse is the edge, the dashed one where the feathering starts;
// drag the centre to move it, the two axis handles to size it, the hollow one
// to turn it.
import { useRef, type PointerEvent as ReactPointerEvent } from 'react'
import { apply, frameLongToScreen, invert, type Box, type FrameSize } from '../../lib/geometry'
import { frameToLong, longToFrame, withDefinition } from '../../lib/masks'
import type { MaskParams } from '../../lib/maskTypes'
import type { EditParams } from '../../lib/types'

type Handle = 'start' | 'end' | 'line' | 'centre' | 'rx' | 'ry' | 'turn'

interface Drag {
  handle: Handle
  pointer: [number, number]
  mask: MaskParams
}

/** The server's bounds (mask_defs.Position, the radii). */
const clampPosition = (value: number) => Math.min(3, Math.max(-2, value))
const clampRadius = (value: number) => Math.min(3, Math.max(0.005, value))

export function ShapeHandles({
  mask,
  frame,
  geometry,
  box,
  onChange,
}: {
  mask: MaskParams
  frame: FrameSize
  geometry: EditParams['geometry']
  box: Box
  onChange: (mask: MaskParams, commit: boolean) => void
}) {
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<Drag | null>(null)
  const toScreen = frameLongToScreen(frame, geometry, box)
  const toLong = invert(toScreen)

  /** The pointer in frame long-edge units. */
  const pointer = (event: ReactPointerEvent): [number, number] => {
    const rect = svg.current!.getBoundingClientRect()
    return apply(toLong, event.clientX - rect.left, event.clientY - rect.top)
  }

  const start = (handle: Handle) => (event: ReactPointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    ;(event.target as Element).setPointerCapture(event.pointerId)
    drag.current = { handle, pointer: pointer(event), mask }
  }

  const move = (event: ReactPointerEvent, commit: boolean) => {
    const current = drag.current
    if (!current) return
    if (commit) drag.current = null
    const [px, py] = pointer(event)
    const shift = [px - current.pointer[0], py - current.pointer[1]]
    const d = current.mask.definition
    const moved = (x: number, y: number): [number, number] => {
      const [lx, ly] = frameToLong(frame, x, y)
      const [fx, fy] = longToFrame(frame, lx + shift[0], ly + shift[1])
      return [clampPosition(fx), clampPosition(fy)]
    }

    let patch: MaskParams['definition'] = {}
    if (current.mask.kind === 'linear') {
      const [x0, y0] = moved(d.x0 ?? 0.5, d.y0 ?? 0)
      const [x1, y1] = moved(d.x1 ?? 0.5, d.y1 ?? 0.5)
      if (current.handle === 'start') patch = { x0, y0 }
      else if (current.handle === 'end') patch = { x1, y1 }
      else patch = { x0, y0, x1, y1 }
      // The server refuses a gradient with both ends in one place.
      const next = { ...d, ...patch }
      if (Math.abs(next.x1! - next.x0!) < 1e-4 && Math.abs(next.y1! - next.y0!) < 1e-4) return
    } else {
      const [cx, cy] = frameToLong(frame, d.cx ?? 0.5, d.cy ?? 0.5)
      const theta = ((d.angle ?? 0) * Math.PI) / 180
      const w = [px - cx, py - cy]
      if (current.handle === 'centre') {
        const [ncx, ncy] = moved(d.cx ?? 0.5, d.cy ?? 0.5)
        patch = { cx: ncx, cy: ncy }
      } else if (current.handle === 'rx') {
        patch = { rx: clampRadius(Math.abs(w[0] * Math.cos(theta) - w[1] * Math.sin(theta))) }
      } else if (current.handle === 'ry') {
        patch = { ry: clampRadius(Math.abs(w[0] * Math.sin(theta) + w[1] * Math.cos(theta))) }
      } else {
        // The major axis runs along (cos, -sin): counter-clockwise on screen.
        const angle = (Math.atan2(-w[1], w[0]) * 180) / Math.PI
        patch = { angle: Math.round(angle * 10) / 10 }
      }
    }
    onChange(withDefinition(current.mask, patch), commit)
  }

  const handlers = {
    onPointerMove: (event: ReactPointerEvent) => move(event, false),
    onPointerUp: (event: ReactPointerEvent) => move(event, true),
    onPointerCancel: (event: ReactPointerEvent) => move(event, true),
  }

  const d = mask.definition
  const at = (x: number, y: number) => apply(toScreen, ...frameToLong(frame, x, y))

  let body: JSX.Element
  if (mask.kind === 'linear') {
    const s0 = at(d.x0 ?? 0.5, d.y0 ?? 0)
    const s1 = at(d.x1 ?? 0.5, d.y1 ?? 0.5)
    const dx = s1[0] - s0[0]
    const dy = s1[1] - s0[1]
    const length = Math.hypot(dx, dy) || 1
    // Perpendicular, long enough to cross any screen; the SVG clips it.
    const n = [(-dy / length) * 4000, (dx / length) * 4000]
    const across = (p: [number, number], dashed: boolean, handle: Handle) => (
      <>
        <line x1={p[0] - n[0]} y1={p[1] - n[1]} x2={p[0] + n[0]} y2={p[1] + n[1]}
          className="stroke-ink-50" strokeWidth={1.25} strokeDasharray={dashed ? '6 5' : undefined} />
        <line x1={p[0] - n[0]} y1={p[1] - n[1]} x2={p[0] + n[0]} y2={p[1] + n[1]}
          stroke="transparent" strokeWidth={12} className="cursor-move" style={{ pointerEvents: 'stroke' }}
          onPointerDown={start(handle)} {...handlers} />
      </>
    )
    body = (
      <>
        {across(s0, false, 'start')}
        {across(s1, true, 'end')}
        <line x1={s0[0]} y1={s0[1]} x2={s1[0]} y2={s1[1]} className="stroke-ink-50/70" strokeWidth={1} />
        <Dot at={s0} onPointerDown={start('start')} handlers={handlers} />
        <Dot at={s1} onPointerDown={start('end')} handlers={handlers} />
        <Dot at={[(s0[0] + s1[0]) / 2, (s0[1] + s1[1]) / 2]} onPointerDown={start('line')}
          handlers={handlers} large />
      </>
    )
  } else {
    const [cx, cy] = frameToLong(frame, d.cx ?? 0.5, d.cy ?? 0.5)
    const theta = ((d.angle ?? 0) * Math.PI) / 180
    const rx = d.rx ?? 0.25
    const ry = d.ry ?? 0.25
    const u: [number, number] = [Math.cos(theta), -Math.sin(theta)]
    const v: [number, number] = [Math.sin(theta), Math.cos(theta)]
    const centre = apply(toScreen, cx, cy)
    const end = (axis: [number, number], r: number) => apply(toScreen, cx + axis[0] * r, cy + axis[1] * r)
    const U = end(u, rx)
    const V = end(v, ry)
    const turn = end(u, rx * 1.18)
    const zoom = Math.sqrt(Math.abs(toScreen.a * toScreen.d - toScreen.b * toScreen.c))
    const degrees = (Math.atan2(U[1] - centre[1], U[0] - centre[0]) * 180) / Math.PI
    const inner = 1 - (d.feather ?? 0.5)
    const ellipse = (k: number, dashed: boolean) => (
      <ellipse cx={centre[0]} cy={centre[1]} rx={rx * zoom * k} ry={ry * zoom * k}
        transform={`rotate(${degrees} ${centre[0]} ${centre[1]})`} fill="none"
        className="stroke-ink-50" strokeWidth={1.25} strokeDasharray={dashed ? '6 5' : undefined} />
    )
    body = (
      <>
        {ellipse(1, false)}
        {inner > 0.02 ? ellipse(inner, true) : null}
        <line x1={U[0]} y1={U[1]} x2={turn[0]} y2={turn[1]} className="stroke-ink-50/70" strokeWidth={1} />
        <Dot at={centre} onPointerDown={start('centre')} handlers={handlers} large />
        <Dot at={U} onPointerDown={start('rx')} handlers={handlers} />
        <Dot at={V} onPointerDown={start('ry')} handlers={handlers} />
        <Dot at={turn} onPointerDown={start('turn')} handlers={handlers} hollow />
      </>
    )
  }

  return (
    <svg ref={svg} className="pointer-events-none absolute inset-0 h-full w-full overflow-hidden"
      style={{ clipPath: `inset(${box.top}px calc(100% - ${box.left + box.width}px) calc(100% - ${box.top + box.height}px) ${box.left}px)` }}>
      <g style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }}>{body}</g>
    </svg>
  )
}

function Dot({
  at,
  onPointerDown,
  handlers,
  large,
  hollow,
}: {
  at: [number, number]
  onPointerDown: (event: ReactPointerEvent) => void
  handlers: Record<string, (event: ReactPointerEvent) => void>
  large?: boolean
  hollow?: boolean
}) {
  return (
    <circle cx={at[0]} cy={at[1]} r={large ? 6 : 5}
      className={hollow ? 'fill-ink-950/40 stroke-ink-50' : 'fill-ink-50 stroke-ink-950'}
      strokeWidth={1.5} style={{ pointerEvents: 'all', cursor: 'grab' }}
      onPointerDown={onPointerDown} {...handlers} />
  )
}
