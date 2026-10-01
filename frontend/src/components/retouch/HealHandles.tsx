// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The spots over the photo: for each, the destination (solid) and the source
// (dashed) joined by a line, as Lightroom draws them. With the correttivo in
// the hand every spot shows; otherwise only the selected one.
//
// Drawn in screen pixels from frame coordinates (lib/geometry.ts), like the
// masks' handles, and dragged the same way: continuously while the pointer
// moves, once more on release -- one undo step per gesture. Dragging the
// source makes it the user's: `source_auto` goes false.
import { useRef, type PointerEvent as ReactPointerEvent } from 'react'
import { apply, frameLongToScreen, invert, type Box, type FrameSize } from '../../lib/geometry'
import { frameToLong, longToFrame } from '../../lib/masks'
import type { HealItem } from '../../lib/retouchTypes'
import type { EditParams } from '../../lib/types'

interface Drag {
  id: string
  end: 'target' | 'source'
  pointer: [number, number]
  item: HealItem
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

export function HealHandles({
  spots,
  selected,
  all,
  frame,
  geometry,
  box,
  onSelect,
  onChange,
}: {
  spots: HealItem[]
  selected: string | null
  /** Every spot, not just the selected one: the tool is in the hand. */
  all: boolean
  frame: FrameSize
  geometry: EditParams['geometry']
  box: Box
  onSelect: (id: string) => void
  onChange: (item: HealItem, commit: boolean) => void
}) {
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<Drag | null>(null)
  const toScreen = frameLongToScreen(frame, geometry, box)
  const toLong = invert(toScreen)
  const zoom = Math.sqrt(Math.abs(toScreen.a * toScreen.d - toScreen.b * toScreen.c))

  const pointer = (event: ReactPointerEvent): [number, number] => {
    const rect = svg.current!.getBoundingClientRect()
    return apply(toLong, event.clientX - rect.left, event.clientY - rect.top)
  }

  const start = (item: HealItem, end: Drag['end']) => (event: ReactPointerEvent) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    ;(event.target as Element).setPointerCapture(event.pointerId)
    onSelect(item.id)
    drag.current = { id: item.id, end, pointer: pointer(event), item }
  }

  const move = (event: ReactPointerEvent, commit: boolean) => {
    const current = drag.current
    if (!current) return
    if (commit) drag.current = null
    const [px, py] = pointer(event)
    const shift = [px - current.pointer[0], py - current.pointer[1]]
    const moved = (x: number, y: number): [number, number] => {
      const [lx, ly] = frameToLong(frame, x, y)
      const [fx, fy] = longToFrame(frame, lx + shift[0], ly + shift[1])
      return [clamp01(fx), clamp01(fy)]
    }
    const item = current.item
    if (current.end === 'target') {
      const [cx, cy] = moved(item.cx, item.cy)
      onChange({ ...item, cx, cy }, commit)
    } else {
      const [sx, sy] = moved(item.sx, item.sy)
      onChange({ ...item, sx, sy, source_auto: false }, commit)
    }
  }

  const handlers = {
    onPointerMove: (event: ReactPointerEvent) => move(event, false),
    onPointerUp: (event: ReactPointerEvent) => move(event, true),
    onPointerCancel: (event: ReactPointerEvent) => move(event, true),
  }

  const shown = all ? spots : spots.filter((spot) => spot.id === selected)
  return (
    <svg ref={svg} className="pointer-events-none absolute inset-0 h-full w-full overflow-hidden"
      style={{ clipPath: `inset(${box.top}px calc(100% - ${box.left + box.width}px) calc(100% - ${box.top + box.height}px) ${box.left}px)` }}>
      <g style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }}>
        {shown.map((spot) => {
          const active = spot.id === selected
          const target = apply(toScreen, ...frameToLong(frame, spot.cx, spot.cy))
          const source = apply(toScreen, ...frameToLong(frame, spot.sx, spot.sy))
          const r = Math.max(3, spot.radius * zoom)
          const tone = active ? 'stroke-ink-50' : 'stroke-ink-50/55'
          const hidden = !spot.visible ? 0.45 : 1
          return (
            <g key={spot.id} opacity={hidden}>
              <line x1={target[0]} y1={target[1]} x2={source[0]} y2={source[1]}
                className={tone} strokeWidth={1} />
              <circle cx={source[0]} cy={source[1]} r={r} fill="transparent" className={tone}
                strokeWidth={active ? 1.5 : 1} strokeDasharray="5 4"
                style={{ pointerEvents: active ? 'all' : 'none', cursor: 'grab' }}
                onPointerDown={start(spot, 'source')} {...handlers} />
              <circle cx={target[0]} cy={target[1]} r={r} fill="transparent" className={tone}
                strokeWidth={active ? 2 : 1.25}
                style={{ pointerEvents: 'all', cursor: 'grab' }}
                onPointerDown={start(spot, 'target')} {...handlers} />
            </g>
          )
        })}
      </g>
    </svg>
  )
}
