// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The crop being drawn, over the whole straightened frame: drag inside to
// move it, a corner to size it -- along the chosen ratio, the opposite corner
// staying put -- and, only when the ratio is free, a side.
//
// It is placed on the frame's own proportions inside the image element, not
// on the picture currently shown: opening the tool asks for a render without
// the crop, and until it arrives the picture on screen is still the cropped
// one. The frame is where that render will be.
//
// While it is up, Enter applies the crop and Escape drops it -- caught before
// any other shortcut of the page, so Escape does not also leave a mask.
import { useEffect, useRef, useState, type PointerEvent, type RefObject } from 'react'
import type { Box } from '../../lib/geometry'
import { dragRect, ratioLabel, rectRatio, type CropHandle } from '../../lib/crop'
import type { CropRect } from '../../lib/analysisTypes'
import type { CropEditor } from './useCropEditor'

/** Where a frame of `aspect` sits inside an `object-contain` image element. */
function useFittedBox(
  imageRef: RefObject<HTMLImageElement | null>,
  aspect: number | null,
): Box | null {
  const [box, setBox] = useState<Box | null>(null)
  useEffect(() => {
    const image = imageRef.current
    if (!image || aspect === null) return setBox(null)
    const update = () => {
      const { clientWidth, clientHeight, offsetLeft, offsetTop } = image
      if (!clientWidth || !clientHeight) return setBox(null)
      const height = Math.min(clientHeight, clientWidth / aspect)
      const width = height * aspect
      setBox({
        left: offsetLeft + (clientWidth - width) / 2,
        top: offsetTop + (clientHeight - height) / 2,
        width,
        height,
      })
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(image)
    return () => observer.disconnect()
  }, [imageRef, aspect])
  return box
}

const CORNERS: CropHandle[] = ['nw', 'ne', 'sw', 'se']
const SIDES: CropHandle[] = ['n', 's', 'w', 'e']

const CURSOR: Record<CropHandle, string> = {
  nw: 'nwse-resize',
  se: 'nwse-resize',
  ne: 'nesw-resize',
  sw: 'nesw-resize',
  n: 'ns-resize',
  s: 'ns-resize',
  w: 'ew-resize',
  e: 'ew-resize',
  move: 'move',
}

/** A handle's place on the frame, in percent of its sides. */
function placeOf(handle: CropHandle): { left: string; top: string } {
  const left = handle.includes('w') ? '0%' : handle.includes('e') ? '100%' : '50%'
  const top = handle.includes('n') ? '0%' : handle.includes('s') ? '100%' : '50%'
  return { left, top }
}

interface Drag {
  handle: CropHandle
  x: number
  y: number
  rect: CropRect
}

/** Enter and Escape for the open crop, ahead of the page's own shortcuts. */
function useCropKeys(editor: CropEditor) {
  const { active, apply, cancel } = editor
  useEffect(() => {
    if (!active) return
    const down = (event: KeyboardEvent) => {
      if (event.key !== 'Enter' && event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      if (event.key === 'Enter') apply()
      else cancel()
    }
    window.addEventListener('keydown', down, true)
    return () => window.removeEventListener('keydown', down, true)
  }, [active, apply, cancel])
}

export function CropTool({
  editor,
  imageRef,
}: {
  editor: CropEditor
  imageRef: RefObject<HTMLImageElement | null>
}) {
  const box = useFittedBox(imageRef, editor.active ? editor.aspect : null)
  const drag = useRef<Drag | null>(null)
  useCropKeys(editor)
  const aspect = editor.aspect
  if (!box || !editor.active || aspect === null) return null

  const handlers = (handle: CropHandle) => ({
    onPointerDown: (event: PointerEvent) => {
      event.preventDefault()
      event.stopPropagation()
      ;(event.currentTarget as Element).setPointerCapture(event.pointerId)
      drag.current = { handle, x: event.clientX, y: event.clientY, rect: editor.draft }
    },
    onPointerMove: (event: PointerEvent) => {
      const current = drag.current
      if (!current) return
      const dx = (event.clientX - current.x) / box.width
      const dy = (event.clientY - current.y) / box.height
      editor.setDraft(dragRect(current.rect, current.handle, dx, dy, editor.ratio, aspect))
    },
    onPointerUp: () => {
      drag.current = null
    },
    onPointerCancel: () => {
      drag.current = null
    },
  })

  const { draft } = editor
  const handles = editor.ratio === null ? [...CORNERS, ...SIDES] : CORNERS

  return (
    <div className="pointer-events-none absolute" style={box}>
      <div
        className="pointer-events-auto absolute touch-none border border-ink-50"
        style={{
          left: `${draft.x * 100}%`,
          top: `${draft.y * 100}%`,
          width: `${draft.width * 100}%`,
          height: `${draft.height * 100}%`,
          cursor: CURSOR.move,
          // The outside dimmed with one shadow, as the proposal is drawn.
          boxShadow: '0 0 0 9999px rgba(10, 10, 10, 0.6)',
        }}
        {...handlers('move')}
      >
        {[1, 2].map((i) => (
          <span key={`v${i}`} className="pointer-events-none absolute inset-y-0 w-px bg-ink-50/40"
            style={{ left: `${(i * 100) / 3}%` }} />
        ))}
        {[1, 2].map((i) => (
          <span key={`h${i}`} className="pointer-events-none absolute inset-x-0 h-px bg-ink-50/40"
            style={{ top: `${(i * 100) / 3}%` }} />
        ))}
        <span className="pointer-events-none absolute left-1 top-1 rounded bg-ink-950/75 px-1.5 py-0.5 text-xs tabular-nums text-ink-100">
          {ratioLabel(rectRatio(draft, aspect))}
        </span>
        {handles.map((handle) => (
          <span
            key={handle}
            className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 touch-none rounded-sm border border-ink-900 bg-ink-50"
            style={{ ...placeOf(handle), cursor: CURSOR[handle] }}
            {...handlers(handle)}
          />
        ))}
      </div>
    </div>
  )
}
