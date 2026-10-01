// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The removals over the photo.
//
// With the correttivo in the hand (Q) a click makes a spot of the current
// size: the server looks for its source on the frame it has open, and the
// spot is added, selected, with both circles to drag. The result shows at the
// next preview: a spot is computed at render (docs/SPEC_rimozione.md 4.2).
//
// With the gomma magica (Shift+Q) the brush of the masks paints the area in
// red: a stroke released is a new eraser, or -- with an eraser selected -- a
// change of its area (Alt takes away). The edit is saved at once rather than
// after the two seconds of coalescence, so the worker starts on the fill while
// the pointer is still warm; until it has finished the area breathes over the
// photo as it was, with the job's progress.
import { useState, type PointerEvent as ReactPointerEvent } from 'react'
import { t } from '../../i18n/it'
import { apply, frameLongToScreen, invert, type Box, type FrameSize } from '../../lib/geometry'
import { longToFrame } from '../../lib/masks'
import { itemById, newErase, newHeal, withAddedItem, withItem } from '../../lib/retouch'
import { retouchApi } from '../../lib/retouchApi'
import type { EraseItem, HealItem } from '../../lib/retouchTypes'
import { toast } from '../../lib/toast'
import type { EditParams } from '../../lib/types'
import { BrushCanvas } from '../masks/BrushCanvas'
import type { MaskEditor } from '../masks/useMaskEditor'
import { HealHandles } from './HealHandles'
import { useFillProgress, useRetouchStates } from './useRetouch'

export function RetouchLayer({
  editor,
  photoId,
  params,
  onChange,
  flush,
  frame,
  box,
}: {
  editor: MaskEditor
  photoId: number
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  /** Write the edit now: the gesture that asks for a fill should not wait. */
  flush?: () => Promise<void>
  frame: FrameSize
  box: Box
}) {
  const [pending, setPending] = useState<[number, number] | null>(null)
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const { byId } = useRetouchStates(photoId, params)
  const progress = useFillProgress(photoId)
  const selected = itemById(params, editor.removal)
  const tool = editor.tool
  const toScreen = frameLongToScreen(frame, params.geometry, box)
  const zoom = Math.sqrt(Math.abs(toScreen.a * toScreen.d - toScreen.b * toScreen.c))
  const spots = params.retouch.filter((item): item is HealItem => item.kind === 'heal')
  const computing = params.retouch.filter(
    (item): item is EraseItem =>
      item.kind === 'erase' && item.visible &&
      ['computing', 'stale'].includes(byId.get(item.id)?.state ?? ''),
  )

  const place = async (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || pending) return
    const rect = event.currentTarget.getBoundingClientRect()
    const screen: [number, number] = [event.clientX - rect.left, event.clientY - rect.top]
    const [lx, ly] = apply(invert(toScreen), ...screen)
    const [cx, cy] = longToFrame(frame, lx, ly)
    if (cx < 0 || cx > 1 || cy < 0 || cy > 1) return
    setPending(screen)
    try {
      const spot = { cx, cy, radius: editor.healRadius }
      const source = await retouchApi.source(photoId, params, params.retouch.length, spot)
      const item = newHeal(params, spot, source)
      onChange(withAddedItem(params, item), true)
      editor.setRemoval(item.id)
    } catch (error) {
      toast((error as Error).message)
    } finally {
      setPending(null)
    }
  }

  const painted = (area: string) => {
    if (selected?.kind === 'erase') {
      onChange(withItem(params, { ...selected, area }), true)
    } else {
      const item = newErase(params, area)
      onChange(withAddedItem(params, item), true)
      editor.setRemoval(item.id)
    }
    void flush?.()
  }

  const clip = `inset(${box.top}px calc(100% - ${box.left + box.width}px) calc(100% - ${box.top + box.height}px) ${box.left}px)`
  const radius = Math.max(3, editor.healRadius * zoom)
  return (
    <>
      {computing.map((item) =>
        item.id === selected?.id && tool === 'erase' ? null : (
          <BrushCanvas key={item.id} raster={item.area} frame={frame} geometry={params.geometry}
            box={box} brush={editor.brush} busy setBusy={() => undefined} onPainted={() => undefined}
            onError={toast} readOnly pulse />
        ),
      )}
      {tool === 'erase' ? (
        <BrushCanvas raster={selected?.kind === 'erase' ? selected.area : null} frame={frame}
          geometry={params.geometry} box={box} brush={editor.brush} busy={editor.busy}
          setBusy={editor.setBusy} onPainted={painted} onError={toast}
          pulse={selected !== undefined && computing.some((item) => item.id === selected.id)}
          faint={selected?.kind === 'erase' && byId.get(selected.id)?.state === 'ready'} />
      ) : null}
      {tool === 'heal' ? (
        <div className="absolute inset-0" style={{ clipPath: clip, cursor: 'crosshair' }}
          onPointerDown={(event) => void place(event)}
          onPointerMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect()
            setCursor([event.clientX - rect.left, event.clientY - rect.top])
          }}
          onPointerLeave={() => setCursor(null)}>
          <svg className="pointer-events-none absolute inset-0 h-full w-full">
            {cursor && !pending ? (
              <circle cx={cursor[0]} cy={cursor[1]} r={radius} fill="none"
                className="stroke-ink-50" strokeWidth={1.25}
                style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }} />
            ) : null}
            {pending ? (
              <circle cx={pending[0]} cy={pending[1]} r={radius} fill="none"
                className="animate-pulse stroke-ink-50" strokeWidth={2} />
            ) : null}
          </svg>
        </div>
      ) : null}
      {spots.length > 0 && (tool === 'heal' || selected?.kind === 'heal') ? (
        <HealHandles spots={spots} selected={editor.removal} all={tool === 'heal'} frame={frame}
          geometry={params.geometry} box={box} onSelect={editor.setRemoval}
          onChange={(item, commit) => onChange(withItem(params, item), commit)} />
      ) : null}
      {computing.length > 0 ? (
        <div className="pointer-events-none absolute left-5 top-12 rounded bg-ink-950/80 px-2 py-0.5 text-xs text-ink-100">
          {t('retouch.computing')}
          {progress !== null ? ` ${Math.round(progress * 100)}%` : ''}
        </div>
      ) : null}
    </>
  )
}
