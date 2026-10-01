// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Two or four frames side by side, zoomed together (section 7.6): "è il modo in
// cui si decide davvero tra due scatti quasi identici".
//
// One view state is shared by every pane -- a zoom and a centre, both relative
// to the image -- so panning one pans all of them and the same eyelash is
// under the same pixel in each. The zoom starts at "fit" and goes to 1:1 on the
// camera's focus point (or, when the camera recorded none, the sharpest region
// the analysis found): the point the photographer chose is the point worth
// checking.
//
// 1:1 here is the embedded preview's own pixels, 1616 px across on the A7 III.
// That is a quarter of the sensor's resolution: enough to tell a closed eye
// from an open one and a sharp frame from a shaken one, which is the decision
// being made. The full RAW is decoded only for editing (section 7.2).
import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Maximize2, X, ZoomIn } from 'lucide-react'
import { thumbUrl } from '../../lib/api'
import { focusOf, percent } from '../../lib/culling'
import type { CullPhoto, UserDecision } from '../../lib/cullTypes'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { reasonLabel } from './CullCard'

interface ViewState {
  /** ``null`` is "fit the pane"; a number is image pixels per screen pixel. */
  zoom: number | null
  centre: [number, number]
}

function Pane({
  photo,
  view,
  onPan,
  onZoomHere,
  onDecide,
}: {
  photo: CullPhoto
  view: ViewState
  onPan: (dx: number, dy: number, width: number, height: number, scale: number) => void
  onZoomHere: (point: [number, number]) => void
  onDecide: (decision: UserDecision) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 1, height: 1 })
  const [natural, setNatural] = useState({ width: 0, height: 0 })
  const drag = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const element = box.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const fit =
    natural.width > 0 ? Math.min(size.width / natural.width, size.height / natural.height) : 1
  // 1:1 means one preview pixel per *device* pixel, not per CSS pixel.
  const scale = view.zoom === null ? fit : view.zoom / window.devicePixelRatio
  const [cx, cy] = view.zoom === null ? [0.5, 0.5] : view.centre
  const tx = size.width / 2 - cx * natural.width * scale
  const ty = size.height / 2 - cy * natural.height * scale

  return (
    <div className="flex min-h-0 min-w-0 flex-col border border-ink-700 bg-mat">
      <div
        ref={box}
        className={cn(
          'relative min-h-0 flex-1 overflow-hidden',
          view.zoom === null ? 'cursor-zoom-in' : 'cursor-grab active:cursor-grabbing',
        )}
        onDoubleClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect()
          const x = (event.clientX - rect.left - tx) / (natural.width * scale)
          const y = (event.clientY - rect.top - ty) / (natural.height * scale)
          onZoomHere([Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))])
        }}
        onPointerDown={(event) => {
          drag.current = { x: event.clientX, y: event.clientY }
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (!drag.current || view.zoom === null) return
          onPan(
            event.clientX - drag.current.x,
            event.clientY - drag.current.y,
            natural.width,
            natural.height,
            scale,
          )
          drag.current = { x: event.clientX, y: event.clientY }
        }}
        onPointerUp={() => {
          drag.current = null
        }}
      >
        <img
          src={thumbUrl(photo, 'full')}
          alt={photo.filename}
          draggable={false}
          onLoad={(event) =>
            setNatural({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })
          }
          className="absolute left-0 top-0 max-w-none origin-top-left select-none"
          style={{
            width: natural.width || undefined,
            height: natural.height || undefined,
            transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
            opacity: natural.width ? 1 : 0,
          }}
        />
      </div>
      <footer className="flex items-center gap-2 border-t border-ink-700 bg-ink-900 px-2 py-1 text-[11px] text-ink-300">
        <span className="truncate text-ink-100">{photo.filename}</span>
        <span className="tabular-nums">{percent(photo.score)}</span>
        <span className="truncate">{photo.reasons.map(reasonLabel).join(' · ')}</span>
        <span className="ml-auto flex gap-1">
          {photo.culled ? (
            <Button size="sm" variant="outline" onClick={() => onDecide('keep')}>
              <Check size={12} />
              {t('culling.action.keep')}
            </Button>
          ) : (
            <Button size="sm" variant="ghost" onClick={() => onDecide('discard')}>
              <X size={12} />
              {t('culling.action.discard')}
            </Button>
          )}
        </span>
      </footer>
    </div>
  )
}

export function CompareView({
  photos,
  onDecide,
  onClose,
}: {
  photos: CullPhoto[]
  onDecide: (photo: CullPhoto, decision: UserDecision) => void
  onClose: () => void
}) {
  const [view, setView] = useState<ViewState>({ zoom: null, centre: focusOf(photos[0]) })

  const zoomTo = useCallback((centre: [number, number]) => {
    setView((current) => (current.zoom === null ? { zoom: 1, centre } : { zoom: null, centre }))
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'z' || event.key === 'Z') {
        event.preventDefault()
        zoomTo(focusOf(photos[0]))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [photos, zoomTo])

  const pan = (dx: number, dy: number, width: number, height: number, scale: number) => {
    if (!width || !height) return
    setView((current) => ({
      ...current,
      centre: [
        Math.min(1, Math.max(0, current.centre[0] - dx / (width * scale))),
        Math.min(1, Math.max(0, current.centre[1] - dy / (height * scale))),
      ],
    }))
  }

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-ink-950/95 p-3">
      <header className="mb-2 flex items-center gap-2 text-xs text-ink-300">
        <span className="text-sm text-ink-100">{t('culling.compare.title')}</span>
        <span>{t('culling.compare.hint')}</span>
        <Button size="sm" variant="ghost" className="ml-auto" onClick={() => zoomTo(focusOf(photos[0]))}>
          {view.zoom === null ? <ZoomIn size={13} /> : <Maximize2 size={13} />}
          {view.zoom === null ? t('culling.compare.actual') : t('culling.compare.fit')}
        </Button>
        <Button size="sm" variant="outline" onClick={onClose}>
          <X size={13} />
          {t('common.close')}
        </Button>
      </header>
      <div
        className={cn(
          'grid min-h-0 flex-1 gap-2',
          photos.length > 2 ? 'grid-cols-2 grid-rows-2' : 'grid-cols-2 grid-rows-1',
        )}
      >
        {photos.map((photo) => (
          <Pane
            key={photo.id}
            photo={photo}
            view={view}
            onPan={pan}
            onZoomHere={zoomTo}
            onDecide={(decision) => onDecide(photo, decision)}
          />
        ))}
      </div>
    </div>
  )
}
