// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The left column of the review (section 10): the scenes, each shown by its
// representative, or the individual queue, lowest confidence first, each with
// its first reason. The thumbnails show each photo developed as its current
// version will come out (the neutral proxy until that render is ready), so a
// photo that strays from the rest of its scene is seen in the strip.
import { useEffect, useRef } from 'react'
import { developedUrl } from '../../lib/reviewApi'
import type { ReviewPhoto, ReviewQueueEntry, ReviewScene } from '../../lib/reviewTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { ConfidenceBadge, reasonLabel } from './ConfidenceCard'

/** A photo developed small, or nothing while it has no image at all. */
export function DevelopedImage({ photo, className }: { photo: ReviewPhoto | undefined; className?: string }) {
  const src = photo ? developedUrl(photo) : null
  return src ? <img src={src} alt={photo?.filename} loading="lazy" className={className} /> : null
}

function Thumb({ photo }: { photo: ReviewPhoto | undefined }) {
  return (
    <div className="h-12 w-16 shrink-0 overflow-hidden rounded-sm border border-ink-700 bg-mat">
      <DevelopedImage photo={photo} className="h-full w-full object-cover" />
    </div>
  )
}

function useScrollIntoView(selected: boolean) {
  const ref = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: 'nearest' })
  }, [selected])
  return ref
}

function SceneRow({
  scene,
  photo,
  selected,
  threshold,
  onSelect,
}: {
  scene: ReviewScene
  photo: ReviewPhoto | undefined
  selected: boolean
  threshold: number
  onSelect: () => void
}) {
  const ref = useScrollIntoView(selected)
  return (
    <button
      ref={ref}
      type="button"
      onClick={onSelect}
      className={cn(
        'flex w-full items-center gap-2 px-2 py-1.5 text-left',
        selected ? 'bg-ink-700' : 'hover:bg-ink-800',
      )}
    >
      <Thumb photo={photo} />
      <span className="min-w-0 flex-1">
        <span className="block text-sm text-ink-50">{t('review.scene', { n: scene.cluster })}</span>
        <span className="block text-[11px] text-ink-400">
          {t('review.scenePhotos', { count: scene.photos.length })}
          {scene.queued > 0 ? ` · ${t('review.sceneQueued', { count: scene.queued })}` : ''}
        </span>
      </span>
      <span className="flex flex-col items-end gap-0.5">
        <span
          className={cn(
            'rounded px-1.5 text-[11px]',
            scene.state === 'pending' ? 'bg-ink-600 text-ink-100' : 'bg-ink-800 text-ink-300',
          )}
        >
          {t(`review.sceneState.${scene.state}` as StringKey)}
        </span>
        <ConfidenceBadge value={scene.min_confidence} threshold={threshold} />
      </span>
    </button>
  )
}

export function SceneList({
  scenes,
  photos,
  selected,
  threshold,
  onSelect,
}: {
  scenes: ReviewScene[]
  photos: Map<number, ReviewPhoto>
  selected: number | null
  threshold: number
  onSelect: (cluster: number) => void
}) {
  if (scenes.length === 0) return <p className="p-3 text-sm text-ink-400">{t('review.empty')}</p>
  return (
    <ul className="divide-y divide-ink-800">
      {scenes.map((scene) => (
        <li key={scene.cluster}>
          <SceneRow
            scene={scene}
            photo={photos.get(scene.representative)}
            selected={scene.cluster === selected}
            threshold={threshold}
            onSelect={() => onSelect(scene.cluster)}
          />
        </li>
      ))}
    </ul>
  )
}

function QueueRow({
  entry,
  photo,
  selected,
  threshold,
  onSelect,
}: {
  entry: ReviewQueueEntry
  photo: ReviewPhoto | undefined
  selected: boolean
  threshold: number
  onSelect: () => void
}) {
  const ref = useScrollIntoView(selected)
  return (
    <button
      ref={ref}
      type="button"
      onClick={onSelect}
      className={cn(
        'flex w-full items-center gap-2 px-2 py-1.5 text-left',
        selected ? 'bg-ink-700' : 'hover:bg-ink-800',
      )}
    >
      <Thumb photo={photo} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm text-ink-50">{photo?.filename}</span>
        <span className="block truncate text-[11px] text-ink-400">
          {entry.reasons[0] ? reasonLabel(entry.reasons[0]).replace(/\s*\(%\)$/, '') : ''}
        </span>
      </span>
      <ConfidenceBadge value={entry.confidence} threshold={threshold} />
    </button>
  )
}

export function QueueList({
  queue,
  photos,
  selected,
  threshold,
  onSelect,
}: {
  queue: ReviewQueueEntry[]
  photos: Map<number, ReviewPhoto>
  selected: number | null
  threshold: number
  onSelect: (photoId: number) => void
}) {
  if (queue.length === 0) return <p className="p-3 text-sm text-ink-400">{t('review.queueEmpty')}</p>
  return (
    <ul className="divide-y divide-ink-800">
      {queue.map((entry) => (
        <li key={entry.photo_id}>
          <QueueRow
            entry={entry}
            photo={photos.get(entry.photo_id)}
            selected={entry.photo_id === selected}
            threshold={threshold}
            onSelect={() => onSelect(entry.photo_id)}
          />
        </li>
      ))}
    </ul>
  )
}

/** The other photos of the scene, under the representative. Click to inspect. */
export function SceneStrip({
  scene,
  photos,
  inspecting,
  onInspect,
}: {
  scene: ReviewScene
  photos: Map<number, ReviewPhoto>
  inspecting: number
  onInspect: (photoId: number) => void
}) {
  return (
    <div className="border-t border-ink-700 bg-ink-900 px-3 py-1.5">
      <p className="mb-1 text-[11px] text-ink-400">{t('review.membersHint')}</p>
      <div className="flex gap-1.5 overflow-x-auto pb-1">
        {scene.photos.map((id) => {
          const photo = photos.get(id)
          return (
            <button
              key={id}
              type="button"
              title={photo?.filename}
              onClick={() => onInspect(id)}
              className={cn(
                'relative h-14 w-20 shrink-0 overflow-hidden rounded-sm border bg-mat',
                id === inspecting ? 'border-ink-200' : 'border-ink-700 hover:border-ink-500',
              )}
            >
              <DevelopedImage photo={photo} className="h-full w-full object-cover" />
              {photo?.status === 'approved' ? (
                <span className="absolute bottom-0.5 right-0.5 rounded bg-ink-950/80 px-1 text-[10px] text-ink-100">✓</span>
              ) : photo?.status === 'needs_review' ? (
                <span className="absolute bottom-0.5 right-0.5 rounded bg-ink-100 px-1 text-[10px] text-ink-900">!</span>
              ) : null}
            </button>
          )
        })}
      </div>
    </div>
  )
}
