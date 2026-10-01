// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Every scene at once (section 9.2.2, at the speed of a contact sheet): the
// representative developed large, the other photos of the scene developed
// small beneath it. Scenes that look right are selected and approved together
// with one key; one that does not is opened (double click, or Enter) and
// corrected on its representative as usual.
//
// Approving from here is exactly approving each scene: the photos in the
// individual queue stay there, and one Ctrl+Z takes the whole choice back.
import { useEffect, useState } from 'react'
import { Check, SquareCheck, SquareDashed } from 'lucide-react'
import type { ReviewPhoto, ReviewScene } from '../../lib/reviewTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { ConfidenceBadge } from './ConfidenceCard'
import { DevelopedImage } from './ReviewList'
import { keyBelongsToControl } from '../../lib/shortcuts'

/** Members shown under a representative; the rest are counted. */
const MEMBERS_SHOWN = 6

function SceneTile({
  scene,
  photos,
  selected,
  focused,
  threshold,
  onToggle,
  onOpen,
}: {
  scene: ReviewScene
  photos: Map<number, ReviewPhoto>
  selected: boolean
  focused: boolean
  threshold: number
  onToggle: () => void
  onOpen: () => void
}) {
  const members = scene.photos.filter((id) => id !== scene.representative)
  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={onToggle}
      onDoubleClick={onOpen}
      className={cn(
        'flex cursor-pointer flex-col gap-1 rounded border bg-ink-900 p-1.5',
        selected ? 'border-ink-100' : 'border-ink-700 hover:border-ink-500',
        focused && !selected ? 'border-ink-400' : '',
      )}
    >
      <div className="relative aspect-[3/2] overflow-hidden rounded-sm bg-mat">
        <DevelopedImage photo={photos.get(scene.representative)} className="h-full w-full object-contain" />
        {selected ? (
          <span className="absolute left-1 top-1 rounded bg-ink-100 p-0.5 text-ink-900">
            <Check size={12} />
          </span>
        ) : null}
      </div>
      {members.length > 0 ? (
        <div className="flex gap-0.5">
          {members.slice(0, MEMBERS_SHOWN).map((id) => (
            <div key={id} className="relative aspect-[3/2] w-[calc((100%-0.625rem)/6)] overflow-hidden rounded-[2px] bg-mat">
              <DevelopedImage photo={photos.get(id)} className="h-full w-full object-cover" />
              {photos.get(id)?.status === 'needs_review' ? (
                <span className="absolute bottom-0 right-0 bg-ink-100 px-0.5 text-[9px] leading-tight text-ink-900">!</span>
              ) : null}
            </div>
          ))}
          {members.length > MEMBERS_SHOWN ? (
            <span className="self-center pl-0.5 text-[10px] text-ink-400">+{members.length - MEMBERS_SHOWN}</span>
          ) : null}
        </div>
      ) : null}
      <div className="flex items-center gap-1.5 text-[11px]">
        <span className="text-ink-100">{t('review.scene', { n: scene.cluster })}</span>
        <span className="text-ink-400">
          {t('review.scenePhotos', { count: scene.photos.length })}
          {scene.queued > 0 ? ` · ${t('review.sceneQueued', { count: scene.queued })}` : ''}
        </span>
        <span className="ml-auto flex items-center gap-1">
          <span className={cn('rounded px-1', scene.state === 'pending' ? 'bg-ink-600 text-ink-100' : 'bg-ink-800 text-ink-300')}>
            {t(`review.sceneState.${scene.state}` as StringKey)}
          </span>
          <ConfidenceBadge value={scene.min_confidence} threshold={threshold} />
        </span>
      </div>
    </div>
  )
}

export function SceneGrid({
  scenes,
  photos,
  threshold,
  busy,
  notice,
  onApprove,
  onOpen,
}: {
  scenes: ReviewScene[]
  photos: Map<number, ReviewPhoto>
  threshold: number
  busy: boolean
  /** The outcome of the last action, as the footer of the other views shows it. */
  notice: string | null
  onApprove: (clusters: number[]) => void
  onOpen: (cluster: number) => void
}) {
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [focus, setFocus] = useState(0)
  const pending = scenes.filter((s) => s.state === 'pending')

  // A scene that is no longer there, or no longer pending, is not selected.
  useEffect(() => {
    setSelected((current) => {
      const open = new Set(scenes.filter((s) => s.state === 'pending').map((s) => s.cluster))
      const kept = new Set([...current].filter((c) => open.has(c)))
      return kept.size === current.size ? current : kept
    })
  }, [scenes])

  const toggle = (cluster: number) =>
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(cluster)) next.delete(cluster)
      else next.add(cluster)
      return next
    })

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (keyBelongsToControl(event)) return
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const key = event.key.toLowerCase()
      if (key === 'a' && selected.size > 0 && !busy) {
        onApprove([...selected])
      } else if (event.key === 'Escape') {
        setSelected(new Set())
      } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        setFocus((f) => Math.max(0, Math.min(scenes.length - 1, f + (event.key === 'ArrowRight' ? 1 : -1))))
      } else if (event.key === ' ' && scenes[focus]) {
        event.preventDefault()
        toggle(scenes[focus].cluster)
      } else if (event.key === 'Enter' && scenes[focus]) {
        onOpen(scenes[focus].cluster)
      }
    }
    window.addEventListener('keydown', down)
    return () => window.removeEventListener('keydown', down)
  }, [busy, focus, onApprove, onOpen, scenes, selected])

  if (scenes.length === 0) return <p className="grid h-full place-items-center text-sm text-ink-400">{t('review.empty')}</p>

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-ink-700 bg-ink-900 px-3 py-1.5 text-xs">
        <span className="text-ink-200">{t('review.gridSummary', { pending: pending.length, selected: selected.size })}</span>
        {notice ? <span className="truncate text-ink-300">{notice}</span> : null}
        <span className="ml-auto flex items-center gap-1">
          <Button size="sm" variant="ghost" disabled={pending.length === 0}
            onClick={() => setSelected(new Set(pending.map((s) => s.cluster)))}>
            <SquareCheck size={13} />{t('review.gridSelectPending')}
          </Button>
          <Button size="sm" variant="ghost" disabled={selected.size === 0} onClick={() => setSelected(new Set())}>
            <SquareDashed size={13} />{t('review.gridSelectNone')}
          </Button>
          <Button size="sm" variant="primary" disabled={selected.size === 0 || busy}
            onClick={() => onApprove([...selected])}>
            <Check size={13} />{t('review.gridApprove', { count: selected.size })} (A)
          </Button>
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-2">
          {scenes.map((scene, index) => (
            <SceneTile key={scene.cluster} scene={scene} photos={photos} threshold={threshold}
              selected={selected.has(scene.cluster)} focused={index === focus}
              onToggle={() => { setFocus(index); if (scene.state === 'pending') toggle(scene.cluster) }}
              onOpen={() => onOpen(scene.cluster)} />
          ))}
        </div>
      </div>
      <p className="shrink-0 bg-ink-900 px-3 py-1 text-[11px] text-ink-500">{t('review.gridHint')}</p>
    </div>
  )
}
