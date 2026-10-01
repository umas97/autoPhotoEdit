// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The left column of the review (section 10): the three ways in -- the grid of
// every scene, one scene at a time, the individual queue -- the list that goes
// with them, and how far the project is.
import type {
  ReviewOverview,
  ReviewPhoto,
  ReviewQueueEntry,
  ReviewScene,
} from '../../lib/reviewTypes'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { QueueList, SceneList } from './ReviewList'

export type ReviewTab = 'grid' | 'scenes' | 'queue'

const TABS: ReviewTab[] = ['grid', 'scenes', 'queue']

export function ReviewSidebar({
  tab,
  onTab,
  review,
  scenes,
  queue,
  photos,
  selectedScene,
  selectedPhoto,
  threshold,
  onScene,
  onPhoto,
}: {
  tab: ReviewTab
  onTab: (tab: ReviewTab) => void
  review: ReviewOverview | undefined
  scenes: ReviewScene[]
  queue: ReviewQueueEntry[]
  photos: Map<number, ReviewPhoto>
  selectedScene: number | null
  selectedPhoto: number | null
  threshold: number
  onScene: (cluster: number) => void
  onPhoto: (photoId: number) => void
}) {
  const label = (name: ReviewTab) =>
    name === 'grid'
      ? t('review.tabGrid')
      : name === 'scenes'
        ? t('review.tabScenes', { done: review?.counts.scenes_done ?? 0, total: review?.counts.scenes ?? 0 })
        : t('review.tabQueue', { count: review?.counts.queue ?? 0 })
  return (
    <aside className="flex min-h-0 flex-col border-r border-ink-700 bg-ink-900">
      <div className="flex shrink-0 gap-1 border-b border-ink-700 p-1.5">
        {TABS.map((name) => (
          <button key={name} type="button" onClick={() => onTab(name)}
            className={cn('flex-1 whitespace-nowrap rounded px-1.5 py-1 text-xs',
              tab === name ? 'bg-ink-700 text-ink-50' : 'text-ink-300 hover:bg-ink-800')}>
            {label(name)}
          </button>
        ))}
      </div>
      {review && review.pending > 0 ? (
        <p className="px-2 py-1.5 text-xs text-ink-300">{t('review.pending', { count: review.pending })}</p>
      ) : null}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab !== 'queue' ? (
          <SceneList scenes={scenes} photos={photos} selected={selectedScene} threshold={threshold} onSelect={onScene} />
        ) : (
          <QueueList queue={queue} photos={photos} selected={selectedPhoto} threshold={threshold} onSelect={onPhoto} />
        )}
      </div>
      {review ? (
        <p className="shrink-0 border-t border-ink-700 px-2 py-1.5 text-[11px] text-ink-400">
          {review.counts.total > 0 && review.counts.approved === review.counts.total
            ? t('review.allDone')
            : t('review.summary', { approved: review.counts.approved, total: review.counts.total, queue: review.counts.queue })}
        </p>
      ) : null}
    </aside>
  )
}
