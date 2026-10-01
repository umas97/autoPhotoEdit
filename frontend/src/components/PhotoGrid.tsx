// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strip of proxies on the left of the viewer. Section 10: never the RAW,
// always the cached 2048 px proxy, on a neutral grey mat with a thin border.
//
// A photo whose proxy is not built yet shows the camera's embedded preview
// instead (section 7.2) -- the same picture, already on the card -- and only a
// photo with neither shows a placeholder. A photo discarded in culling is
// dimmed, never tinted: saturated colour stays away from the previews
// (section 10). So is a frame an accepted merge stands in for, when
// "Mostra scatti sorgente" shows it.
//
// A merged photo carries the badge of section 25.6, "HDR · 3"; a click on the
// badge opens its frames under it. Ctrl+click (Shift+click for a run) picks
// photos for a merge made by hand, without moving the photo being edited.
import { Fragment, useState, type MouseEvent } from 'react'
import { proxyUrl, thumbUrl } from '../lib/api'
import type { Photo } from '../lib/types'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { badgeLabel } from './merges/MergeCard'
import { MergeMembers } from './merges/MergeMembers'

/** Whether a photo can be one of the frames of a new merge. */
export function pickable(photo: Photo): boolean {
  return photo.merge === null && !photo.superseded && !photo.missing
}

export function PhotoGrid({
  projectId,
  photos,
  selectedId,
  onSelect,
  picked,
  onPick,
}: {
  projectId: number
  photos: Photo[]
  selectedId: number | null
  onSelect: (photo: Photo) => void
  picked?: ReadonlySet<number>
  /** `range`: every pickable photo between the last one picked and this. */
  onPick?: (photo: Photo, mode: 'toggle' | 'range') => void
}) {
  const [expanded, setExpanded] = useState<number | null>(null)
  if (photos.length === 0) {
    return <p className="p-4 text-sm text-ink-400">{t('photos.empty')}</p>
  }

  const click = (photo: Photo, event: MouseEvent) => {
    if (onPick && pickable(photo) && (event.ctrlKey || event.metaKey)) onPick(photo, 'toggle')
    else if (onPick && pickable(photo) && event.shiftKey && picked?.size) onPick(photo, 'range')
    else onSelect(photo)
  }

  return (
    <ul className="grid grid-cols-2 gap-1.5 p-1.5 2xl:grid-cols-3">
      {photos.map((photo) => {
        const selected = photo.id === selectedId
        const isPicked = picked?.has(photo.id) ?? false
        return (
          <Fragment key={photo.id}>
            <li className="relative">
              <button
                type="button"
                onClick={(event) => click(photo, event)}
                className={cn(
                  'group block w-full overflow-hidden rounded border bg-mat text-left',
                  isPicked
                    ? 'border-ink-50 ring-1 ring-ink-50'
                    : selected
                      ? 'border-ink-200'
                      : 'border-ink-700 hover:border-ink-500',
                )}
                aria-current={selected}
                aria-pressed={onPick ? isPicked : undefined}
              >
                <div className="relative aspect-[3/2] w-full bg-mat">
                  {photo.has_proxy || photo.has_thumb ? (
                    <img
                      src={photo.has_proxy ? proxyUrl(photo) : thumbUrl(photo)}
                      alt={photo.filename}
                      loading="lazy"
                      decoding="async"
                      className={cn(
                        'h-full w-full object-contain',
                        (photo.culled || photo.superseded) && 'opacity-35',
                      )}
                    />
                  ) : (
                    <span className="absolute inset-0 grid place-items-center px-2 text-center text-[11px] text-ink-400">
                      {photo.missing ? t('photos.missing') : t('photos.noProxy')}
                    </span>
                  )}
                </div>
                <span className="block truncate px-1.5 py-1 text-[11px] text-ink-300">
                  {photo.culled ? `${t('photos.culledMark')} · ` : ''}
                  {photo.superseded ? `${t('merges.sourceMark')} · ` : ''}
                  {photo.filename}
                </span>
              </button>
              {photo.merge && photo.merge_group_id !== null ? (
                <button
                  type="button"
                  className="absolute left-1 top-1 rounded bg-ink-950/80 px-1.5 py-0.5 text-[10px] font-medium text-ink-50 hover:bg-ink-800"
                  title={t('merges.expand')}
                  aria-expanded={expanded === photo.id}
                  onClick={() => setExpanded(expanded === photo.id ? null : photo.id)}
                >
                  {badgeLabel(photo.merge.kind, photo.merge.sources)}
                </button>
              ) : null}
            </li>
            {expanded === photo.id && photo.merge_group_id !== null ? (
              <li className="col-span-full">
                <MergeMembers projectId={projectId} groupId={photo.merge_group_id} />
              </li>
            ) : null}
          </Fragment>
        )
      })}
    </ul>
  )
}
