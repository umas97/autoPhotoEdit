// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// One card of the culling grid (section 7.6): the camera's preview, its score,
// an icon for each reason it was discarded, and -- for a burst -- the "1 di 7"
// that opens the others.
//
// A discarded photo is dimmed, never recoloured: section 10 keeps saturated
// colour away from the previews, and a red tint over a photograph is exactly
// that. The reason icons sit on the dark frame around the image, not on it.
import {
  Aperture,
  Copy,
  Focus,
  Hand,
  ListFilter,
  Moon,
  Sun,
  Wind,
  type LucideIcon,
} from 'lucide-react'
import { thumbUrl } from '../../lib/api'
import { percent } from '../../lib/culling'
import type { CullMerge, CullPhoto, CullReason } from '../../lib/cullTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'

export const REASON_ICON: Record<CullReason, LucideIcon> = {
  out_of_focus: Focus,
  motion_blur: Wind,
  overexposed: Sun,
  underexposed: Moon,
  burst_duplicate: Copy,
  below_target: ListFilter,
  user: Hand,
}

export function reasonLabel(reason: CullReason): string {
  return t(`culling.reason.${reason}` as StringKey)
}

export function mergeLabel(merge: CullMerge): string {
  return t('culling.merge.badge', {
    kind: t(`culling.merge.kind.${merge.kind}` as StringKey),
    count: merge.members.length,
  })
}

export function CullCard({
  photo,
  burst,
  burstKept,
  merge,
  current,
  onSelect,
  onOpenBurst,
}: {
  photo: CullPhoto
  /** Members of the burst this card stands for, proposed frame first. */
  burst: number[] | null
  /** How many frames of that burst are kept: the "1" of "1 di 7". */
  burstKept: number
  merge: CullMerge | undefined
  current: boolean
  onSelect: () => void
  onOpenBurst: () => void
}) {
  const aspect = photo.width && photo.height ? photo.width / photo.height : 3 / 2
  return (
    <li
      className="[contain-intrinsic-size:auto_12rem] [content-visibility:auto]"
      data-photo={photo.id}
    >
      <div
        role="button"
        tabIndex={-1}
        onClick={onSelect}
        className={cn(
          'group relative block w-full cursor-pointer overflow-hidden rounded border bg-ink-850 text-left',
          current ? 'border-ink-100' : 'border-ink-700 hover:border-ink-500',
        )}
        aria-current={current}
        title={photo.filename}
      >
        <div className="relative flex aspect-[3/2] w-full items-center justify-center bg-mat">
          <img
            src={thumbUrl(photo)}
            alt={photo.filename}
            loading="lazy"
            decoding="async"
            style={{ aspectRatio: aspect }}
            className={cn(
              'max-h-full max-w-full object-contain transition-opacity',
              photo.culled && 'opacity-35',
            )}
          />
        </div>

        <div className="flex h-6 items-center gap-1.5 px-1.5 text-[11px] text-ink-300">
          <span
            className={cn('tabular-nums', photo.culled ? 'text-ink-400' : 'text-ink-100')}
            title={t('culling.card.score')}
          >
            {percent(photo.score)}
          </span>
          {photo.reasons.map((reason) => {
            const Icon = REASON_ICON[reason]
            return (
              <span key={reason} title={reasonLabel(reason)} className="text-ink-300">
                <Icon size={12} />
              </span>
            )
          })}
          {merge ? (
            <span className="flex items-center gap-0.5 text-ink-200" title={t('culling.merge.protected')}>
              <Aperture size={12} />
              {mergeLabel(merge)}
            </span>
          ) : null}
          {burst ? (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                onOpenBurst()
              }}
              className="ml-auto rounded border border-ink-600 px-1 tabular-nums text-ink-200 hover:border-ink-400"
              title={t('culling.burst.open')}
            >
              {t('culling.burst.counter', { kept: burstKept, total: burst.length })}
            </button>
          ) : (
            <span className="ml-auto truncate text-ink-400">{photo.filename}</span>
          )}
        </div>
      </div>
    </li>
  )
}
