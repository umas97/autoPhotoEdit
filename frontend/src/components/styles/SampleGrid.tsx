// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The training pairs of a profile, each with the reference it learned from and
// how well the pipeline could reproduce it (section 8.2.5): the residual ΔE, or
// why the pair is not used. A pair can be left out of training, or removed.
import { Ban, RotateCcw, Trash2 } from 'lucide-react'
import { sampleThumbUrl } from '../../lib/stylesApi'
import type { StyleSample } from '../../lib/styleTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'

function statusLabel(sample: StyleSample): string {
  if (sample.excluded) return t('styles.sample.excluded')
  if (sample.status === 'ready') return t('styles.sample.ready', { delta: sample.delta_e?.toFixed(1) ?? '—' })
  return t(`styles.sample.${sample.status}` as StringKey)
}

export function SampleGrid({
  samples,
  editable,
  onExclude,
  onRemove,
}: {
  samples: StyleSample[]
  editable: boolean
  onExclude: (sample: StyleSample, excluded: boolean) => void
  onRemove: (sample: StyleSample) => void
}) {
  if (samples.length === 0) return <p className="text-sm text-ink-400">{t('styles.samples.none')}</p>
  return (
    <ul className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-2">
      {samples.map((sample) => {
        const unusable = sample.excluded || sample.status === 'failed' || sample.status === 'unreproducible'
        return (
          <li
            key={sample.id}
            className={cn('rounded-md border border-ink-700 bg-ink-900 p-1.5', unusable && 'opacity-60')}
            title={sample.error ?? undefined}
          >
            <div className="aspect-[3/2] overflow-hidden rounded border border-ink-700 bg-mat">
              {sample.has_thumbnail ? (
                <img src={sampleThumbUrl(sample.id)} alt={sample.reference ?? ''} loading="lazy"
                  className="h-full w-full object-contain" />
              ) : null}
            </div>
            <div className="mt-1 truncate text-xs text-ink-100" title={sample.reference_path ?? ''}>
              {sample.reference}
            </div>
            <div className="truncate text-[11px] text-ink-400" title={sample.raw_path ?? ''}>
              {sample.raw}
              {sample.pairing ? ` · ${t(`styles.pairing.${sample.pairing}` as StringKey)}` : ''}
            </div>
            <div className="mt-1 flex items-center gap-1">
              <span
                className={cn(
                  'truncate text-[11px] tabular-nums',
                  sample.status === 'ready' && !sample.excluded ? 'text-ink-200' : 'text-warn',
                )}
              >
                {statusLabel(sample)}
              </span>
              {!sample.raw_available ? (
                <span className="text-[11px] text-ink-500" title={t('styles.sample.rawMissing')}>·RAW</span>
              ) : null}
              {editable ? (
                <span className="ml-auto flex">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6"
                    title={sample.excluded ? t('styles.sample.include') : t('styles.sample.exclude')}
                    aria-label={sample.excluded ? t('styles.sample.include') : t('styles.sample.exclude')}
                    disabled={sample.status !== 'ready'}
                    onClick={() => onExclude(sample, !sample.excluded)}
                  >
                    {sample.excluded ? <RotateCcw size={12} /> : <Ban size={12} />}
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6"
                    title={t('styles.sample.remove')}
                    aria-label={t('styles.sample.remove')}
                    onClick={() => onRemove(sample)}
                  >
                    <Trash2 size={12} />
                  </Button>
                </span>
              ) : null}
            </div>
          </li>
        )
      })}
    </ul>
  )
}
