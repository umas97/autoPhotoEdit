// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Perché questa foto è tra gli scarti?" answered for the photo under the
// cursor (section 7.4: "l'utente deve poter capire perché una foto è finita tra
// gli scarti").
//
// Each criterion's score is shown against the threshold the aggressiveness
// sets, with its weight next to it, and the combined score is the weighted mean
// of exactly those rows. Nothing is summarised away: the numbers on this panel
// are the numbers the server used.
import { Aperture } from 'lucide-react'
import { percent } from '../../lib/culling'
import type { CullMerge, CullPhoto, CullSettings, ScoredCriterion } from '../../lib/cullTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn, formatShutter } from '../../lib/utils'
import { mergeLabel, reasonLabel } from './CullCard'

const ORDER: ScoredCriterion[] = ['sharpness', 'motion', 'exposure', 'faces', 'aesthetic']

export function WhyPanel({
  photo,
  settings,
  merge,
  burstPosition,
}: {
  photo: CullPhoto
  settings: CullSettings
  merge: CullMerge | undefined
  /** ``[position, total]`` inside the photo's burst, when it has one. */
  burstPosition: [number, number] | null
}) {
  const verdict = photo.culled ? t('culling.why.culled') : t('culling.why.kept')
  return (
    <section className="border-t border-ink-700 px-3 py-2.5 text-xs">
      <h3 className="mb-1 flex items-baseline gap-2">
        <span className="truncate text-sm text-ink-50">{photo.filename}</span>
        <span className={cn('ml-auto shrink-0', photo.culled ? 'text-ink-300' : 'text-ink-100')}>
          {verdict}
        </span>
      </h3>
      {photo.reasons.length > 0 ? (
        <p className="mb-1.5 text-ink-200">{photo.reasons.map(reasonLabel).join(' · ')}</p>
      ) : null}
      {photo.decided_by === 'user' ? (
        <p className="mb-1.5 text-ink-300">{t('culling.why.byUser')}</p>
      ) : null}

      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr className="text-ink-400">
            <th className="w-2/5 text-left font-normal">{t('culling.why.criterion')}</th>
            <th className="text-left font-normal">{t('culling.why.score')}</th>
            <th className="w-12 text-right font-normal">{t('culling.why.weight')}</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.filter((name) => photo.criteria[name] !== undefined).map((name) => {
            const value = photo.criteria[name] ?? 0
            const failing = value < settings.threshold && name !== 'faces' && name !== 'aesthetic'
            return (
              <tr key={name}>
                <td className="truncate py-0.5 text-ink-200">
                  {t(`culling.criterion.${name}` as StringKey)}
                </td>
                <td className="py-0.5">
                  <div className="relative h-1.5 w-full rounded-full bg-ink-700">
                    <div
                      className={cn('h-full rounded-full', failing ? 'bg-ink-400' : 'bg-ink-200')}
                      style={{ width: `${Math.round(value * 100)}%` }}
                    />
                    <div
                      className="absolute top-[-3px] h-3 w-px bg-ink-300"
                      style={{ left: `${Math.round(settings.threshold * 100)}%` }}
                      title={t('culling.threshold', { value: Math.round(settings.threshold * 100) })}
                    />
                  </div>
                </td>
                <td className="py-0.5 text-right tabular-nums text-ink-300">
                  {percent(settings.weights[name])}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-1.5 text-ink-200">
        {t('culling.why.total', { score: percent(photo.score) })}
      </p>

      {burstPosition ? (
        <p className="mt-1 text-ink-300">
          {t('culling.why.burst', { position: burstPosition[0], total: burstPosition[1] })}
        </p>
      ) : null}
      {merge ? (
        <p className="mt-1 flex items-center gap-1 text-ink-300">
          <Aperture size={12} />
          {mergeLabel(merge)} · {t('culling.merge.protected')}
        </p>
      ) : null}
      {photo.sharpness === null && photo.status !== 'failed' ? (
        <p className="mt-1 text-ink-400">{t('culling.why.notAssessable')}</p>
      ) : null}
      {photo.error ? <p className="mt-1 text-bad">{photo.error}</p> : null}
      {photo.iso ? (
        <p className="mt-1.5 tabular-nums text-ink-400">
          {t('viewer.exposureLine', {
            iso: photo.iso,
            aperture: photo.aperture?.toFixed(1) ?? '—',
            shutter: formatShutter(photo.shutter),
            focal: photo.focal_length?.toFixed(0) ?? '—',
          })}
        </p>
      ) : null}
    </section>
  )
}
