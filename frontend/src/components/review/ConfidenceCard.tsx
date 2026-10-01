// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The confidence badge and the reasons of section 9.2.3: "ciascuna con il motivo
// esplicito". The reasons are the terms of backend/ape/review/confidence.py
// that pulled the score down, heaviest first, in words -- with the measured
// value where one makes the sentence concrete ("luci bruciate: 6%").
import type { ConfidenceTerm, PhotoReview, ReasonCode } from '../../lib/reviewTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'

export function percent(value: number | null | undefined): number {
  return Math.round((value ?? 0) * 100)
}

export function reasonLabel(code: ReasonCode, value?: number | null): string {
  const shown =
    value !== null && value !== undefined && (code === 'burnt' || code === 'crushed')
      ? (value * 100).toFixed(1)
      : ''
  return t(`review.reason.${code}` as StringKey, { value: shown })
}

/** A neutral badge: the interface stays grey near the photographs (section 10). */
export function ConfidenceBadge({
  value,
  threshold,
  className,
}: {
  value: number | null | undefined
  threshold: number
  className?: string
}) {
  const low = value !== null && value !== undefined && value < threshold
  return (
    <span
      className={cn(
        'rounded px-1.5 py-px text-[11px] tabular-nums',
        low ? 'bg-ink-100 text-ink-900' : 'bg-ink-700 text-ink-200',
        className,
      )}
    >
      {value === null || value === undefined ? '—' : `${percent(value)}%`}
    </span>
  )
}

export function ConfidenceCard({ review }: { review: PhotoReview | undefined }) {
  if (!review?.available) return null
  const threshold = review.threshold ?? 0.55
  const terms = new Map<string, ConfidenceTerm>((review.terms ?? []).map((term) => [term.code, term]))
  const decision = review.review?.decision
  const codes: ReasonCode[] = [
    ...(decision === 'rejected' ? (['rejected'] as ReasonCode[]) : []),
    ...(review.review?.queued ? (['scene_rejected'] as ReasonCode[]) : []),
    ...(review.reasons ?? []),
  ]
  return (
    <section className="border-b border-ink-700 px-3 py-2">
      <div className="flex items-center gap-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-ink-400">{t('review.why')}</h3>
        <ConfidenceBadge value={review.confidence} threshold={threshold} className="ml-auto" />
        <span className="text-[11px] text-ink-400">
          {t('review.threshold', { value: percent(threshold) })}
        </span>
      </div>
      {decision === 'approved' ? (
        <p className="mt-1 text-xs text-ink-100">{t('review.decided.approved')}</p>
      ) : null}
      {codes.length > 0 ? (
        <ul className="mt-1 space-y-0.5">
          {codes.map((code) => (
            <li key={code} className="text-xs text-ink-200">
              · {reasonLabel(code, terms.get(code)?.value)}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1 text-xs text-ink-400">{t('review.confidence', { value: percent(review.confidence) })}</p>
      )}
    </section>
  )
}
