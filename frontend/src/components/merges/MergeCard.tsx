// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// One group on the merges screen (section 25.6): what it is and why it was
// proposed, in words the user can judge without opening the photos; its
// members, the reference marked; the 1024 px preview, and what the preview's
// report found worth saying before three minutes of CPU are spent on it.
//
// The buttons follow the group's decision. Every one of them is a request and
// the card redraws from the answer: nothing here guesses a state the server
// has not confirmed.
import { useState } from 'react'
import { AlertTriangle, Check, Eye, Pencil, RotateCcw, Star, X } from 'lucide-react'
import { memberThumbUrl } from '../../lib/mergesApi'
import type { MergeGroupView, MergeKind, MergeReasons, MergeReport } from '../../lib/mergeTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Checkbox } from '../ui/Field'

export type MergeAction = 'preview' | 'accept' | 'reject' | 'undo' | 'edit' | 'open'

const number = (value: number, digits = 1) =>
  value.toLocaleString('it-IT', { maximumFractionDigits: digits })

/** Stops with a real minus sign, and a plus where the sign is the point. */
export function formatEv(value: number): string {
  if (Math.abs(value) < 0.05) return '0'
  return `${value > 0 ? '+' : '−'}${number(Math.abs(value))}`
}

export function kindLabel(kind: MergeKind): string {
  return t(`merges.kind.${kind}` as StringKey)
}

export function badgeLabel(kind: MergeKind, count: number): string {
  return t('merges.badge', { kind: kindLabel(kind), count })
}

/** "5 scatti, EV da −2 a +2, intervallo 0,8 s" (section 25.2). */
export function reasonsText(reasons: MergeReasons, members: number): string {
  // Detection's count is of the group as found; after an edit, the members are.
  const parts = [t('merges.reason.frames', { count: reasons.manual ? members : reasons.frames ?? members })]
  if (reasons.ev_min !== undefined && reasons.ev_max !== undefined) {
    parts.push(t('merges.reason.ev', { min: formatEv(reasons.ev_min), max: formatEv(reasons.ev_max) }))
  }
  if (reasons.overlap_min !== undefined && reasons.overlap_max !== undefined) {
    parts.push(
      t('merges.reason.overlap', {
        min: Math.round(reasons.overlap_min * 100),
        max: Math.round(reasons.overlap_max * 100),
      }),
    )
  }
  if (reasons.focus_positions) parts.push(t('merges.reason.focusExif'))
  else if (reasons.focus_moves) parts.push(t('merges.reason.focusPixels'))
  if (reasons.sweep) parts.push(t('merges.reason.sweep'))
  if (reasons.span_s !== undefined) parts.push(t('merges.reason.span', { seconds: number(reasons.span_s) }))
  if (reasons.camera_bracket) parts.push(t('merges.reason.cameraBracket'))
  return parts.join(', ')
}

/** What the merge's report asks the user to look at, one sentence each. */
function reportNotes(report: MergeReport | null | undefined): string[] {
  if (!report) return []
  const notes: string[] = []
  if (report.misaligned && report.residual_px !== undefined) {
    notes.push(t('merges.report.misaligned', { px: number(report.residual_px) }))
  }
  if (report.ghost_fraction !== undefined && report.ghost_fraction >= 0.005) {
    notes.push(t('merges.report.ghosts', { percent: number(report.ghost_fraction * 100, 0) }))
  }
  if (report.uncovered_fraction !== undefined && report.uncovered_fraction >= 0.005) {
    notes.push(t('merges.report.uncovered', { percent: number(report.uncovered_fraction * 100, 0) }))
  }
  if (report.used && report.members !== undefined && report.used.length < report.members) {
    notes.push(t('merges.report.used', { used: report.used.length, members: report.members }))
  }
  if (report.too_large) notes.push(report.too_large)
  if (report.crop) notes.push(t('merges.report.crop'))
  return notes
}

/** "Proiezione cilindrica, 88° di ampiezza, 48 MP": what the full merge will make. */
function panoramaSummary(group: MergeGroupView): string | null {
  // The full report describes what was made; the preview's, what will be.
  const report = group.full_report ?? group.preview.report
  if (!report?.projection || !report.span_deg) return null
  const megapixels = group.full_report ? report.output?.megapixels : report.full?.megapixels
  if (megapixels === undefined) return null
  return t(report.vertical ? 'merges.report.panoramaVertical' : 'merges.report.panorama', {
    projection: t(`merges.projection.${report.projection}`).toLowerCase(),
    span: Math.round(report.span_deg[report.vertical ? 1 : 0]),
    mp: number(megapixels, 0),
  })
}

function Members({ group }: { group: MergeGroupView }) {
  return (
    <ul className="flex gap-1.5 overflow-x-auto pb-1">
      {group.members.map((member) => (
        <li key={member.photo_id} className="w-28 shrink-0">
          <div
            className={cn(
              'relative aspect-[3/2] overflow-hidden rounded border bg-mat',
              member.reference ? 'border-ink-200' : 'border-ink-700',
            )}
          >
            {member.has_thumb || member.has_proxy ? (
              <img src={memberThumbUrl(member)} alt={member.filename ?? ''}
                decoding="async" className="h-full w-full object-contain" />
            ) : null}
            {member.reference ? (
              <span className="absolute left-1 top-1 rounded bg-ink-950/80 p-0.5 text-ink-50"
                title={t('merges.member.reference')}>
                <Star size={10} />
              </span>
            ) : null}
          </div>
          <div className="truncate text-[11px] text-ink-300" title={member.filename ?? ''}>
            {member.filename}
            {group.kind === 'hdr' && member.ev_offset !== null
              ? ` · ${t('merges.member.ev', { ev: formatEv(member.ev_offset) })}`
              : ''}
            {member.missing ? ` · ${t('merges.member.missing')}` : ''}
          </div>
        </li>
      ))}
    </ul>
  )
}

function Preview({ group }: { group: MergeGroupView }) {
  const [coverage, setCoverage] = useState(false)
  const { preview } = group
  if (preview.state === 'queued' || preview.state === 'running') {
    return <p className="text-xs text-ink-300">{t('merges.preview.working')}</p>
  }
  if (preview.state === 'error') {
    return <p className="text-xs text-bad">{t('merges.preview.error', { reason: preview.error ?? '' })}</p>
  }
  if (preview.state === 'stale') return <p className="text-xs text-warn">{t('merges.preview.stale')}</p>
  if (preview.state !== 'ready' || !preview.url) {
    return <p className="text-xs text-ink-400">{t('merges.preview.none')}</p>
  }
  return (
    <div>
      <div className="relative overflow-hidden rounded border border-ink-700 bg-mat">
        <img src={preview.url} alt={t('merges.preview.alt')} className="max-h-80 w-full object-contain" />
        {coverage && preview.coverage_url ? (
          <img src={preview.coverage_url} alt="" aria-hidden
            className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
        ) : null}
      </div>
      {preview.coverage_url ? (
        <Checkbox checked={coverage} onChange={setCoverage} label={t('merges.preview.coverage')} />
      ) : null}
    </div>
  )
}

export function MergeCard({
  group,
  pending,
  onAction,
}: {
  group: MergeGroupView
  /** A request for this card is on its way: the buttons wait for its answer. */
  pending: boolean
  onAction: (action: MergeAction) => void
}) {
  const { decision } = group
  const job = group.merge_job
  const notes = reportNotes(decision === 'accepted' ? group.full_report ?? group.preview.report : group.preview.report)
  const summary = panoramaSummary(group)
  const canPreview = group.preview.state === 'none' || group.preview.state === 'stale' || group.preview.state === 'error'

  return (
    <article className={cn('rounded-lg border bg-ink-850 p-3', decision === 'rejected' ? 'border-ink-800 opacity-70' : 'border-ink-700')}>
      <header className="mb-2 flex flex-wrap items-baseline gap-x-2 text-sm">
        <span className="font-semibold text-ink-50">{badgeLabel(group.kind, group.members.length)}</span>
        <span className="text-xs text-ink-400">
          {group.confidence !== null
            ? t('merges.confidence', { value: Math.round(group.confidence * 100) })
            : t('merges.manual')}
          {' · '}
          {t(`merges.decision.${decision}` as StringKey)}
        </span>
      </header>
      <p className="mb-2 text-xs text-ink-200">{reasonsText(group.reasons, group.members.length)}</p>
      <Members group={group} />

      {decision !== 'rejected' ? (
        <div className="mt-2">
          <Preview group={group} />
        </div>
      ) : null}

      {summary ? <p className="mt-2 text-xs text-ink-300">{summary}</p> : null}
      {notes.length ? (
        <ul className="mt-2 space-y-1">
          {notes.map((note) => (
            <li key={note} className="flex items-start gap-1.5 text-xs text-warn">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              {note}
            </li>
          ))}
        </ul>
      ) : null}

      {job ? (
        <div className="mt-2">
          <div className="h-1 w-full overflow-hidden rounded-full bg-ink-700">
            <div className="h-full bg-ink-300 transition-[width]" style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <p className="mt-1 text-xs text-ink-300">
            {job.state === 'running'
              ? t('merges.running', { percent: Math.round(job.progress * 100) })
              : t('merges.queued')}
          </p>
        </div>
      ) : null}
      {decision === 'failed' && group.error ? (
        <p className="mt-2 text-xs text-bad">{t('merges.failed', { reason: group.error })}</p>
      ) : null}
      {decision === 'accepted' && group.result_photo_id !== null ? (
        <p className="mt-2 text-xs text-good">{t('merges.done')}</p>
      ) : null}

      <footer className="mt-3 flex flex-wrap gap-1.5">
        {decision === 'proposed' || decision === 'failed' ? (
          <>
            {canPreview ? (
              <Button size="sm" variant="outline" disabled={pending} onClick={() => onAction('preview')}>
                <Eye size={13} />
                {t('merges.preview')}
              </Button>
            ) : null}
            <Button size="sm" variant="primary" disabled={pending} onClick={() => onAction('accept')}>
              <Check size={13} />
              {decision === 'failed' ? t('merges.retry') : t('merges.accept')}
            </Button>
            <Button size="sm" variant="ghost" disabled={pending} onClick={() => onAction('edit')}>
              <Pencil size={13} />
              {t('merges.edit')}
            </Button>
            <Button size="sm" variant="ghost" disabled={pending} onClick={() => onAction('reject')}>
              <X size={13} />
              {t('merges.reject')}
            </Button>
          </>
        ) : null}
        {decision === 'accepted' ? (
          <>
            {group.result_photo_id !== null ? (
              <Button size="sm" variant="outline" onClick={() => onAction('open')}>
                {t('merges.open')}
              </Button>
            ) : null}
            <Button size="sm" variant="ghost" disabled={pending} onClick={() => onAction('undo')}>
              <RotateCcw size={13} />
              {t('merges.undo')}
            </Button>
          </>
        ) : null}
        {decision === 'rejected' ? (
          <Button size="sm" variant="ghost" disabled={pending} onClick={() => onAction('undo')}>
            <RotateCcw size={13} />
            {t('merges.restore')}
          </Button>
        ) : null}
      </footer>
    </article>
  )
}
