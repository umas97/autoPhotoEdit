// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The batches of the Export screen: progress, pause and resume (phase 8), and
// the photos that did not make it with the reason in Italian and "Riprova"
// (section 19). A skipped photo is listed apart from a failed one: skipping
// is what the user asked for, not a problem (section 16.2).
import { Pause, Play, RotateCcw, Square } from 'lucide-react'
import type { ExportBatch } from '../../lib/exportTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'

function when(iso: string): string {
  return new Date(iso).toLocaleString('it-IT', { dateStyle: 'short', timeStyle: 'short' })
}

export function BatchList({
  batches,
  busy,
  onAction,
}: {
  batches: ExportBatch[]
  busy: boolean
  onAction: (batch: ExportBatch, action: 'pause' | 'resume' | 'cancel' | 'retry') => void
}) {
  if (batches.length === 0) return null
  return (
    <section className="space-y-2">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-300">
        {t('export.batches')}
      </h2>
      <ul className="space-y-2">
        {batches.map((batch) => {
          const { done, skipped, failed, queued } = batch.counts
          const finished = done + skipped + failed
          const fraction = batch.total ? finished / batch.total : 0
          const live = batch.state === 'running' || batch.state === 'paused'
          return (
            <li key={batch.id} className="rounded-md border border-ink-700 p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-ink-50">{t('export.batch', { date: when(batch.created_at) })}</span>
                <span className={cn('rounded px-1.5 text-[11px]',
                  batch.state === 'done' ? 'bg-good/30 text-ink-50'
                    : batch.state === 'running' ? 'bg-ink-600 text-ink-50' : 'bg-ink-700 text-ink-200')}>
                  {t(`export.batchState.${batch.state}` as StringKey)}
                </span>
                <span className="ml-auto flex gap-1">
                  {batch.state === 'running' ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => onAction(batch, 'pause')}>
                      <Pause size={13} />
                      {t('export.pause')}
                    </Button>
                  ) : null}
                  {batch.state === 'paused' ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => onAction(batch, 'resume')}>
                      <Play size={13} />
                      {t('export.resume')}
                    </Button>
                  ) : null}
                  {live ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => onAction(batch, 'cancel')}>
                      <Square size={13} />
                      {t('export.cancel')}
                    </Button>
                  ) : null}
                  {failed > 0 && !live ? (
                    <Button size="sm" variant="ghost" disabled={busy} onClick={() => onAction(batch, 'retry')}>
                      <RotateCcw size={13} />
                      {t('export.retry')}
                    </Button>
                  ) : null}
                </span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded bg-ink-800">
                <div className="h-full bg-ink-200 transition-[width]" style={{ width: `${fraction * 100}%` }} />
              </div>
              <p className="mt-1 text-xs tabular-nums text-ink-300">
                {t('export.batchCounts', { done, skipped, failed, queued })}
              </p>
              {(batch.outcomes.renamed ?? 0) + (batch.outcomes.overwritten ?? 0) > 0 ? (
                <p className="text-xs text-ink-400">
                  {t('export.outcomes', {
                    renamed: batch.outcomes.renamed ?? 0,
                    overwritten: batch.outcomes.overwritten ?? 0,
                  })}
                </p>
              ) : null}
              {batch.output_dir ? (
                <p className="truncate text-xs text-ink-500" title={batch.output_dir}>
                  {t('export.to', { folder: batch.output_dir })}
                </p>
              ) : null}
              {batch.problems && batch.problems.length > 0 ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-ink-200">
                    {t('export.problems')} ({batch.problems.length})
                  </summary>
                  <ul className="mt-1 max-h-48 space-y-0.5 overflow-y-auto text-xs">
                    {batch.problems.map((problem) => (
                      <li key={problem.item_id} className="flex gap-2">
                        <span className="shrink-0 text-ink-100">{problem.filename}</span>
                        <span className={problem.state === 'failed' ? 'text-bad' : 'text-ink-400'}>
                          {problem.state === 'failed' ? problem.error : t('export.skippedItem')}
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
