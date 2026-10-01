// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Problems panel (section 19): the photos in state "failed" with their
// reason in plain Italian, the traceback behind a "Dettagli tecnici" that
// closes again and copies, and "Riprova" for one photo or for all. Below them,
// the jobs that failed on something that is not a failed photo -- a training
// pair, a segmentation -- and the merges that could not be made (section 25.6),
// with "Riprova" and the way to the merges screen to change them first. The
// badge that leads here is in every header, fed by
// the progress socket: the count of failures is always visible.
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, FileArchive, RotateCw } from 'lucide-react'
import { api } from '../lib/api'
import { useProgress } from '../lib/useProgress'
import { formatWhen } from '../lib/utils'
import { t } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { DiagnosticsDialog } from '../components/problems/DiagnosticsDialog'
import { kindLabel } from '../components/merges/MergeCard'
import { Button } from '../components/ui/Button'

function Details({ text }: { text: string | null }) {
  const [copied, setCopied] = useState(false)
  if (!text) return null
  return (
    <details className="mt-1 text-xs">
      <summary className="cursor-pointer select-none text-ink-400 hover:text-ink-200">{t('problems.details')}</summary>
      <div className="relative mt-1">
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded bg-ink-950 p-2 pr-16 text-[11px] text-ink-200">
          {text}
        </pre>
        <Button size="sm" variant="outline" className="absolute right-1 top-1"
          onClick={() => void navigator.clipboard.writeText(text).then(() => setCopied(true))}>
          {t(copied ? 'problems.copied' : 'problems.copy')}
        </Button>
      </div>
    </details>
  )
}

export function ProblemsPage() {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { progress } = useProgress()
  // The socket says when the count moved; the list follows it.
  const problems = useQuery({
    queryKey: ['problems', progress?.problems ?? null],
    queryFn: () => api.problems(),
    placeholderData: (previous) => previous,
  })
  const [notice, setNotice] = useState<string | null>(null)
  const [diagnostics, setDiagnostics] = useState(false)
  const retry = useMutation({
    mutationFn: api.retryProblems,
    onSuccess: (answer) => {
      setNotice(t('problems.retried', { count: answer.retried }))
      void queryClient.invalidateQueries({ queryKey: ['problems'] })
    },
  })
  const data = problems.data

  return (
    <AppShell
      actions={
        <Button size="sm" variant="ghost" onClick={() => setDiagnostics(true)}>
          <FileArchive size={14} />
          {t('problems.diagnostics')}
        </Button>
      }>
      <div className="mx-auto h-full max-w-4xl overflow-y-auto p-4">
        <p className="text-sm text-ink-300">{t('problems.intro')}</p>
        <div className="mt-3 flex items-center gap-2">
          <Button size="sm" variant="primary" disabled={!data?.count || retry.isPending} onClick={() => retry.mutate({})}>
            <RotateCw size={13} />
            {t('problems.retryAll')}
          </Button>
          {notice ? <span className="text-xs text-ink-300">{notice}</span> : null}
        </div>

        {data && data.count === 0 ? <p className="mt-6 text-sm text-ink-200">{t('problems.none')}</p> : null}

        {data?.photos.length ? (
          <section className="mt-5">
            <h2 className="text-sm font-semibold text-ink-100">{t('problems.photos', { count: data.photos.length })}</h2>
            <ul className="mt-2 space-y-2">
              {data.photos.map((photo) => (
                <li key={photo.photo_id} className="rounded border border-ink-700 bg-ink-850 p-2.5">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={15} className="mt-0.5 shrink-0 text-bad" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink-50">
                        {photo.filename}
                        <span className="ml-2 text-xs text-ink-400">
                          {photo.project}
                          {photo.stage ? ` · ${t('problems.stage', { stage: photo.stage })}` : ''}
                          {photo.at ? ` · ${formatWhen(photo.at)}` : ''}
                        </span>
                      </div>
                      <p className="text-sm text-ink-200">{photo.reason}</p>
                      <Details text={photo.details} />
                    </div>
                    <Button size="sm" variant="outline" disabled={retry.isPending}
                      onClick={() => retry.mutate({ photo_ids: [photo.photo_id] })}>
                      {t('problems.retry')}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {data?.merges.length ? (
          <section className="mt-5">
            <h2 className="text-sm font-semibold text-ink-100">{t('problems.merges', { count: data.merges.length })}</h2>
            <ul className="mt-2 space-y-2">
              {data.merges.map((merge) => (
                <li key={merge.group_id} className="rounded border border-ink-700 bg-ink-850 p-2.5">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={15} className="mt-0.5 shrink-0 text-bad" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink-50">
                        {t('problems.mergeTitle', { kind: kindLabel(merge.kind), frames: merge.frames })}
                        <span className="ml-2 text-xs text-ink-400">{merge.project ?? ''}</span>
                      </div>
                      <p className="text-sm text-ink-200">{merge.reason}</p>
                    </div>
                    <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${merge.project_id}/fusioni`)}>
                      {t('problems.mergeEdit')}
                    </Button>
                    <Button size="sm" variant="outline" disabled={retry.isPending}
                      onClick={() => retry.mutate({ merge_ids: [merge.group_id] })}>
                      {t('problems.retry')}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {data?.jobs.length ? (
          <section className="mt-5">
            <h2 className="text-sm font-semibold text-ink-100">{t('problems.jobs', { count: data.jobs.length })}</h2>
            <ul className="mt-2 space-y-2">
              {data.jobs.map((job) => (
                <li key={job.job_id} className="rounded border border-ink-700 bg-ink-850 p-2.5">
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink-50">
                        {job.stage}
                        <span className="ml-2 text-xs text-ink-400">
                          {job.project ?? ''}
                          {job.photo_id ? ` · ${t('problems.photoOf', { id: job.photo_id })}` : ''}
                          {job.at ? ` · ${formatWhen(job.at)}` : ''}
                        </span>
                      </div>
                      <p className="text-sm text-ink-200">{job.reason}</p>
                      <Details text={job.details} />
                    </div>
                    <Button size="sm" variant="outline" disabled={retry.isPending}
                      onClick={() => retry.mutate({ job_ids: [job.job_id] })}>
                      {t('problems.retry')}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
      <DiagnosticsDialog open={diagnostics} onOpenChange={setDiagnostics} />
    </AppShell>
  )
}
