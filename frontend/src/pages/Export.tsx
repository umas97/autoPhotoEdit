// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Export screen (section 10.6): the options on the left, on the right what
// they will do -- how many photos, the warnings, the files already in the
// destination -- the button, and the batches with pause and resume.
//
// Nothing here decides anything the server has not checked: every change of
// an option comes back with the server's own plan, and a start the server
// refuses because files collide opens the dialog of section 16.2 and tries
// again with the answers. While a batch runs the screen asks for news every
// second and a half, and stops asking when none runs (section 26: no polling
// at rest).
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, ClipboardCheck, Download } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { exportApi, exportConflicts } from '../lib/exportApi'
import type {
  ExportBatch,
  ExportConflictInfo,
  ExportScreen,
  ExportSettings,
} from '../lib/exportTypes'
import { t } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { ExportForm } from '../components/export/ExportForm'
import { BatchList } from '../components/export/BatchList'
import { ConflictDialog, type ConflictAnswers } from '../components/export/ConflictDialog'

export function ExportPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [conflicts, setConflicts] = useState<ExportConflictInfo[] | null>(null)

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const screen = useQuery({
    queryKey: ['export', id],
    queryFn: () => exportApi.exportScreen(id),
    refetchInterval: (query) =>
      query.state.data?.batches.some((b) => b.state === 'running') ? 1_500 : false,
  })
  const setScreen = (data: ExportScreen) => queryClient.setQueryData(['export', id], data)

  const update = useMutation({
    mutationFn: (changes: Partial<ExportSettings>) => exportApi.updateExport(id, changes),
    onSuccess: setScreen,
  })
  const global = useMutation({
    mutationFn: ({ key, value }: { key: 'artist' | 'copyright'; value: string }) =>
      api.writeSetting(key, value),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['export', id] }),
  })
  const start = useMutation({
    mutationFn: (answers: ConflictAnswers | null) => exportApi.startExport(id, answers ?? {}),
    onSuccess: () => {
      setConflicts(null)
      void queryClient.invalidateQueries({ queryKey: ['export', id] })
    },
    onError: (error) => {
      const pending = exportConflicts(error)
      if (pending) setConflicts(pending)
    },
  })
  const action = useMutation({
    mutationFn: ({ batch, verb }: { batch: ExportBatch; verb: 'pause' | 'resume' | 'cancel' | 'retry' }) =>
      exportApi.exportAction(batch.id, verb),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['export', id] }),
  })

  const data = screen.data
  const plan = data?.plan
  const running = data?.batches.some((b) => b.state === 'running') ?? false
  const startError =
    start.error && !exportConflicts(start.error) ? (start.error as ApiError).message : null

  return (
    <AppShell
      back={{ to: `/progetti/${id}/foto`, label: t('nav.viewer') }}
      title={project.data?.name}
      actions={
        <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/revisione`)}>
          <ClipboardCheck size={14} />
          {t('nav.review')}
        </Button>
      }
    >
      <div className="h-full overflow-y-auto p-4">
        <div className="mx-auto grid max-w-6xl gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <div>
            <h1 className="mb-3 text-base font-semibold text-ink-50">{t('export.title')}</h1>
            {data ? (
              <ExportForm
                projectId={id}
                settings={data.settings}
                artist={data.artist}
                copyright={data.copyright}
                sourceDir={data.source_dir}
                onChange={(changes) => update.mutate(changes)}
                onGlobal={(key, value) => global.mutate({ key, value })}
              />
            ) : (
              <p className="text-sm text-ink-400">{t('common.loading')}</p>
            )}
            {update.error ? (
              <p className="mt-2 text-xs text-bad">{(update.error as ApiError).message}</p>
            ) : null}
          </div>

          <aside className="space-y-4 lg:sticky lg:top-0 lg:self-start">
            <section className="space-y-2 rounded-md border border-ink-700 p-3">
              <p className="text-sm text-ink-50">
                {plan?.ok ? t('export.plan', { count: plan.count }) : t('export.planNone')}
              </p>
              {plan && !plan.ok ? <p className="text-xs text-bad">{plan.error}</p> : null}
              {plan?.ok && plan.conflicts.length > 0 ? (
                <p className="text-xs text-warn">
                  {t('export.conflicts', { count: plan.conflicts.length })}
                </p>
              ) : null}
              {(plan?.warnings ?? []).map((warning) => (
                <p key={warning} className="flex gap-1.5 text-xs text-ink-300">
                  <AlertTriangle size={13} className="mt-0.5 shrink-0 text-warn" />
                  {warning}
                </p>
              ))}
              <Button variant="primary" className="w-full"
                disabled={!plan?.ok || start.isPending || running}
                onClick={() => start.mutate(null)}>
                <Download size={14} />
                {start.isPending ? t('export.starting') : t('export.start', { count: plan?.count ?? 0 })}
              </Button>
              {startError ? <p className="text-xs text-bad">{startError}</p> : null}
            </section>

            <BatchList
              batches={data?.batches ?? []}
              busy={action.isPending}
              onAction={(batch, verb) => action.mutate({ batch, verb })}
            />
          </aside>
        </div>
      </div>
      {conflicts ? (
        <ConflictDialog
          conflicts={conflicts}
          folder={data?.settings.output_dir ?? ''}
          onCancel={() => setConflicts(null)}
          onDone={(answers) => start.mutate(answers)}
        />
      ) : null}
    </AppShell>
  )
}
