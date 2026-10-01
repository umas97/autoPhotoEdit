// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The scene model (CLIP), offered and never imposed (section 17): what it is,
// its licence and size, a download the user starts, with progress and a way to
// cancel. Without it the scenes are grouped on the image's own features and
// the screen says so -- nothing waits for a download.
//
// The progress is asked for only while a download is running: at rest the
// screen makes no requests (section 26).
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, X } from 'lucide-react'
import { api } from '../../lib/api'
import type { ModelState } from '../../lib/analysisTypes'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'

export function ModelCard({ projectId }: { projectId: number }) {
  const queryClient = useQueryClient()
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: (query) =>
      query.state.data?.some((model) => model.download?.state === 'running') ? 1_000 : false,
  })
  const model: ModelState | undefined = models.data?.find((m) => m.feature === 'embedding')

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['models'] })
    void queryClient.invalidateQueries({ queryKey: ['scenes', projectId] })
  }
  const start = useMutation({ mutationFn: () => api.downloadModel('embedding'), onSuccess: refresh })
  const cancel = useMutation({
    mutationFn: () => api.cancelModelDownload('embedding'),
    onSuccess: refresh,
  })
  const backfill = useMutation({
    mutationFn: () => api.queueAnalysis(projectId, true),
    onSuccess: refresh,
  })

  if (!model) return null
  const running = model.download?.state === 'running'
  const failed = model.download?.state === 'failed'
  const size = model.size_mb.toFixed(0)

  return (
    <section className="rounded-md border border-ink-700 bg-ink-900 p-3">
      <h2 className="text-sm font-semibold text-ink-100">{t('models.title')}</h2>
      <p className="mt-1 text-xs text-ink-300">{t('models.embedding.body', { size })}</p>

      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {model.available ? (
          <>
            <span className="text-ink-200">{t('models.ready')}</span>
            <Button size="sm" variant="ghost" disabled={backfill.isPending} onClick={() => backfill.mutate()}>
              {t('models.backfill')}
            </Button>
          </>
        ) : running ? (
          <>
            <progress
              className="h-1.5 w-40 accent-ink-200"
              value={model.download?.fraction ?? 0}
              max={1}
            />
            <span className="tabular-nums text-ink-300">
              {t('models.downloading', {
                percent: Math.round((model.download?.fraction ?? 0) * 100),
              })}
            </span>
            <Button size="sm" variant="ghost" onClick={() => cancel.mutate()}>
              <X size={12} />
              {t('models.cancel')}
            </Button>
          </>
        ) : model.reason === 'runtime_missing' ? (
          <span className="text-warn">{t('models.reason.runtime_missing')}</span>
        ) : (
          <>
            <Button size="sm" variant="outline" disabled={start.isPending} onClick={() => start.mutate()}>
              <Download size={12} />
              {t('models.download', { size })}
            </Button>
            {model.reason === 'checksum_mismatch' ? (
              <span className="text-warn">{t('models.reason.checksum_mismatch')}</span>
            ) : null}
          </>
        )}
        {failed ? (
          <span className="text-bad">
            {t('models.failed', { error: model.download?.error ?? '' })}
          </span>
        ) : null}
      </div>
    </section>
  )
}

