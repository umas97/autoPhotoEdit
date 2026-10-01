// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The eraser's IA engine is not part of the program (section 17, decision
// 37): the first time "Motore: IA" is chosen without it, the controls offer
// the download here, with its size, licence and the Places2 notice, as the
// "Soggetto" masks do. When it has arrived, the fills that failed for want of
// it are queued again -- the one thing the download itself asks for.
import { useEffect, useRef } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Download } from 'lucide-react'
import { t, type StringKey } from '../../i18n/it'
import { api } from '../../lib/api'
import { retouchApi } from '../../lib/retouchApi'
import { toast } from '../../lib/toast'
import { Button } from '../ui/Button'

const FEATURE = 'inpaint'

export function ModelOffer({ photoId, reason }: { photoId: number; reason: string | null }) {
  const queryClient = useQueryClient()
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: (query) =>
      query.state.data?.some((m) => m.download?.state === 'running') ? 1_000 : false,
  })
  const download = useMutation({
    mutationFn: () => api.downloadModel(FEATURE),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['models'] }),
    onError: (error: Error) => toast(error.message),
  })
  const model = models.data?.find((m) => m.feature === FEATURE)
  const running = model?.download?.state === 'running'

  // Arrived: forget the failures and fill again (only once per download).
  const done = model?.download?.state === 'done' && model.available
  const retried = useRef(false)
  useEffect(() => {
    if (!done || retried.current) return
    retried.current = true
    retouchApi
      .retry(photoId)
      .catch((error: Error) => toast(error.message))
      .finally(() => void queryClient.invalidateQueries({ queryKey: ['retouchStates', photoId] }))
  }, [done, photoId, queryClient])

  return (
    <div className="mt-1.5 rounded border border-ink-700 bg-ink-900 px-2 py-1.5 text-xs text-ink-300">
      <p className="text-ink-200">{t('retouch.ml.unavailable', { reason: reason ?? '' })}</p>
      {model && !model.available ? (
        <>
          <p className="mt-0.5">
            {t('masks.segment.needsModel', { size: model.size_mb.toFixed(0), licence: model.licence })}
          </p>
          {model.notice ? (
            <p className="mt-0.5 text-warn">{t(`culling.notice.${model.notice}` as StringKey)}</p>
          ) : null}
          <Button size="sm" variant="outline" className="mt-1"
            disabled={running || download.isPending || !model.downloadable}
            onClick={() => download.mutate()}>
            <Download size={12} />
            {running
              ? t('masks.segment.downloading', {
                  percent: Math.round((model.download?.fraction ?? 0) * 100),
                })
              : t('masks.segment.download')}
          </Button>
          {model.download?.state === 'failed' ? (
            <p className="mt-0.5 text-bad">{model.download.error}</p>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
