// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Soggetto": a mask the segmentation finds -- the sky, the people, the skin.
//
// The models are not part of the program (section 17): the first time a
// subject is asked for, the panel offers the download, with its size and
// licence. Finding a subject is a job in a worker, about a second; the panel
// waits for it and then adds the mask, selected, like any other. Asked again
// on the same photo, the answer is already there.
import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Cloud, Download, UserRound, X } from 'lucide-react'
import { t, type StringKey } from '../../i18n/it'
import { api } from '../../lib/api'
import { newMask, withAddedMask } from '../../lib/masks'
import { SUBJECT_FEATURE, findSegment, segments } from '../../lib/masksApi'
import type { SegmentSubject } from '../../lib/maskTypes'
import { toast } from '../../lib/toast'
import type { EditParams } from '../../lib/types'
import { Button } from '../ui/Button'
import type { MaskEditor } from './useMaskEditor'

const SUBJECTS: Array<{ subject: SegmentSubject; icon: typeof Cloud }> = [
  { subject: 'sky', icon: Cloud },
  { subject: 'person', icon: UserRound },
  { subject: 'skin', icon: UserRound },
]

export function SegmentControls({
  photoId,
  params,
  onChange,
  editor,
}: {
  photoId: number
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  editor: MaskEditor
}) {
  const queryClient = useQueryClient()
  /** The subject the user asked for, added as soon as it is found. */
  const [wanted, setWanted] = useState<SegmentSubject | null>(null)
  const latest = useRef(params)
  latest.current = params

  const states = useQuery({
    queryKey: ['segments', photoId],
    queryFn: () => segments(photoId),
    refetchInterval: (query) =>
      query.state.data?.some((s) => s.state === 'queued' || s.state === 'running') ? 700 : false,
  })
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: (query) =>
      query.state.data?.some((m) => m.download?.state === 'running') ? 1_000 : false,
  })

  const find = useMutation({
    mutationFn: (subject: SegmentSubject) => findSegment(photoId, subject),
    onSuccess: (state) => {
      queryClient.setQueryData(['segments', photoId], (old: typeof states.data) =>
        old?.map((s) => (s.subject === state.subject ? state : s)),
      )
      void queryClient.invalidateQueries({ queryKey: ['segments', photoId] })
    },
    onError: (error) => {
      setWanted(null)
      toast((error as Error).message)
    },
  })

  const download = useMutation({
    mutationFn: (feature: string) => api.downloadModel(feature),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['models'] }),
  })

  // When the job has answered: add the mask, or say why not.
  const current = states.data?.find((s) => s.subject === wanted)
  useEffect(() => {
    if (!wanted || !current) return
    if (current.state === 'ready' && current.raster) {
      const mask = newMask('segment', latest.current, null, current.raster)
      mask.definition = { subject: wanted, raster: current.raster }
      mask.name = `${t(`masks.segment.${wanted}` as StringKey)}`
      const next = withAddedMask(latest.current, mask)
      onChange(next, true)
      editor.setSelected(next.masks.length - 1)
      editor.setSegmenting(false)
      setWanted(null)
    } else if (current.state === 'failed') {
      toast(current.error ?? t('masks.segment.failed'))
      setWanted(null)
    }
  }, [current, wanted, onChange, editor])

  // A model that finished downloading makes its subjects available.
  const finished = models.data?.some((m) => m.download?.state === 'done')
  useEffect(() => {
    if (finished) void queryClient.invalidateQueries({ queryKey: ['segments', photoId] })
  }, [finished, photoId, queryClient])

  return (
    <section className="mb-2 rounded-md border border-ink-700 bg-ink-900 p-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-ink-100">{t('masks.segment.title')}</h3>
        <Button size="sm" variant="ghost" onClick={() => editor.setSegmenting(false)}
          aria-label={t('common.close')}>
          <X size={13} />
        </Button>
      </div>
      <div className="mt-1 space-y-1.5">
        {SUBJECTS.map(({ subject, icon: Icon }) => {
          const state = states.data?.find((s) => s.subject === subject)
          const model = models.data?.find((m) => m.feature === SUBJECT_FEATURE[subject])
          const waiting = wanted === subject || state?.state === 'queued' || state?.state === 'running'
          const running = model?.download?.state === 'running'
          return (
            <div key={subject}>
              {state?.available ? (
                <Button size="sm" variant="secondary" className="w-full justify-start"
                  disabled={waiting || wanted !== null}
                  onClick={() => {
                    setWanted(subject)
                    find.mutate(subject)
                  }}>
                  <Icon size={13} />
                  {t(`masks.segment.${subject}` as StringKey)}
                  <span className="ml-auto text-[11px] text-ink-400">
                    {waiting
                      ? t('masks.segment.finding')
                      : state.state === 'ready'
                        ? t('masks.segment.found')
                        : ''}
                  </span>
                </Button>
              ) : model ? (
                <div className="rounded border border-ink-800 px-2 py-1.5 text-xs text-ink-300">
                  <p className="flex items-center gap-1.5 text-ink-200">
                    <Icon size={13} />
                    {t(`masks.segment.${subject}` as StringKey)}
                  </p>
                  <p className="mt-0.5">
                    {t('masks.segment.needsModel', {
                      size: model.size_mb.toFixed(0),
                      licence: model.licence,
                    })}
                  </p>
                  {model.notice ? (
                    <p className="mt-0.5 text-warn">{t(`culling.notice.${model.notice}` as StringKey)}</p>
                  ) : null}
                  <Button size="sm" variant="outline" className="mt-1"
                    disabled={running || download.isPending || !model.downloadable}
                    onClick={() => download.mutate(model.feature)}>
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
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
      <p className="mt-1.5 text-[11px] text-ink-500">{t('masks.segment.hint')}</p>
    </section>
  )
}
