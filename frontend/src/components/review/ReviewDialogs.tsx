// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Two dialogs of the review.
//
// **Soglia e pesi** -- section 9.1: the threshold of escalation "regolabile in
// UI" and the weights "configurabili". Each slider is saved when released; the
// server answers with the whole review, so the queue on the left changes under
// the dialog as the threshold moves.
//
// **Le tue correzioni** -- section 9.2.4: the corrections become training pairs
// only "su conferma esplicita dell'utente a fine progetto". A built-in profile
// cannot be retrained, so its corrections found a new profile, whose name the
// user can change here.
import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../../lib/api'
import { reviewApi } from '../../lib/reviewApi'
import { sampleThumbUrl } from '../../lib/stylesApi'
import type { ReviewOverview, WeightCode } from '../../lib/reviewTypes'
import { t, type StringKey } from '../../i18n/it'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Field, TextInput } from '../ui/Field'
import { Slider } from '../ui/Slider'

const WEIGHTS: WeightCode[] = [
  'far',
  'ambiguous',
  'extrapolation',
  'white_balance',
  'burnt',
  'crushed',
  'geometry',
  'lens',
]

export function ReviewSettingsDialog({
  projectId,
  review,
  open,
  onOpenChange,
}: {
  projectId: number
  review: ReviewOverview
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const queryClient = useQueryClient()
  const [threshold, setThreshold] = useState(review.threshold)
  const [weights, setWeights] = useState(review.weights)
  useEffect(() => {
    if (open) {
      setThreshold(review.threshold)
      setWeights(review.weights)
    }
  }, [open, review.threshold, review.weights])

  const save = useMutation({
    mutationFn: (body: { threshold?: number; weights?: Partial<Record<WeightCode, number>> }) =>
      reviewApi.reviewSettings(projectId, body),
    onSuccess: (updated) => queryClient.setQueryData(['review', projectId], updated),
  })

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('review.settingsTitle')}
      description={t('review.settingsBody')}
      footer={
        <>
          <Button
            variant="ghost"
            onClick={() => {
              setThreshold(0.55)
              setWeights(review.default_weights)
              save.mutate({ threshold: 0.55, weights: review.default_weights })
            }}
          >
            {t('review.defaults')}
          </Button>
          <Button variant="primary" onClick={() => onOpenChange(false)}>
            {t('common.close')}
          </Button>
        </>
      }
    >
      <div className="space-y-1">
        <Slider
          label={t('review.thresholdLabel')}
          value={threshold}
          min={0}
          max={1}
          step={0.05}
          neutral={0.55}
          onChange={setThreshold}
          onCommit={(value) => save.mutate({ threshold: value })}
        />
        <hr className="my-2 border-ink-700" />
        {WEIGHTS.map((code) => (
          <Slider
            key={code}
            label={t(`review.weight.${code}` as StringKey)}
            value={weights[code]}
            min={0}
            max={1}
            step={0.05}
            neutral={review.default_weights[code]}
            onChange={(value) => setWeights({ ...weights, [code]: value })}
            onCommit={(value) => save.mutate({ weights: { ...weights, [code]: value } })}
          />
        ))}
        <p className="pt-2 text-xs text-ink-400">
          {t('review.summary', {
            approved: review.counts.approved,
            total: review.counts.total,
            queue: review.counts.queue,
          })}
        </p>
      </div>
    </Dialog>
  )
}

export function FeedbackDialog({
  projectId,
  open,
  onOpenChange,
}: {
  projectId: number
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const queryClient = useQueryClient()
  const summary = useQuery({
    queryKey: ['feedback', projectId],
    queryFn: () => reviewApi.feedback(projectId),
    enabled: open,
    // The thumbnails are rendered by a job: ask again until they are all there.
    refetchInterval: (query) =>
      open && query.state.data?.samples.some((s) => !s.has_thumbnail) ? 3_000 : false,
  })
  const [name, setName] = useState('')
  const [done, setDone] = useState<string | null>(null)
  useEffect(() => {
    if (open) setDone(null)
    if (summary.data?.suggested_name && !name) setName(summary.data.suggested_name)
  }, [open, summary.data, name])

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['feedback', projectId] })
    void queryClient.invalidateQueries({ queryKey: ['review', projectId] })
    void queryClient.invalidateQueries({ queryKey: ['styles'] })
  }
  const incorporate = useMutation({
    mutationFn: () => reviewApi.incorporateFeedback(projectId, name),
    onSuccess: (profile) => {
      setDone(t('review.incorporated', { name: profile.name }))
      refresh()
    },
  })
  const discard = useMutation({ mutationFn: () => reviewApi.discardFeedback(projectId), onSuccess: refresh })

  const data = summary.data
  const count = data?.proposed ?? 0
  const builtin = data?.profile?.builtin ?? false
  const error = (incorporate.error ?? discard.error) as ApiError | null
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('review.feedbackTitle')}
      description={t('review.feedbackBody')}
      footer={
        <>
          <Button variant="ghost" disabled={count === 0 || discard.isPending} onClick={() => discard.mutate()}>
            {t('review.discard')}
          </Button>
          <Button
            variant="primary"
            disabled={count === 0 || (builtin && count < 2) || incorporate.isPending}
            onClick={() => incorporate.mutate()}
          >
            {t('review.incorporate')}
          </Button>
        </>
      }
    >
      {done ? <p className="text-sm text-ink-100">{done}</p> : null}
      {!done && count === 0 ? <p className="text-sm text-ink-300">{t('review.feedbackNone')}</p> : null}
      {!done && count > 0 && data?.profile ? (
        <div className="space-y-3">
          <p className="text-sm text-ink-200">
            {builtin
              ? t('review.feedbackBuiltin', { name: data.profile.name, count })
              : t('review.feedbackLearned', { name: data.profile.name, count })}
          </p>
          {builtin ? (
            <Field label={t('review.feedbackName')} hint={count < 2 ? t('review.feedbackFew') : undefined}>
              <TextInput value={name} onChange={(event) => setName(event.target.value)} />
            </Field>
          ) : null}
          <ul className="grid max-h-48 grid-cols-6 gap-1 overflow-y-auto">
            {data.samples.map((sample) => (
              <li key={sample.id} title={sample.filename ?? ''}>
                <div className="aspect-square overflow-hidden rounded-sm border border-ink-700 bg-mat">
                  {sample.has_thumbnail ? (
                    <img src={sampleThumbUrl(sample.id)} alt="" className="h-full w-full object-cover" />
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {error ? <p className="mt-2 text-xs text-bad">{error.message}</p> : null}
    </Dialog>
  )
}
