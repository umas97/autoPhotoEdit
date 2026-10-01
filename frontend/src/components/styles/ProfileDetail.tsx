// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// One profile: what it is, how far its training has got, what it learned from
// and how it predicts. While pairs are still being inverted the page asks
// again every few seconds, and stops when none are left.
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Copy, Download, Trash2 } from 'lucide-react'
import { ApiError } from '../../lib/api'
import { styleExportUrl, stylesApi } from '../../lib/stylesApi'
import type { StyleSummary } from '../../lib/styleTypes'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { ManualPairing } from './ManualPairing'
import { SampleGrid } from './SampleGrid'

export function profileBadge(profile: StyleSummary): string {
  if (profile.builtin) return t('styles.badge.builtin')
  const total = profile.samples.pending + profile.samples.ready + profile.samples.failed +
    profile.samples.unreproducible
  if (profile.samples.pending > 0) {
    return t('styles.badge.training', { done: total - profile.samples.pending, total })
  }
  return profile.trained ? t('styles.badge.trained') : t('styles.badge.untrained')
}

export function ProfileDetail({
  profileId,
  onDeleted,
  onDuplicated,
}: {
  profileId: number
  onDeleted: () => void
  onDuplicated: (profile: StyleSummary) => void
}) {
  const queryClient = useQueryClient()
  const [askDelete, setAskDelete] = useState(false)
  const detail = useQuery({
    queryKey: ['style', profileId],
    queryFn: () => stylesApi.style(profileId),
    refetchInterval: (query) => ((query.state.data?.samples.pending ?? 0) > 0 ? 3_000 : false),
  })
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['style', profileId] })
    void queryClient.invalidateQueries({ queryKey: ['styles'] })
  }
  const exclude = useMutation({
    mutationFn: ({ id, excluded }: { id: number; excluded: boolean }) => stylesApi.excludeSample(id, excluded),
    onSuccess: refresh,
  })
  const remove = useMutation({ mutationFn: (id: number) => stylesApi.deleteSample(id), onSuccess: refresh })
  const duplicate = useMutation({
    mutationFn: (name: string) => stylesApi.duplicateStyle(profileId, name),
    onSuccess: (copy) => {
      void queryClient.invalidateQueries({ queryKey: ['styles'] })
      onDuplicated(copy)
    },
  })
  const destroy = useMutation({
    mutationFn: () => stylesApi.deleteStyle(profileId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['styles'] })
      onDeleted()
    },
  })

  const profile = detail.data
  if (!profile) return <p className="p-3 text-sm text-ink-400">{t('common.loading')}</p>
  const methods = profile.stats?.methods ?? {}
  const error = (duplicate.error ?? destroy.error ?? exclude.error ?? remove.error) as ApiError | null

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-base font-semibold text-ink-50">{profile.name}</h2>
          <p className="text-xs text-ink-300">
            {profileBadge(profile)}
            {profile.builtin ? '' : ` · ${t('styles.pairs', { count: profile.n_pairs })}`}
            {profile.trained_at
              ? ` · ${t('styles.trainedAt', { date: new Date(profile.trained_at).toLocaleString('it-IT') })}`
              : ''}
          </p>
          {profile.notes ? <p className="mt-1 max-w-prose text-sm text-ink-200">{profile.notes}</p> : null}
        </div>
        <div className="flex gap-1.5">
          <Button size="sm" variant="outline" disabled={duplicate.isPending}
            onClick={() => duplicate.mutate(t('styles.duplicateName', { name: profile.name }))}>
            <Copy size={13} />
            {t('styles.duplicate')}
          </Button>
          <a href={styleExportUrl(profile.id)} download>
            <Button size="sm" variant="outline">
              <Download size={13} />
              {t('styles.export')}
            </Button>
          </a>
          {!profile.builtin ? (
            <Button size="sm" variant="ghost" onClick={() => setAskDelete(true)}>
              <Trash2 size={13} />
              {t('styles.delete')}
            </Button>
          ) : null}
        </div>
      </header>

      {profile.builtin ? <p className="text-xs text-ink-400">{t('styles.builtinNote')}</p> : null}
      {profile.few_pairs ? (
        <p className="rounded border border-warn/40 bg-warn/10 p-2 text-xs text-warn">
          {t('styles.fewPairs', {
            count: profile.n_pairs, min: profile.min_pairs, recommended: profile.recommended_pairs,
          })}
        </p>
      ) : null}
      {error ? <p className="text-xs text-bad">{error.message}</p> : null}

      {profile.trained ? (
        <section>
          <h3 className="mb-1 text-sm font-medium text-ink-100">{t('styles.how')}</h3>
          <p className="text-xs text-ink-300">
            {t('styles.how.methods', {
              knn: methods.knn ?? 0, ridge: methods.ridge ?? 0,
              anchor: methods.anchor ?? 0, boost: methods.boost ?? 0,
            })}{' '}
            {profile.stats?.with_embedding ? t('styles.how.embedding') : t('styles.how.features')}
          </p>
        </section>
      ) : null}

      {!profile.builtin ? (
        <>
          <section>
            <h3 className="mb-2 text-sm font-medium text-ink-100">{t('styles.samples')}</h3>
            <SampleGrid
              samples={profile.sample_list}
              editable
              onExclude={(sample, excluded) => exclude.mutate({ id: sample.id, excluded })}
              onRemove={(sample) => remove.mutate(sample.id)}
            />
          </section>
          {profile.reference_dir ? (
            <section>
              <h3 className="mb-2 text-sm font-medium text-ink-100">{t('styles.manual.title')}</h3>
              <ManualPairing profileId={profile.id} />
            </section>
          ) : null}
        </>
      ) : null}

      <Dialog
        open={askDelete}
        onOpenChange={setAskDelete}
        title={t('styles.delete')}
        description={t('styles.deleteConfirm', { name: profile.name })}
        footer={
          <>
            <Button variant="ghost" onClick={() => setAskDelete(false)}>{t('common.cancel')}</Button>
            <Button variant="danger" disabled={destroy.isPending} onClick={() => destroy.mutate()}>
              {t('styles.delete')}
            </Button>
          </>
        }
      />
    </div>
  )
}
