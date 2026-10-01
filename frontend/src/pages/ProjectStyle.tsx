// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A project's choice of style (section 8.4): every usable profile with its
// affinity to the project, the proposed one marked and preselected, and the
// user's override. Applying queues a prediction per photo; while they run the
// page asks every few seconds, and each answer also applies the finished
// predictions (the server does that lazily, on this very request).
//
// The "Coerenza" slider is section 8.4's lambda: how far photos of one scene
// move towards each other. It is saved when the handle is released.
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, ClipboardCheck, Palette } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { stylesApi } from '../lib/stylesApi'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { Slider } from '../components/ui/Slider'

export function ProjectStylePage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const profiles = useQuery({ queryKey: ['styles'], queryFn: stylesApi.styles })
  const state = useQuery({
    queryKey: ['projectStyle', id],
    queryFn: () => stylesApi.projectStyle(id),
    refetchInterval: (query) => ((query.state.data?.pending ?? 0) > 0 ? 3_000 : false),
  })
  const [choice, setChoice] = useState<number | null>(null)
  const [lambda, setLambda] = useState<number | null>(null)
  useEffect(() => {
    if (state.data && choice === null) setChoice(state.data.current ?? state.data.proposed)
    if (state.data && lambda === null) setLambda(state.data.coherence_lambda)
  }, [state.data, choice, lambda])

  const apply = useMutation({
    mutationFn: (body: { profile: number | null; lambda?: number }) =>
      stylesApi.chooseStyle(id, body.profile, body.lambda),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['projectStyle', id] })
      void queryClient.invalidateQueries({ queryKey: ['photos', id] })
    },
  })

  const usable = useMemo(() => (profiles.data ?? []).filter((p) => p.usable), [profiles.data])
  const data = state.data
  const current = data?.current ?? null

  return (
    <AppShell back={{ to: `/progetti/${id}/foto`, label: t('nav.viewer') }} title={project.data?.name}>
      <div className="h-full overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          <header className="flex items-baseline gap-3">
            <h1 className="text-base font-semibold text-ink-50">{t('projectStyle.title')}</h1>
            <Button size="sm" variant="ghost" className="ml-auto" onClick={() => navigate('/stili')}>
              <Palette size={14} />
              {t('projectStyle.library')}
            </Button>
          </header>
          <p className="text-sm text-ink-300">{t('projectStyle.intro')}</p>

          <ul className="divide-y divide-ink-700 rounded-md border border-ink-700">
            {usable.map((profile) => {
              const affinity = data?.affinity[String(profile.id)]
              return (
                <li key={profile.id}>
                  <label className={cn('flex cursor-pointer items-center gap-3 px-3 py-2',
                    choice === profile.id ? 'bg-ink-800' : 'hover:bg-ink-850')}>
                    <input type="radio" name="style" checked={choice === profile.id}
                      onChange={() => setChoice(profile.id)} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm text-ink-50">{profile.name}</span>
                      <span className="block truncate text-xs text-ink-400">{profile.notes}</span>
                    </span>
                    <span className="text-xs tabular-nums text-ink-300">
                      {affinity !== null && affinity !== undefined
                        ? t('projectStyle.affinity', { value: Math.round(affinity * 100) })
                        : profile.builtin ? '' : t('projectStyle.noAffinity')}
                    </span>
                    {data?.proposed === profile.id ? (
                      <span className="rounded bg-ink-600 px-1.5 text-[11px] text-ink-100">
                        {t('projectStyle.proposed')}
                      </span>
                    ) : null}
                    {current === profile.id ? (
                      <span className="rounded bg-good/30 px-1.5 text-[11px] text-ink-50">
                        {t('projectStyle.current')}
                      </span>
                    ) : null}
                  </label>
                </li>
              )
            })}
          </ul>

          <div className="flex items-center gap-3">
            <Button variant="primary" disabled={choice === null || choice === current || apply.isPending}
              onClick={() => apply.mutate({ profile: choice })}>
              <Check size={14} />
              {apply.isPending ? t('projectStyle.applying') : t('projectStyle.apply')}
            </Button>
            {apply.error ? <span className="text-xs text-bad">{(apply.error as ApiError).message}</span> : null}
            {current !== null && (data?.pending ?? 0) === 0 ? (
              <Button variant="secondary" className="ml-auto" onClick={() => navigate(`/progetti/${id}/revisione`)}>
                <ClipboardCheck size={14} />
                {t('nav.review')}
              </Button>
            ) : null}
          </div>

          <div className="max-w-sm">
            {lambda !== null ? (
              <Slider label={t('projectStyle.coherence')} value={lambda} min={0} max={1} step={0.05}
                neutral={0.35} onChange={setLambda}
                onCommit={(value) => apply.mutate({ profile: current, lambda: value })} />
            ) : null}
            <p className="text-xs text-ink-400">{t('projectStyle.coherenceHint')}</p>
          </div>

          <p className="text-sm text-ink-300">
            {!current
              ? t('projectStyle.none')
              : (data?.pending ?? 0) > 0
                ? t('projectStyle.pending', { count: data!.pending })
                : data
                  ? t('projectStyle.applied', {
                      written: data.status.styled,
                      kept: data.status.kept_user_edit,
                    })
                  : ''}
          </p>
        </div>
      </div>
    </AppShell>
  )
}
