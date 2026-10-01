// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The scenes of a project (section 9.2) and what the analysis of phase 5 needs
// from the user: the lenses without a profile, and the scene model if they want
// it. It is the screen the review of phase 7 will grow from, and already the
// place to see how the program has understood the card.
//
// Each scene shows its representative -- the medoid, a real photo -- large, and
// the rest small. A click opens the photo in the viewer.
//
// While analyses are running the scenes are not recomputed (clustering half a
// card would renumber them under the user's eyes); the page says how many are
// left and asks again every few seconds, and stops asking when none are.
import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, proxyUrl } from '../lib/api'
import type { Photo } from '../lib/types'
import { t } from '../i18n/it'
import { useUi } from '../lib/store'
import { AppShell } from '../components/AppShell'
import { LensesCard } from '../components/scenes/LensesCard'
import { ModelCard } from '../components/scenes/ModelCard'

export function ScenesPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const select = useUi((state) => state.select)

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const scenes = useQuery({
    queryKey: ['scenes', id],
    queryFn: () => api.scenes(id),
    refetchInterval: (query) => ((query.state.data?.pending ?? 0) > 0 ? 3_000 : false),
  })
  const photos = useQuery({
    queryKey: ['photos', id, false],
    queryFn: () => api.listPhotos(id, 0, 2000, false),
    enabled: scenes.data !== undefined,
  })
  const byId = useMemo(() => {
    const map = new Map<number, Photo>()
    for (const photo of photos.data?.items ?? []) map.set(photo.id, photo)
    return map
  }, [photos.data])

  const open = (photoId: number) => {
    select(id, photoId)
    navigate(`/progetti/${id}/foto`)
  }

  const data = scenes.data
  return (
    <AppShell back={{ to: `/progetti/${id}/foto`, label: t('nav.viewer') }} title={project.data?.name}>
      <div className="grid h-full grid-cols-[1fr_minmax(18rem,26rem)] gap-3 overflow-hidden p-3">
        <section className="min-h-0 overflow-y-auto pr-1">
          <header className="mb-2 flex flex-wrap items-baseline gap-3">
            <h1 className="text-base font-semibold text-ink-50">{t('scenes.title')}</h1>
            {data ? (
              <span className="text-sm text-ink-300">
                {t('scenes.summary', { clusters: data.clusters.length, photos: data.clustered })}
              </span>
            ) : null}
          </header>
          {data && data.pending > 0 ? (
            <p className="mb-2 text-xs text-ink-300">{t('scenes.pending', { count: data.pending })}</p>
          ) : null}
          {data ? (
            <p className="mb-3 text-xs text-ink-400">
              {data.basis === 'embedding' ? t('scenes.basis.embedding') : t('scenes.basis.features')}
            </p>
          ) : null}
          {data && data.clusters.length === 0 ? (
            <p className="text-sm text-ink-400">{t('scenes.none')}</p>
          ) : null}

          <ul className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3">
            {(data?.clusters ?? []).map((scene) => {
              const lead = byId.get(scene.representative)
              return (
                <li key={scene.cluster} className="rounded-md border border-ink-700 bg-ink-900 p-2">
                  <div className="mb-1 flex items-baseline justify-between text-xs">
                    <span className="font-medium text-ink-100">{t('scenes.scene', { n: scene.cluster })}</span>
                    <span className="text-ink-400">{t('scenes.photos', { count: scene.photos.length })}</span>
                  </div>
                  <button
                    type="button"
                    title={t('scenes.open')}
                    onClick={() => open(scene.representative)}
                    className="block aspect-[3/2] w-full overflow-hidden rounded border border-ink-700 bg-mat hover:border-ink-400"
                  >
                    {lead?.has_proxy ? (
                      <img src={proxyUrl(lead)} alt={lead.filename} loading="lazy"
                        className="h-full w-full object-contain" />
                    ) : null}
                  </button>
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {scene.photos.slice(1, 13).map((photoId) => {
                      const photo = byId.get(photoId)
                      return (
                        <button
                          key={photoId}
                          type="button"
                          onClick={() => open(photoId)}
                          className="h-10 w-14 overflow-hidden rounded-sm border border-ink-700 bg-mat hover:border-ink-400"
                          title={photo?.filename}
                        >
                          {photo?.has_proxy ? (
                            <img src={proxyUrl(photo)} alt={photo.filename} loading="lazy"
                              className="h-full w-full object-cover" />
                          ) : null}
                        </button>
                      )
                    })}
                    {scene.photos.length > 13 ? (
                      <span className="self-center text-xs text-ink-400">+{scene.photos.length - 13}</span>
                    ) : null}
                  </div>
                </li>
              )
            })}
          </ul>
        </section>

        <aside className="min-h-0 space-y-3 overflow-y-auto">
          <LensesCard projectId={id} />
          <ModelCard projectId={id} />
        </aside>
      </div>
    </AppShell>
  )
}
