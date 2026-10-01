// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The merges screen (section 25.6), reached after the import and from culling
// and the viewer. Opening it is also what runs detection in a project that
// skipped culling (backend/ape/merge/service.py), so the first answer may say
// "searching" and the cards arrive as the panorama search ends.
//
// The list is asked for again only while something is running -- a preview, a
// full merge, the search -- and not at all otherwise: an idle screen costs no
// CPU (section 26).
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Image, ListFilter } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { busy, mergesApi } from '../lib/mergesApi'
import type { MergeDecision, MergeGroupView } from '../lib/mergeTypes'
import { useUi } from '../lib/store'
import { toast } from '../lib/toast'
import { t } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { MergeCard, type MergeAction } from '../components/merges/MergeCard'
import { MergeEditDialog, type MergeEdit } from '../components/merges/MergeEditDialog'
import { Button } from '../components/ui/Button'

const SECTIONS: MergeDecision[] = ['proposed', 'failed', 'accepted', 'rejected']

export function MergesPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const select = useUi((state) => state.select)
  const [editing, setEditing] = useState<MergeGroupView | null>(null)

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const merges = useQuery({
    queryKey: ['merges', id],
    queryFn: () => mergesApi.list(id),
    refetchInterval: (query) => (busy(query.state.data) ? 2_000 : false),
  })

  const settle = (group: MergeGroupView) => {
    queryClient.setQueryData(['merges', id], (view: typeof merges.data) =>
      view ? { ...view, groups: view.groups.map((g) => (g.id === group.id ? group : g)) } : view,
    )
    // A merge changes which photos the grid shows, and the undo brings them back.
    void queryClient.invalidateQueries({ queryKey: ['photos', id] })
    void queryClient.invalidateQueries({ queryKey: ['merges', id] })
  }
  const act = useMutation({
    mutationFn: ({ group, action }: { group: MergeGroupView; action: Exclude<MergeAction, 'edit' | 'open'> }) =>
      mergesApi[action](group.id),
    onSuccess: settle,
    onError: (error) => toast((error as ApiError).message),
  })
  const edit = useMutation({
    mutationFn: async ({ group, change }: { group: MergeGroupView; change: MergeEdit }) => {
      await mergesApi.edit(group.id, change)
      // Section 25.5.3: change it, then see it again.
      return mergesApi.preview(group.id)
    },
    onSuccess: (group) => {
      setEditing(null)
      settle(group)
    },
    onError: (error) => toast((error as ApiError).message),
  })

  const onAction = (group: MergeGroupView, action: MergeAction) => {
    if (action === 'edit') setEditing(group)
    else if (action === 'open') {
      if (group.result_photo_id !== null) select(id, group.result_photo_id)
      navigate(`/progetti/${id}/foto`)
    } else act.mutate({ group, action })
  }

  const view = merges.data
  const pendingId = act.isPending ? act.variables?.group.id : edit.isPending ? edit.variables?.group.id : null

  return (
    <AppShell
      back={{ to: `/progetti/${id}/foto`, label: t('nav.viewer') }}
      title={project.data?.name}
      actions={
        <>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/cernita`)}>
            <ListFilter size={14} />
            {t('nav.culling')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/foto`)}>
            <Image size={14} />
            {t('nav.viewer')}
          </Button>
        </>
      }
    >
      <div className="mx-auto h-full max-w-4xl overflow-y-auto px-6 py-5">
        <header className="mb-4">
          <h1 className="text-lg font-semibold text-ink-50">{t('merges.title')}</h1>
          <p className="text-sm text-ink-300">{t('merges.subtitle')}</p>
        </header>

        {merges.error ? <p className="text-sm text-bad">{(merges.error as ApiError).message}</p> : null}
        {!view ? (merges.isLoading ? <p className="text-sm text-ink-400">{t('common.loading')}</p> : null) : (
          <>
            {!view.enabled ? <p className="mb-3 text-sm text-ink-300">{t('merges.disabled')}</p> : null}
            {view.searching ? <p className="mb-3 text-sm text-ink-300">{t('merges.searching')}</p> : null}
            {view.groups.length === 0 && !view.searching ? (
              <p className="text-sm text-ink-300">{t('merges.empty')}</p>
            ) : null}
            {SECTIONS.map((decision) => {
              const groups = view.groups.filter((g) => g.decision === decision)
              if (groups.length === 0) return null
              return (
                <section key={decision} className="mb-5">
                  <h2 className="mb-2 text-sm font-medium text-ink-100">
                    {t(`merges.section.${decision}`, { count: groups.length })}
                  </h2>
                  <div className="space-y-3">
                    {groups.map((group) => (
                      <MergeCard key={group.id} group={group} pending={pendingId === group.id}
                        onAction={(action) => onAction(group, action)} />
                    ))}
                  </div>
                </section>
              )
            })}
          </>
        )}
      </div>
      <MergeEditDialog projectId={id} group={editing} onClose={() => setEditing(null)}
        onSave={(group, change) => edit.mutate({ group, change })} />
    </AppShell>
  )
}
