// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Screen 1 of section 10: the projects, their state, and the way back into one
// that was left half-done.
//
// Creating a project is where the folder is chosen. The browser's own picker
// is no use: it gives a server a copy of the files, never a path, and copying
// a card of RAWs to hand it back to a program on the same machine would be
// absurd. So the server lists its folders and "Sfoglia…" walks them
// (components/FolderBrowser.tsx); the path can still be typed, and either way
// the server answers whether it exists -- the check that matters (section 2).
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FolderOpen, FolderSearch, ListFilter, Plus, Trash2 } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import type { Project } from '../lib/types'
import { t } from '../i18n/it'
import { formatWhen } from '../lib/utils'
import { AppShell } from '../components/AppShell'
import { FolderBrowser } from '../components/FolderBrowser'
import { Button } from '../components/ui/Button'
import { Dialog } from '../components/ui/Dialog'
import { Field, TextInput } from '../components/ui/Field'

export function ProjectsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [toDelete, setToDelete] = useState<Project | null>(null)

  const projects = useQuery({ queryKey: ['projects'], queryFn: api.listProjects })
  const remove = useMutation({
    mutationFn: (id: number) => api.deleteProject(id),
    // Out of the list at once, then the server's word on it.
    onSuccess: (_, id) => {
      queryClient.setQueryData<Project[]>(['projects'], (list) => list?.filter((p) => p.id !== id))
      return queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  return (
    <AppShell
      actions={
        <Button size="sm" variant="primary" onClick={() => setCreating(true)}>
          <Plus size={14} />
          {t('projects.new')}
        </Button>
      }
    >
      <div className="h-full overflow-y-auto px-6 py-5">
        <header className="mb-5">
          <h1 className="text-lg font-semibold text-ink-50">{t('projects.title')}</h1>
          <p className="text-sm text-ink-300">{t('projects.subtitle')}</p>
        </header>

        {projects.isLoading ? <p className="text-sm text-ink-400">{t('common.loading')}</p> : null}
        {projects.error ? (
          <p className="text-sm text-bad">{(projects.error as ApiError).message}</p>
        ) : null}
        {projects.data?.length === 0 ? (
          <p className="text-sm text-ink-400">{t('projects.empty')}</p>
        ) : null}

        <ul className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {projects.data?.map((project) => (
            <li
              key={project.id}
              className="rounded-lg border border-ink-700 bg-ink-850 p-3 transition-colors hover:border-ink-500"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h2 className="truncate font-medium text-ink-50">{project.name}</h2>
                  <p className="truncate text-xs text-ink-400" title={project.source_dir}>
                    {project.source_dir}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={t('projects.delete')}
                  onClick={() => setToDelete(project)}
                >
                  <Trash2 size={14} />
                </Button>
              </div>

              <dl className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-300">
                <dd>{t('projects.card.photos', { count: project.photo_count })}</dd>
                {project.missing_count > 0 ? (
                  <dd className="text-warn">
                    {t('projects.card.missing', { count: project.missing_count })}
                  </dd>
                ) : null}
                {project.pending_jobs > 0 ? (
                  <dd>{t('projects.card.pending', { count: project.pending_jobs })}</dd>
                ) : null}
                <dd className="ml-auto text-ink-400">
                  {t('projects.card.updated', { when: formatWhen(project.updated_at) })}
                </dd>
              </dl>

              {project.source_missing ? (
                <p className="mt-2 text-xs text-warn">{t('projects.card.sourceMissing')}</p>
              ) : null}

              <div className="mt-3 flex gap-2">
                <Button
                  size="sm"
                  onClick={() =>
                    // A project still being culled opens where the user left
                    // it: nothing is developed yet, so the viewer would be a
                    // grid of placeholders.
                    navigate(
                      project.status === 'culling'
                        ? `/progetti/${project.id}/cernita`
                        : `/progetti/${project.id}/foto`,
                    )
                  }
                >
                  <FolderOpen size={14} />
                  {t('projects.open')}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => navigate(`/progetti/${project.id}/cernita`)}
                >
                  <ListFilter size={14} />
                  {t('nav.culling')}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => navigate(`/progetti/${project.id}/import`)}
                >
                  {t('nav.import')}
                </Button>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <CreateProjectDialog open={creating} onOpenChange={setCreating} />

      <Dialog
        open={toDelete !== null}
        onOpenChange={(open) => !open && setToDelete(null)}
        title={t('projects.deleteTitle')}
        description={
          toDelete ? t('projects.deleteBody', { name: toDelete.name }) : undefined
        }
        footer={
          <>
            <Button variant="ghost" onClick={() => setToDelete(null)}>
              {t('common.cancel')}
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                if (toDelete) remove.mutate(toDelete.id)
                setToDelete(null)
              }}
            >
              {t('projects.delete')}
            </Button>
          </>
        }
      />
    </AppShell>
  )
}

function CreateProjectDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [source, setSource] = useState('')
  const [browsing, setBrowsing] = useState(false)

  // The project is created empty and the import screen does the importing:
  // that is where section 7.1's choice -- editing straight away, or culling
  // first -- is asked, and importing here would answer it without asking.
  const create = useMutation({
    mutationFn: () =>
      api.createProject({ name: name.trim(), source_dir: source.trim(), import_now: false }),
    onSuccess: async (project) => {
      await queryClient.invalidateQueries({ queryKey: ['projects'] })
      onOpenChange(false)
      setName('')
      setSource('')
      setBrowsing(false)
      navigate(`/progetti/${project.id}/import`)
    },
  })

  const ready = name.trim().length > 0 && source.trim().length > 0

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('projects.new')}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="primary"
            disabled={!ready || create.isPending}
            onClick={() => create.mutate()}
          >
            {create.isPending ? t('projects.form.creating') : t('projects.form.create')}
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Field label={t('projects.form.name')}>
          <TextInput
            value={name}
            placeholder={t('projects.form.namePlaceholder')}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        <Field label={t('projects.form.source')}>
          <div className="flex gap-2">
            <TextInput
              value={source}
              placeholder={t('projects.form.sourcePlaceholder')}
              spellCheck={false}
              onChange={(event) => setSource(event.target.value)}
            />
            <Button type="button" variant="outline" aria-expanded={browsing}
              onClick={() => setBrowsing((open) => !open)}>
              <FolderSearch size={14} />
              {t('folders.browse')}
            </Button>
          </div>
        </Field>
        {browsing ? (
          <FolderBrowser
            start={source.trim()}
            onClose={() => setBrowsing(false)}
            onChoose={(path) => {
              setSource(path)
              // The folder's own name is usually the event's: a start, not a rule.
              if (!name.trim()) setName(path.split('/').filter(Boolean).pop() ?? '')
              setBrowsing(false)
            }}
          />
        ) : null}
        <p className="text-xs text-ink-400">{t('projects.form.nextStep')}</p>
        {create.error ? (
          <p className="text-sm text-bad">{(create.error as ApiError).message}</p>
        ) : null}
      </div>
    </Dialog>
  )
}
