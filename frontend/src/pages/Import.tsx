// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Screen 2 of section 10. It scans before it imports, because the number of
// RAWs in a folder is the one fact that tells the user they picked the right
// one, and a scan writes nothing.
//
// The choice between "straight to editing" and "cull first" (section 7.1) is
// asked here and asked explicitly: "nessun default nascosto". On a first import
// neither is preselected and the import button waits for an answer; on a
// re-import the project's own earlier answer is shown, because that is the
// user's choice and not a default.
//
// Culling first means no proxy is built at all: the embedded previews are
// analysed and nothing is developed until the user confirms a selection.
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FolderSearch, Import as ImportIcon } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { useProgress } from '../lib/useProgress'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { Checkbox, Field, TextInput } from '../components/ui/Field'

type After = 'editing' | 'culling'

export function ImportPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [folder, setFolder] = useState('')
  const [buildProxies, setBuildProxies] = useState(true)
  const [afterImport, setAfterImport] = useState<After | null>(null)

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const { progress } = useProgress(id)

  // A project that has been imported before already holds the user's answer.
  useEffect(() => {
    if (afterImport === null && project.data && project.data.photo_count > 0) {
      setAfterImport(project.data.culling_enabled ? 'culling' : 'editing')
    }
  }, [afterImport, project.data])

  const scan = useMutation({
    mutationFn: () => api.scan(id, folder.trim() || undefined),
  })

  const runImport = useMutation({
    mutationFn: () =>
      api.importFolder(id, {
        folder: folder.trim() || null,
        build_proxies: buildProxies,
        culling: afterImport === 'culling',
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['photos', id] })
      void queryClient.invalidateQueries({ queryKey: ['projects'] })
      void queryClient.invalidateQueries({ queryKey: ['project', id] })
      void queryClient.invalidateQueries({ queryKey: ['culling', id] })
    },
  })

  return (
    <AppShell back={{ to: '/', label: t('nav.projects') }} title={project.data?.name}>
      <div className="mx-auto h-full max-w-2xl overflow-y-auto px-6 py-5">
        <header className="mb-5">
          <h1 className="text-lg font-semibold text-ink-50">{t('import.title')}</h1>
          <p className="text-sm text-ink-300">{t('import.subtitle')}</p>
        </header>

        <div className="space-y-4 rounded-lg border border-ink-700 bg-ink-850 p-4">
          <Field label={t('import.folder')} hint={t('import.folderHint')}>
            <TextInput
              value={folder}
              spellCheck={false}
              placeholder={project.data?.source_dir}
              onChange={(event) => setFolder(event.target.value)}
            />
          </Field>

          <div className="flex gap-2">
            <Button variant="outline" disabled={scan.isPending} onClick={() => scan.mutate()}>
              <FolderSearch size={14} />
              {scan.isPending ? t('import.scanning') : t('import.scan')}
            </Button>
            <Button
              variant="primary"
              disabled={runImport.isPending || afterImport === null}
              title={afterImport === null ? t('import.chooseFirst') : undefined}
              onClick={() => runImport.mutate()}
            >
              <ImportIcon size={14} />
              {runImport.isPending ? t('import.running') : t('import.run')}
            </Button>
          </div>

          {scan.error ? (
            <p className="text-sm text-bad">{(scan.error as ApiError).message}</p>
          ) : null}
          {scan.data ? (
            <div className="rounded-md border border-ink-700 bg-ink-800 p-3 text-sm text-ink-200">
              <p>
                {t('import.scanResult', {
                  raws: scan.data.raws,
                  sidecars: scan.data.sidecars,
                  others: scan.data.other_files,
                })}
              </p>
              {scan.data.subdirectories.length > 0 ? (
                <p className="mt-1 text-xs text-ink-400">
                  {t('import.scanSubdirs', { names: scan.data.subdirectories.join(', ') })}
                </p>
              ) : null}
              {Object.keys(scan.data.rejected_formats).length > 0 ? (
                <p className="mt-1 text-xs text-warn">
                  {t('import.rejected', {
                    formats: Object.entries(scan.data.rejected_formats)
                      .map(([extension, count]) => `${extension} (${count})`)
                      .join(', '),
                  })}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>

        <section
          className={cn(
            'mt-4 space-y-1 rounded-lg border bg-ink-850 p-4',
            afterImport === null ? 'border-ink-500' : 'border-ink-700',
          )}
        >
          <h2 className="text-sm font-medium text-ink-100">{t('import.afterTitle')}</h2>
          {afterImport === null ? (
            <p className="text-xs text-ink-300">{t('import.chooseFirst')}</p>
          ) : null}
          {(['editing', 'culling'] as const).map((choice) => (
            <label
              key={choice}
              className="flex cursor-pointer items-start gap-2.5 rounded-md p-1.5 hover:bg-ink-800"
            >
              <input
                type="radio"
                name="after-import"
                className="mt-0.5 h-4 w-4 accent-ink-200"
                checked={afterImport === choice}
                onChange={() => setAfterImport(choice)}
              />
              <span>
                <span className="block text-sm text-ink-100">
                  {choice === 'editing' ? t('import.afterEditing') : t('import.afterCulling')}
                </span>
                <span className="block text-xs text-ink-400">
                  {choice === 'editing' ? t('import.afterEditingHint') : t('import.afterCullingHint')}
                </span>
              </span>
            </label>
          ))}
          {afterImport === 'editing' ? (
            <Checkbox
              checked={buildProxies}
              onChange={setBuildProxies}
              label={t('import.buildProxies')}
            />
          ) : null}
        </section>

        {runImport.error ? (
          <p className="mt-4 text-sm text-bad">{(runImport.error as ApiError).message}</p>
        ) : null}

        {runImport.data ? (
          <section className="mt-4 rounded-lg border border-ink-700 bg-ink-850 p-4">
            <h2 className="text-sm font-medium text-ink-50">{t('import.done')}</h2>
            <p className="mt-1 whitespace-pre-line text-sm text-ink-200">
              {runImport.data.summary}
            </p>
            {progress && progress.total > 0 ? (
              <div className="mt-3">
                <div className="h-1 w-full overflow-hidden rounded-full bg-ink-700">
                  <div
                    className="h-full bg-ink-300 transition-[width]"
                    style={{ width: `${Math.round(progress.progress * 100)}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-ink-400">
                  {afterImport === 'culling'
                    ? t('import.progressCulling', {
                        done: progress.photos_analysed,
                        total: progress.photos,
                      })
                    : t('import.progress', {
                        done: progress.photos_with_proxy,
                        total: progress.photos,
                      })}
                </p>
              </div>
            ) : null}
            <Button
              className="mt-3"
              variant="primary"
              onClick={() =>
                navigate(afterImport === 'culling' ? `/progetti/${id}/cernita` : `/progetti/${id}/foto`)
              }
            >
              {afterImport === 'culling' ? t('import.goToCulling') : t('import.goToPhotos')}
            </Button>
            <Button className="ml-2 mt-3" variant="ghost" onClick={() => navigate(`/progetti/${id}/fusioni`)}>
              {t('nav.merges')}
            </Button>
          </section>
        ) : null}
      </div>
    </AppShell>
  )
}
