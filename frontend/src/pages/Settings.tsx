// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Settings screen (section 10): the preferences that belong to no project.
// Author and copyright (section 16.3), the disk -- how much each kind of file
// takes, the cache quota and "Svuota cache", which says *before* running what
// it frees, how many merges it costs and what it never touches (section 20.3)
// -- what closing the window does while work is running (section 21.3), and
// how to back up the catalogue (section 20.2), which is a terminal command --
// and the lensfun data the lens correction uses, which are the program's.
import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { StorageCategory } from '../lib/types'
import { formatBytes } from '../lib/utils'
import { t, type StringKey } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { CommittedInput } from '../components/ui/CommittedInput'
import { Dialog } from '../components/ui/Dialog'
import { Field } from '../components/ui/Field'
import { LensfunData } from '../components/LensfunData'

const ORDER: StorageCategory[] = [
  'proxies', 'previews', 'developed', 'stages', 'merges', 'intermediates', 'masks', 'models',
]
const CLOSE = ['ask', 'continue', 'pause', 'stop'] as const

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-ink-700 bg-ink-850 p-4">
      <h2 className="text-sm font-semibold text-ink-50">{title}</h2>
      {hint ? <p className="mt-1 text-xs text-ink-300">{hint}</p> : null}
      <div className="mt-3">{children}</div>
    </section>
  )
}

export function SettingsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings })
  const storage = useQuery({ queryKey: ['storage'], queryFn: api.storage, staleTime: 0 })
  const [quotaError, setQuotaError] = useState(false)
  const [asking, setAsking] = useState(false)
  const [cleared, setCleared] = useState<number | null>(null)

  const write = useMutation({
    mutationFn: ({ key, value }: { key: string; value: unknown }) => api.writeSetting(key, value),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
      void queryClient.invalidateQueries({ queryKey: ['storage'] })
      void queryClient.invalidateQueries({ queryKey: ['window'] })
    },
  })
  const clear = useMutation({
    mutationFn: api.clearCache,
    onSuccess: (answer) => {
      setAsking(false)
      setCleared(answer.freed)
      void queryClient.invalidateQueries({ queryKey: ['storage'] })
      void queryClient.invalidateQueries({ queryKey: ['photos'] })
    },
  })

  const values = settings.data ?? {}
  const text = (key: string) => (typeof values[key] === 'string' ? (values[key] as string) : '')
  const disk = storage.data
  const close = (values.on_window_close as string | null | undefined) ?? 'ask'

  return (
    <AppShell>
      <div className="mx-auto h-full max-w-3xl space-y-4 overflow-y-auto p-4">
        <Section title={t('settings.author')} hint={t('settings.authorHint')}>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t('settings.artist')}>
              <CommittedInput value={text('artist')} onCommit={(value) => write.mutate({ key: 'artist', value })} />
            </Field>
            <Field label={t('settings.copyright')}>
              <CommittedInput value={text('copyright')} onCommit={(value) => write.mutate({ key: 'copyright', value })} />
            </Field>
          </div>
        </Section>

        <Section title={t('settings.storage')} hint={t('settings.storageHint')}>
          {disk ? (
            <>
              <div className="flex flex-wrap items-end gap-3">
                <Field label={t('settings.quota')}>
                  <CommittedInput value={String(Math.round(disk.limit_gb * 10) / 10)} inputMode="numeric"
                    onCommit={(raw) => {
                      const value = Number(raw.replace(',', '.'))
                      const valid = Number.isFinite(value) && value >= 1 && value <= 10_000
                      setQuotaError(!valid)
                      if (valid) write.mutate({ key: 'cache_max_gb', value })
                    }} />
                </Field>
                <p className="pb-2 text-xs text-ink-200">
                  {t('settings.cacheUsed', { used: formatBytes(disk.cache_bytes), limit: formatBytes(disk.limit_bytes) })}
                </p>
                <Button className="ml-auto" variant="outline" disabled={!disk.cache_bytes}
                  onClick={() => { setCleared(null); setAsking(true) }}>
                  {t('settings.clear')}
                </Button>
              </div>
              {quotaError ? <p className="mt-1 text-xs text-bad">{t('settings.quotaInvalid')}</p> : null}
              {cleared !== null ? <p className="mt-2 text-xs text-ink-200">{t('settings.cleared', { size: formatBytes(cleared) })}</p> : null}
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-ink-700">
                <div className="h-full bg-ink-300" style={{ width: `${Math.min(100, (100 * disk.cache_bytes) / disk.limit_bytes)}%` }} />
              </div>
              <table className="mt-3 w-full text-xs">
                <tbody>
                  {ORDER.map((key) => (
                    <tr key={key} className="border-t border-ink-800">
                      <td className="py-1 text-ink-200">{t(`settings.cat.${key}` as StringKey)}</td>
                      <td className="py-1 text-right tabular-nums text-ink-300">{t('settings.files', { count: disk.categories[key].files })}</td>
                      <td className="w-24 py-1 text-right tabular-nums text-ink-100">{formatBytes(disk.categories[key].bytes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : null}
        </Section>

        <Section title={t('settings.lenses')} hint={t('settings.lensesHint')}>
          <LensfunData />
        </Section>

        <Section title={t('settings.window')} hint={t('settings.windowHint')}>
          <div className="flex flex-wrap gap-2">
            {CLOSE.map((choice) => (
              <Button key={choice} size="sm" variant={close === choice ? 'secondary' : 'ghost'}
                onClick={() => write.mutate({ key: 'on_window_close', value: choice === 'ask' ? null : choice })}>
                {t(`settings.close.${choice}` as StringKey)}
              </Button>
            ))}
          </div>
        </Section>

        <Section title={t('settings.backup')} hint={t('settings.backupHint')}>
          <pre className="rounded bg-ink-950 p-2 text-xs text-ink-100">
            autophotoedit backup ~/catalogo-autophotoedit.db{'\n'}autophotoedit restore ~/catalogo-autophotoedit.db
          </pre>
          <p className="mt-2 text-xs text-ink-300">{t('settings.restoreHint')}</p>
          {disk ? <p className="mt-1 text-xs text-ink-300">{t('settings.backupMasks', { path: disk.masks_dir })}</p> : null}
          {disk ? <p className="mt-1 text-xs text-ink-300">{t('settings.backupRetouch', { path: disk.retouch_dir })}</p> : null}
        </Section>

        <Section title={t('settings.diagnostics')} hint={t('settings.diagnosticsHint')}>
          <Button variant="outline" onClick={() => navigate('/problemi')}>{t('settings.openProblems')}</Button>
        </Section>
      </div>

      <Dialog open={asking} onOpenChange={setAsking} title={t('settings.clearTitle')}
        footer={
          <>
            <Button variant="ghost" onClick={() => setAsking(false)}>{t('common.cancel')}</Button>
            <Button variant="primary" disabled={clear.isPending} onClick={() => clear.mutate()}>{t('settings.clearYes')}</Button>
          </>
        }>
        {disk ? (
          <div className="space-y-2 text-sm text-ink-100">
            <p>{t('settings.clearFrees', { size: formatBytes(disk.clear.frees_bytes) })}</p>
            {disk.clear.merges_to_rebuild ? (
              <p className="text-warn">{t('settings.clearMerges', { count: disk.clear.merges_to_rebuild })}</p>
            ) : null}
            <p className="text-ink-300">{t('settings.clearProxies')}</p>
            <p className="text-ink-300">
              {t('settings.clearKeeps', {
                masks: formatBytes(disk.clear.keeps.masks?.bytes ?? 0),
                models: formatBytes(disk.clear.keeps.models?.bytes ?? 0),
              })}
            </p>
          </div>
        ) : null}
      </Dialog>
    </AppShell>
  )
}
