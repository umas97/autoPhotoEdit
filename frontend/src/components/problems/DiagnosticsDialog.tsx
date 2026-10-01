// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Esporta diagnostica" (section 19): the list of what is about to go into the
// zip, file by file, with its content one click away -- *before* the zip is
// made. The user is about to attach it to a public issue and must be able to
// see every line of it. The zip itself is a download: the server writes
// nothing, the browser asks where.
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { formatBytes as size } from '../../lib/utils'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'

export function DiagnosticsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const plan = useQuery({ queryKey: ['diagnosticsPlan'], queryFn: api.diagnosticsPlan, enabled: open, staleTime: 0 })
  const [shown, setShown] = useState<string | null>(null)
  const entries = plan.data ?? []
  const total = entries.reduce((sum, entry) => sum + entry.size, 0)

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t('diagnostics.title')}
      description={t('diagnostics.description')}
      footer={
        <>
          <span className="mr-auto self-center text-xs text-ink-400">{t('diagnostics.cli')}</span>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>{t('common.cancel')}</Button>
          <a href="/api/diagnostics/bundle.zip" download
            className="inline-flex h-9 shrink-0 items-center whitespace-nowrap rounded-md bg-ink-100 px-3 text-sm font-medium text-ink-900 hover:bg-white"
            aria-disabled={!entries.length} onClick={() => onOpenChange(false)}>
            {t('diagnostics.download', { size: size(total) })}
          </a>
        </>
      }>
      {plan.isLoading ? <p className="text-sm text-ink-300">{t('diagnostics.loading')}</p> : null}
      <ul className="max-h-[50vh] space-y-2 overflow-y-auto">
        {entries.map((entry) => (
          <li key={entry.name} className="rounded border border-ink-700 p-2 text-xs">
            <div className="flex items-baseline gap-2">
              <code className="text-ink-50">{entry.name}</code>
              <span className="tabular-nums text-ink-400">{size(entry.size)}</span>
              <button type="button" className="ml-auto text-ink-300 underline-offset-2 hover:underline"
                onClick={() => setShown(shown === entry.name ? null : entry.name)}>
                {t(shown === entry.name ? 'diagnostics.hide' : 'diagnostics.show')}
              </button>
            </div>
            <p className="mt-0.5 text-ink-300">{entry.description}</p>
            {shown === entry.name ? (
              <>
                {entry.name.startsWith('log/') ? <p className="mt-1 text-ink-500">{t('diagnostics.logTail')}</p> : null}
                <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-all rounded bg-ink-950 p-2 text-[11px] text-ink-200">
                  {entry.preview}
                </pre>
              </>
            ) : null}
          </li>
        ))}
      </ul>
    </Dialog>
  )
}
