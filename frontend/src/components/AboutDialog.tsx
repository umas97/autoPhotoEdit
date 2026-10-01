// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Informazioni", from the main menu: what the program is, who built it, and
// which version is running. The version is the server's (``ape.__version__``
// through /api/health), because that is the program actually answering.
import { useQuery } from '@tanstack/react-query'
import { ExternalLink } from 'lucide-react'
import { api } from '../lib/api'
import { t } from '../i18n/it'
import { Button } from './ui/Button'
import { Dialog } from './ui/Dialog'

/** A name and an address, not words: they stay the same in every language. */
const AUTHOR = { name: 'umas97', url: 'https://github.com/umas97' }

export function AboutDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, enabled: open })
  const version = health.data
    ? t('about.version', { version: health.data.version })
    : health.isError
      ? t('about.versionUnknown')
      : t('common.loading')

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('about.title')}
      description={version}
      footer={
        <Button variant="ghost" onClick={() => onOpenChange(false)}>
          {t('common.close')}
        </Button>
      }
    >
      <div className="space-y-3 text-sm text-ink-200">
        <p>{t('about.summary')}</p>
        <p>{t('about.features')}</p>
        <div className="border-t border-ink-700 pt-3">
          <p>{t('about.credits', { author: AUTHOR.name })}</p>
          {/* A new window: the application's own has no address bar to come back with. */}
          <a
            href={AUTHOR.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-center gap-1.5 text-ink-50 underline-offset-2 hover:underline"
          >
            <ExternalLink size={13} />
            {AUTHOR.url.replace('https://', '')}
          </a>
          <p className="mt-2 text-xs text-ink-300">{t('about.license')}</p>
        </div>
      </div>
    </Dialog>
  )
}
