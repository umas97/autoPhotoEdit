// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Funziona tutto, ma l'esperienza è degradata e l'utente deve sapere perché"
// (section 21.2, fallback 3). The server knows which link of the chain answered
// and hands us the sentence; this shows it, offers the install button when the
// browser has one, and can be dismissed for good.
import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { installState, promptInstall, subscribeInstall, type InstallState } from '../pwa/install'
import { useUi } from '../lib/store'
import { t } from '../i18n/it'
import { Button } from './ui/Button'

export function WindowBanner() {
  const { data } = useQuery({ queryKey: ['window'], queryFn: api.window, staleTime: 30_000 })
  const [install, setInstall] = useState<InstallState>(installState)
  const dismissed = useUi((state) => state.tabNoticeDismissed)
  const dismiss = useUi((state) => state.dismissTabNotice)

  useEffect(() => subscribeInstall(setInstall), [])

  if (!data || data.mode !== 'tab' || dismissed) return null

  return (
    <div className="flex items-start gap-3 border-b border-ink-700 bg-ink-850 px-4 py-2 text-sm">
      <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-warn" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-ink-100">{t('window.tabNotice.title')}</p>
        <p className="text-ink-300">{data.note}</p>
      </div>
      {install.available ? (
        <Button size="sm" variant="outline" onClick={() => void promptInstall()}>
          {t('nav.install')}
        </Button>
      ) : null}
      <Button size="sm" variant="ghost" onClick={dismiss}>
        {t('window.tabNotice.dismiss')}
      </Button>
    </div>
  )
}
