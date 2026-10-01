// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The frame every screen sits in, and the part of section 10 that is about the
// window rather than about photographs:
//
//   * every screen has a visible way back, because there is no Back button;
//   * Alt+Left and Alt+Right are handled here, not left to the browser, so that
//     history works the same in the dedicated window and in a tab;
//   * "Esci dall'applicazione" is in the interface, because the way to quit a
//     program should not be to find the terminal that started it.
//
// The main menu (Progetti, Stili, Problemi, Impostazioni, Informazioni) stays
// on every screen outside a project: those screens pass no `back`. A project's
// screens pass one and get their own route back instead.
import { useEffect, useState, type ReactNode } from 'react'
import { useNavigate, NavLink, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Info, LogOut, Power } from 'lucide-react'
import { api } from '../lib/api'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { Button } from './ui/Button'
import { Dialog } from './ui/Dialog'
import { WindowBanner } from './WindowBanner'
import { CloseGuard } from './CloseGuard'
import { JobBar } from './JobBar'
import { AboutDialog } from './AboutDialog'

export function AppShell({
  children,
  back,
  title,
  actions,
}: {
  children: ReactNode
  /** Where "indietro" goes on this screen. Absent on the root. */
  back?: { to: string; label: string }
  title?: string
  actions?: ReactNode
}) {
  const navigate = useNavigate()
  const location = useLocation()
  const [quitting, setQuitting] = useState(false)
  const [askQuit, setAskQuit] = useState(false)
  const [showAbout, setShowAbout] = useState(false)
  const { data: window_ } = useQuery({ queryKey: ['window'], queryFn: api.window })

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!event.altKey) return
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        navigate(-1)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        navigate(1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate])

  const quit = async () => {
    setQuitting(true)
    try {
      await api.quit()
      // The server closes the window itself; this is for the tab case, where
      // there is no window of ours to close.
      setTimeout(() => window.close(), 500)
    } catch {
      setQuitting(false)
    }
  }

  return (
    <div className="flex h-full flex-col bg-ink-900">
      <WindowBanner />
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-ink-700 px-3">
        {back ? (
          <Button variant="ghost" size="sm" onClick={() => navigate(back.to)}>
            <ArrowLeft size={14} />
            {back.label}
          </Button>
        ) : (
          <span className="px-1.5 text-sm font-semibold text-ink-50">{t('app.name')}</span>
        )}

        {/* The route back is already the button on the left; a second way to the
            same screen is clutter in a window this dense (section 10). */}
        <nav className={cn('flex items-center gap-1', back && 'hidden')}>
          <NavLink
            to="/"
            className={({ isActive }) =>
              cn(
                'rounded-md px-2.5 py-1 text-sm',
                isActive || location.pathname === '/'
                  ? 'bg-ink-700 text-ink-50'
                  : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100',
              )
            }
          >
            {t('nav.projects')}
          </NavLink>
          <NavLink
            to="/stili"
            className={({ isActive }) =>
              cn(
                'rounded-md px-2.5 py-1 text-sm',
                isActive ? 'bg-ink-700 text-ink-50' : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100',
              )
            }
          >
            {t('nav.styles')}
          </NavLink>
          <NavLink
            to="/problemi"
            className={({ isActive }) =>
              cn(
                'rounded-md px-2.5 py-1 text-sm',
                isActive ? 'bg-ink-700 text-ink-50' : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100',
              )
            }
          >
            {t('nav.problems')}
          </NavLink>
          <NavLink
            to="/impostazioni"
            className={({ isActive }) =>
              cn(
                'rounded-md px-2.5 py-1 text-sm',
                isActive ? 'bg-ink-700 text-ink-50' : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100',
              )
            }
          >
            {t('nav.settings')}
          </NavLink>
        </nav>

        {title ? (
          <span className="truncate text-sm text-ink-200" title={title}>
            {title}
          </span>
        ) : null}

        <div className="ml-auto flex items-center gap-2">
          {actions}
          <JobBar />
          {back ? null : (
            <Button
              variant="ghost"
              size="icon"
              title={t('about.open')}
              aria-label={t('about.open')}
              onClick={() => setShowAbout(true)}
            >
              <Info size={15} />
            </Button>
          )}
          {window_?.can_quit ? (
            <Button
              variant="ghost"
              size="icon"
              title={t('nav.quit')}
              aria-label={t('nav.quit')}
              onClick={() => setAskQuit(true)}
            >
              <Power size={15} />
            </Button>
          ) : null}
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-hidden">{children}</main>

      <CloseGuard />
      <AboutDialog open={showAbout} onOpenChange={setShowAbout} />
      <Dialog
        open={askQuit}
        onOpenChange={setAskQuit}
        title={t('quit.title')}
        description={t('quit.body')}
        footer={
          <>
            <Button variant="ghost" onClick={() => setAskQuit(false)}>
              {t('common.cancel')}
            </Button>
            <Button variant="danger" disabled={quitting} onClick={() => void quit()}>
              <LogOut size={14} />
              {quitting ? t('nav.quitting') : t('quit.confirm')}
            </Button>
          </>
        }
      />
    </div>
  )
}
