// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A window with no address bar has no reload button and no way back (section
// 10), so a crashed interface must not leave a blank rectangle: it shows what
// happened, offers a reload, and offers the diagnostics of section 19 -- which
// here is a file the user can attach to a report, assembled in the browser
// because a crashed frontend cannot be sure the server is the healthy one.
import { Component, type ErrorInfo, type ReactNode } from 'react'
import { t } from '../i18n/it'
import { Button } from './ui/Button'

interface State {
  error: Error | null
  stack: string | null
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, stack: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ stack: info.componentStack ?? null })
    console.error('autoPhotoEdit:', error, info)
  }

  private exportDiagnostics = async (): Promise<void> => {
    let health: unknown = null
    try {
      health = await (await fetch('/api/health')).json()
    } catch {
      health = { status: 'irraggiungibile' }
    }
    const bundle = {
      generato: new Date().toISOString(),
      errore: this.state.error?.message ?? null,
      stack: this.state.error?.stack ?? null,
      componenti: this.state.stack,
      url: location.href,
      userAgent: navigator.userAgent,
      display: window.matchMedia('(display-mode: standalone)').matches ? 'standalone' : 'browser',
      server: health,
    }
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' }),
    )
    const link = document.createElement('a')
    link.href = url
    link.download = `autophotoedit-diagnostica-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children
    return (
      <div className="grid min-h-full place-items-center bg-ink-900 p-8">
        <div className="w-full max-w-xl">
          <h1 className="text-lg font-semibold text-ink-50">{t('error.title')}</h1>
          <p className="mt-2 text-sm leading-relaxed text-ink-200">{t('error.body')}</p>
          <div className="mt-5 flex gap-2">
            <Button variant="primary" onClick={() => location.reload()}>
              {t('error.reload')}
            </Button>
            <Button variant="outline" onClick={this.exportDiagnostics}>
              {t('error.diagnostics')}
            </Button>
          </div>
          <details className="mt-6 rounded-md border border-ink-700 bg-ink-850 p-3">
            <summary className="cursor-pointer text-xs text-ink-300">
              {t('error.details')}
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto text-xs text-ink-300">
              {this.state.error.message}
              {'\n'}
              {this.state.error.stack}
            </pre>
          </details>
        </div>
      </div>
    )
  }
}
