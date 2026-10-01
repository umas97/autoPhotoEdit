// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "?" anywhere: the shortcuts of the screen in front of the user, from the
// table in lib/shortcuts.ts. The other screens' are listed after, folded, so
// the list answers "what can I press here" first.
import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { t } from '../i18n/it'
import { SHORTCUTS } from '../lib/shortcuts'
import { Dialog } from './ui/Dialog'

export function ShortcutsHelp() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const screen = pathname.split('/').filter(Boolean).pop() ?? ''

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (event.key === '?' && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault()
        setOpen((value) => !value)
      }
    }
    window.addEventListener('keydown', down)
    return () => window.removeEventListener('keydown', down)
  }, [])

  const here = SHORTCUTS.filter(
    (section) => section.screens.includes('*') || section.screens.includes(screen as never),
  )
  const elsewhere = SHORTCUTS.filter((section) => !here.includes(section))

  const table = (section: (typeof SHORTCUTS)[number]) => (
    <section key={section.label} className="mb-3">
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-300">
        {t(section.label)}
      </h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        {section.items.map((item) => (
          <div key={item.label + item.keys.join()} className="contents">
            <dt className="flex flex-wrap gap-1">
              {item.keys.map((key) => (
                <kbd key={key}
                  className="rounded border border-ink-600 bg-ink-800 px-1.5 py-0.5 font-mono text-xs text-ink-100">
                  {key}
                </kbd>
              ))}
            </dt>
            <dd className="text-ink-200">{t(item.label)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )

  return (
    <Dialog open={open} onOpenChange={setOpen} title={t('shortcuts.title')}
      description={t('shortcuts.description')}>
      <div className="max-h-[60vh] overflow-y-auto pr-1">
        {here.map(table)}
        {elsewhere.length > 0 ? (
          <details className="border-t border-ink-700 pt-2">
            <summary className="cursor-pointer text-xs text-ink-400">{t('shortcuts.otherScreens')}</summary>
            <div className="mt-2">{elsewhere.map(table)}</div>
          </details>
        ) : null}
      </div>
    </Dialog>
  )
}
