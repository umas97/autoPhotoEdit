// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The corner where lib/toast.ts messages appear. Polite for information,
// assertive for errors, so a screen reader says a failure at once.
import { X } from 'lucide-react'
import { useToasts } from '../lib/toast'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'

export function Toasts() {
  const toasts = useToasts((state) => state.toasts)
  const dismiss = useToasts((state) => state.dismiss)
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2">
      {toasts.map((item) => (
        <div
          key={item.id}
          role={item.tone === 'error' ? 'alert' : 'status'}
          className={cn(
            'pointer-events-auto flex items-start gap-2 rounded-md border px-3 py-2 text-sm shadow-lg',
            item.tone === 'error'
              ? 'border-bad/60 bg-ink-900 text-ink-50'
              : 'border-ink-600 bg-ink-900 text-ink-100',
          )}
        >
          <span className="min-w-0 flex-1 break-words">{item.message}</span>
          <button
            type="button"
            className="shrink-0 text-ink-400 hover:text-ink-100"
            aria-label={t('common.close')}
            onClick={() => dismiss(item.id)}
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  )
}
