// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A dialog of the application, never the browser's own (section 21.3): in a
// window with no chrome, a `confirm()` box looks like something that escaped.
import * as RadixDialog from '@radix-ui/react-dialog'
import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

export interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  children?: ReactNode
  footer?: ReactNode
  /** A closing dialog has no escape hatch: every route out is a decision. */
  dismissible?: boolean
}

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  dismissible = true,
}: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-ink-950/70 backdrop-blur-[1px]" />
        <RadixDialog.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-50 w-[min(32rem,calc(100vw-2rem))]',
            '-translate-x-1/2 -translate-y-1/2 rounded-lg border border-ink-600',
            'bg-ink-850 p-5 shadow-2xl',
          )}
          onEscapeKeyDown={(event) => {
            if (!dismissible) event.preventDefault()
          }}
          onInteractOutside={(event) => {
            if (!dismissible) event.preventDefault()
          }}
        >
          <RadixDialog.Title className="text-base font-semibold text-ink-50">
            {title}
          </RadixDialog.Title>
          {description ? (
            <RadixDialog.Description className="mt-1.5 text-sm text-ink-200">
              {description}
            </RadixDialog.Description>
          ) : null}
          {children ? <div className="mt-4">{children}</div> : null}
          {footer ? <div className="mt-5 flex justify-end gap-2">{footer}</div> : null}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
