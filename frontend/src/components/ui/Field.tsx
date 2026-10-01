// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/utils'

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-ink-200">{label}</span>
      <div className="mt-1">{children}</div>
      {hint ? <p className="mt-1 text-xs text-ink-400">{hint}</p> : null}
    </label>
  )
}

export function TextInput({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'h-9 w-full rounded-md border border-ink-600 bg-ink-800 px-2.5 text-sm text-ink-50',
        'placeholder:text-ink-400 focus:border-ink-400 focus:outline-none',
        className,
      )}
      {...props}
    />
  )
}

export function Checkbox({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  hint?: string
  disabled?: boolean
}) {
  return (
    <label
      className={cn(
        'flex cursor-pointer items-start gap-2.5 rounded-md p-1.5',
        disabled ? 'cursor-not-allowed opacity-50' : 'hover:bg-ink-800',
      )}
    >
      <input
        type="checkbox"
        className="mt-0.5 h-4 w-4 accent-ink-200"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        <span className="block text-sm text-ink-100">{label}</span>
        {hint ? <span className="block text-xs text-ink-400">{hint}</span> : null}
      </span>
    </label>
  )
}
