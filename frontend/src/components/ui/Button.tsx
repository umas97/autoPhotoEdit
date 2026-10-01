// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
import { cva, type VariantProps } from 'class-variance-authority'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/utils'

const button = cva(
  'inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors ' +
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink-300 ' +
    'disabled:pointer-events-none disabled:opacity-40 whitespace-nowrap',
  {
    variants: {
      variant: {
        primary: 'bg-ink-100 text-ink-900 hover:bg-ink-50',
        secondary: 'bg-ink-600 text-ink-50 hover:bg-ink-500',
        ghost: 'text-ink-200 hover:bg-ink-700 hover:text-ink-50',
        outline: 'border border-ink-500 text-ink-100 hover:bg-ink-700',
        danger: 'bg-bad/80 text-ink-50 hover:bg-bad',
      },
      size: {
        sm: 'h-7 px-2.5 text-xs',
        md: 'h-9 px-3.5 text-sm',
        lg: 'h-10 px-5 text-sm',
        icon: 'h-8 w-8',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  },
)

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {
  children?: ReactNode
}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(button({ variant, size }), className)} {...props} />
}
