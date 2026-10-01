// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The control section 10 is measured on.
//
// It reports two different things, and the difference is the whole point:
// `onChange` fires continuously while the handle moves and drives the 1024 px
// preview; `onCommit` fires once when it is released and asks for 2048 px. A
// slider that only reported the final value would feel dead; one that asked for
// 2048 px on every frame would be slower than the eye tolerates.
//
// Double-click returns the parameter to its neutral value -- the one place
// where "what does this do by default" is answerable without a manual.
import * as RadixSlider from '@radix-ui/react-slider'
import { cn, formatNumber } from '../../lib/utils'

export interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  neutral: number
  digits?: number
  unit?: string
  disabled?: boolean
  onChange: (value: number) => void
  onCommit: (value: number) => void
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  neutral,
  digits = 2,
  unit = '',
  disabled,
  onChange,
  onCommit,
}: SliderProps) {
  const modified = Math.abs(value - neutral) > step / 2
  return (
    <div className={cn('py-1.5', disabled && 'opacity-40')}>
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className={cn('text-ink-200', modified && 'text-ink-50')}>{label}</span>
        <span
          className={cn(
            'tabular-nums text-ink-300',
            modified && 'text-ink-100',
          )}
        >
          {formatNumber(value, digits, unit)}
        </span>
      </div>
      <RadixSlider.Root
        className="relative mt-1.5 flex h-4 w-full touch-none items-center select-none"
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([next]) => onChange(next)}
        onValueCommit={([next]) => onCommit(next)}
        onDoubleClick={() => {
          onChange(neutral)
          onCommit(neutral)
        }}
        aria-label={label}
      >
        <RadixSlider.Track className="relative h-[3px] w-full rounded-full bg-ink-600">
          <RadixSlider.Range className="absolute h-full rounded-full bg-ink-400" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          className={cn(
            'block h-3.5 w-3.5 rounded-full border border-ink-400 bg-ink-100 shadow-sm',
            'transition-colors hover:bg-ink-50 focus-visible:outline-none',
            'focus-visible:ring-2 focus-visible:ring-ink-300',
          )}
        />
      </RadixSlider.Root>
    </div>
  )
}
