// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The right-hand panel of section 10, built from the table in lib/params.ts.
//
// It reports a change twice: once while the handle moves and once when it is
// released. The viewer uses the first for the 1024 px preview and the second
// both for the 2048 px render and for the undo stack -- so an undo steps back
// one *gesture*, not one animation frame.
import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { GROUPS, HSL_BANDS } from '../lib/controls'
import { readBand, readPath, withBand, withPath } from '../lib/params'
import type { EditParams, HSLBandName } from '../lib/types'
import { t, type StringKey } from '../i18n/it'
import { useUi } from '../lib/store'
import { cn } from '../lib/utils'
import { Slider } from './ui/Slider'
import { Checkbox } from './ui/Field'

export interface ParamPanelProps {
  params: EditParams
  /** `commit` is true when the gesture ended: render big, and record an undo. */
  onChange: (next: EditParams, commit: boolean) => void
  disabled?: boolean
  /** Rendered above the groups: the geometry section, which is not a table row. */
  header?: ReactNode
}

export function ParamPanel({ params, onChange, disabled, header }: ParamPanelProps) {
  const collapsed = useUi((state) => state.collapsedGroups)
  const toggle = useUi((state) => state.toggleGroup)
  const asShot = params.white_balance.mode === 'as_shot'

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-ink-850 px-3 py-2">
      <h2 className="sticky top-0 z-10 -mx-3 mb-1 bg-ink-850 px-3 pb-2 pt-1 text-xs font-semibold uppercase tracking-wide text-ink-300">
        {t('params.title')}
      </h2>
      {header}

      {GROUPS.map((group) => {
        const isOpen = group.open ? !collapsed.includes(group.key) : collapsed.includes(group.key)
        return (
          <section key={group.key} className="border-b border-ink-800 py-1.5 last:border-b-0">
            <button
              type="button"
              className="flex w-full items-center justify-between py-1 text-left text-xs font-medium text-ink-100"
              onClick={() => toggle(group.key)}
              aria-expanded={isOpen}
            >
              {t(group.label)}
              <ChevronDown
                size={14}
                className={cn('text-ink-400 transition-transform', isOpen && 'rotate-180')}
              />
            </button>

            {isOpen ? (
              <div className="pb-1">
                {group.key === 'white_balance' ? (
                  <Checkbox
                    checked={asShot}
                    disabled={disabled}
                    label={t('params.white_balance.mode')}
                    onChange={(checked) => {
                      const next = structuredClone(params)
                      next.white_balance.mode = checked ? 'as_shot' : 'custom'
                      onChange(next, true)
                    }}
                  />
                ) : null}

                {group.controls.map((control) => (
                  <Slider
                    key={control.path}
                    label={t(control.label)}
                    value={readPath(params, control.path)}
                    min={control.min}
                    max={control.max}
                    step={control.step}
                    neutral={control.neutral}
                    digits={control.digits}
                    unit={control.unit}
                    disabled={disabled || (group.key === 'white_balance' && asShot)}
                    onChange={(value) => onChange(withPath(params, control.path, value), false)}
                    onCommit={(value) => onChange(withPath(params, control.path, value), true)}
                  />
                ))}

                {group.key === 'color' ? (
                  <HslBands params={params} onChange={onChange} disabled={disabled} />
                ) : null}
              </div>
            ) : null}
          </section>
        )
      })}
    </div>
  )
}

function HslBands({ params, onChange, disabled }: ParamPanelProps) {
  const fields: Array<{ key: 'hue' | 'saturation' | 'luminance'; label: StringKey }> = [
    { key: 'hue', label: 'params.hsl.hue' },
    { key: 'saturation', label: 'params.hsl.saturation' },
    { key: 'luminance', label: 'params.hsl.luminance' },
  ]
  return (
    <details className="mt-2 rounded-md border border-ink-800 px-2 py-1">
      <summary className="cursor-pointer py-1 text-xs font-medium text-ink-200">
        {t('params.group.hsl')}
      </summary>
      {HSL_BANDS.map((band: HSLBandName) => (
        <div key={band} className="border-t border-ink-800 py-1 first:border-t-0">
          <p className="pt-1 text-[11px] uppercase tracking-wide text-ink-400">
            {t(`params.hsl.band.${band}` as StringKey)}
          </p>
          {fields.map((field) => (
            <Slider
              key={field.key}
              label={t(field.label)}
              value={readBand(params, band, field.key)}
              min={-1}
              max={1}
              step={0.01}
              neutral={0}
              digits={2}
              disabled={disabled}
              onChange={(value) => onChange(withBand(params, band, field.key, value), false)}
              onCommit={(value) => onChange(withBand(params, band, field.key, value), true)}
            />
          ))}
        </div>
      ))}
    </details>
  )
}
