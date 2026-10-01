// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The controls of *what* a mask selects, which depend on its kind: the
// feathering of an ellipse, the brush, the ranges of a parametric selection,
// the subject of a segmentation -- and for every shape, the two refinements
// section 6.3 asks for most: "only these tones", "only this colour".
import { Brush, Eraser } from 'lucide-react'
import { t, type StringKey } from '../../i18n/it'
import {
  SELECTION_CONTROLS,
  limitStep,
  orderedRange,
  withDefinition,
  withLimit,
  withLimitRange,
  type Limit,
} from '../../lib/masks'
import type { HueRange, MaskParams, ValueRange } from '../../lib/maskTypes'
import type { Control } from '../../lib/controls'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Checkbox } from '../ui/Field'
import { Slider } from '../ui/Slider'
import { BRUSH_SIZE, type MaskEditor } from './useMaskEditor'

type Change = (mask: MaskParams, commit: boolean) => void

export function TableSlider({
  control,
  value,
  onChange,
  disabled,
}: {
  control: Control
  value: number
  onChange: (value: number, commit: boolean) => void
  disabled?: boolean
}) {
  return (
    <Slider label={t(control.label)} value={value} min={control.min} max={control.max}
      step={control.step} neutral={control.neutral} digits={control.digits} unit={control.unit}
      disabled={disabled} onChange={(next) => onChange(next, false)}
      onCommit={(next) => onChange(next, true)} />
  )
}

function Range({
  range,
  onChange,
  disabled,
}: {
  range: ValueRange
  onChange: (range: ValueRange, commit: boolean) => void
  disabled?: boolean
}) {
  const set = (field: 'low' | 'high' | 'feather') => (value: number, commit: boolean) =>
    onChange(orderedRange(range, field, value), commit)
  return (
    <>
      <TableSlider control={SELECTION_CONTROLS.low} value={range.low} onChange={set('low')} disabled={disabled} />
      <TableSlider control={SELECTION_CONTROLS.high} value={range.high} onChange={set('high')} disabled={disabled} />
      <TableSlider control={SELECTION_CONTROLS.rangeFeather} value={range.feather}
        onChange={set('feather')} disabled={disabled} />
    </>
  )
}

function Hues({
  range,
  onChange,
  disabled,
}: {
  range: HueRange
  onChange: (range: HueRange, commit: boolean) => void
  disabled?: boolean
}) {
  const set = (field: keyof HueRange) => (value: number, commit: boolean) =>
    onChange({ ...range, [field]: value }, commit)
  return (
    <>
      {/* The wheel of the HSL bands, so the slider shows which hue it is on. */}
      <div className="mt-1 h-1.5 rounded-full"
        style={{ background: 'linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00)' }} />
      <TableSlider control={SELECTION_CONTROLS.center} value={range.center} onChange={set('center')} disabled={disabled} />
      <TableSlider control={SELECTION_CONTROLS.width} value={range.width} onChange={set('width')} disabled={disabled} />
      <TableSlider control={SELECTION_CONTROLS.hueFeather} value={range.feather}
        onChange={set('feather')} disabled={disabled} />
    </>
  )
}

const DEFAULT_RANGE: ValueRange = { low: 0, high: 1, feather: 0.1 }
const DEFAULT_HUE: HueRange = { center: 220, width: 60, feather: 20 }

/** The ranges of a parametric mask: each one on or off, the ones on multiplied. */
function Parametric({ mask, onChange, disabled }: { mask: MaskParams; onChange: Change; disabled?: boolean }) {
  const d = mask.definition
  const toggle = (field: 'luminance' | 'saturation' | 'hue', on: boolean) =>
    onChange(withDefinition(mask, { [field]: on ? (field === 'hue' ? DEFAULT_HUE : DEFAULT_RANGE) : null }), true)
  return (
    <>
      {(['luminance', 'saturation'] as const).map((field) => (
        <div key={field}>
          <Checkbox checked={Boolean(d[field])} disabled={disabled}
            label={t(`masks.range.${field}` as StringKey)} onChange={(on) => toggle(field, on)} />
          {d[field] ? (
            <Range range={d[field]!} disabled={disabled}
              onChange={(range, commit) => onChange(withDefinition(mask, { [field]: range }), commit)} />
          ) : null}
        </div>
      ))}
      <Checkbox checked={Boolean(d.hue)} disabled={disabled} label={t('masks.range.hue')}
        onChange={(on) => toggle('hue', on)} />
      {d.hue ? (
        <Hues range={d.hue} disabled={disabled}
          onChange={(range, commit) => onChange(withDefinition(mask, { hue: range }), commit)} />
      ) : null}
      {!d.luminance && !d.saturation && !d.hue ? (
        <p className="px-1.5 text-xs text-ink-400">{t('masks.range.none')}</p>
      ) : null}
    </>
  )
}

/** "Only these tones" / "only this colour" on a shape: an intersected parametric step. */
function Limits({ mask, onChange, disabled }: { mask: MaskParams; onChange: Change; disabled?: boolean }) {
  const combine = mask.definition.combine ?? []
  const row = (limit: Limit) => {
    const at = limitStep(mask, limit)
    const step = at >= 0 ? combine[at] : null
    return (
      <div key={limit}>
        <Checkbox checked={step !== null} disabled={disabled}
          label={t(`masks.limit.${limit}` as StringKey)} onChange={(on) => onChange(withLimit(mask, limit, on), true)} />
        {step && limit === 'luminance' ? (
          <Range range={step.definition.luminance!} disabled={disabled}
            onChange={(range, commit) => {
              let next = mask
              for (const field of ['low', 'high', 'feather'] as const)
                next = withLimitRange(next, limit, field, range[field])
              onChange(next, commit)
            }} />
        ) : null}
        {step && limit === 'hue' ? (
          <Hues range={step.definition.hue!} disabled={disabled}
            onChange={(range, commit) => {
              let next = mask
              for (const field of ['center', 'width', 'feather'] as const)
                next = withLimitRange(next, limit, field, range[field])
              onChange(next, commit)
            }} />
        ) : null}
      </div>
    )
  }
  return <div className="mt-1 border-t border-ink-800 pt-1">{row('luminance')}{row('hue')}</div>
}

function BrushControls({ editor }: { editor: MaskEditor }) {
  const { brush, setBrush } = editor
  return (
    <>
      <div className="flex gap-1 py-1">
        <Button size="sm" variant={brush.erase ? 'ghost' : 'secondary'} onClick={() => setBrush({ erase: false })}>
          <Brush size={13} />{t('masks.brush.paint')}
        </Button>
        <Button size="sm" variant={brush.erase ? 'secondary' : 'ghost'} onClick={() => setBrush({ erase: true })}>
          <Eraser size={13} />{t('masks.brush.erase')}
        </Button>
      </div>
      <Slider label={t('masks.brush.size')} value={brush.size} min={BRUSH_SIZE.min} max={BRUSH_SIZE.max}
        step={0.001} neutral={0.03} digits={3} onChange={(size) => setBrush({ size })}
        onCommit={(size) => setBrush({ size })} />
      <Slider label={t('masks.brush.softness')} value={brush.softness} min={0} max={1} step={0.01}
        neutral={0.6} onChange={(softness) => setBrush({ softness })} onCommit={(softness) => setBrush({ softness })} />
      <Slider label={t('masks.brush.flow')} value={brush.flow} min={0.05} max={1} step={0.01}
        neutral={0.7} onChange={(flow) => setBrush({ flow })} onCommit={(flow) => setBrush({ flow })} />
      <p className={cn('px-1.5 text-xs', editor.busy ? 'text-ink-200' : 'text-ink-400')}>
        {t(editor.busy ? 'masks.brush.saving' : 'masks.brush.hint')}
      </p>
    </>
  )
}

export function MaskSelection({
  mask,
  editor,
  onChange,
  disabled,
}: {
  mask: MaskParams
  editor: MaskEditor
  onChange: Change
  disabled?: boolean
}) {
  return (
    <div className="pb-1">
      <p className="px-1.5 pb-1 text-xs text-ink-400">{t(`masks.kind.${mask.kind}.hint` as StringKey)}</p>
      {mask.kind === 'radial' ? (
        <TableSlider control={SELECTION_CONTROLS.feather} value={mask.definition.feather ?? 0.5}
          disabled={disabled} onChange={(feather, commit) => onChange(withDefinition(mask, { feather }), commit)} />
      ) : null}
      {mask.kind === 'brush' ? <BrushControls editor={editor} /> : null}
      {mask.kind === 'parametric' ? <Parametric mask={mask} onChange={onChange} disabled={disabled} /> : null}
      {mask.kind !== 'parametric' ? <Limits mask={mask} onChange={onChange} disabled={disabled} /> : null}
    </div>
  )
}
