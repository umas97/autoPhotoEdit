// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Lens correction, straightening and crop (section 6.4), at the top of the
// parameter panel.
//
// Each control says *why* it is where it is. The lens correction names the
// profile it uses -- or says there is none and offers to associate one, which
// is the fix section 6.4 asks for. The rotation says what the analysis found:
// levelled by so much, already level, or left alone and why. A photographer
// who disagrees with an automatic choice should be able to see it was one.
import { ChevronDown, Crop, RotateCcw } from 'lucide-react'
import type { EditParams } from '../lib/types'
import type { PhotoAnalysis } from '../lib/analysisTypes'
import { t } from '../i18n/it'
import { useUi } from '../lib/store'
import { cn } from '../lib/utils'
import { Button } from './ui/Button'
import { Checkbox } from './ui/Field'
import { Slider } from './ui/Slider'

export interface GeometryPanelProps {
  params: EditParams
  analysis: PhotoAnalysis | null
  onChange: (next: EditParams, commit: boolean) => void
  onAssociateLens: () => void
  disabled?: boolean
}

/** The slider covers the auto-straightening window with room to spare. */
const ROTATION_RANGE = 10

function lensLine(analysis: PhotoAnalysis | null): string {
  if (!analysis || analysis.lens === undefined) return t('geometry.lensPending')
  if (analysis.lens === null) return t('geometry.lensMissing')
  if (analysis.lens.source === 'applied') return t('geometry.lensApplied')
  return analysis.lens.source === 'override'
    ? t('geometry.lensProfileManual', { model: analysis.lens.model })
    : t('geometry.lensProfile', { model: analysis.lens.model })
}

function straightenLine(analysis: PhotoAnalysis | null): string | null {
  const found = analysis?.straighten
  if (!found) return null
  const degrees = Math.abs(found.measured_deg ?? found.rotation_deg).toFixed(2)
  switch (found.outcome) {
    case 'rotated':
      return t('geometry.straighten.rotated', { deg: Math.abs(found.rotation_deg).toFixed(2) })
    case 'level':
      return t('geometry.straighten.level')
    case 'too_large':
      return t('geometry.straighten.too_large', { deg: degrees })
    case 'contradictory':
      return t('geometry.straighten.contradictory')
    default:
      return t('geometry.straighten.no_lines')
  }
}

export function GeometryPanel({
  params,
  analysis,
  onChange,
  onAssociateLens,
  disabled,
}: GeometryPanelProps) {
  const collapsed = useUi((state) => state.collapsedGroups)
  const toggle = useUi((state) => state.toggleGroup)
  const open = !collapsed.includes('geometry')
  const geometry = params.geometry
  const automatic = analysis?.straighten?.rotation_deg ?? 0
  const crop = geometry.crop

  const set = (change: Partial<EditParams['geometry']>, commit: boolean) => {
    const next = structuredClone(params)
    next.geometry = { ...next.geometry, ...change }
    onChange(next, commit)
  }

  return (
    <section className="border-b border-ink-800 py-1.5">
      <button
        type="button"
        className="flex w-full items-center justify-between py-1 text-left text-xs font-medium text-ink-100"
        onClick={() => toggle('geometry')}
        aria-expanded={open}
      >
        {t('geometry.title')}
        <ChevronDown
          size={14}
          className={cn('text-ink-400 transition-transform', open && 'rotate-180')}
        />
      </button>
      {open ? (
        <div className="space-y-1 pb-1">
          <Checkbox
            checked={geometry.lens_correction}
            disabled={disabled}
            label={t('geometry.lens')}
            hint={lensLine(analysis)}
            onChange={(checked) => set({ lens_correction: checked }, true)}
          />
          {analysis?.lens === null ? (
            <Button size="sm" variant="ghost" onClick={onAssociateLens}>
              {t('geometry.lensAssociate')}
            </Button>
          ) : null}

          <Slider
            label={t('geometry.rotation')}
            value={geometry.rotation_deg}
            min={-ROTATION_RANGE}
            max={ROTATION_RANGE}
            step={0.05}
            neutral={0}
            digits={2}
            unit="°"
            disabled={disabled}
            onChange={(value) => set({ rotation_deg: value }, false)}
            onCommit={(value) => set({ rotation_deg: value }, true)}
          />
          {straightenLine(analysis) ? (
            <p className="px-1.5 text-xs text-ink-400">{straightenLine(analysis)}</p>
          ) : null}
          {Math.abs(geometry.rotation_deg - automatic) > 1e-3 && analysis?.straighten ? (
            <Button
              size="sm"
              variant="ghost"
              disabled={disabled}
              onClick={() => set({ rotation_deg: automatic }, true)}
            >
              <RotateCcw size={12} />
              {t('geometry.straighten.restore')}
            </Button>
          ) : null}

          <div className="flex items-center justify-between gap-2 px-1.5 pt-1 text-xs text-ink-300">
            <span className="flex items-center gap-1.5">
              <Crop size={12} />
              {crop
                ? t('geometry.cropSet', {
                    w: Math.round(crop.width * 100),
                    h: Math.round(crop.height * 100),
                  })
                : t('geometry.cropNone')}
            </span>
            {crop ? (
              <Button size="sm" variant="ghost" disabled={disabled} onClick={() => set({ crop: null }, true)}>
                {t('geometry.cropClear')}
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  )
}
