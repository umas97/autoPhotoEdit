// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The controls of section 7.4, and the formula behind them in plain sight.
//
// Nothing here computes a selection. Every control sends its value to the
// server, which answers with the photos whose outcome changed; that is what
// keeps a move of the aggressiveness slider under the 100 ms of section 14
// without a second, TypeScript copy of the rules that could disagree with the
// first.
//
// The two criteria that need a model are shown even when they cannot run --
// disabled, with the reason and the cost -- because a missing option the user
// was never told about is indistinguishable from a feature that does not
// exist. The aesthetic one carries its licence warning whether it is
// available or not (section 7.3).
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { CullCriterion, CullSettings, CullView, ScoredCriterion } from '../../lib/cullTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn, formatNumber } from '../../lib/utils'
import { Checkbox } from '../ui/Field'
import { Slider } from '../ui/Slider'

const CRITERIA: CullCriterion[] = ['sharpness', 'motion', 'exposure', 'burst', 'faces', 'aesthetic']
const SCORED: ScoredCriterion[] = ['sharpness', 'motion', 'exposure', 'faces', 'aesthetic']
const MODEL: Array<'faces' | 'aesthetic'> = ['faces', 'aesthetic']

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-b border-ink-700 px-3 py-2.5">
      <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-ink-300">{title}</h3>
      {children}
    </section>
  )
}

export function CullingPanel({
  settings,
  availability,
  summary,
  latencyMs,
  onChange,
}: {
  settings: CullSettings
  availability: CullView['availability']
  summary: CullView['summary']
  latencyMs: number | null
  onChange: (patch: Partial<Omit<CullSettings, 'threshold'>>) => void
}) {
  // The slider moves locally at once and asks the server on every step; the
  // number shown is the one under the hand, not the last one confirmed.
  const [aggressiveness, setAggressiveness] = useState(settings.aggressiveness)
  const [target, setTarget] = useState(settings.target ?? 30)
  // Answers to earlier steps of a drag arrive while the hand is still moving;
  // following them would pull the handle back under the cursor.
  const dragging = useRef(false)
  useEffect(() => {
    if (!dragging.current) setAggressiveness(settings.aggressiveness)
  }, [settings.aggressiveness])

  const setMode = (mode: CullSettings['mode']) => {
    if (mode === 'conservative') onChange({ mode })
    else {
      const value = mode === 'target_percent' ? Math.min(target, 100) : Math.max(1, target)
      setTarget(value)
      onChange({ mode, target: value })
    }
  }

  const shortfall =
    summary.target_count !== null && summary.selected < summary.target_count
      ? summary.target_count - summary.selected
      : 0

  return (
    <div className="h-full overflow-y-auto text-sm">
      <Section title={t('culling.panel.mode')}>
        {(['conservative', 'target_percent', 'target_count'] as const).map((mode) => (
          <label key={mode} className="flex cursor-pointer items-start gap-2 rounded p-1 hover:bg-ink-800">
            <input
              type="radio"
              name="culling-mode"
              className="mt-0.5 accent-ink-200"
              checked={settings.mode === mode}
              onChange={() => setMode(mode)}
            />
            <span>
              <span className="block text-ink-100">{t(`culling.mode.${mode}` as StringKey)}</span>
              <span className="block text-xs text-ink-400">
                {t(`culling.mode.${mode}.hint` as StringKey)}
              </span>
            </span>
          </label>
        ))}
        {settings.mode !== 'conservative' ? (
          <div className="mt-1.5 flex items-center gap-2 pl-6 text-xs text-ink-200">
            <input
              type="number"
              min={1}
              max={settings.mode === 'target_percent' ? 100 : summary.total}
              value={target}
              onChange={(event) => setTarget(Number(event.target.value))}
              onBlur={() => onChange({ target })}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onChange({ target })
              }}
              className="h-7 w-20 rounded border border-ink-600 bg-ink-800 px-2 tabular-nums text-ink-50"
            />
            <span>
              {settings.mode === 'target_percent'
                ? t('culling.target.percent', { count: summary.target_count ?? 0 })
                : t('culling.target.count')}
            </span>
          </div>
        ) : null}
        {shortfall > 0 ? (
          <p className="mt-1.5 pl-6 text-xs text-warn">
            {t('culling.target.shortfall', { count: shortfall })}
          </p>
        ) : null}
      </Section>

      <Section title={t('culling.panel.aggressiveness')}>
        <Slider
          label={t('culling.aggressiveness')}
          value={aggressiveness}
          min={0}
          max={1}
          step={0.05}
          neutral={0.5}
          digits={2}
          onChange={(value) => {
            dragging.current = true
            setAggressiveness(value)
            onChange({ aggressiveness: value })
          }}
          onCommit={(value) => {
            dragging.current = false
            onChange({ aggressiveness: value })
          }}
        />
        <p className="text-xs text-ink-400">
          {t('culling.threshold', { value: Math.round(settings.threshold * 100) })}
          {latencyMs !== null ? ` · ${t('culling.latency', { ms: latencyMs })}` : ''}
        </p>
      </Section>

      <Section title={t('culling.panel.criteria')}>
        {CRITERIA.map((criterion) => {
          const model = MODEL.includes(criterion as 'faces' | 'aesthetic')
            ? availability[criterion as 'faces' | 'aesthetic']
            : null
          const hint = [t(`culling.criterion.${criterion}.hint` as StringKey)]
          if (model && !model.available && model.reason) {
            hint.push(t(`culling.unavailable.${model.reason}` as StringKey))
          }
          if (model?.notice) hint.push(t(`culling.notice.${model.notice}` as StringKey))
          return (
            <Checkbox
              key={criterion}
              checked={settings.criteria[criterion]}
              disabled={model !== null && !model.available}
              onChange={(checked) => onChange({ criteria: { ...settings.criteria, [criterion]: checked } })}
              label={t(`culling.criterion.${criterion}` as StringKey)}
              hint={hint.join(' ')}
            />
          )
        })}
      </Section>

      <Section title={t('culling.panel.weights')}>
        <p className="mb-1 text-xs text-ink-400">{t('culling.weights.formula')}</p>
        {SCORED.filter((name) => settings.criteria[name]).map((name) => (
          <WeightSlider
            key={name}
            name={name}
            value={settings.weights[name]}
            onCommit={(value) => onChange({ weights: { ...settings.weights, [name]: value } })}
          />
        ))}
      </Section>

      <p className={cn('px-3 py-2 text-xs text-ink-400')}>
        {t('culling.panel.footnote', {
          analysed: summary.analysed,
          total: summary.total,
          failed: summary.failed,
        })}
        {' '}
        {formatNumber(summary.total ? (summary.selected / summary.total) * 100 : 0, 0, '%')}
      </p>
    </div>
  )
}

/**
 * A weight moves under the hand and is sent when released: a change of weight
 * re-scores every photo, and two thousand new scores per pixel of drag is
 * traffic nobody can see the difference of.
 */
function WeightSlider({
  name,
  value,
  onCommit,
}: {
  name: ScoredCriterion
  value: number
  onCommit: (value: number) => void
}) {
  const [local, setLocal] = useState(value)
  useEffect(() => setLocal(value), [value])
  return (
    <Slider
      label={t(`culling.criterion.${name}` as StringKey)}
      value={local}
      min={0}
      max={1}
      step={0.05}
      neutral={DEFAULT_WEIGHTS[name]}
      digits={2}
      onChange={setLocal}
      onCommit={onCommit}
    />
  )
}

/** backend/ape/culling/select.py, DEFAULT_WEIGHTS: where a double click returns. */
const DEFAULT_WEIGHTS: Record<ScoredCriterion, number> = {
  sharpness: 0.4,
  motion: 0.25,
  exposure: 0.35,
  faces: 0.3,
  aesthetic: 0.2,
}
