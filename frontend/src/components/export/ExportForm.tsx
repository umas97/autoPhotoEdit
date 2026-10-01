// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The options of the Export screen (section 10.6). Every change is saved on
// the project at once and answered with the new plan, so the count, the name
// preview and the collisions on the right are always those of what is shown
// here. Text fields are saved when they lose focus (or on Enter), not on every
// keystroke: a folder path half typed is not a folder worth asking about.
//
// The file-name template has its own live preview (section 16.1), asked for
// while typing -- a cheap request that renders one name -- because a template
// is only understood by seeing what it does to a real file name.
import { useEffect, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { exportApi } from '../../lib/exportApi'
import type {
  ExportConflict,
  ExportFormat,
  ExportSettings,
  OutputSharpening,
  OutputSpace,
} from '../../lib/exportTypes'
import { t, type StringKey } from '../../i18n/it'
import { CommittedInput } from '../ui/CommittedInput'
import { Checkbox, Field } from '../ui/Field'
import { Slider } from '../ui/Slider'

const TOKENS = [
  ['basename', '{basename}'],
  ['counter', '{counter:03}'],
  ['seq', '{seq}'],
  ['date', '{date:%Y%m%d}'],
  ['time', '{time:%H%M%S}'],
  ['project', '{project}'],
  ['camera', '{camera}'],
  ['lens', '{lens}'],
  ['iso', '{iso}'],
  ['focal', '{focal}'],
  ['ext', '{ext}'],
] as const

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2 rounded-md border border-ink-700 p-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-300">{title}</h2>
      {children}
    </section>
  )
}

function Select<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T
  options: Array<[T, StringKey]>
  onChange: (value: T) => void
  disabled?: boolean
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value as T)}
      className="h-9 w-full rounded-md border border-ink-600 bg-ink-800 px-2 text-sm text-ink-100 disabled:opacity-50"
    >
      {options.map(([option, label]) => (
        <option key={option} value={option}>
          {t(label)}
        </option>
      ))}
    </select>
  )
}

export function ExportForm({
  projectId,
  settings,
  artist,
  copyright,
  sourceDir,
  onChange,
  onGlobal,
}: {
  projectId: number
  settings: ExportSettings
  artist: string
  copyright: string
  sourceDir: string
  onChange: (changes: Partial<ExportSettings>) => void
  onGlobal: (key: 'artist' | 'copyright', value: string) => void
}) {
  const [template, setTemplate] = useState(settings.template)
  useEffect(() => setTemplate(settings.template), [settings.template])
  const [quality, setQuality] = useState(settings.quality)
  useEffect(() => setQuality(settings.quality), [settings.quality])
  const preview = useQuery({
    queryKey: ['exportName', projectId, template, settings.format],
    queryFn: () => exportApi.exportName(projectId, template),
    placeholderData: (previous) => previous,
  })
  const sidecars = settings.xmp_darktable || settings.xmp_adobe

  return (
    <div className="space-y-3">
      <Section title={t('export.destination')}>
        <CommittedInput
          value={settings.output_dir ?? ''}
          placeholder="/home/…/Esportate"
          onCommit={(value) => onChange({ output_dir: value.trim() || null })}
        />
        <p className="text-xs text-ink-400">
          {t('export.destinationHint')} <span className="text-ink-500">({sourceDir})</span>
        </p>
      </Section>

      <Section title={t('export.what')}>
        <Checkbox checked={settings.images} label={t('export.images')}
          onChange={(images) => onChange({ images })} />
        <Checkbox checked={settings.only_approved} label={t('export.onlyApproved')}
          hint={t('export.onlyApprovedHint')}
          onChange={(only_approved) => onChange({ only_approved })} />
      </Section>

      <Section title={t('export.format')}>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t('export.format')}>
            <Select<ExportFormat> value={settings.format} disabled={!settings.images}
              options={[['jpeg', 'export.format.jpeg'], ['tiff8', 'export.format.tiff8'],
                ['tiff16', 'export.format.tiff16']]}
              onChange={(format) => onChange({ format })} />
          </Field>
          <Field label={t('export.space')}>
            <Select<OutputSpace> value={settings.output_space} disabled={!settings.images}
              options={[['srgb', 'export.space.srgb'], ['display-p3', 'export.space.display-p3'],
                ['adobe-rgb', 'export.space.adobe-rgb'], ['rec2020', 'export.space.rec2020']]}
              onChange={(output_space) => onChange({ output_space })} />
          </Field>
        </div>
        {settings.format === 'jpeg' ? (
          <Slider label={t('export.quality')} value={quality} min={50} max={100} step={1}
            neutral={92} digits={0} disabled={!settings.images}
            onChange={setQuality} onCommit={(value) => onChange({ quality: value })} />
        ) : null}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Checkbox checked={settings.long_edge !== null} label={t('export.resize')}
              disabled={!settings.images}
              onChange={(on) => onChange({ long_edge: on ? 2048 : null })} />
            {settings.long_edge !== null ? (
              <Field label={t('export.longEdge')}>
                <CommittedInput value={String(settings.long_edge)} inputMode="numeric"
                  onCommit={(value) => {
                    const edge = Number.parseInt(value, 10)
                    if (Number.isFinite(edge)) onChange({ long_edge: edge })
                  }} />
              </Field>
            ) : null}
          </div>
          <Field label={t('export.sharpening')} hint={t('export.sharpeningHint')}>
            <Select<OutputSharpening> value={settings.sharpening} disabled={!settings.images}
              options={[['none', 'export.sharpening.none'], ['low', 'export.sharpening.low'],
                ['standard', 'export.sharpening.standard'], ['high', 'export.sharpening.high']]}
              onChange={(sharpening) => onChange({ sharpening })} />
          </Field>
        </div>
        {settings.format !== 'tiff16' ? (
          <Checkbox checked={settings.dither} label={t('export.dither')} disabled={!settings.images}
            onChange={(dither) => onChange({ dither })} />
        ) : null}
      </Section>

      <Section title={t('export.template')}>
        <CommittedInput value={settings.template} onDraft={setTemplate}
          onCommit={(value) => onChange({ template: value })} />
        <div className="flex flex-wrap gap-1">
          {TOKENS.map(([key, token]) => (
            <button key={key} type="button" title={token}
              className="rounded border border-ink-600 px-1.5 py-0.5 text-[11px] text-ink-200 hover:bg-ink-700"
              onClick={() => {
                const next = template.replace(/\.\{ext\}$/, '') + (key === 'ext' ? '' : `_${token}`) + '.{ext}'
                setTemplate(next)
                onChange({ template: next })
              }}>
              {t(`export.token.${key}` as StringKey)}
            </button>
          ))}
        </div>
        <p className="text-xs text-ink-400">{t('export.templateHint')}</p>
        {preview.data ? (
          preview.data.error ? (
            <p className="text-xs text-bad">{preview.data.error}</p>
          ) : (
            <p className="text-xs text-ink-200">
              {t('export.preview', { name: preview.data.name ?? '' })}{' '}
              <span className="text-ink-500">
                {t('export.previewOf', { filename: preview.data.filename ?? '' })}
              </span>
            </p>
          )
        ) : null}
        <Field label={t('export.conflict')}>
          <Select<ExportConflict> value={settings.on_conflict}
            options={[['ask', 'export.conflict.ask'], ['rename', 'export.conflict.rename'],
              ['overwrite', 'export.conflict.overwrite'], ['skip', 'export.conflict.skip']]}
            onChange={(on_conflict) => onChange({ on_conflict })} />
        </Field>
      </Section>

      <Section title={t('export.metadata')}>
        <Checkbox checked={settings.strip_gps} label={t('export.stripGps')}
          hint={t('export.stripGpsHint')} onChange={(strip_gps) => onChange({ strip_gps })} />
        <div className="grid grid-cols-2 gap-3">
          <Field label={t('export.artist')}>
            <CommittedInput value={artist} onCommit={(value) => onGlobal('artist', value)} />
          </Field>
          <Field label={t('export.copyright')}>
            <CommittedInput value={copyright} onCommit={(value) => onGlobal('copyright', value)} />
          </Field>
        </div>
        <p className="text-xs text-ink-400">{t('export.globalHint')}</p>
      </Section>

      <Section title={t('export.sidecars')}>
        <Checkbox checked={settings.xmp_darktable} label={t('export.xmpDarktable')}
          onChange={(xmp_darktable) => onChange({ xmp_darktable })} />
        <Checkbox checked={settings.xmp_adobe} label={t('export.xmpAdobe')}
          onChange={(xmp_adobe) => onChange({ xmp_adobe })} />
        <p className="text-xs text-ink-400">{t('export.sidecarsHint')}</p>
        <Checkbox checked={settings.xmp_beside_raw} label={t('export.besideRaw')}
          disabled={!sidecars} onChange={(xmp_beside_raw) => onChange({ xmp_beside_raw })} />
        {settings.xmp_beside_raw ? (
          <p className="rounded border border-warn/50 p-2 text-xs text-warn">
            {t('export.besideRawWarning')}
          </p>
        ) : null}
      </Section>
    </div>
  )
}
