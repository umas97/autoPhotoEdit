// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The masks tab of the right-hand panel (section 6.3): the list of the photo's
// masks, the ways to add one, and for the selected mask what it selects and
// what it does.
//
// Every change goes through the same onChange as the global sliders, so the
// preview, the undo stack and "unsaved" treat a mask like any other
// parameter: nothing is saved until the photo is.
import { useState, type ReactNode } from 'react'
import { Blend, Brush, Circle, Copy, FlipVertical2, Rows3, ScanFace, Trash2 } from 'lucide-react'
import { t, type StringKey } from '../../i18n/it'
import { proxyUrl } from '../../lib/api'
import { useFrameSize } from '../../lib/geometry'
import {
  ADDABLE,
  MASK_GROUPS,
  SELECTION_CONTROLS,
  adjusts,
  newMask,
  nextName,
  readMaskPath,
  withAddedMask,
  withMask,
  withMaskPath,
  withoutMask,
} from '../../lib/masks'
import type { MaskKind, MaskParams } from '../../lib/maskTypes'
import { toast } from '../../lib/toast'
import type { EditParams, Photo } from '../../lib/types'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Checkbox } from '../ui/Field'
import { uploadEmptyRaster } from './BrushCanvas'
import { MaskSelection, TableSlider } from './MaskSelection'
import { SegmentControls } from './SegmentControls'
import { RetouchPanel } from '../retouch/RetouchPanel'
import type { MaskEditor } from './useMaskEditor'

export interface RetouchContext {
  projectId: number
  /** Write the edit now: a fill is asked for by a finished gesture. */
  flush: () => Promise<void>
  /** The photos picked in the grid, for "Copia punti". */
  copyTargets?: number[]
}

export const KIND_ICON: Record<MaskKind, typeof Circle> = {
  linear: Rows3,
  radial: Circle,
  brush: Brush,
  parametric: Blend,
  segment: ScanFace,
}

export function MaskPanel({
  params,
  onChange,
  editor,
  photo,
  disabled,
  retouch,
}: {
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  editor: MaskEditor
  photo: Pick<Photo, 'id' | 'has_proxy' | 'proxy_rev'>
  disabled?: boolean
  /** What the Rimozione group needs beyond the edit. */
  retouch?: RetouchContext
}) {
  const frame = useFrameSize(photo.has_proxy ? proxyUrl(photo) : null)
  const [adding, setAdding] = useState(false)
  const index = editor.selected
  const mask = index !== null ? params.masks[index] : undefined

  const add = async (kind: MaskKind) => {
    if (kind === 'segment') {
      // A subject is added by the segmentation, once it has run.
      editor.setSelected(null)
      editor.setSegmenting(true)
      return
    }
    setAdding(true)
    try {
      const raster = kind === 'brush' && frame ? await uploadEmptyRaster(frame) : undefined
      if (kind === 'brush' && !raster) return
      const next = withAddedMask(params, newMask(kind, params, frame, raster))
      onChange(next, true)
      editor.setSelected(next.masks.length - 1)
    } catch (error) {
      toast((error as Error).message)
    } finally {
      setAdding(false)
    }
  }

  const change = (next: MaskParams, commit: boolean) => onChange(withMask(params, index!, next), commit)

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-ink-850 px-3 py-2">
      <RetouchPanel photoId={photo.id} params={params} onChange={onChange} editor={editor}
        disabled={disabled} projectId={retouch?.projectId} flush={retouch?.flush}
        copyTargets={retouch?.copyTargets} />
      <h3 className="pb-1.5 text-xs font-medium text-ink-100">{t('masks.tab.masks')}</h3>
      <div className="flex flex-wrap gap-1 pb-2">
        {ADDABLE.map((kind) => {
          const Icon = KIND_ICON[kind]
          return (
            <Button key={kind} size="sm" variant="outline" disabled={disabled || adding || !frame}
              title={t(`masks.kind.${kind}.hint` as StringKey)} onClick={() => void add(kind)}>
              <Icon size={13} />
              {t(`masks.kind.${kind}` as StringKey)}
            </Button>
          )
        })}
      </div>

      {editor.segmenting ? (
        <SegmentControls photoId={photo.id} params={params} onChange={onChange} editor={editor} />
      ) : null}

      {params.masks.length === 0 && !editor.segmenting ? (
        <p className="py-2 text-xs text-ink-400">{t('masks.empty')}</p>
      ) : (
        <ul className="space-y-0.5 pb-2" aria-label={t('masks.list')}>
          {params.masks.map((item, i) => {
            const Icon = KIND_ICON[item.kind]
            return (
              <li key={i}>
                <button type="button"
                  className={cn(
                    'flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm',
                    i === index ? 'bg-ink-700 text-ink-50' : 'text-ink-200 hover:bg-ink-800',
                  )}
                  aria-pressed={i === index}
                  onClick={() => {
                    editor.setSegmenting(false)
                    editor.setSelected(i === index ? null : i)
                  }}>
                  <Icon size={14} className="shrink-0 text-ink-400" />
                  <span className="min-w-0 flex-1 truncate">{item.name || nextName(item.kind, [])}</span>
                  {item.invert ? <FlipVertical2 size={12} className="text-ink-400" /> : null}
                  {!adjusts(item) ? <span className="text-[11px] text-ink-500">{t('masks.inert')}</span> : null}
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {mask ? (
        <>
          <div className="flex items-center gap-1 border-t border-ink-800 py-1.5">
            <input
              className="h-7 min-w-0 flex-1 rounded border border-ink-700 bg-ink-800 px-2 text-sm text-ink-50 focus:border-ink-400 focus:outline-none"
              value={mask.name}
              maxLength={60}
              aria-label={t('masks.name')}
              disabled={disabled}
              onChange={(event) => change({ ...mask, name: event.target.value }, false)}
              onBlur={(event) => change({ ...mask, name: event.target.value }, true)}
            />
            <Button size="sm" variant="ghost" title={t('masks.duplicate')} disabled={disabled}
              onClick={() => {
                const copy = { ...structuredClone(mask), name: nextName(mask.kind, params.masks) }
                const next = withAddedMask(params, copy)
                onChange(next, true)
                editor.setSelected(next.masks.length - 1)
              }}>
              <Copy size={13} />
            </Button>
            <Button size="sm" variant="ghost" title={`${t('masks.delete')} (Canc)`} disabled={disabled}
              onClick={() => {
                onChange(withoutMask(params, index!), true)
                editor.setSelected(null)
              }}>
              <Trash2 size={13} />
            </Button>
          </div>

          <Section label="masks.group.selection">
            <MaskSelection mask={mask} editor={editor} onChange={change} disabled={disabled} />
            <Checkbox checked={mask.invert} disabled={disabled} label={t('masks.invert')}
              onChange={(invert) => change({ ...mask, invert }, true)} />
            <TableSlider control={SELECTION_CONTROLS.opacity} value={mask.opacity} disabled={disabled}
              onChange={(opacity, commit) => change({ ...mask, opacity }, commit)} />
            <Checkbox checked={editor.showSelection} label={t('masks.showSelection')}
              onChange={editor.setShowSelection} />
          </Section>

          {MASK_GROUPS.map((group) => (
            <Section key={group.key} label={group.label}>
              {group.controls.map((control) => (
                <TableSlider key={control.path} control={control} disabled={disabled}
                  value={readMaskPath(mask, control.path)}
                  onChange={(value, commit) => change(withMaskPath(mask, control.path, value), commit)} />
              ))}
            </Section>
          ))}
        </>
      ) : params.masks.length > 0 ? (
        <p className="py-2 text-xs text-ink-400">{t('masks.choose')}</p>
      ) : null}

      {params.masks.length > 0 || params.retouch.length > 0 ? (
        <p className="mt-auto border-t border-ink-800 pt-2 text-[11px] text-ink-500">
          {t(params.retouch.length > 0 ? 'retouch.notInSidecars' : 'masks.notInSidecars')}
        </p>
      ) : null}
    </div>
  )
}

function Section({ label, children }: { label: StringKey; children: ReactNode }) {
  return (
    <section className="border-t border-ink-800 py-1.5">
      <h3 className="py-1 text-xs font-medium text-ink-100">{t(label)}</h3>
      {children}
    </section>
  )
}
