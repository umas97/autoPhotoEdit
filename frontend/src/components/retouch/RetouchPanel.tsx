// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Rimozione group of the masks tab (docs/SPEC_rimozione.md R4): its two
// tools, its own list, the selected removal's controls, an area taken from a
// mask, and "Copia punti sulle foto selezionate".
//
// Every change goes through the same onChange as a slider -- one undo step
// per gesture, saved by the autosave -- and nothing here writes a fill: the
// server resolves it (backend/ape/retouch/fills.py).
import { useState } from 'react'
import { Eraser, Eye, EyeOff, Layers, Loader2, MoreHorizontal, Sparkles, Trash2 } from 'lucide-react'
import { t } from '../../i18n/it'
import { itemById, newErase, withAddedItem, withItem, withoutItem } from '../../lib/retouch'
import { retouchApi } from '../../lib/retouchApi'
import type { RetouchItem, RetouchState } from '../../lib/retouchTypes'
import { toast } from '../../lib/toast'
import type { EditParams } from '../../lib/types'
import { cn } from '../../lib/utils'
import { nextName } from '../../lib/masks'
import { Button } from '../ui/Button'
import { Checkbox } from '../ui/Field'
import type { MaskEditor, RetouchTool } from '../masks/useMaskEditor'
import { CopySpotsDialog } from './CopySpotsDialog'
import { RetouchItemControls } from './RetouchItemControls'
import { useRetouchStates } from './useRetouch'

const STATE_TONE: Record<RetouchState, string> = {
  ready: 'text-ink-500',
  computing: 'text-ink-200',
  stale: 'text-warn',
  error: 'text-bad',
}

export function RetouchPanel({
  photoId,
  projectId,
  params,
  onChange,
  editor,
  flush,
  copyTargets,
  disabled,
}: {
  photoId: number
  projectId?: number
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  editor: MaskEditor
  flush?: () => Promise<void>
  /** The photos picked in the grid: where "Copia punti" copies to. */
  copyTargets?: number[]
  disabled?: boolean
}) {
  const { states, byId } = useRetouchStates(photoId, params)
  const [fromMask, setFromMask] = useState(false)
  const [menu, setMenu] = useState(false)
  const [copying, setCopying] = useState(false)
  const selected = itemById(params, editor.removal)

  const pick = (tool: RetouchTool) => {
    const next = editor.tool === tool ? null : tool
    editor.setTool(next)
    if (next && selected && selected.kind !== next) editor.setRemoval(null)
  }

  const select = (item: RetouchItem) => {
    const again = editor.removal === item.id
    editor.setRemoval(again ? null : item.id)
    editor.setTool(again ? null : item.kind)
  }

  const takeMask = async (index: number) => {
    try {
      const { name } = await retouchApi.areaFromMask(photoId, params, index)
      // A little growth: the halo around a selected object belongs to it.
      const item = newErase(params, name, 0.004)
      onChange(withAddedItem(params, item), true)
      editor.setRemoval(item.id)
      editor.setTool('erase')
      setFromMask(false)
      void flush?.()
    } catch (error) {
      toast((error as Error).message)
    }
  }

  const count = { heal: 0, erase: 0 }
  return (
    <section className="mb-2 border-b border-ink-800 pb-2">
      <div className="flex items-center gap-1 pb-1.5">
        <h3 className="flex-1 text-xs font-medium text-ink-100">{t('retouch.title')}</h3>
        <div className="relative">
          <Button size="icon" variant="ghost" title={t('retouch.menu')} onClick={() => setMenu(!menu)}>
            <MoreHorizontal size={14} />
          </Button>
          {menu ? (
            <div className="absolute right-0 z-20 mt-1 w-64 rounded border border-ink-700 bg-ink-900 p-1 shadow-lg">
              <button type="button" className="w-full rounded px-2 py-1.5 text-left text-xs text-ink-100 hover:bg-ink-700"
                onClick={() => {
                  setMenu(false)
                  setCopying(true)
                }}>
                {t('retouch.copy')}
              </button>
            </div>
          ) : null}
        </div>
      </div>
      <div className="flex flex-wrap gap-1">
        <ToolButton active={editor.tool === 'heal'} disabled={disabled} onClick={() => pick('heal')}
          icon={<Sparkles size={13} />} label={t('retouch.heal')} hint={t('retouch.heal.hint')} />
        <ToolButton active={editor.tool === 'erase'} disabled={disabled} onClick={() => pick('erase')}
          icon={<Eraser size={13} />} label={t('retouch.erase')} hint={t('retouch.erase.hint')} />
        <Button size="sm" variant="ghost" disabled={disabled} onClick={() => setFromMask(!fromMask)}>
          <Layers size={13} />
          {t('retouch.fromMask')}
        </Button>
      </div>
      {editor.tool ? (
        <p className="pt-1.5 text-[11px] text-ink-400">{t(`retouch.hint.${editor.tool}`)}</p>
      ) : null}

      {fromMask ? (
        <div className="mt-2 rounded-md border border-ink-700 bg-ink-900 p-2 text-xs">
          <p className="pb-1 font-medium text-ink-100">{t('retouch.fromMask.title')}</p>
          {params.masks.length === 0 ? (
            <p className="text-ink-400">{t('retouch.fromMask.none')}</p>
          ) : (
            <ul className="space-y-0.5">
              {params.masks.map((mask, index) => (
                <li key={index}>
                  <button type="button" className="w-full rounded px-2 py-1 text-left text-ink-200 hover:bg-ink-700"
                    onClick={() => void takeMask(index)}>
                    {mask.name || nextName(mask.kind, [])}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <p className="pt-1.5 text-[11px] text-ink-500">{t('retouch.fromMask.note')}</p>
          <div className="flex justify-end gap-1 pt-1">
            <Button size="sm" variant="ghost" onClick={() => editor.setSegmenting(true)}>
              {t('masks.kind.segment')}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setFromMask(false)}>
              {t('retouch.fromMask.cancel')}
            </Button>
          </div>
        </div>
      ) : null}

      {params.retouch.length === 0 ? (
        <p className="py-2 text-xs text-ink-400">{t('retouch.empty')}</p>
      ) : (
        <ul className="space-y-0.5 py-1.5" aria-label={t('retouch.list')}>
          {params.retouch.map((item) => {
            count[item.kind] += 1
            const state = byId.get(item.id)?.state ?? (item.kind === 'heal' ? 'ready' : 'computing')
            return (
              <li key={item.id} className={cn(
                'flex items-center gap-1 rounded px-2 py-1 text-sm',
                item.id === editor.removal ? 'bg-ink-700 text-ink-50' : 'text-ink-200 hover:bg-ink-800',
              )}>
                <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  aria-pressed={item.id === editor.removal} onClick={() => select(item)}>
                  {item.kind === 'heal' ? <Sparkles size={13} className="shrink-0 text-ink-400" />
                    : <Eraser size={13} className="shrink-0 text-ink-400" />}
                  <span className="min-w-0 flex-1 truncate">
                    {t(`retouch.item.${item.kind}`, { n: count[item.kind] })}
                  </span>
                  <span className={cn('flex items-center gap-1 text-[11px]', STATE_TONE[state])}>
                    {state === 'computing' || state === 'stale' ? <Loader2 size={11} className="animate-spin" /> : null}
                    {t(`retouch.state.${state}`)}
                  </span>
                </button>
                <Button size="icon" variant="ghost" className="h-6 w-6" title={t('retouch.visible')}
                  disabled={disabled}
                  onClick={() => onChange(withItem(params, { ...item, visible: !item.visible }), true)}>
                  {item.visible ? <Eye size={12} /> : <EyeOff size={12} />}
                </Button>
                <Button size="icon" variant="ghost" className="h-6 w-6" title={t('retouch.delete')}
                  disabled={disabled}
                  onClick={() => {
                    onChange(withoutItem(params, item.id), true)
                    if (editor.removal === item.id) editor.setRemoval(null)
                  }}>
                  <Trash2 size={12} />
                </Button>
              </li>
            )
          })}
        </ul>
      )}

      {selected ? (
        <RetouchItemControls photoId={photoId} params={params} item={selected}
          state={byId.get(selected.id)} states={states} onChange={onChange} flush={flush}
          disabled={disabled} />
      ) : null}

      {params.retouch.length > 0 ? (
        <Checkbox checked={editor.showRemovals} label={t('retouch.showRemovals')}
          onChange={editor.setShowRemovals} />
      ) : null}

      {projectId !== undefined ? (
        <CopySpotsDialog open={copying} onOpenChange={setCopying} projectId={projectId}
          photoId={photoId} params={params} targets={copyTargets ?? []} flush={flush} />
      ) : null}
    </section>
  )
}

function ToolButton({
  active,
  disabled,
  onClick,
  icon,
  label,
  hint,
}: {
  active: boolean
  disabled?: boolean
  onClick: () => void
  icon: JSX.Element
  label: string
  hint: string
}) {
  return (
    <Button size="sm" variant={active ? 'primary' : 'outline'} disabled={disabled} title={hint}
      aria-pressed={active} onClick={onClick}>
      {icon}
      {label}
    </Button>
  )
}
