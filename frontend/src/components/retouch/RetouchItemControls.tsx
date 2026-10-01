// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The controls of the selected removal. A spot: its size, fade and opacity,
// and "another source" (the server looks again, away from the ones tried).
// An eraser: how far its area grows, fade, opacity, the engine, and "another
// variant" -- the next seed, the same area, a different fill. The IA engine
// without its model offers the download (`ModelOffer`).
import { useRef, useState } from 'react'
import { RefreshCw, Shuffle } from 'lucide-react'
import { t } from '../../i18n/it'
import { RETOUCH_CONTROLS, withItem } from '../../lib/retouch'
import { retouchApi } from '../../lib/retouchApi'
import type { EraseItem, HealItem, RetouchItemState, RetouchStates } from '../../lib/retouchTypes'
import { toast } from '../../lib/toast'
import type { EditParams } from '../../lib/types'
import { cn } from '../../lib/utils'
import { TableSlider } from '../masks/MaskSelection'
import { Button } from '../ui/Button'
import { ModelOffer } from './ModelOffer'

export function RetouchItemControls({
  photoId,
  params,
  item,
  state,
  states,
  onChange,
  flush,
  disabled,
}: {
  photoId: number
  params: EditParams
  item: HealItem | EraseItem
  state: RetouchItemState | undefined
  states: RetouchStates | undefined
  onChange: (next: EditParams, commit: boolean) => void
  flush?: () => Promise<void>
  disabled?: boolean
}) {
  const set = (patch: Partial<HealItem> | Partial<EraseItem>, commit: boolean) =>
    onChange(withItem(params, { ...item, ...patch } as HealItem | EraseItem), commit)
  const slider = (key: keyof typeof RETOUCH_CONTROLS, field: string) => (
    <TableSlider control={RETOUCH_CONTROLS[key]} disabled={disabled}
      value={(item as unknown as Record<string, number>)[field]}
      onChange={(value, commit) => {
        set({ [field]: value }, commit)
        // An eraser's fill depends on its growth and fade: ask for it now.
        if (commit && item.kind === 'erase' && field !== 'opacity') void flush?.()
      }} />
  )
  // Without its model the IA engine's failure is the offer to download it.
  const offersModel = item.kind === 'erase' && item.engine === 'ml' && states?.ml.available === false
  return (
    <section className="border-t border-ink-800 py-1.5">
      {item.kind === 'heal' ? (
        <>
          {slider('healRadius', 'radius')}
          {slider('healFeather', 'feather')}
          {slider('healOpacity', 'opacity')}
          <OtherSource photoId={photoId} params={params} item={item} onChange={onChange}
            disabled={disabled} />
        </>
      ) : (
        <>
          {slider('eraseExpand', 'expand')}
          {slider('eraseFeather', 'feather')}
          {slider('eraseOpacity', 'opacity')}
          <div className="flex items-center gap-2 py-1 text-xs text-ink-300">
            <span className="w-16">{t('retouch.engine')}</span>
            {(['classic', 'ml'] as const).map((engine) => {
              const unavailable = engine === 'ml' && states?.ml.available === false
              return (
                <button key={engine} type="button" disabled={disabled}
                  title={unavailable ? t('retouch.ml.unavailable', { reason: states?.ml.reason ?? '' }) : undefined}
                  className={cn(
                    'rounded px-2 py-0.5',
                    item.engine === engine ? 'bg-ink-600 text-ink-50' : 'text-ink-300 hover:bg-ink-700',
                  )}
                  onClick={() => {
                    set({ engine }, true)
                    void flush?.()
                  }}>
                  {t(`retouch.engine.${engine}`)}
                </button>
              )
            })}
          </div>
          <Button size="sm" variant="outline" disabled={disabled} className="mt-1"
            onClick={() => {
              set({ seed: item.seed + 1 }, true)
              void flush?.()
            }}>
            <Shuffle size={13} />
            {t('retouch.variant')}
          </Button>
          {offersModel ? (
            <ModelOffer photoId={photoId} reason={states?.ml.reason ?? null} />
          ) : null}
        </>
      )}
      {state?.state === 'error' && !offersModel ? (
        <div className="mt-2 rounded border border-bad/50 bg-bad/10 p-2 text-xs text-ink-100">
          <p>{t('retouch.failed', { message: state.error ?? '' })}</p>
          <Button size="sm" variant="outline" className="mt-1.5"
            onClick={() => retouchApi.retry(photoId).catch((error: Error) => toast(error.message))}>
            <RefreshCw size={13} />
            {t('retouch.retry')}
          </Button>
        </div>
      ) : null}
    </section>
  )
}

function OtherSource({
  photoId,
  params,
  item,
  onChange,
  disabled,
}: {
  photoId: number
  params: EditParams
  item: HealItem
  onChange: (next: EditParams, commit: boolean) => void
  disabled?: boolean
}) {
  const [busy, setBusy] = useState(false)
  /** The sources tried for this spot, so "another" is never one of them again. */
  const tried = useRef(new Map<string, Array<[number, number]>>())
  const another = async () => {
    setBusy(true)
    try {
      const seen = tried.current.get(item.id) ?? []
      const avoid: Array<[number, number]> = [...seen, [item.sx, item.sy]]
      const index = params.retouch.findIndex((other) => other.id === item.id)
      const source = await retouchApi.source(photoId, params, Math.max(0, index), item, avoid)
      tried.current.set(item.id, avoid)
      onChange(withItem(params, { ...item, ...source, source_auto: true }), true)
    } catch (error) {
      toast((error as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Button size="sm" variant="outline" className="mt-1" disabled={disabled || busy}
      onClick={() => void another()}>
      <RefreshCw size={13} />
      {t('retouch.otherSource')}
    </Button>
  )
}
