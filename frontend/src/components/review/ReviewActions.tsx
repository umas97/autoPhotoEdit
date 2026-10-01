// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The bar under the photograph of the review: what is on screen, the last
// notice, the autosave, and the actions with their keys (Z, C, R, A). The
// keys themselves are handled by the page; this only draws the buttons.
import { Check, Columns2, Images, X, ZoomIn } from 'lucide-react'
import type { SaveState } from '../../lib/autosave'
import { t } from '../../i18n/it'
import { SaveStatus } from '../SaveStatus'
import { Button } from '../ui/Button'

export type ReviewAction = 'apply' | 'approve' | 'reject'

export function ReviewActions({
  scenes,
  label,
  notice,
  saveState,
  onRetry,
  zoom,
  onZoom,
  compare,
  onCompare,
  hasPhoto,
  canApply,
  busy,
  onAct,
}: {
  /** Judging scenes rather than the queue: the labels speak of the scene. */
  scenes: boolean
  label: string
  notice: string | null
  saveState: SaveState
  onRetry: () => void
  zoom: boolean
  onZoom: () => void
  compare: boolean
  onCompare: () => void
  hasPhoto: boolean
  /** "Applica alla scena": a correction to send, on the representative. */
  canApply: boolean
  busy: boolean
  onAct: (action: ReviewAction) => void
}) {
  return (
    <>
      <footer className="flex shrink-0 items-center gap-2 border-t border-ink-700 bg-ink-900 px-3 py-1.5 text-xs">
        <span className="truncate text-ink-200">{label}</span>
        {notice ? <span className="truncate text-ink-300">{notice}</span> : null}
        <SaveStatus state={saveState} onRetry={onRetry} />
        <span className="ml-auto flex items-center gap-1">
          <Button size="sm" variant={zoom ? 'secondary' : 'ghost'} title="Z" onClick={onZoom}>
            <ZoomIn size={13} />{t('review.zoom')}
          </Button>
          <Button size="sm" variant={compare ? 'secondary' : 'ghost'} title="C" onClick={onCompare}>
            <Columns2 size={13} />{t('review.compare')}
          </Button>
          <Button size="sm" variant="outline" disabled={!hasPhoto || busy}
            title={t(scenes ? 'review.rejectHintScene' : 'review.rejectHintPhoto')}
            onClick={() => onAct('reject')}>
            <X size={13} />{t(scenes ? 'review.rejectScene' : 'review.reject')} (R)
          </Button>
          {scenes ? (
            <Button size="sm" variant="secondary" disabled={!canApply || busy} onClick={() => onAct('apply')}>
              <Images size={13} />{t('review.applyScene')}
            </Button>
          ) : null}
          <Button size="sm" variant="primary" disabled={!hasPhoto || busy} onClick={() => onAct('approve')}>
            <Check size={13} />{t(scenes ? 'review.approveScene' : 'review.approve')} (A)
          </Button>
        </span>
      </footer>
      <p className="hidden shrink-0 bg-ink-900 px-3 pb-1 text-[11px] text-ink-500 xl:block">{t('review.shortcuts')}</p>
    </>
  )
}
