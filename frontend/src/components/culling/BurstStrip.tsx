// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A burst, opened (section 7.6): every frame side by side, the proposed one
// first, and the means to change which one it is.
//
// "Scegli questo" is the ↑/↓ of the keyboard: the chosen frame is kept and the
// frame it replaces is discarded, as one undoable step. Keeping a second frame
// as well is a different gesture -- "Tieni anche" -- because it is a different
// intention: this burst holds two moments, not one.
import { Check, Plus, X } from 'lucide-react'
import { thumbUrl } from '../../lib/api'
import { percent } from '../../lib/culling'
import type { CullPhoto, UserDecision } from '../../lib/cullTypes'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'

export function BurstStrip({
  members,
  currentId,
  onSelect,
  onChoose,
  onDecide,
  onClose,
}: {
  members: CullPhoto[]
  currentId: number | null
  onSelect: (photo: CullPhoto) => void
  /** Make this frame the proposed one of the burst. */
  onChoose: (photo: CullPhoto) => void
  onDecide: (photo: CullPhoto, decision: UserDecision) => void
  onClose: () => void
}) {
  return (
    <section className="shrink-0 border-b border-ink-700 bg-ink-900 px-3 py-2">
      <header className="mb-1.5 flex items-center gap-2 text-xs text-ink-300">
        <span className="text-ink-100">{t('culling.burst.title', { count: members.length })}</span>
        <span>{t('culling.burst.hint')}</span>
        <Button size="sm" variant="ghost" className="ml-auto" onClick={onClose}>
          <X size={13} />
          {t('common.close')}
        </Button>
      </header>
      <ul className="flex gap-2 overflow-x-auto pb-1">
        {members.map((photo, index) => (
          <li key={photo.id} className="w-52 shrink-0">
            <button
              type="button"
              onClick={() => onSelect(photo)}
              className={cn(
                'block w-full overflow-hidden rounded border bg-mat',
                photo.id === currentId ? 'border-ink-100' : 'border-ink-700 hover:border-ink-500',
              )}
            >
              <img
                src={thumbUrl(photo)}
                alt={photo.filename}
                className={cn('aspect-[3/2] w-full object-contain', photo.culled && 'opacity-35')}
              />
            </button>
            <div className="mt-1 flex items-center gap-1 text-[11px] text-ink-300">
              <span className="tabular-nums text-ink-100">{percent(photo.score)}</span>
              {index === 0 && !photo.culled ? (
                <span className="text-ink-200">{t('culling.burst.proposed')}</span>
              ) : null}
              <span className="ml-auto flex gap-1">
                {photo.culled ? (
                  <>
                    <Button size="sm" variant="outline" onClick={() => onChoose(photo)}>
                      <Check size={12} />
                      {t('culling.burst.choose')}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      title={t('culling.burst.keepToo')}
                      onClick={() => onDecide(photo, 'keep')}
                    >
                      <Plus size={12} />
                    </Button>
                  </>
                ) : (
                  <Button
                    size="sm"
                    variant="ghost"
                    title={t('culling.action.discard')}
                    onClick={() => onDecide(photo, 'discard')}
                  >
                    <X size={12} />
                  </Button>
                )}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
