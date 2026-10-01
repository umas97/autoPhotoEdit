// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Where the saving stands, in a word: the editor saves by itself
// (lib/autosave.ts), so there is no button to press -- only a failure to
// see, with the way to try again.
import { Check, CloudOff, Loader2 } from 'lucide-react'
import type { SaveState } from '../lib/autosave'
import { t } from '../i18n/it'
import { Button } from './ui/Button'

export function SaveStatus({ state, onRetry }: { state: SaveState; onRetry: () => void }) {
  if (state === 'error') {
    return (
      <span className="flex items-center gap-1 text-xs text-bad" title={t('editing.hint')}>
        <CloudOff size={13} />
        {t('editing.unsaved')}
        <Button size="sm" variant="outline" onClick={onRetry}>
          {t('editing.retry')}
        </Button>
      </span>
    )
  }
  const busy = state === 'saving'
  return (
    <span className="flex items-center gap-1 text-xs text-ink-400" title={t('editing.hint')} aria-live="polite">
      {busy ? <Loader2 size={13} className="animate-spin" /> : state === 'saved' ? <Check size={13} /> : null}
      {t(busy ? 'editing.saving' : state === 'pending' ? 'editing.pending' : 'editing.saved')}
    </span>
  )
}
