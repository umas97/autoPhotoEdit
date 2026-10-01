// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The right-hand column of the viewer and of the review: the global
// adjustments, or the masks. Two tabs over one edit -- both write the same
// parameters through the same onChange, so undo and "unsaved" see no
// difference between them.
import type { ReactNode } from 'react'
import { t } from '../../i18n/it'
import type { EditParams, Photo } from '../../lib/types'
import { cn } from '../../lib/utils'
import { ParamPanel } from '../ParamPanel'
import { MaskPanel, type RetouchContext } from './MaskPanel'
import type { MaskEditor } from './useMaskEditor'

export function EditSidebar({
  params,
  onChange,
  editor,
  photo,
  disabled,
  header,
  retouch,
}: {
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  editor: MaskEditor
  photo: Pick<Photo, 'id' | 'has_proxy' | 'proxy_rev'>
  disabled?: boolean
  /** Above the global groups: style, geometry, confidence. */
  header?: ReactNode
  retouch?: RetouchContext
}) {
  const count = params.masks.length + params.retouch.length
  const tab = (key: 'adjust' | 'masks', label: string, hint: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={editor.tab === key}
      title={hint}
      className={cn(
        'flex-1 border-b-2 px-2 py-1.5 text-xs font-medium',
        editor.tab === key
          ? 'border-ink-200 text-ink-50'
          : 'border-transparent text-ink-400 hover:text-ink-200',
      )}
      onClick={() => editor.setTab(key)}
    >
      {label}
    </button>
  )
  return (
    <div className="flex h-full flex-col">
      <div role="tablist" className="flex shrink-0 border-b border-ink-700 bg-ink-850">
        {tab('adjust', t('masks.tab.adjust'), 'G')}
        {tab('masks', count ? t('masks.tab.masksCount', { count }) : t('masks.tab.masks'), 'M')}
      </div>
      <div className="min-h-0 flex-1">
        {editor.tab === 'adjust' ? (
          <ParamPanel params={params} onChange={onChange} disabled={disabled} header={header} />
        ) : (
          <MaskPanel params={params} onChange={onChange} editor={editor} photo={photo}
            disabled={disabled} retouch={retouch} />
        )}
      </div>
    </div>
  )
}
