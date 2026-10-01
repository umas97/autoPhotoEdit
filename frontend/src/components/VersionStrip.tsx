// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The history of one photo under the viewer (section 23): every version with
// where it came from (section 23.1: predetta, modificata da te...), and a
// click to make an old one current again -- as a new version, so that the
// restore is itself undoable. What the autosave still holds is written first,
// so that it is in the history too, and not written over the restore later.
import { useMutation } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { VersionSource } from '../lib/types'
import { t, type StringKey } from '../i18n/it'
import { cn, formatWhen } from '../lib/utils'

export function VersionStrip({
  photoId,
  versions,
  beforeRestore,
  onRestored,
}: {
  photoId: number
  versions: Array<{ id: number; created_at: string; is_current: boolean; source: string }>
  beforeRestore: () => Promise<void>
  onRestored: () => void
}) {
  const restore = useMutation({
    mutationFn: async (versionId: number) => {
      await beforeRestore()
      return api.restoreVersion(photoId, versionId)
    },
    onSuccess: onRestored,
  })

  return (
    <div className="flex items-center gap-2 overflow-x-auto border-t border-ink-700 bg-ink-900 px-3 py-1 text-xs text-ink-400">
      <span className="shrink-0 text-ink-300">{t('viewer.versions')}</span>
      {versions.slice(0, 12).map((version) => (
        <button
          key={version.id}
          type="button"
          disabled={version.is_current || restore.isPending}
          onClick={() => restore.mutate(version.id)}
          className={cn(
            'shrink-0 rounded border px-2 py-0.5',
            version.is_current
              ? 'border-ink-400 text-ink-100'
              : 'border-ink-700 hover:border-ink-500 hover:text-ink-100',
          )}
          title={t('viewer.restore')}
        >
          #{version.id} · {t(`versions.source.${version.source as VersionSource}` as StringKey)} ·{' '}
          {formatWhen(version.created_at)}
        </button>
      ))}
    </div>
  )
}
