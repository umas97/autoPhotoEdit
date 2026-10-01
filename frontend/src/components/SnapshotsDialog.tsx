// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The snapshots of a project (section 23.2): name the state of every photo,
// see the ones taken by themselves at the passages of the workflow, go back
// to one. Going back asks first -- with how many photos change -- and then
// offers its own undo, which is the snapshot of the state it replaced.
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import type { Snapshot } from '../lib/types'
import { formatWhen } from '../lib/utils'
import { t } from '../i18n/it'
import { Button } from './ui/Button'
import { Dialog } from './ui/Dialog'
import { TextInput } from './ui/Field'

export function SnapshotsDialog({
  projectId,
  open,
  onOpenChange,
  flush,
}: {
  projectId: number
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Write the edit waiting for the autosave: it is part of "now", which every count compares with. */
  flush: () => Promise<void>
}) {
  const queryClient = useQueryClient()
  const list = useQuery({
    queryKey: ['snapshots', projectId],
    queryFn: async () => {
      await flush()
      return api.snapshots(projectId)
    },
    enabled: open,
    // Every opening compares with *now*: never a count from before the last edit.
    staleTime: 0,
  })
  const [name, setName] = useState('')
  const [asking, setAsking] = useState<Snapshot | null>(null)
  const [result, setResult] = useState<{ restored: number; missing: number; undo: number } | null>(null)

  const changed = () => {
    for (const queryKey of [['snapshots', projectId], ['photo'], ['photos', projectId], ['review', projectId], ['culling', projectId], ['photoReview']]) {
      void queryClient.invalidateQueries({ queryKey })
    }
  }

  const take = useMutation({
    mutationFn: async () => {
      await flush()
      return api.takeSnapshot(projectId, name.trim())
    },
    onSuccess: () => {
      setName('')
      changed()
    },
  })
  const restore = useMutation({
    mutationFn: async (snapshotId: number) => {
      await flush()
      return api.restoreSnapshot(projectId, snapshotId)
    },
    onSuccess: (answer) => {
      setAsking(null)
      setResult(answer)
      changed()
    },
  })
  const remove = useMutation({
    mutationFn: (snapshotId: number) => api.deleteSnapshot(projectId, snapshotId),
    onSuccess: changed,
  })

  const snapshots = list.data ?? []
  return (
    <Dialog open={open} onOpenChange={(next) => { onOpenChange(next); if (!next) { setAsking(null); setResult(null) } }}
      title={t('snapshots.title')} description={t('snapshots.description')}
      footer={<Button variant="ghost" onClick={() => onOpenChange(false)}>{t('snapshots.close')}</Button>}>
      <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); if (name.trim()) take.mutate() }}>
        <TextInput aria-label={t('snapshots.name')} placeholder={t('snapshots.namePlaceholder')} value={name}
          maxLength={200} onChange={(event) => setName(event.target.value)} />
        <Button type="submit" variant="primary" disabled={!name.trim() || take.isPending}>{t('snapshots.take')}</Button>
      </form>

      {result ? (
        <div className="mt-3 flex items-center gap-2 rounded border border-ink-600 bg-ink-800 px-2 py-1.5 text-xs text-ink-100">
          <span>{t('snapshots.restored', { count: result.restored })}</span>
          {result.missing ? <span className="text-warn">{t('snapshots.missing', { count: result.missing })}</span> : null}
          <Button size="sm" variant="outline" className="ml-auto" disabled={restore.isPending}
            onClick={() => restore.mutate(result.undo)}>{t('snapshots.undo')}</Button>
        </div>
      ) : null}

      {asking ? (
        <div className="mt-3 space-y-2 rounded border border-warn/50 bg-ink-800 p-2 text-xs text-ink-100">
          <p>{t('snapshots.confirm', { name: asking.name, count: asking.differs })}</p>
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setAsking(null)}>{t('common.cancel')}</Button>
            <Button size="sm" variant="primary" disabled={restore.isPending}
              onClick={() => restore.mutate(asking.id)}>{t('snapshots.confirmYes')}</Button>
          </div>
        </div>
      ) : null}

      <ul className="mt-3 max-h-[45vh] space-y-1 overflow-y-auto">
        {snapshots.length === 0 && !list.isLoading ? (
          <li className="text-xs text-ink-400">{t('snapshots.empty')}</li>
        ) : null}
        {snapshots.map((snapshot) => (
          <li key={snapshot.id} className="flex items-center gap-2 rounded px-1.5 py-1 text-xs hover:bg-ink-800">
            <div className="min-w-0 flex-1">
              <div className="truncate text-ink-50">
                {snapshot.name}
                {snapshot.kind === 'auto' ? <span className="ml-1.5 text-ink-400">· {t('snapshots.auto')}</span> : null}
              </div>
              <div className="text-ink-400">
                {formatWhen(snapshot.created_at)} · {t('snapshots.photos', { count: snapshot.photos })} ·{' '}
                {snapshot.differs ? t('snapshots.differs', { count: snapshot.differs }) : t('snapshots.same')}
              </div>
            </div>
            <Button size="sm" variant="outline" disabled={!snapshot.differs || restore.isPending}
              onClick={() => { setResult(null); setAsking(snapshot) }}>{t('snapshots.restore')}</Button>
            <Button size="sm" variant="ghost" disabled={remove.isPending}
              onClick={() => remove.mutate(snapshot.id)}>{t('snapshots.delete')}</Button>
          </li>
        ))}
      </ul>
    </Dialog>
  )
}
