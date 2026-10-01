// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A merge made by hand (section 25.2): "l'utente può sempre creare un gruppo a
// mano selezionando N foto nella griglia e scegliendo il tipo di fusione".
// The bar shows over the grid once two photos are picked; choosing a kind
// creates the group, queues its preview and opens the merges screen on it.
// Nothing is merged until "Accetta" there.
import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { X } from 'lucide-react'
import type { ApiError } from '../../lib/api'
import { mergesApi } from '../../lib/mergesApi'
import type { MergeKind } from '../../lib/mergeTypes'
import type { Photo } from '../../lib/types'
import { toast } from '../../lib/toast'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { pickable } from '../PhotoGrid'
import { kindLabel } from './MergeCard'

const KINDS: MergeKind[] = ['hdr', 'focus_stack', 'panorama']

/** The picked photos, and the Ctrl/Shift+click that changes them. */
export function usePicked(photos: Photo[]) {
  const [picked, setPicked] = useState<number[]>([])
  const pick = useCallback(
    (photo: Photo, mode: 'toggle' | 'range') => {
      setPicked((current) => {
        if (mode === 'toggle') {
          return current.includes(photo.id) ? current.filter((id) => id !== photo.id) : [...current, photo.id]
        }
        const last = photos.findIndex((p) => p.id === current[current.length - 1])
        const here = photos.findIndex((p) => p.id === photo.id)
        if (last < 0 || here < 0) return [...current, photo.id]
        const [from, to] = last < here ? [last, here] : [here, last]
        const run = photos.slice(from, to + 1).filter(pickable).map((p) => p.id)
        return [...current, ...run.filter((id) => !current.includes(id))]
      })
    },
    [photos],
  )
  return { picked, set: new Set(picked), pick, clear: () => setPicked([]) }
}

export function PickBar({
  projectId,
  picked,
  onClear,
}: {
  projectId: number
  picked: number[]
  onClear: () => void
}) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const create = useMutation({
    mutationFn: (kind: MergeKind) => mergesApi.create(projectId, kind, picked),
    onSuccess: () => {
      onClear()
      void queryClient.invalidateQueries({ queryKey: ['merges', projectId] })
      navigate(`/progetti/${projectId}/fusioni`)
    },
    onError: (error) => toast((error as ApiError).message),
  })
  if (picked.length < 2) {
    return picked.length === 1 ? (
      <p className="px-2 pb-1 text-[11px] text-ink-400">{t('merges.pick.hint')}</p>
    ) : null
  }
  return (
    <div className="flex flex-wrap items-center gap-1 border-b border-ink-700 px-2 pb-1.5 text-xs text-ink-200">
      <span className="mr-1">{t('merges.pick.count', { count: picked.length })}</span>
      <span className="text-ink-400">{t('merges.pick.create')}</span>
      {KINDS.map((kind) => (
        <Button key={kind} size="sm" variant="outline" disabled={create.isPending} onClick={() => create.mutate(kind)}>
          {kindLabel(kind)}
        </Button>
      ))}
      <Button size="sm" variant="ghost" title={t('merges.pick.clear')} aria-label={t('merges.pick.clear')}
        onClick={onClear}>
        <X size={12} />
      </Button>
    </div>
  )
}
