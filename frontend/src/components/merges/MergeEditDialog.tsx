// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Modifica gruppo" (section 25.6): add or remove frames, change the reference,
// and the options a merge of this kind has -- deghosting for a bracketing, the
// projection for a panorama (section 25.5.3). The candidates to add are the
// shots taken within a minute of the group: a bracketing or a panorama is a
// few seconds of shooting, and a list of the whole card would bury them.
import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Star } from 'lucide-react'
import { api } from '../../lib/api'
import { memberThumbUrl } from '../../lib/mergesApi'
import type { MergeGroupView } from '../../lib/mergeTypes'
import type { Photo } from '../../lib/types'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Checkbox } from '../ui/Field'

const NEARBY_MS = 60_000
const PROJECTIONS = ['cylindrical', 'spherical', 'plane'] as const
/** Section 25.5.4: the way out of the size limit is a smaller output. */
const OUTPUT_SCALES = [1, 0.75, 0.5] as const

export interface MergeEdit {
  photo_ids: number[]
  reference_id: number
  options: Record<string, unknown>
}

function nearby(photos: Photo[], group: MergeGroupView): Photo[] {
  const times = group.members.map((m) => (m.shot_at ? Date.parse(m.shot_at) : NaN)).filter(Number.isFinite)
  const members = new Set(group.members.map((m) => m.photo_id))
  if (times.length === 0) return photos.filter((p) => members.has(p.id))
  const [first, last] = [Math.min(...times), Math.max(...times)]
  return photos.filter((photo) => {
    if (members.has(photo.id)) return true
    if (photo.merge !== null || photo.superseded || photo.missing || !photo.shot_at) return false
    const at = Date.parse(photo.shot_at)
    return at >= first - NEARBY_MS && at <= last + NEARBY_MS
  })
}

export function MergeEditDialog({
  projectId,
  group,
  onClose,
  onSave,
}: {
  projectId: number
  group: MergeGroupView | null
  onClose: () => void
  onSave: (group: MergeGroupView, edit: MergeEdit) => void
}) {
  const photos = useQuery({
    queryKey: ['photos', projectId, 'merge-candidates'],
    queryFn: () => api.listPhotos(projectId, 0, 2000, true, false),
    enabled: group !== null,
  })
  const [chosen, setChosen] = useState<Set<number>>(new Set())
  const [reference, setReference] = useState<number | null>(null)
  const [options, setOptions] = useState<Record<string, unknown>>({})

  useEffect(() => {
    if (!group) return
    setChosen(new Set(group.members.map((m) => m.photo_id)))
    setReference(group.members.find((m) => m.reference)?.photo_id ?? null)
    setOptions(group.options)
  }, [group])

  const candidates = useMemo(
    () => (group && photos.data ? nearby(photos.data.items, group) : []),
    [group, photos.data],
  )
  if (!group) return null

  const toggle = (id: number) => {
    const next = new Set(chosen)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setChosen(next)
    if (reference === id && !next.has(id)) setReference(null)
  }
  const ids = candidates.filter((p) => chosen.has(p.id)).map((p) => p.id)
  const referenceId = reference !== null && chosen.has(reference) ? reference : ids[Math.floor(ids.length / 2)]

  return (
    <Dialog
      open
      onOpenChange={(open) => (open ? undefined : onClose())}
      title={t('merges.editTitle')}
      description={t('merges.editBody')}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="primary"
            disabled={ids.length < 2}
            title={ids.length < 2 ? t('merges.editTooFew') : undefined}
            onClick={() => onSave(group, { photo_ids: ids, reference_id: referenceId, options })}
          >
            {t('merges.editSave')}
          </Button>
        </>
      }
    >
      <p className="mb-1 text-xs text-ink-400">{t('merges.editNearby')}</p>
      <ul className="max-h-72 space-y-1 overflow-y-auto pr-1">
        {candidates.map((photo) => (
          <li key={photo.id}
            className={cn('flex items-center gap-2 rounded p-1', chosen.has(photo.id) ? 'bg-ink-800' : 'opacity-70')}>
            <input type="checkbox" className="h-4 w-4 accent-ink-200" checked={chosen.has(photo.id)}
              onChange={() => toggle(photo.id)} aria-label={photo.filename} />
            <img src={memberThumbUrl({ photo_id: photo.id, has_thumb: photo.has_thumb, has_proxy: photo.has_proxy,
              proxy_rev: photo.proxy_rev })}
              alt="" className="h-10 w-15 rounded bg-mat object-contain" loading="lazy" />
            <span className="min-w-0 flex-1 truncate text-xs text-ink-200">{photo.filename}</span>
            <Button size="sm" variant={photo.id === referenceId ? 'secondary' : 'ghost'}
              disabled={!chosen.has(photo.id)} onClick={() => setReference(photo.id)}
              title={t('merges.editReference')}>
              <Star size={12} />
              {photo.id === referenceId ? t('merges.editReference') : null}
            </Button>
          </li>
        ))}
      </ul>

      {group.kind === 'hdr' ? (
        <div className="mt-3">
          <Checkbox checked={options.deghost !== false} label={t('merges.option.deghost')}
            onChange={(checked) => setOptions({ ...options, deghost: checked })} />
        </div>
      ) : null}
      {group.kind === 'panorama' ? (
        <fieldset className="mt-3">
          <legend className="text-xs text-ink-300">{t('merges.option.projection')}</legend>
          <div className="mt-1 flex gap-1">
            {PROJECTIONS.map((projection) => (
              <Button key={projection} size="sm"
                variant={(options.projection ?? 'auto') === projection ? 'secondary' : 'ghost'}
                onClick={() => setOptions({ ...options, projection })}>
                {t(`merges.projection.${projection}`)}
              </Button>
            ))}
          </div>
          <legend className="mt-3 text-xs text-ink-300">{t('merges.option.outputScale')}</legend>
          <div className="mt-1 flex gap-1">
            {OUTPUT_SCALES.map((scale) => (
              <Button key={scale} size="sm"
                variant={Number(options.output_scale ?? 1) === scale ? 'secondary' : 'ghost'}
                onClick={() => setOptions({ ...options, output_scale: scale })}>
                {`${Math.round(scale * 100)}%`}
              </Button>
            ))}
          </div>
        </fieldset>
      ) : null}
    </Dialog>
  )
}
