// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The frames behind a merged photo, opened from its badge in the grid
// (section 25.6: "un click espande i membri"). They come from the merges
// screen's own answer, so the grid does not need the hidden frames loaded.
import { useQuery } from '@tanstack/react-query'
import { Star } from 'lucide-react'
import { memberThumbUrl, mergesApi } from '../../lib/mergesApi'
import { t } from '../../i18n/it'

export function MergeMembers({ projectId, groupId }: { projectId: number; groupId: number }) {
  const merges = useQuery({ queryKey: ['merges', projectId], queryFn: () => mergesApi.list(projectId) })
  const group = merges.data?.groups.find((g) => g.id === groupId)
  if (!group) return <p className="px-1 text-[11px] text-ink-400">{t('common.loading')}</p>
  return (
    <ul className="flex gap-1 overflow-x-auto rounded border border-ink-700 bg-ink-850 p-1">
      {group.members.map((member) => (
        <li key={member.photo_id} className="relative w-16 shrink-0" title={member.filename ?? ''}>
          <img src={memberThumbUrl(member)} alt={member.filename ?? ''} loading="lazy"
            className="aspect-[3/2] w-full rounded bg-mat object-contain" />
          {member.reference ? (
            <Star size={9} className="absolute left-0.5 top-0.5 text-ink-50"
              aria-label={t('merges.member.reference')} />
          ) : null}
        </li>
      ))}
    </ul>
  )
}
