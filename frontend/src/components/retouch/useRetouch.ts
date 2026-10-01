// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The removals' state, and the one place that watches their jobs.
//
// No polling (section 26): the progress socket already says which jobs are
// running, four times a second at most and nothing while idle. The screen
// watches it once (`useRetouchWatch`, in the page), and when this photo's
// `retouch_fill` leaves the running list it asks for the states again and
// bumps a revision the preview renders on -- the new fill is resolved by the
// server, the parameters on screen do not change. The panel and the overlay
// read the states from the same query, and the job's progress from the cache.
import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { fillInputs } from '../../lib/retouch'
import { retouchApi } from '../../lib/retouchApi'
import type { RetouchItemState, RetouchStates } from '../../lib/retouchTypes'
import type { EditParams } from '../../lib/types'
import { useProgress } from '../../lib/useProgress'

const FILL_JOB = 'retouch_fill'

/** The page's watcher: returns the revision the preview must follow. */
export function useRetouchWatch(projectId: number | undefined, photoId: number | null): number {
  const queryClient = useQueryClient()
  const { progress } = useProgress(projectId)
  const [revision, setRevision] = useState(0)
  const running = useRef(false)
  const job = photoId === null
    ? undefined
    : progress?.running.find((entry) => entry.kind === FILL_JOB && entry.photo_id === photoId)

  useEffect(() => {
    if (photoId === null) return
    queryClient.setQueryData(['retouchProgress', photoId], job ? job.progress : null)
    if (job) {
      running.current = true
    } else if (running.current) {
      running.current = false
      setRevision((n) => n + 1)
      void queryClient.invalidateQueries({ queryKey: ['retouchStates', photoId] })
    }
  }, [job, photoId, queryClient])

  useEffect(() => {
    running.current = false
  }, [photoId])
  return revision
}

/** The fill job's progress for this photo, 0..1, or null when none runs. */
export function useFillProgress(photoId: number): number | null {
  const query = useQuery<number | null>({
    queryKey: ['retouchProgress', photoId],
    queryFn: () => null,
    enabled: false,
    initialData: null,
  })
  return query.data ?? null
}

/** The state of each removal; asked again whenever what makes a fill changes. */
export function useRetouchStates(photoId: number, params: EditParams): {
  states: RetouchStates | undefined
  byId: Map<string, RetouchItemState>
} {
  const any = params.retouch.length > 0
  const query = useQuery({
    queryKey: ['retouchStates', photoId, fillInputs(params)],
    queryFn: () => retouchApi.states(photoId, params),
    enabled: any,
    // The watcher invalidates it; nothing else makes it stale.
    staleTime: Infinity,
  })
  const byId = new Map((query.data?.items ?? []).map((item) => [item.id, item]))
  return { states: query.data, byId }
}
