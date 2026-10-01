// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The removals' side of the server (backend/ape/api/routes_retouch.py): the
// source of a spot, an area taken from a mask, the state of each removal, and
// "Copia punti sulle foto selezionate".
import { request } from './api'
import type { RetouchStates } from './retouchTypes'
import type { EditParams } from './types'

const post = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const retouchApi = {
  /** Where the spot is best copied from, looked for on the frame before item `index`. */
  source: (
    photoId: number,
    params: EditParams,
    index: number,
    spot: { cx: number; cy: number; radius: number },
    avoid: Array<[number, number]> = [],
  ) =>
    request<{ sx: number; sy: number }>(
      `/api/photos/${photoId}/retouch/source`,
      post({ params, index, ...spot, avoid }),
    ),

  /** A raster of what mask `maskIndex` selects now: the area of a new eraser. */
  areaFromMask: (photoId: number, params: EditParams, maskIndex: number) =>
    request<{ name: string }>(
      `/api/photos/${photoId}/retouch/area`,
      post({ params, mask_index: maskIndex }),
    ),

  /** The state of each removal of `params`; the missing fills get queued. */
  states: (photoId: number, params: EditParams) =>
    request<RetouchStates>(`/api/photos/${photoId}/retouch/states`, post(params)),

  retry: (photoId: number) =>
    request<{ queued: boolean }>(`/api/photos/${photoId}/retouch/retry`, { method: 'POST' }),

  copy: (projectId: number, photoId: number, targets: number[], includeErase: boolean) =>
    request<{ queued: number; skipped: number; items: number }>(
      `/api/projects/${projectId}/retouch/copy`,
      post({ photo_id: photoId, targets, include_erase: includeErase }),
    ),
}
