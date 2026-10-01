// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The masks' side of the server (backend/ape/api/routes_masks.py): painted
// rasters up and down, and the selection of one mask as the preview frames it.
import { ApiError, request } from './api'
import type { SegmentSubject } from './maskTypes'
import type { EditParams } from './types'

async function failure(response: Response): Promise<ApiError> {
  let detail = `errore ${response.status}`
  try {
    const payload = (await response.json()) as { detail?: string }
    if (payload.detail) detail = payload.detail
  } catch {
    /* not JSON */
  }
  return new ApiError(response.status, detail)
}

/** Store a painted mask (a PNG whose alpha is the selection); returns its name. */
export async function uploadRaster(png: Blob): Promise<string> {
  let response: Response
  try {
    response = await fetch('/api/masks/rasters', {
      method: 'POST',
      headers: { 'Content-Type': 'image/png' },
      body: png,
    })
  } catch {
    throw new ApiError(0, 'il server non risponde')
  }
  if (!response.ok) throw await failure(response)
  return ((await response.json()) as { name: string }).name
}

/** A stored raster: one channel, 8 bit. Immutable, since the name is its content. */
export function rasterUrl(name: string): string {
  return `/api/masks/rasters/${name}`
}

/** What mask `index` selects, straightened and cropped as the preview: red, alpha = selection. */
export async function maskSelection(
  photoId: number,
  index: number,
  params: EditParams,
  longEdge: number,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch(
    `/api/photos/${photoId}/masks/${index}/selection?long_edge=${longEdge}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal,
    },
  )
  if (!response.ok) throw await failure(response)
  return response.blob()
}

export type SegmentSubjectState = {
  subject: SegmentSubject
  state: 'none' | 'queued' | 'running' | 'ready' | 'failed'
  raster: string | null
  error: string | null
  /** The subject's model is downloaded and verified. */
  available: boolean
  reason: string | null
}

/** Which subjects have been found on this photo, and whether their models are here. */
export function segments(photoId: number): Promise<SegmentSubjectState[]> {
  return request<SegmentSubjectState[]>(`/api/photos/${photoId}/segments`)
}

/** Find a subject: queued, unless it has been found already. */
export function findSegment(photoId: number, subject: SegmentSubject): Promise<SegmentSubjectState> {
  return request<SegmentSubjectState>(`/api/photos/${photoId}/segments/${subject}`, { method: 'POST' })
}

/** The model each subject needs (backend/ape/analysis/segment.py). */
export const SUBJECT_FEATURE: Record<SegmentSubject, string> = {
  sky: 'segment_sky',
  person: 'segment_person',
  skin: 'segment_person',
}
