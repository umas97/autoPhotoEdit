// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Review (phase 7): backend/ape/api/routes_review.py. Every decision answers with
// what Ctrl+Z needs to take it back, and the feedback it teaches the profile.
import { proxyUrl, request } from './api'
import type {
  FeedbackSummary,
  PhotoReview,
  ReviewOverview,
  ReviewPhoto,
  UndoRecord,
  WeightCode,
} from './reviewTypes'
import type { StyleSummary } from './styleTypes'
import type { EditParams } from './types'

export const reviewApi = {
  review: (projectId: number) => request<ReviewOverview>(`/api/projects/${projectId}/review`),

  reviewSettings: (
    projectId: number,
    body: { threshold?: number; weights?: Partial<Record<WeightCode, number>> },
  ) =>
    request<ReviewOverview>(`/api/projects/${projectId}/review/settings`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),

  /** `action` is apply | approve | reject; `params` the representative on screen. */
  reviewScene: (
    projectId: number,
    cluster: number,
    action: 'apply' | 'approve' | 'reject',
    params?: EditParams | null,
    /** The representative's version the correction on screen started from. */
    baseVersionId?: number | null,
  ) =>
    request<{ undo: UndoRecord; propagated?: number; approved?: number; queued?: number }>(
      `/api/projects/${projectId}/review/scenes/${cluster}/${action}`,
      {
        method: 'POST',
        body: JSON.stringify({ params: params ?? null, base_version_id: baseVersionId ?? null }),
      },
    ),

  /** The grid's "approve the selected": several scenes as they are, one undo. */
  approveScenes: (projectId: number, clusters: number[]) =>
    request<{ undo: UndoRecord; approved: number; scenes: number }>(
      `/api/projects/${projectId}/review/scenes/approve`,
      { method: 'POST', body: JSON.stringify({ clusters }) },
    ),

  reviewPhoto: (
    projectId: number,
    photoId: number,
    action: 'approve' | 'reject' | 'reset',
    params?: EditParams | null,
  ) =>
    request<{ undo: UndoRecord }>(`/api/projects/${projectId}/review/photos/${photoId}/${action}`, {
      method: 'POST',
      body: JSON.stringify({ params: params ?? null }),
    }),

  reviewUndo: (projectId: number, undo: UndoRecord) =>
    request<{ restored: number }>(`/api/projects/${projectId}/review/undo`, {
      method: 'POST',
      body: JSON.stringify({ undo }),
    }),

  photoReview: (photoId: number) => request<PhotoReview>(`/api/photos/${photoId}/review`),

  feedback: (projectId: number) => request<FeedbackSummary>(`/api/projects/${projectId}/feedback`),

  incorporateFeedback: (projectId: number, name?: string) =>
    request<StyleSummary>(`/api/projects/${projectId}/feedback/incorporate`, {
      method: 'POST',
      body: JSON.stringify({ name: name || null }),
    }),

  discardFeedback: (projectId: number) =>
    request<{ discarded: number }>(`/api/projects/${projectId}/feedback`, { method: 'DELETE' }),
}

/**
 * A review photo as its current version develops it, small; the neutral proxy
 * until the render is ready (backend/ape/review/developed.py).
 */
export function developedUrl(photo: ReviewPhoto): string | null {
  if (photo.developed !== null) return `/api/photos/${photo.id}/developed/${photo.developed}`
  return photo.has_proxy ? proxyUrl(photo) : null
}
