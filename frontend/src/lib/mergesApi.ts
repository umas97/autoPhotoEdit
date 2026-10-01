// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The merges' side of the server (backend/ape/api/routes_merges.py). Nothing
// here merges on its own: every call is a click, and the answer is the card as
// it is after that click -- the job it queued shows up in `preview.state` or
// `merge_job`, and the merged photo when that job has run.
import { request } from './api'
import type { MergeGroupView, MergeKind, MergesView } from './mergeTypes'

type Edit = { photo_ids?: number[]; reference_id?: number | null; options?: Record<string, unknown> }

function post(path: string): Promise<MergeGroupView> {
  return request<MergeGroupView>(path, { method: 'POST' })
}

export const mergesApi = {
  /** Also what runs detection in a project that skipped culling. */
  list: (projectId: number) => request<MergesView>(`/api/projects/${projectId}/merges`),

  /** A group made by hand from photos selected in the grid; its preview is queued. */
  create: (projectId: number, kind: MergeKind, photoIds: number[], referenceId?: number) =>
    request<MergeGroupView>(`/api/projects/${projectId}/merges`, {
      method: 'POST',
      body: JSON.stringify({ kind, photo_ids: photoIds, reference_id: referenceId ?? null }),
    }),

  edit: (groupId: number, edit: Edit) =>
    request<MergeGroupView>(`/api/merges/${groupId}`, {
      method: 'PATCH',
      body: JSON.stringify(edit),
    }),

  preview: (groupId: number) => post(`/api/merges/${groupId}/preview`),
  /** Queue the full merge; also "Riprova" on a failed group. */
  accept: (groupId: number) => post(`/api/merges/${groupId}/accept`),
  reject: (groupId: number) => post(`/api/merges/${groupId}/reject`),
  /** The merged photo leaves, the frames return; a rejection is undone the same way. */
  undo: (groupId: number) => post(`/api/merges/${groupId}/undo`),
}

/**
 * A member's picture, a hundred pixels wide: the camera's 400 px preview when
 * the card has one, the 2048 px proxy only when it does not.
 */
export function memberThumbUrl(member: {
  photo_id: number
  has_thumb: boolean
  has_proxy: boolean
  proxy_rev: string | null
}): string {
  if (member.has_thumb || !member.has_proxy) return `/api/photos/${member.photo_id}/thumb`
  return `/api/photos/${member.photo_id}/proxy${member.proxy_rev ? `?v=${member.proxy_rev}` : ''}`
}

/** Whether a card still has work running, so the screen keeps asking. */
export function busy(view: MergesView | undefined): boolean {
  if (!view) return false
  return (
    view.searching ||
    view.groups.some(
      (group) =>
        group.merge_job !== null ||
        group.preview.state === 'queued' ||
        group.preview.state === 'running',
    )
  )
}
