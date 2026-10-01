// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Export (phase 8): backend/ape/api/routes_export.py. The screen, the name a
// template gives, and the batches with their pause, resume and retry.
import { ApiError, request } from './api'
import type {
  ExportBatch,
  ExportConflict,
  ExportConflictInfo,
  ExportScreen,
  ExportSettings,
} from './exportTypes'

export const exportApi = {
  exportScreen: (projectId: number) =>
    request<ExportScreen>(`/api/projects/${projectId}/export`),

  updateExport: (projectId: number, changes: Partial<ExportSettings>) =>
    request<ExportScreen>(`/api/projects/${projectId}/export`, {
      method: 'PATCH',
      body: JSON.stringify(changes),
    }),

  exportName: (projectId: number, template: string) =>
    request<{ name: string | null; error: string | null; filename?: string }>(
      `/api/projects/${projectId}/export/name?template=${encodeURIComponent(template)}`,
    ),

  startExport: (
    projectId: number,
    body: {
      answers?: Array<{ photo_id: number; policy: ExportConflict }>
      apply_to_all?: ExportConflict | null
    },
  ) =>
    request<ExportBatch>(`/api/projects/${projectId}/export`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  exportAction: (batchId: number, action: 'pause' | 'resume' | 'cancel' | 'retry') =>
    request<ExportBatch>(`/api/exports/${batchId}/${action}`, {
      method: 'POST',
      body: action === 'retry' ? JSON.stringify({}) : undefined,
    }),
}

/**
 * The photos a start was refused for, when the policy is "ask" (HTTP 409).
 *
 * The server's ``detail`` is an object there, which ``request`` has already
 * turned into the error's message as JSON.
 */
export function exportConflicts(error: unknown): ExportConflictInfo[] | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null
  try {
    const detail = JSON.parse(error.message) as { conflicts?: ExportConflictInfo[] }
    return detail.conflicts ?? null
  } catch {
    return null
  }
}
