// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Style profiles (phase 6): backend/ape/api/routes_styles.py. Creating a profile
// from two folders, its training pairs, and the profile a project develops with.
import { request } from './api'
import type {
  PairingReport,
  PhotoStyle,
  ProjectStyle,
  StyleDetail,
  StyleSample,
  StyleSummary,
} from './styleTypes'
import type { EditParams } from './types'

export const stylesApi = {
  styles: () => request<StyleSummary[]>('/api/styles'),

  style: (id: number) => request<StyleDetail>(`/api/styles/${id}`),

  pairingPreview: (body: { name: string; raw_dir: string; reference_dir: string }) =>
    request<PairingReport>('/api/styles/pairing-preview', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  createStyle: (body: { name: string; raw_dir: string; reference_dir: string; notes?: string }) =>
    request<StyleSummary & { pairing: PairingReport }>('/api/styles', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  editStyle: (id: number, body: { name?: string; notes?: string }) =>
    request<StyleSummary>(`/api/styles/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),

  deleteStyle: (id: number) => request<void>(`/api/styles/${id}`, { method: 'DELETE' }),

  duplicateStyle: (id: number, name: string) =>
    request<StyleSummary>(`/api/styles/${id}/duplicate`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  unpaired: (id: number) =>
    request<{ raws: string[]; references: string[] }>(`/api/styles/${id}/unpaired`),

  addPair: (id: number, raw: string, reference: string) =>
    request<StyleSample>(`/api/styles/${id}/pairs`, {
      method: 'POST',
      body: JSON.stringify({ raw, reference }),
    }),

  excludeSample: (sampleId: number, excluded: boolean) =>
    request<StyleSample>(`/api/styles/samples/${sampleId}`, {
      method: 'PATCH',
      body: JSON.stringify({ excluded }),
    }),

  deleteSample: (sampleId: number) =>
    request<void>(`/api/styles/samples/${sampleId}`, { method: 'DELETE' }),

  importStyle: (file: Blob) =>
    request<StyleSummary>('/api/styles/import', {
      method: 'POST',
      body: file,
      headers: { 'Content-Type': 'application/octet-stream' },
    }),

  projectStyle: (projectId: number) =>
    request<ProjectStyle>(`/api/projects/${projectId}/style`),

  chooseStyle: (projectId: number, profileId: number | null, coherenceLambda?: number) =>
    request<{ queued: number; profile_id: number | null; coherence_lambda: number }>(
      `/api/projects/${projectId}/style`,
      {
        method: 'PUT',
        body: JSON.stringify({ profile_id: profileId, coherence_lambda: coherenceLambda }),
      },
    ),

  photoStyle: (photoId: number) => request<PhotoStyle>(`/api/photos/${photoId}/style`),

  photoStyleParams: (photoId: number, profileId: number) =>
    request<{ profile_id: number; params: EditParams }>(
      `/api/photos/${photoId}/style/params?profile_id=${profileId}`,
    ),
}

/** The 512 px reference a training pair keeps (section 20.1). */
export function sampleThumbUrl(sampleId: number): string {
  return `/api/styles/samples/${sampleId}/thumbnail`
}

/** Where the browser saves a profile from. A download, not a server-side write. */
export function styleExportUrl(profileId: number): string {
  return `/api/styles/${profileId}/export`
}
