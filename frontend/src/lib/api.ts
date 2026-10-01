// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// One place that talks to the server, so that error handling, JSON decoding and
// the shape of a request are decided once.
//
// Two things here are not boilerplate.
//
// `ApiError` carries the server's Italian `detail` message. The API is written
// to answer with a sentence a person can read (backend/ape/api/app.py), and the
// interface shows that sentence rather than inventing its own: a wrong folder
// should say which folder, and only the server knows.
//
// `renderPreview` is the hot path of section 10. It posts the parameters and
// gets back a JPEG, and it takes an AbortSignal because a slider being dragged
// produces requests faster than the server answers them -- the useful reply is
// always the last one, and the ones before it are cancelled rather than queued.
//
// The screens with many calls of their own keep them beside this file, on
// top of `request`: stylesApi, reviewApi, exportApi, mergesApi, masksApi and
// foldersApi.
import type {
  CropProposal,
  LensCandidate,
  LensEntry,
  LensfunState,
  ModelState,
  ScenesView,
} from './analysisTypes'
import type { CullSelection, CullSettings, CullView, UserDecision } from './cullTypes'
import type {
  DiagnosticsEntry,
  EditParams,
  ImportResult,
  JobSummary,
  Photo,
  PhotoDetail,
  PhotoPage,
  Problems,
  Project,
  ScanPreview,
  Snapshot,
  StorageOverview,
  WindowInfo,
} from './types'

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch (error) {
    // A request cancelled on purpose -- superseded by a newer one -- is not a
    // server that stopped answering, and must not be reported as one.
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    // The server is down, or the socket died mid-request. Everything above this
    // line treats it as one condition: the program is not answering. Status 0 is
    // the marker -- no HTTP status exists for a request that never arrived.
    throw new ApiError(0, 'il server non risponde')
  }

  if (!response.ok) {
    let detail = `errore ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string | unknown }
      if (typeof payload.detail === 'string') detail = payload.detail
      else if (payload.detail) detail = JSON.stringify(payload.detail)
    } catch {
      /* a body that is not JSON tells us nothing more than the status did */
    }
    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  health: () => request<{ status: string; version: string; workers: number }>('/api/health'),

  listProjects: () => request<Project[]>('/api/projects'),

  createProject: (body: {
    name: string
    source_dir: string
    output_dir?: string | null
    import_now: boolean
  }) => request<Project>('/api/projects', { method: 'POST', body: JSON.stringify(body) }),

  readProject: (id: number) => request<Project>(`/api/projects/${id}`),

  deleteProject: (id: number) =>
    request<void>(`/api/projects/${id}`, { method: 'DELETE' }),

  scan: (id: number, folder?: string) =>
    request<ScanPreview>(
      `/api/projects/${id}/scan${folder ? `?folder=${encodeURIComponent(folder)}` : ''}`,
    ),

  importFolder: (
    id: number,
    body: {
      folder?: string | null
      mark_missing?: boolean
      build_proxies?: boolean
      /** Section 7.1: the explicit choice, cull first or straight to editing. */
      culling?: boolean | null
    },
  ) =>
    request<ImportResult>(`/api/projects/${id}/import`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  listPhotos: (projectId: number, offset = 0, limit = 200, includeCulled = true, includeSources = false) =>
    request<PhotoPage>(
      `/api/projects/${projectId}/photos?offset=${offset}&limit=${limit}` +
        `&include_culled=${includeCulled}&include_sources=${includeSources}`,
    ),

  culling: (projectId: number) => request<CullView>(`/api/projects/${projectId}/culling`),

  startCulling: (projectId: number, force = false) =>
    request<{ queued: number }>(`/api/projects/${projectId}/culling/start`, {
      method: 'POST',
      body: JSON.stringify({ force }),
    }),

  cullDecisions: (projectId: number, decisions: Array<{ photo_id: number; decision: UserDecision }>) =>
    request<CullSelection>(`/api/projects/${projectId}/culling/decisions`, {
      method: 'POST',
      body: JSON.stringify({ decisions }),
    }),

  confirmCulling: (projectId: number) =>
    request<{ queued: number; selected: number }>(`/api/projects/${projectId}/culling/confirm`, {
      method: 'POST',
    }),

  readPhoto: (id: number) => request<PhotoDetail>(`/api/photos/${id}`),

  saveParams: (id: number, params: EditParams, source = 'user_edited', keepalive = false) =>
    request<PhotoDetail>(`/api/photos/${id}/params`, {
      method: 'PUT',
      body: JSON.stringify({ params, source }),
      keepalive,
    }),

  restoreVersion: (photoId: number, versionId: number) =>
    request<PhotoDetail>(`/api/photos/${photoId}/versions/${versionId}/restore`, {
      method: 'POST',
    }),

  // --- section 19: problems and diagnostics ------------------------------------

  problems: (projectId?: number) =>
    request<Problems>(`/api/problems${projectId ? `?project_id=${projectId}` : ''}`),

  /** Both lists absent: everything the panel shows. */
  retryProblems: (body: {
    photo_ids?: number[]
    job_ids?: number[]
    merge_ids?: number[]
    project_id?: number
  }) =>
    request<{ retried: number }>('/api/problems/retry', { method: 'POST', body: JSON.stringify(body) }),

  diagnosticsPlan: () => request<DiagnosticsEntry[]>('/api/diagnostics/plan'),

  // --- section 20.3: storage ------------------------------------------------------

  storage: () => request<StorageOverview>('/api/storage'),

  clearCache: () => request<{ removed: number; freed: number }>('/api/storage/clear', { method: 'POST' }),

  // --- section 23.2: snapshots ---------------------------------------------------

  snapshots: (projectId: number) => request<Snapshot[]>(`/api/projects/${projectId}/snapshots`),

  takeSnapshot: (projectId: number, name: string) =>
    request<{ id: number; name: string }>(`/api/projects/${projectId}/snapshots`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  /** `undo` is the snapshot of the state replaced: restoring it is the undo. */
  restoreSnapshot: (projectId: number, snapshotId: number) =>
    request<{ restored: number; missing: number; undo: number }>(
      `/api/projects/${projectId}/snapshots/${snapshotId}/restore`,
      { method: 'POST' },
    ),

  deleteSnapshot: (projectId: number, snapshotId: number) =>
    request<void>(`/api/projects/${projectId}/snapshots/${snapshotId}`, { method: 'DELETE' }),

  // --- phase 5: scenes, lenses, crop proposals, downloads -------------------

  scenes: (projectId: number) => request<ScenesView>(`/api/projects/${projectId}/scenes`),

  queueAnalysis: (projectId: number, embeddings = false) =>
    request<{ queued: number }>(
      `/api/projects/${projectId}/analysis?embeddings=${embeddings}`,
      { method: 'POST' },
    ),

  lenses: (projectId: number) =>
    request<{ lenses: LensEntry[]; lensfun: LensfunState }>(`/api/projects/${projectId}/lenses`),

  searchLenses: (query: string) =>
    request<LensCandidate[]>(`/api/lenses/search?q=${encodeURIComponent(query)}`),

  setLensOverride: (lens: string, maker: string, model: string) =>
    request<{ queued: number }>('/api/lenses/override', {
      method: 'PUT',
      body: JSON.stringify({ lens, maker, model }),
    }),

  clearLensOverride: (lens: string) =>
    request<{ queued: number }>(`/api/lenses/override?lens=${encodeURIComponent(lens)}`, {
      method: 'DELETE',
    }),

  lensfun: () => request<LensfunState>('/api/lensfun'),

  updateLensfun: () => request<LensfunState>('/api/lensfun/update', { method: 'POST' }),

  models: () => request<ModelState[]>('/api/models'),

  downloadModel: (feature: string) =>
    request<ModelState>(`/api/models/${feature}/download`, { method: 'POST' }),

  cancelModelDownload: (feature: string) =>
    request<ModelState>(`/api/models/${feature}/download`, { method: 'DELETE' }),

  applyCrop: (proposal: Pick<CropProposal, 'id'>) =>
    request<PhotoDetail>(`/api/crop-proposals/${proposal.id}/apply`, { method: 'POST' }),

  rejectCrop: (proposal: Pick<CropProposal, 'id'>) =>
    request<PhotoDetail>(`/api/crop-proposals/${proposal.id}/reject`, { method: 'POST' }),

  resumeCropProposals: (projectId: number) =>
    request<{ paused: boolean }>(`/api/projects/${projectId}/crop-proposals/resume`, {
      method: 'POST',
    }),

  jobSummary: (projectId?: number) =>
    request<JobSummary>(
      `/api/jobs/summary${projectId ? `?project_id=${projectId}` : ''}`,
    ),

  activeJobs: () => request<{ active: number }>('/api/jobs/active'),

  window: () => request<WindowInfo>('/api/window'),

  quit: () => request<{ detail: string }>('/api/app/quit', { method: 'POST' }),

  settings: () => request<Record<string, unknown>>('/api/settings'),

  writeSetting: (key: string, value: unknown) =>
    request<Record<string, unknown>>(`/api/settings/${encodeURIComponent(key)}`, {
      method: 'PUT',
      body: JSON.stringify({ value }),
    }),
}

/**
 * The cached browsing JPEG. A plain URL: the browser caches it, we do not. The
 * revision is what makes a regenerated proxy a different URL -- the server
 * marks the response immutable.
 */
export function proxyUrl(photo: Pick<Photo, 'id'> & { proxy_rev?: string | null }): string {
  return `/api/photos/${photo.id}/proxy${photo.proxy_rev ? `?v=${photo.proxy_rev}` : ''}`
}

/**
 * The camera's own embedded preview (section 7.2): what culling shows, and what
 * the grid falls back to before a proxy exists. `full` is the 1616 px preview
 * the comparison view zooms into; `grid` a 400 px copy for the dense grid.
 */
export function thumbUrl(photo: Pick<Photo, 'id'>, size: 'grid' | 'full' = 'grid'): string {
  return `/api/photos/${photo.id}/thumb${size === 'full' ? '?size=full' : ''}`
}

/**
 * Change the culling settings. Cancellable, like a preview: the aggressiveness
 * slider sends one of these per step, and only the last answer matters.
 */
export async function updateCullSettings(
  projectId: number,
  settings: Partial<Omit<CullSettings, 'threshold'>>,
  signal?: AbortSignal,
): Promise<CullSelection> {
  return request<CullSelection>(`/api/projects/${projectId}/culling/settings`, {
    method: 'PUT',
    body: JSON.stringify(settings),
    signal,
  })
}

/**
 * Develop a photo with these parameters, right now, at this size.
 *
 * Returns the JPEG as a blob. Section 10: 1024 px while a slider is moving,
 * 2048 px when it is released, and never the full-resolution RAW.
 */
export async function renderPreview(
  photoId: number,
  params: EditParams,
  longEdge: number,
  signal?: AbortSignal,
  retouch = true,
): Promise<Blob> {
  const query = `long_edge=${longEdge}${retouch ? '' : '&retouch=false'}`
  const response = await fetch(`/api/photos/${photoId}/preview?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
    signal,
  })
  if (!response.ok) {
    let detail = `errore ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) detail = payload.detail
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail)
  }
  return response.blob()
}

/**
 * Tell the server what to do with running jobs while the window closes.
 *
 * `sendBeacon` because the document is being torn down: a fetch would be
 * cancelled halfway (section 21.3).
 */
export function sendCloseIntent(decision: 'continue' | 'pause' | 'stop', remember: boolean): void {
  const payload = new Blob([JSON.stringify({ decision, remember })], {
    type: 'application/json',
  })
  navigator.sendBeacon('/api/window/close-intent', payload)
}

