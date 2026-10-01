// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The shapes the server speaks in, mirroring backend/ape/api/schemas.py field
// for field, and EditParams from backend/ape/pipeline/params.py.
//
// `npm run api:types` regenerates src/lib/openapi.d.ts from the live server's
// OpenAPI document; these declarations are the hand-written surface the
// components use, kept small on purpose. When the two disagree, schemas.py is
// right -- it is the one the server actually validates against.
//
// The screens with many shapes of their own keep them beside this file:
// analysisTypes, cullTypes, styleTypes, reviewTypes, exportTypes, mergeTypes
// and maskTypes.
import type { CropProposal, CropRect, PhotoAnalysis } from './analysisTypes'
import type { MaskParams } from './maskTypes'
import type { MergeBadge, MergeKind } from './mergeTypes'
import type { RetouchItem } from './retouchTypes'

// backend/ape/db/enums.py, value for value.
export type ProjectStatus =
  | 'new'
  | 'importing'
  | 'culling'
  | 'analyzing'
  | 'reviewing'
  | 'exporting'
  | 'done'

export type PhotoStatus =
  | 'imported'
  | 'culled'
  | 'analyzed'
  | 'predicted'
  | 'needs_review'
  | 'approved'
  | 'exported'
  | 'failed'

export type CullingMode = 'conservative' | 'target_percent' | 'target_count'

export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface Project {
  id: number
  name: string
  source_dir: string
  output_dir: string | null
  status: ProjectStatus
  source_missing: boolean
  culling_enabled: boolean
  culling_mode: CullingMode
  culling_target: number | null
  culling_aggressiveness: number
  merge_detection_enabled: boolean
  created_at: string
  updated_at: string
  photo_count: number
  missing_count: number
  pending_jobs: number
}

export interface Photo {
  id: number
  filename: string
  status: PhotoStatus
  missing: boolean
  culled: boolean
  width: number | null
  height: number | null
  shot_at: string | null
  camera: string | null
  lens: string | null
  iso: number | null
  aperture: number | null
  shutter: number | null
  focal_length: number | null
  culling_score: number | null
  burst_group_id: number | null
  cluster_id: number | null
  cluster_rank: number | null
  has_proxy: boolean
  /** Changes with the proxy file; part of its URL, which is cached immutable. */
  proxy_rev: string | null
  /** The camera's embedded preview can be served, even before any proxy. */
  has_thumb: boolean
  error: string | null
  /** A frame an accepted merge stands in for (shown only with "Mostra scatti sorgente"). */
  superseded: boolean
  merge_group_id: number | null
  /** For a merged photo, the badge of section 25.6: kind and number of sources. */
  merge: MergeBadge | null
}

export interface EditVersion {
  id: number
  parent_version_id: number | null
  params_version: number
  source: string
  created_at: string
  is_current: boolean
}

export type VersionSource = 'predicted' | 'user_edited' | 'cluster_applied' | 'reverted' | 'snapshot_restored'

/** A project snapshot (section 23.2, backend/ape/snapshots.py). */
export interface Snapshot {
  id: number
  name: string
  kind: 'manual' | 'auto'
  created_at: string
  photos: number
  /** Photos whose state differs from now: what a rollback would change. */
  differs: number
}

export interface PhotoDetail extends Photo {
  path: string | null
  duplicate_paths: string[] | null
  params: EditParams | null
  current_version_id: number | null
  versions: EditVersion[]
  analysis: PhotoAnalysis | null
  crop_proposal: CropProposal | null
  crop_proposals_paused: boolean
}

export interface PhotoPage {
  items: Photo[]
  total: number
  offset: number
  limit: number
}

export interface ScanPreview {
  folder: string
  raws: number
  sidecars: number
  subdirectories: string[]
  rejected_formats: Record<string, number>
  other_files: number
}

export interface ImportResult {
  imported: number
  already_present: number
  duplicates: number
  restored: number
  marked_missing: number
  queued_jobs: number
  summary: string
}

export interface JobSummary {
  counts: Record<string, number>
  active: number
  total: number
  progress: number
}

export interface ProgressMessage {
  type: 'progress'
  project_id: number | null
  counts: Record<JobState, number>
  active: number
  total: number
  progress: number
  photos: number
  photos_with_proxy: number
  photos_analysed: number
  running: Array<{ id: number; kind: string; progress: number; photo_id: number | null }>
  /** Section 19: failed photos and failed jobs, the badge that is always visible. */
  problems: number
}

/** The Problems panel (backend/ape/problems.py). */
export interface ProblemPhoto {
  photo_id: number
  project_id: number
  project: string | null
  filename: string
  reason: string
  stage: string | null
  job_id: number | null
  details: string | null
  at: string | null
}

export interface ProblemJob {
  job_id: number
  project_id: number | null
  project: string | null
  photo_id: number | null
  stage: string
  reason: string
  details: string | null
  at: string | null
}

/** A merge in state "failed" (section 25.6): its reason is the group's own. */
export interface ProblemMerge {
  group_id: number
  project_id: number
  project: string | null
  kind: MergeKind
  frames: number
  reason: string
}

export interface Problems {
  photos: ProblemPhoto[]
  jobs: ProblemJob[]
  merges: ProblemMerge[]
  count: number
}

/** One file of the diagnostics bundle, shown before the zip exists. */
export interface DiagnosticsEntry {
  name: string
  description: string
  size: number
  preview: string
}

export type WindowMode = 'pwa' | 'app' | 'tab' | 'none'

export interface WindowInfo {
  mode: WindowMode
  browser: string | null
  window_pid: number | null
  alive: boolean
  note: string | null
  url: string | null
  can_quit: boolean
  on_window_close: 'continue' | 'pause' | 'stop' | null
}

// --- EditParams (backend/ape/pipeline/params.py) ------------------------------

export interface WhiteBalanceParams {
  mode: 'as_shot' | 'custom'
  temperature_k: number
  tint: number
}

export interface HSLBand {
  hue: number
  saturation: number
  luminance: number
}

export type HSLBandName =
  | 'red'
  | 'orange'
  | 'yellow'
  | 'green'
  | 'aqua'
  | 'blue'
  | 'purple'
  | 'magenta'

export interface EditParams {
  params_version: number
  white_balance: WhiteBalanceParams
  exposure: { ev: number }
  highlight_recovery: { strength: number; threshold: number }
  noise: { luminance: number; chrominance: number; radius: number }
  tone: {
    black_point_ev: number
    white_point_ev: number
    contrast: number
    pivot: number
    toe: number
    shoulder: number
    chroma_preservation: number
  }
  tone_shaping: { shadows: number; highlights: number; whites: number; blacks: number }
  tone_curve: {
    highlights: number
    lights: number
    darks: number
    shadows: number
    points: Array<[number, number]>
  }
  color: {
    saturation: number
    vibrance: number
    hsl: Partial<Record<HSLBandName, HSLBand>>
    split_toning: {
      shadow_hue: number
      shadow_saturation: number
      highlight_hue: number
      highlight_saturation: number
      balance: number
    }
  }
  local_contrast: { clarity: number; radius: number; shadows: number; highlights: number }
  sharpen: { amount: number; radius: number; threshold: number }
  geometry: { lens_correction: boolean; rotation_deg: number; crop: CropRect | null }
  masks: MaskParams[]
  /** The removals, in the order they apply (lib/retouchTypes.ts). */
  retouch: RetouchItem[]
}

/** Disk use by category and the cache quota (backend/ape/cache.py, section 20.3). */
export type StorageCategory =
  | 'proxies' | 'previews' | 'developed' | 'stages' | 'merges' | 'intermediates' | 'masks' | 'models'

export interface StorageUse {
  bytes: number
  files: number
  evictable: boolean
}

export interface StorageOverview {
  limit_bytes: number
  /** Where the painted masks live: the backup must carry them along. */
  masks_dir: string
  /** Where the magic eraser's fills live: the same. */
  retouch_dir: string
  limit_gb: number
  cache_bytes: number
  categories: Record<StorageCategory, StorageUse>
  clear: {
    frees_bytes: number
    merges_to_rebuild: number
    categories: Partial<Record<StorageCategory, StorageUse>>
    keeps: Partial<Record<StorageCategory, StorageUse>>
  }
}
