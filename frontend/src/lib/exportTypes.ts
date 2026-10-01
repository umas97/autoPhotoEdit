// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Export (phase 8): backend/ape/export/settings.py and api/routes_export.py.
export type ExportFormat = 'jpeg' | 'tiff8' | 'tiff16'
export type ExportConflict = 'ask' | 'rename' | 'overwrite' | 'skip'
export type OutputSpace = 'srgb' | 'display-p3' | 'adobe-rgb' | 'rec2020'
export type OutputSharpening = 'none' | 'low' | 'standard' | 'high'

export interface ExportSettings {
  output_dir: string | null
  template: string
  on_conflict: ExportConflict
  strip_gps: boolean
  images: boolean
  format: ExportFormat
  quality: number
  output_space: OutputSpace
  long_edge: number | null
  sharpening: OutputSharpening
  dither: boolean
  xmp_darktable: boolean
  xmp_adobe: boolean
  xmp_beside_raw: boolean
  only_approved: boolean
}

export interface ExportConflictInfo {
  photo_id: number
  filename: string
  files: string[]
}

export interface ExportPlan {
  ok: boolean
  error?: string
  count: number
  names: Array<{ photo_id: number; filename: string; name: string }>
  conflicts: ExportConflictInfo[]
  warnings?: string[]
}

export type ExportBatchState = 'running' | 'paused' | 'done' | 'cancelled'

export interface ExportBatch {
  id: number
  state: ExportBatchState
  total: number
  counts: { queued: number; done: number; skipped: number; failed: number; cancelled: number }
  outcomes: Record<string, number>
  output_dir: string | null
  created_at: string
  finished_at: string | null
  problems?: Array<{
    item_id: number
    photo_id: number
    filename: string
    state: 'failed' | 'skipped'
    error: string | null
  }>
}

export interface ExportScreen {
  settings: ExportSettings
  plan: ExportPlan
  batches: ExportBatch[]
  artist: string
  copyright: string
  source_dir: string
}
