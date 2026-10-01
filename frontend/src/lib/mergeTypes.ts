// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The merges screen as backend/ape/api/routes_merges.py answers it (section
// 25.6): one card per group, with its members, the reasons it was proposed, and
// where its preview and its full merge are.
export type MergeKind = 'hdr' | 'panorama' | 'focus_stack'
export type MergeDecision = 'proposed' | 'accepted' | 'rejected' | 'failed'

/** The badge of a merged photo in the grid: "HDR · 3". */
export interface MergeBadge {
  kind: MergeKind
  sources: number
}

export interface MergeMember {
  photo_id: number
  position: number
  /** Stops from the reference frame (HDR); null where it means nothing. */
  ev_offset: number | null
  reference: boolean
  filename: string | null
  missing: boolean
  has_thumb: boolean
  has_proxy: boolean
  proxy_rev: string | null
  shot_at: string | null
}

/** What detection saw, in the terms the card spells out (backend/ape/merge/detect*.py). */
export interface MergeReasons {
  manual?: boolean
  frames?: number
  span_s?: number
  ev_min?: number
  ev_max?: number
  step_ev?: number
  camera_bracket?: boolean
  focus_positions?: boolean
  focus_moves?: boolean
  selective_share?: number
  overlap_min?: number
  overlap_max?: number
  /** Panorama shot as a burst while turning: neighbours overlap past 60%. */
  sweep?: boolean
}

/**
 * The preview's state: `none` never asked, `queued`/`running` its job, `ready`
 * for the group as it is now, `stale` for the group before an edit, `error` a
 * merge that could not be made.
 */
export type MergePreviewState = 'none' | 'queued' | 'running' | 'ready' | 'stale' | 'error'

export interface MergeReport {
  residual_px?: number
  misaligned?: boolean
  ghost_fraction?: number
  /** Focus stack: share of the frame no frame had in focus. */
  uncovered_fraction?: number
  /** Panorama: the projection used, how wide it is, what it produces. */
  projection?: 'cylindrical' | 'spherical' | 'plane'
  /** The sweep tilts instead of turning: the span that matters is the height. */
  vertical?: boolean
  span_deg?: [number, number]
  /** Panorama: the members, and the ones stitched (a sweep needs only a few). */
  members?: number
  used?: string[]
  output?: { width: number; height: number; megapixels: number; scale: number }
  ram_estimate_mb?: number
  /** Preview only: the full-resolution output, and the refusal it would meet. */
  full?: { width: number; height: number; megapixels: number; ram_estimate_mb: number }
  too_large?: string
  /** Largest rectangle without empty borders, 0..1: proposed, never applied. */
  crop?: { x: number; y: number; width: number; height: number } | null
  [key: string]: unknown
}

export interface MergeJobState {
  state: 'queued' | 'running'
  progress: number
}

export interface MergeGroupView {
  id: number
  kind: MergeKind
  decision: MergeDecision
  confidence: number | null
  reasons: MergeReasons
  options: Record<string, unknown>
  members: MergeMember[]
  preview: {
    state: MergePreviewState
    url: string | null
    coverage_url: string | null
    error: string | null
    report: MergeReport
  }
  merge_job: MergeJobState | null
  full_report: MergeReport | null
  error: string | null
  /** The merged photo, once the full merge has run and while it stands. */
  result_photo_id: number | null
}

export interface MergesView {
  /** Project.merge_detection_enabled: detection only, never execution. */
  enabled: boolean
  /** The panorama search is queued or running. */
  searching: boolean
  groups: MergeGroupView[]
}
