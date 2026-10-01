// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The culling screen (section 7) as backend/ape/api/schemas_culling.py
// answers it.
import type { FeatureAvailability } from './analysisTypes'
import type { CullingMode, PhotoStatus } from './types'

/** Every criterion of section 7.3, in the order the interface lists them. */
export type CullCriterion = 'sharpness' | 'motion' | 'exposure' | 'burst' | 'faces' | 'aesthetic'
export type ScoredCriterion = Exclude<CullCriterion, 'burst'>

export type CullReason =
  | 'out_of_focus'
  | 'motion_blur'
  | 'overexposed'
  | 'underexposed'
  | 'burst_duplicate'
  | 'below_target'
  | 'user'

export interface CullSettings {
  mode: CullingMode
  target: number | null
  aggressiveness: number
  weights: Record<ScoredCriterion, number>
  criteria: Record<CullCriterion, boolean>
  threshold: number
}

export interface CullSummary {
  total: number
  selected: number
  culled: number
  analysed: number
  pending: number
  failed: number
  target_count: number | null
}

export interface CullDecision {
  id: number
  culled: boolean
  reasons: CullReason[]
  score: number | null
  rank: number
  criteria: Partial<Record<ScoredCriterion, number>>
  decided_by: 'auto' | 'user' | null
}

export interface CullPhoto extends CullDecision {
  filename: string
  shot_at: string | null
  status: PhotoStatus
  missing: boolean
  error: string | null
  burst_group_id: number | null
  merge_group_id: number | null
  focus_point: [number, number] | null
  width: number | null
  height: number | null
  iso: number | null
  aperture: number | null
  shutter: number | null
  focal_length: number | null
  sharpness: number | null
  motion: number | null
  exposure: number | null
  exposure_side: 'over' | 'under' | null
}

export interface CullMerge {
  id: number
  kind: 'hdr' | 'panorama' | 'focus_stack'
  decision: 'proposed' | 'accepted' | 'rejected' | 'failed'
  confidence: number | null
  reasons: { frames?: number; ev_min?: number; ev_max?: number; span_s?: number }
  members: number[]
  reference: number | null
}

export interface CullView {
  summary: CullSummary
  settings: CullSettings
  availability: Record<'faces' | 'aesthetic', FeatureAvailability>
  photos: CullPhoto[]
  bursts: Record<string, number[]>
  merges: CullMerge[]
  confirmed: boolean
}

export interface CullSelection {
  summary: CullSummary
  settings: CullSettings
  decisions: CullDecision[]
}

export type UserDecision = 'keep' | 'discard' | 'auto'
