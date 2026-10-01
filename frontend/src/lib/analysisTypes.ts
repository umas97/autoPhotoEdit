// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Phase 5 as backend/ape/api/routes_analysis.py answers it: straightening,
// lens profiles, crop proposals, scenes, and the optional models and data the
// program downloads only when asked.
export interface LensProfile {
  maker: string
  model: string
  /** `applied`: corrected before these pixels existed (a panorama's frames). */
  source: 'auto' | 'override' | 'applied'
}

export type StraightenOutcome = 'level' | 'rotated' | 'no_lines' | 'contradictory' | 'too_large'

export interface PhotoAnalysis {
  version?: number
  straighten?: {
    rotation_deg: number
    measured_deg: number | null
    confidence: number
    outcome: StraightenOutcome
  }
  /** Absent: not analysed yet. Null: the lens has no lensfun profile. */
  lens?: LensProfile | null
  lens_model?: string | null
  as_shot?: { temperature_k: number; tint: number } | null
}

export interface CropRect {
  x: number
  y: number
  width: number
  height: number
}

export interface CropProposal {
  id: number
  rect: CropRect
  aspect: string | null
  score: number | null
  decision: string
}

export interface SceneCluster {
  cluster: number
  representative: number
  photos: number[]
}

export interface ScenesView {
  clusters: SceneCluster[]
  clustered: number
  total: number
  pending: number
  basis: 'embedding' | 'features'
  embedding: { available: boolean; reason: FeatureAvailability['reason'] }
}

export interface DownloadState {
  state: 'idle' | 'running' | 'done' | 'failed' | 'cancelled'
  fraction: number | null
  error: string | null
}

export interface ModelState {
  name: string
  feature: string
  licence: string
  size_mb: number
  url: string
  notice: string | null
  available: boolean
  reason: FeatureAvailability['reason']
  downloadable: boolean
  download: DownloadState | null
}

export interface LensfunState {
  available: boolean
  source: 'bundled' | 'updated' | null
  updated_at: string | null
  lenses: number
  update: DownloadState | null
}

export interface LensEntry {
  lens: string | null
  photos: number
  analysed: number
  profile: LensProfile | null
  override: { maker: string; model: string } | null
}

export interface LensCandidate {
  maker: string
  model: string
  min_focal: number
  max_focal: number
}

export interface FeatureAvailability {
  available: boolean
  reason: 'runtime_missing' | 'not_pinned' | 'model_missing' | 'checksum_mismatch' | null
  licence: string | null
  notice: string | null
  size_mb: number | null
}
