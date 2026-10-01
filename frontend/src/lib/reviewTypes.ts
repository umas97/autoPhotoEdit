// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Review (phase 7): backend/ape/review/ and api/routes_review.py.
import type { EditParams, PhotoStatus } from './types'

/** The terms of the confidence, review/confidence.py TERMS, plus the user's own. */
export type ReasonCode =
  | 'far'
  | 'ambiguous'
  | 'extrapolation'
  | 'white_balance'
  | 'burnt'
  | 'crushed'
  | 'geometry'
  | 'lens'
  | 'rejected'
  | 'scene_rejected'

export type WeightCode = Exclude<ReasonCode, 'rejected' | 'scene_rejected'>

export type SceneState = 'pending' | 'approved' | 'rejected'

export interface ReviewScene {
  cluster: number
  representative: number
  photos: number[]
  state: SceneState
  queued: number
  approved: number
  min_confidence: number | null
}

export interface ReviewQueueEntry {
  photo_id: number
  confidence: number | null
  reasons: ReasonCode[]
  cluster: number | null
}

export interface ReviewPhoto {
  id: number
  filename: string
  status: PhotoStatus
  confidence: number | null
  cluster_id: number | null
  has_proxy: boolean
  proxy_rev: string | null
  /** The current version, when its developed render is ready (review/developed.py). */
  developed: number | null
}

export interface ReviewOverview {
  profile: { id: number; name: string; builtin: boolean } | null
  pending: number
  /** Developed renders still to come: the screen polls until they are there. */
  developing: number
  threshold: number
  weights: Record<WeightCode, number>
  default_weights: Record<WeightCode, number>
  counts: { total: number; approved: number; queue: number; scenes: number; scenes_done: number }
  scenes: ReviewScene[]
  queue: ReviewQueueEntry[]
  photos: ReviewPhoto[]
  feedback: { proposed: number }
}

export interface ConfidenceTerm {
  code: WeightCode
  risk: number
  weight: number
  value: number | null
}

export type VariantKey =
  | 'neighbours'
  | 'brighter'
  | 'darker'
  | 'neutral'
  | 'camera_wb'
  | 'auto_wb'
  | 'no_rotation'

export interface PhotoReview {
  available: boolean
  photo_id?: number
  confidence?: number
  threshold?: number
  terms?: ConfidenceTerm[]
  reasons?: WeightCode[]
  review?: { decision?: 'approved' | 'rejected'; by?: 'scene' | 'photo'; queued?: boolean } | null
  status?: PhotoStatus
  cluster?: number | null
  variants?: Array<{ key: VariantKey; params: EditParams }>
}

/** What an action hands back so that Ctrl+Z can undo it (review/decisions.py). */
export type UndoRecord = Array<{
  photo_id: number
  version_id: number | null
  review: Record<string, unknown> | null
}>

export interface FeedbackSummary {
  profile: { id: number; name: string; builtin: boolean } | null
  proposed: number
  samples: Array<{ id: number; photo_id: number | null; filename: string | null; has_thumbnail: boolean }>
  suggested_name: string | null
}
