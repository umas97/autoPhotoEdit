// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Style profiles (phase 6): backend/ape/api/routes_styles.py.
export type StyleSampleStatus = 'pending' | 'ready' | 'unreproducible' | 'failed'

export interface StyleSummary {
  id: number
  name: string
  notes: string | null
  builtin: boolean
  rules: boolean
  trained: boolean
  usable: boolean
  n_pairs: number
  samples: { pending: number; ready: number; unreproducible: number; failed: number; excluded: number }
  few_pairs: boolean
  min_pairs: number
  recommended_pairs: number
  trained_at: string | null
  stats: {
    methods?: Record<string, number>
    validation?: Record<string, number[]>
    with_embedding?: boolean
  } | null
  raw_dir: string | null
  reference_dir: string | null
}

export interface StyleSample {
  id: number
  raw: string | null
  reference: string | null
  raw_path: string | null
  reference_path: string | null
  pairing: 'name' | 'xmp' | 'time' | 'manual' | null
  status: StyleSampleStatus
  delta_e: number | null
  excluded: boolean
  error: string | null
  has_thumbnail: boolean
  raw_available: boolean
  weight?: number
}

export interface StyleDetail extends StyleSummary {
  sample_list: StyleSample[]
}

export interface PairingReport {
  pairs: number
  methods: Record<'name' | 'xmp' | 'time', number>
  unpaired_references: string[] | number
  raws?: number
}

export interface ProjectStyle {
  proposed: number | null
  current: number | null
  affinity: Record<string, number | null>
  min_affinity: number
  coherence_lambda: number
  applied: { written: number; kept_user_edit: number; unchanged: number; waiting: number }
  pending: number
  status: { styled: number; kept_user_edit: number }
}

export interface PhotoStyle {
  profile: { id: number; name: string } | null
  method: 'model' | 'rules' | null
  neighbours: StyleSample[]
  kept_user_edit: boolean
  applied: boolean
  neutral_profile: string
}
