// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The parameter panel, described as data.
//
// Every control here carries the same range the pydantic field carries in
// backend/ape/pipeline/params.py. That duplication is deliberate and it is the
// cheap half of a bargain: the server validates and would refuse an out-of-range
// value with a 400, but a slider that can reach a value the server rejects is a
// slider that breaks under the user's hand. The ranges are checked against the
// schema by tests/test_frontend_contract.py, so they cannot drift quietly.
//
// Nothing here knows how to render a control: the panel reads this table and
// the table describes meaning, not layout.
import type { StringKey } from '../i18n/it'
import type { HSLBandName } from './types'

export interface Control {
  /** Dotted path into EditParams, e.g. 'tone.contrast'. */
  path: string
  label: StringKey
  min: number
  max: number
  step: number
  /** Where the slider snaps back to, and what "neutral" looks like. */
  neutral: number
  /** How many decimals to show. */
  digits?: number
  /** Appended to the displayed number. */
  unit?: string
}

export interface Group {
  key: string
  label: StringKey
  controls: Control[]
  /** Open when the panel is first shown. The rest start folded. */
  open?: boolean
}

const unit = (path: string, label: StringKey, neutral = 0): Control => ({
  path,
  label,
  min: -1,
  max: 1,
  step: 0.01,
  neutral,
  digits: 2,
})

export const HSL_BANDS: HSLBandName[] = [
  'red',
  'orange',
  'yellow',
  'green',
  'aqua',
  'blue',
  'purple',
  'magenta',
]

export const GROUPS: Group[] = [
  {
    key: 'white_balance',
    label: 'params.group.white_balance',
    open: true,
    controls: [
      {
        path: 'white_balance.temperature_k',
        label: 'params.white_balance.temperature_k',
        min: 1500,
        max: 25000,
        step: 10,
        neutral: 5500,
        digits: 0,
        unit: ' K',
      },
      {
        path: 'white_balance.tint',
        label: 'params.white_balance.tint',
        min: -150,
        max: 150,
        step: 1,
        neutral: 0,
        digits: 0,
      },
    ],
  },
  {
    key: 'exposure',
    label: 'params.group.exposure',
    open: true,
    controls: [
      {
        path: 'exposure.ev',
        label: 'params.exposure.ev',
        min: -6,
        max: 6,
        step: 0.01,
        neutral: 0,
        digits: 2,
        unit: ' EV',
      },
    ],
  },
  {
    key: 'tone',
    label: 'params.group.tone',
    open: true,
    controls: [
      {
        path: 'tone.contrast',
        label: 'params.tone.contrast',
        min: 0.3,
        max: 3,
        step: 0.01,
        neutral: 1.2,
        digits: 2,
      },
      {
        path: 'tone.pivot',
        label: 'params.tone.pivot',
        min: -3,
        max: 3,
        step: 0.01,
        neutral: 0,
        digits: 2,
        unit: ' EV',
      },
      {
        path: 'tone.black_point_ev',
        label: 'params.tone.black_point_ev',
        min: -16,
        max: -1,
        step: 0.1,
        neutral: -8,
        digits: 1,
        unit: ' EV',
      },
      {
        path: 'tone.white_point_ev',
        label: 'params.tone.white_point_ev',
        min: 1,
        max: 12,
        step: 0.1,
        neutral: 4.5,
        digits: 1,
        unit: ' EV',
      },
      { path: 'tone.toe', label: 'params.tone.toe', min: 0.2, max: 6, step: 0.05, neutral: 1.2, digits: 2 },
      {
        path: 'tone.shoulder',
        label: 'params.tone.shoulder',
        min: 0.2,
        max: 6,
        step: 0.05,
        neutral: 1.5,
        digits: 2,
      },
      {
        path: 'tone.chroma_preservation',
        label: 'params.tone.chroma_preservation',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0.4,
        digits: 2,
      },
    ],
  },
  {
    key: 'tone_shaping',
    label: 'params.group.tone_shaping',
    open: true,
    controls: [
      unit('tone_shaping.shadows', 'params.tone_shaping.shadows'),
      unit('tone_shaping.highlights', 'params.tone_shaping.highlights'),
      unit('tone_shaping.whites', 'params.tone_shaping.whites'),
      unit('tone_shaping.blacks', 'params.tone_shaping.blacks'),
    ],
  },
  {
    key: 'color',
    label: 'params.group.color',
    open: true,
    controls: [
      unit('color.saturation', 'params.color.saturation'),
      unit('color.vibrance', 'params.color.vibrance'),
    ],
  },
  {
    key: 'tone_curve',
    label: 'params.group.tone_curve',
    controls: [
      unit('tone_curve.highlights', 'params.tone_curve.highlights'),
      unit('tone_curve.lights', 'params.tone_curve.lights'),
      unit('tone_curve.darks', 'params.tone_curve.darks'),
      unit('tone_curve.shadows', 'params.tone_curve.shadows'),
    ],
  },
  {
    key: 'local_contrast',
    label: 'params.group.local_contrast',
    controls: [
      unit('local_contrast.highlights', 'params.local_contrast.highlights'),
      unit('local_contrast.shadows', 'params.local_contrast.shadows'),
      unit('local_contrast.clarity', 'params.local_contrast.clarity'),
      {
        path: 'local_contrast.radius',
        label: 'params.local_contrast.radius',
        min: 0.002,
        max: 0.1,
        step: 0.001,
        neutral: 0.02,
        digits: 3,
      },
    ],
  },
  {
    key: 'noise',
    label: 'params.group.noise',
    controls: [
      {
        path: 'noise.luminance',
        label: 'params.noise.luminance',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0,
        digits: 2,
      },
      {
        path: 'noise.chrominance',
        label: 'params.noise.chrominance',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0.25,
        digits: 2,
      },
      {
        path: 'noise.radius',
        label: 'params.noise.radius',
        min: 0.0005,
        max: 0.02,
        step: 0.0005,
        neutral: 0.004,
        digits: 4,
      },
    ],
  },
  {
    key: 'sharpen',
    label: 'params.group.sharpen',
    controls: [
      {
        path: 'sharpen.amount',
        label: 'params.sharpen.amount',
        min: 0,
        max: 3,
        step: 0.01,
        neutral: 0,
        digits: 2,
      },
      {
        path: 'sharpen.radius',
        label: 'params.sharpen.radius',
        min: 0.0002,
        max: 0.01,
        step: 0.0002,
        neutral: 0.0008,
        digits: 4,
      },
      {
        path: 'sharpen.threshold',
        label: 'params.sharpen.threshold',
        min: 0,
        max: 0.2,
        step: 0.005,
        neutral: 0.01,
        digits: 3,
      },
    ],
  },
  {
    key: 'highlight_recovery',
    label: 'params.group.highlight_recovery',
    controls: [
      {
        path: 'highlight_recovery.strength',
        label: 'params.highlight_recovery.strength',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0.7,
        digits: 2,
      },
      {
        path: 'highlight_recovery.threshold',
        label: 'params.highlight_recovery.threshold',
        min: 0.5,
        max: 1,
        step: 0.005,
        neutral: 0.96,
        digits: 3,
      },
    ],
  },
  {
    key: 'split_toning',
    label: 'params.group.split_toning',
    controls: [
      {
        path: 'color.split_toning.shadow_hue',
        label: 'params.split_toning.shadow_hue',
        min: 0,
        max: 359,
        step: 1,
        neutral: 0,
        digits: 0,
        unit: '°',
      },
      {
        path: 'color.split_toning.shadow_saturation',
        label: 'params.split_toning.shadow_saturation',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0,
        digits: 2,
      },
      {
        path: 'color.split_toning.highlight_hue',
        label: 'params.split_toning.highlight_hue',
        min: 0,
        max: 359,
        step: 1,
        neutral: 0,
        digits: 0,
        unit: '°',
      },
      {
        path: 'color.split_toning.highlight_saturation',
        label: 'params.split_toning.highlight_saturation',
        min: 0,
        max: 1,
        step: 0.01,
        neutral: 0,
        digits: 2,
      },
      unit('color.split_toning.balance', 'params.split_toning.balance'),
    ],
  },
]
