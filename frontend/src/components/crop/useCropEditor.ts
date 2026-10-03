// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The hand-drawn crop, between the panel that opens it and the frame drawn
// over the photo. While it is open the preview shows the whole straightened
// frame and the rectangle is a draft, local to the screen: only "Applica"
// writes it into the parameters, as one step of the history.
//
// The draft starts in the frame's own ratio (lib/crop.ts) -- or in the ratio
// of the crop the photo already has -- and stays in it unless the user picks
// another one.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { proxyUrl } from '../../lib/api'
import type { CropRect } from '../../lib/analysisTypes'
import {
  detectAspect,
  fitRatio,
  FULL_FRAME,
  isFullFrame,
  largest,
  targetRatio,
  type CropAspect,
} from '../../lib/crop'
import { useFrameSize } from '../../lib/geometry'
import type { EditParams, Photo } from '../../lib/types'

export interface CropEditor {
  /** The frame's width / height, or null until the proxy has loaded. */
  aspect: number | null
  active: boolean
  draft: CropRect
  setDraft: (rect: CropRect) => void
  choice: CropAspect
  flipped: boolean
  /** Pixel ratio the draft is held to, or null when free. */
  ratio: number | null
  start: () => void
  cancel: () => void
  apply: () => void
  choose: (choice: CropAspect) => void
  flip: () => void
  /** The largest rectangle of the current ratio. */
  reset: () => void
}

const round = (value: number) => Math.round(value * 1e6) / 1e6

/**
 * The parameters to preview: while the crop is drawn, the whole straightened
 * frame, the draft over it; the crop comes back with "Applica" or "Annulla".
 */
export function useShownParams(params: EditParams | null, crop: CropEditor): EditParams | null {
  const active = crop.active
  return useMemo(() => {
    if (!active || !params?.geometry.crop) return params
    return { ...params, geometry: { ...params.geometry, crop: null } }
  }, [active, params])
}

export function useCropEditor(
  photo: Pick<Photo, 'id' | 'has_proxy' | 'proxy_rev'> | null | undefined,
  params: EditParams | null,
  onChange: (next: EditParams, commit: boolean) => void,
): CropEditor {
  const photoId = photo?.id ?? null
  // The proxy is the frame, upright and uncropped: its ratio is the frame's.
  const frame = useFrameSize(photo?.has_proxy ? proxyUrl(photo) : null)
  const aspect = frame ? frame.width / frame.height : null
  const [active, setActive] = useState(false)
  const [draft, setDraft] = useState<CropRect>(FULL_FRAME)
  const [choice, setChoice] = useState<CropAspect>('original')
  const [flipped, setFlipped] = useState(false)

  useEffect(() => setActive(false), [photoId])

  const ratio = aspect === null ? null : targetRatio(choice, flipped, aspect)

  const start = useCallback(() => {
    if (!params || aspect === null) return
    const current = params.geometry.crop
    const found = detectAspect(current, aspect)
    setChoice(found.choice)
    setFlipped(found.flipped)
    setDraft(current ?? FULL_FRAME)
    setActive(true)
  }, [aspect, params])

  const cancel = useCallback(() => setActive(false), [])

  const apply = useCallback(() => {
    if (!params) return
    const next = structuredClone(params)
    const rect = {
      x: round(draft.x),
      y: round(draft.y),
      width: round(Math.min(draft.width, 1 - round(draft.x))),
      height: round(Math.min(draft.height, 1 - round(draft.y))),
    }
    next.geometry = { ...next.geometry, crop: isFullFrame(rect) ? null : rect }
    onChange(next, true)
    setActive(false)
  }, [draft, onChange, params])

  const reshape = useCallback(
    (nextChoice: CropAspect, nextFlipped: boolean) => {
      setChoice(nextChoice)
      setFlipped(nextFlipped)
      if (aspect !== null) {
        setDraft((rect) => fitRatio(rect, targetRatio(nextChoice, nextFlipped, aspect), aspect))
      }
    },
    [aspect],
  )

  const choose = useCallback(
    (nextChoice: CropAspect) => reshape(nextChoice, nextChoice === choice ? flipped : false),
    [choice, flipped, reshape],
  )

  const flip = useCallback(() => reshape(choice, !flipped), [choice, flipped, reshape])

  const reset = useCallback(() => {
    if (aspect !== null) setDraft(largest(ratio, aspect))
  }, [aspect, ratio])

  return {
    aspect,
    active,
    draft,
    setDraft,
    choice,
    flipped,
    ratio,
    start,
    cancel,
    apply,
    choose,
    flip,
    reset,
  }
}
