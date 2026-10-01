// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the masks editor remembers between the panel on the right and the
// handles over the photo: which tab is open, which mask is selected, whether
// its selection is shown, and how the brush is set -- and for the removals of
// the same tab, which tool is in the hand, which removal is selected, the
// spot's size and whether the removals are shown at all. Local to the screen:
// none of it is part of the edit.
import { useCallback, useEffect, useState } from 'react'
import { HEAL_RADIUS } from '../../lib/retouch'

export interface BrushSettings {
  /** Radius, as a fraction of the frame's long edge -- like every radius. */
  size: number
  /** 0 a hard edge, 1 a dab that fades from its centre. */
  softness: number
  /** How much one pass adds, 0..1. */
  flow: number
  erase: boolean
}

export interface MaskEditor {
  tab: 'adjust' | 'masks'
  setTab: (tab: 'adjust' | 'masks') => void
  selected: number | null
  setSelected: (index: number | null) => void
  showSelection: boolean
  setShowSelection: (show: boolean) => void
  brush: BrushSettings
  setBrush: (patch: Partial<BrushSettings>) => void
  /** A stroke is being uploaded: the panel says so, and nothing else paints. */
  busy: boolean
  setBusy: (busy: boolean) => void
  /** The subject picker is open: "Soggetto" was clicked. */
  segmenting: boolean
  setSegmenting: (open: boolean) => void
  /** The removal tool in the hand: a click heals, a stroke erases. */
  tool: RetouchTool | null
  setTool: (tool: RetouchTool | null) => void
  /** The selected removal, by id; selecting one lets go of the mask, and back. */
  removal: string | null
  setRemoval: (id: string | null) => void
  /** Radius of the next spot, in long edges. */
  healRadius: number
  setHealRadius: (radius: number) => void
  /** "Mostra rimozioni": off suspends them in the preview, parameters untouched. */
  showRemovals: boolean
  setShowRemovals: (show: boolean) => void
}

export type RetouchTool = 'heal' | 'erase'

export const BRUSH_SIZE = { min: 0.004, max: 0.2 }

export function useMaskEditor(photoId: number | null, count: number): MaskEditor {
  const [tab, setTab] = useState<'adjust' | 'masks'>('adjust')
  const [selected, setSelectedState] = useState<number | null>(null)
  const [showSelection, setShowSelection] = useState(false)
  const [busy, setBusy] = useState(false)
  const [segmenting, setSegmenting] = useState(false)
  const [tool, setTool] = useState<RetouchTool | null>(null)
  const [removal, setRemovalState] = useState<string | null>(null)
  const [healRadius, setHealRadius] = useState(HEAL_RADIUS.start)
  const [showRemovals, setShowRemovals] = useState(true)
  const [brush, setBrushState] = useState<BrushSettings>({
    size: 0.03,
    softness: 0.6,
    flow: 0.7,
    erase: false,
  })

  // Another photo: nothing selected, the handles gone.
  useEffect(() => {
    setSelectedState(null)
    setShowSelection(false)
    setSegmenting(false)
    setRemovalState(null)
  }, [photoId])

  // An undo can take the selected mask away.
  useEffect(() => {
    if (selected !== null && selected >= count) setSelectedState(count > 0 ? count - 1 : null)
  }, [count, selected])

  // One thing selected at a time: a mask, or a removal.
  const setSelected = useCallback((index: number | null) => {
    setSelectedState(index)
    if (index !== null) {
      setRemovalState(null)
      setTool(null)
    }
  }, [])
  const setRemoval = useCallback((id: string | null) => {
    setRemovalState(id)
    if (id !== null) setSelectedState(null)
  }, [])

  const setBrush = useCallback(
    (patch: Partial<BrushSettings>) => setBrushState((previous) => ({ ...previous, ...patch })),
    [],
  )

  return {
    tab,
    setTab,
    selected: tab === 'masks' ? selected : null,
    setSelected,
    showSelection: tab === 'masks' && showSelection,
    setShowSelection,
    brush,
    setBrush,
    busy,
    setBusy,
    segmenting: tab === 'masks' && segmenting,
    setSegmenting,
    tool: tab === 'masks' ? tool : null,
    setTool,
    removal: tab === 'masks' ? removal : null,
    setRemoval,
    healRadius,
    setHealRadius,
    showRemovals,
    setShowRemovals,
  }
}
