// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The masks' keys, shared by the viewer and the review, which call this first
// from their own keydown handler: M the masks tab (G back to the global
// adjustments), O the selection in red, Canc to delete the selected mask,
// Esc to let go of it, [ and ] the brush size. And the removals' (decision
// 35): Q the correttivo, Shift+Q the gomma magica -- the key Lightroom gives
// its removal tools -- Canc and Esc on the selected removal, [ and ] the size
// of the spot or of the eraser's brush.
import { withoutMask } from '../../lib/masks'
import { HEAL_RADIUS, itemById, withItem, withoutItem } from '../../lib/retouch'
import type { EditParams } from '../../lib/types'
import { BRUSH_SIZE, type MaskEditor } from './useMaskEditor'

function handleRetouchKey(
  event: KeyboardEvent,
  editor: MaskEditor,
  params: EditParams,
  onChange: (next: EditParams, commit: boolean) => void,
): boolean {
  const key = event.key
  if (key === 'q' || key === 'Q') {
    const tool = event.shiftKey ? 'erase' : 'heal'
    editor.setTab('masks')
    editor.setTool(editor.tool === tool ? null : tool)
    return true
  }
  const item = itemById(params, editor.removal)
  if (key === '[' || key === ']') {
    const factor = key === ']' ? 1.25 : 0.8
    if (item?.kind === 'heal') {
      const radius = Math.min(HEAL_RADIUS.max, Math.max(HEAL_RADIUS.min, item.radius * factor))
      onChange(withItem(params, { ...item, radius }), true)
      editor.setHealRadius(radius)
      return true
    }
    if (editor.tool === 'heal') {
      const radius = Math.min(HEAL_RADIUS.max, Math.max(HEAL_RADIUS.min, editor.healRadius * factor))
      editor.setHealRadius(radius)
      return true
    }
    if (editor.tool === 'erase') {
      const size = Math.min(BRUSH_SIZE.max, Math.max(BRUSH_SIZE.min, editor.brush.size * factor))
      editor.setBrush({ size })
      return true
    }
    return false
  }
  if (!item) {
    if (key === 'Escape' && editor.tool) {
      editor.setTool(null)
      return true
    }
    return false
  }
  if (key === 'Delete' || key === 'Backspace') {
    onChange(withoutItem(params, item.id), true)
    editor.setRemoval(null)
    return true
  }
  if (key === 'Escape') {
    // Let go of the removal, keep the tool: the next click or stroke is a new one.
    editor.setRemoval(null)
    return true
  }
  return false
}

/** True when the key was one of the masks': the caller then stops there. */
export function handleMaskKey(
  event: KeyboardEvent,
  editor: MaskEditor,
  params: EditParams | null,
  onChange: (next: EditParams, commit: boolean) => void,
): boolean {
  if (event.ctrlKey || event.metaKey || event.altKey) return false
  const key = event.key
  if (key === 'm' || key === 'M') {
    editor.setTab('masks')
    return true
  }
  if (key === 'g' || key === 'G') {
    editor.setTab('adjust')
    return true
  }
  if (params && handleRetouchKey(event, editor, params, onChange)) return true
  if (editor.tab !== 'masks' || editor.selected === null || !params) return false
  const mask = params.masks[editor.selected]
  if (!mask) return false
  if (key === 'o' || key === 'O') {
    editor.setShowSelection(!editor.showSelection)
    return true
  }
  if (key === 'Delete' || key === 'Backspace') {
    onChange(withoutMask(params, editor.selected), true)
    editor.setSelected(null)
    return true
  }
  if (key === 'Escape') {
    editor.setSelected(null)
    return true
  }
  if (mask.kind === 'brush' && (key === '[' || key === ']')) {
    const factor = key === ']' ? 1.25 : 0.8
    const size = Math.min(BRUSH_SIZE.max, Math.max(BRUSH_SIZE.min, editor.brush.size * factor))
    editor.setBrush({ size })
    return true
  }
  return false
}
