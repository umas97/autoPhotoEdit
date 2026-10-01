// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Every keyboard shortcut of the program, as data: what the ? list shows.
// The handlers live with their screens (pages/Viewer.tsx, pages/Review.tsx,
// pages/Culling.tsx, components/masks/shortcuts.ts); this table is the one
// place that says what they are, and must be changed with them.
import type { StringKey } from '../i18n/it'

export interface Shortcut {
  keys: string[]
  label: StringKey
}

export interface ShortcutSection {
  label: StringKey
  /** The screens it applies to, as the last segment of their route. */
  screens: Array<'foto' | 'revisione' | 'cernita' | '*'>
  items: Shortcut[]
}

const MASKS: Shortcut[] = [
  { keys: ['M'], label: 'shortcuts.masksTab' },
  { keys: ['G'], label: 'shortcuts.adjustTab' },
  { keys: ['O'], label: 'shortcuts.showSelection' },
  { keys: ['Q'], label: 'shortcuts.heal' },
  { keys: ['Shift+Q'], label: 'shortcuts.erase' },
  { keys: ['Canc'], label: 'shortcuts.deleteMask' },
  { keys: ['Esc'], label: 'shortcuts.deselect' },
  { keys: ['[', ']'], label: 'shortcuts.retouchSize' },
  { keys: ['Alt'], label: 'shortcuts.brushErase' },
]

export const SHORTCUTS: ShortcutSection[] = [
  {
    label: 'shortcuts.section.everywhere',
    screens: ['*'],
    items: [{ keys: ['?'], label: 'shortcuts.help' }],
  },
  {
    label: 'shortcuts.section.viewer',
    screens: ['foto'],
    items: [
      { keys: ['←', '→'], label: 'shortcuts.navigate' },
      { keys: ['\\'], label: 'shortcuts.before' },
      { keys: ['Ctrl+Z'], label: 'shortcuts.undo' },
      { keys: ['Ctrl+Shift+Z', 'Ctrl+Y'], label: 'shortcuts.redo' },
      { keys: ['Ctrl+S'], label: 'shortcuts.save' },
    ],
  },
  { label: 'shortcuts.section.masks', screens: ['foto', 'revisione'], items: MASKS },
  {
    label: 'shortcuts.section.review',
    screens: ['revisione'],
    items: [
      { keys: ['←', '→'], label: 'shortcuts.navigate' },
      { keys: ['A'], label: 'shortcuts.approve' },
      { keys: ['R'], label: 'shortcuts.reject' },
      { keys: ['Spazio', '\\'], label: 'shortcuts.before' },
      { keys: ['1', '2', '3'], label: 'shortcuts.variant' },
      { keys: ['Z'], label: 'shortcuts.zoom' },
      { keys: ['C'], label: 'shortcuts.compare' },
      { keys: ['Ctrl+Z'], label: 'shortcuts.undoReview' },
      { keys: ['Ctrl+Shift+Z'], label: 'shortcuts.redo' },
    ],
  },
  {
    label: 'shortcuts.section.reviewGrid',
    screens: ['revisione'],
    items: [
      { keys: ['A'], label: 'shortcuts.gridApprove' },
      { keys: ['←', '→'], label: 'shortcuts.gridMove' },
      { keys: ['Spazio'], label: 'shortcuts.gridSelect' },
      { keys: ['Invio'], label: 'shortcuts.gridOpen' },
      { keys: ['Esc'], label: 'shortcuts.gridClear' },
    ],
  },
  {
    label: 'shortcuts.section.culling',
    screens: ['cernita'],
    items: [
      { keys: ['X'], label: 'shortcuts.discard' },
      { keys: ['P'], label: 'shortcuts.keep' },
      { keys: ['←', '→'], label: 'shortcuts.navigate' },
      { keys: ['↑', '↓'], label: 'shortcuts.burst' },
      { keys: ['Spazio'], label: 'shortcuts.cullCompare' },
      { keys: ['Ctrl+Z'], label: 'shortcuts.cullUndo' },
    ],
  },
]

/** The keys a focused slider moves itself with (Radix, WAI-ARIA slider pattern). */
const SLIDER_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'])

/**
 * The key belongs to the control under focus, not to the screen: anything typed
 * in a text field, and the arrows of a focused slider -- which otherwise moved
 * the slider *and* went to the next photo, saving the step on the wrong one.
 */
export function keyBelongsToControl(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null
  if (!target) return false
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) return true
  return target.getAttribute('role') === 'slider' && SLIDER_KEYS.has(event.key)
}
