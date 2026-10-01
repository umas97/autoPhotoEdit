// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The culling screen's model, as plain functions over plain data.
//
// The server decides; this file only arranges. Every "keep" or "discard" on
// screen comes from backend/ape/culling/select.py, so there is no scoring and
// no threshold here -- a second implementation of the selection in TypeScript
// would be a second truth, and the two would drift.
//
// What does live here is the shape of the screen: which cards are visible
// under a filter, how a burst collapses to its proposed frame (section 7.6,
// "una card mostra la scelta con un contatore 1 di 7"), and how a partial
// answer -- the few photos a slider step changed -- is folded into what the
// screen already holds.
import type {
  CullDecision,
  CullMerge,
  CullPhoto,
  CullSelection,
  CullView,
  UserDecision,
} from './cullTypes'

export type CullFilter = 'all' | 'selected' | 'culled'
export type CullSort = 'time' | 'score' | 'name'

export interface CullState {
  photos: Map<number, CullPhoto>
  /** Shooting order, as the server sends it. */
  order: number[]
  /** Burst id to members, proposed frame first. */
  bursts: Map<number, number[]>
  /** Photo id to the merge proposal it belongs to. */
  merges: Map<number, CullMerge>
  view: Omit<CullView, 'photos' | 'bursts' | 'merges'>
}

function burstsOf(photos: Map<number, CullPhoto>, order: number[]): Map<number, number[]> {
  const bursts = new Map<number, number[]>()
  for (const id of order) {
    const photo = photos.get(id)
    if (!photo || photo.burst_group_id === null) continue
    const members = bursts.get(photo.burst_group_id) ?? []
    members.push(id)
    bursts.set(photo.burst_group_id, members)
  }
  for (const [key, members] of bursts) {
    if (members.length < 2) {
      bursts.delete(key)
      continue
    }
    members.sort((a, b) => photos.get(a)!.rank - photos.get(b)!.rank)
  }
  return bursts
}

export function fromView(view: CullView): CullState {
  const photos = new Map(view.photos.map((photo) => [photo.id, photo]))
  const order = view.photos.map((photo) => photo.id)
  const merges = new Map<number, CullMerge>()
  for (const merge of view.merges) {
    if (merge.decision !== 'proposed' && merge.decision !== 'accepted') continue
    for (const member of merge.members) merges.set(member, merge)
  }
  return {
    photos,
    order,
    bursts: burstsOf(photos, order),
    merges,
    view: {
      summary: view.summary,
      settings: view.settings,
      availability: view.availability,
      confirmed: view.confirmed,
    },
  }
}

/** Fold a partial answer -- only what changed -- into the current state. */
export function applySelection(state: CullState, selection: CullSelection): CullState {
  const photos = new Map(state.photos)
  for (const decision of selection.decisions) {
    const photo = photos.get(decision.id)
    if (photo) photos.set(decision.id, { ...photo, ...(decision as CullDecision) })
  }
  return {
    ...state,
    photos,
    bursts: burstsOf(photos, state.order),
    view: { ...state.view, summary: selection.summary, settings: selection.settings },
  }
}

export interface CullCard {
  photo: CullPhoto
  /** The whole burst, proposed frame first, when this card stands for one. */
  burst: number[] | null
}

/**
 * The cards on screen. With bursts on, a burst is one card -- its proposed
 * frame -- except under "only the discarded", where every discarded frame is
 * shown on its own: that is the filter one opens to find a frame again.
 */
export function visibleCards(
  state: CullState,
  filter: CullFilter,
  sort: CullSort,
): CullCard[] {
  const collapse = state.view.settings.criteria.burst && filter !== 'culled'
  const burstOf = new Map<number, number[]>()
  for (const members of state.bursts.values()) {
    for (const id of members) burstOf.set(id, members)
  }

  const cards: CullCard[] = []
  for (const id of state.order) {
    const photo = state.photos.get(id)!
    const burst = burstOf.get(id) ?? null
    if (collapse && burst) {
      // One card per burst, standing where its proposed frame stands.
      if (burst[0] !== id) continue
      const anyKept = burst.some((member) => !state.photos.get(member)!.culled)
      if (filter === 'selected' && !anyKept) continue
      cards.push({ photo, burst })
      continue
    }
    if (filter === 'selected' && photo.culled) continue
    if (filter === 'culled' && !photo.culled) continue
    cards.push({ photo, burst: collapse ? null : burst })
  }

  if (sort === 'score') {
    cards.sort((a, b) => (b.photo.score ?? -1) - (a.photo.score ?? -1))
  } else if (sort === 'name') {
    cards.sort((a, b) => a.photo.filename.localeCompare(b.photo.filename))
  }
  return cards
}

/** What to send to put a photo back exactly as it is now. For undo. */
export function currentDecision(photo: CullPhoto): UserDecision {
  if (photo.decided_by !== 'user') return 'auto'
  return photo.culled ? 'discard' : 'keep'
}

/**
 * Where to look first in the comparison view: the camera's focus point when it
 * recorded one, the sharpest region otherwise (both come from the server).
 */
export function focusOf(photo: CullPhoto): [number, number] {
  return photo.focus_point ?? [0.5, 0.5]
}

/** A score in [0, 1] as the 0-100 the interface shows. */
export function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : String(Math.round(value * 100))
}
