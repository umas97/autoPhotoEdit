// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Local interface state: what the user is looking at and what they have
// dismissed. Server state is TanStack Query's business and never lands here --
// two copies of the same truth is how an interface starts lying.
//
// The two persisted flags go through localStorage because they are preferences
// of this machine's window, not of the catalogue: a dismissed notice should not
// travel with a backup of the database (section 20).
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UiState {
  /** The photo open in the viewer, per project. */
  selectedPhoto: Record<number, number | undefined>
  select: (projectId: number, photoId: number | undefined) => void

  /** "Non mostrare più" on the tab-mode notice of section 21.2. */
  tabNoticeDismissed: boolean
  dismissTabNotice: () => void

  /** Panel groups the user folded away. */
  collapsedGroups: string[]
  toggleGroup: (key: string) => void
}

export const useUi = create<UiState>()(
  persist(
    (set, get) => ({
      selectedPhoto: {},
      select: (projectId, photoId) =>
        set({ selectedPhoto: { ...get().selectedPhoto, [projectId]: photoId } }),

      tabNoticeDismissed: false,
      dismissTabNotice: () => set({ tabNoticeDismissed: true }),

      collapsedGroups: [],
      toggleGroup: (key) => {
        const collapsed = get().collapsedGroups
        set({
          collapsedGroups: collapsed.includes(key)
            ? collapsed.filter((item) => item !== key)
            : [...collapsed, key],
        })
      },
    }),
    { name: 'ape-ui' },
  ),
)
