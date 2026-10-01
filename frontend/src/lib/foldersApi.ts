// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The folder browser's side of the server (backend/ape/api/routes_folders.py).
// Read-only: it lists folders and counts RAWs, and never opens anything else.
import { request } from './api'

export interface FolderEntry {
  name: string
  path: string
}

export interface FolderPlace extends FolderEntry {
  /** `volume` is a memory card or disk mounted by the desktop. */
  kind: 'home' | 'pictures' | 'volume' | 'root'
}

export interface FolderListing {
  path: string
  parent: string | null
  folders: FolderEntry[]
  truncated: boolean
  /** ARW files directly in `path`: what an import of it would find. */
  raw_count: number
  places: FolderPlace[]
}

export const foldersApi = {
  /** The home folder when `path` is omitted. */
  list: (path?: string) =>
    request<FolderListing>(
      path ? `/api/folders?path=${encodeURIComponent(path)}` : '/api/folders',
    ),
}
