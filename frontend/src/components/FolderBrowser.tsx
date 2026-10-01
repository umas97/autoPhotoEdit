// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Choosing a folder by walking it: the project's RAWs, and the two folders of a
// style profile (components/styles/CreateProfileDialog.tsx). The browser cannot
// give the server a path, but the server is on the same machine and lists the
// folders itself (backend/ape/api/routes_folders.py). A panel, not a second
// dialog: it opens inside the dialog that needs the path, under its field.
//
// One click opens a folder; "Usa questa cartella" takes the one shown, with
// the count of RAWs an import of it would find -- the import reads that folder
// only, not its sub-folders (section 15), so a card is chosen at DCIM/100MSDCF,
// not at its root, and the count says so before the project exists.
import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowUp, Check, Folder, HardDrive, House, Image, Monitor } from 'lucide-react'
import { foldersApi, type FolderPlace } from '../lib/foldersApi'
import { ApiError } from '../lib/api'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { Button } from './ui/Button'

const PLACE_ICONS = { home: House, pictures: Image, volume: HardDrive, root: Monitor }

function placeLabel(place: FolderPlace): string {
  return place.kind === 'volume' ? place.name : t(`folders.place.${place.kind}`)
}

/** "/home/a/b" -> [["/", "/"], ["home", "/home"], ["a", "/home/a"], ["b", "/home/a/b"]]. */
function crumbs(path: string): [string, string][] {
  const parts = path.split('/').filter(Boolean)
  return [['/', '/'], ...parts.map((name, i): [string, string] => [name, '/' + parts.slice(0, i + 1).join('/')])]
}

/** Every ancestor clickable; scrolled to the end, where the folder shown is. */
function Breadcrumbs({ path, onGo }: { path: string; onGo: (path: string) => void }) {
  const row = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (row.current) row.current.scrollLeft = row.current.scrollWidth
  }, [path])
  const items = crumbs(path)
  return (
    <div ref={row} className="flex min-w-0 flex-1 items-center overflow-x-auto whitespace-nowrap text-xs no-scrollbar"
      title={path}>
      {items.map(([name, target], index) => {
        const last = index === items.length - 1
        return (
          <span key={target} className="flex items-center">
            {index > 1 ? <span className="px-0.5 text-ink-500">/</span> : null}
            <button type="button" onClick={() => onGo(target)} disabled={last}
              className={cn(
                'rounded px-1 py-0.5',
                last ? 'font-medium text-ink-50' : 'text-ink-300 hover:bg-ink-700 hover:text-ink-50',
              )}>
              {name}
            </button>
          </span>
        )
      })}
    </div>
  )
}

export function FolderBrowser({
  start,
  counts = 'raws',
  onChoose,
  onClose,
}: {
  /** Where to open: the path already typed, if it is one. */
  start?: string
  /** What the folder is for, and so what its count is of: RAWs, or edited photos. */
  counts?: 'raws' | 'references'
  onChoose: (path: string) => void
  onClose: () => void
}) {
  const [path, setPath] = useState<string | undefined>(
    start && start.startsWith('/') ? start : undefined,
  )
  const listing = useQuery({
    queryKey: ['folders', path ?? '~'],
    queryFn: () => foldersApi.list(path),
    retry: false,
  })
  const data = listing.data
  const count = data ? (counts === 'raws' ? data.raw_count : data.reference_count) : 0
  const countLabel = counts === 'raws'
    ? (count ? t('folders.raws', { count }) : t('folders.noRaws'))
    : (count ? t('folders.references', { count }) : t('folders.noReferences'))

  return (
    <div className="rounded-md border border-ink-600 bg-ink-900">
      <div className="flex flex-wrap gap-1 border-b border-ink-700 p-1.5">
        {data?.places.map((place) => {
          const Icon = PLACE_ICONS[place.kind]
          return (
            <Button key={place.path} size="sm" type="button" title={place.path}
              variant={data.path === place.path ? 'secondary' : 'ghost'}
              onClick={() => setPath(place.path)}>
              <Icon size={13} />
              {placeLabel(place)}
            </Button>
          )
        })}
      </div>

      <div className="flex items-center gap-1.5 border-b border-ink-700 px-1.5 py-1">
        <Button size="icon" variant="ghost" type="button" aria-label={t('folders.up')}
          title={t('folders.up')} disabled={!data?.parent}
          onClick={() => data?.parent && setPath(data.parent)}>
          <ArrowUp size={14} />
        </Button>
        {data ? <Breadcrumbs path={data.path} onGo={setPath} /> : null}
      </div>

      <ul className="h-56 overflow-y-auto p-1" aria-busy={listing.isFetching}>
        {listing.error ? (
          <li className="space-y-2 p-2 text-sm">
            <p className="text-bad">{(listing.error as ApiError).message}</p>
            <Button size="sm" type="button" onClick={() => setPath(undefined)}>
              {t('folders.home')}
            </Button>
          </li>
        ) : null}
        {data && data.folders.length === 0 ? (
          <li className="p-2 text-sm text-ink-400">{t('folders.empty')}</li>
        ) : null}
        {data?.folders.map((folder) => (
          <li key={folder.path}>
            <button type="button" onClick={() => setPath(folder.path)}
              className={cn(
                'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm text-ink-100',
                'hover:bg-ink-700 focus-visible:bg-ink-700 focus-visible:outline-none',
              )}>
              <Folder size={14} className="shrink-0 text-ink-400" />
              <span className="truncate">{folder.name}</span>
            </button>
          </li>
        ))}
        {data?.truncated ? (
          <li className="p-2 text-xs text-ink-400">
            {t('folders.truncated', { count: data.folders.length })}
          </li>
        ) : null}
      </ul>

      <div className="flex items-center gap-2 border-t border-ink-700 p-1.5">
        <p className={cn('min-w-0 flex-1 text-xs', count ? 'text-ink-200' : 'text-ink-400')}>
          {data ? countLabel : null}
        </p>
        <Button size="sm" variant="ghost" type="button" onClick={onClose}>
          {t('folders.close')}
        </Button>
        <Button size="sm" variant="primary" type="button" disabled={!data}
          onClick={() => data && onChoose(data.path)}>
          <Check size={13} />
          {t('folders.choose')}
        </Button>
      </div>
    </div>
  )
}
