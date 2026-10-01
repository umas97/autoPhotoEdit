// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The manual pairing screen of section 8.1: the files of the profile's two
// folders that no clue paired, side by side. Pick one of each, pair them; the
// new pair is inverted in the background like the others.
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link2 } from 'lucide-react'
import { ApiError } from '../../lib/api'
import { stylesApi } from '../../lib/stylesApi'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { Button } from '../ui/Button'

const baseName = (path: string) => path.split('/').pop() ?? path

function FileList({
  title,
  files,
  selected,
  onSelect,
}: {
  title: string
  files: string[]
  selected: string | null
  onSelect: (file: string) => void
}) {
  return (
    <div className="min-w-0">
      <h4 className="mb-1 text-xs font-medium text-ink-300">{title}</h4>
      <ul className="max-h-48 overflow-y-auto rounded border border-ink-700">
        {files.map((file) => (
          <li key={file}>
            <button
              type="button"
              onClick={() => onSelect(file)}
              className={cn(
                'block w-full truncate px-2 py-1 text-left text-xs',
                selected === file ? 'bg-ink-600 text-ink-50' : 'text-ink-200 hover:bg-ink-800',
              )}
              title={file}
            >
              {baseName(file)}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function ManualPairing({ profileId }: { profileId: number }) {
  const queryClient = useQueryClient()
  const [raw, setRaw] = useState<string | null>(null)
  const [reference, setReference] = useState<string | null>(null)
  const unpaired = useQuery({ queryKey: ['unpaired', profileId], queryFn: () => stylesApi.unpaired(profileId) })
  const pair = useMutation({
    mutationFn: () => stylesApi.addPair(profileId, raw!, reference!),
    onSuccess: () => {
      setRaw(null)
      setReference(null)
      void queryClient.invalidateQueries({ queryKey: ['unpaired', profileId] })
      void queryClient.invalidateQueries({ queryKey: ['style', profileId] })
    },
  })

  const data = unpaired.data
  if (!data || data.references.length === 0) {
    return data ? <p className="text-xs text-ink-400">{t('styles.manual.none')}</p> : null
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-ink-300">{t('styles.manual.body')}</p>
      <div className="grid grid-cols-2 gap-2">
        <FileList title={t('styles.manual.raws')} files={data.raws} selected={raw} onSelect={setRaw} />
        <FileList
          title={t('styles.manual.references')}
          files={data.references}
          selected={reference}
          onSelect={setReference}
        />
      </div>
      <div className="flex items-center gap-2">
        <Button size="sm" variant="outline" disabled={!raw || !reference || pair.isPending} onClick={() => pair.mutate()}>
          <Link2 size={13} />
          {t('styles.manual.pair')}
        </Button>
        {pair.error ? <span className="text-xs text-bad">{(pair.error as ApiError).message}</span> : null}
      </div>
    </div>
  )
}
