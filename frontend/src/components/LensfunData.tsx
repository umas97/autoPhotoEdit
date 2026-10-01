// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Which lensfun data the program uses, and the one button that fetches newer
// ones (section 6.4; an explicit download, section 19). The data belong to the
// program, not to a project: the line is in Settings and, next to the lenses
// it concerns, in a project's Scenes screen.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { api } from '../lib/api'
import type { LensfunState } from '../lib/analysisTypes'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { Button } from './ui/Button'

export function LensfunDataLine({
  lensfun,
  onUpdate,
  className,
}: {
  lensfun: LensfunState
  onUpdate: () => void
  className?: string
}) {
  const running = lensfun.update?.state === 'running'
  const date = lensfun.updated_at ? new Date(lensfun.updated_at).toLocaleDateString('it-IT') : ''
  return (
    <div className={cn('flex flex-wrap items-center gap-2 text-xs text-ink-400', className)}>
      <span>
        {lensfun.source === 'updated'
          ? t('lenses.data.updated', { date, count: lensfun.lenses })
          : t('lenses.data.bundled', { count: lensfun.lenses })}
      </span>
      <Button size="sm" variant="ghost" disabled={running} onClick={onUpdate} title={t('lenses.data.updateHint')}>
        <RefreshCw size={12} className={running ? 'animate-spin' : undefined} />
        {running ? t('lenses.data.updating') : t('lenses.data.update')}
      </Button>
      {lensfun.update?.error ? (
        <span className="text-bad">{t('lenses.data.failed', { error: lensfun.update.error })}</span>
      ) : null}
    </div>
  )
}

/** The line on its own, for Settings: it asks the server itself. */
export function LensfunData() {
  const queryClient = useQueryClient()
  const lensfun = useQuery({
    queryKey: ['lensfun'],
    queryFn: api.lensfun,
    // Only while the data are being fetched; at rest, no requests.
    refetchInterval: (query) => (query.state.data?.update?.state === 'running' ? 1_500 : false),
  })
  const update = useMutation({
    mutationFn: api.updateLensfun,
    onSuccess: (state) => {
      queryClient.setQueryData(['lensfun'], state)
      // Every project's lenses and photos: the profiles may have changed.
      void queryClient.invalidateQueries({ queryKey: ['lenses'] })
      void queryClient.invalidateQueries({ queryKey: ['photo'] })
    },
  })
  if (!lensfun.data) return null
  return <LensfunDataLine lensfun={lensfun.data} onUpdate={() => update.mutate()} />
}
