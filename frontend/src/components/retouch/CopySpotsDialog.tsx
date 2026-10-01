// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Copia punti sulle foto selezionate" (docs/SPEC_rimozione.md R6): the one way a
// removal leaves its photo, and only when asked. Meant for the dust on a
// sensor: the spots land on the same photosites of the photos picked in the
// grid, portrait or landscape, their sources found again on each; erasers
// travel only with "anche le gomme". Confirmed with the number of photos.
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { t } from '../../i18n/it'
import { retouchApi } from '../../lib/retouchApi'
import { toast } from '../../lib/toast'
import type { EditParams } from '../../lib/types'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Checkbox } from '../ui/Field'

export function CopySpotsDialog({
  open,
  onOpenChange,
  projectId,
  photoId,
  params,
  targets,
  flush,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  projectId: number
  photoId: number
  params: EditParams
  targets: number[]
  flush?: () => Promise<void>
}) {
  const queryClient = useQueryClient()
  const [withErasers, setWithErasers] = useState(false)
  const others = targets.filter((id) => id !== photoId)
  const spots = params.retouch.filter((item) => item.kind === 'heal').length
  const erasers = params.retouch.filter((item) => item.kind === 'erase').length
  const copy = useMutation({
    mutationFn: async () => {
      // The server copies what is saved: the last gesture first.
      await flush?.()
      return retouchApi.copy(projectId, photoId, others, withErasers)
    },
    onSuccess: (result) => {
      toast(t('retouch.copy.done', { photos: result.queued }))
      if (result.skipped) toast(t('retouch.copy.skipped', { count: result.skipped }))
      for (const id of others) void queryClient.invalidateQueries({ queryKey: ['photo', id] })
      onOpenChange(false)
    },
  })
  const nothing = spots === 0 && !(withErasers && erasers > 0)
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t('retouch.copy.title')}
      description={t('retouch.copy.body')}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t('retouch.fromMask.cancel')}
          </Button>
          <Button variant="primary" disabled={others.length === 0 || nothing || copy.isPending}
            onClick={() => copy.mutate()}>
            {t('retouch.copy.confirm', { photos: others.length })}
          </Button>
        </>
      }>
      {others.length === 0 ? (
        <p className="text-sm text-ink-300">{t('retouch.copy.pick')}</p>
      ) : null}
      {nothing ? <p className="text-sm text-ink-300">{t('retouch.copy.nothing')}</p> : null}
      {erasers > 0 ? (
        <Checkbox checked={withErasers} onChange={setWithErasers} label={t('retouch.copy.erase')} />
      ) : null}
    </Dialog>
  )
}
