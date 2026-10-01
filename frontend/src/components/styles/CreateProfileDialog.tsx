// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Nuovo profilo": a name and two folders (section 8.1). The pairing is shown
// before anything is created -- how many pairs, found how -- so that a wrong
// folder is noticed while it costs nothing, not after ten minutes of training.
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ApiError } from '../../lib/api'
import { stylesApi } from '../../lib/stylesApi'
import type { PairingReport, StyleSummary } from '../../lib/styleTypes'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Field, TextInput } from '../ui/Field'

export function CreateProfileDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (profile: StyleSummary) => void
}) {
  const [name, setName] = useState('')
  const [rawDir, setRawDir] = useState('')
  const [referenceDir, setReferenceDir] = useState('')
  const [report, setReport] = useState<PairingReport | null>(null)

  const body = { name: name.trim(), raw_dir: rawDir.trim(), reference_dir: referenceDir.trim() || rawDir.trim() }
  const ready = body.name !== '' && body.raw_dir !== ''

  const preview = useMutation({
    mutationFn: () => stylesApi.pairingPreview(body),
    onSuccess: setReport,
  })
  const create = useMutation({
    mutationFn: () => stylesApi.createStyle(body),
    onSuccess: (profile) => {
      onCreated(profile)
      onOpenChange(false)
      setName('')
      setRawDir('')
      setReferenceDir('')
      setReport(null)
    },
  })
  const error = (preview.error ?? create.error) as ApiError | null
  const unpaired = report
    ? typeof report.unpaired_references === 'number'
      ? report.unpaired_references
      : report.unpaired_references.length
    : 0

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t('styles.create.title')}
      description={t('styles.create.duration')}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button variant="outline" disabled={!ready || preview.isPending} onClick={() => preview.mutate()}>
            {t('styles.create.preview')}
          </Button>
          <Button variant="primary" disabled={!ready || create.isPending} onClick={() => create.mutate()}>
            {t('styles.create.submit')}
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Field label={t('styles.create.name')}>
          <TextInput value={name} onChange={(event) => setName(event.target.value)} autoFocus />
        </Field>
        <Field label={t('styles.create.rawDir')}>
          <TextInput
            value={rawDir}
            placeholder="/home/…/RAW"
            onChange={(event) => {
              setRawDir(event.target.value)
              setReport(null)
            }}
          />
        </Field>
        <Field label={t('styles.create.referenceDir')} hint={t('styles.create.referenceHint')}>
          <TextInput
            value={referenceDir}
            placeholder={rawDir || '/home/…/Editate'}
            onChange={(event) => {
              setReferenceDir(event.target.value)
              setReport(null)
            }}
          />
        </Field>
        {report ? (
          <p className="rounded border border-ink-700 bg-ink-850 p-2 text-xs text-ink-200">
            {t('styles.create.previewResult', {
              pairs: report.pairs,
              raws: report.raws ?? '—',
              name: report.methods.name,
              xmp: report.methods.xmp,
              time: report.methods.time,
              unpaired,
            })}
          </p>
        ) : null}
        {error ? <p className="text-xs text-bad">{error.message}</p> : null}
      </div>
    </Dialog>
  )
}
