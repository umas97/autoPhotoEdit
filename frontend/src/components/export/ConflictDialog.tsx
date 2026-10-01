// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Section 16.2: "at the first collision the interface shows a dialog with
// Sovrascrivi / Rinomina / Salta and a box 'Applica a tutte le restanti'".
//
// The server lists every colliding photo before the batch starts, so the
// dialog walks through them one at a time; with the box ticked, the choice
// answers the rest and becomes the project's policy. Closing the dialog
// without answering starts nothing -- never a silent default.
import { useState } from 'react'
import type { ExportConflict, ExportConflictInfo } from '../../lib/exportTypes'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Checkbox } from '../ui/Field'

export interface ConflictAnswers {
  answers: Array<{ photo_id: number; policy: ExportConflict }>
  apply_to_all: ExportConflict | null
}

export function ConflictDialog({
  conflicts,
  folder,
  onDone,
  onCancel,
}: {
  conflicts: ExportConflictInfo[]
  folder: string
  onDone: (answers: ConflictAnswers) => void
  onCancel: () => void
}) {
  const [index, setIndex] = useState(0)
  const [all, setAll] = useState(false)
  const [answers, setAnswers] = useState<ConflictAnswers['answers']>([])
  const current = conflicts[index]
  if (!current) return null

  const choose = (policy: ExportConflict) => {
    if (all) {
      onDone({ answers, apply_to_all: policy })
      return
    }
    const next = [...answers, { photo_id: current.photo_id, policy }]
    if (index + 1 >= conflicts.length) {
      onDone({ answers: next, apply_to_all: null })
      return
    }
    setAnswers(next)
    setIndex(index + 1)
  }

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onCancel()
      }}
      title={t('export.ask.title')}
      description={t('export.ask.progress', { n: index + 1, total: conflicts.length })}
      footer={
        <>
          <Button variant="ghost" onClick={onCancel}>
            {t('common.cancel')}
          </Button>
          <Button variant="outline" onClick={() => choose('skip')}>
            {t('export.ask.skip')}
          </Button>
          <Button variant="secondary" onClick={() => choose('overwrite')}>
            {t('export.ask.overwrite')}
          </Button>
          <Button variant="primary" onClick={() => choose('rename')}>
            {t('export.ask.rename')}
          </Button>
        </>
      }
    >
      <p className="text-sm text-ink-100">
        {t('export.ask.body', { files: current.files.join(', '), folder })}
      </p>
      <p className="mt-1 text-xs text-ink-400">{current.filename}</p>
      {conflicts.length - index > 1 ? (
        <div className="mt-3">
          <Checkbox checked={all} onChange={setAll} label={t('export.ask.all')}
            hint={t('export.ask.allHint')} />
        </div>
      ) : null}
    </Dialog>
  )
}
