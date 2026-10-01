// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// "Nuovo profilo": a name and two folders (section 8.1), each typed or chosen
// with the same folder browser as a new project. The pairing is shown before
// anything is created -- how many pairs, found how, and whether that is enough
// -- so that a wrong folder is noticed while it costs nothing, not after ten
// minutes of training.
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FolderSearch } from 'lucide-react'
import { ApiError } from '../../lib/api'
import { stylesApi } from '../../lib/stylesApi'
import type { PairingReport, StyleSummary } from '../../lib/styleTypes'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'
import { FolderBrowser } from '../FolderBrowser'
import { Button } from '../ui/Button'
import { Dialog } from '../ui/Dialog'
import { Field, TextInput } from '../ui/Field'

/** Section 8.1: the least that works, and what makes a good profile. */
const MIN_PAIRS = 8
const RECOMMENDED_PAIRS = 30

function verdict(pairs: number): { text: string; tone: string } {
  if (pairs < MIN_PAIRS) {
    return { text: t('styles.create.verdictFew', { min: MIN_PAIRS }), tone: 'border-warn/40 bg-warn/10 text-warn' }
  }
  if (pairs < RECOMMENDED_PAIRS) {
    return {
      text: t('styles.create.verdictEnough', { recommended: RECOMMENDED_PAIRS }),
      tone: 'border-ink-700 bg-ink-850 text-ink-200',
    }
  }
  return { text: t('styles.create.verdictGood'), tone: 'border-ink-700 bg-ink-850 text-good' }
}

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
  const [browsing, setBrowsing] = useState<'raws' | 'references' | null>(null)

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
      setBrowsing(null)
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
        <p className="rounded border border-ink-700 bg-ink-850 p-2 text-xs text-ink-300">
          {t('styles.create.howMany', { min: MIN_PAIRS, recommended: RECOMMENDED_PAIRS })}
        </p>
        <Field label={t('styles.create.rawDir')}>
          <div className="flex gap-2">
            <TextInput
              value={rawDir}
              placeholder="/home/…/RAW"
              spellCheck={false}
              onChange={(event) => {
                setRawDir(event.target.value)
                setReport(null)
              }}
            />
            <BrowseButton open={browsing === 'raws'}
              onClick={() => setBrowsing((open) => (open === 'raws' ? null : 'raws'))} />
          </div>
        </Field>
        {browsing === 'raws' ? (
          <FolderBrowser
            start={rawDir.trim()}
            onClose={() => setBrowsing(null)}
            onChoose={(path) => {
              setRawDir(path)
              setReport(null)
              setBrowsing(null)
            }}
          />
        ) : null}
        <Field label={t('styles.create.referenceDir')} hint={t('styles.create.referenceHint')}>
          <div className="flex gap-2">
            <TextInput
              value={referenceDir}
              placeholder={rawDir || '/home/…/Editate'}
              spellCheck={false}
              onChange={(event) => {
                setReferenceDir(event.target.value)
                setReport(null)
              }}
            />
            <BrowseButton open={browsing === 'references'}
              onClick={() => setBrowsing((open) => (open === 'references' ? null : 'references'))} />
          </div>
        </Field>
        {browsing === 'references' ? (
          <FolderBrowser
            // The edited photos are usually next to the RAWs: start there.
            start={referenceDir.trim() || rawDir.trim()}
            counts="references"
            onClose={() => setBrowsing(null)}
            onChoose={(path) => {
              setReferenceDir(path)
              setReport(null)
              setBrowsing(null)
            }}
          />
        ) : null}
        {report ? (
          <div className={cn('space-y-1 rounded border p-2 text-xs', verdict(report.pairs).tone)}>
            <p className="text-ink-200">
              {t('styles.create.previewResult', {
                pairs: report.pairs,
                raws: report.raws ?? '—',
                name: report.methods.name,
                xmp: report.methods.xmp,
                time: report.methods.time,
                unpaired,
              })}
            </p>
            <p>{verdict(report.pairs).text}</p>
          </div>
        ) : null}
        {error ? <p className="text-xs text-bad">{error.message}</p> : null}
      </div>
    </Dialog>
  )
}

function BrowseButton({ open, onClick }: { open: boolean; onClick: () => void }) {
  return (
    <Button type="button" variant="outline" aria-expanded={open} onClick={onClick}>
      <FolderSearch size={14} />
      {t('folders.browse')}
    </Button>
  )
}
