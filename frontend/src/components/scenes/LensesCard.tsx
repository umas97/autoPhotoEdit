// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The lenses of a project and their lensfun profiles (section 6.4): which one
// corrects each, which have none, and the two ways to fix a missing one --
// update the lensfun data (an explicit download, section 19) or associate a
// profile by hand, remembered for that lens in every project.
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { LensEntry } from '../../lib/analysisTypes'
import { t } from '../../i18n/it'
import { Button } from '../ui/Button'
import { LensfunDataLine } from '../LensfunData'
import { Dialog } from '../ui/Dialog'
import { TextInput } from '../ui/Field'

function status(entry: LensEntry): { text: string; missing: boolean } {
  if (entry.override) return { text: t('lenses.override', { model: entry.override.model }), missing: false }
  if (entry.profile) return { text: t('lenses.auto', { model: entry.profile.model }), missing: false }
  if (entry.analysed === 0) return { text: t('lenses.pending'), missing: false }
  return { text: t('lenses.missing'), missing: true }
}

function AssociateDialog({
  lens,
  onClose,
  onChosen,
}: {
  lens: string
  onClose: () => void
  onChosen: () => void
}) {
  const [query, setQuery] = useState(lens.replace(/^(FE|E)\s+/, ''))
  const results = useQuery({
    queryKey: ['lens-search', query],
    queryFn: () => api.searchLenses(query),
    enabled: query.trim().length >= 2,
  })
  const choose = useMutation({
    mutationFn: (choice: { maker: string; model: string }) =>
      api.setLensOverride(lens, choice.maker, choice.model),
    onSuccess: onChosen,
  })

  return (
    <Dialog
      open
      onOpenChange={(open) => (open ? undefined : onClose())}
      title={t('lenses.dialogTitle', { lens })}
      description={t('lenses.dialogBody')}
      footer={
        <Button variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
      }
    >
      <label className="block text-xs text-ink-300">
        {t('lenses.search')}
        <TextInput
          value={query}
          placeholder={t('lenses.searchHint')}
          onChange={(event) => setQuery(event.target.value)}
          className="mt-1"
        />
      </label>
      <ul className="mt-2 max-h-64 overflow-y-auto">
        {(results.data ?? []).map((candidate) => (
          <li key={`${candidate.maker}|${candidate.model}`}>
            <button
              type="button"
              disabled={choose.isPending}
              className="w-full rounded px-2 py-1 text-left text-sm text-ink-100 hover:bg-ink-800"
              onClick={() => choose.mutate(candidate)}
            >
              {candidate.maker} {candidate.model}
            </button>
          </li>
        ))}
        {results.data && results.data.length === 0 ? (
          <li className="px-2 py-1 text-xs text-ink-400">{t('lenses.noResults')}</li>
        ) : null}
      </ul>
    </Dialog>
  )
}

export function LensesCard({ projectId }: { projectId: number }) {
  const queryClient = useQueryClient()
  const [associating, setAssociating] = useState<string | null>(null)
  const lenses = useQuery({
    queryKey: ['lenses', projectId],
    queryFn: () => api.lenses(projectId),
    // Only while the lensfun data are being fetched; at rest, no requests.
    refetchInterval: (query) => (query.state.data?.lensfun.update?.state === 'running' ? 1_500 : false),
  })
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['lenses', projectId] })
    void queryClient.invalidateQueries({ queryKey: ['photos', projectId] })
    void queryClient.invalidateQueries({ queryKey: ['photo'] })
  }
  const update = useMutation({ mutationFn: api.updateLensfun, onSuccess: refresh })
  const clear = useMutation({ mutationFn: (lens: string) => api.clearLensOverride(lens), onSuccess: refresh })

  if (!lenses.data) return null
  return (
    <section className="rounded-md border border-ink-700 bg-ink-900 p-3">
      <h2 className="text-sm font-semibold text-ink-100">{t('lenses.title')}</h2>
      <ul className="mt-2 space-y-1.5">
        {lenses.data.lenses.map((entry) => {
          const line = status(entry)
          return (
            <li key={entry.lens ?? '—'} className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span className="text-ink-100">{entry.lens ?? t('lenses.unknown')}</span>
              <span className="text-xs text-ink-400">{t('lenses.photos', { count: entry.photos })}</span>
              <span className={line.missing ? 'text-xs text-warn' : 'text-xs text-ink-300'}>{line.text}</span>
              {entry.lens ? (
                <span className="ml-auto flex gap-1">
                  <Button size="sm" variant="ghost" onClick={() => setAssociating(entry.lens)}>
                    {entry.override || entry.profile ? t('lenses.change') : t('lenses.associate')}
                  </Button>
                  {entry.override ? (
                    <Button size="sm" variant="ghost" onClick={() => clear.mutate(entry.lens!)}>
                      {t('lenses.clear')}
                    </Button>
                  ) : null}
                </span>
              ) : null}
            </li>
          )
        })}
      </ul>
      <LensfunDataLine lensfun={lenses.data.lensfun} onUpdate={() => update.mutate()}
        className="mt-3 border-t border-ink-800 pt-2" />
      {associating ? (
        <AssociateDialog
          lens={associating}
          onClose={() => setAssociating(null)}
          onChosen={() => {
            setAssociating(null)
            refresh()
          }}
        />
      ) : null}
    </section>
  )
}
