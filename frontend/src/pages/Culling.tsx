// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Screen 3 of section 10: culling (section 7.6).
//
// A dense grid of the camera's own previews, one card per moment -- a burst
// collapses to its proposed frame -- with the settings on the right and, at the
// bottom, the count and the one button that leads out: "Procedi con
// l'editing". The program never starts developing anything without that
// click (section 7.6), so the button is always visible and never automatic.
//
// Keys, from section 7.6: X discards, P keeps, ←/→ move, ↑/↓ change the
// proposed frame of a burst, Space compares, Ctrl+Z undoes.
//
// The screen opens on any project, whatever was chosen at import: section 7.1
// makes culling reversible, so a project that went "straight to editing" can
// come here later, and the photos that were never analysed are queued for it
// the moment the screen sees them.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Columns2, Combine, Undo2 } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { visibleCards, type CullFilter, type CullSort } from '../lib/culling'
import { useCulling } from '../lib/useCulling'
import type { CullPhoto, UserDecision } from '../lib/cullTypes'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { BurstStrip } from '../components/culling/BurstStrip'
import { CompareView } from '../components/culling/CompareView'
import { CullCard } from '../components/culling/CullCard'
import { CullingPanel } from '../components/culling/CullingPanel'
import { WhyPanel } from '../components/culling/WhyPanel'
import { keyBelongsToControl } from '../lib/shortcuts'

export function CullingPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const culling = useCulling(id)
  const { state } = culling

  const [filter, setFilter] = useState<CullFilter>('all')
  const [sort, setSort] = useState<CullSort>('time')
  const [currentId, setCurrentId] = useState<number | null>(null)
  const [openBurst, setOpenBurst] = useState<number[] | null>(null)
  const [comparing, setComparing] = useState(false)

  const cards = useMemo(() => (state ? visibleCards(state, filter, sort) : []), [filter, sort, state])
  const current = (currentId !== null && state?.photos.get(currentId)) || cards[0]?.photo || null
  const burstOf = useCallback(
    (photoId: number) => {
      if (!state) return null
      for (const members of state.bursts.values()) if (members.includes(photoId)) return members
      return null
    },
    [state],
  )

  // Photos never analysed -- imported "straight to editing", or added by a
  // later import -- are queued as soon as the screen sees them.
  const summary = state?.view.summary
  const [started, setStarted] = useState(false)
  useEffect(() => {
    if (!summary || started) return
    if (summary.analysed + summary.failed < summary.total && summary.pending === 0) {
      setStarted(true)
      void api.startCulling(id).then(() => culling.refresh())
    }
  }, [culling, id, started, summary])

  const confirm = useMutation({
    mutationFn: () => api.confirmCulling(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['project', id] })
      void queryClient.invalidateQueries({ queryKey: ['photos', id] })
      navigate(`/progetti/${id}/foto`)
    },
  })

  const decide = useCallback(
    (photo: CullPhoto, decision: UserDecision) =>
      culling.decide([{ photo_id: photo.id, decision }]),
    [culling],
  )

  /**
   * A new proposed frame for a burst: keep it, and hand the frames the user
   * had kept by hand back to the automatic selection -- which, with a frame of
   * the burst now chosen, makes them duplicates. Handing back rather than
   * discarding keeps a string of ↑/↓ presses from freezing a trail of "no"s,
   * and the batch is computed when it is sent, so presses queued faster than
   * the server answers still land on the right frames.
   */
  const choose = useCallback(
    (photo: CullPhoto) =>
      void culling.decide((current) => {
        const members = [...current.bursts.values()].find((m) => m.includes(photo.id)) ?? []
        const batch: Array<{ photo_id: number; decision: UserDecision }> = [
          { photo_id: photo.id, decision: 'keep' },
        ]
        for (const member of members) {
          if (member !== photo.id && current.photos.get(member)!.decided_by === 'user') {
            batch.push({ photo_id: member, decision: 'auto' })
          }
        }
        return batch
      }),
    [culling],
  )

  const move = useCallback(
    (delta: number) => {
      if (!cards.length || !state) return
      let index = cards.findIndex(
        (card) => card.photo.id === current?.id || card.burst?.includes(current?.id ?? -1),
      )
      let target = index + delta
      if (index < 0 && current) {
        // The current photo left the visible cards -- discarded under "solo
        // selezionate" -- so step from where it stood in shooting order.
        const position = state.order.indexOf(current.id)
        const at = (card: (typeof cards)[number]) => state.order.indexOf(card.photo.id)
        index = cards.findIndex((card) => at(card) > position)
        if (index < 0) index = cards.length
        target = delta > 0 ? index : index - 1
      }
      const next = cards[Math.min(cards.length - 1, Math.max(0, target))]
      setCurrentId(next.photo.id)
      document
        .querySelector(`[data-photo="${next.photo.id}"]`)
        ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    },
    [cards, current, state],
  )

  const compared = useMemo(() => {
    if (!current || !state) return []
    const members = burstOf(current.id)
    if (members) return members.slice(0, 4).map((member) => state.photos.get(member)!)
    const index = cards.findIndex((card) => card.photo.id === current.id)
    const next = cards[index + 1]?.photo ?? cards[index - 1]?.photo
    return next ? [current, next] : [current]
  }, [burstOf, cards, current, state])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (keyBelongsToControl(event)) return
      if (!current) return
      const key = event.key
      if ((event.ctrlKey || event.metaKey) && (key === 'z' || key === 'Z')) {
        event.preventDefault()
        void culling.undo()
      } else if (key === ' ') {
        event.preventDefault()
        setComparing((open) => !open && compared.length > 1)
      } else if (key === 'Escape') {
        setComparing(false)
        setOpenBurst(null)
      } else if (comparing) {
        return
      } else if (key === 'x' || key === 'X') {
        void decide(current, 'discard')
      } else if (key === 'p' || key === 'P') {
        void decide(current, 'keep')
      } else if (key === 'ArrowLeft' && !event.altKey) {
        event.preventDefault()
        move(-1)
      } else if (key === 'ArrowRight' && !event.altKey) {
        event.preventDefault()
        move(1)
      } else if ((key === 'ArrowUp' || key === 'ArrowDown') && state) {
        const members = burstOf(current.id)
        if (!members) return
        event.preventDefault()
        // Shooting order, not rank: the rank changes with every choice, and
        // stepping through it would bounce between the same two frames.
        const frames = [...members].sort((a, b) => state.order.indexOf(a) - state.order.indexOf(b))
        const position = frames.indexOf(current.id)
        const next = frames[(position + (key === 'ArrowDown' ? 1 : frames.length - 1)) % frames.length]
        choose(state.photos.get(next)!)
        setCurrentId(next)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [burstOf, choose, compared.length, comparing, culling, current, decide, move, state])

  const analysing = summary ? summary.pending > 0 : false
  const burstMembers = openBurst && state ? openBurst.map((member) => state.photos.get(member)!) : null
  const currentBurst = current ? burstOf(current.id) : null

  return (
    <AppShell
      back={{ to: '/', label: t('nav.projects') }}
      title={project.data?.name}
      actions={
        <>
          <Button size="sm" variant="ghost" disabled={!culling.canUndo} onClick={() => void culling.undo()} title="Ctrl+Z">
            <Undo2 size={14} />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={compared.length < 2}
            onClick={() => setComparing(true)}
            title={t('culling.compare.title')}
          >
            <Columns2 size={14} />
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/fusioni`)}>
            <Combine size={14} />
            {t('nav.merges')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/foto`)}>
            {t('culling.toViewer')}
          </Button>
        </>
      }
    >
      <div className="grid h-full grid-cols-[1fr_minmax(16rem,20rem)] overflow-hidden">
        <section className="flex min-h-0 flex-col">
          <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-ink-700 bg-ink-900 px-3 py-1.5 text-xs text-ink-300">
            {(['all', 'selected', 'culled'] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setFilter(value)}
                className={cn(
                  'rounded px-2 py-0.5',
                  filter === value ? 'bg-ink-700 text-ink-50' : 'hover:bg-ink-800 hover:text-ink-100',
                )}
              >
                {t(`culling.filter.${value}`)}
              </button>
            ))}
            <span className="mx-1 h-4 w-px bg-ink-700" />
            <label className="flex items-center gap-1">
              {t('culling.sort')}
              <select
                value={sort}
                onChange={(event) => setSort(event.target.value as CullSort)}
                className="rounded border border-ink-600 bg-ink-800 px-1 py-0.5 text-ink-100"
              >
                <option value="time">{t('culling.sort.time')}</option>
                <option value="score">{t('culling.sort.score')}</option>
                <option value="name">{t('culling.sort.name')}</option>
              </select>
            </label>
            {analysing && summary ? (
              <span className="ml-auto tabular-nums text-ink-200">
                {t('culling.analysing', { done: summary.analysed, total: summary.total })}
              </span>
            ) : (
              <span className="ml-auto hidden text-ink-500 xl:inline">{t('culling.shortcuts')}</span>
            )}
          </div>

          {burstMembers ? (
            <BurstStrip
              members={burstMembers}
              currentId={current?.id ?? null}
              onSelect={(photo) => setCurrentId(photo.id)}
              onChoose={choose}
              onDecide={(photo, decision) => void decide(photo, decision)}
              onClose={() => setOpenBurst(null)}
            />
          ) : null}

          <div className="min-h-0 flex-1 overflow-y-auto bg-ink-900 p-2">
            {culling.loading ? <p className="p-2 text-sm text-ink-400">{t('common.loading')}</p> : null}
            {culling.loadError ? <p className="p-2 text-sm text-bad">{culling.loadError}</p> : null}
            {state && cards.length === 0 ? (
              <p className="p-2 text-sm text-ink-400">{t('culling.empty')}</p>
            ) : null}
            <ul className="grid grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] gap-1.5">
              {cards.map((card) => (
                <CullCard
                  key={card.photo.id}
                  photo={card.photo}
                  burst={card.burst}
                  burstKept={card.burst?.filter((member) => !state!.photos.get(member)!.culled).length ?? 0}
                  merge={state?.merges.get(card.photo.id)}
                  current={card.photo.id === current?.id}
                  onSelect={() => setCurrentId(card.photo.id)}
                  onOpenBurst={() => setOpenBurst(card.burst)}
                />
              ))}
            </ul>
          </div>

          <footer className="flex shrink-0 items-center gap-3 border-t border-ink-700 bg-ink-850 px-3 py-2">
            {summary ? (
              <span className="text-sm text-ink-100">
                {t('culling.bottom.count', { selected: summary.selected, total: summary.total })}
              </span>
            ) : null}
            {culling.error ? <span className="text-xs text-bad">{culling.error}</span> : null}
            {confirm.error ? (
              <span className="text-xs text-bad">{(confirm.error as ApiError).message}</span>
            ) : null}
            <span className="ml-auto text-xs text-ink-400">
              {analysing ? t('culling.bottom.wait') : t('culling.bottom.hint')}
            </span>
            <Button
              variant="primary"
              disabled={!summary || analysing || confirm.isPending || summary.selected === 0}
              onClick={() => confirm.mutate()}
            >
              {t('culling.bottom.proceed')}
              <ArrowRight size={14} />
            </Button>
          </footer>
        </section>

        <aside className="flex min-h-0 flex-col border-l border-ink-700 bg-ink-900">
          {state ? (
            <>
              <div className="min-h-0 flex-1">
                <CullingPanel
                  settings={state.view.settings}
                  availability={state.view.availability}
                  summary={state.view.summary}
                  latencyMs={culling.latencyMs}
                  onChange={(patch) => void culling.updateSettings(patch)}
                />
              </div>
              {current ? (
                <WhyPanel
                  photo={current}
                  settings={state.view.settings}
                  merge={state.merges.get(current.id)}
                  burstPosition={
                    currentBurst ? [currentBurst.indexOf(current.id) + 1, currentBurst.length] : null
                  }
                />
              ) : null}
            </>
          ) : null}
        </aside>
      </div>

      {comparing && compared.length > 1 ? (
        <CompareView
          photos={compared}
          onDecide={(photo, decision) => void decide(photo, decision)}
          onClose={() => setComparing(false)}
        />
      ) : null}
    </AppShell>
  )
}
