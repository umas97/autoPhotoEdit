// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The review, "la schermata principale" of section 10: scenes or queue on the
// left, the photograph in the middle, the parameters on the right.
//
// **Scenes** (section 9.2.2) are judged on their representative. Its sliders
// are the scene's: "Applica alla scena" sends the correction to every photo of
// it as a delta, A approves the scene (applying a pending correction first), R
// sends the whole scene to the individual queue. Clicking another photo of the
// scene inspects it, read-only: the scene speaks through its representative.
//
// **The queue** (section 9.2.3) holds the photos under the confidence threshold,
// lowest first, each with its reasons and two or three variants on keys 1-3.
// A approves what is on screen; R puts the style back to Neutro automatico and
// keeps the photo in the queue, to be fixed by hand.
//
// **The grid** shows every scene at once, developed, so that the scenes that
// look right are approved together (components/review/SceneGrid.tsx); the
// left column's thumbnails and the scene strip are developed too.
//
// The sliders save by themselves (lib/useEditState.ts). Ctrl+Z first steps
// back through the slider gestures made since the last review action, then
// undoes that action -- which the server turns into new versions, so that the
// history of section 23 only ever grows. A correction of a representative is
// measured from the version the screen started from, not from the current one,
// which the autosave has already made the correction.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, GraduationCap, History, SlidersHorizontal } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { reviewApi } from '../lib/reviewApi'
import { useEditState } from '../lib/useEditState'
import { usePreview } from '../lib/usePreview'
import type { EditParams } from '../lib/types'
import type { UndoRecord } from '../lib/reviewTypes'
import { t } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { CropProposalLayer } from '../components/CropOverlay'
import { GeometryPanel } from '../components/GeometryPanel'
import { EditSidebar } from '../components/masks/EditSidebar'
import { MaskLayer } from '../components/masks/MaskLayer'
import { handleMaskKey } from '../components/masks/shortcuts'
import { useMaskEditor } from '../components/masks/useMaskEditor'
import { useRetouchWatch } from '../components/retouch/useRetouch'
import { SnapshotsDialog } from '../components/SnapshotsDialog'
import { StylePanel } from '../components/StylePanel'
import { Button } from '../components/ui/Button'
import { ConfidenceCard } from '../components/review/ConfidenceCard'
import { ReviewActions, type ReviewAction } from '../components/review/ReviewActions'
import { FeedbackDialog, ReviewSettingsDialog } from '../components/review/ReviewDialogs'
import { SceneStrip } from '../components/review/ReviewList'
import { ReviewSidebar, type ReviewTab as Tab } from '../components/review/ReviewSidebar'
import { ReviewStage } from '../components/review/ReviewStage'
import { SceneGrid } from '../components/review/SceneGrid'
import { VariantStrip } from '../components/review/VariantStrip'
import { keyBelongsToControl } from '../lib/shortcuts'

export function ReviewPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  const overview = useQuery({
    queryKey: ['review', id],
    queryFn: () => reviewApi.review(id),
    // While predictions or developed renders are on their way.
    refetchInterval: (query) =>
      (query.state.data?.pending ?? 0) > 0 || (query.state.data?.developing ?? 0) > 0 ? 3_000 : false,
  })
  const data = overview.data
  const photos = useMemo(() => new Map((data?.photos ?? []).map((p) => [p.id, p])), [data])

  const [tab, setTab] = useState<Tab>('grid')
  const [cluster, setCluster] = useState<number | null>(null)
  const [queued, setQueued] = useState<number | null>(null)
  const [inspect, setInspect] = useState<number | null>(null)
  const scenes = useMemo(() => data?.scenes ?? [], [data])
  const queue = useMemo(() => data?.queue ?? [], [data])
  const scene = scenes.find((s) => s.cluster === cluster) ?? scenes.find((s) => s.state === 'pending') ?? scenes[0]
  const entry = queue.find((q) => q.photo_id === queued) ?? queue[0]
  const photoId = tab === 'scenes' ? (inspect ?? scene?.representative ?? null) : (entry?.photo_id ?? null)
  const photo = photoId !== null ? photos.get(photoId) : undefined
  const inspecting = tab === 'scenes' && inspect !== null && inspect !== scene?.representative

  const detail = useQuery({
    queryKey: ['photo', photoId],
    queryFn: () => api.readPhoto(photoId!),
    enabled: photoId !== null,
  })
  const review = useQuery({
    queryKey: ['photoReview', photoId],
    queryFn: () => reviewApi.photoReview(photoId!),
    enabled: photoId !== null,
  })

  const { params, dragging, onChange, stepBack, stepForward, saveState, flush, base, edited, settle } =
    useEditState(detail.data, photoId)
  const editor = useMaskEditor(photoId, params?.masks.length ?? 0)

  const [compareParams, setCompareParams] = useState<EditParams | null>(null)
  useEffect(() => setCompareParams(null), [photoId])
  const revision = useRetouchWatch(id, photoId)
  const preview = usePreview(photoId, compareParams ?? params, dragging, { retouch: editor.showRemovals, revision })

  const [showBefore, setShowBefore] = useState(false)
  const [zoom, setZoom] = useState(false)
  const [compare, setCompare] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [snapshotsOpen, setSnapshotsOpen] = useState(false)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const undoStack = useRef<UndoRecord[]>([])
  const imageRef = useRef<HTMLImageElement | null>(null)

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['review', id] })
    void queryClient.invalidateQueries({ queryKey: ['photo'] })
    void queryClient.invalidateQueries({ queryKey: ['photoReview'] })
  }

  const act = useMutation({
    mutationFn: async (action: ReviewAction): Promise<{ undo: UndoRecord; propagated?: number }> => {
      // What the autosave still holds reaches the server first: the action
      // reads the saved state, and must not be overtaken by a late save.
      await flush()
      const onScreen = !inspecting && action !== 'reject' ? params : null
      if (tab === 'scenes') {
        if (!scene) throw new Error(t('review.empty'))
        return reviewApi.reviewScene(id, scene.cluster, action, onScreen, onScreen ? base?.versionId : null)
      }
      if (!entry || action === 'apply') throw new Error(t('review.queueEmpty'))
      return reviewApi.reviewPhoto(id, entry.photo_id, action, onScreen)
    },
    onSuccess: (result, action) => {
      settle()
      undoStack.current = [...undoStack.current.slice(-49), result.undo]
      setNotice(
        result.propagated ? t('review.propagated', { count: result.propagated }) : null,
      )
      if (action !== 'apply') {
        // Move on to the next thing to look at; a rejected photo stays, to be fixed.
        if (tab === 'scenes' && scene) {
          const index = scenes.indexOf(scene)
          const next = scenes.slice(index + 1).find((s) => s.state === 'pending') ??
            scenes.find((s) => s.state === 'pending' && s !== scene)
          setCluster(next?.cluster ?? scene.cluster)
          setInspect(null)
        } else if (tab === 'queue' && entry && action === 'approve') {
          const index = queue.indexOf(entry)
          setQueued((queue[index + 1] ?? queue[index - 1])?.photo_id ?? null)
        }
      }
      refresh()
    },
    onError: (error) => setNotice((error as ApiError).message),
  })

  const undo = useCallback(async () => {
    if (stepBack()) return
    const record = undoStack.current.pop()
    if (!record) {
      setNotice(t('review.nothingToUndo'))
      return
    }
    await flush()
    await reviewApi.reviewUndo(id, record)
    setNotice(t('review.undone'))
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flush, id, stepBack])

  const approveMany = useMutation({
    mutationFn: async (clusters: number[]) => {
      await flush()
      return reviewApi.approveScenes(id, clusters)
    },
    onSuccess: (result) => {
      settle()
      undoStack.current = [...undoStack.current.slice(-49), result.undo]
      setNotice(t('review.gridApproved', { count: result.scenes }))
      refresh()
    },
    onError: (error) => setNotice((error as ApiError).message),
  })
  const openScene = useCallback((c: number) => {
    setTab('scenes')
    setCluster(c)
    setInspect(null)
  }, [])

  const variants = useMemo(() => review.data?.variants ?? [], [review.data])

  const step = useCallback(
    (delta: number) => {
      if (tab === 'scenes') {
        const index = scene ? scenes.indexOf(scene) : -1
        const next = scenes[index + delta]
        if (next) {
          setCluster(next.cluster)
          setInspect(null)
        }
      } else {
        const index = entry ? queue.indexOf(entry) : -1
        const next = queue[index + delta]
        if (next) setQueued(next.photo_id)
      }
    },
    [tab, scene, scenes, entry, queue],
  )

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (keyBelongsToControl(event)) return
      if (settingsOpen || feedbackOpen || snapshotsOpen) return
      const key = event.key.toLowerCase()
      if (tab === 'grid') {
        // The grid has keys of its own; Ctrl+Z is the page's.
        if ((event.ctrlKey || event.metaKey) && key === 'z') {
          event.preventDefault()
          void undo()
        }
        return
      }
      if (handleMaskKey(event, editor, inspecting ? null : params, onChange)) {
        event.preventDefault()
      } else if (event.key === '\\' || event.key === ' ') {
        event.preventDefault()
        setShowBefore(true)
      } else if (event.key === 'ArrowLeft' && !event.altKey) {
        step(-1)
      } else if (event.key === 'ArrowRight' && !event.altKey) {
        step(1)
      } else if ((event.ctrlKey || event.metaKey) && key === 'z') {
        event.preventDefault()
        if (event.shiftKey) stepForward()
        else void undo()
      } else if (event.ctrlKey || event.metaKey || event.altKey) {
        return
      } else if (key === 'a' && !act.isPending) {
        act.mutate('approve')
      } else if (key === 'r' && !act.isPending) {
        act.mutate('reject')
      } else if (key === 'z') {
        setZoom((on) => !on)
        setCompare(false)
      } else if (key === 'c') {
        setCompare((on) => !on)
        setZoom(false)
      } else if (['1', '2', '3'].includes(event.key) && tab === 'queue') {
        const variant = variants[Number(event.key) - 1]
        if (variant) onChange(variant.params, true)
      }
    }
    const up = (event: KeyboardEvent) => {
      if (event.key === '\\' || event.key === ' ') setShowBefore(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [act, editor, feedbackOpen, inspecting, onChange, params, settingsOpen, snapshotsOpen, step, stepForward, tab, undo, variants])

  const threshold = data?.threshold ?? 0.55
  const noStyle = data !== undefined && data.profile === null

  return (
    <AppShell
      back={{ to: `/progetti/${id}/foto`, label: t('nav.viewer') }}
      title={project.data?.name}
      actions={
        data?.profile ? (
          <>
            <Button size="sm" variant="ghost" onClick={() => setSettingsOpen(true)}>
              <SlidersHorizontal size={14} />
              {t('review.settings')}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setSnapshotsOpen(true)}>
              <History size={14} />
              {t('snapshots.open')}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setFeedbackOpen(true)}>
              <GraduationCap size={14} />
              {t('review.feedback', { count: data.feedback.proposed })}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/export`)}>
              <Download size={14} />
              {t('nav.export')}
            </Button>
          </>
        ) : null
      }
    >
      {noStyle ? (
        <div className="grid h-full place-items-center">
          <div className="max-w-md space-y-3 text-center">
            <p className="text-sm text-ink-200">{t('review.noStyle')}</p>
            <Button variant="primary" onClick={() => navigate(`/progetti/${id}/stile`)}>
              {t('review.chooseStyle')}
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid h-full grid-cols-[minmax(14rem,19rem)_1fr_minmax(15rem,20rem)] overflow-hidden">
          <ReviewSidebar tab={tab} onTab={setTab} review={data} scenes={scenes} queue={queue} photos={photos}
            selectedScene={tab === 'scenes' ? (scene?.cluster ?? null) : null} selectedPhoto={entry?.photo_id ?? null}
            threshold={threshold} onScene={openScene} onPhoto={setQueued} />

          {tab === 'grid' ? (
            <section className="col-span-2 min-h-0 bg-mat">
              <SceneGrid scenes={scenes} photos={photos} threshold={threshold} busy={approveMany.isPending}
                notice={notice}
                onApprove={(clusters) => approveMany.mutate(clusters)} onOpen={openScene} />
            </section>
          ) : (
            <>
            <section className="flex min-h-0 flex-col bg-mat">
              <div className="relative min-h-0 flex-1 p-3">
                {photo ? (
                  <ReviewStage photo={photo} previewUrl={preview.url} showBefore={showBefore} zoom={zoom}
                    compare={compare} imageRef={imageRef}
                    overlay={detail.data && detail.data.id === photo.id ? (<>
                      <CropProposalLayer detail={detail.data} params={params} flush={flush}
                        visible={!dragging && preview.url !== null && editor.tab !== 'masks'} imageRef={imageRef}
                        onUpdated={(updated) => queryClient.setQueryData(['photo', updated.id], updated)} />
                      {params && !inspecting ? (
                        <MaskLayer editor={editor} photo={photo} params={params} onChange={onChange} imageRef={imageRef}
                          dragging={dragging} visible={preview.url !== null && compareParams === null} flush={flush} />
                      ) : null}
                    </>) : null} />
                ) : (
                  <p className="grid h-full place-items-center text-sm text-ink-400">
                    {overview.isLoading ? t('common.loading') : t(tab === 'queue' ? 'review.queueEmpty' : 'review.empty')}
                  </p>
                )}
                {inspecting ? (
                  <div className="absolute left-5 top-5 flex items-center gap-2 rounded bg-ink-950/80 px-2 py-1 text-xs text-ink-100">
                    {t('review.inspecting', { name: photo?.filename ?? '' })}
                    <Button size="sm" variant="outline" onClick={() => setInspect(null)}>
                      {t('review.backToRepresentative')}
                    </Button>
                  </div>
                ) : null}
              </div>

              <ReviewActions scenes={tab === 'scenes'}
                label={`${tab === 'scenes' && scene ? `${t('review.scene', { n: scene.cluster })} · ` : ''}${photo?.filename ?? ''}`}
                notice={notice} saveState={saveState} onRetry={() => void flush()}
                zoom={zoom} onZoom={() => { setZoom(!zoom); setCompare(false) }}
                compare={compare} onCompare={() => { setCompare(!compare); setZoom(false) }}
                hasPhoto={photo !== undefined} canApply={edited && !inspecting} busy={act.isPending}
                onAct={(action) => act.mutate(action)} />

              {tab === 'scenes' && scene && scene.photos.length > 1 ? (
                <SceneStrip scene={scene} photos={photos} inspecting={photoId ?? -1}
                  onInspect={(pid) => setInspect(pid === scene.representative ? null : pid)} />
              ) : null}
              {tab === 'queue' && photoId !== null ? (
                <VariantStrip photoId={photoId} variants={variants} current={params}
                  onChoose={(next) => onChange(next, true)} />
              ) : null}
            </section>

            <aside className="min-h-0 border-l border-ink-700">
              {params && photoId !== null ? (
                <EditSidebar params={params} onChange={onChange} disabled={inspecting} editor={editor} photo={photo ?? { id: photoId, has_proxy: false, proxy_rev: null }} retouch={{ projectId: id, flush }}
                  header={
                    <>
                      <ConfidenceCard review={review.data} />
                      <StylePanel key={photoId} photoId={photoId} onCompare={setCompareParams} />
                      <GeometryPanel params={params} analysis={detail.data?.analysis ?? null} onChange={onChange}
                        onAssociateLens={() => navigate(`/progetti/${id}/scene`)} disabled={inspecting} />
                    </>
                  } />
              ) : (
                <p className="p-3 text-sm text-ink-400">
                  {detail.error ? (detail.error as ApiError).message : overview.error ? (overview.error as ApiError).message : ''}
                </p>
              )}
            </aside>
            </>
          )}
        </div>
      )}
      {data?.profile ? (
        <>
          <ReviewSettingsDialog projectId={id} review={data} open={settingsOpen} onOpenChange={setSettingsOpen} />
          <FeedbackDialog projectId={id} open={feedbackOpen} onOpenChange={setFeedbackOpen} />
          <SnapshotsDialog projectId={id} open={snapshotsOpen} onOpenChange={setSnapshotsOpen}
            flush={flush} />
        </>
      ) : null}
    </AppShell>
  )
}
