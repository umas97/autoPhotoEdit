// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The viewer: grid on the left, photograph in the middle, parameters on the
// right (section 10). It is the screen phase 3 is judged on, and two of its
// behaviours are the judgement:
//
// **The slider under 150 ms.** Moving a handle sets the parameters and marks the
// gesture as in progress; usePreview renders at 1024 px while that is true and
// at 2048 px once it is not. The measured round trip is shown under the image,
// because a number that is claimed in a specification should be visible in the
// program.
//
// **The preview equals the export.** There is no client-side approximation
// anywhere in this file: every pixel comes from the same pipeline the exporter
// uses, through /api/photos/{id}/preview. The only difference between what is
// on screen and what a later export writes is the size, and test 1 of section
// 13 is what says the two agree.
//
// "Before" is the cached proxy: the neutral development of the same RAW. Not a
// second render -- it is already on disk, it is what the import produced, and
// holding a key should not cost a second of CPU.
//
// The photo and what is drawn over it are components/viewer/ViewerStage.tsx;
// the right-hand column, adjustments and masks, components/masks/EditSidebar.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ClipboardCheck, Combine, Download, History, Layers, ListFilter, Palette, Redo2, RotateCcw, Undo2 } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { neutralParams } from '../lib/params'
import { useEditState } from '../lib/useEditState'
import { usePreview } from '../lib/usePreview'
import { useUi } from '../lib/store'
import type { EditParams, Photo } from '../lib/types'
import { t } from '../i18n/it'
import { AppShell } from '../components/AppShell'
import { GeometryPanel } from '../components/GeometryPanel'
import { useCropEditor, useShownParams } from '../components/crop'
import { EditSidebar } from '../components/masks/EditSidebar'
import { handleMaskKey } from '../components/masks/shortcuts'
import { useMaskEditor } from '../components/masks/useMaskEditor'
import { useRetouchWatch } from '../components/retouch/useRetouch'
import { PhotoGrid } from '../components/PhotoGrid'
import { PickBar, usePicked } from '../components/merges/PickBar'
import { SaveStatus } from '../components/SaveStatus'
import { SnapshotsDialog } from '../components/SnapshotsDialog'
import { StylePanel } from '../components/StylePanel'
import { Button } from '../components/ui/Button'
import { Checkbox } from '../components/ui/Field'
import { ViewerStage } from '../components/viewer/ViewerStage'
import { keyBelongsToControl } from '../lib/shortcuts'

export function ViewerPage() {
  const { projectId } = useParams()
  const id = Number(projectId)
  const navigate = useNavigate()

  const selectedByProject = useUi((state) => state.selectedPhoto)
  const select = useUi((state) => state.select)
  const selectedId = selectedByProject[id] ?? null

  const project = useQuery({ queryKey: ['project', id], queryFn: () => api.readProject(id) })
  // Section 7.5: the discarded photos stay in the project, behind "Mostra
  // scarti". Hidden by default once a culling has chosen them; shown when the
  // project never had one, since then there is nothing to hide.
  const [showCulledChoice, setShowCulled] = useState<boolean | null>(null)
  const showCulled = showCulledChoice ?? !project.data?.culling_enabled
  // Section 25.1: the frames an accepted merge stands in for, behind a filter.
  const [showSources, setShowSources] = useState(false)
  const photos = useQuery({
    queryKey: ['photos', id, showCulled, showSources],
    queryFn: () => api.listPhotos(id, 0, 500, showCulled, showSources),
    enabled: project.data !== undefined,
    refetchInterval: (query) =>
      // While proxies are still being built the grid is incomplete; once every
      // photo has one, stop asking.
      // A discarded photo never gets one: it must not keep the polling alive.
      query.state.data?.items.some((photo) => !photo.has_proxy && !photo.missing && !photo.culled)
        ? 3_000
        : false,
  })

  const items = useMemo(() => photos.data?.items ?? [], [photos.data])
  const picking = usePicked(items)
  const hasMerges = showSources || items.some((photo) => photo.merge !== null)
  const current = items.find((photo) => photo.id === selectedId) ?? items[0] ?? null

  useEffect(() => {
    if (current && current.id !== selectedId) select(id, current.id)
  }, [current, id, select, selectedId])

  const detail = useQuery({
    queryKey: ['photo', current?.id],
    queryFn: () => api.readPhoto(current!.id),
    enabled: current !== null,
  })

  // --- the edit under the user's hand ---------------------------------------
  const { params, dragging, onChange, stepBack, stepForward, saveState, flush } = useEditState(
    detail.data,
    current?.id ?? null,
  )
  const editor = useMaskEditor(current?.id ?? null, params?.masks.length ?? 0)
  const crop = useCropEditor(current, params, onChange)
  const [showBefore, setShowBefore] = useState(false)
  const [snapshotsOpen, setSnapshotsOpen] = useState(false)
  const imageRef = useRef<HTMLImageElement | null>(null)

  // "This profile vs neutral" (section 22): a preview-only substitute.
  const [compare, setCompare] = useState<EditParams | null>(null)
  useEffect(() => setCompare(null), [current?.id])
  const revision = useRetouchWatch(id, current?.id ?? null)
  const shown = useShownParams(compare ?? params, crop)
  const preview = usePreview(current?.id ?? null, shown, dragging, {
    retouch: editor.showRemovals,
    revision,
  })

  const step = useCallback(
    (delta: number) => {
      if (!current) return
      const index = items.findIndex((photo) => photo.id === current.id)
      const next = items[index + delta]
      if (next) select(id, next.id)
    },
    [current, id, items, select],
  )

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (keyBelongsToControl(event) || snapshotsOpen) return
      const command = event.ctrlKey || event.metaKey

      if (handleMaskKey(event, editor, params, onChange)) {
        event.preventDefault()
      } else if (event.key === '\\') {
        event.preventDefault()
        setShowBefore(true)
      } else if (event.key === 'ArrowLeft' && !event.altKey) {
        step(-1)
      } else if (event.key === 'ArrowRight' && !event.altKey) {
        step(1)
      } else if (command && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        if (event.shiftKey) stepForward()
        else stepBack()
      } else if (command && event.key.toLowerCase() === 'y') {
        event.preventDefault()
        stepForward()
      } else if (command && event.key === 's') {
        // Saving happens by itself; Ctrl+S only stops waiting for the coalescence.
        event.preventDefault()
        void flush()
      }
    }
    const up = (event: KeyboardEvent) => {
      if (event.key === '\\') setShowBefore(false)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [editor, flush, onChange, params, snapshotsOpen, step, stepBack, stepForward])

  return (
    <AppShell
      back={{ to: '/', label: t('nav.projects') }}
      title={project.data?.name}
      actions={
        <>
          <Checkbox
            checked={showCulled}
            onChange={(checked) => setShowCulled(checked)}
            label={t('viewer.showCulled')}
          />
          {hasMerges ? (
            <Checkbox checked={showSources} onChange={setShowSources} label={t('merges.showSources')} />
          ) : null}
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/cernita`)}>
            <ListFilter size={14} />
            {t('nav.culling')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/fusioni`)}>
            <Combine size={14} />
            {t('nav.merges')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/scene`)}>
            <Layers size={14} />
            {t('nav.scenes')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/stile`)}>
            <Palette size={14} />
            {t('nav.projectStyle')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/revisione`)}>
            <ClipboardCheck size={14} />
            {t('nav.review')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(`/progetti/${id}/export`)}>
            <Download size={14} />
            {t('nav.export')}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setSnapshotsOpen(true)}>
            <History size={14} />
            {t('snapshots.open')}
          </Button>
          <SaveStatus state={saveState} onRetry={() => void flush()} />
          <Button size="sm" variant="ghost" onClick={stepBack} title={`${t('viewer.undo')} (Ctrl+Z)`}>
            <Undo2 size={14} />
          </Button>
          <Button size="sm" variant="ghost" onClick={stepForward}
            title={`${t('viewer.redo')} (Ctrl+Shift+Z)`}>
            <Redo2 size={14} />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            title={t('viewer.resetAll')}
            onClick={() => onChange(neutralParams(), true)}
          >
            <RotateCcw size={14} />
          </Button>
        </>
      }
    >
      <div className="grid h-full grid-cols-[minmax(12rem,18rem)_1fr_minmax(15rem,20rem)] overflow-hidden">
        <aside className="min-h-0 overflow-y-auto border-r border-ink-700 bg-ink-900">
          <div className="sticky top-0 z-10 bg-ink-900">
            <div className="px-2 py-1.5 text-xs text-ink-400">
              {t('photos.count', { shown: items.length, total: photos.data?.total ?? 0 })}
            </div>
            <PickBar projectId={id} picked={picking.picked} onClear={picking.clear} />
          </div>
          <PhotoGrid
            projectId={id}
            photos={items}
            selectedId={current?.id ?? null}
            onSelect={(photo: Photo) => select(id, photo.id)}
            picked={picking.set}
            onPick={picking.pick}
          />
        </aside>

        <ViewerStage projectId={id} current={current} detail={detail.data} params={params}
          onChange={onChange} preview={preview} comparing={compare !== null}
          showBefore={showBefore} dragging={dragging} flush={flush} editor={editor}
          crop={crop} imageRef={imageRef} />

        <aside className="min-h-0 border-l border-ink-700">
          {params && current ? (
            <EditSidebar
              params={params}
              onChange={onChange}
              editor={editor}
              photo={current}
              disabled={current.missing}
              retouch={{ projectId: id, flush, copyTargets: picking.picked }}
              header={
                <>
                  <StylePanel key={current.id} photoId={current.id} onCompare={setCompare} />
                  <GeometryPanel
                    params={params}
                    analysis={detail.data?.analysis ?? null}
                    onChange={onChange}
                    onAssociateLens={() => navigate(`/progetti/${id}/scene`)}
                    crop={crop}
                    disabled={current.missing}
                  />
                </>
              }
            />
          ) : (
            <p className="p-3 text-sm text-ink-400">
              {detail.error ? (detail.error as ApiError).message : t('common.loading')}
            </p>
          )}
        </aside>
      </div>
      <SnapshotsDialog projectId={id} open={snapshotsOpen} onOpenChange={setSnapshotsOpen} flush={flush} />
    </AppShell>
  )
}
