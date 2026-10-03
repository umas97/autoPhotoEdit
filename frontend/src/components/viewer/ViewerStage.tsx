// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The middle of the viewer: the photograph, what is drawn over it -- the crop
// proposal or the crop being drawn, the masks' handles and selection, the
// banners -- and the line of
// facts under it with the measured render time (section 10).
//
// Every pixel of the photo comes from the pipeline through the preview
// endpoint; "before" is the cached neutral proxy, not a second render.
import type { MutableRefObject } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, proxyUrl, thumbUrl } from '../../lib/api'
import type { PreviewState } from '../../lib/usePreview'
import type { EditParams, Photo, PhotoDetail } from '../../lib/types'
import { t } from '../../i18n/it'
import { cn, formatShutter } from '../../lib/utils'
import { CropProposalLayer } from '../CropOverlay'
import { CropTool, type CropEditor } from '../crop'
import { MaskLayer } from '../masks/MaskLayer'
import type { MaskEditor } from '../masks/useMaskEditor'
import { Button } from '../ui/Button'
import { VersionStrip } from '../VersionStrip'

export function ViewerStage({
  projectId,
  current,
  detail,
  params,
  onChange,
  preview,
  comparing,
  showBefore,
  dragging,
  flush,
  editor,
  crop,
  imageRef,
}: {
  projectId: number
  current: Photo | null
  detail: PhotoDetail | undefined
  params: EditParams | null
  onChange: (next: EditParams, commit: boolean) => void
  preview: PreviewState
  /** The neutral comparison of the style panel is on screen, not the edit. */
  comparing: boolean
  showBefore: boolean
  dragging: boolean
  /** Write the edit waiting for the autosave, before an action that reads it. */
  flush: () => Promise<void>
  editor: MaskEditor
  crop: CropEditor
  imageRef: MutableRefObject<HTMLImageElement | null>
}) {
  const queryClient = useQueryClient()
  const recover = useMutation({
    mutationFn: (photoId: number) =>
      api.cullDecisions(projectId, [{ photo_id: photoId, decision: 'keep' }]),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['photos', projectId] })
      void queryClient.invalidateQueries({ queryKey: ['culling', projectId] })
    },
  })
  const developed = !showBefore && preview.url !== null
  const own = detail && current && detail.id === current.id ? detail : null

  return (
    <section className="flex min-h-0 flex-col bg-mat">
      <div className="relative min-h-0 flex-1 p-3">
        {current === null ? (
          <p className="grid h-full place-items-center text-sm text-ink-400">{t('viewer.noPhoto')}</p>
        ) : (
          <>
            <img
              ref={imageRef}
              key={current.id}
              src={
                showBefore && current.has_proxy
                  ? proxyUrl(current)
                  : (preview.url ??
                    (current.has_proxy
                      ? proxyUrl(current)
                      : current.has_thumb
                        ? thumbUrl(current, 'full')
                        : ''))
              }
              alt={current.filename}
              className="mx-auto h-full w-full object-contain"
            />
            {own ? (
              <CropProposalLayer
                detail={own}
                params={params}
                flush={flush}
                visible={developed && !dragging && editor.tab !== 'masks' && !crop.active}
                imageRef={imageRef}
                onUpdated={(updated) => queryClient.setQueryData(['photo', updated.id], updated)}
              />
            ) : null}
            {params && !current.missing ? (
              <MaskLayer editor={editor} photo={current} params={params} onChange={onChange}
                imageRef={imageRef} dragging={dragging}
                visible={developed && !comparing && !crop.active} flush={flush} />
            ) : null}
            {showBefore ? null : <CropTool editor={crop} imageRef={imageRef} />}
            {detail?.crop_proposals_paused ? (
              <div className="absolute bottom-5 left-5 flex items-center gap-2 rounded bg-ink-950/80 px-2 py-1 text-xs text-ink-200">
                {t('crop.paused')}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    api.resumeCropProposals(projectId).then(() =>
                      queryClient.invalidateQueries({ queryKey: ['photo', current.id] }),
                    )
                  }
                >
                  {t('crop.resume')}
                </Button>
              </div>
            ) : null}
            {current.culled ? (
              <div className="absolute right-5 top-5 flex items-center gap-2 rounded bg-ink-950/80 px-2 py-1 text-xs text-ink-100">
                {t('viewer.culledBanner')}
                <Button size="sm" variant="outline" disabled={recover.isPending}
                  onClick={() => recover.mutate(current.id)}>
                  {t('viewer.recover')}
                </Button>
              </div>
            ) : null}
            <span
              className={cn(
                'pointer-events-none absolute left-5 top-5 rounded bg-ink-950/70 px-2 py-0.5',
                'text-xs text-ink-100 transition-opacity',
                showBefore ? 'opacity-100' : 'opacity-0',
              )}
            >
              {t('viewer.before')}
            </span>
          </>
        )}
      </div>

      <footer className="flex shrink-0 items-center gap-3 border-t border-ink-700 bg-ink-900 px-3 py-1.5 text-xs text-ink-400">
        <span className="truncate text-ink-200">{current?.filename}</span>
        {current?.camera ? (
          <span className="truncate">
            {t('viewer.exif', { camera: current.camera, lens: current.lens ?? '—' })}
          </span>
        ) : null}
        {current?.iso ? (
          <span className="truncate tabular-nums">
            {t('viewer.exposureLine', {
              iso: current.iso,
              aperture: current.aperture?.toFixed(1) ?? '—',
              shutter: formatShutter(current.shutter),
              focal: current.focal_length?.toFixed(0) ?? '—',
            })}
          </span>
        ) : null}
        <span className="ml-auto tabular-nums">
          {preview.rendering
            ? t('viewer.rendering')
            : preview.elapsedMs !== null
              ? t('viewer.renderMs', { ms: preview.elapsedMs, edge: preview.edge })
              : ''}
        </span>
        {preview.error ? <span className="text-bad">{preview.error}</span> : null}
        <span className="hidden text-ink-500 xl:inline">{t('viewer.shortcutsBody')}</span>
      </footer>

      {detail && detail.versions.length > 0 ? (
        <VersionStrip
          photoId={detail.id}
          versions={detail.versions}
          beforeRestore={flush}
          onRestored={() => queryClient.invalidateQueries({ queryKey: ['photo', current?.id] })}
        />
      ) : null}
    </section>
  )
}
