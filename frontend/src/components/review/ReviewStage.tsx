// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The photograph in the middle of the review (section 10): the edit, and on
// request the three ways of judging it the specification lists --
//
//   * **prima/dopo** (held \ or Space): the neutral proxy in place of the edit;
//   * **zoom 1:1**: the render at its own pixels, panned by scrolling. It is the
//     2048 px development preview, not the sensor: section 26 keeps the full
//     demosaic for the export, and 2048 px is enough to judge colour and tone,
//     which is what the review decides;
//   * **affiancato**: before and after side by side.
//
// Every pixel of "after" comes from the pipeline through the preview endpoint,
// exactly as in the viewer.
import { useEffect, useRef, type MutableRefObject, type ReactNode } from 'react'
import { proxyUrl } from '../../lib/api'
import type { ReviewPhoto } from '../../lib/reviewTypes'
import { t } from '../../i18n/it'
import { cn } from '../../lib/utils'

function Label({ children, visible = true }: { children: ReactNode; visible?: boolean }) {
  return (
    <span
      className={cn(
        'pointer-events-none absolute left-2 top-2 rounded bg-ink-950/70 px-2 py-0.5 text-xs text-ink-100',
        'transition-opacity',
        visible ? 'opacity-100' : 'opacity-0',
      )}
    >
      {children}
    </span>
  )
}

export function ReviewStage({
  photo,
  previewUrl,
  showBefore,
  zoom,
  compare,
  imageRef,
  overlay,
}: {
  photo: ReviewPhoto
  previewUrl: string | null
  showBefore: boolean
  zoom: boolean
  compare: boolean
  imageRef: MutableRefObject<HTMLImageElement | null>
  /** Drawn over the single, fitted image only: the crop proposal. */
  overlay?: ReactNode
}) {
  const before = photo.has_proxy ? proxyUrl(photo) : ''
  const after = previewUrl ?? before
  const scroller = useRef<HTMLDivElement>(null)

  // Entering 1:1 starts at the centre of the frame, where the subject usually is.
  useEffect(() => {
    const element = scroller.current
    if (!zoom || !element) return
    const centre = () => {
      element.scrollLeft = (element.scrollWidth - element.clientWidth) / 2
      element.scrollTop = (element.scrollHeight - element.clientHeight) / 2
    }
    const image = element.querySelector('img')
    if (image && !image.complete) image.addEventListener('load', centre, { once: true })
    else centre()
  }, [zoom, photo.id])

  if (compare) {
    return (
      <div className="grid h-full grid-cols-2 gap-2">
        <div className="relative min-h-0">
          <img src={before} alt={photo.filename} className="h-full w-full object-contain" />
          <Label>{t('review.before')}</Label>
        </div>
        <div className="relative min-h-0">
          <img src={after} alt={photo.filename} className="h-full w-full object-contain" />
        </div>
      </div>
    )
  }

  if (zoom) {
    return (
      <div ref={scroller} className="relative h-full overflow-auto">
        <img
          src={showBefore ? before : after}
          alt={photo.filename}
          className="max-w-none"
          title={t('review.zoomHint')}
        />
        <Label visible={showBefore}>{t('review.before')}</Label>
      </div>
    )
  }

  return (
    <div className="relative h-full">
      <img
        ref={imageRef}
        key={photo.id}
        src={showBefore ? before : after}
        alt={photo.filename}
        className="mx-auto h-full w-full object-contain"
      />
      {!showBefore ? overlay : null}
      <Label visible={showBefore}>{t('review.before')}</Label>
    </div>
  )
}
