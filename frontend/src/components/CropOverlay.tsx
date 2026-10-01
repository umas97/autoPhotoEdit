// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The proposed crop, drawn over the photo with its two answers (section 6.4):
// "Applica crop proposto" and "Scarta". It is a proposal and looks like one --
// a frame and a dimmed outside, never the photo already cropped -- because
// nothing about it is applied until the user says so.
//
// The rectangle is normalised to the straightened frame, which is exactly what
// the preview shows while the photo has no crop of its own; the overlay is only
// drawn in that state. It follows the image as displayed (`object-contain`
// inside its box), so it is recomputed when either changes size.
import { useEffect, useState, type RefObject } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Check, X } from 'lucide-react'
import { api } from '../lib/api'
import type { EditParams, PhotoDetail } from '../lib/types'
import type { CropProposal } from '../lib/analysisTypes'
import { t } from '../i18n/it'
import { Button } from './ui/Button'

interface Box {
  left: number
  top: number
  width: number
  height: number
}

/** Where an `object-contain` image actually sits inside its element. */
function displayedBox(image: HTMLImageElement): Box | null {
  const { naturalWidth, naturalHeight, clientWidth, clientHeight, offsetLeft, offsetTop } = image
  if (!naturalWidth || !naturalHeight || !clientWidth || !clientHeight) return null
  const scale = Math.min(clientWidth / naturalWidth, clientHeight / naturalHeight)
  const width = naturalWidth * scale
  const height = naturalHeight * scale
  return {
    left: offsetLeft + (clientWidth - width) / 2,
    top: offsetTop + (clientHeight - height) / 2,
    width,
    height,
  }
}

export function CropOverlay({
  imageRef,
  proposal,
  onApply,
  onReject,
  busy,
}: {
  imageRef: RefObject<HTMLImageElement | null>
  proposal: CropProposal
  onApply: () => void
  onReject: () => void
  busy?: boolean
}) {
  const [box, setBox] = useState<Box | null>(null)

  useEffect(() => {
    const image = imageRef.current
    if (!image) return
    const update = () => setBox(displayedBox(image))
    update()
    image.addEventListener('load', update)
    const observer = new ResizeObserver(update)
    observer.observe(image)
    return () => {
      image.removeEventListener('load', update)
      observer.disconnect()
    }
  }, [imageRef, proposal.id])

  if (!box) return null
  const { rect } = proposal
  const frame = {
    left: box.left + rect.x * box.width,
    top: box.top + rect.y * box.height,
    width: rect.width * box.width,
    height: rect.height * box.height,
  }
  const aspect =
    proposal.aspect === 'original' || !proposal.aspect
      ? t('crop.aspect.original')
      : proposal.aspect === 'borders'
        ? t('crop.aspect.borders')
        : proposal.aspect

  return (
    <>
      <div
        className="pointer-events-none absolute border border-ink-100"
        style={{
          ...frame,
          // The outside is dimmed with a shadow the size of the screen: one
          // element instead of four, and no colour anywhere near the photo.
          boxShadow: '0 0 0 9999px rgba(10, 10, 10, 0.55)',
        }}
      >
        {[1, 2].map((i) => (
          <span key={`v${i}`} className="absolute inset-y-0 w-px bg-ink-100/40"
            style={{ left: `${(i * 100) / 3}%` }} />
        ))}
        {[1, 2].map((i) => (
          <span key={`h${i}`} className="absolute inset-x-0 h-px bg-ink-100/40"
            style={{ top: `${(i * 100) / 3}%` }} />
        ))}
      </div>
      <div
        className="absolute flex items-center gap-2 rounded bg-ink-950/85 px-2 py-1 text-xs text-ink-100"
        style={{ left: frame.left, top: Math.max(box.top, frame.top - 34) }}
      >
        <span>{t('crop.proposed', { aspect })}</span>
        <Button size="sm" variant="primary" disabled={busy} onClick={onApply}>
          <Check size={12} />
          {t('crop.apply')}
        </Button>
        <Button size="sm" variant="ghost" disabled={busy} onClick={onReject}>
          <X size={12} />
          {t('crop.reject')}
        </Button>
      </div>
    </>
  )
}

/**
 * The proposal of the open photo, with its two answers wired to the server.
 *
 * Applying writes a new version from the *saved* parameters, so the edit still
 * waiting for the autosave is written first -- otherwise accepting a crop
 * would quietly throw away the last slider moves.
 */
export function CropProposalLayer({
  detail,
  params,
  flush,
  visible,
  imageRef,
  onUpdated,
}: {
  detail: PhotoDetail
  params: EditParams | null
  flush: () => Promise<void>
  visible: boolean
  imageRef: RefObject<HTMLImageElement | null>
  onUpdated: (updated: PhotoDetail) => void
}) {
  const decide = useMutation({
    mutationFn: async (answer: 'apply' | 'reject') => {
      const proposal = detail.crop_proposal!
      if (answer === 'apply') await flush()
      return answer === 'apply' ? api.applyCrop(proposal) : api.rejectCrop(proposal)
    },
    onSuccess: onUpdated,
  })

  const proposal = detail.crop_proposal
  if (!visible || !proposal || params?.geometry.crop) return null
  return (
    <CropOverlay
      imageRef={imageRef}
      proposal={proposal}
      busy={decide.isPending}
      onApply={() => decide.mutate('apply')}
      onReject={() => decide.mutate('reject')}
    />
  )
}
