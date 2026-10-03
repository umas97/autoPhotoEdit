// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the review draws over the fitted photo, as the viewer does
// (components/viewer/ViewerStage.tsx): the crop proposal or the crop being
// drawn, and the masks' handles -- these only for the photo under review, not
// for one of its scene being inspected.
import type { RefObject } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { EditParams, Photo, PhotoDetail } from '../../lib/types'
import { CropProposalLayer } from '../CropOverlay'
import { CropTool, type CropEditor } from '../crop'
import { MaskLayer } from '../masks/MaskLayer'
import type { MaskEditor } from '../masks/useMaskEditor'

export function ReviewOverlay({
  detail,
  photo,
  params,
  onChange,
  flush,
  editor,
  crop,
  imageRef,
  dragging,
  rendered,
  comparing,
  inspecting,
}: {
  detail: PhotoDetail
  photo: Pick<Photo, 'id' | 'has_proxy' | 'proxy_rev'>
  params: EditParams | null
  onChange: (next: EditParams, commit: boolean) => void
  flush: () => Promise<void>
  editor: MaskEditor
  crop: CropEditor
  imageRef: RefObject<HTMLImageElement | null>
  dragging: boolean
  /** The developed preview is on screen, not the neutral proxy. */
  rendered: boolean
  /** The neutral comparison of the style panel is on screen, not the edit. */
  comparing: boolean
  inspecting: boolean
}) {
  const queryClient = useQueryClient()
  return (
    <>
      <CropProposalLayer detail={detail} params={params} flush={flush} imageRef={imageRef}
        visible={!dragging && rendered && editor.tab !== 'masks' && !crop.active}
        onUpdated={(updated) => queryClient.setQueryData(['photo', updated.id], updated)} />
      {params && !inspecting ? (
        <MaskLayer editor={editor} photo={photo} params={params} onChange={onChange}
          imageRef={imageRef} dragging={dragging} flush={flush}
          visible={rendered && !comparing && !crop.active} />
      ) : null}
      <CropTool editor={crop} imageRef={imageRef} />
    </>
  )
}
