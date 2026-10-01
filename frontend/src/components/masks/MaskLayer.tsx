// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Everything the masks editor draws over the photo: the selected mask's
// handles or brush, and on request (O) its selection in red.
//
// Only over the developed preview: the handles are placed with the geometry of
// the parameters, and the neutral proxy shown before the first render -- or
// while "before" is held -- is neither straightened nor cropped.
import type { RefObject } from 'react'
import { proxyUrl } from '../../lib/api'
import { useDisplayedBox, useFrameSize } from '../../lib/geometry'
import { withDefinition, withMask } from '../../lib/masks'
import type { MaskParams } from '../../lib/maskTypes'
import { toast } from '../../lib/toast'
import type { EditParams, Photo } from '../../lib/types'
import { RetouchLayer } from '../retouch/RetouchLayer'
import { BrushCanvas } from './BrushCanvas'
import { SelectionOverlay } from './SelectionOverlay'
import { ShapeHandles } from './ShapeHandles'
import type { MaskEditor } from './useMaskEditor'

export function MaskLayer({
  editor,
  photo,
  params,
  onChange,
  imageRef,
  dragging,
  visible,
  flush,
}: {
  editor: MaskEditor
  photo: Pick<Photo, 'id' | 'has_proxy' | 'proxy_rev'>
  params: EditParams
  onChange: (next: EditParams, commit: boolean) => void
  imageRef: RefObject<HTMLImageElement | null>
  dragging: boolean
  visible: boolean
  /** Write the edit now (the eraser's fill should not wait for the autosave). */
  flush?: () => Promise<void>
}) {
  const frame = useFrameSize(photo.has_proxy ? proxyUrl(photo) : null)
  const box = useDisplayedBox(imageRef, photo.id)
  if (!visible || !frame || !box || editor.tab !== 'masks') return null
  const removals = (
    <RetouchLayer editor={editor} photoId={photo.id} params={params} onChange={onChange}
      flush={flush} frame={frame} box={box} />
  )
  const index = editor.selected
  const mask = index === null ? undefined : params.masks[index]
  if (index === null || !mask) return removals

  const change = (next: MaskParams, commit: boolean) =>
    onChange(withMask(params, index, next), commit)

  return (
    <>
      {removals}
      {editor.showSelection ? (
        <SelectionOverlay photoId={photo.id} params={params} index={index} box={box}
          dragging={dragging} onError={(message) => message && toast(message)} />
      ) : null}
      {mask.kind === 'linear' || mask.kind === 'radial' ? (
        <ShapeHandles mask={mask} frame={frame} geometry={params.geometry} box={box}
          onChange={change} />
      ) : null}
      {mask.kind === 'brush' ? (
        <BrushCanvas raster={mask.definition.raster ?? null} frame={frame}
          geometry={params.geometry} box={box} brush={editor.brush} busy={editor.busy}
          setBusy={editor.setBusy} onError={toast}
          onPainted={(raster) => change(withDefinition(mask, { raster }), true)} />
      ) : null}
    </>
  )
}
