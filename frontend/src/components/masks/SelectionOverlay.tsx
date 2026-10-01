// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What a mask selects, in red over the photo (O).
//
// The selection comes from the server, computed by the pipeline itself and
// framed as the preview is: a luminance range or a segmented subject cannot
// be drawn by the browser, and for the shapes it shows the feathering as it
// will really fall. As with the preview, one request at a time, the previous
// one cancelled, and nothing asked while a handle is still moving.
import { useEffect, useRef, useState } from 'react'
import type { Box } from '../../lib/geometry'
import { maskSelection } from '../../lib/masksApi'
import type { EditParams } from '../../lib/types'

const EDGE = 1024

export function SelectionOverlay({
  photoId,
  params,
  index,
  box,
  dragging,
  onError,
}: {
  photoId: number
  params: EditParams
  index: number
  box: Box
  dragging: boolean
  onError: (message: string | null) => void
}) {
  const [url, setUrl] = useState<string | null>(null)
  const urlRef = useRef<string | null>(null)
  const key = JSON.stringify([photoId, index, params])

  useEffect(() => {
    if (dragging) return
    const abort = new AbortController()
    const timer = window.setTimeout(() => {
      maskSelection(photoId, index, params, EDGE, abort.signal)
        .then((blob) => {
          const next = URL.createObjectURL(blob)
          if (urlRef.current) URL.revokeObjectURL(urlRef.current)
          urlRef.current = next
          setUrl(next)
          onError(null)
        })
        .catch((error: Error) => {
          if (error.name !== 'AbortError') onError(error.message)
        })
    }, 120)
    return () => {
      window.clearTimeout(timer)
      abort.abort()
    }
    // `key` stands for photoId, index and params.
  }, [key, dragging])

  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    },
    [],
  )

  if (!url) return null
  return (
    <img
      src={url}
      alt=""
      className="pointer-events-none absolute opacity-55"
      style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
    />
  )
}
