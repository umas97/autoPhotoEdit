// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The two or three alternatives of section 9.2.3, each rendered by the real
// pipeline at 512 px so that choosing among them is choosing among pictures,
// not among names. The renders run one after another -- the server derives
// the 512 px source from the 2048 px one already open (api/deps.py), so each
// costs a pass of the pipeline, not a decode -- and are cancelled when the
// photo changes.
//
// Keys 1-3 pick one (handled by the page); picking sets the parameters under
// the sliders like any other edit, so it can be adjusted before approving.
import { useEffect, useState } from 'react'
import { renderPreview } from '../../lib/api'
import { sameParams } from '../../lib/params'
import type { EditParams } from '../../lib/types'
import type { VariantKey } from '../../lib/reviewTypes'
import { t, type StringKey } from '../../i18n/it'
import { cn } from '../../lib/utils'

const VARIANT_EDGE = 512

export function VariantStrip({
  photoId,
  variants,
  current,
  onChoose,
}: {
  photoId: number
  variants: Array<{ key: VariantKey; params: EditParams }>
  current: EditParams | null
  onChoose: (params: EditParams) => void
}) {
  const [urls, setUrls] = useState<Record<string, string>>({})

  useEffect(() => {
    const abort = new AbortController()
    const made: string[] = []
    setUrls({})
    void (async () => {
      for (const variant of variants) {
        try {
          const blob = await renderPreview(photoId, variant.params, VARIANT_EDGE, abort.signal)
          const url = URL.createObjectURL(blob)
          made.push(url)
          setUrls((previous) => ({ ...previous, [variant.key]: url }))
        } catch {
          if (abort.signal.aborted) return
        }
      }
    })()
    return () => {
      abort.abort()
      made.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [photoId, variants])

  if (variants.length === 0) return null
  return (
    <div className="border-t border-ink-700 bg-ink-900 px-3 py-1.5">
      <p className="mb-1 text-[11px] text-ink-400">
        {t('review.variants')} · {t('review.variantsHint')}
      </p>
      <div className="flex gap-2">
        {variants.map((variant, index) => {
          const active = current !== null && sameParams(current, variant.params)
          return (
            <button
              key={variant.key}
              type="button"
              onClick={() => onChoose(variant.params)}
              className={cn(
                'w-40 shrink-0 overflow-hidden rounded-sm border text-left',
                active ? 'border-ink-100' : 'border-ink-700 hover:border-ink-500',
              )}
            >
              <div className="aspect-[3/2] bg-mat">
                {urls[variant.key] ? (
                  <img src={urls[variant.key]} alt="" className="h-full w-full object-contain" />
                ) : null}
              </div>
              <span className="block truncate px-1.5 py-0.5 text-[11px] text-ink-200">
                <span className="mr-1 rounded bg-ink-700 px-1 tabular-nums text-ink-100">{index + 1}</span>
                {t(`review.variant.${variant.key}` as StringKey)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
