// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Which style developed this photo, and from which of the user's samples
// (section 8.3: "questa foto è stata editata come i tuoi sample #12 e #31").
// The samples show their 512 px reference, which a profile carries even when
// its RAWs are on another machine (section 20.1).
//
// "Confronta con Neutro automatico" is the comparison section 22 asks to be
// available at any moment: while it is on, the preview shows what the neutral
// profile would do with this photo. It only previews; nothing is saved.
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { sampleThumbUrl, stylesApi } from '../lib/stylesApi'
import type { EditParams } from '../lib/types'
import { t } from '../i18n/it'
import { Checkbox } from './ui/Field'

export function StylePanel({
  photoId,
  onCompare,
}: {
  photoId: number
  onCompare: (params: EditParams | null) => void
}) {
  const [comparing, setComparing] = useState(false)
  const style = useQuery({ queryKey: ['photoStyle', photoId], queryFn: () => stylesApi.photoStyle(photoId) })
  const profiles = useQuery({ queryKey: ['styles'], queryFn: stylesApi.styles })
  const neutral = profiles.data?.find((p) => p.builtin && p.name === style.data?.neutral_profile)

  const toggle = async (on: boolean) => {
    setComparing(on)
    if (!on || !neutral) {
      onCompare(null)
      return
    }
    try {
      onCompare((await stylesApi.photoStyleParams(photoId, neutral.id)).params)
    } catch {
      setComparing(false)
      onCompare(null)
    }
  }

  const data = style.data
  return (
    <section className="border-b border-ink-700 px-3 py-2">
      <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-400">{t('photoStyle.title')}</h3>
      {!data?.profile ? (
        <p className="text-xs text-ink-400">{t('photoStyle.none')}</p>
      ) : (
        <>
          <p className="text-xs text-ink-200">
            {t('photoStyle.by', { name: data.profile.name })}
            {data.method === 'rules' ? ` · ${t('photoStyle.rules')}` : ''}
          </p>
          {data.kept_user_edit ? <p className="mt-1 text-xs text-warn">{t('photoStyle.kept')}</p> : null}
          {data.neighbours.length > 0 ? (
            <div className="mt-2">
              <p className="text-[11px] text-ink-400">{t('photoStyle.samplesHint')}</p>
              <ul className="mt-1 grid grid-cols-5 gap-1">
                {data.neighbours.map((sample) => (
                  <li key={sample.id} title={`${sample.reference ?? ''} · ${Math.round((sample.weight ?? 0) * 100)}%`}>
                    <div className="aspect-square overflow-hidden rounded-sm border border-ink-700 bg-mat">
                      {sample.has_thumbnail ? (
                        <img src={sampleThumbUrl(sample.id)} alt={sample.reference ?? ''} loading="lazy"
                          className="h-full w-full object-cover" />
                      ) : null}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </>
      )}
      {neutral ? (
        <div className="mt-2">
          <Checkbox checked={comparing} onChange={(on) => void toggle(on)} label={t('photoStyle.compare')} />
          {comparing ? <p className="text-[11px] text-warn">{t('photoStyle.comparing')}</p> : null}
        </div>
      ) : null}
    </section>
  )
}
