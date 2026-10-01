// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The style library (section 10, screen 4): the built-in profiles of section 22
// and the user's own, creation from two folders, the pairs and their residuals,
// the manual pairing, and the .apestyle round trip of section 20.1.
//
// The library is global -- a profile belongs to the photographer, not to a
// project -- so the screen hangs off the main navigation, and a project's
// choice of style lives on the project's own screen.
import { useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Upload } from 'lucide-react'
import { ApiError } from '../lib/api'
import { stylesApi } from '../lib/stylesApi'
import type { StyleSummary } from '../lib/styleTypes'
import { t } from '../i18n/it'
import { cn } from '../lib/utils'
import { AppShell } from '../components/AppShell'
import { Button } from '../components/ui/Button'
import { CreateProfileDialog } from '../components/styles/CreateProfileDialog'
import { ProfileDetail, profileBadge } from '../components/styles/ProfileDetail'

function ProfileList({
  title,
  profiles,
  selected,
  onSelect,
}: {
  title: string
  profiles: StyleSummary[]
  selected: number | null
  onSelect: (id: number) => void
}) {
  return (
    <div>
      <h2 className="mb-1 px-1 text-xs font-medium uppercase tracking-wide text-ink-400">{title}</h2>
      <ul className="space-y-0.5">
        {profiles.map((profile) => (
          <li key={profile.id}>
            <button
              type="button"
              onClick={() => onSelect(profile.id)}
              className={cn(
                'w-full rounded-md px-2 py-1.5 text-left',
                selected === profile.id ? 'bg-ink-700 text-ink-50' : 'text-ink-200 hover:bg-ink-800',
              )}
            >
              <span className="block truncate text-sm">{profile.name}</span>
              <span className="block truncate text-[11px] text-ink-400">{profileBadge(profile)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function StylesPage() {
  const { profileId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const selected = profileId ? Number(profileId) : null
  const [creating, setCreating] = useState(false)
  const fileInput = useRef<HTMLInputElement | null>(null)

  const profiles = useQuery({
    queryKey: ['styles'],
    queryFn: stylesApi.styles,
    refetchInterval: (query) =>
      query.state.data?.some((p) => p.samples.pending > 0) ? 3_000 : false,
  })
  const importing = useMutation({
    mutationFn: (file: File) => stylesApi.importStyle(file),
    onSuccess: (profile) => {
      void queryClient.invalidateQueries({ queryKey: ['styles'] })
      navigate(`/stili/${profile.id}`)
    },
  })

  const all = profiles.data ?? []
  const builtin = all.filter((p) => p.builtin)
  const learned = all.filter((p) => !p.builtin)
  const open = (id: number) => navigate(`/stili/${id}`)

  return (
    <AppShell
      actions={
        <>
          <input
            ref={fileInput}
            type="file"
            accept=".apestyle,application/zip"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) importing.mutate(file)
              event.target.value = ''
            }}
          />
          <Button size="sm" variant="ghost" disabled={importing.isPending} onClick={() => fileInput.current?.click()}>
            <Upload size={14} />
            {importing.isPending ? t('styles.importing') : t('styles.import')}
          </Button>
          <Button size="sm" variant="primary" onClick={() => setCreating(true)}>
            <Plus size={14} />
            {t('styles.new')}
          </Button>
        </>
      }
    >
      <div className="grid h-full grid-cols-[minmax(13rem,17rem)_1fr] overflow-hidden">
        <aside className="min-h-0 space-y-4 overflow-y-auto border-r border-ink-700 p-2">
          <ProfileList title={t('styles.builtin')} profiles={builtin} selected={selected} onSelect={open} />
          <ProfileList title={t('styles.learned')} profiles={learned} selected={selected} onSelect={open} />
          {profiles.data && learned.length === 0 ? (
            <p className="px-1 text-xs text-ink-400">{t('styles.none')}</p>
          ) : null}
        </aside>
        <section className="min-h-0 overflow-y-auto p-4">
          {importing.error ? (
            <p className="mb-3 text-xs text-bad">{(importing.error as ApiError).message}</p>
          ) : null}
          {selected !== null ? (
            <ProfileDetail
              key={selected}
              profileId={selected}
              onDeleted={() => navigate('/stili')}
              onDuplicated={(copy) => navigate(`/stili/${copy.id}`)}
            />
          ) : (
            <div className="max-w-prose space-y-2">
              <p className="text-sm text-ink-200">{t('styles.intro')}</p>
              <p className="text-sm text-ink-400">{t('styles.select')}</p>
            </div>
          )}
        </section>
      </div>
      <CreateProfileDialog
        open={creating}
        onOpenChange={setCreating}
        onCreated={(profile) => {
          void queryClient.invalidateQueries({ queryKey: ['styles'] })
          navigate(`/stili/${profile.id}`)
        }}
      />
    </AppShell>
  )
}
