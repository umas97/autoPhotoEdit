// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What the workers are doing, in the corner of the header: the progress socket
// of section 12, shown as one bar and a count. Silent when the queue is empty --
// a permanent progress widget on an idle program is noise.
//
// The count of failures stays, idle or not (section 19: "il conteggio dei
// falliti è sempre visibile"), and leads to the Problems panel.
import { AlertTriangle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useProgress } from '../lib/useProgress'
import { t } from '../i18n/it'

export function JobBar() {
  const { progress, connected } = useProgress()
  const navigate = useNavigate()
  if (!progress) {
    return connected ? null : (
      <span className="text-xs text-warn" title={t('window.serverDown')}>
        ●
      </span>
    )
  }

  const problems = progress.problems ? (
    <button type="button" onClick={() => navigate('/problemi')} title={t('problems.badgeHint')}
      className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-bad hover:bg-ink-800">
      <AlertTriangle size={13} />
      {progress.problems === 1 ? t('problems.badgeOne') : t('problems.badge', { count: progress.problems })}
    </button>
  ) : null
  if (progress.active === 0) return problems

  const percent = Math.round(progress.progress * 100)
  return (
    <>
      {problems}
      <div className="flex items-center gap-2" title={t('jobs.title')}>
        <div className="h-1 w-24 overflow-hidden rounded-full bg-ink-700">
          <div className="h-full bg-ink-300 transition-[width]" style={{ width: `${percent}%` }} />
        </div>
        <span className="tabular-nums text-xs text-ink-300">
          {progress.active} {t('jobs.queued')}
        </span>
      </div>
    </>
  )
}
