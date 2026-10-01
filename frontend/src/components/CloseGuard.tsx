// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Closing the window while the queue is busy (section 21.3).
//
// The browser gives us one hook that fires before the document goes away and
// one that fires as it goes. Neither can hold a dialog open -- `beforeunload`
// can only show the browser's own generic box, which section 21.3 forbids -- so
// the honest arrangement is this: the *application* asks the question up front,
// as soon as there is something to decide about, and the answer is remembered
// and sent with `sendBeacon` when the window actually closes.
//
// With nothing running there is no question, and closing the window closes the
// program, as in any desktop application.
import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, sendCloseIntent } from '../lib/api'
import { t } from '../i18n/it'
import { Button } from './ui/Button'
import { Dialog } from './ui/Dialog'
import { Checkbox } from './ui/Field'

type Decision = 'continue' | 'pause' | 'stop'

export function CloseGuard() {
  const [open, setOpen] = useState(false)
  const [remember, setRemember] = useState(false)
  const { data: window_ } = useQuery({ queryKey: ['window'], queryFn: api.window })
  const { data: active } = useQuery({
    queryKey: ['jobs', 'active'],
    queryFn: api.activeJobs,
    refetchInterval: 2_000,
  })

  const count = active?.active ?? 0
  const remembered = window_?.on_window_close ?? null

  useEffect(() => {
    const beforeUnload = () => {
      // A decision already exists, or there is nothing to decide: say so and let
      // the window go. Section 21.3 makes silence mean "continue".
      sendCloseIntent(remembered ?? 'continue', false)
    }
    const keyDown = (event: KeyboardEvent) => {
      // Ctrl+W in a Chromium app window closes it. If jobs are moving and the
      // user has never chosen, ask now rather than guess.
      if ((event.ctrlKey || event.metaKey) && event.key === 'w' && count > 0 && !remembered) {
        event.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('beforeunload', beforeUnload)
    window.addEventListener('keydown', keyDown)
    return () => {
      window.removeEventListener('beforeunload', beforeUnload)
      window.removeEventListener('keydown', keyDown)
    }
  }, [count, remembered])

  const decide = (decision: Decision) => {
    sendCloseIntent(decision, remember)
    setOpen(false)
    if (decision === 'stop') void api.quit().catch(() => undefined)
    else window.close()
  }

  return (
    <Dialog
      open={open}
      onOpenChange={setOpen}
      dismissible={false}
      title={t('close.title')}
      description={t('close.body', { count })}
      footer={
        <>
          <Button variant="ghost" onClick={() => decide('stop')} title={t('close.stopHint')}>
            {t('close.stop')}
          </Button>
          <Button variant="outline" onClick={() => decide('pause')} title={t('close.pauseHint')}>
            {t('close.pause')}
          </Button>
          <Button
            variant="primary"
            onClick={() => decide('continue')}
            title={t('close.continueHint')}
          >
            {t('close.continue')}
          </Button>
        </>
      }
    >
      <Checkbox checked={remember} onChange={setRemember} label={t('close.remember')} />
    </Dialog>
  )
}
