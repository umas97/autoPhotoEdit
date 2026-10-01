// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The progress socket of section 12, as a hook.
//
// The server sends a whole snapshot four times a second and stays quiet when
// nothing changes (backend/ape/api/ws.py), so there is no event log to replay
// and no state to accumulate here: the last message *is* the state. A client
// that reconnects after a gap has missed nothing.
//
// Reconnection is a fixed second rather than a backoff. The other end is on
// loopback: it is either up, or it is being restarted and will be up shortly.
import { useEffect, useRef, useState } from 'react'
import type { ProgressMessage } from './types'

export function useProgress(projectId?: number): {
  progress: ProgressMessage | null
  connected: boolean
} {
  const [progress, setProgress] = useState<ProgressMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let closed = false
    let timer: number | undefined

    const connect = () => {
      if (closed) return
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
      const query = projectId ? `?project_id=${projectId}` : ''
      const socket = new WebSocket(`${scheme}://${location.host}/ws${query}`)
      socketRef.current = socket

      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => {
        try {
          setProgress(JSON.parse(event.data as string) as ProgressMessage)
        } catch {
          /* a frame we cannot parse is a frame we can ignore: the next is whole */
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!closed) timer = window.setTimeout(connect, 1000)
      }
      socket.onerror = () => socket.close()
    }

    connect()
    return () => {
      closed = true
      if (timer) window.clearTimeout(timer)
      socketRef.current?.close()
    }
  }, [projectId])

  return { progress, connected }
}
