// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A text field saved when it loses focus or on Enter, not at every keystroke:
// the Export screen and the Settings screen write global preferences with it.
import { useEffect, useState } from 'react'
import { TextInput } from './Field'

/** A text field saved when it loses focus, not at every keystroke. */
export function CommittedInput({
  value,
  onCommit,
  onDraft,
  placeholder,
  inputMode,
}: {
  value: string
  onCommit: (value: string) => void
  onDraft?: (value: string) => void
  placeholder?: string
  inputMode?: 'numeric'
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => {
    if (draft !== value) onCommit(draft)
  }
  return (
    <TextInput
      value={draft}
      placeholder={placeholder}
      inputMode={inputMode}
      onChange={(event) => {
        setDraft(event.target.value)
        onDraft?.(event.target.value)
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') commit()
      }}
    />
  )
}
