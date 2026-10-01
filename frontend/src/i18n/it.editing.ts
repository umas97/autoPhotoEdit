// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the saving that happens by itself (lib/autosave.ts), in the
// viewer and in the review. Spread into the one table in it.ts.
export const editing = {
  'editing.saved': 'Salvato',
  'editing.pending': 'Da salvare…',
  'editing.saving': 'Salvataggio…',
  'editing.unsaved': 'Non salvato',
  'editing.retry': 'Riprova',
  'editing.hint':
    'Le modifiche si salvano da sole: ogni gesto diventa una versione (più movimenti dello stesso cursore entro 2 secondi ne fanno una). Ctrl+S salva subito.',
  'editing.saveFailed': 'Modifica non salvata: {message}. Riprovo al prossimo gesto, o premi «Riprova».',
} as const
