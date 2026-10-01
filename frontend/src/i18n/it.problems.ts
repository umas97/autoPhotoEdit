// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the Problems panel and of the diagnostics bundle (section 19).
// Spread into the one table in it.ts.
export const problems = {
  'nav.problems': 'Problemi',
  'problems.badge': '{count} problemi',
  'problems.badgeOne': '1 problema',
  'problems.badgeHint': 'Foto o lavorazioni non riuscite: apri il pannello Problemi',
  'problems.none': 'Nessun problema: tutte le foto e le lavorazioni sono andate a buon fine.',
  'problems.intro':
    'Un errore su una foto non ferma mai le altre. Qui trovi cosa non è riuscito e perché; «Riprova» rimette in coda il lavoro.',
  'problems.photos': 'Foto non riuscite ({count})',
  'problems.jobs': 'Altre lavorazioni non riuscite ({count})',
  'problems.merges': 'Fusioni non riuscite ({count})',
  'problems.mergeTitle': '{kind} di {frames} scatti',
  'problems.mergeEdit': 'Cambia i parametri',
  'problems.stage': 'fase: {stage}',
  'problems.photoOf': 'foto {id}',
  'problems.details': 'Dettagli tecnici',
  'problems.copy': 'Copia',
  'problems.copied': 'Copiato',
  'problems.retry': 'Riprova',
  'problems.retryAll': 'Riprova tutte',
  'problems.retried': 'Rimesse in coda: {count}.',
  'problems.diagnostics': 'Esporta diagnostica',

  'diagnostics.title': 'Esporta diagnostica',
  'diagnostics.description':
    'Uno zip da allegare a una segnalazione. Ecco cosa conterrà, file per file, prima di crearlo: mai i RAW, mai i pixel né il contenuto delle foto, nessun nome di progetto né autore, e la tua cartella home scritta come ~.',
  'diagnostics.loading': 'Preparo l’elenco…',
  'diagnostics.show': 'Mostra il contenuto',
  'diagnostics.hide': 'Nascondi il contenuto',
  'diagnostics.logTail': 'ultime righe; nello zip c’è il file intero',
  'diagnostics.download': 'Scarica lo zip ({size})',
  'diagnostics.cli': 'Dal terminale: autophotoedit diagnose',
} as const
