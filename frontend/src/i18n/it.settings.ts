// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the Settings screen (sections 10, 16.3, 20, 21.3). Spread into
// the one table in it.ts.
export const settings = {
  'nav.settings': 'Impostazioni',

  'settings.author': 'Autore e copyright',
  'settings.authorHint': 'Scritti nei metadati di ogni foto esportata, in tutti i progetti.',
  'settings.artist': 'Autore',
  'settings.copyright': 'Copyright',

  'settings.storage': 'Spazio su disco',
  'settings.storageHint':
    'La cache si rigenera dai RAW: oltre la quota si liberano per primi i file usati meno di recente, le fusioni per ultime. Le maschere dipinte a mano e i modelli non sono cache e non vengono mai rimossi.',
  'settings.quota': 'Quota della cache (GB)',
  'settings.quotaInvalid': 'Serve un numero fra 1 e 10000.',
  'settings.cacheUsed': 'Cache: {used} su {limit}',
  'settings.cat.proxies': 'Anteprime di lavoro (proxy)',
  'settings.cat.previews': 'Anteprime della fotocamera',
  'settings.cat.developed': 'Miniature sviluppate',
  'settings.cat.stages': 'Stadi intermedi',
  'settings.cat.intermediates': 'Intermedi delle fusioni',
  'settings.cat.merges': 'Anteprime delle fusioni',
  'settings.cat.masks': 'Maschere dipinte a mano (non cache)',
  'settings.cat.models': 'Modelli scaricati (non cache)',
  'settings.files': '{count} file',
  'settings.clear': 'Svuota cache…',
  'settings.clearTitle': 'Svuotare la cache?',
  'settings.clearFrees': 'Libera {size}.',
  'settings.clearMerges': '{count} fusioni andranno rigenerate: può richiedere minuti ciascuna.',
  'settings.clearProxies': 'Le anteprime di lavoro si ricreano da sole mentre sfogli le foto.',
  'settings.clearKeeps': 'Non cancella: il catalogo, gli edit, le maschere dipinte a mano ({masks}), i modelli ({models}).',
  'settings.clearYes': 'Svuota la cache',
  'settings.cleared': 'Cache svuotata: liberati {size}.',

  'settings.window': 'Chiusura della finestra',
  'settings.windowHint': 'Cosa fare se chiudi la finestra mentre ci sono lavorazioni in corso.',
  'settings.close.ask': 'Chiedi ogni volta',
  'settings.close.continue': 'Continua in background',
  'settings.close.pause': 'Metti in pausa',
  'settings.close.stop': 'Interrompi',

  'settings.backup': 'Backup del catalogo',
  'settings.backupHint':
    'Il catalogo contiene progetti, edit, storico e stili; non contiene cache né proxy. Da un terminale:',
  'settings.backupMasks': 'Copia anche la cartella delle maschere dipinte a mano: {path}',
  'settings.restoreHint': 'Il ripristino mette da parte il catalogo attuale; va fatto con il programma chiuso.',

  'settings.lenses': 'Obiettivi',
  'settings.lensesHint':
    'I profili lensfun correggono distorsione, vignettatura e aberrazione cromatica. Quelli inclusi nel programma non conoscono gli obiettivi più recenti: aggiornali una volta, valgono per tutti i progetti.',
  'settings.diagnostics': 'Diagnostica',
  'settings.diagnosticsHint': 'Foto non riuscite, dettagli tecnici e lo zip da allegare a una segnalazione.',
  'settings.openProblems': 'Apri Problemi',
} as const
