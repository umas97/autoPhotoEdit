// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strings of the folder browser used to choose a project's RAWs and a
// style profile's two folders. Spread
// into the one table in it.ts.
export const folders = {
  'folders.browse': 'Sfoglia…',
  'folders.close': 'Chiudi',
  'folders.up': 'Cartella superiore',
  'folders.place.home': 'Home',
  'folders.place.pictures': 'Immagini',
  'folders.place.root': 'Computer',
  'folders.empty': 'Nessuna sottocartella.',
  'folders.truncated': 'Mostrate solo le prime {count} cartelle.',
  'folders.raws': '{count} RAW in questa cartella',
  'folders.noRaws': 'Nessun RAW qui: si leggono solo i file della cartella scelta, non le sottocartelle.',
  'folders.references': '{count} foto editate (JPEG o TIFF) in questa cartella',
  'folders.noReferences':
    'Nessun JPEG o TIFF qui: si leggono solo i file della cartella scelta, non le sottocartelle.',
  'folders.choose': 'Usa questa cartella',
  'folders.home': 'Torna alla home',
} as const
