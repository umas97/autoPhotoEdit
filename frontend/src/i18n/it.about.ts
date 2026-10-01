// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the "Informazioni" dialog of the main menu. Spread into the one
// table in it.ts.
export const about = {
  'about.open': 'Informazioni',
  'about.title': 'Informazioni su autoPhotoEdit',
  'about.version': 'Versione {version}',
  'about.versionUnknown': 'Versione non disponibile',
  'about.summary':
    'autoPhotoEdit sviluppa in automatico i file RAW Sony (.ARW). Impara il tuo stile dalle foto che hai già post-prodotto e lo applica agli scatti nuovi, chiedendoti conferma solo su quelli su cui non è abbastanza sicuro.',
  'about.features':
    'Si occupa anche della cernita, delle fusioni (HDR, focus stacking, panorama), delle maschere, della rimozione di imperfezioni e dell’export con i sidecar XMP per Lightroom e darktable. Lavora tutto in locale e non modifica mai i RAW originali.',
  'about.credits': 'Progetto costruito in vibe coding da {author}.',
  'about.license': 'Software libero, con licenza GPL-3.0-or-later.',
} as const
