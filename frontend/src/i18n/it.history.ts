// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the history (section 23): where each version of a photo came
// from, and the snapshots of a project. Spread into the one table in it.ts.
export const history = {
  'versions.source.predicted': 'predetta',
  'versions.source.user_edited': 'modificata da te',
  'versions.source.cluster_applied': 'applicata dalla scena',
  'versions.source.reverted': 'ripristinata',
  'versions.source.snapshot_restored': 'da snapshot',

  'snapshots.open': 'Snapshot',
  'snapshots.title': 'Snapshot del progetto',
  'snapshots.description':
    'Uno snapshot ricorda quale versione era corrente per ogni foto e le decisioni di cernita e revisione. Non copia pixel. Tornare a uno snapshot non cancella nulla: crea nuove versioni, e prima fotografa lo stato attuale, così puoi annullare.',
  'snapshots.name': 'Nome',
  'snapshots.namePlaceholder': 'es. prima della revisione manuale',
  'snapshots.take': 'Crea snapshot',
  'snapshots.empty': 'Nessuno snapshot ancora. Se ne crea uno da solo a fine cernita, fine predizione e fine revisione.',
  'snapshots.auto': 'automatico',
  'snapshots.photos': '{count} foto',
  'snapshots.differs': '{count} diverse da ora',
  'snapshots.same': 'uguale a ora',
  'snapshots.restore': 'Torna qui',
  'snapshots.confirm': 'Tornare a «{name}»? {count} foto cambiano; lo stato attuale resta in uno snapshot automatico.',
  'snapshots.confirmYes': 'Sì, torna qui',
  'snapshots.delete': 'Elimina',
  'snapshots.restored': 'Ripristinate {count} foto.',
  'snapshots.missing': '{count} foto dello snapshot non ci sono più.',
  'snapshots.undo': 'Annulla il ripristino',
  'snapshots.close': 'Chiudi',
} as const
