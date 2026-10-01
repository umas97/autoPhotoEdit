// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strings of phase 5 -- geometry, crop proposals, scenes, lenses, the scene
// model -- kept apart only so that it.ts stays readable. They are spread into
// the one table there, and `t()` does not know the difference.
export const analysis = {
  'nav.scenes': 'Scene',

  'geometry.title': 'Geometria',
  'geometry.lens': 'Correzione obiettivo',
  'geometry.lensProfile': 'Profilo: {model}',
  'geometry.lensProfileManual': 'Profilo scelto da te: {model}',
  'geometry.lensApplied': 'Già corretto sugli scatti, prima di unirli',
  'geometry.lensMissing': 'Obiettivo senza profilo lensfun: nessuna correzione',
  'geometry.lensPending': 'Profilo obiettivo non ancora cercato',
  'geometry.lensAssociate': 'Associa un profilo',
  'geometry.rotation': 'Rotazione',
  'geometry.straighten.rotated': 'Raddrizzata automaticamente di {deg}°',
  'geometry.straighten.level': 'Orizzonte già dritto',
  'geometry.straighten.no_lines': 'Nessuna linea affidabile: non ruotata',
  'geometry.straighten.contradictory': 'Linee contraddittorie: non ruotata',
  'geometry.straighten.too_large': 'Inclinazione di {deg}°, troppo per essere un errore: non ruotata',
  'geometry.straighten.restore': 'Ripristina la rotazione automatica',
  'geometry.crop': 'Crop',
  'geometry.cropNone': 'Nessun crop',
  'geometry.cropSet': '{w}% × {h}% del fotogramma',
  'geometry.cropClear': 'Rimuovi il crop',

  'crop.proposed': 'Crop proposto ({aspect})',
  'crop.apply': 'Applica crop proposto',
  'crop.reject': 'Scarta',
  'crop.paused': 'Proposte di crop sospese in questo progetto dopo due rifiuti di fila.',
  'crop.resume': 'Riattiva le proposte',
  'crop.aspect.original': 'proporzioni originali',
  'crop.aspect.borders': 'senza i bordi vuoti',

  'scenes.title': 'Scene',
  'scenes.summary': '{clusters} scene su {photos} foto',
  'scenes.pending': 'Analisi in corso: {count} foto in coda. Le scene si aggiornano alla fine.',
  'scenes.none': 'Nessuna foto analizzata. L’analisi parte da sola dopo le anteprime.',
  'scenes.photos': '{count} foto',
  'scenes.representative': 'Rappresentante',
  'scenes.scene': 'Scena {n}',
  'scenes.basis.embedding': 'Raggruppate per contenuto (modello di scena) e orario di scatto.',
  'scenes.basis.features':
    'Raggruppate per luce, colori ed EXIF. Il modello di scena le distinguerebbe anche per contenuto.',
  'scenes.open': 'Apri nel visualizzatore',

  'models.title': 'Modello di scena',
  'models.embedding.body':
    'CLIP ViT-B/32 (licenza MIT), {size} MB, scaricato una volta da Hugging Face e verificato con SHA-256. Serve a raggruppare le foto per contenuto e, più avanti, a scegliere lo stile.',
  'models.download': 'Scarica ({size} MB)',
  'models.downloading': 'Download… {percent}%',
  'models.cancel': 'Annulla download',
  'models.ready': 'Installato e verificato.',
  'models.failed': 'Download non riuscito: {error}',
  'models.reason.runtime_missing': 'ONNX Runtime non è installato (extra «ml»).',
  'models.reason.checksum_mismatch':
    'Il file presente non corrisponde al checksum: verrà sostituito dal download.',
  'models.backfill': 'Calcola il modello di scena anche per le foto già analizzate',

  'lenses.title': 'Obiettivi',
  'lenses.photos': '{count} foto',
  'lenses.auto': 'Profilo lensfun: {model}',
  'lenses.override': 'Associato da te: {model}',
  'lenses.missing': 'Senza profilo: nessuna correzione applicata',
  'lenses.pending': 'In attesa di analisi',
  'lenses.unknown': 'Obiettivo non indicato negli EXIF',
  'lenses.associate': 'Associa profilo',
  'lenses.change': 'Cambia',
  'lenses.clear': 'Rimuovi associazione',
  'lenses.search': 'Cerca nel database lensfun',
  'lenses.searchHint': 'Per esempio: tamron 28-75',
  'lenses.noResults': 'Nessun profilo trovato.',
  'lenses.dialogTitle': 'Profilo per «{lens}»',
  'lenses.dialogBody':
    'La scelta vale per questo obiettivo in tutti i progetti. Anteprime e analisi delle sue foto vengono rifatte.',
  'lenses.data.bundled': 'Dati lensfun inclusi nel programma ({count} obiettivi).',
  'lenses.data.updated': 'Dati lensfun aggiornati il {date} ({count} obiettivi).',
  'lenses.data.update': 'Aggiorna dati lensfun',
  'lenses.data.updating': 'Aggiornamento dei dati lensfun…',
  'lenses.data.updateHint':
    'Scarica da lensfun.github.io i profili più recenti. Unico accesso alla rete, e solo quando lo chiedi.',
  'lenses.data.failed': 'Aggiornamento non riuscito: {error}',
} as const
