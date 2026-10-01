// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strings of phase 9 -- the masks, the segmentation, the shortcuts -- kept
// apart so that it.ts stays readable. They are spread into the one table there.
export const masks = {
  'viewer.undo': 'Annulla',
  'viewer.redo': 'Ripristina',

  'masks.tab.adjust': 'Regolazioni',
  'masks.tab.masks': 'Maschere',
  'masks.tab.masksCount': 'Maschere ({count})',
  'masks.list': 'Maschere della foto',
  'masks.empty':
    'Nessuna maschera. Una maschera applica esposizione, colore e contrasto solo a una parte della foto.',
  'masks.choose': 'Scegli una maschera per vederne le maniglie e regolarla.',
  'masks.inert': 'nessun effetto',
  'masks.name': 'Nome della maschera',
  'masks.duplicate': 'Duplica',
  'masks.delete': 'Elimina',
  'masks.invert': 'Inverti la selezione',
  'masks.opacity': 'Opacità',
  'masks.feather': 'Sfumatura',
  'masks.showSelection': 'Mostra la selezione in rosso (O)',
  'masks.notInSidecars':
    'Le maschere valgono nell’export JPEG/TIFF; gli XMP per darktable e Lightroom non le riportano.',

  'masks.group.selection': 'Selezione',
  'masks.group.light': 'Luce',
  'masks.group.color': 'Colore',
  'masks.group.detail': 'Dettaglio',

  'masks.kind.linear': 'Lineare',
  'masks.kind.radial': 'Radiale',
  'masks.kind.brush': 'Pennello',
  'masks.kind.parametric': 'Intervallo',
  'masks.kind.segment': 'Soggetto',
  'masks.kind.linear.hint':
    'Sfumatura lineare: piena oltre la linea continua, nulla oltre quella tratteggiata. Trascina gli estremi o il centro.',
  'masks.kind.radial.hint':
    'Ellisse: piena all’interno di quella tratteggiata, nulla fuori. Trascina il centro, le maniglie degli assi o quella vuota per ruotarla.',
  'masks.kind.brush.hint':
    'Dipingi sulla foto; Alt mentre dipingi cancella. [ e ] cambiano la dimensione.',
  'masks.kind.parametric.hint':
    'Seleziona per valori della foto: luminosità, saturazione, tinta. Gli intervalli attivi si moltiplicano.',
  'masks.kind.segment.hint': 'Il soggetto trovato dal modello di segmentazione.',

  'masks.range.luminance': 'Luminosità',
  'masks.range.saturation': 'Saturazione',
  'masks.range.hue': 'Tinta',
  'masks.range.low': 'Da',
  'masks.range.high': 'A',
  'masks.range.none': 'Nessun intervallo attivo: la maschera seleziona tutta la foto.',
  'masks.hue.center': 'Tinta centrale',
  'masks.hue.width': 'Ampiezza',
  'masks.limit.luminance': 'Solo in un intervallo di luminosità',
  'masks.limit.hue': 'Solo in una gamma di colori',

  'masks.brush.paint': 'Dipingi',
  'masks.brush.erase': 'Cancella',
  'masks.brush.size': 'Dimensione',
  'masks.brush.softness': 'Morbidezza',
  'masks.brush.flow': 'Flusso',
  'masks.brush.hint': 'Ogni tratto viene salvato quando rilasci il pulsante.',
  'masks.brush.saving': 'Salvataggio del tratto…',

  'masks.segment.title': 'Trova un soggetto',
  'masks.segment.sky': 'Cielo',
  'masks.segment.person': 'Persone',
  'masks.segment.skin': 'Pelle',
  'masks.segment.finding': 'ricerca…',
  'masks.segment.found': 'già trovato',
  'masks.segment.failed': 'La segmentazione non è riuscita.',
  'masks.segment.needsModel': 'Serve un modello da scaricare: {size} MB, licenza {licence}.',
  'masks.segment.download': 'Scarica il modello',
  'masks.segment.downloading': 'Download… {percent}%',
  'masks.segment.hint':
    'Calcolato sull’anteprima della foto. Le persone piccole in lontananza possono sfuggire: per quelle c’è il pennello.',
} as const
