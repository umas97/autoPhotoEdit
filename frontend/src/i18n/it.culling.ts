// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strings of the culling screen (section 7) -- filters, the panel of
// criteria, the comparison view and "why" -- kept apart so that it.ts stays
// readable. They are spread into the one table there.
export const culling = {
  'culling.toViewer': 'Vai alle foto',
  'culling.empty': 'Nessuna foto con questo filtro.',
  'culling.filter.all': 'Tutte',
  'culling.filter.selected': 'Solo selezionate',
  'culling.filter.culled': 'Solo scarti',
  'culling.sort': 'Ordina per',
  'culling.sort.time': 'orario',
  'culling.sort.score': 'punteggio',
  'culling.sort.name': 'nome file',
  'culling.analysing': 'Valutazione in corso: {done} di {total}',
  'culling.shortcuts':
    'X scarta · P tieni · ←/→ naviga · ↑/↓ cambia la scelta nella raffica · Spazio confronta · Ctrl+Z annulla',
  'culling.bottom.count': '{selected} selezionate su {total}',
  'culling.bottom.proceed': 'Procedi con l’editing',
  'culling.bottom.hint': 'Lo sviluppo parte solo da qui.',
  'culling.bottom.wait': 'Attendi la fine della valutazione.',

  'culling.card.score': 'Punteggio complessivo (0–100)',
  'culling.action.keep': 'Tieni',
  'culling.action.discard': 'Scarta',

  'culling.reason.out_of_focus': 'fuori fuoco',
  'culling.reason.motion_blur': 'mosso',
  'culling.reason.overexposed': 'alte luci perse oltre il recupero',
  'culling.reason.underexposed': 'troppo scura anche per il RAW',
  'culling.reason.burst_duplicate': 'doppione della raffica',
  'culling.reason.below_target': 'oltre l’obiettivo di selezione',
  'culling.reason.user': 'scartata da te',

  'culling.burst.counter': '{kept} di {total}',
  'culling.burst.open': 'Apri la raffica',
  'culling.burst.title': 'Raffica · {count} scatti',
  'culling.burst.hint': 'La proposta è il primo; scegline un altro o tienine più d’uno.',
  'culling.burst.proposed': 'proposta',
  'culling.burst.choose': 'Scegli',
  'culling.burst.keepToo': 'Tieni anche questa',

  'culling.merge.badge': '{kind} · {count}',
  'culling.merge.kind.hdr': 'HDR',
  'culling.merge.kind.panorama': 'Panorama',
  'culling.merge.kind.focus_stack': 'Focus stack',
  'culling.merge.protected':
    'Proposta di fusione: gli scatti non vengono valutati come foto singole.',

  'culling.compare.title': 'Confronto',
  'culling.compare.hint':
    'Doppio clic o Z per l’1:1 sul punto di fuoco · trascina per spostarti · Esc chiude',
  'culling.compare.actual': '1:1',
  'culling.compare.fit': 'Adatta',

  'culling.panel.mode': 'Modalità',
  'culling.mode.conservative': 'Conservativa',
  'culling.mode.conservative.hint':
    'Scarta solo ciò che è compromesso oltre il recupero e i doppioni delle raffiche.',
  'culling.mode.target_percent': 'Tieni una percentuale',
  'culling.mode.target_percent.hint':
    'Prima un fotogramma per ogni momento, poi i secondi delle raffiche.',
  'culling.mode.target_count': 'Tieni un numero di foto',
  'culling.mode.target_count.hint':
    'Prima un fotogramma per ogni momento, poi i secondi delle raffiche.',
  'culling.target.percent': '% (= {count} foto)',
  'culling.target.count': 'foto',
  'culling.target.shortfall':
    'Mancano {count} foto all’obiettivo: le altre sono scarti tecnici e non vengono usate per riempirlo.',

  'culling.panel.aggressiveness': 'Aggressività',
  'culling.aggressiveness': 'Aggressività',
  'culling.threshold': 'Soglia tecnica: {value}/100',
  'culling.latency': 'selezione aggiornata in {ms} ms',

  'culling.panel.criteria': 'Criteri',
  'culling.criterion.sharpness': 'Nitidezza',
  'culling.criterion.sharpness.hint':
    'Misurata sulla zona più nitida: uno sfondo sfocato non penalizza.',
  'culling.criterion.motion': 'Mosso',
  'culling.criterion.motion.hint': 'Distingue il micromosso dal fuori fuoco.',
  'culling.criterion.exposure': 'Esposizione',
  'culling.criterion.exposure.hint':
    'Penalizza solo ciò che il RAW non recupera (circa 2 stop nelle luci, 4 nelle ombre).',
  'culling.criterion.burst': 'Raffiche e quasi-doppioni',
  'culling.criterion.burst.hint':
    'Scatti a meno di 2 secondi e visivamente simili: ne viene proposto uno.',
  'culling.criterion.faces': 'Volti',
  'culling.criterion.faces.hint':
    'Occhi chiusi ed espressione del soggetto principale. Costo: circa +0,15 s per foto.',
  'culling.criterion.aesthetic': 'Estetica',
  'culling.criterion.aesthetic.hint':
    'Ordinamento soggettivo, usalo come suggerimento. Costo: circa +0,25 s per foto.',
  'culling.unavailable.runtime_missing':
    'Non disponibile: richiede ONNX Runtime, che non è installato (pacchetto opzionale «ml»).',
  'culling.unavailable.not_pinned':
    'Non disponibile: il modello non è ancora verificato in questa versione; arriverà con la gestione dei modelli.',
  'culling.unavailable.model_missing': 'Non disponibile: il modello non è stato scaricato.',
  'culling.unavailable.checksum_mismatch':
    'Non disponibile: il file del modello non corrisponde al checksum atteso ed è stato scartato.',
  'culling.notice.ava_research_only':
    'Modello addestrato su un dataset di sola ricerca; valuta tu l’idoneità per un uso commerciale.',
  'culling.notice.places2_terms':
    'Addestrato su Places2, le cui immagini sono concesse per ricerca non commerciale: valuta tu l’idoneità per un uso commerciale.',
  'culling.notice.skyseg_provenance':
    'Chi pubblica il modello del cielo non dichiara su quali dati è stato addestrato: valuta tu l’idoneità per un uso commerciale.',

  'culling.panel.weights': 'Pesi',
  'culling.weights.formula':
    'Punteggio = media dei criteri attivi, pesata con questi valori.',
  'culling.panel.footnote': 'Valutate {analysed} di {total}, {failed} illeggibili. Selezionate:',

  'culling.why.kept': 'Tenuta',
  'culling.why.culled': 'Scartata',
  'culling.why.byUser': 'Decisione tua: nessun ricalcolo la cambia.',
  'culling.why.criterion': 'Criterio',
  'culling.why.score': 'Punteggio',
  'culling.why.weight': 'Peso',
  'culling.why.total': 'Punteggio complessivo: {score}/100',
  'culling.why.burst': 'Raffica: {position}° di {total}',
  'culling.why.notAssessable':
    'Nitidezza non valutabile: la foto non ha dettagli da mettere a fuoco (cielo, nebbia). Non viene scartata per questo.',
} as const
