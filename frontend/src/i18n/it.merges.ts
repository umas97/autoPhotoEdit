// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The strings of phase 11 -- the merges screen, the badge in the grid, the
// selection that makes a group by hand -- kept apart so that it.ts stays
// readable. They are spread into the one table there.
export const merges = {
  'nav.merges': 'Fusioni',

  'merges.title': 'Fusioni',
  'merges.subtitle':
    'Bracketing, focus stacking e panoramiche trovati fra gli scatti. Nessuna fusione parte senza «Accetta»: i RAW restano intatti e una fusione si può sempre annullare.',
  'merges.disabled':
    'Il rilevamento delle fusioni è spento per questo progetto. Puoi comunque creare una fusione a mano dal visualizzatore.',
  'merges.searching': 'Ricerca delle panoramiche in corso…',
  'merges.empty':
    'Nessuna fusione proposta. Per crearne una a mano, nel visualizzatore scegli più foto con Ctrl+clic e poi «Crea fusione».',
  'merges.section.proposed': 'Da decidere ({count})',
  'merges.section.failed': 'Non riuscite ({count})',
  'merges.section.accepted': 'Accettate ({count})',
  'merges.section.rejected': 'Rifiutate ({count})',

  'merges.kind.hdr': 'HDR',
  'merges.kind.panorama': 'Panorama',
  'merges.kind.focus_stack': 'Focus stack',
  'merges.badge': '{kind} · {count}',
  'merges.confidence': 'confidenza {value}%',
  'merges.manual': 'creata a mano',

  'merges.reason.frames': '{count} scatti',
  'merges.reason.ev': 'EV da {min} a {max}',
  'merges.reason.span': 'intervallo {seconds} s',
  'merges.reason.cameraBracket': 'bracketing della fotocamera',
  'merges.reason.focusExif': 'messa a fuoco che si sposta (EXIF)',
  'merges.reason.focusPixels': 'zona nitida che si sposta',
  'merges.reason.overlap': 'sovrapposizione {min}–{max}%',
  'merges.reason.sweep': 'scattata a raffica ruotando',

  'merges.member.reference': 'Riferimento',
  'merges.member.missing': 'mancante',
  'merges.member.ev': '{ev} EV',

  'merges.preview': 'Anteprima',
  'merges.preview.none': 'Nessuna anteprima: «Anteprima» la calcola a 1024 px in pochi secondi.',
  'merges.preview.working': 'Anteprima in calcolo…',
  'merges.preview.stale': 'Il gruppo è cambiato: l’anteprima va ricalcolata.',
  'merges.preview.error': 'Anteprima non riuscita: {reason}',
  'merges.preview.alt': 'Anteprima della fusione',
  'merges.preview.coverage': 'Mostra la copertura',
  'merges.report.misaligned':
    'Allineamento incerto: fino a {px} px di scarto fra gli scatti. Controlla i bordi prima di accettare.',
  'merges.report.ghosts': 'Soggetti in movimento: sul {percent}% della foto conta solo il riferimento.',
  'merges.report.uncovered':
    'Nessuno scatto è a fuoco sul {percent}% della foto: la copertura mostra dove.',
  'merges.report.crop': 'Crop proposto per i bordi irregolari: lo trovi fra le proposte della foto.',
  'merges.report.used':
    'Cuce {used} dei {members} scatti: gli altri si sovrappongono tanto da non aggiungere nulla.',

  'merges.accept': 'Accetta',
  'merges.reject': 'Rifiuta',
  'merges.edit': 'Modifica gruppo',
  'merges.retry': 'Riprova',
  'merges.undo': 'Annulla la fusione',
  'merges.restore': 'Riproponi',
  'merges.open': 'Apri la foto',
  'merges.running': 'Fusione a piena risoluzione in corso… {percent}%',
  'merges.queued': 'Fusione in coda',
  'merges.failed': 'Non riuscita: {reason}',
  'merges.done': 'Fusione fatta: la foto derivata è nel progetto al posto dei suoi scatti.',
  'merges.decision.proposed': 'proposta',
  'merges.decision.accepted': 'accettata',
  'merges.decision.rejected': 'rifiutata',
  'merges.decision.failed': 'non riuscita',

  'merges.editTitle': 'Modifica gruppo',
  'merges.editBody':
    'Scegli gli scatti da fondere e quello di riferimento: da lui la foto derivata prende bilanciamento, geometria e metadati.',
  'merges.editNearby': 'Scatti vicini',
  'merges.editReference': 'Riferimento',
  'merges.editSave': 'Salva e ricalcola l’anteprima',
  'merges.editTooFew': 'Servono almeno due scatti.',
  'merges.option.deghost': 'Rimuovi i fantasmi (soggetti in movimento)',
  'merges.option.projection': 'Proiezione',
  'merges.projection.cylindrical': 'Cilindrica',
  'merges.projection.spherical': 'Sferica',
  'merges.projection.plane': 'Piana',
  'merges.option.outputScale': 'Risoluzione della panoramica',
  'merges.report.panorama': 'Proiezione {projection}, {span}° di ampiezza, {mp} MP a piena risoluzione',
  'merges.report.panoramaVertical':
    'Panoramica verticale, proiezione {projection}, {span}° di altezza, {mp} MP a piena risoluzione',

  'merges.pick.count': '{count} foto scelte',
  'merges.pick.create': 'Crea fusione:',
  'merges.pick.clear': 'Annulla la scelta',
  'merges.pick.hint': 'Ctrl+clic per scegliere più foto da fondere',
  'merges.showSources': 'Mostra scatti sorgente',
  'merges.sourceMark': 'sorgente',
  'merges.expand': 'Mostra gli scatti di questa fusione',
} as const
