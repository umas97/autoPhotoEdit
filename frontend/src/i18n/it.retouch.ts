// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The words of the Rimozione group of the masks tab: the spot healing brush,
// the magic eraser, their states, "Copia punti". Spread into the one table in
// it.ts.
export const retouch = {
  'retouch.title': 'Rimozione',
  'retouch.heal': 'Correttivo',
  'retouch.heal.hint': 'Correttivo a punto (Q): clicca su una macchia, un brufolo, un granello di polvere',
  'retouch.erase': 'Gomma magica',
  'retouch.erase.hint': 'Gomma magica (Maiusc+Q): dipingi sopra ciò che vuoi togliere',
  'retouch.fromMask': 'Da una maschera…',
  'retouch.fromMask.title': 'Rimuovi quello che seleziona una maschera',
  'retouch.fromMask.note':
    'L’area si fotografa adesso: se poi modifichi la maschera, la rimozione resta com’era.',
  'retouch.fromMask.none':
    'Questa foto non ha maschere. Per il cielo o una persona crea prima una maschera «Soggetto».',
  'retouch.fromMask.cancel': 'Annulla',
  'retouch.list': 'Rimozioni della foto',
  'retouch.item.heal': 'Correttivo {n}',
  'retouch.item.erase': 'Gomma {n}',
  'retouch.state.ready': 'pronto',
  'retouch.state.computing': 'in calcolo',
  'retouch.state.stale': 'da ricalcolare',
  'retouch.state.error': 'errore',
  'retouch.visible': 'Mostra o nascondi',
  'retouch.delete': 'Elimina (Canc)',
  'retouch.size': 'Dimensione',
  'retouch.feather': 'Sfumatura',
  'retouch.opacity': 'Opacità',
  'retouch.expand': 'Espandi',
  'retouch.engine': 'Motore',
  'retouch.engine.classic': 'Classico',
  'retouch.engine.ml': 'IA',
  'retouch.variant': 'Altra variante',
  'retouch.otherSource': 'Scegli un’altra sorgente',
  'retouch.showRemovals': 'Mostra le rimozioni',
  'retouch.empty':
    'Nessuna rimozione. Il correttivo copre una piccola imperfezione con una zona vicina; la gomma magica toglie un oggetto e lo riempie con ciò che lo circonda.',
  'retouch.hint.heal':
    'Clicca sulla macchia. Trascina il cerchio pieno per spostarla, quello tratteggiato per cambiare la sorgente; [ e ] cambiano la dimensione.',
  'retouch.hint.erase':
    'Dipingi sopra l’oggetto: al rilascio viene riempito. Con una gomma selezionata il pennello ne corregge l’area (Alt per togliere); Esc per iniziarne un’altra.',
  'retouch.computing': 'Riempimento in corso…',
  'retouch.retry': 'Riprova',
  'retouch.failed': 'Rimozione non riuscita: {message}',
  'retouch.menu': 'Altre azioni',
  'retouch.copy': 'Copia punti sulle foto selezionate…',
  'retouch.copy.title': 'Copia i punti su altre foto',
  'retouch.copy.body':
    'Per le macchie del sensore: ogni correttivo va nello stesso punto del sensore delle foto selezionate, anche se sono verticali, con la sorgente cercata di nuovo su ciascuna. Si aggiunge alle rimozioni che hanno già.',
  'retouch.copy.pick': 'Seleziona prima le foto nella griglia a sinistra, con Ctrl+clic o Maiusc+clic.',
  'retouch.copy.erase': 'Anche le gomme',
  'retouch.copy.confirm': 'Copia su {photos} foto',
  'retouch.copy.done': 'Rimozioni in copia su {photos} foto',
  'retouch.copy.skipped': '{count} fusioni saltate: non hanno un sensore su cui mettere i punti',
  'retouch.copy.nothing': 'Questa foto non ha correttivi da copiare',
  'retouch.notInSidecars':
    'Maschere e rimozioni valgono nell’export JPEG/TIFF; gli XMP per darktable e Lightroom non le riportano.',
  'retouch.ml.unavailable': 'Motore IA non disponibile: {reason}',

  'shortcuts.heal': 'Correttivo a punto',
  'shortcuts.erase': 'Gomma magica',
  'shortcuts.retouchSize': 'Correttivo o pennello più piccolo / più grande',
  'settings.cat.retouch': 'Riempimenti della gomma magica (non cache)',
  'settings.backupRetouch': 'E quella dei riempimenti della gomma magica: {path}',
}
