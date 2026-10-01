# Rimozione: gomma magica e correttivo a punto

Prompt per l'agente implementatore. È un'estensione di [`SPEC.md`](SPEC.md), che resta il
riferimento normativo: tutto ciò che non è ripetuto qui vale comunque, in particolare §2 (non
distruttività), §6.2 (ordine delle operazioni e indipendenza dalla risoluzione), §6.3 (maschere
manuali, mai automatiche), §13 (test), §17 (modelli), §23 (storico) e §26 (regole di lavoro).
Le decisioni già prese con l'utente non si riaprono.

Le scelte di prodotto qui sotto sono state **decise con l'utente il 2026-09-30**. Registrale come
decisioni numerate nel diario di sviluppo prima di scrivere codice.

---

## 1. Cosa si vuole

Due strumenti nuovi nel tab **Maschere** dell'editor (Visualizzatore e Revisione), raccolti in
un gruppo **«Rimozione»** separato dalle maschere di regolazione:

1. **Gomma magica.** L'utente indica un'area (un oggetto, una persona, un palo, un cavo) e il
   programma la riempie in modo plausibile con il contenuto circostante.
2. **Correttivo a punto.** L'utente clicca su una piccola imperfezione (un brufolo, una macchia
   del sensore, un granello di polvere) e il programma la copre con una zona sorgente vicina,
   fondendo texture e colore. La sorgente viene scelta in automatico e l'utente può trascinarla
   altrove.

**Fuori dal perimetro** (non implementarli, e non aggiungere punti di aggancio «per dopo»):

- il timbro clone (copia esatta senza fusione);
- il rilevamento automatico delle imperfezioni. Violerebbe §6.3: nessuno strumento propone o
  applica rimozioni da solo;
- la traduzione delle rimozioni negli XMP di darktable o Lightroom (vedi §7).

## 2. Decisioni prese con l'utente

| # | Tema | Decisione |
|---|---|---|
| R1 | Strumenti | Gomma magica (pennello) e correttivo a punto. Niente clone e niente rilevamento automatico. |
| R2 | Motore | **Classico di base, ML opzionale.** Il riempimento classico funziona senza modelli e senza rete. Un modello ONNX di inpainting, scaricato su richiesta, migliora gli oggetti grandi. |
| R3 | Calcolo | **Una volta, poi salvato.** La parte costosa o non ripetibile (ricerca della sorgente, riempimento) si calcola una volta in un worker e si salva, indirizzata per contenuto. Il render compone il risultato e non lo ricalcola. |
| R4 | UI | Gruppo «Rimozione» a parte nel tab Maschere, con una lista propria di interventi. |
| R5 | Area della gomma | Si dipinge col pennello, **oppure** si parte da una maschera esistente (pennello, radiale, segmentazione…) e la si espande. |
| R6 | Propagazione | Le rimozioni restano **proprie della foto**: sono escluse da «applica alla scena», dall'apprendimento dello stile, dai predefiniti e dai profili `.apestyle`. L'unica eccezione è un'azione **esplicita**, «Copia punti sulle foto selezionate», pensata per le macchie del sensore. |

## 3. Modello dei dati

### 3.1 `EditParams.retouch`

Aggiungi una lista **`EditParams.retouch: list[RetouchItem]`**, distinta da `masks[]`. Una
rimozione non ha regolazioni, e mescolarla alle maschere confonderebbe sia il modello sia la UI.

Il bump di `PARAMS_VERSION` si fa con una migrazione nel registro `MIGRATIONS` di `params.py`:
una versione precedente si carica con `retouch = []`. Il test 2 di §13 (round-trip e caricamento
delle versioni precedenti) deve coprirla.

**Una lista vuota deve dare un render identico al bit a quello di oggi.** Serve un test esplicito,
e l'intera suite esistente deve continuare a passare invariata.

Gli interventi si applicano **nell'ordine della lista**: ogni intervento vede il risultato dei
precedenti, come in Lightroom.

Campi comuni a ogni intervento:

- `id` stabile, che serve alla UI e alla chiave della patch;
- `kind`: `"heal"` (correttivo a punto) o `"erase"` (gomma magica);
- `visible` (bool) e `opacity` (0–1);
- `feather`, come frazione del raggio o del bordo dell'area.

Tutte le coordinate sono **normalizzate nel fotogramma**, con la stessa convenzione di
`mask_defs.py` (dopo la correzione lente, prima di rotazione e crop). I raggi sono frazioni del
lato lungo.

**`heal`** (correttivo a punto):

- centro `cx, cy` e raggio `radius`;
- `source`: la posizione della sorgente (`sx, sy`) e `source_auto: bool`. Quando l'utente
  trascina la sorgente, `source_auto` diventa falso;
- la sorgente scelta in automatico **si salva nei parametri** al momento della creazione. Così
  il render è una funzione pura dei parametri, deterministica, e non cerca mai una sorgente da
  sé.

**`erase`** (gomma magica):

- `area`: nome SHA-256 di un raster 8 bit, come `RasterDef.raster`, salvato con `masks_store`;
- `expand`: allargamento dell'area, come frazione del lato lungo. Serve a coprire l'alone
  intorno a un oggetto;
- `engine`: `"classic"` o `"ml"`;
- `seed` (int): «Altra variante» lo incrementa. Il risultato resta deterministico per un dato
  seme, ma l'utente può chiederne un altro;
- `fill`: riferimento alla patch calcolata (§4.3), oppure `null` finché il calcolo non è finito.

Se l'area nasce da una maschera esistente (R5), la selezione di quella maschera **si fotografa
in un raster nuovo** al momento della creazione. Modificare dopo la maschera di partenza non
cambia la rimozione e non la fa ricalcolare di nascosto. Dillo nella UI con una riga di testo.

### 3.2 Contratto con il frontend

Aggiungi tipi, tabelle e schemi in `frontend/src/lib/` (`types.ts`, un `retouch.ts` accanto a
`masks.ts`) ed estendi `tests/test_frontend_contract.py`, così che backend e frontend non possano
divergere.

## 4. Pipeline

### 4.1 Posizione nell'ordine di §6.2

Aggiungi uno stadio **`retouch` subito dopo `lens` e prima di `noise`** in
`pipeline/stages.py`. I motivi:

- le coordinate del fotogramma nascono dopo la correzione lente, quindi prima non si può;
- il riempimento resta in lineare scene-referred, dove §6.2 mette tutto ciò che è fisico;
- la riduzione rumore passa uniforme sull'area riempita e sul resto, così la grana non tradisce
  la toppa;
- il risultato dipende solo dal decode, dalla correzione lente e dagli interventi precedenti.
  Muovere qualunque cursore (WB, esposizione, tono, colore, maschere di regolazione) non invalida
  nessuna patch.

Aggiorna lo schema di §6.2 nel commento di `STAGES` e documenta la posizione nel diario di sviluppo.

La chiave di cache dello stadio contiene la lista `retouch` (parametri e riferimenti alle patch).
Gli stadi a valle ereditano già la chiave a monte: verificalo con un test come quello di
`test_masks.py`, «stage cache = render fresco».

### 4.2 Operazioni pure

Ogni operazione è una funzione pura `(ndarray, params) -> ndarray` (§26), con una docstring che
dichiara spazio colore e range in ingresso e in uscita (Rec.2020 lineare, float32, bianco a 1.0).

**Correttivo a punto.** Copia la texture dalla sorgente e adatta colore e luminosità al bordo
della destinazione, con una fusione di tipo Poisson o un'equivalente a bassa frequenza. È
economico, quindi **si calcola al render**, a qualunque risoluzione. Il raggio è in coordinate
normalizzate e l'invarianza di risoluzione viene per costruzione. La scelta automatica della
sorgente cerca, fra le posizioni vicine, la zona più simile in texture e luminanza con la
destinazione esclusa. Gira una volta sul proxy, alla creazione o su «Scegli un'altra sorgente»,
e il risultato si salva nei parametri (§3.1).

**Gomma magica, motore classico.** Un riempimento per esempi (patch-based, PatchMatch o simili)
dal contorno verso l'interno. Deve funzionare senza modelli né rete (§17: le funzioni di default
non dipendono da alcun modello). `cv2.inpaint` (Telea/NS) da solo non basta oltre qualche decina
di pixel, perché impasta: usalo al massimo come inizializzazione. `opencv-python-headless` non
contiene `xphoto`. Passare a `opencv-contrib-python-headless`, o aggiungere qualunque altra
dipendenza, è una **domanda da fare all'utente prima** (§26).

**Gomma magica, motore ML.** Un modello ONNX di inpainting su CPU, via onnxruntime, in un worker
e mai nel server. Il modello va scelto secondo §5.

### 4.3 Patch salvate e invarianza di risoluzione

L'inpainting (classico o ML) calcolato a risoluzioni diverse inventa contenuti diversi. Se si
ricalcolasse al render, l'anteprima a 1024 px e l'export a piena risoluzione non
coinciderebbero, e il test 1 di §13 fallirebbe. Perciò la patch si separa in due livelli:

1. **Struttura.** Il worker calcola il riempimento **una sola volta, sul proxy**, nel riquadro
   dell'area più un margine, alla risoluzione di lavoro del motore. Salva il risultato come
   **patch lineare a 16 bit (o float16)**, indirizzata per contenuto (SHA-256) e scritta in modo
   atomico con `guarded_open`, come i raster. La chiave del contenuto include:
   - l'area e `expand`;
   - `engine` e `seed`;
   - la versione dell'algoritmo o del modello;
   - l'impronta di ciò che sta a monte: proxy token, parametri di `lens`, e le patch e i
     parametri degli interventi precedenti nella lista.
2. **Dettaglio.** Al render, a qualunque risoluzione, la struttura salvata si ricampiona sull'area
   e il **dettaglio ad alta frequenza** (grana, micro-texture) si sintetizza in modo
   deterministico dall'intorno reale dell'area, a quella risoluzione. Il metodo va scelto in modo
   che il render a 1024 px e il full-res ridotto a 1024 px restino sotto ΔE2000 medio 1,5
   nell'area.

Dove stanno le patch:

- sono **dati dell'utente, non cache**: un riempimento ML non si rigenera se il modello è stato
  rimosso. Vanno in `~/.local/share/autophotoedit/retouch/`, accanto a `masks/`;
- non si rimuovono mai in automatico;
- compaiono nella schermata Impostazioni fra le categorie dello spazio su disco;
- sono citate da `autophotoedit backup` come le maschere.

**Patch vecchie.** Se l'impronta a monte cambia (parametri della lente, proxy rigenerato, un
intervento precedente modificato), la patch non vale più. Il render allora applica la patch
vecchia, se c'è, e la UI segna l'intervento come «da ricalcolare», rimettendolo in coda in
automatico. È l'unico ricalcolo automatico ammesso, perché è la conseguenza di un gesto
dell'utente.

**Export.** Una foto con un `erase` senza patch valida calcola la patch dentro il job di export,
con la stessa funzione e lo stesso seme, quindi con lo stesso risultato. **Non si esporta mai in
silenzio senza la rimozione.** Se il motore ML è richiesto ma il modello non è disponibile, la
foto fallisce con un motivo chiaro nel pannello Problemi (§19).

### 4.4 Job

- Un job nuovo, `retouch_fill`, in un `jobs/handlers_retouch.py` (non allungare
  `handlers_masks.py`). Gira nei worker con priorità interattiva e avvisa il frontend via
  WebSocket quando ha finito.
- Mai nel batch automatico (§6.3): parte solo da un gesto dell'utente, da una patch diventata
  vecchia, o dall'export di una foto che ha già l'intervento.
- Un job superato da un gesto più recente sullo stesso intervento si annulla o si scarta, e non
  sovrascrive il risultato nuovo.

## 5. Modello ML (opzionale)

Seguono le regole di §17: registry dichiarativo in `models_registry.py`,
download solo su richiesta dalla UI (alla prima scelta di «Motore: IA»), revisione fissa, SHA-256
pubblicato dall'autore e verificato scaricando il file, voce in `models/LICENSES.md`.

- **Budget (§26):** ≤ 120 MB per il modello e ≤ 350 MB in totale. Misura il totale attuale dei
  modelli nel registry: sul disco dell'utente ci sono già ~100 MB di segmentazione, più CLIP e gli
  altri. Riporta i numeri.
- **Candidati da verificare**, senza darli per buoni:
  - **MI-GAN** (Picsart AI Research, dichiarato MIT, ~30 MB, 512 px);
  - **LaMa** (dichiarato Apache-2.0). Il big-lama FP32 in ONNX pesa ~200 MB e sfora: valuta
    FP16 o int8, misurando la qualità persa.

  Per ciascuno verifica la licenza **dei pesi** (non solo del codice), la provenienza della
  conversione ONNX e i dati di addestramento. Places2 e CelebA hanno termini propri: dichiara il
  dubbio all'utente nella UI e in `LICENSES.md`, come per SkySeg.
- **Vietati:** i modelli con licenza non commerciale o solo di ricerca sui pesi, anche se
  migliori.
- **Prima di fissare il modello, chiedi all'utente.** Presenta i candidati con licenza, peso,
  tempo misurato su CPU a 512 px, RAM di picco del worker, e 3–4 confronti visivi sulle foto vere
  contro il motore classico. Poi registra la scelta come decisione nel diario di sviluppo, con tutti i
  dettagli di pin (revisione, file, dimensione, SHA-256) come per i modelli di segmentazione.
- Determinismo: fissa `intra_op_num_threads` e `inter_op_num_threads` della sessione e verifica
  che due esecuzioni diano byte identici (test 4 di §13). Se non è così, dillo con i numeri
  prima di aggirarlo.
- Caricamento pigro e scaricamento dopo 5 minuti di inattività, come gli altri modelli.

## 6. Interfaccia

Tutti i testi sono in italiano in `src/i18n/it*.ts`, letti con `t()`. Nessun file oltre le ~400
righe (`test_no_frontend_file_is_oversized`): `MaskPanel.tsx` è già a 205 righe, quindi il gruppo
Rimozione va in componenti propri in `components/masks/` (o in `components/retouch/`).

**Pannello.** Il tab Maschere mostra la sezione «Rimozione» sopra o sotto la lista delle
maschere, secondo lo stile che c'è. Contiene:

- i due strumenti, «Correttivo» e «Gomma magica»;
- la lista degli interventi: tipo, numero d'ordine, stato (pronto / in calcolo / da ricalcolare
  / errore), occhio per `visible`, cestino;
- per l'intervento selezionato: dimensione, sfumatura, opacità. Per la gomma anche espandi,
  motore (Classico / IA) e «Altra variante». Per il correttivo anche «Scegli un'altra sorgente»;
- in fondo, la riga che già avverte che gli XMP non contengono le maschere, estesa alle
  rimozioni.

**Correttivo a punto.**

- Un clic crea l'intervento con il raggio corrente e la sorgente automatica.
- Si vedono due cerchi (destinazione e sorgente) uniti da una linea, come `ShapeHandles`. Si
  trascina l'uno o l'altro; `[` e `]` cambiano il raggio.
- Il risultato appare subito, perché è calcolato al render (§4.2).

**Gomma magica.**

- Riusa `BrushCanvas` (dimensione, morbidezza, Alt per cancellare) con una sovrapposizione rossa
  semitrasparente. Al rilascio del puntatore carica il raster e accoda il job.
- Mentre il job lavora, l'area mostra un indicatore di avanzamento sopra l'immagine originale.
- Il pulsante «Da una maschera…» elenca le maschere della foto (una segmentazione non ancora
  eseguita si può lanciare da lì, on demand), fotografa la selezione, applica `expand` e crea
  l'intervento.

**Scorciatoie.** Proponi i tasti per i due strumenti (Lightroom usa Q per la rimozione). Verifica
che non collidano in `lib/shortcuts.ts` e aggiungili alla tabella, così che `?` li mostri. Canc
ed Esc valgono per l'intervento selezionato come per le maschere.

**Prima/dopo.** Il confronto esistente, se c'è, mostra anche l'effetto delle rimozioni. Serve
anche un interruttore «mostra rimozioni» per sospenderle tutte durante il lavoro, senza cambiare
i parametri.

**Storico (§23).** Ogni gesto concluso è un passo di Ctrl+Z e rientra nella
coalescenza del salvataggio automatico: un clic, un rilascio del pennello, il trascinamento di
una sorgente, un cambio di motore o di variante. L'arrivo di una patch dal worker **non** è un
gesto e non crea una `EditVersion` nuova visibile nella striscia come «modificata da te». Decidi
come agganciare il riferimento alla patch senza sporcare lo storico, e spiegalo nel diario di sviluppo.

**Copia punti sulle foto selezionate (R6).**

- Si trova nel menu della sezione Rimozione. Copia i correttivi a punto (le gomme solo se l'utente
  spunta «anche le gomme») sulle foto selezionate in griglia o nella striscia.
- La posizione va conservata **sul sensore, non nel fotogramma raddrizzato**: una foto verticale
  ha l'orientamento LibRaw (`flip`, vedi `raw/embedded.py`) diverso da una orizzontale, e la
  macchia del sensore resta nello stesso punto del sensore. Converti le coordinate passando per
  l'orientamento.
- La sorgente automatica si ricalcola per ogni foto di destinazione.
- Si aggiunge in coda agli interventi della foto di destinazione, senza sostituirli. Conferma
  prima col numero di foto toccate.
- È un'azione esplicita, mai automatica, e non passa da «applica alla scena».

## 7. Integrazioni da non dimenticare

- **Esclusioni (R6):**
  - «applica alla scena» (`review/decisions.apply_to_scene`: le maschere sono già proprie della
    foto, vedi il docstring a l.10) lascia `retouch` com'è sulla foto di destinazione;
  - `style/vector.py` e `style/apply.py` non leggono e non scrivono `retouch`;
  - lo stesso vale per i predefiniti di §22 e per i profili `.apestyle` di §20.1.

  Serve un test per ciascuno.
- **XMP:** i sidecar darktable e Adobe non contengono le rimozioni. Estendi l'avviso del piano di
  export (`export/service.py`, quello delle maschere): «N foto hanno rimozioni: le immagini
  esportate le contengono, gli XMP no».
- **Snapshot e ripristino (§23.2):** una versione ripristinata con rimozioni ritrova le sue patch
  (sono dati dell'utente e non spariscono). Serve un test.
- **Fusioni (§25):** una foto derivata si comporta come uno scatto singolo. Il proxy token
  dell'impronta è quello della derivata.
- **Non distruttività (§2):** ogni scrittura di raster o patch passa da `assert_outside_source`.
  Il test 8 di §13 si estende con un ciclo che include una gomma e un correttivo.

## 8. Test richiesti

Vanno scritti **insieme al codice** (§26), in `tests/test_retouch.py`,
`tests/test_retouch_api.py` e, se serve, in `tests/test_retouch_model.py` (`slow`, saltato se il
modello manca):

1. **Neutralità:** `retouch = []` dà un render identico al bit a quello attuale.
2. **Località:** i pixel fuori dall'area (più la sfumatura) sono identici al bit al render senza
   l'intervento.
3. **Efficacia:** su un'immagine sintetica con un oggetto noto su uno sfondo uniforme o
   texturizzato, l'area riempita ha ΔE medio rispetto allo sfondo vero sotto una soglia che
   fisserai e motiverai nel commento. Vale per correttivo, gomma classica e gomma ML.
4. **Invarianza di risoluzione** (test 1 di §13): render a 1024 px contro full-res ridotto, ΔE2000
   medio < 1,5 anche **dentro** le aree rimosse, con entrambi i motori.
5. **Determinismo** (test 4): stessa foto, stessi parametri e stesso seme danno byte identici,
   anche ricalcolando la patch da zero. Un seme diverso dà una patch diversa.
6. **Ordine:** due interventi sovrapposti danno risultati diversi se scambiati, e il secondo vede
   il primo.
7. **Patch vecchia:** cambiare i parametri della lente segna l'intervento da ricalcolare. Cambiare
   WB, esposizione o una maschera di regolazione non lo segna.
8. **Export senza patch:** la patch viene calcolata e l'immagine esportata coincide con quella
   fatta con la patch già pronta. Con ML richiesto e modello assente, la foto fallisce con il
   motivo giusto.
9. **Round-trip e migrazione dei parametri** (test 2 di §13).
10. **Esclusioni:** scena, stile, predefiniti, `.apestyle` (§7).
11. **Copia punti:** una macchia copiata da una foto orizzontale a una verticale cade nello stesso
    punto del sensore.
12. **API:** caricamento dell'area, job, patch, rifiuto di nomi e raster non validi (come
    `test_masks_api.py`), sorgente intatta.
13. **Contratto frontend** e **licenze** (`test_licence_isolation.py`: nessuna dipendenza vietata;
    `LICENSES.md` elenca il modello).

## 9. Prestazioni e budget

Sono obiettivi da misurare e riportare con i numeri, non da stimare (§26). Se non li raggiungi,
dillo.

- Correttivo a punto: la ricerca automatica della sorgente < 300 ms sul proxy, e il ridisegno
  dell'anteprima dopo un trascinamento non deve essere visibilmente più lento del render attuale.
- Gomma classica: < 2 s sul proxy per un'area fino al 5% del fotogramma, < 6 s fino al 20%.
- Gomma ML: < 3 s su CPU per un'area fino al 20%, modello già caricato. Il primo caricamento va
  misurato a parte.
- RAM: server < 400 MB, worker < 1,5 GB anche con il modello ML caricato e una foto a piena
  risoluzione.
- Export: misura quanto aggiunge una foto con 5 correttivi e 1 gomma rispetto alla stessa foto
  senza. Per confronto, le maschere costano ~1,1 s a maschera.
- Bundle JS < 1,5 MB gzip (oggi ~160 KB). A idle ~0% di CPU: nessun polling dello stato dei job,
  si usa il WebSocket.

## 10. Consegna

1. Prima di scrivere codice, registra nel diario di sviluppo le decisioni R1–R6 come decisioni numerate e
   fai all'utente le domande aperte:
   - la dipendenza per il riempimento classico, se ne serve una nuova;
   - il modello ML, dopo il confronto di §5.
2. Implementa in quest'ordine, verificando ogni passo con i test prima di andare avanti:
   1. modello dati e migrazione;
   2. stadio `retouch` con il correttivo a punto;
   3. UI del correttivo;
   4. gomma classica con patch e job;
   5. UI della gomma, compreso «Da una maschera…»;
   6. copia punti;
   7. motore ML.

   Il correttivo è consegnabile e utile anche da solo.
3. **Prova tutto sui RAW veri dell'utente** (`tests/fixtures/`, `tests/rawEdited/raw`): almeno
   una macchia del sensore su cielo, un'imperfezione su un volto, un oggetto piccolo e uno grande
   (una persona sullo sfondo), con entrambi i motori. Riporta tempi, RAM e il giudizio visivo.
4. Aggiorna il diario di sviluppo con una sezione datata «Rimozione: cosa è stato fatto», con
   «misurato» e «aperti» come per le altre fasi, e aggiorna i conteggi della suite. Aggiorna
   il README se cambia qualcosa per l'utente (nuove scorciatoie, nuovo modello scaricabile, nuova
   cartella da includere nei backup).
