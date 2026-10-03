# autoPhotoEdit — Specifica di implementazione

> **Destinatario:** agente di coding (Claude Code) che implementa il progetto da zero.
> **Lingua del codice/commenti:** inglese. **Lingua della UI:** italiano.
> Questo documento è la fonte di verità. Se una scelta implementativa contraddice questo file,
> vince questo file; se questo file è ambiguo su un dettaglio, chiedi prima di inventare.

---

## 1. Obiettivo

Costruire un'applicazione desktop-locale che esegue **post-produzione automatica di file RAW Sony
(A7 II / A7 III, `.ARW`)**, imparando lo stile da foto già post-prodotte fornite dall'utente, e
chiedendo conferma all'utente solo sugli scatti in cui non è sufficientemente sicura.

Il programma deve:

1. **Non toccare mai i file originali.** L'intera applicazione è a sola lettura sui RAW.
2. Offrire, in modo **opzionale e scelto dall'utente all'import**, una fase di **cernita**
   (selezione degli scatti migliori) prima dell'editing.
3. Sviluppare i RAW con una pipeline propria (esposizione, bilanciamento colore, contrasto, luci,
   ombre, bianchi, neri, saturazione/vividezza, curve, contrasto locale, maschere parametriche).
4. Raddrizzare automaticamente (orizzonte + verticali) e applicare correzione lente da profilo.
5. **Proporre** un crop compositivo che l'utente accetta o rifiuta (mai applicato in automatico).
6. Imparare uno o più **profili di stile** da coppie RAW + JPEG/TIFF già editati.
7. Scegliere automaticamente il profilo di stile più adatto alla scena, mantenendo coerenza
   stilistica all'interno del progetto.
8. Esportare JPEG/TIFF sviluppati **e** sidecar XMP (darktable + Adobe/Lightroom).
9. Presentare all'utente, in una UI web locale, i cluster rappresentativi e gli outlier a bassa
   confidenza per la validazione.
10. **Proporre** la fusione di gruppi di scatti — HDR da bracketing, panorami, focus stacking —
    che l'utente conferma o rifiuta, producendo una foto derivata che entra nel flusso normale
    di editing, revisione ed export (§25).

### Non-obiettivi (v1)

- Nessun ritocco pixel-level (rimozione imperfezioni, liquify, cloni).
- Nessun supporto a RAW di altri produttori oltre a Sony ARW (l'architettura non deve però
  precludere l'estensione: usa LibRaw, che li supporta già).
- Nessun cloud, nessuna API esterna a pagamento, nessun upload. Tutto gira offline.
- Nessuna gestione di video.
- Nessun allineamento/fusione di scatti a mano libera con parallasse forte: le fusioni di §25
  assumono scatti ripresi dalla stessa posizione (HDR, stacking) o con rotazione attorno al
  punto nodale (panorama). Quando l'assunzione cade, il programma lo dichiara e non fonde.

---

## 2. Non distruttività — invariante assoluto

**I file RAW originali non vengono mai modificati, spostati, rinominati o cancellati. Mai, per
nessun motivo, in nessuna fase.** Questo vincolo ha precedenza su ogni altro requisito di questo
documento: se una funzionalità richiedesse di toccare un originale, la funzionalità si taglia.

Regole operative che l'implementazione deve rispettare:

1. **La cartella sorgente è trattata come sola lettura.** Il backend apre i RAW esclusivamente con
   `open(path, 'rb')` / `rawpy.imread`. Non esiste in tutto il codice una sola scrittura,
   `os.rename`, `os.remove`, `shutil.move` o `chmod` con destinazione dentro `Project.source_dir`.
2. **Test automatico obbligatorio** (`tests/test_non_destructive.py`): hash SHA-256 e `mtime` di
   ogni file della cartella sorgente calcolati prima di un ciclo completo
   import → cernita → analisi → edit → export e riverificati dopo. Qualunque differenza fa
   fallire il test. Gira in CI e non può essere marcato `skip`.
3. **Guardia a runtime:** una funzione `assert_outside_source(path, project)` viene chiamata da
   ogni funzione che apre un file in scrittura e solleva un'eccezione se la destinazione cade
   dentro la cartella sorgente (confronto su `Path.resolve()`, symlink inclusi). È l'ultima rete
   di sicurezza, non una formalità.
4. **Tutto ciò che il programma produce vive altrove:**
   - database, cache, proxy, maschere raster, modelli → `~/.local/share/autophotoedit/`
     (XDG, `$XDG_DATA_HOME`);
   - export → cartella di destinazione scelta dall'utente, che deve essere **diversa** dalla
     sorgente (la UI rifiuta la coincidenza);
   - sidecar XMP → di default nella cartella di export, **non** accanto ai RAW. Scriverli accanto
     ai RAW resta possibile, ma solo dietro un'opzione esplicita disattivata di default e con un
     avviso chiaro che aggiunge file (senza modificarne alcuno) nella cartella sorgente.
5. **Le modifiche sono dati, non pixel.** Ogni edit è un `EditParams` JSON nel database. I pixel
   sviluppati esistono solo come cache rigenerabile e come export finale: cancellare l'intera
   cache non perde nulla di irrecuperabile.
6. **Nessuna scrittura EXIF sugli originali**, in nessun caso, nemmeno rating o etichette: rating
   e flag di cernita vivono nel database.

---

## 3. Vincoli di ambiente (macchina target, già verificata)

| Voce | Valore |
|---|---|
| OS | Ubuntu (kernel 7.0.x) |
| CPU | Intel i5 13ª gen — 16 thread logici |
| GPU | **solo grafica integrata** — nessuna CUDA, nessun PyTorch GPU |
| RAM | 15 GB totali, ~8 GB tipicamente disponibili |
| Python | 3.12.3 (già presente) |
| Node | 24.21 (già presente) |
| Browser | **Google Chrome** (`/usr/bin/google-chrome`) e Firefox — verificati |
| Non installati | `darktable-cli`, `rawtherapee-cli`, `exiftool` |

**Conseguenze vincolanti:**

- Tutta l'inferenza ML gira su **CPU** tramite **ONNX Runtime** (`onnxruntime`, CPU EP). Niente
  PyTorch a runtime; se serve per convertire modelli, è una dipendenza solo di sviluppo.
- Il budget RAM per worker è ~1.5 GB. Un ARW 24 MP in float32 RGB = ~290 MB: la pipeline
  full-res deve lavorare in-place dove possibile e non tenere più di 3 buffer contemporanei.
- Il parallelismo è per-immagine (process pool), non intra-immagine.
- `exiftool` è una dipendenza opzionale: usa `exiv2`/`pyexiv2` come default e degrada con grazia.
- La finestra dedicata di §21.2 ha su questa macchina il browser che le serve (Chrome, famiglia
  Chromium): **il percorso normale è la finestra, non la scheda.** Il fallback a scheda esiste per
  altre macchine, non per questa, e non va usato come scusa per non curare la modalità finestra.

---

## 4. Stack tecnologico (vincolante)

### Backend

- **Python 3.12**, gestione ambiente con **`uv`** (`pyproject.toml`, niente `requirements.txt`).
- **FastAPI** + **uvicorn** — API REST + WebSocket per il progresso.
- **rawpy** (LibRaw) — decodifica RAW.
- **NumPy** + **OpenCV** (`opencv-python-headless`) — pipeline immagine.
- **SciPy** — ottimizzazione (`scipy.optimize`), interpolazione curve.
- **lensfunpy** — profili obiettivo (distorsione, vignettatura, TCA).
- **colour-science** — conversioni di spazio colore corrette (Lab, Rec.2020, sRGB, CAT).
- **onnxruntime** — embedding di scena e (opzionale) segmentazione.
- **SQLAlchemy 2.x** + **SQLite** — catalogo e stato.
- **Pydantic v2** — schemi API e serializzazione dei parametri di edit.
- **pyexiv2** — lettura EXIF e scrittura sidecar XMP. *(GPL-3.0: determina la licenza del
  progetto, vedi §24. Tutto l'accesso a exiv2 resta confinato in `raw/metadata.py` e in
  `export/`, così resta sostituibile.)*
- **pytest** — test.

### Frontend

- **React 18 + TypeScript + Vite**, **Tailwind CSS**, **shadcn/ui** per i componenti,
  **TanStack Query** per lo stato server, **Zustand** per lo stato UI locale.
- **Lingua UI: solo italiano**, ma nessuna stringa hardcoded nei componenti: tutte le stringhe
  visibili vivono in `src/i18n/it.ts`, accedute via una funzione `t('chiave')`. Aggiungere una
  lingua in futuro deve essere un lavoro di traduzione, non di refactoring.
- Servito da uvicorn come static build in produzione; `vite dev` con proxy in sviluppo.
- **L'applicazione è una PWA installabile e si apre in una finestra propria, non in una scheda del
  browser.** Il comando `autophotoedit` avvia il server su `http://127.0.0.1:8787` e apre una
  finestra dedicata senza barra degli indirizzi, schede o segnalibri (§21.2). Il browser resta il
  motore di rendering, ma non deve mai essere visibile *come* browser.
- **`vite-plugin-pwa`** genera `manifest.webmanifest` e il service worker. Il manifest dichiara
  `display: "standalone"`, `scope: "/"`, `start_url: "/"`, un `id` stabile, nome e descrizione in
  italiano, `theme_color`/`background_color` coerenti col tema della UI, e icone PNG 192 e 512 più
  una 512 `maskable`.
- **Il service worker fa cache del solo app shell** (HTML, JS, CSS, font, icone), in precache con
  revisione al deploy. Tutto ciò che sta sotto `/api` e `/ws`, e le immagini proxy, **non viene mai
  intercettato né messo in cache**: lo stato vive sul server e una risposta stantia mostrerebbe
  all'utente una realtà falsa. A server spento il service worker serve una pagina "autoPhotoEdit
  non è in esecuzione" al posto dell'errore di rete del browser.
- `http://127.0.0.1` è un *secure context*: service worker e installabilità funzionano senza HTTPS
  e senza certificati. **Non introdurre TLS locale.**

### Motivazione della scelta SQLite (decisione presa — non riaprirla)

Il volume tipico è 500–2000 foto per progetto, a volte di più. **SQLite in modalità WAL è la
scelta corretta e non comporta complessità significativa**: è un singolo file, zero
amministrazione, gestisce comodamente centinaia di migliaia di righe e dà gratuitamente il
*resume* dei job interrotti — che a questi volumi è un requisito, non un lusso. Non spezzare
l'elaborazione in sessioni manuali: la coda persistente su SQLite risolve il problema alla radice.

---

## 5. Architettura

```
autoPhotoEdit/
├── pyproject.toml
├── docs/SPEC.md                   # questo file
├── README.md
├── LICENSE                        # GPL-3.0 (§24)
├── NOTICE.md                      # dipendenze e loro licenze, verificato in CI
├── install.sh                     # installa launcher + .desktop (§18)
├── uninstall.sh
├── backend/
│   └── ape/
│       ├── __main__.py            # entrypoint CLI: `autophotoedit [--port] [--no-window]
│       │                           #                 [--window] [--browser PATH]`
│       ├── config.py              # settings (pydantic-settings), path XDG
│       ├── models_registry.py     # modelli ONNX: URL, SHA-256, licenza, feature (§17)
│       ├── diagnostics.py         # bundle diagnostico, logging rotante (§19)
│       ├── launcher.py            # apertura della finestra PWA, rilevamento browser (§21.2)
│       ├── single_instance.py     # lockfile, istanza singola, refocus finestra (§21.1)
│       ├── db/
│       │   ├── models.py          # SQLAlchemy ORM
│       │   ├── session.py
│       │   └── migrations/        # alembic
│       ├── raw/
│       │   ├── decode.py          # rawpy → float32 linear scene-referred
│       │   ├── metadata.py        # EXIF, obiettivo, camera, WB as-shot
│       │   └── proxy.py           # generazione proxy 2048px per la UI
│       ├── pipeline/
│       │   ├── params.py          # EditParams (pydantic) — IL contratto centrale
│       │   ├── ops/               # un modulo per operazione, tutte firma (img, params)->img
│       │   │   ├── white_balance.py
│       │   │   ├── exposure.py
│       │   │   ├── highlight_recovery.py
│       │   │   ├── tone.py        # sigmoid/filmic, black/white point, shadows/highlights
│       │   │   ├── tone_curve.py  # curva parametrica + spline libera
│       │   │   ├── color.py       # saturazione, vividezza, HSL a 8 canali, split toning
│       │   │   ├── local_contrast.py  # clarity via guided filter
│       │   │   ├── noise.py       # denoise luminanza/croma leggero
│       │   │   ├── sharpen.py
│       │   │   └── masks.py       # maschere parametriche + maschere manuali
│       │   ├── geometry.py        # lens correction, rotazione, crop
│       │   ├── colorspace.py      # working space, trasformazioni, output profile
│       │   └── render.py          # orchestrazione: EditParams + RAW → array finale
│       ├── culling/
│       │   ├── technical.py       # nitidezza, esposizione, mosso — su anteprima incorporata
│       │   ├── burst.py           # raggruppamento raffiche/quasi-duplicati
│       │   ├── faces.py           # occhi chiusi, espressione (opzionale, ONNX)
│       │   ├── aesthetic.py       # punteggio estetico (opzionale, ONNX)
│       │   └── select.py          # punteggio combinato e strategia di selezione
│       ├── merge/
│       │   ├── detect.py          # rilevamento gruppi candidati (HDR, pano, stack) (§25)
│       │   ├── align.py           # allineamento: omografia, ECC con scala, deghosting
│       │   ├── hdr.py             # fusione in lineare pesata sugli EV
│       │   ├── panorama.py        # stima omografie a scala ridotta + warp/blend a bande
│       │   ├── focus_stack.py     # piramidi laplaciane, mappa di nitidezza locale
│       │   └── virtual.py         # creazione della Photo derivata e del suo intermedio
│       ├── analysis/
│       │   ├── embed.py           # embedding CLIP ONNX della scena
│       │   ├── scene.py           # feature ingegnerizzate (istogrammi, EXIF, statistiche)
│       │   ├── straighten.py      # stima angolo orizzonte/verticali
│       │   ├── crop.py            # proposta crop compositivo
│       │   ├── cluster.py         # clustering e scelta rappresentanti
│       │   └── confidence.py      # punteggio di confidenza per foto
│       ├── style/
│       │   ├── learn.py           # inversione parametri da coppie RAW+edited
│       │   ├── profile.py         # StyleProfile: modello + metadati + persistenza
│       │   ├── predict.py         # scena → EditParams
│       │   ├── match.py           # scelta del profilo più adatto
│       │   ├── builtin.py         # Neutro automatico + 3 profili predefiniti (§22)
│       │   └── portable.py        # export/import .apestyle (§20)
│       ├── export/
│       │   ├── image.py           # JPEG/TIFF, ICC, metadata
│       │   ├── naming.py          # template nomi file, collisioni (§16)
│       │   ├── xmp_darktable.py
│       │   └── xmp_adobe.py
│       ├── jobs/
│       │   ├── queue.py           # coda persistente su SQLite
│       │   └── worker.py          # process pool
│       └── api/
│           ├── app.py
│           ├── routes_projects.py
│           ├── routes_photos.py
│           ├── routes_styles.py
│           ├── routes_review.py
│           └── ws.py
├── frontend/
│   ├── public/
│   │   └── icons/                 # 192, 512, 512-maskable, SVG sorgente (§18, §21.2)
│   ├── src/
│   │   ├── pages/  (Projects, Import, Culling, Styles, Review, Export,
│   │   │             Settings, Problems)
│   │   ├── components/
│   │   ├── pwa/
│   │   │   ├── manifest.ts        # manifest.webmanifest generato da vite-plugin-pwa (§4)
│   │   │   ├── sw.ts              # service worker: app shell, mai /api né /ws (§4)
│   │   │   ├── offline.html       # "autoPhotoEdit non è in esecuzione"
│   │   │   └── install.ts         # beforeinstallprompt, rilevamento display-mode (§21.2)
│   │   ├── i18n/it.ts             # TUTTE le stringhe UI, nessuna hardcoded
│   │   └── lib/api.ts             # client tipizzato generato da OpenAPI
│   └── ...
├── models/                        # modelli ONNX scaricati on-demand, con checksum (§17)
│   └── LICENSES.md                # licenza dichiarata di ogni modello
└── tests/
```

### Principio architetturale non negoziabile

**`EditParams` è l'unico contratto tra tutti i sottosistemi.** Lo style learning produce
`EditParams`; la UI mostra e modifica `EditParams`; il renderer consuma `EditParams`; gli
esportatori XMP traducono `EditParams`. Nessun modulo deve produrre pixel senza passare da un
`EditParams` serializzabile: è ciò che rende il risultato riproducibile, versionabile e
modificabile a mano.

---

## 6. Pipeline di sviluppo RAW

### 6.1 Spazio di lavoro

1. `rawpy.postprocess` con: `output_bps=16`, `gamma=(1,1)` (lineare), `no_auto_bright=True`,
   `use_camera_wb=False`, `user_wb=` moltiplicatori espliciti, `output_color=rawpy.ColorSpace.raw`
   oppure demosaicing con `half_size=False` e `demosaic_algorithm=AHD` (DCB come opzione qualità).
2. Applica la matrice camera→XYZ presa dai metadati LibRaw, poi XYZ→**Rec.2020 lineare**:
   il working space wide-gamut evita clipping dei colori saturi prima del tone mapping.
3. Tutto internamente in **float32, scene-referred, lineare**, normalizzato con il bianco a 1.0.
4. Output: conversione a sRGB (default) o Display-P3/AdobeRGB, con gamut mapping perceptual
   (compressione della croma, non clipping).

### 6.2 Ordine delle operazioni (vincolante)

```
[fusione multi-scatto, se la foto è derivata: N RAW → 1 buffer lineare]   (§25)
decode RAW (lineare)
 → correzione lente (distorsione, TCA, vignettatura)    [lensfunpy, opzionale]
 → riduzione rumore (scene-referred, prima del tone map)
 → bilanciamento del bianco (temp/tint → moltiplicatori)
 → recupero alte luci (ricostruzione canali clippati)
 → esposizione (guadagno lineare, in EV)
 → regolazioni locali su maschere parametriche         [in lineare]
 → tone mapping (sigmoid: black point, white point, contrasto, pivot)
 → ombre / luci / bianchi / neri (curva di shaping post-sigmoid)
 → curva tonale (parametrica + spline utente)
 → colore (vividezza, saturazione, HSL 8 canali, split toning)
 → contrasto locale / clarity (guided filter)
 → rotazione + crop
 → nitidezza (output-referred, dopo il resize finale)
 → conversione spazio colore di output + gamut mapping
 → encode
```

**Regole:**
- **La fusione sta a monte di tutto.** Una foto derivata (§25) sostituisce lo stadio "decode RAW"
  con il proprio intermedio lineare scene-referred: da lì in poi la pipeline è identica, bit per
  bit, a quella di uno scatto singolo. Nessuna operazione sotto conosce l'esistenza delle fusioni.
- Tutto ciò che è *fisico* (WB, esposizione, recupero luci, denoise) avviene in lineare.
- Tutto ciò che è *percettivo* (contrasto, ombre/luci, saturazione) avviene dopo il tone mapping.
- Il tone mapping è una **sigmoid** parametrica (stile filmic/AgX), non una gamma: parametri
  `black_point_ev`, `white_point_ev`, `contrast`, `pivot`, `toe`, `shoulder`.
- Le operazioni devono essere **risoluzione-indipendenti**: lo stesso `EditParams` applicato al
  proxy 2048px e al full-res deve dare lo stesso risultato visivo. Ogni operazione con un raggio
  in pixel (clarity, sharpen, guided filter) scala il raggio con la dimensione dell'immagine.
  *Questo è un requisito testabile: vedi §13.*

### 6.3 Maschere

**Default automatico: solo maschere parametriche.** Selezione continua (feathered, 0–1 float) su:
- range di luminanza (con transizioni morbide),
- range di tinta/saturazione,
- opzionalmente combinate con AND/OR/NOT.

**Maschere manuali (aggiunte dall'utente nella UI, mai automatiche):**
- gradiente lineare, gradiente radiale, pennello (raster, salvato come PNG 8-bit in cache),
- maschera da segmentazione soggetto (cielo / persone / pelle) — eseguita **on-demand** quando
  l'utente la richiede, tramite modello ONNX leggero su CPU. Non deve mai girare nel batch
  automatico: è costosa e non richiesta dal flusso di default.

Ogni maschera è un elemento di `EditParams.masks[]`, con la propria sotto-lista di regolazioni.

### 6.4 Geometria

- **Raddrizzamento automatico (applicato):** rileva le linee dominanti con `cv2.createLineSegmentDetector`
  (o `FastLineDetector`), separa quelle quasi-orizzontali da quelle quasi-verticali, stima
  l'angolo con una mediana pesata sulla lunghezza dei segmenti. Applica la rotazione solo se
  |angolo| è tra 0.15° e 8° e se la coerenza dei segmenti supera una soglia; altrimenti non
  ruotare e abbassa la confidenza geometrica. Il crop implicito della rotazione è il minimo
  necessario a eliminare i bordi vuoti.
- **Correzione lente (applicata):** identifica l'obiettivo dai metadati EXIF (`LensModel`,
  focale, apertura) e cerca il profilo in **lensfun**. Applica distorsione + TCA + vignettatura.
  Se il profilo non esiste, registra l'obiettivo come "senza profilo", non applicare nulla e
  segnalalo nella UI (l'utente può associare manualmente un profilo, scelta memorizzata per
  quell'obiettivo in tutti i progetti futuri).
- **Crop compositivo (solo proposta):** calcola una mappa di salienza (`cv2.saliency` fine-grained
  o un modello ONNX leggero), valuta un insieme di crop candidati **solo nel rapporto originale
  del fotogramma** (3:2 o 2:3 per uno scatto singolo; per un panorama il rapporto che risulta
  dall'unione) con un punteggio che combina: copertura della salienza, allineamento
  dei picchi di salienza ai punti di forza (regola dei terzi), penalità per taglio di volti/soggetti,
  penalità per perdita di area. **Non applicare mai automaticamente.** Mostra il rettangolo
  proposto in overlay con un pulsante "Applica crop proposto". Se l'utente rifiuta due proposte
  consecutive nello stesso progetto, smetti di proporre in quel progetto. Unica eccezione al
  rapporto originale fra le proposte: il rettangolo senza bordi vuoti di un panorama (§25.5.5).
- **Crop manuale:** "Ritaglia" nel pannello Geometria apre un riquadro da trascinare sulla foto,
  bloccato sul rapporto originale. Altri rapporti (libero, 1:1, 4:5, 16:9, con l'orientamento
  invertibile) sono una scelta esplicita dell'utente, mai un default.

---

## 7. Cernita (selezione degli scatti) — fase opzionale

### 7.1 Scelta all'import

Quando l'utente carica una cartella, la schermata di import presenta **due percorsi, con scelta
esplicita e nessun default nascosto**:

- **Vai direttamente all'editing** — tutte le foto importate entrano in elaborazione.
- **Fai prima una cernita** — il programma analizza e propone una selezione; l'editing parte solo
  dopo la conferma dell'utente.

La scelta è memorizzata in `Project.culling_enabled` ed è **reversibile**: si può aprire la
cernita anche dopo, o saltarla e tornarci. Non è un vicolo cieco.

### 7.2 Analisi di cernita — leggerezza prima di tutto

Questa fase deve essere veloce su 2000+ file su una macchina senza GPU. Il punto chiave:

> **La cernita non decodifica i RAW.** Usa l'**anteprima JPEG incorporata** nell'ARW
> (`rawpy.extract_thumb()`), che le Sony includono a piena risoluzione e che è già sviluppata
> dalla fotocamera. Estrarla costa ~30–60 ms contro ~1 s di una decodifica completa.

Il demosaicing completo avviene solo dopo la conferma della selezione, e **solo sulle foto
selezionate**: è questo che rende il programma utilizzabile su questo hardware.

### 7.3 Criteri, tutti come toggle indipendenti nella UI

**Attivi di default (leggeri, nessun modello ML):**

1. **Nitidezza / messa a fuoco** — varianza del Laplaciano ed energia dei gradienti misurate
   *sulla regione più nitida* dell'immagine, non sull'intero fotogramma (una foto con sfondo
   sfocato e soggetto a fuoco è nitida). Normalizzata per ISO e contenuto: il valore assoluto non
   è confrontabile tra scene diverse, quindi il punteggio è **relativo** all'interno del gruppo di
   scatti simili.
2. **Mosso da movimento** — distingue il micromosso dal fuori fuoco tramite l'anisotropia dello
   spettro dei gradienti (il mosso ha una direzione preferenziale, lo sfocato no).
3. **Esposizione** — percentuale di clipping in alto e in basso ed EV di deviazione dall'ottimo,
   tenendo conto che un RAW recupera ~2 EV in alto e ~4 in basso: penalizza solo ciò che è
   **irrecuperabile**, non ciò che è semplicemente scuro.
4. **Raffiche e quasi-duplicati** — raggruppa per prossimità temporale EXIF (< 2 s) combinata con
   similarità visiva (hash percettivo + istogrammi sulle miniature). Dentro ogni gruppo ordina per
   punteggio e propone **una sola** foto, tenendo le altre immediatamente accessibili.
   **Eccezione vincolante:** un bracketing di esposizione, una sequenza di focus stacking o una
   panoramica *sembrano* una raffica e non lo sono. Prima del raggruppamento raffiche gira il
   rilevamento delle fusioni (§25.2); gli scatti appartenenti a un gruppo candidato sono esclusi
   dal raggruppamento raffiche, e i frame scuri o chiari di un bracketing **non** vengono mai
   penalizzati dal criterio di esposizione — sono sottoesposti per progetto, non per errore.
   Scartare i frame di un bracketing è il modo più rapido per rendere il programma inutile a chi
   fotografa in bracketing.

**Attivabili su richiesta (toggle, modelli ONNX su CPU, costo dichiarato in UI):**

5. **Volti** — rilevamento volti (YuNet, ~5 ms per immagine su questa CPU), poi per ogni volto
   apertura degli occhi (EAR da landmark o piccolo classificatore ONNX) ed espressione.
   Penalizza gli occhi chiusi **solo sul soggetto principale** — il volto più grande o più
   centrale — perché in una foto di gruppo l'occhio chiuso di una comparsa in secondo piano non è
   uno scarto. Costo stimato: +0.15 s per foto.
6. **Estetica** — punteggio di gradevolezza/composizione con un modello NIMA-like convertito in
   ONNX (backbone MobileNet, per restare leggero). Costo stimato: +0.25 s per foto. Da presentare
   in UI come *"ordinamento soggettivo, usalo come suggerimento"*.
   **Avviso di licenza obbligatorio:** i pesi NIMA pubblici derivano dal dataset **AVA**, il cui
   uso è dichiarato *research-only*. Il modello non viene ridistribuito con il programma (si
   scarica on-demand, §17) e il toggle mostra una nota: *"modello addestrato su un dataset di
   sola ricerca; valuta tu l'idoneità per un uso commerciale"*. Il toggle resta **off** di default.

Attivare o disattivare un toggle ricalcola **solo** il punteggio mancante, senza rianalizzare ciò
che è già stato calcolato. I punteggi parziali sono persistiti sulla `Photo`.

### 7.4 Punteggio e strategia di selezione

Il punteggio finale è una combinazione pesata dei criteri attivi, con **pesi visibili e
regolabili**. Non nascondere la formula: l'utente deve poter capire perché una foto è finita tra
gli scarti.

Due modalità, entrambe disponibili in UI:

- **Conservativa (preselezionata)** — scarta solo ciò che è tecnicamente compromesso oltre il
  recupero (fuori fuoco netto, mosso grave, clipping irrecuperabile) e i doppioni di raffica.
  Tutto ciò che è valido resta.
- **Target numerico** — l'utente imposta una percentuale ("tieni il 30%") o un numero assoluto
  ("tieni 200 foto"); il programma ordina per punteggio e taglia, garantendo però di prendere una
  foto da **ogni** gruppo di raffica prima di prenderne una seconda da qualunque gruppo
  (copertura prima di densità: meglio 200 momenti diversi che 200 varianti dello stesso).

Uno **slider di aggressività** (0–100) modula entrambe le modalità spostando le soglie.
Il cambio di modalità o di slider aggiorna la selezione **istantaneamente, senza rianalisi**:
l'analisi produce punteggi, la selezione è solo una soglia applicata a punteggi già calcolati.
Questa separazione è vincolante.

### 7.5 Destino degli scarti

**Nessun file viene spostato, rinominato o toccato** (vedi §2). Le foto scartate restano nel
progetto con `Photo.culled = true` e il motivo dello scarto, nascoste dietro un filtro
"Mostra scarti" e recuperabili con un click in qualunque momento, anche a elaborazione avviata.

### 7.6 Schermata di cernita

- Griglia densa di miniature con badge di punteggio e icona del motivo di scarto.
- **Gruppi di raffica collassati**: una card mostra la scelta con un contatore "1 di 7";
  espandendola si confrontano gli scatti del gruppo affiancati e si cambia il vincitore.
- Confronto a due o quattro riquadri con **zoom sincronizzato 1:1** sui volti o sul punto di
  fuoco — è il modo in cui si decide davvero tra due scatti quasi identici.
- Filtri: solo selezionate / solo scarti / tutte; ordinamento per punteggio, orario o cartella.
- Scorciatoie: `X` scarta, `P` promuovi, `←/→` naviga, `↑/↓` cambia vincitore nel gruppo,
  `Space` confronto affiancato, `Ctrl+Z` annulla.
- Barra inferiore persistente: "**342 selezionate su 1180** — Procedi con l'editing", aggiornata
  in tempo reale. Quel pulsante è l'unico modo per passare all'elaborazione: **il programma non
  inizia mai a elaborare senza consenso esplicito.**

### 7.7 Prestazioni richieste

Su 2000 file, con i soli criteri di default, su 14 worker: **analisi di cernita ≤ 3 minuti**.
Con volti ed estetica attivi: ≤ 9 minuti. Se questi numeri non sono raggiungibili, riportali
misurati anziché degradare la qualità in silenzio.

---

## 8. Apprendimento dello stile

### 8.1 Input

L'utente fornisce **coppie**: file RAW originale + la sua versione già post-prodotta
(JPEG/TIFF). L'accoppiamento avviene per nome file base; se fallisce, la UI mostra una
schermata di accoppiamento manuale. Minimo utile: **8 coppie**; sotto questa soglia avvisa che
il profilo sarà poco affidabile. Consigliato: 30–100 coppie che coprano scene diverse.

**Fallback senza RAW** (l'utente ha solo i JPEG editati): estrai comunque le statistiche target
(istogrammi per canale, temperatura percepita, contrasto, saturazione media, distribuzione in Lab)
e usale come obiettivo per l'ottimizzazione applicata ai RAW del progetto corrente. Qualità
inferiore — dichiaralo nella UI.

### 8.2 Inversione dei parametri (il cuore del sistema)

Per ogni coppia (RAW, riferimento editato):

1. Sviluppa il RAW con `EditParams` neutri e ridimensiona a lato lungo 512 px.
2. Ridimensiona il riferimento alla stessa dimensione; allinea geometricamente (il riferimento
   può essere croppato/ruotato: stima una omografia con feature matching ECC/ORB e, se il crop
   è significativo, ritaglia il RAW sviluppato alla stessa inquadratura prima di confrontare).
3. **Ottimizza il vettore di `EditParams` globali** per minimizzare una loss composita:
   - ΔE2000 medio in CIELab (peso 1.0),
   - distanza tra istogrammi di luminanza (Wasserstein 1-D, peso 0.5),
   - distanza tra istogrammi di croma a/b (peso 0.3),
   - differenza di contrasto locale (deviazione standard del gradiente, peso 0.2),
   - regolarizzazione L2 verso i parametri neutri (peso 0.05) per evitare soluzioni estreme.
4. Ottimizzatore: **CMA-ES** (`cma`) o `scipy.optimize.differential_evolution` per una prima
   fase globale, poi `Nelder-Mead` per il raffinamento. Vincola ogni parametro al suo range
   fisico. Budget: ≤ 400 valutazioni per coppia a 512 px (≈ 10–25 s per coppia su questa CPU).
5. Salva il vettore risultante insieme alla loss finale. Le coppie con loss residua alta sono
   **escluse** dal training e segnalate come "non riproducibili dalla pipeline".

Il vettore ottimizzato comprende: WB (temp, tint), esposizione, recupero luci, black/white point,
contrasto, pivot, toe, shoulder, ombre, luci, bianchi, neri, vividezza, saturazione, HSL per gli
8 canali (hue/sat/lum), clarity, split toning. **Non** comprende geometria e maschere.

### 8.3 Dal vettore al modello predittivo

Un profilo di stile **non è un preset fisso**: è una funzione scena → parametri.

1. Per ogni coppia calcola un **vettore di feature della scena**:
   - embedding CLIP ViT-B/32 ONNX (512-d) del RAW sviluppato neutro → ridotto a 32-d con PCA
     addestrata sul dataset del profilo;
   - feature ingegnerizzate: EV di scena, ISO, apertura, focale, media/mediana/percentili 1-5-50-95-99
     della luminanza, saturazione media, temperatura correlata stimata, percentuale di pixel clippati,
     entropia dell'istogramma, dominanza cromatica (istogramma di tinta a 8 bin), presenza di volti
     (conteggio + area), indicatore interno/esterno.
2. Addestra una regressione **k-NN pesata (k=5, kernel gaussiano sulla distanza)** sui vettori
   scena→parametri, con un **fallback a ridge regression** e, se il dataset supera 60 coppie,
   un gradient boosting leggero. Il k-NN è preferito perché con pochi campioni generalizza in modo
   prevedibile e permette di mostrare all'utente *quali* sample hanno determinato l'edit
   ("questa foto è stata editata come i tuoi sample #12 e #31") — requisito di trasparenza della UI.
3. Il profilo salva: modello, PCA, statistiche di normalizzazione, i vettori delle coppie, i
   riferimenti ai file sample, metadati (nome, data, note, camera/obiettivi coperti).

### 8.4 Selezione automatica del profilo e coerenza stilistica

- I profili sono salvati nel catalogo globale e **riutilizzabili in progetti futuri**.
- All'apertura di un progetto, calcola l'embedding medio delle foto e confrontalo con l'embedding
  medio di ogni profilo: proponi il profilo più vicino, mostrando il punteggio di affinità e
  lasciando all'utente l'override (requisito esplicito: selezione automatica **con** override).
- **Coerenza stilistica intra-progetto:** dopo aver predetto i parametri per tutte le foto,
  applica una regolarizzazione:
  - all'interno di un cluster di scena, sposta i parametri di ogni foto verso la mediana del
    cluster di un fattore λ (default 0.35, regolabile con uno slider "Coerenza" in UI);
  - per scatti consecutivi (< 60 s di distanza EXIF, stessa focale ± 10%, stesso ambiente),
    applica uno smoothing più forte (λ = 0.6) su WB ed esposizione: è ciò che impedisce che due
    foto della stessa scena abbiano temperature diverse.
  - Lo slider a 0 disattiva del tutto la regolarizzazione.

---

## 9. Confidenza e flusso di revisione

Requisito: **cluster + outlier combinati**.

### 9.1 Punteggio di confidenza (0–1) per foto

Combina, con pesi configurabili:
- **distanza dal vicino più prossimo** nello spazio scena del profilo (lontano = bassa confidenza);
- **dispersione dei k vicini** (i sample vicini suggeriscono edit diversi tra loro = ambiguità);
- **ambiguità del bilanciamento del bianco** (illuminanti misti: varianza alta della temperatura
  stimata su patch diverse dell'immagine);
- **clipping severo** (> 2% di pixel bruciati o > 5% in nero dopo l'edit);
- **incertezza geometrica** (orizzonte non rilevato con sicurezza, linee contraddittorie);
- **obiettivo senza profilo lensfun**;
- **esposizione fuori dal range dei sample** (estrapolazione anziché interpolazione).

Soglia di escalation regolabile in UI (default 0.55).

### 9.2 Flusso in UI

1. **Clustering:** raggruppa le foto del progetto con clustering agglomerativo sull'embedding
   ridotto + prossimità temporale. Numero di cluster scelto con soglia di distanza, non fisso.
2. **Revisione per cluster:** per ogni cluster mostra il rappresentante (il medoide) con
   prima/dopo, i parametri, e i sample che l'hanno ispirato. L'utente approva, corregge
   (modificando i parametri con gli slider) o rifiuta. **Le correzioni si propagano a tutto il
   cluster come delta**, non come valori assoluti — così le variazioni interne al cluster
   restano preservate.
3. **Coda outlier:** le foto sotto la soglia di confidenza, indipendentemente dal cluster, entrano
   in una coda di revisione individuale, ordinate per confidenza crescente, ciascuna con il motivo
   esplicito ("bilanciamento del bianco ambiguo: luce mista", "orizzonte non rilevato").
   Per ognuna la UI propone **2–3 varianti alternative** tra cui scegliere, oltre alla regolazione
   manuale.
4. **Apprendimento dal feedback:** ogni correzione dell'utente viene salvata come nuova coppia
   (RAW + parametri accettati) e, su conferma esplicita dell'utente a fine progetto, incorporata
   nel profilo di stile. Questo chiude il ciclo: il programma migliora progetto dopo progetto.

---

## 10. UI (web locale)

Design: **moderno, scuro di default, denso ma respirabile**; il contenuto sono le foto, la
cromia dell'interfaccia deve essere neutra (grigi) per non falsare la percezione del colore.
Nessun colore saturo vicino alle anteprime. Tipografia di sistema (Inter). Tutte le anteprime
su sfondo grigio neutro ~#1a1a1a con un bordo sottile.

### Schermate

1. **Progetti** — elenco, creazione, stato di avanzamento, riapertura di una sessione interrotta.
2. **Import** — selezione cartella RAW, scansione, **scelta esplicita tra "vai subito
   all'editing" e "fai prima una cernita"** (§7.1), avanzamento via WebSocket, riepilogo
   (numero foto, camere, obiettivi, obiettivi senza profilo, spazio richiesto dalla cache).
3. **Cernita** (se attivata) — vedi §7.6.
3-bis. **Fusioni** — gruppi HDR / focus stack / panorama proposti, con anteprima rapida,
   Accetta / Rifiuta / Modifica gruppo (§25.6).
4. **Stili** — libreria dei profili salvati; creazione di un profilo da coppie RAW+editate con
   accoppiamento automatico e correzione manuale; visualizzazione dell'avanzamento del training
   e delle coppie scartate; affinità del profilo con il progetto corrente.
5. **Revisione** — la schermata principale:
   - griglia dei cluster a sinistra, rappresentante grande al centro, pannello parametri a destra;
   - toggle prima/dopo (tasto `\`), zoom 1:1, confronto affiancato;
   - overlay del crop proposto con Applica/Scarta;
   - badge di confidenza e motivo dell'escalation;
   - pannello "Sample di riferimento" che mostra le foto dei sample che hanno guidato l'edit;
   - scorciatoie da tastiera: `←/→` naviga, `A` approva, `R` rifiuta, `Space` prima/dopo,
     `1..3` sceglie una variante, `Ctrl+Z` annulla.
   - editor maschere manuali (gradiente/radiale/pennello) attivabile su richiesta.
6. **Export** — scelta formato (JPEG qualità N / TIFF 16-bit), spazio colore, ridimensionamento,
   nitidezza di output, cartella di destinazione, **template del nome file con anteprima live**
   (§16.1), toggle "Rimuovi dati di posizione" (§16.3), **e** generazione dei sidecar XMP
   (darktable e/o Adobe). Coda con avanzamento e possibilità di mettere in pausa/riprendere.
7. **Impostazioni** — autore e copyright per gli export, quota e uso della cache con "Svuota
   cache" (§20), gestione dei modelli ONNX scaricati (§17), comportamento alla chiusura della
   finestra (§21.3), modalità di apertura in uso (finestra dedicata o scheda di ripiego) con il
   pulsante "Installa come app" quando il browser lo consente (§21.2), backup/ripristino del
   catalogo, livello di log.
8. **Problemi** — elenco delle foto in stato `failed` con motivo in italiano, dettagli tecnici
   a scomparsa, "Riprova" singolo o massivo, ed "Esporta diagnostica" (§19).

### Requisiti della finestra applicativa

La UI gira in una finestra senza cromatura del browser (§21.2): non c'è il pulsante Indietro, non
c'è la barra degli indirizzi, non c'è un ricaricamento a portata di mano.

- **La navigazione vive tutta dentro l'app:** ogni schermata ha un percorso di ritorno visibile, e
  `Alt+←/→` sono gestite dal router, non lasciate al browser.
- **Nessuna funzione raggiungibile solo digitando un URL.** Gli URL restano leggibili e stabili
  (servono in debug), ma non sono mai l'unica strada.
- **I link esterni** (licenze, documentazione) si aprono nel browser predefinito e non devono mai
  sostituire il contenuto della finestra dell'applicazione.
- **Un errore fatale del frontend non lascia una finestra bianca senza uscite:** un error boundary
  mostra il messaggio, un pulsante "Ricarica" e "Esporta diagnostica" (§19).
- La finestra è utilizzabile da **1280×800** in su; sotto quella soglia le schermate si
  ridispongono, non si troncano.

### Requisiti di reattività

- La UI non mostra mai il RAW full-res: usa i **proxy 2048 px** in cache.
- L'anteprima di una modifica dei parametri deve aggiornarsi in **< 150 ms**: renderizza a 1024 px
  durante il trascinamento dello slider, a 2048 px quando l'utente rilascia.
- Il rendering di anteprima gira in un worker dedicato con cache dei buffer intermedi: se cambia
  solo un parametro *post* tone mapping, non ridecodificare il RAW né ricalcolare gli stadi
  precedenti. **Implementa la pipeline come una catena di stadi con cache invalidata per stadio.**

---

## 11. Modello dati (SQLite)

```
Project(id, name, source_dir, output_dir, style_profile_id, coherence_lambda,
        confidence_threshold, culling_enabled, culling_mode, culling_target,
        culling_aggressiveness, culling_weights JSON, culling_criteria JSON,
        export_template, export_on_conflict, export_strip_gps, source_missing BOOL,
        merge_detection_enabled BOOL,
        created_at, updated_at, status)
      # culling_mode: conservative | target_percent | target_count
      # culling_criteria: toggle attivi (technical, burst, faces, aesthetic)
      # export_on_conflict: ask | rename | overwrite | skip   (default: ask)
      # source_missing: la cartella sorgente non è raggiungibile → offri il re-mapping (§15)

Photo(id, project_id, path, filename, hash, camera, lens, iso, aperture,
      shutter, focal_length, shot_at, width, height, orientation,
      proxy_path, embedding BLOB, scene_features BLOB, cluster_id,
      confidence REAL, escalation_reasons JSON, status, created_at,
      sharpness REAL, motion_blur REAL, exposure_score REAL, face_score REAL,
      aesthetic_score REAL, culling_score REAL, burst_group_id, burst_rank,
      culled BOOL, cull_reasons JSON, cull_decided_by,
      sidecar_jpeg_path, duplicate_paths JSON, missing BOOL,
      kind, merge_group_id, intermediate_path)
      # kind: raw | merged   — una foto 'merged' non ha un file sorgente proprio (§25):
      #       path è NULL, intermediate_path punta all'EXR/TIFF float32 lineare in cache,
      #       rigenerabile dai RAW dei membri, che restano intatti
      # sidecar_jpeg_path: JPEG affiancato allo scatto, NON importato come foto a sé (§15)
      # duplicate_paths: altri path con hash identico, importati una sola volta
      # missing: il file non è più nella sorgente — la riga NON viene cancellata
      # status: imported | culled | analyzed | predicted | needs_review | approved | exported | failed
      # cull_decided_by: auto | user — una decisione dell'utente non viene MAI sovrascritta
      #                                da un ricalcolo automatico

EditVersion(id, photo_id, parent_version_id, params JSON, params_version,
            source, created_at, is_current)
      # source: predicted | user_edited | cluster_applied | reverted | snapshot_restored
      # parent_version_id: la storia è un albero, il ripristino non cancella nulla (§23)

CropProposal(id, photo_id, rect JSON, aspect, score, decision)
      # decision: pending | applied | rejected

MergeGroup(id, project_id, kind, decision, confidence, detect_reasons JSON,
           params JSON, result_photo_id, error, created_at)
      # kind: hdr | panorama | focus_stack
      # decision: proposed | accepted | rejected | failed
      #   'rejected' è permanente per quel gruppo: non riproporlo a ogni riapertura
      # params: opzioni di fusione scelte (deghosting, proiezione, ordine, allineamento)
MergeMember(group_id, photo_id, position, ev_offset, role)
      # role: member | reference   — il frame di riferimento per l'allineamento
      #       e per l'ereditarietà dei metadati della foto derivata

StyleProfile(id, name, notes, model BLOB, pca BLOB, norm_stats JSON,
             embedding_centroid BLOB, n_pairs, builtin BOOL, builtin_rules JSON,
             created_at, updated_at)
      # builtin: profilo fornito con il programma (§22). model/pca sono NULL:
      #          la predizione usa builtin_rules. Non cancellabile, non sovrascrivibile:
      #          modificarlo crea una copia utente con builtin = false.

StyleSample(id, profile_id, raw_path, reference_path, params JSON,
            scene_features BLOB, embedding BLOB, residual_loss REAL, excluded BOOL)

LensProfileOverride(lens_model, lensfun_maker, lensfun_model)   # globale, cross-progetto

Setting(key, value JSON)          # preferenze globali, cross-progetto:
      # artist, copyright, cache_max_gb, on_window_close (ask|continue|pause|stop),
      # log_level, aesthetic_enabled_ack (conferma dell'avviso di licenza §17)

ProjectSnapshot(id, project_id, name, kind, created_at)
      # kind: manual | auto   — gli auto sono creati ai passaggi di fase, max 10 conservati
SnapshotEntry(snapshot_id, photo_id, edit_version_id, culled, status)
      # uno snapshot registra quale versione era corrente, non i pixel: pochi KB (§23)

Job(id, project_id, kind, payload JSON, state, progress, error, attempts,
    started_at, finished_at)
      # kind: proxy | analyze | predict | render_preview | export
      # state: queued | running | done | failed | cancelled
```

- Tutti i parametri di edit sono JSON validato con Pydantic e **versionato** (`params_version`):
  quando il formato cambia, scrivi una migrazione, non rompere i progetti esistenti.
- WAL abilitato, `synchronous=NORMAL`.
- I `Job` in stato `running` all'avvio vengono rimessi in `queued` (resume dopo un crash).

---

## 12. Concorrenza e prestazioni

- Process pool con `N = max(2, os.cpu_count() - 2)` = 14 worker su questa macchina.
- Ogni worker limita i thread di BLAS/OpenCV a 1 (`cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1`)
  per evitare oversubscription.
- La coda è persistita su SQLite; i worker prendono i job con una transazione atomica.
- Il progresso viene pubblicato su WebSocket in batch (max 4 messaggi/s), non per singola foto.

### Target di prestazione (da verificare con benchmark, in `tests/bench/`)

| Operazione | Target (24 MP, questa macchina) |
|---|---|
| Estrazione anteprima incorporata (cernita) | ≤ 60 ms per foto (1 core) |
| Punteggio tecnico di cernita (nitidezza, mosso, esposizione) | ≤ 80 ms per foto (1 core) |
| Volti (opzionale) | ≤ 150 ms per foto (1 core) |
| Estetica (opzionale) | ≤ 250 ms per foto (1 core) |
| **Cernita di 2000 foto, criteri di default, 14 worker** | **≤ 3 minuti** |
| Decodifica RAW + proxy 2048 px | ≤ 1.2 s per foto (1 core) |
| Analisi (embedding + feature + geometria) | ≤ 0.6 s per foto |
| Predizione parametri | ≤ 5 ms per foto |
| Render anteprima 1024 px da cache di stadio | ≤ 120 ms |
| Export full-res JPEG | ≤ 3.5 s per foto (1 core) |
| Rilevamento dei gruppi di fusione (su miniature, §25.2) | ≤ 40 ms per foto (1 core) |
| Fusione HDR, 3 frame da 24 MP, con allineamento | ≤ 25 s (2 core) |
| Focus stack, 8 frame da 24 MP | ≤ 60 s (2 core) |
| Panorama, 6 frame da 24 MP, output ~80 MP | ≤ 180 s (2 core), picco RAM ≤ 3 GB |
| **Batch 1000 foto: import + analisi** | **≤ 4 minuti** con 14 worker |
| **Batch 1000 foto: export** | **≤ 5 minuti** con 14 worker |

Se un target non è raggiungibile, dillo esplicitamente con i numeri misurati anziché
abbassare silenziosamente la qualità.

---

## 13. Test richiesti

1. **Invarianza di risoluzione:** per 10 immagini e 20 `EditParams` casuali, il render a 1024 px
   e il render full-res ridimensionato a 1024 px devono avere ΔE2000 medio < 1.5.
2. **Round-trip dei parametri:** `EditParams → JSON → EditParams` identico; il JSON versionato di
   una versione precedente si carica correttamente.
3. **Convergenza dell'inversione:** sviluppa un RAW con parametri noti, salva il JPEG, esegui
   l'inversione: i parametri recuperati devono avere ΔE2000 residuo < 2.0 sul render.
4. **Determinismo:** la stessa foto con gli stessi parametri produce byte identici.
5. **Neutralità:** un RAW di un color checker sviluppato con parametri neutri e WB corretto deve
   dare le patch grigie con a*,b* entro ±2.
6. **Resume dei job:** uccidi il processo a metà di un batch, riavvia, il batch riprende senza
   duplicare né perdere foto.
7. **Sidecar:** l'XMP darktable generato deve essere apribile in darktable senza errori; l'XMP
   Adobe deve essere letto da Lightroom/Camera Raw (test manuale documentato, non automatizzabile).
8. **Non distruttività (§2):** hash SHA-256 e `mtime` di tutti i file sorgente invariati dopo un
   ciclo completo import → cernita → analisi → edit → export. Test obbligatorio, non skippabile.
   In più: `assert_outside_source` deve sollevare su un percorso dentro la sorgente, anche
   raggiunto attraverso un symlink.
9. **Cernita:** su un set con scarti noti (10 foto volutamente mosse / fuori fuoco / bruciate tra
   100), la modalità conservativa identifica almeno 8 dei 10 scarti e non ne scarta più di 3
   valide; le raffiche note vengono raggruppate correttamente.
10. **Idempotenza delle decisioni utente:** una foto promossa o scartata manualmente mantiene la
    decisione dopo un ricalcolo dei punteggi o il cambio di un toggle.
11. **Robustezza:** RAW corrotto, EXIF mancante, obiettivo sconosciuto, immagine completamente
   bruciata o completamente nera: nessun crash del worker, la foto entra in stato `failed` con
   messaggio leggibile.
12. **Import idempotente (§15):** reimportare la stessa cartella non crea duplicati e preserva
    edit, decisioni di cernita, cluster e approvazioni; aggiungere un file lo importa da solo;
    rimuovere un file lo marca `missing` senza cancellare la riga né toccare il disco. Due file
    con contenuto identico producono **una** `Photo` con due `duplicate_paths`. Un JPEG con lo
    stesso basename di un ARW non diventa una `Photo`.
13. **Naming e collisioni in export (§16):** un template che usa tutti i token produce nomi validi
    su filesystem; con `on_conflict=rename` nessun file preesistente nella cartella di
    destinazione cambia hash; con `skip` il file viene lasciato intatto e il job riportato come
    saltato, non fallito.
14. **Metadati di export (§16.3):** l'export contiene EXIF di scatto, `Software` valorizzato e
    `Orientation=1`; i campi Artist/Copyright riflettono le preferenze; il GPS è presente o
    assente secondo il toggle, e con il toggle attivo **nessun** tag GPS sopravvive.
15. **Round-trip del profilo (§20):** esporta un profilo in `.apestyle`, importalo in un catalogo
    vuoto, predici su 10 foto: i parametri devono essere identici a quelli del profilo originale.
    L'import deve riuscire anche con i RAW dei sample assenti dal disco.
16. **Storico e snapshot (§23):** 20 modifiche consecutive, ripristino alla versione 5, snapshot,
    altre modifiche, rollback allo snapshot: nessuna versione perduta, `is_current` sempre unica
    per foto, e il rollback è a sua volta annullabile.
17. **Fusioni multi-scatto (§25):**
    - un bracketing di 3 scatti EXIF-coerenti viene rilevato come gruppo HDR e **non** come
      raffica, e nessuno dei suoi frame viene scartato dalla cernita per esposizione;
    - la fusione HDR di 3 esposizioni sintetiche generate dallo stesso RAW ricostruisce la scena
      originale con ΔE2000 medio < 2.0;
    - un focus stack sintetico (stessa immagine sfocata a zone complementari) ricostruisce
      l'originale nitido con SSIM > 0.95;
    - una panoramica che non si cuce (sovrapposizione insufficiente) fallisce con un messaggio
      leggibile, lascia il gruppo in stato `failed` e non crea una foto derivata rotta;
    - un gruppo `rejected` non viene riproposto alla riapertura del progetto;
    - **§2 vale anche qui:** dopo rilevamento, fusione ed export, gli hash dei RAW membri sono
      invariati; cancellando l'intermedio in cache, la foto derivata si rigenera identica
      (test di determinismo) senza toccare gli originali.
18. **Integrità dei modelli (§17):** un file ONNX con SHA-256 non corrispondente viene scartato,
    la feature resta disabilitata con un messaggio, nessun crash e nessun modello corrotto in
    cache. Con la rete assente, le feature opzionali si disattivano con grazia e quelle di
    default continuano a funzionare.
19. **Finestra applicativa e PWA (§21):** con un browser Chromium presente, `autophotoedit` avvia
    esattamente un processo figlio in modalità `--app` con il `WM_CLASS` atteso; rilanciare il
    comando non apre una seconda finestra ed esce con codice 0; con il rilevamento del browser
    forzato a vuoto il launcher ricade sulla scheda **e lo dichiara nella UI**; `--no-window` non
    avvia alcun processo figlio. Il `manifest.webmanifest` generato è valido e soddisfa i criteri
    di installabilità (icone 192 e 512, `display: standalone`, `start_url` nello `scope`), e il
    service worker **non** intercetta né mette in cache `/api` e `/ws`: due letture consecutive
    dello stesso endpoint, con lo stato cambiato sul server tra l'una e l'altra, restituiscono
    valori diversi.

---

## 14. Fasi di sviluppo

Consegna incrementale. **Al termine di ogni fase il programma deve essere eseguibile e utile**,
non un mezzo scheletro. Non passare alla fase successiva senza i criteri di accettazione.

**Fase 1 — Fondamenta della pipeline (nessuna UI)**
Decodifica RAW, spazio di lavoro, `EditParams`, tutte le operazioni di §6.2 tranne maschere e
geometria, export JPEG/TIFF, CLI `ape render <raw> --params p.json -o out.jpg`.
*Accettazione:* test 1, 2, 4, 5 passano; un ARW dell'A7III si sviluppa con qualità visivamente
confrontabile con darktable a parametri equivalenti.

**Fase 2 — Catalogo, coda, proxy, API**
SQLite + modelli, coda persistente, worker pool, generazione proxy, FastAPI con gli endpoint di
progetto e foto, WebSocket di progresso.
*Accettazione:* test 6 e 12 passano; importa 1000 foto rispettando il target di §12; il
re-import della stessa cartella non duplica nulla.

**Fase 3 — Frontend base e finestra dell'applicazione**
Vite/React/Tailwind, schermate Progetti, Import, e un visualizzatore con pannello parametri
funzionante sul proxy. Manifest PWA, service worker dell'app shell e `launcher.py` con la finestra
dedicata e la sua catena di fallback (§21.2).
*Accettazione:* modifica di uno slider aggiornata in < 150 ms; l'anteprima corrisponde all'export;
`autophotoedit` apre una finestra **senza barra degli indirizzi né schede**, e un secondo avvio
riporta in primo piano quella esistente invece di aprirne un'altra.

**Fase 4 — Cernita**
Estrazione anteprime incorporate, punteggi tecnici, raggruppamento raffiche, strategie di
selezione, schermata di cernita con confronto e gruppi. Volti ed estetica dietro toggle, se i
modelli ONNX sono disponibili; altrimenti il toggle resta disabilitato con una spiegazione.
*Accettazione:* test 9 e 10 passano; 2000 foto analizzate entro il target di §7.7; cambiare
modalità o slider aggiorna la selezione senza rianalisi percepibile (< 100 ms).

**Fase 5 — Analisi e geometria**
Embedding ONNX, feature di scena, clustering, raddrizzamento automatico, correzione lente
lensfun, proposta di crop.
*Accettazione:* su 50 foto di test il raddrizzamento non peggiora mai un orizzonte già corretto;
gli obiettivi Sony comuni vengono riconosciuti da lensfun.

**Fase 6 — Style learning**
Inversione dei parametri, profili, predizione, selezione automatica, regolarizzazione di coerenza,
**profili predefiniti di §22** ed export/import `.apestyle`.
*Accettazione:* test 3 e 15 passano; dato un profilo addestrato su 30 coppie, su un set di
validazione di 10 foto non viste il ΔE2000 medio rispetto all'edit manuale dell'utente è < 6 e
nessuna foto supera 12. I tre profili predefiniti producono risultati presentabili sui RAW di
test senza alcun addestramento, e nessuno dei tre è visivamente peggiore del Neutro automatico.

**Fase 7 — Revisione, confidenza, escalation**
Punteggio di confidenza, coda outlier, revisione per cluster, propagazione delta, varianti
alternative, apprendimento dal feedback.
*Accettazione:* su un progetto da 500 foto, meno del 20% richiede revisione individuale e il
flusso completo di validazione si chiude in meno di 15 minuti di lavoro utente.

**Fase 8 — Export e sidecar**
Export batch, XMP darktable e Adobe, opzioni di output, pausa/ripresa.
*Accettazione:* test 7, 8, 11, 13 e 14 passano; target di export di §12 rispettato.

**Fase 9 — Maschere manuali e rifinitura**
Editor maschere (gradiente, radiale, pennello), segmentazione on-demand, scorciatoie da
tastiera complete, gestione errori in UI.

**Fase 10 — Confezionamento**
`install.sh` / `uninstall.sh` con `.desktop`, icone e `StartupWMClass` (§18), schermata
Impostazioni, storico e snapshot (§23), backup/restore e quota cache (§20), pannello Problemi e
bundle diagnostico (§19), `LICENSE` + `NOTICE.md` + `models/LICENSES.md` (§24).
*Accettazione:* test 16, 18 e 19 passano; su una macchina pulita `install.sh` porta a un'icona nel
menu di Ubuntu che apre l'applicazione **nella sua finestra**, con icona e nome corretti nella
dock (mai quelli di Chromium), e `uninstall.sh` la rimuove senza toccare i dati dell'utente.

**Fase 11 — Fusioni multi-scatto (§25)**
Nell'ordine, ciascuna consegnabile da sola: rilevamento dei gruppi e interazione con la cernita →
**HDR** → **focus stacking** → **panorama**. Il panorama va per ultimo perché è il più costoso in
RAM e l'unico che può fallire in modo irrimediabile.
*Accettazione:* test 17 passa; i target di §12 per le fusioni sono misurati e riportati; su un
bracketing e una panoramica reali dell'utente il risultato è utilizzabile senza ritocchi esterni,
e la foto derivata attraversa stile, revisione ed export esattamente come uno scatto singolo.

---

## 15. Import e scansione della cartella

- **Scansione non ricorsiva:** vengono importati solo i file presenti nella cartella selezionata.
  Le eventuali sottocartelle sono ignorate, ma il riepilogo di import le dichiara
  ("3 sottocartelle ignorate") perché il silenzio qui si trasforma in foto mancanti.
- Estensione accettata: `.arw` (case-insensitive). Altri formati RAW vengono riconosciuti ma
  rifiutati con un messaggio esplicito, non ignorati in silenzio.
- **JPEG affiancati ignorati:** un `.jpg`/`.jpeg` con lo stesso basename di un ARW non diventa
  una `Photo`. Il suo path finisce in `Photo.sidecar_jpeg_path` e può essere usato come anteprima
  immediata nella cernita, risparmiando anche l'`extract_thumb()`.
- **Deduplica per contenuto:** chiave rapida = dimensione in byte + SHA-256 dei primi 1 MB; solo
  in caso di collisione della chiave rapida si calcola l'hash completo. I file con contenuto
  identico producono **una sola** `Photo`; gli altri path sono conservati in `duplicate_paths` e
  il riepilogo di import dichiara quanti duplicati sono stati trovati.
- **Re-import idempotente:** reimportare la stessa cartella in un progetto esistente aggiunge solo
  i file nuovi. Le foto già presenti conservano edit, versioni, decisioni di cernita, cluster,
  confidenza e approvazioni — **un re-import non deve mai costare all'utente del lavoro già
  fatto**. I file spariti dalla sorgente vengono marcati `missing`: restano nel catalogo, visibili
  con un badge, e non vengono cancellati da nessuna parte.
- **Sorgente ri-mappabile:** se `source_dir` non esiste più (disco esterno rimontato altrove), il
  progetto si apre in stato "sorgente non trovata" e offre di ri-puntarla a una nuova cartella.
  Il riaggancio avviene **per hash** — autorevole, indipendente dai nomi — con fallback sul
  filename, e riporta quanti file sono stati riagganciati e quanti restano irreperibili.
- Tutto questo è sola lettura: il calcolo degli hash apre i file in `'rb'` e nient'altro (§2).

---

## 16. Export: naming, collisioni, metadati

### 16.1 Template del nome file

Configurabile per progetto, default `{basename}.{ext}`. Token disponibili:
`{basename}` `{ext}` `{counter:03}` `{seq}` `{date:%Y%m%d}` `{time:%H%M%S}` `{project}`
`{camera}` `{lens}` `{iso}` `{focal}`.

- I caratteri illegali per il filesystem vengono sostituiti con `_`; un template vuoto o che
  produce nomi duplicati per foto diverse viene rifiutato prima di iniziare la coda.
- La UI mostra un'**anteprima live** del nome risultante sulla prima foto del progetto: è l'unico
  modo perché l'utente capisca un template senza esportare 1000 file per scoprirlo.

### 16.2 Collisioni

**Mai una sovrascrittura silenziosa.** Alla prima collisione la UI mostra un dialogo con
**Sovrascrivi / Rinomina / Salta** e una casella "Applica a tutte le restanti".
La scelta è persistita in `Project.export_on_conflict`. "Rinomina" aggiunge un suffisso `_1`,
`_2`… Da CLI: `--on-conflict=rename|overwrite|skip`, default `rename`.
Una foto saltata è **saltata**, non fallita: non finisce nel pannello Problemi.

### 16.3 Metadati scritti nei file esportati

**Copiati dall'originale:** marca e modello del corpo, obiettivo, ISO, tempo, diaframma, focale,
`DateTimeOriginal`. L'orientamento è già applicato ai pixel, quindi `Orientation` viene scritto
a `1`: scriverlo diversamente raddoppierebbe la rotazione in metà dei visualizzatori.

**Scritti dal programma:** `Software` e `XMP:xmp:CreatorTool` = `autoPhotoEdit <versione>`.

**Configurabili nelle Impostazioni globali** (validi per tutti i progetti):
`Artist` / `XMP:dc:creator` e `Copyright` / `XMP:dc:rights`.

**GPS:** copiato **di default**, con un toggle "Rimuovi dati di posizione" nella schermata Export,
persistito per progetto. Quando è attivo non deve sopravvivere alcun tag GPS, in nessun blocco
(EXIF, XMP o IPTC): è un test, non una buona intenzione (§13.14).

**Non copiati:** numeri di serie del corpo e dell'obiettivo.

---

## 17. Modelli ONNX: provenienza, integrità, licenze

Un registry dichiarativo in `backend/ape/models_registry.py` descrive ogni modello: nome, URL,
SHA-256, dimensione attesa, licenza, e la feature che lo richiede.

**Comportamento:** il download avviene **alla prima attivazione della feature** che lo richiede,
non all'avvio dell'applicazione — l'utente che non attiva né volti né estetica né segmentazione
non scarica nulla. Barra di avanzamento, annullabile, destinazione
`$XDG_DATA_HOME/autophotoedit/models/`. La verifica SHA-256 è obbligatoria: se non corrisponde il
file viene cancellato e la feature resta disabilitata con un messaggio che spiega cosa è successo.
Senza rete, le feature opzionali restano disabilitate con una spiegazione e un link; **le feature
di default non dipendono da alcun modello e continuano a funzionare.**

| Modello | Usato per | Licenza | Peso indicativo |
|---|---|---|---|
| CLIP ViT-B/32, encoder visivo, ONNX int8 (`Xenova/clip-vit-base-patch32`) | embedding di scena (§8.3) | MIT | ~90 MB |
| YuNet (OpenCV Zoo) | rilevamento volti (§7.3) | MIT | ~0.3 MB |
| NIMA MobileNet ONNX | punteggio estetico (§7.3) | ⚠️ pesi derivati da **AVA**, *research-only* | ~14 MB |
| U²-Netp *oppure* BiRefNet-lite | segmentazione on-demand (§6.3) | Apache-2.0 / MIT | ~5 / ~45 MB |

**Vietati** perché con licenza non commerciale, anche se tecnicamente adatti: **MODNet**,
**RMBG-1.4**. Se serve un'alternativa, si sceglie tra licenze permissive e la si documenta.

`models/LICENSES.md` elenca per ciascun modello licenza, autore e URL di origine. Poiché i modelli
**non** sono ridistribuiti nella repo, le loro licenze non si propagano al progetto (§24) — ma
vanno comunque dichiarate all'utente, che è chi li scarica.

---

## 18. Distribuzione e installazione

Il programma si installa e si usa come un'applicazione desktop normale, non come uno script.

`install.sh`, **idempotente** e senza `sudo` salvo dove indicato:

1. verifica Python 3.12 e le librerie di sistema (`libraw`, `lensfun` + dati aggiornati,
   `libexiv2`). Se mancano, **stampa il comando `apt` esatto e si ferma**: non installa pacchetti
   di sistema di sua iniziativa;
2. `uv sync`;
3. costruisce il frontend se Node è presente; altrimenti usa la build precompilata in
   `backend/ape/static/`, che viene committata ai tag di release **proprio perché l'utente finale
   non debba avere Node**. La build include manifest, service worker e icone della PWA (§4);
4. installa il launcher `~/.local/bin/autophotoedit`, le icone in
   `~/.local/share/icons/hicolor/{48x48,128x128,256x256,scalable}/apps/autophotoedit.*` e
   `~/.local/share/applications/autophotoedit.desktop` (`Categories=Graphics;Photography;`,
   `Icon=autophotoedit`, e **`StartupWMClass=autophotoedit`, identico al `--class` passato al
   browser in §21.2** — è ciò che dà alla finestra l'icona e il nome giusti nella dock invece di
   quelli di Chromium), poi aggiorna la cache del database desktop;
5. stampa un riepilogo di cosa ha installato e dove.

`uninstall.sh` rimuove launcher e `.desktop` e **non tocca i dati dell'utente**; chiede
separatamente, con conferma esplicita, se rimuovere anche `~/.local/share/autophotoedit/`.

Il `.desktop` lancia `autophotoedit`, che avvia il server e apre la finestra dedicata
dell'applicazione (§21.2). Dal punto di vista dell'utente si comporta come qualunque altro
programma del menu: un'icona, una finestra, nessun browser in vista.

---

## 19. Errori e diagnostica

- **Nessuna telemetria, mai.** Le uniche chiamate di rete ammesse in tutto il programma sono il
  download dei modelli ONNX (§17) e l'aggiornamento dei dati lensfun, entrambe su azione
  esplicita dell'utente. Nessun controllo aggiornamenti automatico, nessuna analytics. **Il
  service worker non parla con nulla che non sia `127.0.0.1:<porta>`**, e il profilo browser
  dedicato di §21.2 nasce senza account, sincronizzazione o estensioni.
- **Log su file rotante** in `$XDG_STATE_HOME/autophotoedit/logs/` (5 file × 5 MB), livello via
  `APE_LOG_LEVEL`. Non è una feature a sé: è ciò che rende non vuoto il bundle diagnostico.
- **Pannello "Problemi"** in UI: le foto in stato `failed` con il motivo scritto in italiano
  comprensibile ("file RAW illeggibile", "obiettivo senza profilo lensfun"), il traceback
  copiabile dietro un "Dettagli tecnici" richiudibile, e "Riprova" per singola foto o per tutte.
  **Un errore su una foto non ferma mai il batch**; il conteggio dei falliti è sempre visibile.
- **Bundle diagnostico:** pulsante "Esporta diagnostica" e comando `autophotoedit diagnose`.
  Produce uno zip con log, versioni di Python, delle dipendenze e delle librerie di sistema,
  schema e statistiche del database, ed EXIF delle foto fallite. **Mai i RAW, mai i pixel, mai il
  contenuto delle foto.** I path sono anonimizzati sostituendo la home con `~`. La UI mostra
  l'elenco di ciò che sta per finire nello zip **prima** di scriverlo: l'utente deve poter vedere
  cosa sta per allegare a una issue pubblica.

---

## 20. Portabilità, backup e cache

### 20.1 Profili di stile trasportabili

Un profilo si esporta come singolo file `.apestyle` (uno zip con `format_version`):
`profile.json` con metadati e statistiche di normalizzazione, il modello e la PCA serializzati,
i vettori scena→parametri delle coppie, le **miniature 512 px dei sample** e i path originali dei
RAW.

L'import deve funzionare **anche se i RAW dei sample non esistono su quella macchina**: il profilo
resta pienamente utilizzabile e le miniature coprono il pannello "Sample di riferimento" della
revisione (§9.2), che altrimenti sarebbe vuoto proprio quando serve la trasparenza.

### 20.2 Backup del catalogo

- `autophotoedit backup <dest.db>` usa `VACUUM INTO`, consistente anche con il server acceso.
- `autophotoedit restore <src.db>` mette da parte il catalogo corrente prima di sostituirlo.
- Il backup esclude cache e proxy: sono rigenerabili, e includerli renderebbe il backup così
  grande da non essere fatto mai.

### 20.3 Quota della cache

Impostazione `cache_max_gb`, default **20 GB**. Superata la soglia, eviction LRU dei proxy e dei
buffer di stadio. Gli **intermedi delle fusioni** (§25) sono cache a tutti gli effetti — si
rigenerano dai RAW — ma hanno la **priorità di eviction più bassa** e una categoria propria nel
riepilogo: rigenerare un panorama costa minuti, non millisecondi. Prima di rimuoverne uno,
"Svuota cache" dichiara quante fusioni andranno rigenerate. **Le maschere raster disegnate a mano dall'utente non sono cache**: vivono in
`masks/`, sono conteggiate a parte e non vengono mai rimosse automaticamente, perché sono l'unica
cosa in quella cartella che non si può rigenerare.

La schermata Impostazioni mostra l'uso corrente per categoria (proxy / stadi / maschere / modelli)
e un pulsante "Svuota cache" che dichiara **in anticipo** quanti GB libera e cosa non cancella.

---

## 21. Server locale e finestra dell'applicazione

### 21.1 Server

- **Bind esclusivamente su `127.0.0.1`.** Non esiste un flag per esporlo sulla rete locale: è un
  vincolo della specifica, non un default modificabile.
- **Istanza singola:** lockfile in `$XDG_RUNTIME_DIR/autophotoedit.lock` con PID del server, porta
  e PID della finestra. Se il server è già attivo e risponde, rilanciare il comando (o cliccare
  l'icona) **riporta in primo piano la finestra esistente** — non ne apre una seconda — ed esce con
  codice 0. Se il server è vivo ma la finestra non c'è più (chiusa in modalità scheda, o avvio con
  `--no-window`), ne apre una nuova sull'istanza esistente. Un lockfile stantio (PID inesistente)
  viene rimosso e l'avvio prosegue.
- Se la porta è occupata da un processo estraneo: errore esplicito con l'indicazione di usare
  `--port`. **Nessun fallback silenzioso su un'altra porta**, che renderebbe imprevedibile
  l'indirizzo dell'applicazione.

### 21.2 Finestra dedicata (PWA)

**L'utente non deve mai vedere un browser:** niente barra degli indirizzi, niente schede, niente
segnalibri. `launcher.py` apre la UI in una finestra applicativa, con questa catena di fallback:

1. **Modalità applicazione di un browser Chromium** — cercato in quest'ordine (`chromium`,
   `chromium-browser`, `google-chrome`, `google-chrome-stable`, `brave-browser`,
   `microsoft-edge`, poi i rispettivi Flatpak) e sovrascrivibile con `APE_BROWSER` o
   `--browser PATH`:

   ```
   --app=http://127.0.0.1:8787/
   --user-data-dir=$XDG_DATA_HOME/autophotoedit/window-profile
   --class=autophotoedit --name=autophotoedit
   --window-size=1400,900 --no-first-run --no-default-browser-check
   ```

   - il **profilo dedicato** tiene la finestra separata dalla navigazione personale dell'utente,
     ne conserva geometria e stato, e fa sì che l'app si apra allo stesso modo sia che il browser
     sia già aperto sia che non lo sia. Non è un'ottimizzazione: senza di esso la finestra eredita
     estensioni, sessione e policy del profilo personale;
   - `--class` fissa il `WM_CLASS` a cui punta `StartupWMClass` nel `.desktop` (§18). Senza,
     la finestra compare nella dock con l'icona di Chromium e un nome sbagliato;
   - il processo della finestra è **figlio del launcher**: è così che "Esci" può chiuderla e che il
     launcher si accorge quando l'utente la chiude.
2. **PWA già installata** (l'utente ha usato "Installa come app"): la si apre per `app id` se il
   browser lo consente; altrimenti si ricade sul punto 1, che dà una finestra equivalente.
3. **Nessun browser Chromium disponibile** (p.es. solo Firefox, che non ha una modalità
   applicazione): si apre il browser predefinito in una scheda normale e **la UI mostra un avviso
   non bloccante, richiudibile per sempre**, che spiega in una riga come ottenere la finestra
   dedicata. Funziona tutto, ma l'esperienza è degradata e l'utente deve sapere perché.

`--no-window` avvia il solo server senza aprire nulla (sviluppo, uso via SSH, script); `--window`
forza l'apertura. Il launcher **non installa e non scarica mai un browser**: se non ne trova uno,
lo dice.

**"Installa come app":** la UI espone il pulsante — basato su `beforeinstallprompt` — solo quando
non è già in `display-mode: standalone` e il browser lo supporta. A installazione avvenuta la PWA
ha la sua voce di menu, creata dal browser: l'app non deve mai generarne una seconda.

### 21.3 Chiusura

- **Chiusura della finestra con job in corso:** un dialogo dell'applicazione (non il popup generico
  del browser) propone tre scelte — **Continua in background**, **Metti in pausa**,
  **Interrompi** — e la decisione viene inviata al server con `navigator.sendBeacon` prima della
  chiusura. Una casella "Ricorda la scelta" salva la preferenza in `Setting.on_window_close`.
  Se la chiusura è brusca (crash, kill), il comportamento è **continua**: i job sono su una coda
  persistente e alla riapertura la UI mostra lo stato reale.
- **Chiusura della finestra senza job in corso:** il launcher rileva l'uscita del processo figlio,
  ferma i worker, chiude il server e rilascia il lockfile. Chiudere la finestra spegne
  l'applicazione, come in qualunque programma desktop. **Questo non vale in modalità scheda
  (fallback 3) né con `--no-window`:** lì il server resta vivo, perché la chiusura di una scheda
  non è un segnale affidabile di volontà di uscire.
- Voce **"Esci dall'applicazione"** nella UI: ferma i worker in modo pulito, chiude il server,
  termina il processo della finestra e rilascia il lockfile. È il modo previsto per terminare,
  senza cercare il terminale.

---

## 22. Cold start: profili predefiniti

**Il programma deve essere utile al primo avvio, prima che l'utente abbia addestrato qualsiasi
cosa.** Un sistema che chiede 30 coppie RAW+editate prima di mostrare un risultato non viene mai
provato abbastanza a lungo da arrivare a quelle 30 coppie.

**Neutro automatico** — sempre presente, non cancellabile. Non è un modello, è una procedura
deterministica: auto-WB (grey-world robusto con reiezione degli outlier, vincolato a un intervallo
plausibile attorno alla temperatura as-shot), auto-esposizione sui percentili dell'istogramma,
sigmoid standard, contrasto e saturazione moderati, nessuna deriva cromatica voluta. È anche la
**baseline di confronto**: la UI deve poter mostrare in qualunque momento "questo profilo *vs*
neutro".

**Tre profili predefiniti**, costruiti a mano dall'agente come `EditParams` più regole di
adattamento alla scena, validati sui RAW di test dell'utente e documentati (cosa fanno e perché):

- **Naturale** — fedele, contrasto medio, colori calibrati; il più vicino a "come lo vedeva
  l'occhio".
- **Ritratto** — toni della pelle protetti (canali HSL arancio e rosso vincolati), ombre più
  morbide, saturazione contenuta, clarity bassa (la clarity sui volti invecchia).
- **Paesaggio** — contrasto e contrasto locale più marcati, vividezza selettiva su verdi e blu,
  recupero delle alte luci più aggressivo.

I predefiniti sono **duplicabili e modificabili ma non sovrascrivibili**: modificarne uno crea
automaticamente una copia utente. Sono righe `StyleProfile` con `builtin = true` e `model = NULL`;
la predizione usa `builtin_rules` invece del k-NN, mentre confidenza, coerenza intra-progetto,
revisione e feedback funzionano **esattamente come per i profili appresi** — nessun percorso
speciale nel codice a valle della predizione.

---

## 23. Storico delle modifiche e snapshot

### 23.1 Storico per foto

Ogni modifica dei parametri crea una nuova `EditVersion` con `parent_version_id`: **non esiste
un update in place dei parametri**. La UI mostra la cronologia della foto con origine (predetto /
modificato da te / applicato dal cluster / ripristinato da snapshot) e orario, e permette di
tornare a qualunque versione precedente. Il ripristino **crea una nuova versione**, non cancella
la storia: annullare un annullamento deve essere sempre possibile.

`Ctrl+Z` / `Ctrl+Shift+Z` navigano questa catena, non uno stack separato in memoria: ciò che
l'utente annulla oggi deve essere annullabile anche dopo un riavvio.

**Coalescenza:** i movimenti consecutivi dello stesso slider entro 2 secondi producono **una**
versione, non venti. Senza questa regola la cronologia diventa illeggibile e il database cresce
per nulla.

### 23.2 Snapshot di progetto

- L'utente può nominare uno stato dell'intero progetto ("prima della revisione manuale") e
  tornarci. Uno snapshot registra **quale versione era corrente** per ogni foto, più lo stato di
  cernita: costa pochi KB, non duplica alcun pixel.
- Snapshot automatici ai passaggi di fase (fine cernita, fine predizione, fine revisione),
  etichettati come automatici, limitati agli ultimi 10 con eliminazione dei più vecchi.
- Il ripristino di uno snapshot crea implicitamente uno snapshot dello stato precedente, quindi è
  a sua volta annullabile.

---

## 24. Licenza

**Il progetto è GPL-3.0-or-later.** Non è una preferenza: è una conseguenza tecnica. `pyexiv2` è
GPL-3.0 ed `exiv2` GPL-2.0-or-later; qualunque licenza permissiva sarebbe incompatibile finché si
linka exiv2. Tutte le altre dipendenze sono compatibili: LibRaw (LGPL-2.1 / CDDL), lensfun
(LGPL-3), rawpy (MIT), OpenCV (Apache-2.0), NumPy/SciPy (BSD), ONNX Runtime (MIT),
FastAPI/Pydantic/SQLAlchemy/uvicorn (MIT/BSD), React/Vite/Tailwind/TanStack/Zustand (MIT),
shadcn-ui (MIT), vite-plugin-pwa e Workbox (MIT).

Richiesto:

- `LICENSE` con il testo completo della GPL-3.0 e un'intestazione di licenza nei file sorgente
  principali;
- `NOTICE.md` con l'elenco delle dipendenze e delle rispettive licenze, **generato e verificato in
  CI** (una dipendenza nuova con licenza incompatibile deve rompere la build, non passare
  inosservata);
- `models/LICENSES.md` (§17).

**Nota progettuale per il futuro:** l'unico ostacolo a una licenza permissiva è pyexiv2,
sostituibile con `exiftool` via subprocess o con una scrittura XMP diretta (l'XMP è solo XML).
Perché quella sostituzione resti un lavoro di poche ore anziché un refactoring, **tutto l'accesso
a exiv2 deve restare confinato dietro le interfacce di `raw/metadata.py` e `export/`**: nessun
`import pyexiv2` altrove nel codice, verificato da un test.

---

## 25. Fusioni multi-scatto: HDR, focus stacking, panorama

Sono le uniche operazioni **molti-a-uno** del programma. Tutto il resto della specifica assume
"un RAW → una foto": questa sezione definisce l'unico punto in cui quell'assunzione si rompe, e
lo fa in modo da non propagare la rottura a valle.

### 25.1 Principio: la fusione produce una *foto derivata*

Una fusione accettata crea una nuova `Photo` con `kind = 'merged'`, che:

- **non ha un file sorgente proprio** (`path` è NULL): il suo "RAW" è un **intermedio float32
  lineare scene-referred** (EXR mezza precisione, o TIFF 32 bit) salvato in cache e puntato da
  `intermediate_path`;
- è **rigenerabile** in qualunque momento dai RAW dei membri, che restano intatti: cancellare
  l'intermedio non perde nulla di irrecuperabile, esattamente come per i proxy (§2.5, §20.3);
- da lì in poi **è una foto come tutte le altre**. Stesso `EditParams`, stesso profilo di stile,
  stesso clustering, stessa confidenza, stessa revisione, stesso export, stessi sidecar XMP.

> **Questa è la regola architetturale della sezione:** la fusione sostituisce lo stadio
> `decode RAW` e nient'altro. Nessun modulo a valle — tone mapping, stile, revisione, export —
> deve contenere una sola riga che sappia dell'esistenza delle fusioni. Se ti accorgi di scrivere
> `if photo.kind == 'merged'` fuori da `merge/` e dalla decodifica, l'astrazione è sbagliata.

I membri restano nel progetto come foto normali, marcate come appartenenti al gruppo e nascoste di
default dietro il filtro "Mostra scatti sorgente": non vengono mai scartate né cancellate.

La foto derivata eredita i metadati dal frame **di riferimento** (`role = 'reference'`), con
`DateTimeOriginal` del primo scatto della sequenza.

### 25.2 Rilevamento dei gruppi — proposta, mai applicazione

Il rilevamento gira sulle **miniature incorporate** (§7.2), mai sui RAW decodificati, e costa
≤ 40 ms per foto. Produce `MergeGroup` in stato `proposed`. **Nessuna fusione viene mai eseguita
senza conferma esplicita dell'utente**, coerentemente con il crop proposto di §6.4.

Il rilevamento gira **prima** del raggruppamento raffiche e ne esclude i membri (§7.3.4).

**HDR / bracketing** — scatti consecutivi entro 3 s, stessa focale e stesso diaframma, con
`ExposureBiasValue` o tempo di posa che variano in modo monotòno, tipicamente 3/5/7 frame.
Indizio forte: i tag Sony di bracketing (`BracketShotNumber`, `SequenceNumber`). Conferma visiva:
le miniature sono allineate e differiscono quasi solo in luminanza.

**Focus stacking** — scatti consecutivi con esposizione **costante** e distanza di messa a fuoco
(`FocusDistance`, o posizione dell'attuatore AF quando disponibile) che varia in modo monotòno;
inquadratura pressoché identica; la mappa di nitidezza locale si sposta tra un frame e l'altro.

**Panorama** — scatti consecutivi con esposizione ed EXIF omogenei e **sovrapposizione parziale**
stimata sulle miniature (feature matching rapido, ORB su immagini a 256 px): la sovrapposizione
utile è 15–60%, con uno spostamento prevalentemente su un solo asse. Sotto il 15% non si propone.

Ogni gruppo porta una **confidenza** e i motivi del rilevamento in chiaro ("5 scatti, EV da −2 a
+2, intervallo 0.8 s"), perché l'utente possa giudicare senza aprire le foto.

**Un gruppo rifiutato resta rifiutato:** `decision = 'rejected'` è permanente per quel gruppo e
non viene riproposto a ogni riapertura del progetto. L'utente può sempre creare un gruppo a mano
selezionando N foto nella griglia e scegliendo il tipo di fusione: il rilevamento automatico è una
comodità, non l'unica strada.

### 25.3 HDR merge

La fusione avviene **in lineare scene-referred, prima di qualunque tone mapping**, ed è per questo
che si integra senza attriti: il RAW è già lineare, quindi **non serve stimare la curva di
risposta del sensore** — niente Debevec, niente Robertson. Il RAW è il caso facile dell'HDR.

1. Decodifica ogni membro con `EditParams` neutri, stesso WB (quello del frame di riferimento):
   un WB diverso tra i frame introdurrebbe deriva cromatica nella fusione.
2. Allineamento: omografia globale stimata con ORB + RANSAC sui frame a scala ridotta, poi
   raffinata con ECC. Se lo spostamento residuo supera pochi pixel, dichiaralo e lascia decidere.
3. Normalizza ogni frame per il proprio guadagno relativo, ricavato dagli EV EXIF e **verificato**
   sui pixel comuni ben esposti (l'EV nominale e quello reale differiscono spesso di qualche
   decimo).
4. Media pesata per pixel con pesi che favoriscono i valori lontani dai due estremi: peso ~0 sui
   pixel clippati e su quelli affogati nel rumore di lettura, peso massimo a metà scala. Il
   risultato è un buffer lineare con range dinamico esteso, **non tone-mappato**.
5. **Deghosting** (attivabile, default on): rileva i pixel la cui deviazione dal frame di
   riferimento eccede il rumore atteso e, in quelle zone, usa il solo frame di riferimento
   riscalato. È ciò che salva foglie al vento e persone che camminano.

Il risultato entra nella pipeline normale: il tone mapping sigmoid di §6.2 **è già** l'operatore
di tone mapping HDR, con i suoi `black_point_ev` / `white_point_ev` semplicemente più distanti.
Nessun operatore locale di tone mapping in stile "HDR 2010": il programma non deve produrre
quell'estetica.

### 25.4 Focus stacking

1. Allineamento con **correzione di scala**: il focus breathing cambia l'ingrandimento tra i
   frame, quindi una semplice traslazione non basta. Stima una similarità (traslazione +
   rotazione + scala) con ECC, partendo da ORB.
2. Per ogni frame, mappa di nitidezza locale = risposta di un laplaciano su piramide gaussiana.
3. Fusione con **piramidi laplaciane**: a ogni livello si prende il coefficiente del frame con
   nitidezza locale maggiore, con una transizione morbida fra i vincitori per evitare cuciture
   visibili. Il lavoro è a tile: la RAM non è un vincolo qui.
4. Le zone in cui **nessun** frame è a fuoco vengono segnalate: una mappa di copertura mostra in
   overlay dove lo stack è incompleto, invece di restituire in silenzio una foto sfocata a metà.

### 25.5 Panorama

È la funzione più costosa e l'unica che può fallire in modo irrimediabile. Implementazione con
`cv2.Stitcher` **a parametri controllati**, non come scatola nera, con una strategia esplicita per
la memoria — su 8 GB disponibili, 6 frame da 24 MP in float32 sono già 1.7 GB di soli input.

1. **Stima a scala ridotta:** omografie, ordine dei frame, compensazione dell'esposizione e
   ricerca delle cuciture avvengono su frame a **1/4 di lato** (1/16 dei pixel). È qui che si
   decide tutto, e costa poco.
2. **Composizione a bande:** il warping e il multi-band blending finali sono eseguiti a piena
   risoluzione **per bande orizzontali**, tenendo in RAM solo la banda corrente più i margini di
   sovrapposizione. Il risultato si scrive incrementalmente sull'intermedio.
3. **Proiezione** selezionabile: cilindrica (default per panoramiche orizzontali), sferica,
   piana (per panoramiche di pochi scatti a focale lunga). La proposta dipende dall'ampiezza
   angolare stimata; l'utente può cambiarla e rivedere l'anteprima.
4. **Tetto dichiarato:** oltre ~200 MP di output o oltre 12 frame in ingresso, il programma non
   procede in silenzio: propone di ridurre la risoluzione di output o di dividere la panoramica,
   dichiarando la RAM stimata. **Meglio un rifiuto spiegato che un OOM killer.**
5. Il crop del panorama (i bordi irregolari) segue la regola di §6.4: **proposto**, non applicato.
   Si propone il rettangolo interno massimo privo di pixel vuoti.
6. Se la cucitura fallisce (sovrapposizione insufficiente, parallasse, soggetto in movimento sulla
   giunzione), il gruppo va in `failed` con un motivo leggibile e **nessuna foto derivata viene
   creata**. Fallire bene è un requisito: una panoramica rotta che entra nel flusso di export è
   peggio di una panoramica mancante.

### 25.6 UI

- Una schermata **Fusioni**, raggiungibile dopo l'import (e dalla cernita), elenca i gruppi
  proposti come card: tipo, numero di scatti, miniature dei membri, motivi del rilevamento,
  confidenza. Per ciascuna: **Anteprima**, **Accetta**, **Rifiuta**, **Modifica gruppo**
  (aggiungi/rimuovi scatti, cambia il frame di riferimento).
- **L'anteprima si calcola a risoluzione ridotta** (lato lungo 1024 px) ed è veloce: l'utente deve
  poter vedere se il panorama si cuce prima di spendere tre minuti di CPU sulla versione full-res.
  La fusione a piena risoluzione parte solo dopo l'accettazione, come job in coda (§12).
- Nella griglia del progetto la foto derivata mostra un badge con il tipo di fusione e il numero
  di sorgenti ("HDR · 3"); un click espande i membri.
- Il pannello Problemi (§19) accoglie le fusioni fallite, con "Riprova" e la possibilità di
  cambiare i parametri prima di riprovare.
- Una fusione si può **annullare**: la foto derivata viene rimossa, i membri tornano visibili, il
  gruppo torna in `proposed`. Come ogni altra operazione, rientra nello storico di §23.

### 25.7 Vincoli che valgono comunque

- **§2 non ammette eccezioni qui.** I RAW membri sono aperti in sola lettura; l'intermedio vive in
  cache; la foto derivata non esiste su disco fuori dalla cache e dall'export.
- Le fusioni sono **opt-in per progetto** (`Project.merge_detection_enabled`, default attivo solo
  per il rilevamento, mai per l'esecuzione).
- Nessuna dipendenza nuova: solo OpenCV e NumPy. Niente Hugin, enblend, enfuse, PyTorch.
- Una foto derivata può essere usata come **sample di uno stile** (§8): ha un intermedio lineare,
  che è tutto ciò che serve all'inversione dei parametri.
- La cernita non valuta mai la nitidezza o l'esposizione di un membro di gruppo candidato con i
  criteri pensati per gli scatti singoli (§7.3.4).

---

## 26. Regole di lavoro per l'agente implementatore

- **Chiedi prima di inventare** su: comportamento di default che cambia il risultato visivo,
  formato dei file esportati, qualunque dipendenza non elencata in §4.
- **Non introdurre PyTorch, TensorFlow o dipendenze GPU a runtime.**
- **§2 (non distruttività) è un invariante, non una linea guida.** Ogni PR che introduce una
  scrittura su file deve passare da `assert_outside_source`.
- **Budget di leggerezza — la macchina non ha GPU dedicata.** Verifica questi limiti a ogni fase:
  - picco RAM del processo server a riposo < 400 MB; per worker < 1.5 GB;
  - **budget dei modelli ONNX: ≤ 120 MB per modello, ≤ 350 MB in totale.** *(Questi numeri
    sostituiscono i 25/100 MB di una stesura precedente della spec: CLIP ViT-B/32 pesa ~90 MB
    anche quantizzato int8, quindi il limite originale era irrealizzabile. Se si trova un
    embedding di qualità equivalente sotto i 40 MB, usalo e riporta il confronto.)*
  - i modelli si caricano **pigramente**, alla prima richiesta, e si scaricano dopo 5 minuti di
    inattività;
  - a idle il programma consuma ~0% di CPU: nessun polling, nessun timer di background;
  - il bundle JS del frontend resta sotto 1.5 MB gzip; niente librerie di componenti pesanti
    oltre a quelle elencate in §4;
  - **preferisci sempre l'anteprima incorporata o il proxy al RAW full-res**; il demosaicing
    completo avviene solo all'export o quando l'utente zooma a 1:1 su una singola foto.
- Scrivi i test della fase corrente *insieme* al codice, non dopo.
- Ogni operazione della pipeline è una funzione pura `(np.ndarray, params) -> np.ndarray` senza
  stato globale, con docstring che dichiara: spazio colore atteso in ingresso, range, e in uscita.
- Commenta il *perché* delle scelte numeriche (coefficienti, soglie), non il *cosa*.
- Nessun file di codice sopra le ~400 righe: se cresce, si divide.
- Prima di dichiarare completa una fase, **esegui** il programma sui RAW reali dell'utente e
  riporta i numeri misurati, non le stime.
- Quando un target di prestazione o qualità non è raggiunto, dillo esplicitamente con i numeri.

---

## 27. Primo passo

1. Crea `pyproject.toml` con `uv`, la struttura di directory di §5, il file `LICENSE` (GPL-3.0,
   §24) e un `README.md` con le istruzioni di installazione (incluse le dipendenze di sistema:
   `libraw`, `lensfun`, `libexiv2`, e i dati di lensfun aggiornati).
2. Implementa la Fase 1.
3. Chiedi all'utente **3–5 file ARW di esempio** (A7 II e A7 III, di scene diverse: interno,
   esterno, controluce) e, se disponibili, le loro versioni già post-prodotte, da mettere in
   `tests/fixtures/`. Sono necessari per validare la Fase 1 e indispensabili per la Fase 6.
