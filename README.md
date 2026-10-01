# autoPhotoEdit

Post-produzione automatica di file RAW Sony (`.ARW`), interamente locale e **non distruttiva**.

Il programma impara uno stile dalle tue foto già post-prodotte, sviluppa i RAW con una pipeline
propria e ti chiede conferma solo sugli scatti su cui non è abbastanza sicuro.

> **I file RAW originali non vengono mai modificati, spostati, rinominati o cancellati.**
> È l'invariante su cui è costruito tutto il resto: ogni modifica è un dato nel database, mai un
> pixel scritto sull'originale. Vedi `docs/SPEC.md` §2.

La specifica completa è in [`docs/SPEC.md`](docs/SPEC.md) ed è la fonte di verità del progetto.

---

## Stato di avanzamento

| Fase | Contenuto | Stato |
|---|---|---|
| 1 | Fondamenta della pipeline: decodifica RAW, `EditParams`, operazioni tonali e colore, export, CLI | **completa**, validata su 24 ARW dell'A7 III |
| 2 | Catalogo SQLite, coda persistente, worker pool, proxy, API REST e WebSocket | **completa** |
| 3 | Frontend, PWA e finestra dell'applicazione | **completa** |
| 4 | Cernita: punteggi tecnici sull'anteprima incorporata, raffiche, bracketing protetti, schermata di cernita | **completa**; volti ed estetica disattivati (modelli non ancora verificabili) |
| 5 | Analisi e geometria: correzione lente, raddrizzamento, crop proposto, feature di scena, embedding, cluster | **completa**; import+analisi di 1000 foto in 3,7 min (target 4) |
| 6 | Apprendimento dello stile: inversione dei parametri, profili, predizione, coerenza, profili predefiniti, `.apestyle` | **implementata**; test 3 e 15 ✓, criterio ΔE sulle 65 coppie dell'utente al limite |
| 7 | Revisione, confidenza, escalation | **implementata**; 14,8% in revisione singola (target < 20%), flusso di 500 foto stimato al limite dei 15 min |
| 8 | Export e sidecar XMP | **implementata**; test 7, 8, 11, 13, 14 ✓; target di velocità dell'export **non** raggiunti (vedi sotto) |
| 9 | Maschere manuali, segmentazione su richiesta, scorciatoie, errori in UI | **completa** |
| 10 | Installazione, Impostazioni, storico e snapshot, backup, quota della cache, Problemi e diagnostica | **implementata**; test 16, 18, 19 ✓; icona e nome nella dock da verificare a mano |
| 11 | Fusioni multi-scatto (HDR, focus stacking, panorama) | **implementata**; test 17 ✓, target di §12 ✓, verificata su bracketing e panoramiche vere dell'utente (manca un focus stack vero) |
| — | Rimozione: correttivo a punto e gomma magica ([`docs/SPEC_rimozione.md`](docs/SPEC_rimozione.md)) | **implementata**: motore classico e motore IA (LaMa int8, scaricabile) |

---

## Dipendenze di sistema

Su Linux x86_64 **nessuna**: i pacchetti Python installati da `uv` includono già le librerie
native che servono.

| Libreria | Serve per | Arriva con | Da quale fase |
|---|---|---|---|
| LibRaw | decodifica dei file ARW | `rawpy` | 1 |
| exiv2 | lettura EXIF e scrittura dei sidecar XMP | `pyexiv2` | 1 (lettura), 8 (scrittura) |
| lensfun + database | profili obiettivo: distorsione, vignettatura, aberrazione cromatica | `lensfunpy` | 5 |

Il database lensfun incluso in `lensfunpy` è datato e non conosce alcuni obiettivi recenti (per
esempio il Tamron 28-75 G2 A063). Si aggiorna dall'applicazione, schermata **Scene** →
«Aggiorna dati lensfun», che scarica il database nel formato 1 (l'unico che la libreria inclusa
accetta) in `~/.local/share/autophotoedit/lensfun/`. Non servono `lensfun-update-data` né i
pacchetti apt.

Su piattaforme per cui questi pacchetti non hanno wheel precompilati, `uv` li compila e servono
allora le librerie di sistema (`libraw`, `libexiv2`, `liblensfun`) con i loro header.

## Installazione

```sh
./install.sh      # per l'utente corrente, senza sudo; rilanciarlo aggiorna
./uninstall.sh    # toglie programma, voce di menu e icone; i dati restano salvo un «sì» esplicito
```

`install.sh` controlla che ci siano `uv` e Python 3.12 (se mancano stampa il comando esatto e si
ferma: non installa mai pacchetti di sistema), esegue `uv sync`, costruisce l'interfaccia se c'è
Node (altrimenti usa la build già presente in `backend/ape/static/`), poi installa:

- il comando `~/.local/bin/autophotoedit`;
- le icone in `~/.local/share/icons/hicolor/{48x48,128x128,256x256,scalable}/apps/`;
- la voce di menu `~/.local/share/applications/autophotoedit.desktop`, con
  `StartupWMClass=autophotoedit`, lo stesso `--class` che il programma dà alla finestra: è ciò che
  mette nella dock l'icona e il nome di autoPhotoEdit invece di quelli di Chromium.

I dati (catalogo, maschere, modelli, cache) stanno in `~/.local/share/autophotoedit/`, i log in
`~/.local/state/autophotoedit/logs/`.

## Installazione (sviluppo)

Serve [`uv`](https://docs.astral.sh/uv/) e Python 3.12.

```sh
uv sync --extra metadata                     # solo la pipeline e la lettura EXIF
uv sync --extra metadata --extra server      # + catalogo, coda e server locale (Fase 2)
uv sync --all-extras                         # tutto, inclusi lensfun e onnxruntime
```

## Uso

### Il programma

```sh
# avvia il server locale (è anche ciò che fa il comando senza argomenti)
uv run autophotoedit                       # http://127.0.0.1:8787/
uv run autophotoedit serve --port 9000
uv run autophotoedit serve --no-workers    # solo server, nessun job in background

# importa una cartella da riga di comando e aspetta le anteprime
uv run ape import ~/Foto/2026-02-matrimonio --wait
```

Il server si lega **solo** a `127.0.0.1` (§21.1) e non esiste un'opzione per esporlo in rete.
L'API è documentata da sola su `/docs`.

### La finestra (§21.2)

`autophotoedit` apre l'interfaccia in una **finestra applicativa**: niente barra degli indirizzi,
niente schede, niente segnalibri. La catena è: PWA installata nel profilo dedicato → Chromium in
modalità `--app` → scheda del browser predefinito. Il terzo caso funziona lo stesso, ma la UI
mostra un avviso che spiega perché sembra un browser.

```sh
uv run autophotoedit                       # server + finestra
uv run autophotoedit --no-window           # solo server (sviluppo, SSH, script)
uv run autophotoedit --browser /usr/bin/chromium   # anche con APE_BROWSER
```

Il browser viene cercato in quest'ordine: `chromium`, `chromium-browser`, `google-chrome`,
`google-chrome-stable`, `brave-browser`, `microsoft-edge`, poi i rispettivi Flatpak. **Nessun
browser viene mai installato o scaricato**: se non ce n'è, il programma lo dice.

Rilanciare il comando **non apre una seconda istanza**: se il server è già attivo, riporta in
primo piano la finestra esistente ed esce con codice 0; se la finestra non c'è più (chiusa in
modalità scheda o avvio con `--no-window`), ne apre una sull'istanza già in esecuzione. Per
portare in primo piano una finestra esistente serve `wmctrl` o `xdotool`: senza, il comando lo
dice invece di aprirne un'altra.

```sh
sudo apt install wmctrl        # facoltativo: consente il «riporta in primo piano» di §21.1
```

Chiudere la finestra chiude il programma, come in qualunque applicazione desktop — **tranne**
quando ci sono lavorazioni in corso: in quel caso la UI chiede se continuare in background,
mettere in pausa o interrompere, e la scelta viaggia con `navigator.sendBeacon` (§21.3).

### Manutenzione (Fase 10)

- **Impostazioni** (in alto, accanto a «Problemi»): autore e copyright, spazio occupato per
  categoria, quota della cache (20 GB di default: oltre, si liberano per primi i file usati meno
  di recente, le fusioni per ultime; le maschere dipinte a mano e i modelli mai), «Svuota cache»
  che dice prima quanto libera e cosa non tocca, e cosa fare chiudendo la finestra con lavori in
  corso.
- **Snapshot** (Visualizzatore e Revisione): uno stato nominato del progetto — versione corrente
  di ogni foto, decisioni di cernita e revisione — a cui tornare; tornarci non cancella nulla ed è
  a sua volta annullabile. Se ne creano da soli a fine cernita, fine predizione e fine revisione
  (gli ultimi 10). Le modifiche si salvano da sole: più movimenti dello stesso cursore entro 2
  secondi fanno una versione sola; la striscia delle versioni dice da dove viene ciascuna.
- **Problemi**: le foto non riuscite con il motivo, i dettagli tecnici copiabili, «Riprova». Il
  contatore rosso nell'intestazione compare appena qualcosa fallisce.

```sh
uv run autophotoedit backup ~/catalogo.db    # VACUUM INTO: coerente anche col programma aperto
uv run autophotoedit restore ~/catalogo.db   # a programma chiuso; il catalogo attuale va da parte
uv run autophotoedit diagnose -o ~/Scrivania # zip per una segnalazione: mai RAW né pixel
```

Il backup contiene solo il catalogo: le maschere dipinte a mano stanno in
`~/.local/share/autophotoedit/masks/` e i riempimenti della gomma magica in
`~/.local/share/autophotoedit/retouch/`, e vanno copiati insieme (il comando lo ricorda). La diagnostica (anche dal pannello
Problemi, «Esporta diagnostica») mostra prima di scriverlo ogni file che conterrà; la cartella home
vi compare come `~`. Il livello del log su file si regola con `APE_LOG_LEVEL` (default `INFO`).

### Il frontend

```sh
cd frontend
npm install
npm run dev      # http://127.0.0.1:5173, con proxy di /api e /ws verso il server
npm run build    # scrive in backend/ape/static/, da dove uvicorn lo serve
npm run typecheck
```

In sviluppo servono due processi (`uv run autophotoedit --no-window` e `npm run dev`); in
produzione uno solo, perché la build finisce dentro il pacchetto Python. La build **non è
committata** (§18.3): senza di essa il server risponde solo sotto `/api`, e i test della PWA si
saltano da soli.

Le icone dell'applicazione (PWA e voce di menu, in `packaging/icons/`) si rigenerano con:

```sh
cd frontend && ../.venv/bin/python scripts/make_icons.py
```

`NOTICE.md` contiene l'elenco completo delle dipendenze distribuite con le loro licenze, generato
da `packaging/notice.py`; `tests/test_notice.py` rompe la build se una licenza non è compatibile
con la GPL-3.0 o se l'elenco non è aggiornato (`uv run python packaging/notice.py --write`).

### Gli stili (Fase 6)

Dalla schermata **Stili** (in alto, accanto a «Progetti»): «Nuovo profilo», un nome, la cartella
dei RAW originali e quella delle versioni editate (JPEG o TIFF, anche la stessa cartella).
L'accoppiamento avviene per nome, poi per il nome originale che Lightroom e Camera Raw scrivono
nell'XMP, poi per ora di scatto; ciò che resta si accoppia a mano nella stessa schermata. Ogni
coppia viene invertita in background (~11 s di un core) e il profilo si addestra da solo quando
l'ultima è pronta. Nessun file delle due cartelle viene modificato.

Nel progetto, il pulsante **Stile** del visualizzatore apre la scelta: il profilo più affine è
proposto, il cursore «Coerenza» avvicina fra loro le foto della stessa scena. Le foto modificate
a mano non vengono toccate. Nel visualizzatore il riquadro «Stile» mostra i sample che hanno
guidato l'edit e permette il confronto con il *Neutro automatico*.

I quattro profili predefiniti (Neutro automatico, Naturale, Ritratto, Paesaggio) funzionano
senza addestramento. Un profilo si esporta e importa come file `.apestyle`.

### L'export (Fase 8)

Dalla schermata **Esporta** (pulsante nel visualizzatore e nella revisione): cartella di
destinazione (mai quella dei RAW), JPEG con qualità o TIFF 8/16 bit, spazio colore,
ridimensionamento, nitidezza di output, modello del nome con anteprima dal vivo, cosa fare se un
file esiste già (chiedi / rinomina / sovrascrivi / salta), rimozione dei dati di posizione,
autore e copyright, e i sidecar XMP per darktable (`DSC0001.ARW.xmp`) e Lightroom / Camera Raw
(`DSC0001.xmp`). Il batch si mette in pausa, riprende (anche dopo un arresto del programma) e
riprova le foto non riuscite. Esporta tutte le foto tenute, o solo le approvate.

```sh
# da riga di comando: stesso batch della schermata; le collisioni per default si rinominano
uv run autophotoedit export "Nome progetto" -o ~/Immagini/esportate --xmp darktable,adobe
uv run autophotoedit export 1 -o /media/scheda/web --long-edge 2048 --sharpening standard \
    --template "{date:%Y%m%d}_{counter:03}.{ext}" --strip-gps
uv run autophotoedit export 1 -o ~/xmp --xmp-only --xmp adobe   # solo i sidecar
```

I sidecar sono un punto di partenza nei controlli di quel programma, non gli stessi pixel. Quello
darktable si apre in darktable 5.6 con tutti i moduli caricati (test 7, automatico se darktable è
installato) e, rispetto al nostro render, dista ΔE2000 2–4 in media sui look reali
(`tests/bench/bench_sidecar.py`). Quello Lightroom è tarato sulle 65 edit dell'utente
(bianco, esposizione, convenzioni del crop); la lettura in Lightroom va verificata a mano.

### Le fusioni (Fase 11)

Dopo l'import il programma cerca da solo, sulle anteprime incorporate, i **bracketing** (anche
quelli a 1/3 di stop, se la fotocamera dice che stava facendo bracketing), i **focus stack** e le
**panoramiche**, comprese quelle scattate a raffica ruotando. Sono solo proposte: la schermata
**Fusioni** (dall'import, dalla cernita o dal visualizzatore) le mostra con i motivi in chiaro,
un'anteprima a 1024 px in pochi secondi, **Accetta**, **Rifiuta** e **Modifica gruppo**. Un
gruppo si crea anche a mano con Ctrl+clic nella griglia. I fotogrammi di una fusione non vengono
mai scartati dalla cernita come raffica o come esposizione sbagliata.

Accettata, la fusione diventa una foto derivata («HDR · 5», «Panorama · 17») che attraversa
stile, revisione ed export come uno scatto singolo; i suoi scatti si fanno da parte («Mostra
scatti sorgente»), e annullando tornano. Le panoramiche usano solo i fotogrammi che servono (4
dei 17 di una raffica), riconoscono da sole quelle verticali e propongono il crop senza bordi
vuoti. Le foto derivate si esportano senza XMP: non c'è un RAW da descrivere.

### La rimozione

Nella scheda **Maschere** del Visualizzatore e della Revisione, il gruppo **Rimozione** ha due
strumenti e una lista propria:

- **Correttivo** (Q): un clic su una macchia, un brufolo, un granello di polvere la copre con una
  zona vicina scelta dal programma, fondendone colore e luce. Si vedono due cerchi: il pieno è la
  macchia, il tratteggiato la sorgente; si trascinano entrambi, `[` e `]` cambiano la dimensione,
  «Scegli un'altra sorgente» ne propone una diversa.
- **Gomma magica** (Maiusc+Q): si dipinge sopra ciò che si vuole togliere; al rilascio il
  programma lo riempie con ciò che lo circonda (qualche secondo, in un worker). Con una gomma
  selezionata il pennello ne corregge l'area (Alt per togliere); «Espandi» allarga l'area,
  «Altra variante» rifà il riempimento in un altro modo. **Da una maschera…** parte dalla
  selezione di una maschera esistente (anche un «Soggetto»), fotografata in quel momento.
- **Motore**: «Classico» funziona sempre, senza modelli né rete, ed è il migliore sulle superfici
  lisce (cielo, ghiaccio, muri). «IA» usa LaMa (int8, 93 MB, Apache-2.0) e va meglio sugli
  oggetti grandi in scene complesse (una panchina, una persona sullo sfondo). La prima volta che
  lo scegli il pannello offre il download, con dimensione e licenza; il modello è addestrato su
  Places2, le cui immagini sono concesse per ricerca non commerciale, e il pannello lo dice.
  Circa 2 s a riempimento, più ~3 s al primo uso per caricarlo.
- «Mostra le rimozioni» le sospende tutte mentre si lavora; `\` confronta prima e dopo.
- **Copia punti sulle foto selezionate…** (menu ⋯, con le foto scelte nella griglia con
  Ctrl+clic o Maiusc+clic): per le macchie del sensore; i correttivi vanno nello stesso punto del
  sensore anche sulle foto verticali, con la sorgente cercata di nuovo su ciascuna.

Le rimozioni sono della foto: non passano con «Applica alla scena», con gli stili, i
predefiniti o i `.apestyle`. Le immagini esportate le contengono; gli XMP no, e il piano di
export lo dice. I riempimenti stanno in `~/.local/share/autophotoedit/retouch/`, non sono cache e
non si cancellano mai da soli.

### La pipeline

```sh
# sviluppa un RAW con i parametri predefiniti
uv run ape render scatto.ARW -o sviluppato.jpg

# con parametri espliciti
uv run ape render scatto.ARW --params parametri.json -o sviluppato.tif --format tiff16

# scrivi un file di parametri predefiniti da cui partire
uv run ape params --output parametri.json

# mostra i metadati letti dal RAW
uv run ape info scatto.ARW
```

Il comando `render` **non scrive mai** dentro la cartella che contiene il RAW di partenza se
quella cartella è dichiarata come sorgente di un progetto; in generale il percorso di
destinazione passa sempre da `assert_outside_source` (§2.3).

## Test

```sh
uv run pytest                     # tutti i test che non richiedono fixture
uv run pytest -m fixtures         # test su RAW reali (vedi tests/fixtures/README.md)
```

I test della Fase 1 (`docs/SPEC.md` §13) sono:

| # | Test | File |
|---|---|---|
| 1 | Invarianza di risoluzione | `tests/test_resolution_invariance.py` |
| 2 | Round-trip dei parametri | `tests/test_params_roundtrip.py` |
| 4 | Determinismo | `tests/test_determinism.py` |
| 5 | Neutralità (color checker) | `tests/test_neutrality.py` |
| 8 | Non distruttività | `tests/test_non_destructive.py` |

Quelli della Fase 2:

| # | Test | File |
|---|---|---|
| 6 | Resume dei job dopo un crash | `tests/test_job_resume.py` |
| 12 | Import idempotente | `tests/test_import_idempotent.py` |
| — | Superficie HTTP e WebSocket | `tests/test_api.py` |

Quelli della Fase 3:

| # | Test | File |
|---|---|---|
| 19 | Finestra applicativa e istanza singola | `tests/test_window_launcher.py` |
| 19 | Manifest, service worker e freschezza di `/api` | `tests/test_pwa_shell.py` |
| — | Slider e stringhe della UI allineati a `EditParams` e a `it.ts` | `tests/test_frontend_contract.py` |

Quelli della Fase 4:

| # | Test | File |
|---|---|---|
| 9 | Cernita su 100 foto con 10 scarti noti | `tests/test_culling.py` |
| 10 | Le decisioni dell'utente sopravvivono a ricalcoli e cambi di impostazioni | `tests/test_culling_select.py`, `tests/test_culling_api.py` |
| 17 (parte) | Un bracketing è una proposta di fusione, non una raffica, e non viene scartato | `tests/test_merge_detect.py` |
| — | Selezione pura, modalità a obiettivo, latenza su 2000 foto | `tests/test_culling_select.py`, `tests/test_culling_api.py` |
| — | Anteprima incorporata, orientamento, punto di fuoco, migrazione dello schema | `tests/test_embedded_preview.py` |

Il test 9 usa un set sintetico (`tests/culling_scenes.py`): gli scarti sono «noti» per
costruzione. I test marcati `fixtures` lo completano sui 24 ARW veri, nessuno dei quali deve
risultare difettoso.

Quelli della Fase 6:

| # | Test | File |
|---|---|---|
| 3 | Convergenza dell'inversione (sintetico e ARW reale) | `tests/test_style_invert.py` |
| 15 | Round-trip del profilo `.apestyle` in un catalogo vuoto, RAW assenti | `tests/test_style_portable.py` |
| — | Modello (k-NN, ridge, boosting, scelta per parametro), coerenza, accoppiamento | `tests/test_style_model.py` |
| — | Profilo da cartelle, accoppiamento manuale, predizione applicata, modifiche utente preservate, §2 | `tests/test_style_flow.py` (`fixtures`, `slow`) |

Quelli delle Fasi 9 e 10:

| # | Test | File |
|---|---|---|
| 16 | Storico e snapshot: 20 modifiche, ripristino, snapshot, rollback annullabile | `tests/test_history_snapshots.py` |
| 18 | Integrità dei modelli, rete assente | `tests/test_analysis.py` |
| 19 | Finestra applicativa, PWA | `tests/test_window_launcher.py`, `tests/test_pwa_shell.py` |
| — | Maschere, selezioni, segmentazione | `tests/test_masks.py`, `tests/test_masks_api.py`, `tests/test_segment.py` |
| — | Rimozione: correttivo, gomma, patch, esclusioni, API, copia punti | `tests/test_retouch.py`, `tests/test_retouch_fill.py`, `tests/test_retouch_exclusions.py`, `tests/test_retouch_api.py`, `tests/test_retouch_model.py` (col modello: `APE_TEST_MODELS`) |
| — | Log, Problemi, diagnostica, backup e restore | `tests/test_maintenance.py` |
| — | Quota della cache, «Svuota cache», proxy persi e rigenerati | `tests/test_cache.py` |
| — | `install.sh` / `uninstall.sh` in una HOME di prova | `tests/test_install.py` |
| — | Licenze delle dipendenze e `NOTICE.md` generato | `tests/test_notice.py` |

Quelli della Fase 11:

| # | Test | File |
|---|---|---|
| 17 | Bracketing rilevato e non scartato; rifiuto permanente | `tests/test_merge_detect.py` |
| 17 | HDR sintetico ΔE2000 < 2; residuo di allineamento | `tests/test_merge_hdr.py` |
| 17 | Focus stack sintetico SSIM > 0,95; mappa di copertura | `tests/test_merge_focus.py` |
| 17 | Panoramica che non si cuce: motivo leggibile, gruppo `failed`, nessuna derivata | `tests/test_merge_panorama.py`, `tests/test_merge_flow.py`, `tests/test_merge_problems.py` |
| 17 | §2: hash dei RAW invariati, intermedio rigenerato identico | `tests/test_merge_flow.py` |
| — | Focus stack e panoramiche (anche a raffica) rilevati, raffiche no | `tests/test_merge_detect_more.py` |
| — | Rilevamento e scelta dei fotogrammi sui file veri di `tests/fixtures/fase11` | `tests/test_merge_real.py` (`fixtures`) |
| — | API delle fusioni: anteprima, accetta, sorgenti, annulla | `tests/test_merge_api.py` |

I due test end-to-end del launcher avviano il programma vero in un sottoprocesso e sono marcati
`slow`; `uv run pytest -m "not slow"` li salta.

## Com'è fatta la pipeline

Due spazi attraversano l'elaborazione e non vanno confusi:

- **scene-referred** — lineare, primari Rec.2020, bianco D65, float32, saturazione
  del sensore a 1.0. I valori sopra 1.0 sono legittimi e portano le alte luci
  speculari. Qui avviene tutto ciò che è fisico: bilanciamento del bianco,
  esposizione, recupero delle alte luci, riduzione del rumore.
- **display-referred** — ancora Rec.2020, ma codificato percettivamente con una
  potenza 1/2.4 e limitato a [0, 1]. Qui avviene tutto ciò che è percettivo:
  ombre/luci, curve, colore, contrasto locale, nitidezza.

Il **tone mapping sigmoid** è l'unico ponte fra i due. È costruito con due
segmenti che si incontrano nel pivot, ciascuno con pendenza esatta nel pivot e
asintoto al nero e al bianco: `contrast = 1.0` riproduce la pendenza che avrebbe
un trasferimento lineare→display, quindi «nessuna curva a S», e il default 1.2
aggiunge la S che porta una resa fotografica normale.

Per la stabilità delle tinte la croma viene compressa verso l'asse acromatico
*prima* della sigmoid e riespansa dopo, con l'espansione volutamente più debole
della compressione (approccio AgX): è la differenza fra le due che produce la
desaturazione filmica delle alte luci. Il parametro `chroma_preservation`
(default 0.4) regola entrambe; a 0 si ottiene la sigmoid per canale classica,
con lo slittamento di tinta che comporta.

**Ogni raggio è una frazione del lato lungo, mai un numero di pixel.** È questo
che rende le operazioni indipendenti dalla risoluzione: gli stessi `EditParams`
sul proxy e sul full-res descrivono lo stesso intorno fisico, e il test 1 di §13
lo misura.

## Prestazioni misurate

Misure, non stime (§26). Macchina: Intel i5-13500H, 12 core / 16 thread, 16 GB.

### Sviluppo di una foto, 1 core, 24 MP

| Caso | Tempo | Target §12 |
|---|---|---|
| Render completo, parametri neutri con correzione obiettivo (Fase 8) | 2.85 s (era 3.41) | 3.5 s ✓ |
| Render completo, un look reale dell'utente (Fase 8, a bande) | 5.0 s | 3.5 s ✗ |
| Export JPEG completo di un look reale: decodifica 1,6 + render 5,0 + codifica 0,7 | 7.3 s | 3.5 s ✗ |
| Picco di memoria di un export a piena risoluzione | 1.29 GB | < 1.5 GB/worker ✓ |
| Decodifica + proxy 2048 px (il lavoro di un job `proxy`) | 1.08 s | 1.2 s ✓ |
| **Export di 104 foto reali, 14 worker (7 export insieme)** | **2.03 s a foto** (≈ 34 min per 1000) | 5 min per 1000 ✗ |

L'export resta lontano dai target e va detto con i numeri: il render è limitato dalla banda di
memoria (più export insieme non aiutano: 5 → 8 contemporanei guadagnano il 12% e iniziano a
usare lo swap) e la decodifica LibRaw da sola costa 1,6 s.

### Anteprima interattiva (§10), misurata via HTTP sul server reale

| Azione | Tempo | Target §10 |
|---|---|---|
| Trascinamento di uno slider *dopo* il tone mapping (saturazione), 1024 px | 79 ms | < 150 ms ✓ |
| Trascinamento di uno slider *prima* del tone mapping (esposizione), 1024 px | 110–117 ms | < 150 ms ✓ |
| Clarity (filtro spaziale), 1024 px | 77–133 ms | < 150 ms ✓ |
| Rilascio: stessa modifica a 2048 px | 288–301 ms | — |
| Prima apertura di una foto (decodifica inclusa) | 1,1 s | — |

§10 chiede di renderizzare «a 1024 px durante il trascinamento, a 2048 px al rilascio»: la
sorgente viene decodificata **alla dimensione richiesta**, quindi il trascinamento fa passare
tutta la pipeline a 1024 px anziché ridimensionare alla fine. Un quarto dei pixel in ogni stadio
è un quarto del lavoro e un quarto della memoria.

Quella scorciatoia è ciò che rende lo slider immediato, e la domanda che si porta dietro è se
cambi l'immagine. Misurata su un ARW reale a 1024 px, con esposizione, contrasto, saturazione,
vividezza e chiarezza attivi, la distanza fra l'anteprima servita dalla UI e l'immagine che
l'export scrive partendo dal sensore intero è **0,96 dE2000 medio** (tolleranza 1,5 del test 1).
Il massimo locale arriva a 24 dE2000 su qualche bordo in aliasing: lì le due strade campionano il
sensore in modo diverso, ed è il prezzo dichiarato della scorciatoia.
`tests/test_api.py::test_the_preview_the_viewer_shows_is_the_image_an_export_writes` lo misura.

### Memoria del processo server

| Stato | RSS | Budget §26 |
|---|---|---|
| A riposo, nessuna foto aperta | 99 MB | < 400 MB ✓ |
| Una foto aperta a 1024 px | 241 MB | |
| La stessa foto a 1024 e 2048 px | 385 MB | |

La cache per stadio tiene quattro checkpoint (`render.CHECKPOINTS`), non tutti e tredici gli
stadi: tenerli tutti costerebbe 300 MB per foto, cioè l'intero budget. Le sorgenti che nessuno
guarda da cinque minuti vengono lasciate cadere al primo accesso successivo — senza timer di
background, perché §26 chiede ~0% di CPU a riposo.

### Import di un batch, 14 worker

| Operazione | Misurato su 96 foto | Estrapolato a 1000 | Target §12 |
|---|---|---|---|
| Scansione e catalogo | 0,8 ms/foto | 0,8 s | — |
| Anteprime 2048 px | 231 ms/foto | 231 s (3,9 min) | 4 min ✓ |

L'estrapolazione è lineare da un campione piccolo e va riverificata su 1000 file veri. L'analisi
di §12 (embedding, feature di scena) non è ancora inclusa perché arriva con la Fase 5.

### Cernita (§7.7), 14 worker

| Operazione | Misurato | Target |
|---|---|---|
| Estrazione anteprima incorporata, 1 core | 15 ms mediana, 24 ms max | ≤ 60 ms ✓ (§12) |
| Punteggio tecnico (nitidezza, mosso, esposizione), 1 core | 54 ms mediana, 70 ms max | ≤ 80 ms ✓ (§12) |
| Analisi di 2000 foto, coda e worker veri | 65,5 s | ≤ 180 s ✓ (§7.7) |
| Raggruppamento + selezione di 2000 foto | 0,4 s | — |
| Slider o cambio di modalità su 2000 foto, via HTTP | 63–65 ms mediana, ~110 ms p90 | < 100 ms ✓ (§14) |

Le 2000 righe puntano a turno ai 24 file di `tests/fixtures/` (48 GB di copie non servono a
misurare l'analisi): i file stanno nella cache di pagina, quindi il numero misura la CPU e non un
lettore di schede lento. Volti ed estetica non sono misurabili finché i loro modelli non ci sono.

**Il pinning dei thread è metà di quel numero.** OpenMP legge `OMP_NUM_THREADS` una volta sola,
quando il suo runtime si inizializza — cioè quando LibRaw viene caricata, non quando il worker
prende il primo job. Impostando le variabili dopo l'import si ottengono 14 processi da 16 thread
l'uno su 12 core: 397 ms/foto invece di 231. `backend/ape/jobs/_preload.py` esiste solo per fare
le due cose nell'ordine giusto.

### Stili (§8, §14 Fase 6), sulle 65 coppie RAW + Lightroom dell'utente

| Operazione | Misurato | Target |
|---|---|---|
| Accoppiamento di 104 RAW con 65 JPEG rinominati | 0,07 s, 65/65 dall'XMP | — |
| Inversione di una coppia, 1 core | 11,2 s mediana, 17,6 max; ≤ 398 render | 10–25 s, ≤ 400 render ✓ (§8.2) |
| Residuo dell'inversione (ΔE medio sulla coppia) | 3,4 a scala di colore, 4,0 per pixel | — |
| Predizione di una foto | 0,25 ms | ≤ 5 ms ✓ (§12) |
| 30 coppie → 10 foto non viste, split canonico | ΔE medio 5,76, peggiore 11,91 | < 6, nessuna > 12 ✓ |
| Idem, 50 split casuali | media < 6 in 27/50, nessuna > 12 in 40/50, entrambi 25/50 | ✗ in metà degli split |

Per rimisurare:

```sh
uv run python tests/bench/bench_style.py RAW_DIR EDITATE_DIR --splits 50
uv run python tests/bench/bench_render.py                # 24 MP sintetici, 1 core
uv run python tests/bench/bench_render.py scatto.ARW     # un file reale
uv run python tests/bench/bench_catalog.py --repeat 4    # import + anteprime, pool intero
uv run python tests/bench/bench_culling.py               # cernita di 2000 foto, pool intero
```

## Dove Sony mette il grigio medio

LibRaw normalizza la saturazione del sensore a 1.0; il tone mapping è ancorato a un grigio medio
scene-linear di 0.1845. La distanza fra i due decide se uno sviluppo neutro esce alla luminosità
che il fotografo ha misurato, ed è un numero **misurato per corpo macchina**, non dedotto.

ISO 12232 dice +0.53 EV in teoria. Chiedendolo alla fotocamera — ogni ARW contiene il JPEG che la
macchina stessa avrebbe prodotto, e il livello di sensore che rende a grigio medio è la risposta
di Sony — l'A7 III risponde **+1.09 EV**: mediana su 21 scatti da ISO 100 a 1600, deviazione
standard 0.058 EV. Con il valore da norma uno sviluppo neutro usciva 7,5 L\* più scuro del JPEG
della macchina; con quello misurato la differenza media è 0,3 L\*.

```sh
uv run python tests/bench/measure_baseline_exposure.py   # rimisura, su qualunque cartella
```

I corpi non ancora misurati usano il numero da norma: sbagliato di una quantità nota, che è
meglio di sbagliato di una quantità ignota. L'A7 II non è ancora stato misurato perché fra i file
di esempio non ce n'è nessuno.

## File RAW di esempio

I RAW non vengono committati: `tests/fixtures/` è vuota nel repository e i test che la
richiedono sono marcati `fixtures` e vengono saltati, non falliti. Vedi
[`tests/fixtures/README.md`](tests/fixtures/README.md) per quali scatti servono e perché.

`tests/fixtures/fase11/` (46 ARW, non committata) contiene i bracketing e le panoramiche
scattati dall'utente per la Fase 11; `tests/test_merge_real.py` ci riverifica il rilevamento.

`tests/rawEdited/` (3,2 GB, non committata) contiene un evento reale: 104 RAW in `raw/` e le 65
foto consegnate, editate in Lightroom Classic, in `edited/`. È ciò su cui la Fase 6 è stata
misurata (`tests/bench/bench_style.py`).

## Licenza

GPL-3.0-or-later. Vedi [`LICENSE`](LICENSE), [`NOTICE.md`](NOTICE.md) per le licenze delle
dipendenze e [`models/LICENSES.md`](models/LICENSES.md) per quelle dei modelli ONNX scaricabili.

La licenza è una conseguenza tecnica, non una preferenza: `pyexiv2` è GPL-3.0. Vedi `docs/SPEC.md`
§24 per il ragionamento completo e per la nota su come mantenere reversibile quella scelta.
