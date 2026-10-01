# Licenze dei modelli ONNX

I modelli **non sono inclusi nella repository**. Vengono scaricati alla prima
attivazione della funzione che li richiede (`docs/SPEC.md` §17), verificati con
SHA-256 e salvati in `$XDG_DATA_HOME/autophotoedit/models/`. Chi non attiva né
volti, né estetica, né segmentazione non scarica nulla.

Poiché non sono ridistribuiti, le loro licenze non si propagano ad
autoPhotoEdit. Vanno comunque dichiarate: chi li scarica è l'utente, ed è
l'utente a doverne valutare l'idoneità per il proprio uso.

| Modello | Funzione | Fase | Licenza | Peso |
|---|---|---|---|---|
| CLIP ViT-B/32, encoder visivo, ONNX int8 (`Xenova/clip-vit-base-patch32`) | embedding di scena (§8.3) | 5 | MIT | ~90 MB |
| YuNet (OpenCV Zoo) | rilevamento volti (§7.3) | 4 | MIT | ~0.3 MB |
| NIMA MobileNet ONNX | punteggio estetico (§7.3) | 4 | ⚠️ vedi sotto | ~14 MB |
| MediaPipe selfie multiclass 256×256, ONNX (`senty-au/selfie_multiclass_256x256-ONNX`) | maschere «Persone» e «Pelle» (§6.3) | 9 | Apache-2.0 | ~16,5 MB |
| SkySeg U²-Net FP16 (`voyagerfromeast/skyseg`) | maschera «Cielo» (§6.3) | 9 | ⚠️ MIT, vedi sotto | ~88 MB |
| LaMa int8, OpenCV Zoo (`opencv/inpainting_lama`) | gomma magica, motore «IA» (`docs/SPEC_rimozione.md` §5) | Rimozione | ⚠️ Apache-2.0, vedi sotto | ~92,6 MB |

Totale se tutto è scaricato: ~286 MB, entro i 350 di §26.

## ⚠️ LaMa e il dataset Places2

- **Codice** (`advimman/lama`): Apache-2.0. **File ONNX**: la cartella `opencv/inpainting_lama`
  dichiara Apache-2.0 per tutti i suoi file; è la quantizzazione int8 di OpenCV di
  `Carve/LaMa-ONNX` (`lama_fp32.onnx`), una conversione di terzi. Nessuna clausola non
  commerciale sui pesi.
- **Dati**: il modello migliore di LaMa è addestrato su Places2, i cui termini d'uso dicono
  «You will use the data only for non-commercial research and educational purposes»
  (places2.csail.mit.edu). Se questi termini passino ai pesi addestrati è una questione
  giuridica aperta; la UI lo dice accanto al download (`places2_terms`), valuta tu l'idoneità
  per un uso commerciale.
- Scartati: MI-GAN (MIT anche sui pesi, 28 MB, più veloce, ma sugli oggetti grandi dei file
  dell'utente inventava contenuti), LaMa FP32 di Carve (208 MB, oltre i 120 per modello), i
  modelli addestrati su CelebA/CelebA-HQ o FFHQ (dati di sola ricerca non commerciale).

## ⚠️ NIMA e il dataset AVA

I pesi NIMA pubblici derivano dal dataset **AVA**, il cui uso è dichiarato
*research-only*. Di conseguenza:

- il toggle «Estetica» resta **disattivato di default**;
- attivandolo, la UI mostra la nota: *«modello addestrato su un dataset di sola
  ricerca; valuta tu l'idoneità per un uso commerciale»*;
- la conferma dell'utente è registrata in `Setting.aesthetic_enabled_ack`.

## Segmentazione: perché questi due, e il dubbio su SkySeg

U²-Netp e BiRefNet, previsti da §17, segmentano l'*oggetto saliente*: non
distinguono cielo, persone e pelle, che sono i soggetti di §6.3. Scelti al
loro posto (revisioni e SHA-256 fissati in
`backend/ape/models_registry.py`):

- **MediaPipe selfie multiclass** (Google, Apache-2.0), convertito in ONNX da
  terzi. Sei classi: sfondo, capelli, pelle del corpo, pelle del viso, vestiti,
  altro. «Persone» = 1 − P(sfondo); «Pelle» = P(pelle del corpo) + P(pelle del
  viso). Addestrato su ritratti: su figure piccole in un paesaggio rende meno.
- **SkySeg** (U²-Net addestrato sul cielo), pubblicato con licenza MIT.
  **Dubbio dichiarato:** chi lo pubblica non dice su quali dati è stato
  addestrato, e il file FP16 è una conversione di terzi. La UI lo mostra accanto
  al pulsante di download (`skyseg_provenance`); valuta tu l'idoneità per un uso
  commerciale.

Il risultato della segmentazione è una maschera a 8 bit salvata nella cartella
delle maschere, per foto e soggetto, e calcolata sul proxy: non si ricalcola a
ogni render.

## Modelli vietati

**MODNet** e **RMBG-1.4** non vanno usati, anche se tecnicamente adatti alla
segmentazione: hanno licenze non commerciali. Se serve un'alternativa si sceglie
tra licenze permissive e la si documenta qui.

## Budget (§26)

- ≤ 120 MB per modello, ≤ 350 MB in totale;
- caricamento pigro alla prima richiesta, scarico dopo 5 minuti di inattività;
- se si trova un embedding di qualità equivalente a CLIP ViT-B/32 sotto i 40 MB,
  va usato quello e va riportato il confronto.
