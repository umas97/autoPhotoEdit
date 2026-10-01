# File di esempio per i test

Questa cartella è vuota di proposito: i RAW non vengono committati. I test che
li richiedono sono marcati `fixtures` e vengono **saltati** finché la cartella
resta vuota — non falliscono.

```sh
uv run pytest              # tutto ciò che non richiede fixture
uv run pytest -m fixtures  # solo i test sui file reali
```

## Cosa serve

### Per la Fase 1 — validazione della pipeline

Da 3 a 5 file `.ARW`, copiati qui (gli originali non vengono toccati: il
programma apre tutto in sola lettura, e `tests/test_non_destructive.py` lo
verifica). Servono scene **diverse fra loro**, perché ciascuna mette alla prova
una parte diversa della pipeline:

| File suggerito | Scena | Cosa verifica |
|---|---|---|
| `a7iii-esterno.ARW` | esterno, luce diurna piena | bilanciamento del bianco as-shot, resa dei verdi |
| `a7iii-controluce.ARW` | controluce con cielo bruciato | recupero delle alte luci, spalla della sigmoid |
| `a7iii-interno.ARW` | interno, luce artificiale calda | adattamento cromatico lontano da D65 |
| `a7ii-ombra.ARW` | sottoesposto / ombra, ISO alto | piede della sigmoid, riduzione del rumore |
| `a7ii-ritratto.ARW` | un volto in luce naturale | toni della pelle, protezione della vividezza |

Almeno uno da **A7 II** e uno da **A7 III**: le due matrici colore sono diverse
ed è proprio quel percorso che va verificato su entrambe.

### Per la Fase 6 — apprendimento dello stile

Le **versioni già post-prodotte** degli stessi scatti (JPEG o TIFF), con lo
stesso nome base: `a7iii-esterno.jpg` accanto a `a7iii-esterno.ARW`.
L'accoppiamento avviene per nome del file.

Se ne hai molte di più (30-100 coppie) non metterle qui: la Fase 6 le leggerà
da una cartella a scelta. Qui bastano le coppie dei file di test.

Le coppie su cui la Fase 6 è stata misurata stanno in `tests/rawEdited/` (non
committata): `raw/` con i 104 ARW di un evento, `edited/` con le 65 foto
consegnate, rinominate da Lightroom. L'accoppiamento non ha bisogno dei nomi:
legge il nome originale che Lightroom scrive nell'XMP.

```sh
uv run python tests/bench/bench_style.py tests/rawEdited/raw tests/rawEdited/edited
```

## Un numero che aspetta questi file

`backend/ape/raw/decode.py` ancora l'esposizione con `BASELINE_EXPOSURE_EV`,
derivato dalla norma ISO 12232 (grigio medio al 12.8% della saturazione del
sensore, cioè +0.53 EV). È un valore standard, non una stima: appena ci sono
degli ARW reali va **misurato** sulle Sony e confrontato, perché ogni costruttore
si tiene un margine leggermente diverso sopra il bianco.
