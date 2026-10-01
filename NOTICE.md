# Dipendenze e licenze

autoPhotoEdit è distribuito sotto **GPL-3.0-or-later** (vedi [`LICENSE`](LICENSE)).

Non è una preferenza, è una conseguenza tecnica: `pyexiv2` è GPL-3.0 ed `exiv2`
è GPL-2.0-or-later. Finché il programma si collega a exiv2, qualunque licenza
permissiva sarebbe incompatibile. Il ragionamento completo è in `docs/SPEC.md` §24.

> **Nota per il futuro.** L'unico ostacolo a una licenza permissiva è pyexiv2,
> sostituibile con `exiftool` via subprocess o con una scrittura XMP diretta.
> Perché quella sostituzione resti un lavoro di poche ore, tutto l'accesso a
> exiv2 è confinato dietro `backend/ape/raw/metadata.py` e `backend/ape/export/`.
> Il vincolo è verificato da `tests/test_licence_isolation.py`: un `import
> pyexiv2` altrove rompe la build.

## Dipendenze Python

| Pacchetto | Licenza | Compatibile con GPL-3.0 | Note |
|---|---|---|---|
| pyexiv2 | GPL-3.0 | sì (impone la licenza) | confinato in `raw/metadata.py` e `export/` |
| libexiv2 (sistema) | GPL-2.0-or-later | sì | usata tramite pyexiv2 |
| rawpy | MIT | sì | binding di LibRaw |
| LibRaw (sistema) | LGPL-2.1 / CDDL-1.0 | sì | |
| numpy | BSD-3-Clause | sì | |
| scipy | BSD-3-Clause | sì | |
| opencv-python-headless | Apache-2.0 | sì | wheel senza moduli contrib |
| colour-science | BSD-3-Clause | sì | |
| pillow | MIT-CMU | sì | usata solo per JPEG e per la lettura |
| pydantic / pydantic-settings | MIT | sì | |
| lensfunpy | MIT | sì | fase 5, extra `lens` |
| lensfun (sistema) | LGPL-3.0 | sì | fase 5 |
| onnxruntime | MIT | sì | fase 4+, extra `ml`, solo CPU |
| fastapi | MIT | sì | fase 2, extra `server` |
| uvicorn | BSD-3-Clause | sì | fase 2 |
| sqlalchemy | MIT | sì | fase 2 |
| alembic | MIT | sì | fase 2 |
| pytest | MIT | sì | solo sviluppo |
| ruff | MIT | sì | solo sviluppo |

## Dipendenze del frontend (fase 3)

| Pacchetto | Licenza |
|---|---|
| React / React DOM | MIT |
| React Router | MIT |
| Vite / @vitejs/plugin-react | MIT |
| TypeScript | Apache-2.0 |
| Tailwind CSS / @tailwindcss/vite | MIT |
| Radix UI (`react-slider`, `react-dialog`, `react-switch`) | MIT |
| class-variance-authority, clsx, tailwind-merge | MIT |
| lucide-react | ISC |
| TanStack Query | MIT |
| Zustand | MIT |
| vite-plugin-pwa / Workbox | MIT |
| openapi-typescript | MIT |

I componenti sono scritti a mano nello stile di **shadcn/ui** (MIT) sopra le primitive Radix:
shadcn/ui è un ricettario da copiare, non un pacchetto da installare, e non compare fra le
dipendenze.

## Modelli ONNX

I modelli **non** sono ridistribuiti con il programma: vengono scaricati su
richiesta esplicita dell'utente, quindi le loro licenze non si propagano a
questo progetto. Sono comunque dichiarate in
[`models/LICENSES.md`](models/LICENSES.md), perché chi li scarica è l'utente.

## Verifica

La tabella qui sopra spiega le dipendenze principali; l'**elenco completo** qui sotto è
**generato** da `packaging/notice.py` leggendo i metadati dei pacchetti installati (dipendenze ed
extra del progetto seguiti nelle loro dipendenze; per il frontend le voci non di sviluppo di
`frontend/package-lock.json` più il runtime di Workbox incluso nel service worker).

`tests/test_notice.py` rompe la build se un pacchetto distribuito ha una licenza non compatibile
con la GPL-3.0 (o sconosciuta), se uno strumento di sviluppo non ha una licenza aperta nota, o se
questo elenco non corrisponde più alle dipendenze. Dopo un cambio di dipendenze:

```sh
uv run python packaging/notice.py --write
```

## Elenco completo

## Elenco completo

<!-- inizio: generato da packaging/notice.py, non modificare a mano -->

### Pacchetti Python distribuiti (dipendenze ed extra, con le loro dipendenze)

| Pacchetto | Licenza |
|---|---|
| alembic | MIT |
| annotated-doc | MIT |
| annotated-types | MIT |
| anyio | MIT |
| click | BSD-3-Clause |
| colour-science | BSD-3-Clause |
| fastapi | MIT |
| flatbuffers | Apache-2.0 |
| greenlet | MIT AND PSF-2.0 |
| h11 | MIT |
| httptools | MIT |
| idna | BSD-3-Clause |
| lensfunpy | MIT |
| Mako | MIT |
| MarkupSafe | BSD-3-Clause |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | MIT |
| opencv-python-headless | Apache-2.0 |
| packaging | Apache-2.0 OR BSD-2-Clause |
| pillow | MIT-CMU |
| protobuf | BSD-3-Clause |
| pydantic | MIT |
| pydantic_core | MIT |
| pydantic-settings | MIT |
| pyexiv2 | GPL-3.0 |
| python-dotenv | BSD-3-Clause |
| PyYAML | MIT |
| rawpy | MIT |
| scipy | BSD-3-Clause |
| SQLAlchemy | MIT |
| starlette | BSD-3-Clause |
| typing_extensions | PSF-2.0 |
| typing-inspection | MIT |
| uvicorn | BSD-3-Clause |
| uvloop | MIT |
| watchfiles | MIT |
| websockets | BSD-3-Clause |

### Pacchetti del frontend inclusi nella build

| Pacchetto | Licenza |
|---|---|
| @radix-ui/number | MIT |
| @radix-ui/primitive | MIT |
| @radix-ui/react-collection | MIT |
| @radix-ui/react-compose-refs | MIT |
| @radix-ui/react-context | MIT |
| @radix-ui/react-dialog | MIT |
| @radix-ui/react-direction | MIT |
| @radix-ui/react-dismissable-layer | MIT |
| @radix-ui/react-focus-guards | MIT |
| @radix-ui/react-focus-scope | MIT |
| @radix-ui/react-id | MIT |
| @radix-ui/react-portal | MIT |
| @radix-ui/react-presence | MIT |
| @radix-ui/react-primitive | MIT |
| @radix-ui/react-slider | MIT |
| @radix-ui/react-slot | MIT |
| @radix-ui/react-switch | MIT |
| @radix-ui/react-use-callback-ref | MIT |
| @radix-ui/react-use-controllable-state | MIT |
| @radix-ui/react-use-effect-event | MIT |
| @radix-ui/react-use-layout-effect | MIT |
| @radix-ui/react-use-previous | MIT |
| @radix-ui/react-use-size | MIT |
| @remix-run/router | MIT |
| @tanstack/query-core | MIT |
| @tanstack/react-query | MIT |
| @types/prop-types | MIT |
| @types/react | MIT |
| @types/react-dom | MIT |
| aria-hidden | MIT |
| class-variance-authority | Apache-2.0 |
| clsx | MIT |
| csstype | MIT |
| detect-node-es | MIT |
| get-nonce | MIT |
| js-tokens | MIT |
| loose-envify | MIT |
| lucide-react | ISC |
| react | MIT |
| react-dom | MIT |
| react-remove-scroll | MIT |
| react-remove-scroll-bar | MIT |
| react-router | MIT |
| react-router-dom | MIT |
| react-style-singleton | MIT |
| scheduler | MIT |
| tailwind-merge | MIT |
| tslib | 0BSD |
| use-callback-ref | MIT |
| use-sidecar | MIT |
| workbox-background-sync | MIT |
| workbox-broadcast-update | MIT |
| workbox-build | MIT |
| workbox-cacheable-response | MIT |
| workbox-core | MIT |
| workbox-expiration | MIT |
| workbox-google-analytics | MIT |
| workbox-navigation-preload | MIT |
| workbox-precaching | MIT |
| workbox-range-requests | MIT |
| workbox-recipes | MIT |
| workbox-routing | MIT |
| workbox-strategies | MIT |
| workbox-streams | MIT |
| workbox-sw | MIT |
| workbox-window | MIT |
| zustand | MIT |

<!-- fine della parte generata -->
