#!/usr/bin/env bash
# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Installs autoPhotoEdit for the current user, as a desktop application
# (docs/SPEC.md section 18): the launcher in ~/.local/bin, the icons in the
# hicolor theme, the menu entry, whose StartupWMClass is the identity the
# browser window ends up with -- which is what puts the right icon and name in
# the dock instead of Chromium's, or another local app's.
#
# Idempotent: running it again updates what it installed and nothing else.
# Never uses sudo and never installs system packages: when something is
# missing it prints the exact command and stops.
#
# For tests and development only:
#   APE_INSTALL_SKIP_SYNC=1      do not run `uv sync` (the .venv must exist)
#   APE_INSTALL_SKIP_FRONTEND=1  do not build the frontend (backend/ape/static must exist)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/autophotoedit"
APPS_DIR="$DATA_HOME/applications"
DESKTOP="$APPS_DIR/autophotoedit.desktop"
ICONS_DIR="$DATA_HOME/icons/hicolor"
MARK="# autoPhotoEdit launcher, written by install.sh"

say() { printf '%s\n' "$*"; }
stop() {
  printf 'errore: %s\n' "$1" >&2
  shift
  for line in "$@"; do printf '  %s\n' "$line" >&2; done
  exit 1
}

# --- 1. what must already be on the machine -----------------------------------
command -v uv >/dev/null 2>&1 || stop "manca uv, che installa le dipendenze Python." \
  "Installalo con:" "curl -LsSf https://astral.sh/uv/install.sh | sh" "poi riapri il terminale e rilancia ./install.sh"

if ! uv python find 3.12 >/dev/null 2>&1; then
  stop "manca Python 3.12." "Installalo con:" "sudo apt install python3.12 python3.12-venv"
fi

# --- 2. the Python environment ------------------------------------------------
cd "$REPO"
if [ "${APE_INSTALL_SKIP_SYNC:-0}" != "1" ]; then
  say "Installo le dipendenze Python (uv sync)…"
  uv sync --all-extras --python 3.12
fi
PYTHON="$REPO/.venv/bin/python"
[ -x "$REPO/.venv/bin/autophotoedit" ] || stop "l'ambiente $REPO/.venv non contiene autophotoedit." \
  "Rilancia ./install.sh senza APE_INSTALL_SKIP_SYNC."

# The wheels carry LibRaw, exiv2 and lensfun on Linux x86_64; where they do
# not, uv builds them and the system libraries are needed.
if ! "$PYTHON" -c "import rawpy, pyexiv2, lensfunpy" >/dev/null 2>&1; then
  stop "le librerie native per RAW, EXIF e profili obiettivo non si caricano." \
    "Installa le librerie di sistema con:" \
    "sudo apt install libraw-dev libexiv2-dev liblensfun-dev liblensfun-data-v1 pkg-config build-essential" \
    "poi rilancia ./install.sh"
fi

# --- 3. the interface ---------------------------------------------------------
if [ "${APE_INSTALL_SKIP_FRONTEND:-0}" != "1" ] && command -v npm >/dev/null 2>&1; then
  say "Costruisco l'interfaccia (npm)…"
  (cd frontend && npm ci --no-audit --no-fund && npm run build)
elif [ -f "$REPO/backend/ape/static/index.html" ]; then
  say "Uso l'interfaccia già costruita in backend/ape/static/ (Node non serve)."
else
  stop "manca l'interfaccia costruita e Node non è installato." \
    "Usa un pacchetto di release (contiene backend/ape/static/) oppure installa Node:" \
    "sudo apt install nodejs npm"
fi
[ -f "$REPO/backend/ape/static/manifest.webmanifest" ] || stop "la build dell'interfaccia è incompleta (manca il manifest)."

# --- 4. launcher, icons, menu entry -------------------------------------------
mkdir -p "$BIN_DIR" "$APPS_DIR"
if [ -e "$LAUNCHER" ] && ! grep -qF "$MARK" "$LAUNCHER" 2>/dev/null; then
  stop "$LAUNCHER esiste già e non l'ha scritto install.sh." "Spostalo o rimuovilo, poi rilancia."
fi
cat >"$LAUNCHER.tmp" <<EOF
#!/bin/sh
$MARK
exec "$REPO/.venv/bin/autophotoedit" "\$@"
EOF
chmod 755 "$LAUNCHER.tmp"
mv -f "$LAUNCHER.tmp" "$LAUNCHER"

for size in 48 128 256; do
  install -D -m 644 "$REPO/packaging/icons/autophotoedit-$size.png" \
    "$ICONS_DIR/${size}x${size}/apps/autophotoedit.png"
done
install -D -m 644 "$REPO/packaging/icons/autophotoedit.svg" "$ICONS_DIR/scalable/apps/autophotoedit.svg"

# The identity of the window (backend/ape/launcher.py, window_class). Not a name
# of our choosing: under Wayland Chromium derives it from the window's URL and
# the browser installed, so it is asked of the code that opens the window.
WM_CLASS="$("$PYTHON" -c "from ape.launcher import window_class; print(window_class())" 2>/dev/null || true)"
[ -n "$WM_CLASS" ] || WM_CLASS="chrome-127.0.0.1__autophotoedit-Default"

# Written and validated aside -- the validator wants the .desktop name -- then
# moved in place, so the menu never sees half an entry.
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
cat >"$STAGING/autophotoedit.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.5
Name=autoPhotoEdit
GenericName=Post-produzione RAW
Comment=Post-produzione automatica di RAW Sony, in locale e non distruttiva
Exec=$LAUNCHER
Icon=autophotoedit
Terminal=false
Categories=Graphics;Photography;
Keywords=RAW;ARW;Sony;foto;sviluppo;fotografia;
StartupNotify=true
StartupWMClass=$WM_CLASS
EOF
if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$STAGING/autophotoedit.desktop" ||
    stop "la voce di menu generata non è valida (vedi sopra)."
fi
install -m 644 "$STAGING/autophotoedit.desktop" "$DESKTOP.tmp"
mv -f "$DESKTOP.tmp" "$DESKTOP"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$APPS_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t -f "$ICONS_DIR" 2>/dev/null || true

# --- 5. what was installed, and where -----------------------------------------
BROWSER="$("$PYTHON" -c "from ape.launcher import find_browser; b = find_browser(); print(' '.join(b) if b else '')" 2>/dev/null || true)"
say ""
say "autoPhotoEdit installato."
say "  programma:     $LAUNCHER  →  $REPO/.venv/bin/autophotoedit"
say "  voce di menu:  $DESKTOP"
say "  finestra/dock: $WM_CLASS"
say "  icone:         $ICONS_DIR/{48x48,128x128,256x256,scalable}/apps/"
say "  i tuoi dati:   ${XDG_DATA_HOME:-$HOME/.local/share}/autophotoedit/ (creata al primo avvio)"
if [ -n "$BROWSER" ]; then
  say "  finestra:      $BROWSER in modalità applicazione"
else
  say "  finestra:      nessun browser Chromium trovato: l'app si aprirà in una scheda del browser"
  say "                 predefinito. Per la finestra dedicata installa Chromium (sudo snap install chromium)."
fi
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "  nota:          $BIN_DIR non è nel PATH: dal terminale usa $LAUNCHER" ;;
esac
say ""
say "Aprilo dal menu delle applicazioni («autoPhotoEdit») o con: autophotoedit"
say "Per disinstallarlo: ./uninstall.sh (i tuoi dati restano, salvo conferma esplicita)."
