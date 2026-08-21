#!/usr/bin/env bash

set -e

echo "=== NVIDIA MuseGlimmer + Codex setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODEX_DIR="$HOME/.codex"
CONFIG="$CODEX_DIR/config.toml"
PROFILE="$CODEX_DIR/nvidia-muse.config.toml"
CATALOG="$CODEX_DIR/nvidia-muse-model.json"
PROXY="$SCRIPT_DIR/nvidia-codex-proxy"

MODEL="meta/muse-glimmer-30b"
BASE_URL="http://127.0.0.1:11435/v1"

# --------------------------------------------------
# Vérifications
# --------------------------------------------------

echo
echo "[0/6] Vérification des dépendances..."

if ! command -v go >/dev/null 2>&1; then
    echo "Erreur : Go n'est pas installé."
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Erreur : npm n'est pas installé."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Erreur : Python 3 est nécessaire."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/main.go" ]; then
    echo "Erreur : main.go introuvable dans :"
    echo "$SCRIPT_DIR"
    exit 1
fi

# --------------------------------------------------
# NVIDIA API KEY
# --------------------------------------------------

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo
    echo "NVIDIA_API_KEY n'est pas définie."
    echo
    read -rsp "Entre ta clé NVIDIA API : " NVIDIA_API_KEY
    echo
    export NVIDIA_API_KEY
fi

# --------------------------------------------------
# 1. Installer Codex
# --------------------------------------------------

echo
echo "[1/6] Installation de Codex..."

npm install -g @openai/codex

echo
codex --version || true

# --------------------------------------------------
# 2. Compiler le proxy
# --------------------------------------------------

echo
echo "[2/6] Compilation du proxy NVIDIA..."

cd "$SCRIPT_DIR"

go build -o "$PROXY" main.go

chmod +x "$PROXY"

echo "Proxy compilé : $PROXY"

# --------------------------------------------------
# 3. Préparer ~/.codex
# --------------------------------------------------

echo
echo "[3/6] Préparation de ~/.codex..."

mkdir -p "$CODEX_DIR"

touch "$CONFIG"

# --------------------------------------------------
# Nettoyage ancien système de profil
#
# Supprime :
#
# profile = "nvidia-muse"
#
# et :
#
# [profiles.nvidia-muse]
# ...
#
# Le profil est maintenant dans son propre fichier.
# --------------------------------------------------

python3 - "$CONFIG" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])

text = path.read_text() if path.exists() else ""

# Supprimer un ancien :
# profile = "..."
text = re.sub(
    r'(?m)^[ \t]*profile[ \t]*=[ \t]*["\'][^"\']*["\'][ \t]*\n?',
    '',
    text
)

# Supprimer l'ancienne table :
# [profiles.nvidia-muse]
#
# jusqu'à la prochaine table TOML.
lines = text.splitlines()
out = []
skip = False

for line in lines:

    stripped = line.strip()

    if stripped == "[profiles.nvidia-muse]":
        skip = True
        continue

    if skip and stripped.startswith("["):
        skip = False

    if not skip:
        out.append(line)

path.write_text(
    "\n".join(out).rstrip() + "\n"
)

PY

# --------------------------------------------------
# 4. Créer le catalogue du modèle
# --------------------------------------------------

echo
echo "[4/6] Création du catalogue MuseGlimmer..."

cat > "$CATALOG" <<EOF
{
  "models": [
    {
      "slug": "$MODEL",
      "display_name": "MuseGlimmer 30B",
      "context_window": 128000,
      "shell_type": "default",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "truncation_policy": {
        "mode": "tokens",
        "limit": 10000
      },
      "input_modalities": [
        "text"
      ],
      "base_instructions": "",
      "support_verbosity": true,
      "default_verbosity": "low",
      "supports_parallel_tool_calls": false,
      "supports_reasoning_summaries": false,
      "supported_reasoning_levels": [],
      "experimental_supported_tools": []
    }
  ]
}
EOF

echo "Catalogue : $CATALOG"

# --------------------------------------------------
# 5. Créer le profil NVIDIA
# --------------------------------------------------

echo
echo "[5/6] Création du profil Codex NVIDIA..."

cat > "$PROFILE" <<EOF
model = "$MODEL"
model_provider = "nvidia-muse"
model_catalog_json = "$CATALOG"

[model_providers.nvidia-muse]
name = "NVIDIA MuseGlimmer"
base_url = "$BASE_URL"
wire_api = "responses"
EOF

echo
echo "Profil :"
cat "$PROFILE"

# --------------------------------------------------
# Vérification de config.toml
# --------------------------------------------------

echo
echo "Vérification de config.toml..."

if grep -q '^\[profiles\.nvidia-muse\]' "$CONFIG"; then
    echo "ERREUR : ancienne configuration nvidia-muse détectée."
    exit 1
fi

if grep -qE '^[[:space:]]*profile[[:space:]]*=' "$CONFIG"; then
    echo "ERREUR : ancien sélecteur profile détecté."
    exit 1
fi

echo "Configuration principale OK."

# --------------------------------------------------
# 6. Lancer le proxy puis Codex
# --------------------------------------------------

echo
echo "[6/6] Démarrage du proxy NVIDIA..."

# Éviter plusieurs instances du proxy
pkill -f "$PROXY" 2>/dev/null || true

sleep 0.5

"$PROXY" &
PROXY_PID=$!

cleanup() {
    echo
    echo "Arrêt du proxy NVIDIA..."
    kill "$PROXY_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

sleep 1

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Erreur : impossible de démarrer le proxy."
    exit 1
fi

echo
echo "=========================================="
echo " NVIDIA MuseGlimmer prêt"
echo "=========================================="
echo
echo "Modèle : $MODEL"
echo "Proxy  : $BASE_URL"
echo "Profil : nvidia-muse"
echo
echo "Lancement de Codex..."
echo

cd "$SCRIPT_DIR"

exec codex --profile nvidia-muse