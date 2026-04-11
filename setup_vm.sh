#!/usr/bin/env bash
# setup_vm.sh — Einmalig nach git clone auf der Linux-VM ausführen
# Erstellt runtime.conf.yaml, palaestrai-DB und installiert das Paket.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[setup] provider-sim VM-Setup"

# runtime.conf.yaml (machine-specific, in .gitignore)
if [ -f runtime.conf.yaml ]; then
    echo "[setup] runtime.conf.yaml bereits vorhanden — übersprungen"
else
    cat > runtime.conf.yaml << 'EOF'
%YAML 1.2
---
fork_method: fork
broker_uri: ipc://*
executor_bus_port: 4242
logger_port: 4243
store_uri: sqlite:///palaestrai.db
EOF
    echo "[setup] runtime.conf.yaml erstellt (fork_method: fork)"
fi

# Paket installieren
echo "[setup] pip install -e .[dev] ..."
pip install -e ".[dev]" -q
echo "[setup] Paket installiert"

# palaestrAI-Datenbank
if [ -f palaestrai.db ]; then
    echo "[setup] palaestrai.db bereits vorhanden — übersprungen"
else
    palaestrai database-create
    echo "[setup] palaestrai.db erstellt"
fi

echo ""
echo "[setup] Fertig. Kurztest:"
echo "  palaestrai experiment-start experiments/configs/soja_arl_dummy.yaml"
