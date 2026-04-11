#!/usr/bin/env bash
# experiments/run_ablation.sh
# Executes all 4 ablation variants of the datasource ablation study.
# Runtime: 4 variants × N seeds × (budget + ~30s validation) = ~110 min at defaults.
#
# Usage:
#   bash experiments/run_ablation.sh              # 5 seeds, 300s budget (full study)
#   bash experiments/run_ablation.sh 1 15         # 1 seed, 15s budget (quick smoke test)

set -e
cd "$(dirname "$0")/.."

N_SEEDS="${1:-5}"
BUDGET="${2:-300}"
LOG_DIR="experiments/checkpoints/ablation_logs"
mkdir -p "$LOG_DIR"

echo "=== PROVIDER Ablationsstudie Datenquellen ==="
echo "Seeds: $N_SEEDS | Budget: ${BUDGET}s | Start: $(date)"
echo "Log-Verzeichnis: $LOG_DIR"

run_variant() {
    local LABEL="$1"; shift
    echo ""
    echo "--- Variante $LABEL ($(date +%H:%M:%S)) ---"
    python experiments/ablation_train.py "$@" --n-seeds "$N_SEEDS" --budget "$BUDGET" \
        2>&1 | tee "$LOG_DIR/ablation_${LABEL}.log"
    echo "--- $LABEL fertig ---"
}

run_variant "v0"
run_variant "v1_baci"      --baci
run_variant "v2_icio"      --icio
run_variant "v3_baci_icio" --baci --icio

echo ""
echo "=== Alle Varianten abgeschlossen: $(date) ==="
echo ""
echo "Ergebnisse:"
column -t -s $'\t' experiments/checkpoints/ablation_results.tsv 2>/dev/null || \
    cat experiments/checkpoints/ablation_results.tsv
