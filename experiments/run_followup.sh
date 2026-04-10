#!/usr/bin/env bash
# experiments/run_followup.sh
# Follow-up-Experimente zur Ablationsstudie:
#   - Interaktionshypothese: BACI mit Scale=2.0 (große Exporteure unkonstrained)
#   - ICIO-Normalisierung: sqrt und softmax
#   - Budget-Sensitivität: V0 und V2 mit 15 Minuten Training
#
# Nutzung:
#   bash experiments/run_followup.sh              # Vollstudie
#   bash experiments/run_followup.sh 1 15         # Schnelltest

set -e
cd "$(dirname "$0")/.."

N_SEEDS="${1:-5}"
BUDGET="${2:-300}"
BUDGET_LONG="${3:-900}"       # 15min für Budget-Sensitivität
N_SEEDS_LONG="${4:-3}"        # 3 Seeds reichen für Budget-Test

LOG_DIR="experiments/checkpoints/ablation_logs"
mkdir -p "$LOG_DIR"

echo "=== PROVIDER Follow-up Ablationsstudie ==="
echo "Seeds: $N_SEEDS | Budget: ${BUDGET}s | Start: $(date)"

run_variant() {
    local LABEL="$1"; shift
    echo ""
    echo "--- $LABEL ($(date +%H:%M:%S)) ---"
    python experiments/ablation_train.py "$@" --n-seeds "$N_SEEDS" --budget "$BUDGET" \
        2>&1 | tee "$LOG_DIR/followup_${LABEL}.log"
    echo "--- $LABEL fertig ---"
}

run_variant_long() {
    local LABEL="$1"; shift
    echo ""
    echo "--- $LABEL LONG ($(date +%H:%M:%S)) ---"
    python experiments/ablation_train.py "$@" --n-seeds "$N_SEEDS_LONG" --budget "$BUDGET_LONG" \
        2>&1 | tee "$LOG_DIR/followup_${LABEL}.log"
    echo "--- $LABEL fertig ---"
}

# Test 1: Interaktionshypothese — BACI-Scale=2.0
# Große Exporteure (Brazil=2.0) nicht mehr constrained, kleine (Argentina=0.60) leicht
run_variant "v1_baci_scale2" --baci --baci-scale 2.0

# Test 2: ICIO sqrt-Normalisierung (sanftere Gewichte)
run_variant "v2_icio_sqrt"   --icio --icio-norm sqrt

# Test 3: ICIO softmax-Normalisierung (schärfere Dominanz)
run_variant "v2_icio_softmax" --icio --icio-norm softmax

# Test 4: Budget-Sensitivität — V0 und V2 mit 15min Training
run_variant_long "v0_15min"
run_variant_long "v2_icio_15min"  --icio

echo ""
echo "=== Follow-up abgeschlossen: $(date) ==="
echo ""
echo "Ergebnisse (alle Varianten inkl. Hauptstudie):"
column -t -s $'\t' experiments/checkpoints/ablation_results.tsv 2>/dev/null || \
    cat experiments/checkpoints/ablation_results.tsv
