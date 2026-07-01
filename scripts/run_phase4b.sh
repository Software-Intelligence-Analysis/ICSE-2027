#!/bin/bash

# ==============================================================================
#  Phase 4b -- 2x2 Experiment: Model Family x Monitoring
# ==============================================================================
#
#  Design:
#
#    Condition A: Llama 3.1 8B  -- baseline pipeline  (no monitoring)
#    Condition B: Llama 3.1 8B  -- monitoring pipeline
#    Condition C: Code Llama 7B -- baseline pipeline  (no monitoring)
#    Condition D: Code Llama 7B -- monitoring pipeline
#
#  Each condition is replicated 3 independent trials for statistical reliability.
#  Total: 6 agent.py invocations (each --version both produces A+B or C+D).
#  Dataset: N=100, matching Phase 2 and Phase 3 methodology.
#
#  Caching:
#    Each trial gets its own cache file per model so trials are truly independent.
#    Within a trial, baseline and monitoring evaluate the SAME generated code
#    (baseline generates + caches, monitoring loads from cache).
#
#  Resume after interruption:
#    Set RESUME_FROM to the run number you want to restart from (1-6).
#    Completed runs are not re-executed and their results stay in the master CSV.
#    Example:
#      RESUME_FROM=4 bash run_phase4b.sh
#
# ==============================================================================

set -e

RESUME_FROM=${RESUME_FROM:-1}

# ── Configuration ──────────────────────────────────────────────────────────────
LLAMA_MODEL="llama3.1:8b"
CODELLAMA_MODEL="codellama:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
TEMPERATURE="0.0"
THRESHOLD="0.6"
DATASET_SIZE="100"
MASTER_CSV="results/master_results_phase4b.csv"

mkdir -p results

# ── Clear results only on a fresh start (not when resuming) ───────────────────
if [ "$RESUME_FROM" -eq 1 ]; then
    echo "Fresh run -- clearing previous Phase 4b results and caches..."
    rm -f "$MASTER_CSV"
    rm -f results/cache_llama31_trial1.json
    rm -f results/cache_llama31_trial2.json
    rm -f results/cache_llama31_trial3.json
    rm -f results/cache_codellama_trial1.json
    rm -f results/cache_codellama_trial2.json
    rm -f results/cache_codellama_trial3.json
else
    echo "Resuming from run $RESUME_FROM -- existing results preserved."
fi

echo ""
echo "=============================================================================="
echo "  PHASE 4b: 2x2 EXPERIMENT (MODEL FAMILY x MONITORING)"
echo "  N=$DATASET_SIZE, 3 trials per model, threshold=$THRESHOLD"
echo "=============================================================================="

# ── Helper ────────────────────────────────────────────────────────────────────
run_experiment() {
    local RUN_NUM=$1
    local LABEL=$2
    local PLANNER=$3
    local CACHE=$4

    if [ "$RUN_NUM" -lt "$RESUME_FROM" ]; then
        echo ""
        echo "  [Run $RUN_NUM] Skipping (already completed)."
        return
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Run $RUN_NUM / 6 -- $LABEL"
    echo "  Planner: $PLANNER | Cache: $CACHE"
    echo "----------------------------------------------------------------------"
    echo ""

    python3 src/agent.py \
        --planner-model  "$PLANNER"      \
        --critic-model   "$CRITIC_MODEL" \
        --fixer-model    "$FIXER_MODEL"  \
        --temperature    "$TEMPERATURE"  \
        --threshold      "$THRESHOLD"    \
        --dataset-size   "$DATASET_SIZE" \
        --version        both            \
        --run-label      "$LABEL"        \
        --master-csv     "$MASTER_CSV"   \
        --cache-planner-output "$CACHE"

    echo ""
    echo "  Run $RUN_NUM complete."
}

# ── Llama 3.1 8B -- 3 trials (Conditions A and B) ─────────────────────────────
run_experiment 1 "llama31_trial1" "$LLAMA_MODEL" "results/cache_llama31_trial1.json"
run_experiment 2 "llama31_trial2" "$LLAMA_MODEL" "results/cache_llama31_trial2.json"
run_experiment 3 "llama31_trial3" "$LLAMA_MODEL" "results/cache_llama31_trial3.json"

# ── Code Llama 7B -- 3 trials (Conditions C and D) ────────────────────────────
run_experiment 4 "codellama_trial1" "$CODELLAMA_MODEL" "results/cache_codellama_trial1.json"
run_experiment 5 "codellama_trial2" "$CODELLAMA_MODEL" "results/cache_codellama_trial2.json"
run_experiment 6 "codellama_trial3" "$CODELLAMA_MODEL" "results/cache_codellama_trial3.json"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo "  PHASE 4b COMPLETE"
echo "=============================================================================="
echo ""
echo "  Master CSV : $MASTER_CSV"
echo "  Rows       : 12 (6 runs x 2 versions each)"
echo ""
echo "  To compute per-condition means across 3 trials:"
echo "    Group by (planner_model, version), average pass@1_rate over trials."
echo ""
echo "  Key comparisons:"
echo "    A vs B  -- Does monitoring help Llama 3.1 8B?"
echo "    C vs D  -- Does monitoring help Code Llama 7B?"
echo "    A vs C  -- Does switching to Code Llama 7B alone help (no monitoring)?"
echo "    A vs D  -- Combined effect: code-specialized model + monitoring vs generic unassisted"
echo ""
