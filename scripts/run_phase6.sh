#!/bin/bash

# ==============================================================================
#  Phase 6 -- Threshold Sweep (tau) + Temperature Retry Study
# ==============================================================================
#
#  Part A: tau sweep (runs 1-9)
#    For each of 3 models, run tau = {0.6, 0.7, 0.75} with --version both.
#    Same planner cache is reused across tau values per model, so the SAME
#    generated code is being re-evaluated at different fixer thresholds.
#    This cleanly isolates tau's effect from planner variance.
#
#    Models: Llama 3.1 8B | Code Llama 7B | Qwen2.5 Coder 7B
#    Tau values: 0.60, 0.70, 0.75
#    Runs: 3 models x 3 tau values = 9 runs
#
#  Part B: Temperature retry study (runs 10-15)
#    For each of 3 models, run --use-iterative with retry_temperature = {0.3, 0.5}.
#    At T_retry > 0 the second attempt is a genuinely different output,
#    making pass@2 a real independent draw rather than a deterministic repeat.
#    tau is fixed at 0.6 (same as all prior phases).
#
#    Models: Llama 3.1 8B | Code Llama 7B | Qwen2.5 Coder 7B
#    T_retry: 0.3, 0.5
#    Runs: 3 models x 2 T_retry values = 6 runs
#
#  Total: 15 runs, 30 CSV rows (--version both = 2 rows per run)
#  Master CSV: results/master_results_phase6.csv
#
#  Resume after interruption:
#    RESUME_FROM=<run number 1-15> bash run_phase6.sh
#
# ==============================================================================

set -e

RESUME_FROM=${RESUME_FROM:-1}

# -- Keep laptop awake --------------------------------------------------------
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo ''; echo 'caffeinate stopped.'" EXIT
echo "caffeinate started (PID $CAFFEINATE_PID) -- laptop will stay awake."

# -- Configuration ------------------------------------------------------------
LLAMA_MODEL="llama3.1:8b"
CODELLAMA_MODEL="codellama:7b"
QWEN_MODEL="qwen2.5-coder:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
TEMPERATURE="0.0"
DATASET_SIZE="100"
MASTER_CSV="results/master_results_phase6.csv"

mkdir -p results

# -- Clear results only on a fresh start --------------------------------------
if [ "$RESUME_FROM" -eq 1 ]; then
    echo "Fresh run -- clearing previous Phase 6 results..."
    rm -f "$MASTER_CSV"
    rm -f results/cache_p6_llama.json
    rm -f results/cache_p6_codellama.json
    rm -f results/cache_p6_qwen.json
else
    echo "Resuming from run $RESUME_FROM -- existing results preserved."
fi

echo ""
echo "=============================================================================="
echo "  PHASE 6: TAU SWEEP + TEMPERATURE RETRY STUDY"
echo "  N=$DATASET_SIZE | Models: Llama 3.1 8B, Code Llama 7B, Qwen2.5 Coder 7B"
echo "=============================================================================="

# -- Helpers ------------------------------------------------------------------
run_tau() {
    local RUN_NUM=$1
    local LABEL=$2
    local PLANNER=$3
    local CACHE=$4
    local TAU=$5

    if [ "$RUN_NUM" -lt "$RESUME_FROM" ]; then
        echo "  [Run $RUN_NUM] Skipping (already completed)."
        return
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Run $RUN_NUM / 15 -- $LABEL  [tau=$TAU]"
    echo "  Planner: $PLANNER | Cache: $CACHE"
    echo "----------------------------------------------------------------------"
    echo ""

    python3 src/agent.py \
        --planner-model        "$PLANNER"      \
        --critic-model         "$CRITIC_MODEL" \
        --fixer-model          "$FIXER_MODEL"  \
        --temperature          "$TEMPERATURE"  \
        --threshold            "$TAU"          \
        --dataset-size         "$DATASET_SIZE" \
        --version              both            \
        --run-label            "$LABEL"        \
        --master-csv           "$MASTER_CSV"   \
        --cache-planner-output "$CACHE"

    echo ""
    echo "  Run $RUN_NUM complete."
}

run_retry() {
    local RUN_NUM=$1
    local LABEL=$2
    local PLANNER=$3
    local CACHE=$4
    local T_RETRY=$5

    if [ "$RUN_NUM" -lt "$RESUME_FROM" ]; then
        echo "  [Run $RUN_NUM] Skipping (already completed)."
        return
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Run $RUN_NUM / 15 -- $LABEL  [T_retry=$T_RETRY]"
    echo "  Planner: $PLANNER | Cache: $CACHE"
    echo "----------------------------------------------------------------------"
    echo ""

    python3 src/agent.py \
        --planner-model        "$PLANNER"      \
        --critic-model         "$CRITIC_MODEL" \
        --fixer-model          "$FIXER_MODEL"  \
        --temperature          "$TEMPERATURE"  \
        --retry-temperature    "$T_RETRY"      \
        --threshold            "0.6"           \
        --dataset-size         "$DATASET_SIZE" \
        --version              both            \
        --use-iterative                        \
        --run-label            "$LABEL"        \
        --master-csv           "$MASTER_CSV"   \
        --cache-planner-output "$CACHE"

    echo ""
    echo "  Run $RUN_NUM complete."
}

# == PART A: TAU SWEEP ========================================================
echo ""
echo "  -- PART A: TAU SWEEP (runs 1-9) --"

# Llama 3.1 8B
run_tau  1 "llama31_tau60"  "$LLAMA_MODEL" "results/cache_p6_llama.json"     "0.60"
run_tau  2 "llama31_tau70"  "$LLAMA_MODEL" "results/cache_p6_llama.json"     "0.70"
run_tau  3 "llama31_tau75"  "$LLAMA_MODEL" "results/cache_p6_llama.json"     "0.75"

# Code Llama 7B
run_tau  4 "codellama_tau60" "$CODELLAMA_MODEL" "results/cache_p6_codellama.json" "0.60"
run_tau  5 "codellama_tau70" "$CODELLAMA_MODEL" "results/cache_p6_codellama.json" "0.70"
run_tau  6 "codellama_tau75" "$CODELLAMA_MODEL" "results/cache_p6_codellama.json" "0.75"

# Qwen2.5 Coder 7B
run_tau  7 "qwen_tau60"  "$QWEN_MODEL" "results/cache_p6_qwen.json" "0.60"
run_tau  8 "qwen_tau70"  "$QWEN_MODEL" "results/cache_p6_qwen.json" "0.70"
run_tau  9 "qwen_tau75"  "$QWEN_MODEL" "results/cache_p6_qwen.json" "0.75"

# == PART B: TEMPERATURE RETRY ================================================
echo ""
echo "  -- PART B: TEMPERATURE RETRY (runs 10-15) --"

# Llama 3.1 8B
run_retry 10 "llama31_retry_t03" "$LLAMA_MODEL" "results/cache_p6_llama.json"     "0.3"
run_retry 11 "llama31_retry_t05" "$LLAMA_MODEL" "results/cache_p6_llama.json"     "0.5"

# Code Llama 7B
run_retry 12 "codellama_retry_t03" "$CODELLAMA_MODEL" "results/cache_p6_codellama.json" "0.3"
run_retry 13 "codellama_retry_t05" "$CODELLAMA_MODEL" "results/cache_p6_codellama.json" "0.5"

# Qwen2.5 Coder 7B
run_retry 14 "qwen_retry_t03" "$QWEN_MODEL" "results/cache_p6_qwen.json" "0.3"
run_retry 15 "qwen_retry_t05" "$QWEN_MODEL" "results/cache_p6_qwen.json" "0.5"

# == SUMMARY ==================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 6 COMPLETE"
echo "=============================================================================="
echo ""
echo "  Master CSV : $MASTER_CSV"
echo "  Rows       : 30 (15 runs x 2 versions each)"
echo ""
echo "  Part A -- Tau sweep key comparisons:"
echo "    For each model: does higher tau reduce gate_revert_rate and recover pass@1?"
echo "    Hypothesis: tau=0.75 should help Qwen/DeepSeek, may hurt Code Llama."
echo ""
echo "  Part B -- T retry key comparisons:"
echo "    For each model: does T_retry=0.3 or 0.5 produce meaningfully higher pass@2?"
echo "    Hypothesis: Code Llama benefits most (baseline low, retry ceiling is high)."
echo ""
