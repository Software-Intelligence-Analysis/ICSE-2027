#!/bin/bash

# ==============================================================================
#  Phase 5 -- Code-Pretrained Model Expansion
# ==============================================================================
#
#  Design:
#    Three new code-specialized planners, each vs the same monitoring pipeline
#    (Llama 3.1 8B critic + fixer, T=0.0, threshold=0.6).
#
#    Condition E: DeepSeek Coder 6.7B  -- baseline + monitoring
#    Condition F: StarCoder2 7B        -- baseline + monitoring
#    Condition G: Qwen2.5 Coder 7B     -- baseline + monitoring
#
#  Each condition is replicated 3 independent trials (T=0.0, deterministic).
#  Total: 9 agent.py invocations (--version both), 18 CSV rows.
#
#  Caching:
#    Each trial gets its own cache file so trials are truly independent.
#    Within a trial, baseline and monitoring evaluate the SAME generated code.
#
#  Resume after interruption:
#    Set RESUME_FROM to the run number you want to restart from (1-9).
#    Example:
#      RESUME_FROM=4 bash run_phase5.sh
#
#  Caffeinate:
#    caffeinate -dims keeps the display on and prevents idle/disk sleep.
#    It is launched in the background and killed automatically on exit.
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
DEEPSEEK_MODEL="deepseek-coder:6.7b"
STARCODER_MODEL="starcoder2:7b"
QWEN_MODEL="qwen2.5-coder:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
TEMPERATURE="0.0"
THRESHOLD="0.6"
DATASET_SIZE="100"
MASTER_CSV="results/master_results_phase5.csv"

mkdir -p results

# -- Clear results only on a fresh start (not when resuming) ------------------
if [ "$RESUME_FROM" -eq 1 ]; then
    echo "Fresh run -- clearing previous Phase 5 results and caches..."
    rm -f "$MASTER_CSV"
    rm -f results/cache_deepseek_trial1.json
    rm -f results/cache_deepseek_trial2.json
    rm -f results/cache_deepseek_trial3.json
    rm -f results/cache_starcoder_trial1.json
    rm -f results/cache_starcoder_trial2.json
    rm -f results/cache_starcoder_trial3.json
    rm -f results/cache_qwen_trial1.json
    rm -f results/cache_qwen_trial2.json
    rm -f results/cache_qwen_trial3.json
else
    echo "Resuming from run $RESUME_FROM -- existing results preserved."
fi

echo ""
echo "=============================================================================="
echo "  PHASE 5: CODE-PRETRAINED MODEL EXPANSION"
echo "  N=$DATASET_SIZE, 3 trials per model, threshold=$THRESHOLD"
echo "  Models: DeepSeek Coder 6.7B | StarCoder2 7B | Qwen2.5 Coder 7B"
echo "=============================================================================="

# -- Helper -------------------------------------------------------------------
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
    echo "  Run $RUN_NUM / 9 -- $LABEL"
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

# -- DeepSeek Coder 6.7B -- 3 trials (Condition E) ----------------------------
run_experiment 1 "deepseek_trial1" "$DEEPSEEK_MODEL" "results/cache_deepseek_trial1.json"
run_experiment 2 "deepseek_trial2" "$DEEPSEEK_MODEL" "results/cache_deepseek_trial2.json"
run_experiment 3 "deepseek_trial3" "$DEEPSEEK_MODEL" "results/cache_deepseek_trial3.json"

# -- StarCoder2 7B -- 3 trials (Condition F) -----------------------------------
run_experiment 4 "starcoder_trial1" "$STARCODER_MODEL" "results/cache_starcoder_trial1.json"
run_experiment 5 "starcoder_trial2" "$STARCODER_MODEL" "results/cache_starcoder_trial2.json"
run_experiment 6 "starcoder_trial3" "$STARCODER_MODEL" "results/cache_starcoder_trial3.json"

# -- Qwen2.5 Coder 7B -- 3 trials (Condition G) --------------------------------
run_experiment 7 "qwen_trial1" "$QWEN_MODEL" "results/cache_qwen_trial1.json"
run_experiment 8 "qwen_trial2" "$QWEN_MODEL" "results/cache_qwen_trial2.json"
run_experiment 9 "qwen_trial3" "$QWEN_MODEL" "results/cache_qwen_trial3.json"

# -- Summary ------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "  PHASE 5 COMPLETE"
echo "=============================================================================="
echo ""
echo "  Master CSV : $MASTER_CSV"
echo "  Rows       : 18 (9 runs x 2 versions each)"
echo ""
echo "  Key comparisons (vs Phase 4b Condition A as generic unmonitored baseline):"
echo "    E-baseline vs A    -- Does DeepSeek Coder alone beat generic Llama 3.1?"
echo "    F-baseline vs A    -- Does StarCoder2 alone beat generic Llama 3.1?"
echo "    G-baseline vs A    -- Does Qwen2.5 Coder alone beat generic Llama 3.1?"
echo "    E-monitored vs D   -- How does DeepSeek Coder+monitoring compare to CodeLlama+monitoring?"
echo "    F-monitored vs D   -- How does StarCoder2+monitoring compare to CodeLlama+monitoring?"
echo "    G-monitored vs D   -- How does Qwen2.5 Coder+monitoring compare to CodeLlama+monitoring?"
echo ""
echo "  To pull Phase 4b + Phase 5 together for full comparison:"
echo "    cat results/master_results_phase4b.csv results/master_results_phase5.csv"
echo ""
