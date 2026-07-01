#!/bin/bash

# ==============================================================================
#  Phase 7 -- Confidence Interval Study for Inverse Capability Hypothesis
# ==============================================================================
#
#  Motivation:
#    The +32 pp (Code Llama 7B) and -3 pp (Qwen2.5 Coder 7B) results are the
#    two endpoints of H3 (Inverse Capability Hypothesis) -- the central claim
#    of the paper. Both currently come from single-trial T=0.0 runs, which
#    means zero variance and no confidence intervals. Reviewers at ICSE/FSE
#    will challenge these numbers without error bars.
#
#    Phase 7 runs 3 genuinely independent trials per model by using T=0.3 for
#    the planner (non-deterministic), so each trial produces different generated
#    code and a real variance estimate. Each trial gets its own cache file.
#    Within a trial, tau=0.60 and tau=0.70 both evaluate the SAME generated
#    code (same cache), so threshold comparisons are clean.
#
#  Design:
#    Models  : Code Llama 7B (H3 positive end) | Qwen2.5 Coder 7B (H3 negative end)
#    Trials  : 3 independent trials per model (T=0.3 planner, fresh cache each)
#    Tau     : 0.60 and 0.70 per trial (same cache reused within trial)
#    Critic  : Llama 3.1 8B (same as Phase 3/4/5/6)
#    Fixer   : Llama 3.1 8B (same as Phase 3/4/5/6)
#    N       : 100 HumanEval problems per run
#
#  Run map (12 runs total, 24 CSV rows with --version both):
#    Runs  1-2  : Code Llama trial 1  (tau=0.60, tau=0.70)
#    Runs  3-4  : Code Llama trial 2  (tau=0.60, tau=0.70)
#    Runs  5-6  : Code Llama trial 3  (tau=0.60, tau=0.70)
#    Runs  7-8  : Qwen trial 1        (tau=0.60, tau=0.70)
#    Runs  9-10 : Qwen trial 2        (tau=0.60, tau=0.70)
#    Runs 11-12 : Qwen trial 3        (tau=0.60, tau=0.70)
#
#  Why T=0.3 (not T=0.0):
#    T=0.0 is fully deterministic -- all 3 trials would be identical (same
#    generated code, same critic scores, same results). This happened with
#    DeepSeek in Phase 5 and gives zero variance. T=0.3 produces meaningfully
#    different outputs while keeping generation quality high.
#
#  Cache note:
#    Each trial uses a FRESH, EMPTY cache file. The first tau run (0.60)
#    generates planner output and saves it; the second tau run (0.70) loads
#    from that cache. This means the two tau values see identical generated
#    code within each trial -- threshold effect is isolated from planner
#    variance, and trials are genuinely independent from each other.
#
#  Expected outputs:
#    - 3 pass@1 estimates per model per tau -> compute mean + 95% CI
#    - Confirm +32 pp (Code Llama) and -3 pp (Qwen) are robust, not flukes
#    - CI width will indicate whether single-trial Phase 4/5 values are reliable
#
#  Resume after interruption:
#    RESUME_FROM=<run number 1-12> bash run_phase7.sh
#    Example: RESUME_FROM=5 bash run_phase7.sh
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
CODELLAMA_MODEL="codellama:7b"
QWEN_MODEL="qwen2.5-coder:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
PLANNER_TEMPERATURE="0.3"       # Non-deterministic: gives genuine trial variance
DATASET_SIZE="100"
MASTER_CSV="results/master_results_phase7.csv"

mkdir -p results

# -- Clear results only on a fresh start --------------------------------------
if [ "$RESUME_FROM" -eq 1 ]; then
    echo "Fresh run -- clearing previous Phase 7 results and caches..."
    rm -f "$MASTER_CSV"
    rm -f results/cache_p7_codellama_trial1.json
    rm -f results/cache_p7_codellama_trial2.json
    rm -f results/cache_p7_codellama_trial3.json
    rm -f results/cache_p7_qwen_trial1.json
    rm -f results/cache_p7_qwen_trial2.json
    rm -f results/cache_p7_qwen_trial3.json
else
    echo "Resuming from run $RESUME_FROM -- existing results preserved."
fi

echo ""
echo "=============================================================================="
echo "  PHASE 7: CI STUDY -- INVERSE CAPABILITY HYPOTHESIS"
echo "  N=$DATASET_SIZE | T_planner=$PLANNER_TEMPERATURE | 3 trials x 2 tau x 2 models"
echo "  Models: Code Llama 7B | Qwen2.5 Coder 7B"
echo "  Tau values: 0.60, 0.70"
echo "=============================================================================="

# -- Unload all models from Ollama to free RAM/VRAM between runs --------------
flush_ollama() {
    echo "  [flush] Unloading models from Ollama to free memory..."
    ollama stop "$CODELLAMA_MODEL" 2>/dev/null || true
    ollama stop "$QWEN_MODEL"      2>/dev/null || true
    ollama stop "$CRITIC_MODEL"    2>/dev/null || true
    ollama stop "$FIXER_MODEL"     2>/dev/null || true
    echo "  [flush] Sleeping 30s for memory and thermals to settle..."
    sleep 30
}

# -- Helper -------------------------------------------------------------------
run_experiment() {
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
    echo "  Run $RUN_NUM / 12 -- $LABEL  [tau=$TAU]"
    echo "  Planner: $PLANNER | T=$PLANNER_TEMPERATURE | Cache: $CACHE"
    echo "----------------------------------------------------------------------"
    echo ""

    python3 src/agent.py \
        --planner-model        "$PLANNER"           \
        --critic-model         "$CRITIC_MODEL"      \
        --fixer-model          "$FIXER_MODEL"       \
        --temperature          "$PLANNER_TEMPERATURE" \
        --threshold            "$TAU"               \
        --dataset-size         "$DATASET_SIZE"      \
        --version              both                 \
        --run-label            "$LABEL"             \
        --master-csv           "$MASTER_CSV"        \
        --cache-planner-output "$CACHE"

    echo ""
    echo "  Run $RUN_NUM complete."
    flush_ollama
}

# == CODE LLAMA 7B -- 3 trials ================================================
echo ""
echo "  -- CODE LLAMA 7B (runs 1-6) --"

run_experiment  1 "codellama_p7_t1_tau60" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial1.json" "0.60"
run_experiment  2 "codellama_p7_t1_tau70" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial1.json" "0.70"

run_experiment  3 "codellama_p7_t2_tau60" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial2.json" "0.60"
run_experiment  4 "codellama_p7_t2_tau70" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial2.json" "0.70"

run_experiment  5 "codellama_p7_t3_tau60" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial3.json" "0.60"
run_experiment  6 "codellama_p7_t3_tau70" "$CODELLAMA_MODEL" "results/cache_p7_codellama_trial3.json" "0.70"

# == QWEN2.5 CODER 7B -- 3 trials =============================================
echo ""
echo "  -- QWEN2.5 CODER 7B (runs 7-12) --"

run_experiment  7 "qwen_p7_t1_tau60" "$QWEN_MODEL" "results/cache_p7_qwen_trial1.json" "0.60"
run_experiment  8 "qwen_p7_t1_tau70" "$QWEN_MODEL" "results/cache_p7_qwen_trial1.json" "0.70"

run_experiment  9 "qwen_p7_t2_tau60" "$QWEN_MODEL" "results/cache_p7_qwen_trial2.json" "0.60"
run_experiment 10 "qwen_p7_t2_tau70" "$QWEN_MODEL" "results/cache_p7_qwen_trial2.json" "0.70"

run_experiment 11 "qwen_p7_t3_tau60" "$QWEN_MODEL" "results/cache_p7_qwen_trial3.json" "0.60"
run_experiment 12 "qwen_p7_t3_tau70" "$QWEN_MODEL" "results/cache_p7_qwen_trial3.json" "0.70"

# == SUMMARY ==================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 7 COMPLETE"
echo "=============================================================================="
echo ""
echo "  Master CSV : $MASTER_CSV"
echo "  Rows       : 24 (12 runs x 2 versions each)"
echo ""
echo "  Next step -- compute CIs from the 3-trial results:"
echo ""
echo "    python3 src/compute_phase7_ci.py"
echo ""
echo "  Expected outputs:"
echo "    Code Llama tau=0.60: mean Δpp across 3 trials + 95% CI"
echo "    Code Llama tau=0.70: mean Δpp across 3 trials + 95% CI"
echo "    Qwen       tau=0.60: mean Δpp across 3 trials + 95% CI"
echo "    Qwen       tau=0.70: mean Δpp across 3 trials + 95% CI"
echo ""
echo "  Key question: do the CIs on Code Llama (+32 pp) and Qwen (-3 pp)"
echo "  exclude zero? If yes, H3 is statistically defensible at ICSE."
echo ""
