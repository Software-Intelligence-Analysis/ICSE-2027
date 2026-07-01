#!/bin/bash

# ==================================================================================
# Phase 4 Experiment Runner — Code Llama + Few-shot + Iterative
# ==================================================================================
#
# Purpose:
#   Test Code Llama (7B) + Few-shot prompting + Iterative pass@1 vs pass@2
#   using a smaller dataset (20 problems) for rapid iteration.
#
# Corrected Design (fixes caching flaw):
#   Run 1: Baseline (Code Llama, no few-shot) → generates & caches to baseline_cache
#   Run 2: Few-shot (Code Llama + examples) → FRESH GENERATION (separate cache)
#   Run 3: Iterative (Code Llama + retry) → loads baseline cache
#
#   KEY: Run 2 MUST regenerate with few-shot to test prompt effect.
#        Using separate cache files ensures this.
#
# Time estimate: ~4.5 hours total (Run 1: ~2h, Run 2: ~1.2h cached, Run 3: ~1.2h cached)
# ==================================================================================

set -e

echo "=============================================================================="
echo "  PHASE 4: CODE LLAMA + FEW-SHOT + ITERATIVE (20 PROBLEMS)"
echo "  CORRECTED: Separate cache files to test few-shot effect"
echo "=============================================================================="
echo ""

# ── Configuration ────────────────────────────────────────────────────────────────
PLANNER_MODEL="codellama:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
TEMPERATURE="0.0"
THRESHOLD="0.6"
MASTER_CSV="results/master_results_phase4.csv"

# SEPARATE cache files for different experimental conditions:
# This ensures Run 2 generates FRESH code with few-shot examples
BASELINE_CACHE="results/planner_cache_baseline.json"
FEWSHOT_CACHE="results/planner_cache_fewshot.json"

# ── Helper function ──────────────────────────────────────────────────────────────
run_experiment() {
    local RUN_NUM=$1
    local RUN_NAME=$2
    local PLANNER_ARGS=$3
    local CACHE_FILE=$4

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  RUN $RUN_NUM: $RUN_NAME (cache: $CACHE_FILE)"
    echo "──────────────────────────────────────────────────────────────────────"
    echo ""

    # Run the experiment
    python3 src/agent.py \
        --planner-model "$PLANNER_MODEL" \
        --critic-model "$CRITIC_MODEL" \
        --fixer-model "$FIXER_MODEL" \
        --temperature "$TEMPERATURE" \
        --threshold "$THRESHOLD" \
        --dataset-size 20 \
        --version both \
        --run-label "phase4_$RUN_NAME" \
        --master-csv "$MASTER_CSV" \
        --cache-planner-output "$CACHE_FILE" \
        $PLANNER_ARGS

    echo ""
    echo "  ✓ Run $RUN_NUM complete"
    echo ""
}

# ── Clear previous results ───────────────────────────────────────────────────────
echo "Clearing previous Phase 4 results and caches..."
rm -f $MASTER_CSV $BASELINE_CACHE $FEWSHOT_CACHE
mkdir -p results

echo ""
echo "Dataset: First 20 HumanEval problems (0–19)"
echo "  (Ideally: 5 easy + 10 medium + 5 hard, but currently loads sequential)"
echo ""

# ── RUN 1: Baseline (Code Llama, no few-shot) ───────────────────────────────────
# Generates and caches to BASELINE_CACHE
run_experiment 1 "codellama_baseline" "" "$BASELINE_CACHE"

# ── RUN 2: Few-shot prompting (Code Llama + few-shot examples) ───────────────────
# SEPARATE cache file — ensures fresh generation with few-shot prompt
run_experiment 2 "codellama_fewshot" "--use-fewshot" "$FEWSHOT_CACHE"

# ── RUN 3: Iterative pass@2 (Code Llama + retry on failure) ───────────────────────
# Loads BASELINE cache (already generated, no few-shot)
# Tests iterative retry logic (pass@1 vs pass@2)
run_experiment 3 "codellama_iterative" "--use-iterative" "$BASELINE_CACHE"

# ── Summary ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo "  PHASE 4 COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to:"
echo "  Master CSV:      $MASTER_CSV"
echo "  Baseline cache:  $BASELINE_CACHE"
echo "  Few-shot cache:  $FEWSHOT_CACHE"
echo ""
echo "Key comparisons to make:"
echo "  1. Run 1 vs Run 2 → Few-shot effect (does prompting help?)"
echo "  2. Run 3 pass@1 vs pass@2 → Iterative improvement"
echo "  3. Run 1 vs Phase 3 baseline → Code Llama vs Llama 3.2 3B"
echo ""
echo "Next steps:"
echo "  1. Review results in $MASTER_CSV"
echo "  2. Compare pass@1_rate across runs"
echo "  3. Check pass@2_rate improvement in Run 3"
echo "  4. Decide: Run full validation (100 problems) if promising"
echo ""
echo "Per-run details in results/TIMESTAMP_*.csv files"
echo ""
