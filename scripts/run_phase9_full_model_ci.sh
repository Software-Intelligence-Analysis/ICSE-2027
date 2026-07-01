#!/usr/bin/env bash
# =============================================================================
# Phase 9 — Full-Model CI Study (Extending Phase 7 to All 5 Models)
# =============================================================================
#
# Phase 7 ran 3 trials at T=0.3 for Code Llama and Qwen only — the two
# endpoints of the Inverse Capability Hypothesis (H3). Phase 9 fills in the
# remaining three models to give a complete H3 picture with 95% CIs:
#
#   Model                  Baseline pass@1  H3 prediction
#   ─────────────────────  ───────────────  ─────────────
#   Code Llama 7B          ~20%             Large benefit  (Phase 7 ✓)
#   Llama 3.1 8B           ~72%             Moderate loss  (Phase 9)
#   DeepSeek Coder 6.7B    ~77%             Moderate loss  (Phase 9)
#   Qwen2.5-Coder 7B       ~90%             Small loss     (Phase 7 ✓)
#   StarCoder2 7B          ~0%              Near zero      (Phase 9)
#
# Design (identical to Phase 7):
#   - 3 genuinely independent trials per model (T=0.3 planner)
#   - τ = 0.60 and τ = 0.70 per trial (same cache reused within trial)
#   - Pre-generated planner caches in results/cache/ (T=0.3, N=100)
#   - 18 total runs (3 models × 3 trials × 2 tau values)
#
# Cache files used (pre-generated, T=0.3, N=100, few_shot=False):
#   results/cache/cache_llama31_trial{1,2,3}.json
#   results/cache/cache_deepseek_trial{1,2,3}.json
#   results/cache/cache_starcoder_trial{1,2,3}.json
#
# Models:
#   Planner : per-model (llama3.1:8b | deepseek-coder:6.7b | starcoder2:7b)
#   Critic  : llama3.1:8b  (T=0.0, same as all phases)
#   Fixer   : llama3.1:8b  (T=0.0, same as all phases)
#
# Note on Llama 3.1 8B: the planner and critic/fixer are the same model.
# This is fine — the cache ensures planner outputs are pre-computed, so only
# the critic+fixer (8B) runs in the monitoring pass.
#
# Note on StarCoder2: baseline pass@1 from Phase 5 was ~0%. The monitoring
# pass will likely show near-zero benefit — this is expected and important
# evidence: H3 breaks down at the very bottom of capability because there is
# nothing for the fixer to work with.
#
# Runtime estimate: ~25-30 min per monitoring run × 18 = ~8-9 hours total
# Recommend running overnight or using RESUME_FROM to continue after interruption.
#
# Resume after interruption:
#   RESUME_FROM=<run number 1-18> bash scripts/run_phase9_full_model_ci.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RESUME_FROM=${RESUME_FROM:-1}

# ── Keep laptop awake ─────────────────────────────────────────────────────────
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo ''; echo 'caffeinate stopped.'" EXIT
echo "caffeinate started (PID $CAFFEINATE_PID) — laptop will stay awake."

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_MODEL="llama3.1:8b"
DEEPSEEK_MODEL="deepseek-coder:6.7b"
STARCODER_MODEL="starcoder2:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"
PLANNER_TEMPERATURE="0.3"
DATASET_SIZE="100"
MASTER_CSV="results/master_results_phase9.csv"

mkdir -p results

# Clear results only on a fresh start
if [ "$RESUME_FROM" -eq 1 ]; then
    echo "Fresh run — clearing previous Phase 9 results..."
    rm -f "$MASTER_CSV"
else
    echo "Resuming from run $RESUME_FROM — existing results preserved."
fi

echo ""
echo "=============================================================================="
echo "  PHASE 9: FULL-MODEL CI STUDY — INVERSE CAPABILITY HYPOTHESIS"
echo "  N=$DATASET_SIZE | T_planner=$PLANNER_TEMPERATURE | 3 trials × 2 tau × 3 models"
echo "  Models: Llama 3.1 8B | DeepSeek Coder 6.7B | StarCoder2 7B"
echo "  Tau values: 0.60, 0.70"
echo "  Master CSV: $MASTER_CSV"
echo "=============================================================================="

# ── Flush Ollama between heavy runs to free VRAM ─────────────────────────────
flush_ollama() {
    echo "  [flush] Unloading models from Ollama to free memory..."
    ollama stop "$LLAMA_MODEL"    2>/dev/null || true
    ollama stop "$DEEPSEEK_MODEL" 2>/dev/null || true
    ollama stop "$STARCODER_MODEL" 2>/dev/null || true
    ollama stop "$CRITIC_MODEL"   2>/dev/null || true
    ollama stop "$FIXER_MODEL"    2>/dev/null || true
    echo "  [flush] Sleeping 30s for memory and thermals to settle..."
    sleep 30
}

# ── Helper ────────────────────────────────────────────────────────────────────
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
    echo "  Run $RUN_NUM / 18 — $LABEL  [tau=$TAU]"
    echo "  Planner: $PLANNER | T=$PLANNER_TEMPERATURE | Cache: $CACHE"
    echo "----------------------------------------------------------------------"
    echo ""

    python3 src/agent.py \
        --planner-model        "$PLANNER"              \
        --critic-model         "$CRITIC_MODEL"         \
        --fixer-model          "$FIXER_MODEL"          \
        --temperature          "$PLANNER_TEMPERATURE"  \
        --threshold            "$TAU"                  \
        --dataset-size         "$DATASET_SIZE"         \
        --version              both                    \
        --run-label            "$LABEL"                \
        --master-csv           "$MASTER_CSV"           \
        --cache-planner-output "$CACHE"

    echo ""
    echo "  Run $RUN_NUM complete."
    flush_ollama
}

# =============================================================================
# LLAMA 3.1 8B — 3 trials × 2 tau
# Note: planner and critic/fixer are the same model.
# Pre-generated cache: results/cache/cache_llama31_trial{1,2,3}.json
# =============================================================================
echo ""
echo "  ── LLAMA 3.1 8B (runs 1–6) ──"

run_experiment  1  "llama31_p9_t1_tau60"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial1.json"  "0.60"
run_experiment  2  "llama31_p9_t1_tau70"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial1.json"  "0.70"

run_experiment  3  "llama31_p9_t2_tau60"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial2.json"  "0.60"
run_experiment  4  "llama31_p9_t2_tau70"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial2.json"  "0.70"

run_experiment  5  "llama31_p9_t3_tau60"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial3.json"  "0.60"
run_experiment  6  "llama31_p9_t3_tau70"  "$LLAMA_MODEL"  "results/cache/cache_llama31_trial3.json"  "0.70"

# =============================================================================
# DEEPSEEK CODER 6.7B — 3 trials × 2 tau
# Pre-generated cache: results/cache/cache_deepseek_trial{1,2,3}.json
# =============================================================================
echo ""
echo "  ── DEEPSEEK CODER 6.7B (runs 7–12) ──"

run_experiment  7  "deepseek_p9_t1_tau60"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial1.json"  "0.60"
run_experiment  8  "deepseek_p9_t1_tau70"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial1.json"  "0.70"

run_experiment  9  "deepseek_p9_t2_tau60"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial2.json"  "0.60"
run_experiment 10  "deepseek_p9_t2_tau70"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial2.json"  "0.70"

run_experiment 11  "deepseek_p9_t3_tau60"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial3.json"  "0.60"
run_experiment 12  "deepseek_p9_t3_tau70"  "$DEEPSEEK_MODEL"  "results/cache/cache_deepseek_trial3.json"  "0.70"

# =============================================================================
# STARCODER2 7B — 3 trials × 2 tau
# Pre-generated cache: results/cache/cache_starcoder_trial{1,2,3}.json
# Note: Phase 5 baseline for StarCoder2 was ~0% pass@1. Monitoring benefit
# is expected to be near-zero — the fixer cannot fix what it cannot understand.
# =============================================================================
echo ""
echo "  ── STARCODER2 7B (runs 13–18) ──"

run_experiment 13  "starcoder_p9_t1_tau60"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial1.json"  "0.60"
run_experiment 14  "starcoder_p9_t1_tau70"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial1.json"  "0.70"

run_experiment 15  "starcoder_p9_t2_tau60"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial2.json"  "0.60"
run_experiment 16  "starcoder_p9_t2_tau70"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial2.json"  "0.70"

run_experiment 17  "starcoder_p9_t3_tau60"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial3.json"  "0.60"
run_experiment 18  "starcoder_p9_t3_tau70"  "$STARCODER_MODEL"  "results/cache/cache_starcoder_trial3.json"  "0.70"

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 9 COMPLETE"
echo "  Results in: $MASTER_CSV"
echo ""
echo "  Run labels:"
echo "    llama31_p9_t{1,2,3}_tau{60,70}   — Llama 3.1 8B"
echo "    deepseek_p9_t{1,2,3}_tau{60,70}  — DeepSeek Coder 6.7B"
echo "    starcoder_p9_t{1,2,3}_tau{60,70} — StarCoder2 7B"
echo ""
echo "  Combine with Phase 7 for full H3 picture:"
echo "    results/master_results_phase7.csv  — Code Llama + Qwen (already run)"
echo "    results/master_results_phase9.csv  — Llama 3.1 + DeepSeek + StarCoder"
echo ""
echo "  Next: send both CSVs and compute 95% CIs for all 5 models across H3 spectrum."
echo "=============================================================================="
