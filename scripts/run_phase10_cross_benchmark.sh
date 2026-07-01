#!/usr/bin/env bash
# =============================================================================
# Phase 10 — Cross-Benchmark Generalization + Phase 9b Cache Regeneration
# =============================================================================
#
# This phase has two goals:
#
#   GOAL 1: Phase 9b — genuine multi-trial CIs for the middle three models
#   -----------------------------------------------------------------------
#   Phase 9 was degenerate because the planner caches were built at T=0.0.
#   Phase 10 regenerates those caches at T=0.3 and runs 3 truly independent
#   trials for Llama 3.1 8B, DeepSeek Coder 6.7B, and StarCoder2 7B.
#   This fills in the confidence intervals missing from the H3 curve.
#
#   Models:   Llama 3.1 8B, DeepSeek Coder 6.7B, StarCoder2 7B
#   Design:   3 independent trials × 2 tau values (0.60, 0.70)
#   N:        100 problems per trial (HumanEval)
#
#   GOAL 2: Cross-benchmark generalization (HumanEval+ and MBPP)
#   -----------------------------------------------------------------------
#   All Phases 1-9 used HumanEval-100. Phase 10 tests whether the Inverse
#   Capability Hypothesis holds on harder benchmarks:
#
#   HumanEval+ (liu2024your): 164 problems — extends HumanEval with extra
#     test cases that catch edge-case failures missed by the original tests.
#     Code Llama's +32pp gain may partially reflect HumanEval's bias toward
#     syntactically simple failures. HumanEval+ tests whether the fixer
#     actually fixes logic, not just syntax.
#
#   MBPP (austin2021program): 374 problems — broader Python programming
#     benchmark with different problem distribution than HumanEval.
#     Tests generalization of H3 beyond the specific benchmark.
#
#   Models:   Code Llama 7B (expected benefit) + Qwen2.5 Coder 7B (expected loss)
#   Design:   Single trial per condition (establishes pattern), then 3-trial
#             CI if pattern holds
#   N:        100 problems per run (subsampled for speed)
#
# Prerequisites:
#   - Ollama running (either locally or via docker-compose)
#   - Models pulled: llama3.1:8b, deepseek-coder:6.7b, starcoder2:7b,
#                    codellama:7b, qwen2.5-coder:7b
#   - HumanEval+ dataset: pip install evalplus
#   - MBPP dataset: pip install datasets (HuggingFace)
#
# Usage:
#   # Local (laptop/GPU machine with Ollama already running):
#   bash scripts/run_phase10_cross_benchmark.sh
#
#   # Docker (professor's GPU machine):
#   docker compose up --build
#   (runs this script inside the container automatically)
#
#   # Resume after interruption:
#   RESUME_FROM=<run number 1-N> bash scripts/run_phase10_cross_benchmark.sh
#
# Runtime estimate (GPU machine with 12GB+ VRAM):
#   Phase 9b cache generation: ~2h  (300 planner completions at T=0.3)
#   Phase 9b monitoring runs:  ~3h  (18 runs × ~10min each)
#   Phase 10 benchmarks:       ~4h  (8 benchmark conditions × ~30min each)
#   Total:                     ~9h  (recommend overnight)
#
# Runtime estimate (CPU only):
#   Multiply all estimates × 8-10x. Not recommended.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RESUME_FROM=${RESUME_FROM:-1}
RUN_COUNTER=0

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_MODEL="llama3.1:8b"
DEEPSEEK_MODEL="deepseek-coder:6.7b"
STARCODER_MODEL="starcoder2:7b"
CODELLAMA_MODEL="codellama:7b"
QWEN_MODEL="qwen2.5-coder:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL="llama3.1:8b"

T_PLANNER="0.3"
T_DETERMINISTIC="0.0"
DATASET_SIZE="100"

CACHE_DIR="results/cache"
RESULTS_DIR="results"
PHASE9B_CSV="results/summary/master_results_phase9b.csv"
PHASE10_CSV="results/summary/master_results_phase10.csv"

mkdir -p "$CACHE_DIR" results/raw results/summary

# ── Helpers ───────────────────────────────────────────────────────────────────
flush_ollama() {
    echo "  [flush] Unloading models from Ollama to free VRAM..."
    for m in "$LLAMA_MODEL" "$DEEPSEEK_MODEL" "$STARCODER_MODEL" "$CODELLAMA_MODEL" "$QWEN_MODEL"; do
        ollama stop "$m" 2>/dev/null || true
    done
    sleep 20
    echo "  [flush] Done."
}

check_run() {
    RUN_COUNTER=$((RUN_COUNTER + 1))
    if [ "$RUN_COUNTER" -lt "$RESUME_FROM" ]; then
        echo "  [Run $RUN_COUNTER] Skipping (already completed)."
        return 1
    fi
    return 0
}

run_monitoring() {
    local RUN_NUM=$1
    local LABEL=$2
    local PLANNER=$3
    local CACHE=$4
    local TAU=$5
    local MASTER_CSV=$6

    check_run || return

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Run $RUN_NUM — $LABEL  [tau=$TAU]"
    echo "  Planner: $PLANNER | T=$T_PLANNER | Cache: $(basename $CACHE)"
    echo "──────────────────────────────────────────────────────────────────────"

    python3 src/agent.py \
        --planner-model        "$PLANNER"           \
        --critic-model         "$CRITIC_MODEL"      \
        --fixer-model          "$FIXER_MODEL"       \
        --temperature          "$T_PLANNER"         \
        --threshold            "$TAU"               \
        --dataset-size         "$DATASET_SIZE"      \
        --version              both                 \
        --run-label            "$LABEL"             \
        --master-csv           "$MASTER_CSV"        \
        --cache-planner-output "$CACHE"

    flush_ollama
}

# =============================================================================
# PART 1: PHASE 9B — REGENERATE CACHES AT T=0.3, THEN RUN CI STUDY
# =============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 9B: GENUINE MULTI-TRIAL CI STUDY"
echo "  Regenerating planner caches at T=0.3 for 3 middle models"
echo "  N=$DATASET_SIZE | 3 trials × 2 tau values per model"
echo "=============================================================================="

# ── Step 1: Generate fresh T=0.3 caches (this is what Phase 9 was missing) ───
echo ""
echo "  ── Generating T=0.3 planner caches ──"

generate_cache() {
    local MODEL=$1
    local CACHE_OUT=$2

    if [ -f "$CACHE_OUT" ]; then
        echo "  [cache] $CACHE_OUT already exists — skipping."
        return
    fi
    echo "  [cache] Generating: $CACHE_OUT"
    python3 src/agent.py \
        --planner-model     "$MODEL"            \
        --temperature       "$T_PLANNER"        \
        --dataset-size      "$DATASET_SIZE"     \
        --version           baseline            \
        --run-label         "cache_gen_$(basename $CACHE_OUT .json)" \
        --cache-planner-output "$CACHE_OUT"
    flush_ollama
}

generate_cache "$LLAMA_MODEL"    "$CACHE_DIR/cache_p10_llama31_t1.json"
generate_cache "$LLAMA_MODEL"    "$CACHE_DIR/cache_p10_llama31_t2.json"
generate_cache "$LLAMA_MODEL"    "$CACHE_DIR/cache_p10_llama31_t3.json"
generate_cache "$DEEPSEEK_MODEL" "$CACHE_DIR/cache_p10_deepseek_t1.json"
generate_cache "$DEEPSEEK_MODEL" "$CACHE_DIR/cache_p10_deepseek_t2.json"
generate_cache "$DEEPSEEK_MODEL" "$CACHE_DIR/cache_p10_deepseek_t3.json"
generate_cache "$STARCODER_MODEL" "$CACHE_DIR/cache_p10_starcoder_t1.json"
generate_cache "$STARCODER_MODEL" "$CACHE_DIR/cache_p10_starcoder_t2.json"
generate_cache "$STARCODER_MODEL" "$CACHE_DIR/cache_p10_starcoder_t3.json"

echo ""
echo "  ── Running Phase 9b monitoring (18 runs) ──"

# Llama 3.1 8B — 3 trials × 2 tau
run_monitoring  1  "llama31_p9b_t1_tau60" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t1.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring  2  "llama31_p9b_t1_tau70" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t1.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring  3  "llama31_p9b_t2_tau60" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t2.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring  4  "llama31_p9b_t2_tau70" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t2.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring  5  "llama31_p9b_t3_tau60" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t3.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring  6  "llama31_p9b_t3_tau70" "$LLAMA_MODEL"  "$CACHE_DIR/cache_p10_llama31_t3.json"  "0.70"  "$PHASE9B_CSV"

# DeepSeek Coder 6.7B — 3 trials × 2 tau
run_monitoring  7  "deepseek_p9b_t1_tau60" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t1.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring  8  "deepseek_p9b_t1_tau70" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t1.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring  9  "deepseek_p9b_t2_tau60" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t2.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring 10  "deepseek_p9b_t2_tau70" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t2.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring 11  "deepseek_p9b_t3_tau60" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t3.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring 12  "deepseek_p9b_t3_tau70" "$DEEPSEEK_MODEL"  "$CACHE_DIR/cache_p10_deepseek_t3.json"  "0.70"  "$PHASE9B_CSV"

# StarCoder2 7B — 3 trials × 2 tau (expected floor: 0pp, but let's confirm with real variance)
run_monitoring 13  "starcoder_p9b_t1_tau60" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t1.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring 14  "starcoder_p9b_t1_tau70" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t1.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring 15  "starcoder_p9b_t2_tau60" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t2.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring 16  "starcoder_p9b_t2_tau70" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t2.json"  "0.70"  "$PHASE9B_CSV"
run_monitoring 17  "starcoder_p9b_t3_tau60" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t3.json"  "0.60"  "$PHASE9B_CSV"
run_monitoring 18  "starcoder_p9b_t3_tau70" "$STARCODER_MODEL"  "$CACHE_DIR/cache_p10_starcoder_t3.json"  "0.70"  "$PHASE9B_CSV"

echo ""
echo "  Phase 9b complete. Results in: $PHASE9B_CSV"
echo "  Run compute_phase7_ci.py on this file to get CIs for all 5 models."


# =============================================================================
# PART 2: PHASE 10 — CROSS-BENCHMARK (HumanEval+ and MBPP)
# =============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 10: CROSS-BENCHMARK GENERALIZATION"
echo "  HumanEval+ (164 problems) and MBPP (100 subsampled)"
echo "  Models: Code Llama 7B (expected benefit) + Qwen2.5 Coder 7B (expected loss)"
echo "  tau=0.60 (primary) and tau=0.70 (comparison)"
echo "=============================================================================="

run_benchmark() {
    local RUN_NUM=$1
    local LABEL=$2
    local PLANNER=$3
    local TAU=$4
    local BENCHMARK=$5   # humaneval_plus | mbpp

    check_run || return

    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Run $RUN_NUM — $LABEL  [tau=$TAU, benchmark=$BENCHMARK]"
    echo "──────────────────────────────────────────────────────────────────────"

    python3 src/agent.py \
        --planner-model  "$PLANNER"          \
        --critic-model   "$CRITIC_MODEL"     \
        --fixer-model    "$FIXER_MODEL"      \
        --temperature    "$T_DETERMINISTIC"  \
        --threshold      "$TAU"              \
        --dataset-size   "$DATASET_SIZE"     \
        --benchmark      "$BENCHMARK"        \
        --version        both                \
        --run-label      "$LABEL"            \
        --master-csv     "$PHASE10_CSV"

    flush_ollama
}

echo ""
echo "  ── HumanEval+ (tau=0.60 and 0.70) ──"
run_benchmark 19  "codellama_p10_hep_tau60"  "$CODELLAMA_MODEL"  "0.60"  "humaneval_plus"
run_benchmark 20  "codellama_p10_hep_tau70"  "$CODELLAMA_MODEL"  "0.70"  "humaneval_plus"
run_benchmark 21  "qwen_p10_hep_tau60"       "$QWEN_MODEL"       "0.60"  "humaneval_plus"
run_benchmark 22  "qwen_p10_hep_tau70"       "$QWEN_MODEL"       "0.70"  "humaneval_plus"

echo ""
echo "  ── MBPP (tau=0.60 and 0.70) ──"
run_benchmark 23  "codellama_p10_mbpp_tau60"  "$CODELLAMA_MODEL"  "0.60"  "mbpp"
run_benchmark 24  "codellama_p10_mbpp_tau70"  "$CODELLAMA_MODEL"  "0.70"  "mbpp"
run_benchmark 25  "qwen_p10_mbpp_tau60"       "$QWEN_MODEL"       "0.60"  "mbpp"
run_benchmark 26  "qwen_p10_mbpp_tau70"       "$QWEN_MODEL"       "0.70"  "mbpp"


# =============================================================================
# Done
# =============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE 10 COMPLETE"
echo ""
echo "  Phase 9b results: $PHASE9B_CSV"
echo "  Phase 10 results: $PHASE10_CSV"
echo ""
echo "  Next steps:"
echo "    1. python3 src/compute_phase7_ci.py --input $PHASE9B_CSV"
echo "       (computes 95% CIs for Llama 3.1, DeepSeek, StarCoder2)"
echo "    2. Compare Phase 10 HumanEval+ vs HumanEval results"
echo "       to assess benchmark specificity of Code Llama +32pp gain"
echo "    3. Update paper: results.tex, discussion.tex, conclusion.tex"
echo "=============================================================================="
