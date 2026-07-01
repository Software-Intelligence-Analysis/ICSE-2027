#!/usr/bin/env bash
# =============================================================================
# Phase 8 — Component Ablation Study
# =============================================================================
# Purpose: Isolate the contribution of each of the three changes made between
#          Phase 2 (net-negative) and Phase 3 (net-positive):
#
#   Change A — Fixer model upgrade:    Llama 3.2 3B  →  Llama 3.1 8B
#   Change B — Problem context added:  fixer sees problem spec + issues + code
#                                      (vs. issues + code only)
#   Change C — Selective reversion gate: fixer output re-scored; reverted if
#                                       gate_score >= pre_score
#
# Design: one-at-a-time ablation (OAT) starting from the full Phase 3 config.
#   Condition 0  — ALL ON    : 8B + context + gate   (Phase 3 winner — control)
#   Condition A  — FIXER 3B  : 3B + context + gate   (ablate model upgrade)
#   Condition B  — NO CONTEXT: 8B + no context + gate (ablate problem context)
#   Condition C  — NO GATE   : 8B + context + no gate (ablate gate)
#
# Shared planner cache: all 5 runs (1 baseline + 4 monitoring conditions) use
# the same pre-generated Code Llama outputs so the only variable is the fix
# strategy — not the planner. This makes the comparison clean.
#
# Model setup:
#   Planner : codellama:7b   (T=0.0, deterministic)
#   Critic  : llama3.1:8b   (T=0.0, always)
#   Fixer   : llama3.1:8b   (Conditions 0/B/C) OR llama3.2:3b (Condition A)
#
# Runtime estimate: ~25-30 min per monitoring run × 4 = ~2 hours total
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
PLANNER_MODEL="codellama:7b"
CRITIC_MODEL="llama3.1:8b"
FIXER_MODEL_8B="llama3.1:8b"
FIXER_MODEL_3B="llama3.2:3b"
TEMPERATURE=0.0
THRESHOLD=0.60
N=100
MASTER_CSV="results/master_results_phase8.csv"

# Shared planner cache — generated once from Phase 6 Code Llama T=0.0 run.
# All monitoring conditions evaluate the exact same planner outputs.
PLANNER_CACHE="results/cache/cache_p6_codellama.json"

echo "============================================================"
echo "  PHASE 8 — ABLATION STUDY"
echo "  Planner : $PLANNER_MODEL  (T=$TEMPERATURE)"
echo "  Critic  : $CRITIC_MODEL"
echo "  N=$N  threshold=$THRESHOLD"
echo "  Shared cache : $PLANNER_CACHE"
echo "  Master CSV   : $MASTER_CSV"
echo "============================================================"
echo ""

# ── Helper: flush Ollama between heavy runs to free VRAM ─────────────────────
flush_ollama() {
    echo "  [flush] Stopping Ollama models to free memory..."
    ollama stop "$FIXER_MODEL_8B" 2>/dev/null || true
    ollama stop "$FIXER_MODEL_3B" 2>/dev/null || true
    ollama stop "$CRITIC_MODEL"   2>/dev/null || true
    ollama stop "$PLANNER_MODEL"  2>/dev/null || true
    sleep 15
}

# =============================================================================
# STEP 0 — Baseline (no monitoring, shared cache)
# Run once — all monitoring conditions compare against this same baseline.
# =============================================================================
echo "------------------------------------------------------------"
echo "  STEP 0 — Baseline (shared across all conditions)"
echo "------------------------------------------------------------"
python3 src/agent.py \
    --version        baseline \
    --planner-model  "$PLANNER_MODEL" \
    --critic-model   "$CRITIC_MODEL" \
    --fixer-model    "$FIXER_MODEL_8B" \
    --temperature    "$TEMPERATURE" \
    --threshold      "$THRESHOLD" \
    --dataset-size   "$N" \
    --master-csv     "$MASTER_CSV" \
    --run-label      "p8_baseline" \
    --cache-planner-output "$PLANNER_CACHE"

flush_ollama

# =============================================================================
# STEP 1 — Condition 0: ALL ON (full Phase 3 config — the control)
# Fixer=8B, problem context=ON, gate=ON
# This is the configuration that produced net-positive results in Phase 3.
# =============================================================================
echo "------------------------------------------------------------"
echo "  STEP 1 — Condition 0: ALL ON (Phase 3 full config)"
echo "  8B fixer + problem context + gate"
echo "------------------------------------------------------------"
python3 src/agent.py \
    --version        monitoring \
    --planner-model  "$PLANNER_MODEL" \
    --critic-model   "$CRITIC_MODEL" \
    --fixer-model    "$FIXER_MODEL_8B" \
    --temperature    "$TEMPERATURE" \
    --threshold      "$THRESHOLD" \
    --dataset-size   "$N" \
    --master-csv     "$MASTER_CSV" \
    --run-label      "p8_all_on" \
    --cache-planner-output "$PLANNER_CACHE"

flush_ollama

# =============================================================================
# STEP 2 — Condition A: FIXER MODEL ABLATION (3B instead of 8B)
# Isolates the contribution of upgrading the fixer from 3B to 8B.
# Everything else stays ON.
# =============================================================================
echo "------------------------------------------------------------"
echo "  STEP 2 — Condition A: FIXER MODEL ABLATION"
echo "  3B fixer + problem context + gate"
echo "------------------------------------------------------------"
python3 src/agent.py \
    --version        monitoring \
    --planner-model  "$PLANNER_MODEL" \
    --critic-model   "$CRITIC_MODEL" \
    --fixer-model    "$FIXER_MODEL_3B" \
    --temperature    "$TEMPERATURE" \
    --threshold      "$THRESHOLD" \
    --dataset-size   "$N" \
    --master-csv     "$MASTER_CSV" \
    --run-label      "p8_ablate_fixer_3b" \
    --cache-planner-output "$PLANNER_CACHE"

flush_ollama

# =============================================================================
# STEP 3 — Condition B: PROBLEM CONTEXT ABLATION (no context in fixer prompt)
# Isolates the contribution of giving the fixer the problem specification.
# Fixer only sees critic issues + broken code.
# Gate stays ON, fixer stays 8B.
# =============================================================================
echo "------------------------------------------------------------"
echo "  STEP 3 — Condition B: PROBLEM CONTEXT ABLATION"
echo "  8B fixer + NO problem context + gate"
echo "------------------------------------------------------------"
python3 src/agent.py \
    --version           monitoring \
    --planner-model     "$PLANNER_MODEL" \
    --critic-model      "$CRITIC_MODEL" \
    --fixer-model       "$FIXER_MODEL_8B" \
    --temperature       "$TEMPERATURE" \
    --threshold         "$THRESHOLD" \
    --dataset-size      "$N" \
    --master-csv        "$MASTER_CSV" \
    --run-label         "p8_ablate_no_context" \
    --no-fixer-context \
    --cache-planner-output "$PLANNER_CACHE"

flush_ollama

# =============================================================================
# STEP 4 — Condition C: GATE ABLATION (gate disabled)
# Isolates the contribution of the selective reversion gate.
# Fixer output is always kept — no re-scoring, no reversion.
# Fixer stays 8B, context stays ON.
# =============================================================================
echo "------------------------------------------------------------"
echo "  STEP 4 — Condition C: GATE ABLATION"
echo "  8B fixer + problem context + NO gate"
echo "------------------------------------------------------------"
python3 src/agent.py \
    --version        monitoring \
    --planner-model  "$PLANNER_MODEL" \
    --critic-model   "$CRITIC_MODEL" \
    --fixer-model    "$FIXER_MODEL_8B" \
    --temperature    "$TEMPERATURE" \
    --threshold      "$THRESHOLD" \
    --dataset-size   "$N" \
    --master-csv     "$MASTER_CSV" \
    --run-label      "p8_ablate_no_gate" \
    --no-gate \
    --cache-planner-output "$PLANNER_CACHE"

flush_ollama

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "  PHASE 8 COMPLETE"
echo "  Results in: $MASTER_CSV"
echo ""
echo "  Conditions run:"
echo "    p8_baseline          — no monitoring (control)"
echo "    p8_all_on            — full Phase 3 config"
echo "    p8_ablate_fixer_3b   — 3B fixer (change A off)"
echo "    p8_ablate_no_context — no problem context (change B off)"
echo "    p8_ablate_no_gate    — no gate (change C off)"
echo ""
echo "  Interpretation:"
echo "    gap(all_on - ablate_X) = contribution of change X"
echo "    Largest gap = most important change"
echo "============================================================"
