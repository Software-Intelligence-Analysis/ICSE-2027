#!/bin/bash
# =============================================================================
#  run_experiments.sh — Phase 3 Ablation Study
#
#  WHAT THIS RUNS:
#    Phase 3 experiment matrix: 8B critic + 8B fixer + enhanced prompt +
#    selective reversion gate. 12 conditions × 3 trials = 36 runs.
#
#    Conditions:
#      Temperatures  : 0.0, 0.3, 0.7  (Planner only — critic/fixer always T=0.0)
#      Thresholds    : 0.4, 0.5, 0.6  (monitoring only)
#      Variants      : baseline, monitoring
#
#  USAGE:
#    chmod +x run_experiments.sh      # first time only
#    ./run_experiments.sh             # runs all 36 conditions
#
#  TIME ESTIMATE: ~12 hours (based on Phase 2 timing with 8B critic on local GPU)
#  TIP: Run overnight. Results accumulate in results/master_results.csv live.
#
#  MODELS:
#    Planner : llama3.2:3b
#    Critic  : llama3.1:8b   (always T=0.0)
#    Fixer   : llama3.1:8b   (always T=0.0)
# =============================================================================

set -e   # stop on any unexpected error

PYTHON="python3 src/agent.py"
N=100
TRIALS=3
PLANNER="llama3.2:3b"
CRITIC="llama3.1:8b"
FIXER="llama3.1:8b"
MASTER_CSV="results/master_results_phase3.csv"
RESUME_FROM=19   # Set this to skip already-completed runs (e.g. 10 to resume from run 10)

TOTAL_RUNS=$(( (3 + 9) * TRIALS ))   # 12 conditions × 3 trials = 36

echo ""
echo "============================================================"
echo "  LLMOps Phase 3 — Ablation Study"
echo "  Dataset size : $N HumanEval problems per run"
echo "  Trials       : $TRIALS per condition"
echo "  Total runs   : $TOTAL_RUNS"
echo "  Models       : Planner=$PLANNER | Critic=$CRITIC | Fixer=$FIXER"
echo "  Results      : $MASTER_CSV"
echo "============================================================"
echo ""

RUN=0

for TRIAL in 1 2 3; do

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  TRIAL $TRIAL of $TRIALS"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Baselines (control — no fixer, threshold irrelevant) ────────────────────

  for TEMP in 0.0 0.3 0.7; do
    RUN=$(( RUN + 1 ))
    [ $RUN -lt $RESUME_FROM ] && echo ">>> [$RUN/$TOTAL_RUNS] Skipping (already done)" && continue
    LABEL="phase3-baseline-t${TEMP}-r${TRIAL}"
    echo ""
    echo ">>> [$RUN/$TOTAL_RUNS] Baseline — temp=$TEMP | trial=$TRIAL"
    $PYTHON \
      --temperature $TEMP \
      --threshold 0.6 \
      --dataset-size $N \
      --version baseline \
      --planner-model $PLANNER \
      --critic-model  $CRITIC \
      --fixer-model   $FIXER \
      --master-csv    "$MASTER_CSV" \
      --run-label "$LABEL"
  done

  # ── Monitoring (treatment — full pipeline with selective reversion gate) ─────

  for TEMP in 0.0 0.3 0.7; do
    for THRESH in 0.4 0.5 0.6; do
      RUN=$(( RUN + 1 ))
      [ $RUN -lt $RESUME_FROM ] && echo ">>> [$RUN/$TOTAL_RUNS] Skipping (already done)" && continue
      LABEL="phase3-mon-t${TEMP}-th${THRESH}-r${TRIAL}"
      echo ""
      echo ">>> [$RUN/$TOTAL_RUNS] Monitoring — temp=$TEMP | threshold=$THRESH | trial=$TRIAL"
      $PYTHON \
        --temperature $TEMP \
        --threshold $THRESH \
        --dataset-size $N \
        --version monitoring \
        --planner-model $PLANNER \
        --critic-model  $CRITIC \
        --fixer-model   $FIXER \
        --run-label "$LABEL"
    done
  done

done


# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  ALL $TOTAL_RUNS RUNS COMPLETE"
echo ""
echo "  Results:"
echo "    results/master_results.csv   ← open in Excel for Table 1"
echo "    results/*.csv                ← per-run detail files"
echo ""
echo "  MLflow UI:"
echo "    mlflow ui --backend-store-uri sqlite:///mlruns.db"
echo "    Open http://127.0.0.1:5000"
echo "============================================================"
echo ""
