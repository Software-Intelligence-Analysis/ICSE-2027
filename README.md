# An Empirical Study of Runtime Hallucination Monitoring in Multi-Agent Code Generation: Benefits Decline with Model Capability

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?logo=mlflow)

> **Replication package** for this paper, submitted to ICSE 2027.

---

## Overview

This repository contains the full experimental implementation for a study of runtime hallucination monitoring in multi-agent code generation. The core question: can a separate critic agent drive a separate fixer agent to improve code correctness in a single pass, at small model scales (3B--8B parameters), without retraining?

The central finding is the **Inverse Capability Hypothesis (H3)**: monitoring benefit declines with baseline model capability. Low-capability models gain significantly (Code Llama 7B: +30.3 pp, p=0.002); high-capability models lose (Qwen2.5 Coder 7B: -4.0 pp, p=0.020). The practical deployment threshold is approximately 60% baseline pass@1.

---

## Pipeline Architecture

Two variants run side-by-side for every experimental condition:

**Baseline**
```
Planner --> Post-Critic
```

**Monitoring Pipeline**
```
Planner --> Pre-Critic --> [Fixer if score > τ] --> Reversion Gate --> Post-Critic
```

The selective reversion gate accepts a fixer output only when the same critic's post-fix score strictly improves over the pre-fix score. It reverted 73.4% of fixer attempts in Phase 3 and provides a structural "do no harm" guarantee; problem context and fixer scale are what make monitoring net-positive.

| Agent | Role |
|---|---|
| Planner | Generates a Python function from the problem statement |
| Pre-Critic | Scores hallucination risk (0.0 to 1.0) and identifies issues |
| Fixer | Rewrites code when Pre-Critic score exceeds threshold τ |
| Post-Critic | Scores the final output (same prompt as Pre-Critic) |

---

## Key Results

| Model | Baseline pass@1 | Monitoring pass@1 | Δpp | 95% CI | p |
|---|---|---|---|---|---|
| Code Llama 7B | 20% | 50% | +30.3 | [+24.1, +36.6] | 0.002 |
| Llama 3.1 8B | 65% | 65.7% | +0.7 | [-2.2, +3.5] | 0.423 |
| DeepSeek Coder 6.7B | 74% | 72.3% | -1.7 | [-4.5, +1.2] | 0.130 |
| Qwen2.5 Coder 7B | 90% | 86.0% | -4.0 | [-6.5, -1.5] | 0.020 |

Critic and Fixer fixed at Llama 3.1 8B throughout. τ=0.60, T=0.3, N=100, 3 trials for CI rows.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM Inference | Ollama (local, no API) |
| Agent Orchestration | LangGraph (StateGraph) |
| Experiment Tracking | MLflow (SQLite backend) |
| Evaluation Benchmark | HumanEval, HumanEval+, MBPP (100 problems each) |
| Language | Python 3.12 |

---

## Replication

Two paths are provided. The local path (Path 1) matches how experiments were run as described in the paper. The Docker path (Path 2) is an automated convenience wrapper that handles environment setup for you.

---

### Path 1: Local (Ollama already installed)

**Requirements:** Python 3.12, [Ollama](https://ollama.com) running locally.

```bash
git clone [anonymous repository]
cd agentic-llmops

# Create virtual environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Pull models
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama pull codellama:7b
ollama pull deepseek-coder:6.7b
ollama pull qwen2.5-coder:7b
ollama pull starcoder2:7b
```

Run a single condition to verify everything works:
```bash
python3 src/agent.py \
  --planner-model codellama:7b \
  --critic-model llama3.1:8b \
  --fixer-model llama3.1:8b \
  --temperature 0.0 \
  --threshold 0.60 \
  --dataset-size 10 \
  --version both
```

Then run any full phase:
```bash
bash scripts/run_phase7.sh
```

To resume after interruption:
```bash
RESUME_FROM=10 bash scripts/run_phase7.sh
```

View MLflow results:
```bash
mlflow ui --backend-store-uri sqlite:///results/tracking.db
# Open http://127.0.0.1:5000
```

---

### Path 2: Docker (automated)

This path automates environment setup — Docker handles all dependencies and Ollama runs as a separate container. It is not how the original experiments were run, but produces identical results.

**Requirements:** Docker, Docker Compose, and ideally an NVIDIA GPU with 12 GB+ VRAM and `nvidia-container-toolkit` installed. CPU-only works but is 8–10x slower.

**Step 1 — Clone and enter the repo**
```bash
git clone [anonymous repository]
cd agentic-llmops
```

**Step 2 — Start the Ollama service**
```bash
docker compose up ollama -d
```
Wait ~20 seconds for Ollama to be ready, then verify:
```bash
curl http://localhost:11434/api/tags
```

**Step 3 — Pull the required models** (one-time, ~20 GB total)
```bash
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull codellama:7b
docker compose exec ollama ollama pull deepseek-coder:6.7b
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull starcoder2:7b
```

**Step 4 — Build the experiment runner**
```bash
docker compose build runner
```

**Step 5 — Run the phase you want to replicate**

Each phase maps to a script. Run any phase with:
```bash
docker compose run runner bash scripts/<script_name>.sh
```

| Paper Section | Phase | Script | Runtime estimate (GPU) |
|---|---|---|---|
| Section V-A | Phase 3 (main ablation) | `run_experiments.sh` | ~12 hours |
| Section V-C | Phase 4b (Code Llama cross-model) | `run_phase4b.sh` | ~2 hours |
| Section V-C | Phase 5 (DeepSeek + Qwen) | `run_phase5.sh` | ~3 hours |
| Section V-D | Phase 6 (threshold sweep) | `run_phase6.sh` | ~4 hours |
| Section V-E | Phase 7 (H3 confidence intervals) | `run_phase7.sh` | ~6 hours |
| Section V-F | Phase 8 (component ablation) | `run_phase8_ablation.sh` | ~2 hours |
| Section V-G | Phase 9b + 10 (cross-benchmark) | `run_phase10_cross_benchmark.sh` | ~9 hours |

Example — replicate the Phase 7 H3 confidence intervals:
```bash
docker compose run runner bash scripts/run_phase7.sh
```

**Step 6 — Results land on your host machine**

All CSVs are written to `./results/` on your host (Docker mounts this as a volume). Raw per-run CSVs go to `results/raw/`; aggregated summaries go to `results/summary/`.

**Step 7 — Resume after interruption**

All scripts support `RESUME_FROM` to skip already-completed runs:
```bash
RESUME_FROM=10 docker compose run runner bash scripts/run_phase7.sh
```

**Step 8 — View results in MLflow**
```bash
docker compose run -p 5000:5000 runner mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///results/tracking.db
# Open http://localhost:5000
```

**CPU-only machines:** Use `docker-compose.cpu.yml` instead (omits the nvidia `deploy` section):
```bash
docker compose -f docker-compose.cpu.yml up ollama -d
docker compose -f docker-compose.cpu.yml run runner bash scripts/run_phase7.sh
```

---

### Verifying Results Against the Paper

The pre-computed results from all phases are included in `results/summary/`. To verify a number from the paper without re-running experiments:

| Paper claim | File | Key column |
|---|---|---|
| Code Llama +30.3 pp (Table III) | `results/summary/master_results_phase7.csv` | `mean_delta`, `ci_lower`, `ci_upper` |
| Qwen -4.0 pp (Table III) | `results/summary/master_results_phase7.csv` | `mean_delta`, `ci_lower`, `ci_upper` |
| Phase 3 +4.85 pp mean (Section V-B) | `results/summary/master_results_phase3.csv` | `delta_pass_at_1` |
| Threshold sweep (Table II) | `results/summary/master_results_phase6.csv` | `delta_pass_at_1`, `trigger_rate` |
| Component ablation (Table IV) | `results/summary/master_results_phase8.csv` | `delta_pass_at_1` |
| Cross-benchmark (Table V) | `results/summary/master_results_phase10.csv` | `delta_pass_at_1` |

Confidence intervals can be recomputed from raw CSVs using:
```bash
python3 src/compute_phase7_ci.py
```

---

## Repository Structure

```
agentic-llmops/
├── src/
│   ├── agent.py                   # Pipeline: Planner, Critic, Fixer, reversion gate
│   └── compute_phase7_ci.py       # Confidence interval computation (Phase 7)
│
├── scripts/
│   ├── run_experiments.sh         # Phases 1--3 ablation runner
│   ├── run_phase4.sh
│   ├── run_phase4b.sh
│   ├── run_phase5.sh
│   ├── run_phase6.sh
│   ├── run_phase7.sh
│   ├── run_phase8_ablation.sh
│   ├── run_phase9_full_model_ci.sh
│   └── run_phase10_cross_benchmark.sh
│
├── results/
│   ├── raw/                       # Timestamped CSV for every individual run
│   ├── cache/                     # Planner output cache (per model, per trial)
│   └── summary/                   # master_results_phase*.csv aggregated files
│
├── notebooks/
│   └── phase10_colab_setup.ipynb  # Colab setup for Phase 10 cross-benchmark runs
│
├── paper/                         # LaTeX source
│   ├── main.tex
│   ├── references.bib
│   ├── sections/
│   └── figures/
│
├── Research/
│   └── papers/                    # Reference PDFs for cited work
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Experimental Phases

| Phase | Description | Key Result |
|---|---|---|
| 1 | Pilot (N=50, 3B models) | Measurement artifact identified and fixed |
| 2a / 2b | Critic scale ablation (3B vs 8B critic, 3B fixer) | H1 rejected; fixer is the bottleneck |
| 3 | Full pipeline (8B fixer, context prompt, reversion gate) | H2 confirmed: +4.85 pp mean (p=0.010) |
| 4 / 4b | Cross-model: Llama 3.1 8B, Code Llama 7B | First evidence for H3 |
| 5 | Cross-model: DeepSeek Coder, Qwen2.5 Coder | H3 pattern holds |
| 6 | Threshold sweep (τ = 0.60, 0.70, 0.75) | τ=0.70 is practical optimum |
| 7 | Confidence intervals for H3 endpoints (3 trials, T=0.3) | H3 confirmed statistically |
| 8 | Component ablation (one-at-a-time) | Context +4 pp; model +1 pp; gate -1 pp |
| 9b | Full CI coverage for intermediate models | Mid-capability results are indistinguishable from zero |
| 10 | Cross-benchmark: HumanEval+, MBPP | H3 holds on HumanEval+; MBPP Code Llama confounded |

---

## References

- Chen et al. (2021). [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- Liu et al. (2023). [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210) (EvalPlus)
- Shinn et al. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- Huang et al. (2023). [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
- Geifman & El-Yaniv (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
