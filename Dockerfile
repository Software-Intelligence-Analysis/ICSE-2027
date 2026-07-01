# =============================================================================
# Dockerfile — Agentic LLMOps Experiment Runner
# =============================================================================
#
# This container runs the experiment scripts (src/agent.py, scripts/run_phase*.sh).
# It does NOT contain the LLM models themselves. Ollama runs in a separate
# container (see docker-compose.yml) and this container calls it over the
# internal Docker network.
#
# Build:  docker build -t llmops-runner .
# Run:    docker compose up   (preferred — handles Ollama service too)
# =============================================================================

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Results directory will be mounted as a volume (see docker-compose.yml)
# so experiment outputs persist after the container exits
RUN mkdir -p results/raw results/summary results/cache

# Default: print usage. Override in docker-compose or docker run command.
CMD ["bash", "-c", "echo 'Run: docker compose run runner bash scripts/run_phase10_cross_benchmark.sh'"]
