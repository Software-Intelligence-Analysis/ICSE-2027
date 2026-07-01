import os
import sys
import ast
import csv
import time
import re
import json
import logging
import warnings
import argparse
import signal
import builtins
from io import StringIO
from datetime import datetime

import mlflow
from datasets import load_dataset
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# ========================= CLI ARGUMENTS =========================
# Run with defaults:          python3 src/agent.py
# Custom temperature/thresh:  python3 src/agent.py --temperature 0.7 --threshold 0.4
# Only baseline:              python3 src/agent.py --version baseline
# Full docs:                  python3 src/agent.py --help

parser = argparse.ArgumentParser(
    description="LLMOps Hallucination Monitoring — Ablation Runner",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--temperature",       type=float, default=0.0,
                    help="LLM sampling temperature (0.0 = deterministic)")
parser.add_argument("--retry-temperature", type=float, default=None,
                    help="Temperature for iterative retry attempt. Defaults to --temperature "
                         "if not set. Set to 0.3 or 0.5 to get genuine diversity on pass@2.")
parser.add_argument("--threshold",      type=float, default=0.6,
                    help="Hallucination score above which the Fixer triggers (0–1)")
parser.add_argument("--dataset-size",   type=int,   default=20,
                    help="Number of HumanEval problems to evaluate")
parser.add_argument("--version",        choices=["baseline", "monitoring", "both"],
                    default="both",
                    help="Which pipeline variant(s) to run")
parser.add_argument("--run-label",      type=str,   default="",
                    help="Short label for this run (e.g. 'run1', 'high-temp') — shows in CSV & MLflow")
parser.add_argument("--planner-model",  type=str,   default="llama3.2:3b",
                    help="Ollama model name for the Planner agent")
parser.add_argument("--critic-model",   type=str,   default="llama3.1:8b",
                    help="Ollama model name for the Critic agent (pre + post)")
parser.add_argument("--fixer-model",    type=str,   default="llama3.1:8b",
                    help="Ollama model name for the Fixer agent")
parser.add_argument("--master-csv",     type=str,   default="results/master_results.csv",
                    help="Path to the master CSV file (append mode). Use a phase-specific "
                         "path to avoid column mismatch with prior runs (e.g. "
                         "results/master_results_phase3.csv)")
parser.add_argument("--use-fewshot",    action="store_true",
                    help="Include few-shot examples in Planner prompt")
parser.add_argument("--use-iterative",  action="store_true",
                    help="Enable iterative re-generation on failed tests (pass@1 vs pass@2)")
parser.add_argument("--cache-planner-output", type=str, default="",
                    help="Path to cache file for Planner output. If provided and exists, loads from cache. "
                         "Otherwise, generates and saves to cache.")
# ── Phase 8 Ablation flags ────────────────────────────────────────────────────
# These let us isolate the contribution of each Phase 3 improvement.
# Default is ON for all (= full Phase 3 config). Turn one off per ablation run.
parser.add_argument("--no-gate", action="store_true",
                    help="Phase 8 ablation: disable selective reversion gate — fixer output is always "
                         "kept regardless of post-fix critic score")
parser.add_argument("--no-fixer-context", action="store_true",
                    help="Phase 8 ablation: remove problem specification from fixer prompt — fixer "
                         "only sees the critic issues and the broken code, not the original problem")
# ── Phase 10 cross-benchmark flag ─────────────────────────────────────────────
parser.add_argument("--benchmark", type=str, default="humaneval",
                    choices=["humaneval", "humaneval_plus", "mbpp"],
                    help="Benchmark dataset to evaluate on (default: humaneval). "
                         "humaneval_plus requires evalplus. mbpp uses google-research-datasets/mbpp.")
args = parser.parse_args()


# ========================= SETUP =========================

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlruns.db')}")
mlflow.set_experiment("agentic-llmops-hallucination-monitoring")

PLANNER_MODEL           = args.planner_model
CRITIC_MODEL            = args.critic_model
FIXER_MODEL             = args.fixer_model
TEMPERATURE             = args.temperature
RETRY_TEMPERATURE       = args.retry_temperature if args.retry_temperature is not None else args.temperature
HALLUCINATION_THRESHOLD = args.threshold
DATASET_SIZE            = args.dataset_size
USE_FEWSHOT             = args.use_fewshot
USE_ITERATIVE           = args.use_iterative
CACHE_PLANNER_OUTPUT    = args.cache_planner_output
USE_GATE                = not args.no_gate          # Phase 8: selective reversion gate on/off
USE_FIXER_CONTEXT       = not args.no_fixer_context # Phase 8: problem spec in fixer prompt on/off
BENCHMARK               = args.benchmark            # Phase 10: which benchmark to evaluate on

# Separate LLM instances per agent role — Phase 2 redesign:
# 3B Planner (lightweight generation), 8B Critic (reliable signal),
# 8B Fixer (capable enough to act on the critic's feedback).
# Critic and Fixer always run at T=0.0 for deterministic, stable scoring —
# only the Planner varies temperature across experimental conditions.
# llm_planner_retry uses RETRY_TEMPERATURE for the iterative second attempt,
# allowing genuine output diversity on pass@2 when retry_temperature > 0.
# Respect OLLAMA_HOST env var so Docker containers can point at the Ollama service
# (docker-compose sets OLLAMA_HOST=http://ollama:11434; defaults to localhost otherwise)
_OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

llm_planner       = ChatOllama(model=PLANNER_MODEL, temperature=TEMPERATURE,       num_predict=1024, base_url=_OLLAMA_BASE)
llm_planner_retry = ChatOllama(model=PLANNER_MODEL, temperature=RETRY_TEMPERATURE, num_predict=1024, base_url=_OLLAMA_BASE)
llm_critic        = ChatOllama(model=CRITIC_MODEL,  temperature=0.0,               num_predict=128,  base_url=_OLLAMA_BASE)
llm_fixer         = ChatOllama(model=FIXER_MODEL,   temperature=0.0,               num_predict=1024, base_url=_OLLAMA_BASE)

# Results directory — all CSVs land here
os.makedirs("results", exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_CSV    = args.master_csv


# ========================= HELPERS =========================

def extract_code(text):
    """Pull out ```python ... ``` block, or return raw text if no block found."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_score(text):
    """Parse 'SCORE: 0.7' from critic output.
    Defaults to 0.0 (assume clean) if the model didn't follow the format.
    We prefer false negatives (missing a hallucination) over false positives
    (triggering the fixer on good code and potentially breaking it).
    """
    match = re.search(r"SCORE:\s*([0-9.]+)", text)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(1))))


def extract_issues(text):
    """Pull out the ISSUES: block from critic output."""
    match = re.search(r"ISSUES:(.*?)(?:SCORE:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else "No specific issues identified."


def _extract_mbpp_entry_point(test_list):
    """Extract the function name from MBPP test assertions like 'assert func_name(...)'."""
    for test in test_list:
        m = re.search(r'assert\s+(\w+)\s*\(', test)
        if m:
            return m.group(1)
    return "solution"


def static_analysis(code: str):
    """
    Layer 1 of the hybrid critic — deterministic, AST-based analysis.
    Catches hard errors that provably cause test failures without any
    risk of LLM hallucination in the scoring.

    Checks:
      1. Syntax validity (ast.parse)
      2. Undefined names — used in the function body but never assigned,
         not a Python builtin, and not imported

    Returns:
        score  (float): 0.0 = clean, 0.7 = undefined names, 0.9 = syntax error
        issues (str):   Human-readable description of what was found
    """
    # ── 1. Syntax check ────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return 0.9, f"Syntax error at line {e.lineno}: {e.msg}"

    # ── 2. Undefined name detection ────────────────────────────────────────────
    defined: set = set()
    used:    set = set()

    for node in ast.walk(tree):
        # Names introduced by function/class definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        # Function arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = (
                node.args.args
                + node.args.posonlyargs
                + node.args.kwonlyargs
            )
            for arg in all_args:
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
        # Assignment targets
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        # Name loads (usage)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        # For-loop variables
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined.add(elt.id)
        # Comprehension variables
        if isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        # Walrus operator (:=)
        if isinstance(node, ast.NamedExpr):
            defined.add(node.target.id)

    builtin_names = set(dir(builtins))
    # Filter out private/dunder internals that appear in any module scope
    undefined = used - defined - builtin_names - {"__name__", "__file__", "_"}

    if undefined:
        return 0.7, f"Potentially undefined names: {', '.join(sorted(undefined))}"

    return 0.0, "None"


# Calibrated critic prompt — used by both pre_critic and post_critic.
# Key calibration choices:
#   • Static analysis findings are fed in as context so the LLM isn't
#     starting blind and doesn't duplicate what AST already caught.
#   • Explicit score anchors prevent the model from using arbitrary
#     thresholds (the Phase 4 Run 1 problem: passing code scored 0.8).
#   • "NOT hallucination" block eliminates style-based false positives
#     that plagued Code Llama's terse output format.
#   • Conservative bias: if the LLM can't point to a specific failing
#     line, it must score below 0.4.
_CRITIC_PROMPT = """\
You are a strict Python code reviewer checking for hallucination risk.

Hallucination means code that would FAIL to execute or produce WRONG results:
  - Calling methods or attributes that do not exist (e.g. list.sort_by())
  - Using variables that are never defined in the function scope
  - Logic that provably returns wrong values for the given problem
  - Incorrect assumptions about argument types that cause crashes

NOT hallucination — do NOT penalize these:
  - Terse, compact, or unconventional style
  - Missing docstrings or comments
  - Using a different algorithm than expected (as long as it is correct)
  - Code that is correct but could be written more cleanly

Static analysis has already checked this code and found:
  {static_issues}

Code to evaluate:
{code}

Score anchors — use these to calibrate your response:
  0.0–0.2 : Code looks correct; would likely pass tests
  0.3–0.5 : Minor concern — possible edge case issue but probably works
  0.6–0.8 : Specific line(s) likely to cause failure — you must name them
  0.9–1.0 : Definitely broken — clear undefined name or syntax-level error

IMPORTANT: Be conservative. If you cannot point to a SPECIFIC line that
would cause a crash or wrong output, score below 0.4.
Style concerns, unusual formatting, or terse code are NOT hallucination.

Respond in this EXACT format (both lines required):
ISSUES: <specific line or problem that would cause failure, or "None" if code looks correct>
SCORE: <single number 0.0–1.0>
"""


def run_hybrid_critic(code: str):
    """
    Two-layer hybrid critic combining static analysis + calibrated LLM judge.

    Layer 1 — Static analysis (deterministic):
        Catches syntax errors and undefined names with zero hallucination risk.
        If a syntax error is found, the LLM is skipped entirely.

    Layer 2 — Calibrated LLM (semantic):
        Only called for issues the AST cannot see — logical errors, wrong
        return types, misused APIs that exist but are called incorrectly.
        Static findings are fed as context so the LLM isn't starting blind.

    Scoring:
        Static layer sets a hard floor. LLM can only push the score higher,
        never lower. This prevents the LLM from cancelling out a real error
        the static checker already found.

    Returns:
        score  (float): combined hallucination risk 0.0–1.0
        issues (str):   description of the primary problem found
    """
    static_score, static_issues = static_analysis(code)

    # Syntax error is a hard stop — LLM call would be wasted
    if static_score >= 0.9:
        return static_score, static_issues

    # Build calibrated prompt with static context injected
    prompt = _CRITIC_PROMPT.format(static_issues=static_issues, code=code)
    response   = llm_critic.invoke([HumanMessage(content=prompt)])
    llm_score  = extract_score(response.content)
    llm_issues = extract_issues(response.content)

    # Static is the floor; LLM can only push higher
    final_score  = max(static_score, llm_score)
    final_issues = (
        llm_issues
        if llm_issues not in ("No specific issues identified.", "None")
        else static_issues
    )

    return final_score, final_issues


def run_tests(example: dict, code: str, timeout: int = 10) -> bool:
    """
    Unified test runner for HumanEval, HumanEval+, and MBPP.

    HumanEval / HumanEval+:
        Combines generated code + the check() function definition + check(entry_point) call.
        Returns True only if every test case passes.

    MBPP:
        Combines generated code + raw assert statements from test_list.
        Returns True only if all asserts pass.

    Uses SIGALRM to enforce a hard timeout — generated code at high temperatures
    can contain infinite loops. SIGALRM interrupts exec() without spawning a
    subprocess or causing re-import side effects.
    """
    def _timeout_handler(signum, frame):
        raise TimeoutError("test execution exceeded time limit")

    if example.get("is_mbpp"):
        # MBPP: code + assert statements (no wrapper function needed)
        full_code = code + "\n\n" + example["test"]
    else:
        # HumanEval / HumanEval+: code + check() definition + check(entry_point) call
        full_code = code + "\n\n" + example["test"] + f"\ncheck({example['entry_point']})"

    old_stdout = sys.stdout
    sys.stdout  = StringIO()

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        exec(full_code, {})
        signal.alarm(0)
        return True
    except TimeoutError:
        return False
    except Exception:
        signal.alarm(0)
        return False
    finally:
        sys.stdout = old_stdout


def append_to_master_csv(row: dict):
    """
    Append one aggregate-result row to the master CSV.
    This is your single source of truth for cross-run analysis.
    Every run adds a row here — open it in Excel when you're done.
    """
    file_exists = os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ========================= NODES =========================

def planner(state):
    """
    Agent 1 — Planner.
    Generates the initial Python function for a given problem prompt.

    Phase 4 Enhancement: Supports few-shot examples and caching.
    - Few-shot: Include 2-3 examples of correct code to improve output quality.
    - Caching: Load pre-generated code from cache to speed up experiments on same Planner output.

    CRITICAL: Cache is only valid if it was generated with the same few-shot configuration.
    The cache file stores metadata to verify compatibility. If few-shot flag differs from
    what the cache was generated with, the cache is invalidated and fresh code is generated.
    """
    problem_id = state.get('problem_id', None)

    # ── Check cache (with metadata validation) ─────────────────────────────
    if CACHE_PLANNER_OUTPUT and os.path.exists(CACHE_PLANNER_OUTPUT):
        with open(CACHE_PLANNER_OUTPUT, 'r') as f:
            cache = json.load(f)

        # Extract metadata (if it exists, cache must be valid for current config)
        cache_metadata = cache.get("__metadata__", {})
        cache_few_shot = cache_metadata.get("few_shot", False)

        # Only load from cache if the few-shot flag matches what the cache was created with
        if cache_few_shot == USE_FEWSHOT and str(problem_id) in cache:
            code = cache[str(problem_id)]
            return {
                **state,
                "generated_code": code,
                "original_code": code,
                "from_cache": True,
            }

    # ── Choose LLM instance: retry uses llm_planner_retry ───────────────────
    active_llm = llm_planner_retry if state.get("is_retry", False) else llm_planner

    # ── Build prompt with optional few-shot examples ────────────────────────
    few_shot_examples = ""
    if USE_FEWSHOT:
        few_shot_examples = """Here are examples of correct, clean Python functions:

Example 1:
Problem: def sum_two(a, b): return the sum of two numbers
def sum_two(a, b):
    return a + b

Example 2:
Problem: def is_prime(n): check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

Now generate a similar function:

"""

    prompt = (
        f"{few_shot_examples}"
        "Write ONLY the Python function.\n"
        "Do not include explanation.\n"
        "Ensure all variables and functions are defined.\n"
        "Do not hallucinate APIs or imports.\n\n"
        f"{state['problem']}"
    )
    response = active_llm.invoke([HumanMessage(content=prompt)])
    code = extract_code(response.content)

    # ── Save to cache if caching is enabled ─────────────────────────────────
    if CACHE_PLANNER_OUTPUT:
        os.makedirs(os.path.dirname(CACHE_PLANNER_OUTPUT) or ".", exist_ok=True)
        cache = {}
        if os.path.exists(CACHE_PLANNER_OUTPUT):
            with open(CACHE_PLANNER_OUTPUT, 'r') as f:
                cache = json.load(f)

        # Store metadata so future runs can verify cache compatibility
        cache["__metadata__"] = {"few_shot": USE_FEWSHOT}
        cache[str(problem_id)] = code

        with open(CACHE_PLANNER_OUTPUT, 'w') as f:
            json.dump(cache, f)

    return {
        **state,
        "generated_code": code,
        "original_code": code,
        "from_cache": False,
    }


def pre_critic(state):
    """
    Agent 2 (Pre-Fix) — Hybrid Critic.
    Scores hallucination risk in the Planner's output BEFORE the Fixer runs.

    Uses run_hybrid_critic: static analysis (AST) first, then calibrated
    LLM judge for semantic issues the AST cannot catch. Static findings
    are fed to the LLM as context so it isn't starting blind.
    """
    score, issues = run_hybrid_critic(state["generated_code"])
    return {
        **state,
        "pre_hallucination_score": score,
        "critic_issues": issues,
    }


def fixer(state):
    """
    Agent 3 — Fixer (Phase 2 redesign).

    Three changes vs. original:
    1. Uses llm_fixer (8B) instead of the 3B planner model — capable enough to
       act on the critic's feedback without introducing new errors.
    2. Receives the original problem specification so it understands intent,
       not just the broken code and issues in isolation.
    3. Selective-fix gate: after generating the fix, the critic immediately
       re-scores it. If the new score is no better than the pre-score, the
       fix is discarded and the original code is restored (gate_reverted=True).
       This implements the selective prediction principle — only keep fixes
       that the critic verifies as an improvement.
    """
    if state["pre_hallucination_score"] > HALLUCINATION_THRESHOLD:

        # ── Build fixer prompt ─────────────────────────────────────────────
        # Phase 8 ablation: USE_FIXER_CONTEXT controls whether the problem
        # specification is included. With context (default): fixer understands
        # the intent. Without context: fixer only sees issues + broken code.
        if USE_FIXER_CONTEXT:
            prompt = f"""You are a Python expert fixing bugs in code.

Problem specification (what the function must do):
{state['problem']}

The following issues were identified in the current implementation:
{state['critic_issues']}

Fix these issues in the code below.
Return ONLY the corrected Python function with no explanation.

Code:
{state['generated_code']}
"""
        else:
            prompt = f"""You are a Python expert fixing bugs in code.

The following issues were identified in the current implementation:
{state['critic_issues']}

Fix these issues in the code below.
Return ONLY the corrected Python function with no explanation.

Code:
{state['generated_code']}
"""

        response = llm_fixer.invoke([HumanMessage(content=prompt)])
        new_code = extract_code(response.content)
        fixer_changed = new_code.strip() != state["generated_code"].strip()

        # ── Selective-fix gate ─────────────────────────────────────────────
        # Phase 8 ablation: USE_GATE controls whether we re-score and revert.
        # With gate (default): only keep fixes the critic verifies as better.
        # Without gate: always keep whatever the fixer produces.
        if USE_GATE:
            gate_score, _ = run_hybrid_critic(new_code)
            if gate_score >= state["pre_hallucination_score"]:
                # Fixer made it worse or no better — revert to original
                return {
                    **state,
                    "fixer_triggered": True,
                    "fixer_changed":   False,
                    "gate_reverted":   True,
                }
        else:
            gate_score = None   # gate not run

        return {
            **state,
            "generated_code": new_code,
            "fixer_triggered": True,
            "fixer_changed":   fixer_changed,
            "gate_reverted":   False,
        }

    return {**state, "fixer_triggered": False, "fixer_changed": False, "gate_reverted": False}


def post_critic(state):
    """
    Agent 2 (Post-Fix) — Hybrid Critic.
    Re-scores hallucination AFTER fixing using the identical hybrid critic
    as pre_critic — only the code under evaluation differs. This preserves
    the Phase 2 fix that eliminated the +0.344 measurement artifact.
    """
    score, _ = run_hybrid_critic(state["generated_code"])
    return {**state, "post_hallucination_score": score}


# ========================= GRAPHS =========================

def baseline_graph():
    """
    Control condition: Planner → Post-Critic only.
    No fixing. Used to measure the unassisted success rate.
    """
    wf = StateGraph(dict)
    wf.add_node("planner",     planner)
    wf.add_node("post_critic", post_critic)
    wf.set_entry_point("planner")
    wf.add_edge("planner",     "post_critic")
    wf.add_edge("post_critic", END)
    return wf.compile()


def monitoring_graph():
    """
    Treatment condition: Planner → Pre-Critic → Fixer → Post-Critic.
    The full monitoring pipeline.
    """
    wf = StateGraph(dict)
    wf.add_node("planner",     planner)
    wf.add_node("pre_critic",  pre_critic)
    wf.add_node("fixer",       fixer)
    wf.add_node("post_critic", post_critic)
    wf.set_entry_point("planner")
    wf.add_edge("planner",     "pre_critic")
    wf.add_edge("pre_critic",  "fixer")
    wf.add_edge("fixer",       "post_critic")
    wf.add_edge("post_critic", END)
    return wf.compile()


# ========================= RUN EXPERIMENT =========================

def _load_benchmark_dataset(benchmark, size):
    """
    Load and normalise a benchmark dataset into a flat list of dicts with keys:
        prompt       — problem description fed to the Planner
        test         — test code (check() body for HumanEval/+, assert stmts for MBPP)
        entry_point  — function name under test
        is_mbpp      — True only for MBPP (changes how run_tests() assembles the exec block)
    """
    if benchmark == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split=f"test[:{size}]")
        return [{"prompt": ex["prompt"], "test": ex["test"],
                 "entry_point": ex["entry_point"], "is_mbpp": False} for ex in ds]

    elif benchmark == "humaneval_plus":
        ds = load_dataset("evalplus/humanevalplus", split=f"test[:{size}]")
        return [{"prompt": ex["prompt"], "test": ex["test"],
                 "entry_point": ex["entry_point"], "is_mbpp": False} for ex in ds]

    elif benchmark == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", split=f"test[:{size}]")
        problems = []
        for ex in ds:
            entry_point = _extract_mbpp_entry_point(ex["test_list"])
            # Build a HumanEval-style prompt so the Planner knows the function name
            prompt = (
                f"Write a Python function named `{entry_point}` to solve this problem:\n"
                f"{ex['text']}\n\n"
                f"def {entry_point}("
            )
            test_code = "\n".join(ex["test_list"])
            problems.append({
                "prompt": prompt,
                "test": test_code,
                "entry_point": entry_point,
                "is_mbpp": True,
            })
        return problems

    raise ValueError(f"Unknown benchmark: {benchmark}")

dataset = _load_benchmark_dataset(BENCHMARK, DATASET_SIZE)

versions_to_run = (
    ["baseline", "with_monitoring"]
    if args.version == "both"
    else ["baseline" if args.version == "baseline" else "with_monitoring"]
)

for version in versions_to_run:

    print("\n" + "=" * 70)
    print(f"  RUNNING  : {version.upper()}")
    print(f"  Temp={TEMPERATURE}   Threshold={HALLUCINATION_THRESHOLD}   N={DATASET_SIZE}"
          + (f"   Label={args.run_label}" if args.run_label else ""))
    print("=" * 70)

    graph = baseline_graph() if version == "baseline" else monitoring_graph()

    # MLflow run name encodes all key parameters — easy to filter in the UI
    run_name = (
        f"{version}__T{TEMPERATURE}__TH{HALLUCINATION_THRESHOLD}__N{DATASET_SIZE}"
        + (f"__{args.run_label}" if args.run_label else "")
    )

    per_problem_rows    = []

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "planner_model":           PLANNER_MODEL,
            "critic_model":            CRITIC_MODEL,
            "fixer_model":             FIXER_MODEL,
            "temperature":             TEMPERATURE,
            "retry_temperature":       RETRY_TEMPERATURE,
            "dataset_size":            DATASET_SIZE,
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "version":                 version,
            "run_label":               args.run_label or "—",
            "use_gate":                int(USE_GATE),
            "use_fixer_context":       int(USE_FIXER_CONTEXT),
            "benchmark":               BENCHMARK,
        })

        successes            = 0
        pre_scores           = []
        post_scores          = []
        improvements         = []
        latencies            = []
        fixer_trigger_count  = 0
        fixer_change_count   = 0
        gate_revert_count    = 0

        for i, example in enumerate(dataset):

            print(f"  [{i+1:02d}/{DATASET_SIZE}] {example['entry_point']:<35}", end="", flush=True)

            start_time = time.time()
            try:
                result = graph.invoke({
                    "problem": example["prompt"],
                    "problem_id": i,  # Track problem ID for caching
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            latency = time.time() - start_time
            latencies.append(latency)

            code    = result["generated_code"]
            success = run_tests(example, code)
            successes += int(success)

            # ── Iterative: Try again if failed (pass@1 vs pass@2) ─────────────
            success_at_iteration = 1 if success else None
            if USE_ITERATIVE and not success:
                # Show error to Planner and let it retry
                error_feedback = f"Previous attempt failed tests. Problem:\n{example['prompt']}"
                try:
                    result_retry = graph.invoke({
                        "problem":    error_feedback,
                        "problem_id": f"{i}_retry",
                        "is_retry":   True,
                    })
                    code_retry = result_retry["generated_code"]
                    success_retry = run_tests(example, code_retry)
                    if success_retry:
                        success_at_iteration = 2
                except:
                    pass  # Keep original failure if retry errored

            pre_score  = result.get("pre_hallucination_score", None)
            post_score = result.get("post_hallucination_score", 0.0)
            post_scores.append(post_score)

            improvement = 0.0
            if version == "with_monitoring" and pre_score is not None:
                pre_scores.append(pre_score)
                improvement = pre_score - post_score
                improvements.append(improvement)
                if result.get("fixer_triggered"): fixer_trigger_count += 1
                if result.get("fixer_changed"):   fixer_change_count  += 1
                if result.get("gate_reverted"):   gate_revert_count   += 1

            # Inline status so you can watch progress in real time
            status_icon = "✓" if success else "✗"
            pre_str = f"pre={pre_score:.2f} " if pre_score is not None else ""
            print(f"{status_icon}  {pre_str}post={post_score:.2f}  lat={latency:.1f}s")

            # Log per-problem metrics to MLflow
            mlflow.log_metrics({
                f"p{i}_success":    int(success),
                f"p{i}_latency":    round(latency, 3),
                f"p{i}_post_score": round(post_score, 3),
                **({
                    f"p{i}_pre_score":   round(pre_score, 3),
                    f"p{i}_improvement": round(improvement, 3),
                } if pre_score is not None else {}),
            })

            # Collect row for per-run CSV
            per_problem_rows.append({
                "run_timestamp":          RUN_TIMESTAMP,
                "run_label":              args.run_label or "—",
                "version":                version,
                "temperature":            TEMPERATURE,
                "threshold":              HALLUCINATION_THRESHOLD,
                "problem_index":          i,
                "entry_point":            example["entry_point"],
                "pass_at_1":              int(success),
                "pass_at_2":              int(1 if success_at_iteration == 2 else success),
                "pre_score":              round(pre_score, 3) if pre_score is not None else "",
                "post_score":             round(post_score, 3),
                "improvement":            round(improvement, 3),
                "latency_s":              round(latency, 3),
                "fixer_triggered":        int(result.get("fixer_triggered", False)),
                "fixer_changed":          int(result.get("fixer_changed", False)),
                "gate_reverted":          int(result.get("gate_reverted", False)),
                "from_planner_cache":     int(result.get("from_cache", False)),
                "use_gate":               int(USE_GATE),
                "use_fixer_context":      int(USE_FIXER_CONTEXT),
            })

        # ── Aggregate ──────────────────────────────────────────────────────
        n                  = len(dataset)
        success_rate       = successes / n
        avg_pre            = sum(pre_scores)   / len(pre_scores)   if pre_scores   else 0.0
        avg_post           = sum(post_scores)  / len(post_scores)  if post_scores  else 0.0
        avg_improvement    = sum(improvements) / len(improvements) if improvements else 0.0
        avg_latency        = sum(latencies)    / len(latencies)    if latencies    else 0.0
        fixer_trigger_rate = fixer_trigger_count / n
        fixer_change_rate  = fixer_change_count  / n
        gate_revert_rate   = gate_revert_count   / n

        mlflow.log_metrics({
            "success_rate":                  round(success_rate, 4),
            "avg_pre_hallucination":         round(avg_pre, 4),
            "avg_post_hallucination":        round(avg_post, 4),
            "avg_hallucination_improvement": round(avg_improvement, 4),
            "avg_latency":                   round(avg_latency, 3),
            "fixer_trigger_rate":            round(fixer_trigger_rate, 4),
            "fixer_change_rate":             round(fixer_change_rate, 4),
            "gate_revert_rate":              round(gate_revert_rate, 4),
        })

        # ── Save per-run CSV ───────────────────────────────────────────────
        run_csv = (
            f"results/{RUN_TIMESTAMP}_{version}"
            f"_T{TEMPERATURE}_TH{HALLUCINATION_THRESHOLD}_N{DATASET_SIZE}"
            + (f"_{args.run_label}" if args.run_label else "")
            + ".csv"
        )
        if per_problem_rows:
            with open(run_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_problem_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_problem_rows)

        # ── Append to master CSV (one row per run) ─────────────────────────
        # Calculate pass@2 rate if iterative was enabled
        pass_at_2_count = sum(1 for row in per_problem_rows if row.get("pass_at_2", 0))
        pass_at_2_rate = pass_at_2_count / n if USE_ITERATIVE else success_rate

        append_to_master_csv({
            "run_timestamp":          RUN_TIMESTAMP,
            "run_label":              args.run_label or "—",
            "version":                version,
            "temperature":            TEMPERATURE,
            "retry_temperature":      RETRY_TEMPERATURE,
            "threshold":              HALLUCINATION_THRESHOLD,
            "dataset_size":           DATASET_SIZE,
            "pass_at_1_rate":         round(success_rate, 4),
            "pass_at_1_count":        successes,
            "pass_at_2_rate":         round(pass_at_2_rate, 4) if USE_ITERATIVE else "",
            "pass_at_2_count":        pass_at_2_count if USE_ITERATIVE else "",
            "avg_pre_hallucination":  round(avg_pre, 4),
            "avg_post_hallucination": round(avg_post, 4),
            "avg_improvement":        round(avg_improvement, 4),
            "avg_latency_s":          round(avg_latency, 3),
            "fixer_trigger_rate":     round(fixer_trigger_rate, 4),
            "fixer_change_rate":      round(fixer_change_rate, 4),
            "gate_revert_rate":       round(gate_revert_rate, 4),
            "use_fewshot":            int(USE_FEWSHOT),
            "use_iterative":          int(USE_ITERATIVE),
            "use_gate":               int(USE_GATE),
            "use_fixer_context":      int(USE_FIXER_CONTEXT),
            "benchmark":              BENCHMARK,
        })

        # ── Console summary ────────────────────────────────────────────────
        print()
        print(f"  ── SUMMARY : {version.upper()} ──────────────────────────────")
        print(f"  Pass@1 Rate            : {success_rate:.1%}  ({successes}/{n})")
        if USE_ITERATIVE:
            print(f"  Pass@2 Rate            : {pass_at_2_rate:.1%}  ({pass_at_2_count}/{n})  ← iterative improvement")
        print(f"  Avg Pre-Hallucination  : {avg_pre:.3f}")
        print(f"  Avg Post-Hallucination : {avg_post:.3f}")
        print(f"  Avg Improvement        : {avg_improvement:+.3f}   ← core research metric")
        print(f"  Avg Latency            : {avg_latency:.2f}s / problem")
        if version == "with_monitoring":
            print(f"  Fixer Triggered        : {fixer_trigger_count}/{n}  ({fixer_trigger_rate:.0%})")
            print(f"  Fixer Actually Changed : {fixer_change_count}/{n}  ({fixer_change_rate:.0%})")
            print(f"  Gate Reverted          : {gate_revert_count}/{n}  ({gate_revert_rate:.0%})  ← fixes rejected by selective gate")
        if USE_FEWSHOT:
            print(f"  Few-shot Prompting     : ENABLED  ← prompts include examples")
        print(f"  Per-run CSV  → {run_csv}")
        print(f"  Master CSV   → {MASTER_CSV}")
        print(f"  MLflow run   → {run_name}")
        print()
