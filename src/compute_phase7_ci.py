"""
compute_phase7_ci.py
--------------------
Reads master_results_phase7.csv and computes mean Δpass@1 + 95% CI
for each (model, tau) combination across 3 independent trials.

Usage:
    python3 src/compute_phase7_ci.py

Output:
    Printed table + results/phase7_ci_summary.csv
"""

import csv
import math
import os
from collections import defaultdict

# ── t-critical values for 95% CI, 2 degrees of freedom (n=3 trials) ─────────
# scipy not required; hardcoded t* for df=2, two-tailed 95%
T_CRIT_DF2 = 4.303   # t_{0.025, df=2}

CSV_PATH    = "results/master_results_phase7.csv"
OUTPUT_PATH = "results/phase7_ci_summary.csv"


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def ci_95(vals):
    """Return (mean, lower, upper) using t-distribution for small samples."""
    n = len(vals)
    m = mean(vals)
    if n < 2:
        return m, float("nan"), float("nan")
    se = std(vals) / math.sqrt(n)
    margin = T_CRIT_DF2 * se
    return m, m - margin, m + margin


def load_results(path):
    """
    Returns a dict:
        key  : (model_label, tau, version)
        value: list of pass@1 floats (one per trial)
    """
    data = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label   = row.get("run_label", "")
            version = row.get("version", "")
            tau     = row.get("threshold", "")
            pass1   = row.get("pass_at_1_rate", None)

            if pass1 is None:
                continue

            # Infer model from label prefix
            if "codellama" in label.lower():
                model = "Code Llama 7B"
            elif "qwen" in label.lower():
                model = "Qwen2.5 Coder 7B"
            else:
                model = label

            data[(model, tau, version)].append(float(pass1))

    return data


def compute_delta(data):
    """
    For each (model, tau), pair baseline and monitored trials by trial index
    and compute per-trial Δpp = monitored - baseline.
    Returns dict: (model, tau) -> list of Δpp values
    """
    deltas = defaultdict(list)

    # Collect models and taus
    combos = set((m, t) for (m, t, v) in data.keys())

    for model, tau in sorted(combos):
        baseline_vals   = data.get((model, tau, "baseline"), [])
        monitored_vals  = data.get((model, tau, "with_monitoring"), [])

        n = min(len(baseline_vals), len(monitored_vals))
        if n == 0:
            print(f"  WARNING: no data for ({model}, tau={tau})")
            continue

        for i in range(n):
            delta_pp = (monitored_vals[i] - baseline_vals[i]) * 100
            deltas[(model, tau)].append(delta_pp)

    return deltas


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run run_phase7.sh first.")
        return

    print(f"\nReading: {CSV_PATH}")
    data   = load_results(CSV_PATH)
    deltas = compute_delta(data)

    print("\n" + "=" * 70)
    print("  PHASE 7 -- CI RESULTS (95% confidence, t-distribution, df=2)")
    print("=" * 70)
    print(f"  {'Model':<22} {'tau':>5}  {'N':>3}  {'Mean Δpp':>9}  {'95% CI':>20}  {'Excludes 0?':>12}")
    print("-" * 70)

    rows = []
    for (model, tau), vals in sorted(deltas.items()):
        n = len(vals)
        m, lo, hi = ci_95(vals)
        excludes_zero = "YES ✓" if (lo > 0 or hi < 0) else "no"
        print(f"  {model:<22} {tau:>5}  {n:>3}  {m:>+8.1f}pp  [{lo:+.1f}, {hi:+.1f}]pp  {excludes_zero:>12}")
        rows.append({
            "model":        model,
            "tau":          tau,
            "n_trials":     n,
            "mean_delta_pp": round(m, 2),
            "ci_lower":     round(lo, 2),
            "ci_upper":     round(hi, 2),
            "excludes_zero": excludes_zero.startswith("YES"),
        })

    print("=" * 70)

    # Also print raw per-trial values for transparency
    print("\n  Per-trial Δpp breakdown:")
    print("-" * 70)
    for (model, tau), vals in sorted(deltas.items()):
        trial_str = "  |  ".join(f"T{i+1}: {v:+.1f}pp" for i, v in enumerate(vals))
        print(f"  {model:<22} tau={tau}  ->  {trial_str}")
    print("=" * 70 + "\n")

    # Save summary CSV
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Summary saved: {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()
