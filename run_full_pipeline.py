"""
FULL-DATASET PIPELINE RUNNER
============================
Re-runs Steps 4 to 13 on the complete 243,323-compound cleaned dataset,
replacing the earlier 40,000-compound run.

Why the change
--------------
The 40,000-compound subset was drawn because this machine has 7.9 GB of RAM.
A controlled experiment - identical test set, identical hyperparameters, only
the training size varying - showed that training on all 243,323 compounds
raises the mean AUC-ROC from 0.8187 to 0.8738, a gain of +0.0551 against a
bootstrap confidence interval of +/-0.0139. Every one of the nine hazard
classes improved. The subset was therefore a real limitation, and this run
removes it.

Step 4 reuses the descriptor matrix computed on Google Colab rather than
recomputing an identical result locally.

Each step runs in its own process so that memory is fully released between
them, which matters at this scale.

Usage:
    python run_full_pipeline.py

Author : Sareer Ahmad
"""

import os
import sys
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = [
    ("4",  "src/step4_full_from_colab.py",       "Feature matrix, full dataset"),
    ("5",  "src/step5_scaffold_split.py",        "Scaffold split (stratified by group)"),
    ("6",  "src/step6_imbalance.py",             "Class imbalance handling"),
    ("7",  "src/step7_model_training.py",        "Model training"),
    ("8",  "src/step8_hyperparameter_tuning.py", "Hyperparameter tuning"),
    ("9",  "src/step9_evaluation.py",            "Model evaluation"),
    ("10", "src/step10_shap_analysis.py",        "SHAP interpretability"),
    ("11", "src/step11_malaysia_validation.py",  "Malaysian validation"),
    ("13", "src/step13_publication.py",          "Publication materials"),
    ("99", "src/final_report.py",                "Final summary report"),
]


def run(label, script, description):
    """Run one step in its own process and report the outcome."""
    print("\n" + "#" * 78)
    print(f"# STEP {label}: {description}")
    print(f"# started {datetime.now().strftime('%H:%M:%S')}")
    print("#" * 78, flush=True)

    log_path = os.path.join(PROJECT_ROOT, "logs", f"full_step{label}.log")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([PYTHON, "-u", os.path.join(PROJECT_ROOT, script)],
                              cwd=PROJECT_ROOT, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started

    with open(log_path, encoding="utf-8") as log:
        tail = log.readlines()[-40:]
    for line in tail:
        print(line.rstrip())

    ok = proc.returncode == 0
    print(f"\n# STEP {label} {'COMPLETED' if ok else 'FAILED'} in "
          f"{elapsed/60:.1f} min (exit {proc.returncode})")
    print(f"# log: {log_path}", flush=True)
    return ok, elapsed


def main():
    """Run every step in order, stopping at the first failure."""
    print("=" * 78)
    print("FULL-DATASET PIPELINE - 243,323 compounds")
    print("=" * 78)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    wanted = STEPS
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        wanted = [s for s in STEPS if s[0] in only]

    results, total_start = [], time.time()
    for label, script, description in wanted:
        ok, elapsed = run(label, script, description)
        results.append((label, description, ok, elapsed))
        if not ok:
            print(f"\nSTOPPING: step {label} failed. Fix and re-run with:")
            print(f"    python run_full_pipeline.py {label}")
            break

    print("\n" + "=" * 78)
    print("FULL-DATASET PIPELINE SUMMARY")
    print("=" * 78)
    print(f"{'Step':<7}{'Description':<44}{'Result':<11}{'Minutes':>9}")
    print("-" * 78)
    for label, description, ok, elapsed in results:
        print(f"{label:<7}{description:<44}{'OK' if ok else 'FAILED':<11}"
              f"{elapsed/60:>9.1f}")
    print("-" * 78)
    print(f"{'TOTAL':<62}{(time.time()-total_start)/60:>9.1f}")
    print("=" * 78)
    return 0 if all(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
