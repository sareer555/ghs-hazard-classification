"""
MASTER PIPELINE RUNNER
======================
Runs Steps 3 to 13 in order, and then the final summary report. Each step is
run in its own Python process so that a crash in one step cannot corrupt the
memory of the next, and so that memory is fully released between steps - which
matters on a machine with 7.9 GB of RAM.

Usage
-----
    python run_pipeline.py              # run every step from 3 to 13
    python run_pipeline.py 7            # run only step 7
    python run_pipeline.py 7 9          # run steps 7 through 9

Step 2 (data collection) is deliberately excluded from the default run,
because it takes about an hour and its output is already saved. Run it
explicitly with:  python src/step2_data_collection.py

Author : Sareer Ahmad
"""

import os
import sys
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# Step number -> (script path, human-readable description)
STEPS = {
    2:  ("src/step2_data_collection.py",     "Data collection from PubChem"),
    3:  ("src/step3_data_cleaning.py",       "Data cleaning and validation"),
    4:  ("src/step4_descriptors.py",         "Molecular descriptor computation"),
    5:  ("src/step5_scaffold_split.py",      "Scaffold-based dataset splitting"),
    6:  ("src/step6_imbalance.py",           "Class imbalance handling"),
    7:  ("src/step7_model_training.py",      "Model training"),
    8:  ("src/step8_hyperparameter_tuning.py", "Hyperparameter tuning"),
    9:  ("src/step9_evaluation.py",          "Model evaluation"),
    10: ("src/step10_shap_analysis.py",      "SHAP interpretability analysis"),
    11: ("src/step11_malaysia_validation.py", "Malaysian industrial validation"),
    13: ("src/step13_publication.py",        "Publication preparation"),
    99: ("src/final_report.py",              "Final project summary report"),
}


def run_step(number):
    """
    Run one step as a separate process and report whether it succeeded.

    Returns
    -------
    (ok, elapsed_seconds)
    """
    script, description = STEPS[number]
    label = "FINAL" if number == 99 else f"STEP {number}"
    print("\n" + "#" * 78)
    print(f"# {label}: {description}")
    print(f"# started {datetime.now().strftime('%H:%M:%S')}")
    print("#" * 78, flush=True)

    started = time.time()
    log_path = os.path.join(PROJECT_ROOT, "logs",
                            f"pipeline_step{number:02d}.log")

    # -u forces unbuffered output so the log fills in live rather than at the
    # end, which makes a long-running step possible to monitor.
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(
            [PYTHON, "-u", os.path.join(PROJECT_ROOT, script)],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)

    elapsed = time.time() - started

    # Echo the tail of the log so the console shows the progress report.
    with open(log_path, encoding="utf-8") as log:
        lines = log.readlines()
    for line in lines[-45:]:
        print(line.rstrip())

    ok = process.returncode == 0
    print(f"\n# {label} {'COMPLETED' if ok else 'FAILED'} "
          f"in {elapsed / 60:.1f} minutes (exit code {process.returncode})")
    print(f"# full log: {log_path}", flush=True)
    return ok, elapsed


def main():
    """Parse the command line and run the requested range of steps."""
    order = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 99]

    if len(sys.argv) == 2:
        wanted = [int(sys.argv[1])]
    elif len(sys.argv) == 3:
        first, last = int(sys.argv[1]), int(sys.argv[2])
        wanted = [s for s in order if first <= s <= last]
    else:
        wanted = order

    print("=" * 78)
    print("GHS HAZARD CLASSIFICATION - PIPELINE RUN")
    print("=" * 78)
    print(f"Python : {PYTHON}")
    print(f"Steps  : {wanted}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    total_start = time.time()
    for number in wanted:
        if number not in STEPS:
            print(f"Unknown step {number} - skipped.")
            continue
        ok, elapsed = run_step(number)
        results.append((number, STEPS[number][1], ok, elapsed))
        if not ok:
            # Later steps depend on earlier ones, so continuing after a failure
            # would only produce a cascade of confusing errors.
            print(f"\nSTOPPING: step {number} failed. Fix it and re-run with:")
            print(f"    python run_pipeline.py {number}")
            break

    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    print(f"{'Step':<8}{'Description':<44}{'Result':<12}{'Minutes':>9}")
    print("-" * 78)
    for number, description, ok, elapsed in results:
        label = "FINAL" if number == 99 else str(number)
        print(f"{label:<8}{description:<44}{'OK' if ok else 'FAILED':<12}"
              f"{elapsed / 60:>9.1f}")
    print("-" * 78)
    print(f"{'TOTAL':<52}{'':<12}{(time.time() - total_start) / 60:>9.1f}")
    print("=" * 78)

    return 0 if all(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
