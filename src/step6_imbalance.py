"""
STEP 6 - CLASS IMBALANCE HANDLING
=================================
The problem
-----------
Hazard classes are wildly unequal in size. Tens of thousands of compounds
carry the irritant pictogram, while only a few hundred are explosives. Left
alone, a classifier learns that predicting "not explosive" for everything
scores 99.8% accuracy - a useless model that would never warn anyone about an
explosive.

Four defences are applied here, exactly as the proposal specifies.

6a  SMOTE oversampling of the minority class, applied to the TRAINING SET
    ONLY. SMOTE invents new synthetic minority examples by interpolating
    between real ones. Applying it to validation or test data would mean
    scoring the model on molecules that do not exist, so it is never done.

6b  Class weights, so that a mistake on a rare class costs the model more
    than a mistake on a common one.

6c  The precision-recall threshold-calibration framework that Step 9 uses to
    replace the arbitrary 0.5 decision cut-off.

6d  Selection of the primary evaluation metric per class - Matthews
    correlation coefficient for rare classes, AUC-ROC for common ones.

A note on SMOTE in a multi-label setting
----------------------------------------
SMOTE is a single-label algorithm: it cannot balance nine labels at once,
because oversampling for one hazard distorts the balance of the other eight.
The standard solution, used here, is to build one SMOTE-balanced training set
per hazard class. Each of the nine binary classifiers is then trained on its
own balanced view of the data. The nine balanced sets are saved together.

Author : Sareer Ahmad
"""

import os
import gc
import sys
import json
import time
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_SPLITS, DIR_FEATURES, DIR_LOGS,
                        GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

ISSUE_LOG = []

# A class with fewer than this many positives in the training set is "rare",
# and is judged by Matthews correlation coefficient rather than AUC-ROC (6d).
RARE_CLASS_TRAIN_THRESHOLD = 500

# Above this many training rows, SMOTE is skipped for that class and class
# weighting alone is relied upon. Interpolating ~1200-dimensional synthetic
# molecules for tens of thousands of rows would exhaust this machine's memory,
# and for a class that already has many positives SMOTE adds little anyway.
SMOTE_MAX_OUTPUT_ROWS = 120000


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ===========================================================================
# 6a - SMOTE
# ===========================================================================
def apply_smote_for_class(X_train, y_train_class, class_name):
    """
    Build a balanced training set for one hazard class.

    Fallback chain required by the proposal:
        SMOTE  ->  ADASYN  ->  RandomOverSampler

    SMOTE creates a synthetic molecule by picking a real minority compound,
    picking one of its k nearest minority neighbours, and taking a point on
    the line between them in descriptor space. It needs at least k+1 minority
    examples to work, so k is reduced automatically when a class is very rare.

    Returns
    -------
    (X_balanced, y_balanced, method_used)
    """
    n_positive = int(y_train_class.sum())
    n_negative = int(len(y_train_class) - n_positive)

    # Nothing to do if the class is already balanced or has no examples.
    if n_positive == 0:
        log_issue("6a", f"{class_name}: no positive examples in the training "
                        f"set - SMOTE skipped, class cannot be learned.")
        return X_train, y_train_class, "none (no positives)"
    if n_positive == n_negative:
        return X_train, y_train_class, "none (already balanced)"

    # For one hazard in this dataset - the irritant pictogram - it is the
    # POSITIVE examples that form the majority, because the ECHA notification
    # inventory that dominates PubChem assigns it to most substances. SMOTE
    # always oversamples whichever class is smaller, so it still applies; only
    # the size projection below has to account for either direction.
    projected_rows = 2 * max(n_positive, n_negative)
    if projected_rows > SMOTE_MAX_OUTPUT_ROWS:
        log_issue("6a", f"{class_name}: SMOTE would produce {projected_rows:,} "
                        f"rows, exceeding the {SMOTE_MAX_OUTPUT_ROWS:,} memory "
                        f"budget of this machine. Class weighting (6b) is used "
                        f"for this class instead.")
        return X_train, y_train_class, "none (memory budget - class weights used)"

    # k_neighbors must be strictly less than the number of minority examples.
    k_neighbors = min(5, n_positive - 1)
    if k_neighbors < 1:
        log_issue("6a", f"{class_name}: only {n_positive} positive example(s); "
                        f"SMOTE needs at least 2. Using RandomOverSampler.")
        try:
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=RANDOM_SEED)
            X_res, y_res = sampler.fit_resample(X_train, y_train_class)
            return X_res, y_res, "RandomOverSampler (too few positives for SMOTE)"
        except Exception as exc:
            log_issue("6a", f"{class_name}: RandomOverSampler failed too: {exc}")
            return X_train, y_train_class, "none (all samplers failed)"

    # ---- primary: SMOTE ---------------------------------------------------
    try:
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
        X_res, y_res = sampler.fit_resample(X_train, y_train_class)
        return X_res, y_res, f"SMOTE (k_neighbors={k_neighbors})"
    except Exception as exc:
        log_issue("6a", f"{class_name}: SMOTE failed ({type(exc).__name__}: "
                        f"{exc}). Falling back to ADASYN.")

    # ---- fallback 1: ADASYN ------------------------------------------------
    try:
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=RANDOM_SEED,
                         n_neighbors=min(5, max(1, n_positive - 1)))
        X_res, y_res = sampler.fit_resample(X_train, y_train_class)
        return X_res, y_res, "ADASYN (FALLBACK - SMOTE failed)"
    except Exception as exc:
        log_issue("6a", f"{class_name}: ADASYN failed too ({exc}). "
                        f"Falling back to RandomOverSampler.")

    # ---- fallback 2: RandomOverSampler -------------------------------------
    try:
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(X_train, y_train_class)
        return X_res, y_res, "RandomOverSampler (FALLBACK - SMOTE and ADASYN failed)"
    except Exception as exc:
        log_issue("6a", f"{class_name}: every oversampler failed ({exc}). "
                        f"Training on the raw imbalanced data with class weights.")
        return X_train, y_train_class, "none (all samplers failed)"


# ===========================================================================
# 6b - CLASS WEIGHTS
# ===========================================================================
def compute_class_weights(y_train):
    """
    Compute, for every hazard, how many negatives there are per positive.

    This ratio is handed to XGBoost as `scale_pos_weight` and tells the model
    to treat one positive example as if it were worth that many negatives.
    """
    print("\n[6b] Computing class weights (negatives per positive) ...")
    weights = {}
    print("-" * 78)
    print(f"{'Class':<22}{'n positive':>12}{'n negative':>12}{'weight':>12}")
    print("-" * 78)
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        n_positive = int(y_train[:, class_index].sum())
        n_negative = int(y_train.shape[0] - n_positive)
        # Guard against division by zero for a class with no positives.
        weight = (n_negative / n_positive) if n_positive else 1.0
        weights[column] = round(float(weight), 4)
        print(f"{column:<22}{n_positive:>12,}{n_negative:>12,}{weight:>12.2f}")
    print("-" * 78)
    return weights


# ===========================================================================
# 6c - THRESHOLD CALIBRATION FRAMEWORK
# ===========================================================================
def build_threshold_calibration_framework():
    """
    Define how Step 9 will choose each class's decision threshold.

    A classifier outputs a probability. Turning that into a yes/no answer
    needs a cut-off, and the usual 0.5 is arbitrary and badly wrong for
    imbalanced data. Two better cut-offs are computed per class in Step 9:

      * threshold_f1        - the cut-off giving the best balance of precision
                              and recall.
      * threshold_recall90  - the lowest cut-off that still catches at least
                              90% of genuinely hazardous compounds. This is
                              the safety-first setting a regulator would want,
                              accepting more false alarms in exchange for
                              missing fewer real hazards.

    Both are fitted on the VALIDATION set only, then applied unchanged to the
    test set. Fitting them on the test set would be a form of cheating.
    """
    print("\n[6c] Preparing the threshold-calibration framework ...")
    framework = {
        "fit_on": "validation set only",
        "applied_to": "test set",
        "thresholds_computed_per_class": [
            {"name": "threshold_f1",
             "rule": "argmax of F1 over the precision-recall curve",
             "use_case": "balanced screening"},
            {"name": "threshold_recall90",
             "rule": "highest threshold whose recall is still >= 0.90",
             "use_case": "safety-first regulatory screening"},
            {"name": "threshold_default",
             "rule": "fixed at 0.5",
             "use_case": "baseline for comparison"},
        ],
        "rationale": ("A fixed 0.5 cut-off is inappropriate for imbalanced "
                      "data. Both thresholds are fitted on validation data "
                      "only and then applied unchanged to the test set."),
    }
    for entry in framework["thresholds_computed_per_class"]:
        print(f"      {entry['name']:<20} {entry['rule']}")
    return framework


# ===========================================================================
# 6d - PRIMARY EVALUATION METRIC PER CLASS
# ===========================================================================
def choose_primary_metrics(y_train):
    """
    Decide which metric is the headline number for each hazard class.

    Matthews correlation coefficient (MCC) is used for rare classes. MCC only
    scores well when all four cells of the confusion matrix are good, so it
    cannot be fooled by a model that simply predicts "not hazardous" for
    everything - which is exactly the failure mode of a rare class.

    AUC-ROC is used for the common classes, where it is stable and is the
    metric most readily compared with other published QSAR work.
    """
    print("\n[6d] Choosing the primary evaluation metric for each class ...")
    print("-" * 90)
    print(f"{'Class':<22}{'meaning':<34}{'train n+':>10}  {'primary metric'}")
    print("-" * 90)
    metrics = {}
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        n_positive = int(y_train[:, class_index].sum())
        is_rare = n_positive < RARE_CLASS_TRAIN_THRESHOLD
        metric = "MCC" if is_rare else "AUC-ROC"
        metrics[column] = {
            "primary_metric": metric,
            "train_positives": n_positive,
            "is_rare": bool(is_rare),
            "reason": ("fewer than 500 training positives - MCC resists the "
                       "all-negative trap" if is_rare else
                       "500 or more training positives - AUC-ROC is stable "
                       "and comparable with the literature"),
        }
        meaning = GHS_TRUE_MEANING[column].split("(")[0].strip()
        print(f"{column:<22}{meaning:<34}{n_positive:>10,}  {metric}")
    print("-" * 90)
    return metrics


# ===========================================================================
# MAIN
# ===========================================================================
def handle_imbalance():
    """Run every sub-step of Step 6 and save the balanced training data."""
    start_time = time.time()
    print("=" * 78)
    print("STEP 6 - CLASS IMBALANCE HANDLING")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy"))
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    val_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_val_indices.npy"))
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))

    X_train, y_train = X[train_idx], y[train_idx]
    print(f"Training set : {X_train.shape[0]:,} compounds x "
          f"{X_train.shape[1]:,} features")
    print(f"Validation   : {len(val_idx):,} compounds  (never oversampled)")
    print(f"Test         : {len(test_idx):,} compounds  (never oversampled)")

    # ---- 6a ---------------------------------------------------------------
    print("\n[6a] Applying SMOTE to the training set, one hazard class at a time ...")
    print("     (validation and test sets are never touched)")
    print("-" * 90)
    print(f"{'Class':<22}{'before':>18}{'after':>18}  {'method'}")
    print("-" * 90)

    # Each balanced set is written to disk and freed immediately. Holding all
    # nine in memory at once would need roughly 1.8 GB on this dataset, which
    # this machine cannot spare.
    balanced_dir = os.path.join(DIR_FEATURES, "STEP6_balanced")
    os.makedirs(balanced_dir, exist_ok=True)

    balanced_sizes = {}
    smote_report = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_class = y_train[:, class_index].astype(int)
        n_positive_before = int(y_class.sum())

        X_balanced, y_balanced, method = apply_smote_for_class(
            X_train, y_class, column)

        # Capture every number the report needs BEFORE the arrays are freed.
        n_positive_after = int(y_balanced.sum())
        n_total_after = int(len(y_balanced))
        before_text = f"{n_positive_before:,}/{len(y_class):,}"
        after_text = f"{n_positive_after:,}/{n_total_after:,}"
        print(f"{column:<22}{before_text:>18}{after_text:>18}  {method}")

        np.save(os.path.join(balanced_dir, f"X_{column}.npy"),
                X_balanced.astype(np.float32))
        np.save(os.path.join(balanced_dir, f"y_{column}.npy"),
                y_balanced.astype(np.int8))
        balanced_sizes[column] = n_total_after
        del X_balanced, y_balanced      # free before building the next class
        gc.collect()

        smote_report.append({
            "GHS_Column": column,
            "Meaning": GHS_TRUE_MEANING[column],
            "Train_Positives_Before": n_positive_before,
            "Train_Total_Before": int(len(y_class)),
            "Train_Positives_After": n_positive_after,
            "Train_Total_After": n_total_after,
            "Method": method,
        })
    print("-" * 90)

    # ---- 6b, 6c, 6d --------------------------------------------------------
    class_weights = compute_class_weights(y_train)
    threshold_framework = build_threshold_calibration_framework()
    primary_metrics = choose_primary_metrics(y_train)

    # ---- save --------------------------------------------------------------
    print("\n[6] Saving balanced training data ...")
    print(f"      Nine per-class balanced sets written to {balanced_dir}")

    # The proposal also names two single-array files. A single array cannot
    # hold nine differently-sized sets, so the class with the largest
    # oversampled set is copied there as the representative example.
    representative = max(balanced_sizes, key=balanced_sizes.get)
    shutil.copyfile(os.path.join(balanced_dir, f"X_{representative}.npy"),
                    stamped("STEP6_X_train_balanced.npy"))
    shutil.copyfile(os.path.join(balanced_dir, f"y_{representative}.npy"),
                    stamped("STEP6_y_train_balanced.npy"))
    print(f"      Representative single-array files use "
          f"{representative} ({balanced_sizes[representative]:,} rows).")
    archive_path = balanced_dir

    pd.DataFrame(smote_report).to_csv(stamped("STEP6_smote_report.csv"), index=False)

    config = {
        "class_weights": class_weights,
        "threshold_calibration_framework": threshold_framework,
        "primary_metrics": primary_metrics,
        "rare_class_threshold": RARE_CLASS_TRAIN_THRESHOLD,
        "smote_max_output_rows": SMOTE_MAX_OUTPUT_ROWS,
        "random_seed": RANDOM_SEED,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(stamped("STEP6_imbalance_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP6_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 6 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    n_rare = sum(1 for m in primary_metrics.values() if m["is_rare"])
    print("\n" + "=" * 78)
    print("STEP 6 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Built one SMOTE-balanced training set per hazard class,")
    print("                computed class weights, defined the threshold-calibration")
    print("                framework for Step 9, and chose the primary metric for")
    print("                each of the nine classes.")
    print(f"SMOTE         : applied to the training set only "
          f"({len(train_idx):,} compounds); validation and test untouched.")
    print(f"PRIMARY METRIC: MCC for {n_rare} rare class(es), "
          f"AUC-ROC for {9 - n_rare} common class(es)")
    print(f"OUTPUT FILES  : {archive_path}")
    print(f"                {stamped('STEP6_X_train_balanced.npy')}")
    print(f"                {stamped('STEP6_y_train_balanced.npy')}")
    print(f"                {stamped('STEP6_smote_report.csv')}")
    print(f"                {stamped('STEP6_imbalance_config.json')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged")
    print(f"ELAPSED       : {config['elapsed_seconds'] / 60:.1f} minutes")
    print("=" * 78)


if __name__ == "__main__":
    handle_imbalance()
