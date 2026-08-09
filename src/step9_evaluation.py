"""
STEP 9 - MODEL EVALUATION
=========================
This step answers the question the whole project exists to answer: how well
do these models actually predict GHS hazards for chemicals they have never
seen?

9a  Seven metrics computed per model per hazard class.
9b  Decision thresholds calibrated on the validation set, then applied
    unchanged to the test set.
9c  95% confidence intervals on every AUC, by bootstrap resampling.
9d  A single comparison table covering every model, class and metric.
9e  ROC curves, precision-recall curves and confusion matrices.

Why the threshold matters so much
---------------------------------
A classifier outputs a probability, and something must decide where to draw
the line between "hazardous" and "not hazardous". The default 0.5 is
arbitrary. For a rare hazard the model may never be 50% sure of anything, so
a 0.5 cut-off would flag nothing at all - a model with an excellent AUC can
have an F1 score of zero. Two better thresholds are therefore fitted on the
validation data and reported alongside the default.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_SPLITS, DIR_FEATURES, DIR_MODELS,
                        DIR_EVAL, DIR_LOGS, GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        get_ablation_identity, seed_everything, stamped)

seed_everything()
warnings.filterwarnings("ignore")

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

ISSUE_LOG = []
N_BOOTSTRAP = 1000          # as the proposal specifies
MODEL_COLOURS = {"RandomForest": "#1f77b4", "XGBoost": "#d62728",
                 "SVM": "#2ca02c", "RandomForest_NoClassWeight": "#ff7f0e",
                 "RandomForest_SMOTE": "#ff7f0e"}


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


def get_positive_probabilities(model, X):
    """
    Extract the probability of the positive class for all nine hazards.

    Both scikit-learn's MultiOutputClassifier and this project's
    PerLabelClassifier return `predict_proba` as a list of nine (n, 2) arrays.
    Column 1 of each is the probability of the hazard being present.

    Returns
    -------
    numpy array of shape (n_compounds, 9)
    """
    probabilities = model.predict_proba(X)
    columns = []
    for class_index in range(len(GHS_LABEL_COLUMNS)):
        array = probabilities[class_index]
        # A model trained on a single-valued class returns one column only.
        columns.append(array[:, 1] if array.shape[1] > 1 else array[:, 0])
    return np.column_stack(columns)


# ===========================================================================
# 9b - THRESHOLD CALIBRATION
# ===========================================================================
def calibrate_thresholds(y_val, probabilities_val):
    """
    Find two better decision thresholds per class, using validation data only.

    threshold_f1       - the cut-off that maximises the F1 score, which is the
                         harmonic mean of precision and recall. This is the
                         best all-round setting.
    threshold_recall90 - the highest cut-off that still finds at least 90% of
                         the genuinely hazardous compounds. This is the
                         safety-first setting: it accepts more false alarms in
                         exchange for missing fewer real hazards, which is the
                         trade-off a chemical regulator would choose.
    """
    print("\n[9b] Calibrating decision thresholds on the validation set ...")
    thresholds = {}
    print("-" * 84)
    print(f"{'Class':<22}{'thr(F1)':>10}{'F1@thr':>10}"
          f"{'thr(rec>=.90)':>16}{'recall':>10}{'precision':>12}")
    print("-" * 84)

    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_true = y_val[:, class_index]
        y_score = probabilities_val[:, class_index]

        if len(np.unique(y_true)) < 2:
            log_issue("9b", f"{column}: validation set has only one class "
                            f"present - threshold cannot be calibrated, "
                            f"0.5 is used.")
            thresholds[column] = {"threshold_f1": 0.5, "f1_at_threshold": 0.0,
                                  "threshold_recall90": 0.5,
                                  "recall_at_recall90": 0.0,
                                  "precision_at_recall90": 0.0}
            continue

        precision, recall, cut_offs = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns one more precision/recall value than
        # thresholds, so the last entry is dropped to line them up.
        precision, recall = precision[:-1], recall[:-1]

        # ---- best F1 ------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = 2 * precision * recall / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        best_index = int(np.argmax(f1_scores))
        threshold_f1 = float(cut_offs[best_index])

        # ---- highest threshold still reaching 90% recall -------------------
        acceptable = np.where(recall >= 0.90)[0]
        if len(acceptable):
            # Recall falls as the threshold rises, so the last acceptable
            # index is the strictest usable cut-off.
            chosen = int(acceptable[-1])
            threshold_recall90 = float(cut_offs[chosen])
            recall_at = float(recall[chosen])
            precision_at = float(precision[chosen])
        else:
            # The model never reaches 90% recall at any threshold.
            threshold_recall90 = float(cut_offs[0])
            recall_at = float(recall[0])
            precision_at = float(precision[0])
            log_issue("9b", f"{column}: 90% recall is unreachable at any "
                            f"threshold; the lowest available cut-off "
                            f"({threshold_recall90:.4f}, recall "
                            f"{recall_at:.3f}) is used instead.")

        thresholds[column] = {
            "threshold_f1": round(threshold_f1, 6),
            "f1_at_threshold": round(float(f1_scores[best_index]), 4),
            "threshold_recall90": round(threshold_recall90, 6),
            "recall_at_recall90": round(recall_at, 4),
            "precision_at_recall90": round(precision_at, 4),
        }
        print(f"{column:<22}{threshold_f1:>10.4f}{f1_scores[best_index]:>10.4f}"
              f"{threshold_recall90:>16.4f}{recall_at:>10.4f}{precision_at:>12.4f}")
    print("-" * 84)
    return thresholds


# ===========================================================================
# 9c - BOOTSTRAP CONFIDENCE INTERVALS
# ===========================================================================
def bootstrap_auc_ci(y_true, y_score, n_iterations=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Estimate a 95% confidence interval for AUC by bootstrap resampling.

    The test set is resampled with replacement 1000 times and the AUC is
    recomputed each time. The 2.5th and 97.5th percentiles of those 1000
    values bracket the true AUC with 95% confidence. This turns a single
    number into an honest range, which is what a journal referee will expect.
    """
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)

    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    scores = []
    for _ in range(n_iterations):
        indices = rng.randint(0, n_samples, n_samples)
        # A resample can by chance contain only one class; skip those.
        if len(np.unique(y_true[indices])) < 2:
            continue
        scores.append(roc_auc_score(y_true[indices], y_score[indices]))

    if not scores:
        return (np.nan, np.nan, np.nan)
    scores = np.array(scores)
    return (float(np.mean(scores)), float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


# ===========================================================================
# 9a - METRICS
# ===========================================================================
def compute_metrics(y_true, y_score, threshold):
    """
    Compute all seven metrics for one class at one decision threshold.

    Specificity - the fraction of safe compounds correctly identified as safe -
    is not provided by scikit-learn and is derived from the confusion matrix.
    """
    y_predicted = (y_score >= threshold).astype(int)

    metrics = {}
    metrics["AUC_ROC"] = (roc_auc_score(y_true, y_score)
                          if len(np.unique(y_true)) > 1 else np.nan)
    metrics["Average_Precision"] = (average_precision_score(y_true, y_score)
                                    if len(np.unique(y_true)) > 1 else np.nan)
    metrics["F1"] = f1_score(y_true, y_predicted, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_predicted)
    metrics["Precision"] = precision_score(y_true, y_predicted, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_predicted, zero_division=0)

    matrix = confusion_matrix(y_true, y_predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    metrics["Specificity"] = (true_negative / (true_negative + false_positive)
                              if (true_negative + false_positive) else 0.0)
    metrics["TN"] = int(true_negative)
    metrics["FP"] = int(false_positive)
    metrics["FN"] = int(false_negative)
    metrics["TP"] = int(true_positive)
    return metrics


# ===========================================================================
# 9e - PLOTS
# ===========================================================================
def plot_roc_curves(y_test, model_probabilities, output_dir):
    """One ROC curve figure per hazard, showing every model together."""
    print("\n[9e] Generating ROC curve plots ...")
    paths = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        code = column.split("_")[0]
        y_true = y_test[:, class_index]
        if len(np.unique(y_true)) < 2:
            log_issue("9e", f"{column}: test set has one class only - "
                            f"ROC curve cannot be drawn.")
            continue

        fig, ax = plt.subplots(figsize=(7, 6.5))
        for model_name, probabilities in model_probabilities.items():
            y_score = probabilities[:, class_index]
            false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(false_positive_rate, true_positive_rate, linewidth=2,
                    color=MODEL_COLOURS.get(model_name, "grey"),
                    label=f"{model_name} (AUC = {auc:.3f})")

        # The diagonal is what random guessing would achieve.
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6,
                label="Random guess (AUC = 0.500)")
        ax.set_xlabel("False positive rate (1 - specificity)", fontsize=12)
        ax.set_ylabel("True positive rate (recall)", fontsize=12)
        ax.set_title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                     f"ROC curves on the scaffold-split test set "
                     f"(n = {len(y_true):,}, positives = {int(y_true.sum()):,})",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, f"STEP9_ROC_curves_{code}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    print(f"      {len(paths)} ROC figures saved")
    return paths


def plot_pr_curves(y_test, model_probabilities, output_dir):
    """One precision-recall curve figure per hazard."""
    print("\n[9e] Generating precision-recall curve plots ...")
    paths = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        code = column.split("_")[0]
        y_true = y_test[:, class_index]
        if len(np.unique(y_true)) < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 6.5))
        for model_name, probabilities in model_probabilities.items():
            y_score = probabilities[:, class_index]
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            average_precision = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, linewidth=2,
                    color=MODEL_COLOURS.get(model_name, "grey"),
                    label=f"{model_name} (AP = {average_precision:.3f})")

        # For a PR curve the "random" baseline is the class prevalence.
        baseline = y_true.mean()
        ax.axhline(baseline, color="k", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"Random guess (AP = {baseline:.3f})")
        ax.set_xlabel("Recall (fraction of hazards found)", fontsize=12)
        ax.set_ylabel("Precision (fraction of alarms that are real)", fontsize=12)
        ax.set_title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                     f"Precision-recall curves "
                     f"(prevalence = {100 * baseline:.2f}%)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, f"STEP9_PR_curves_{code}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    print(f"      {len(paths)} precision-recall figures saved")
    return paths


def plot_confusion_matrices(y_test, probabilities, thresholds, best_model_name,
                            output_dir):
    """Confusion matrix heat map per hazard, for the best model."""
    print(f"\n[9e] Generating confusion matrices for {best_model_name} ...")
    paths = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        code = column.split("_")[0]
        y_true = y_test[:, class_index]
        threshold = thresholds[column]["threshold_f1"]
        y_predicted = (probabilities[:, class_index] >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_predicted, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5.2))
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False,
                    square=True, linewidths=1.5, linecolor="white",
                    xticklabels=["Predicted safe", "Predicted hazardous"],
                    yticklabels=["Actually safe", "Actually hazardous"],
                    annot_kws={"fontsize": 15, "fontweight": "bold"}, ax=ax)
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        # A false negative is a missed hazard - the dangerous kind of error.
        ax.set_title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                     f"{best_model_name}, threshold = {threshold:.3f}\n"
                     f"Missed hazards (false negatives) = {false_negative:,}",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(output_dir, f"STEP9_confusion_matrix_{code}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    print(f"      {len(paths)} confusion matrices saved")
    return paths


# ===========================================================================
# MAIN
# ===========================================================================
def evaluate_models():
    """Run the whole of Step 9 and save every table and figure."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 9 - MODEL EVALUATION")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    val_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_val_indices.npy"))
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Validation set: {X_val.shape[0]:,} compounds (threshold calibration)")
    print(f"Test set      : {X_test.shape[0]:,} compounds (final scoring)")

    # ---- load every trained model ----------------------------------------
    # Step 7 pickled its wrapper class from __main__, so the class has to be
    # registered here before those files can be read. See the function's
    # docstring in step7_model_training.py.
    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()
    _ablation_name, _ablation_file, _ablation_meta = get_ablation_identity()
    if _ablation_meta:
        print(f"   ablation resolved as: {_ablation_name} "
              f"({_ablation_meta.get('what_this_ablation_measures', '')[:70]})")

    # The Step 8 files are preferred where they exist: they carry the tuned
    # hyperparameters AND were refitted on the training set. The Step 7 files
    # are the fallback if tuning failed for that algorithm.
    models = {}
    model_files = [
        ("RandomForest", "STEP8_rf_tuned.pkl"),
        ("RandomForest", "STEP7_rf_model.pkl"),
        ("XGBoost", "STEP8_xgb_tuned.pkl"),
        ("XGBoost", "STEP7_xgb_models.pkl"),
        ("SVM", "STEP7_svm_model.pkl"),
        # The ablation's name depends on what it actually measured; see
        # get_ablation_identity(). Hard-coding "RandomForest_SMOTE" here
        # once mislabelled a run in which SMOTE never executed.
        (_ablation_name, _ablation_file),
    ]
    for name, filename in model_files:
        if name in models:
            continue      # the tuned Random Forest already loaded
        path = os.path.join(DIR_MODELS, filename)
        if os.path.exists(path):
            try:
                candidate = joblib.load(path)
                # Reject a model built for a different feature count rather
                # than letting it fail later with an opaque shape error. This
                # happens when a model file survives from a run on a different
                # dataset, and it must not be quietly evaluated.
                expected = getattr(candidate, "n_features_in_", None)
                if expected is None:
                    sub = (candidate.models if hasattr(candidate, "models")
                           else getattr(candidate, "estimators_", []))
                    for estimator in sub:
                        expected = getattr(estimator, "n_features_in_", None)
                        if expected is not None:
                            break
                feature_indices = getattr(candidate, "feature_indices", None)
                allowed = (len(feature_indices) if feature_indices is not None
                           else X_test.shape[1])
                if expected is not None and int(expected) != int(allowed):
                    log_issue("9", f"{filename} was trained on {int(expected)} "
                                   f"features but the current matrix has "
                                   f"{int(allowed)}. It is left over from a run "
                                   f"on different data and has been SKIPPED. "
                                   f"Delete it and re-run the step that "
                                   f"produces it.")
                    continue
                models[name] = candidate
                print(f"   loaded {name:<20} from {filename}")
            except Exception as exc:
                log_issue("9", f"could not load {filename}: {exc}")
    if not models:
        raise RuntimeError("Step 9 failed: no trained models found.")

    # ---- predictions -------------------------------------------------------
    print("\n[9a] Computing predictions ...")
    val_probabilities, test_probabilities = {}, {}
    for name, model in models.items():
        started = time.time()
        val_probabilities[name] = get_positive_probabilities(model, X_val)
        test_probabilities[name] = get_positive_probabilities(model, X_test)
        print(f"      {name:<20} predicted in {time.time() - started:.1f}s")

    # ---- 9b threshold calibration (per model) ------------------------------
    all_thresholds = {}
    for name in models:
        print(f"\n   --- thresholds for {name} ---")
        all_thresholds[name] = calibrate_thresholds(y_val, val_probabilities[name])

    with open(stamped("STEP9_calibrated_thresholds.json"), "w",
              encoding="utf-8") as fh:
        json.dump(all_thresholds, fh, indent=2)

    # ---- 9a + 9c metrics on the test set -----------------------------------
    print(f"\n[9a + 9c] Computing metrics and {N_BOOTSTRAP} bootstrap "
          f"confidence intervals ...")
    rows = []
    for name in models:
        probabilities = test_probabilities[name]
        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            y_true = y_test[:, class_index]
            y_score = probabilities[:, class_index]
            thresholds = all_thresholds[name][column]

            # The same metrics at three different decision thresholds.
            for threshold_name, threshold in [
                    ("default_0.5", 0.5),
                    ("calibrated_F1", thresholds["threshold_f1"]),
                    ("safety_recall90", thresholds["threshold_recall90"])]:
                metrics = compute_metrics(y_true, y_score, threshold)
                row = {"Model": name, "GHS_Column": column,
                       "Pictogram_Code": column.split("_")[0],
                       "Actual_Meaning": GHS_TRUE_MEANING[column],
                       "Threshold_Type": threshold_name,
                       "Threshold_Value": round(float(threshold), 6),
                       "N_Test": int(len(y_true)),
                       "N_Test_Positive": int(y_true.sum())}
                row.update({k: (round(float(v), 4) if isinstance(v, float) else v)
                            for k, v in metrics.items()})
                rows.append(row)

            # Bootstrap CI depends only on the scores, not the threshold, so
            # it is computed once per model+class.
            mean_auc, low, high = bootstrap_auc_ci(y_true, y_score)
            for row in rows[-3:]:
                row["AUC_bootstrap_mean"] = (round(mean_auc, 4)
                                             if np.isfinite(mean_auc) else np.nan)
                row["AUC_CI95_lower"] = round(low, 4) if np.isfinite(low) else np.nan
                row["AUC_CI95_upper"] = (round(high, 4)
                                         if np.isfinite(high) else np.nan)
        print(f"      {name} done")

    results = pd.DataFrame(rows)
    results.to_csv(stamped("STEP9_model_comparison_results.csv"), index=False)
    results.to_csv(os.path.join(DIR_EVAL,
                                f"STEP9_model_comparison_results_{TODAY}.csv"),
                   index=False)

    # ---- 9d comparison table -----------------------------------------------
    print("\n[9d] Model comparison - AUC-ROC on the scaffold-split test set")
    print("=" * 100)
    auc_table = results[results["Threshold_Type"] == "calibrated_F1"].pivot(
        index="GHS_Column", columns="Model", values="AUC_ROC")
    auc_table = auc_table.reindex(GHS_LABEL_COLUMNS)
    header = f"{'Class':<22}{'meaning':<26}"
    for name in auc_table.columns:
        header += f"{name:>20}"
    print(header)
    print("-" * 100)
    for column in GHS_LABEL_COLUMNS:
        meaning = GHS_TRUE_MEANING[column].split("(")[0].strip()[:24]
        line = f"{column:<22}{meaning:<26}"
        for name in auc_table.columns:
            value = auc_table.loc[column, name]
            line += f"{value:>20.4f}" if pd.notna(value) else f"{'n/a':>20}"
        print(line)
    print("-" * 100)
    mean_line = f"{'MEAN':<22}{'':<26}"
    for name in auc_table.columns:
        mean_line += f"{auc_table[name].mean():>20.4f}"
    print(mean_line)
    print("=" * 100)

    auc_table.to_csv(stamped("STEP9_auc_comparison_table.csv"))

    # Also show the calibrated-threshold F1 and MCC, which is where the
    # differences between models really show up.
    for metric in ("F1", "MCC"):
        table = results[results["Threshold_Type"] == "calibrated_F1"].pivot(
            index="GHS_Column", columns="Model", values=metric).reindex(
            GHS_LABEL_COLUMNS)
        print(f"\n{metric} at the calibrated threshold")
        print("-" * 100)
        print(f"{'Class':<22}" + "".join(f"{n:>20}" for n in table.columns))
        for column in GHS_LABEL_COLUMNS:
            line = f"{column:<22}"
            for name in table.columns:
                value = table.loc[column, name]
                line += f"{value:>20.4f}" if pd.notna(value) else f"{'n/a':>20}"
            print(line)
        print("-" * 100)
        print(f"{'MEAN':<22}" + "".join(f"{table[n].mean():>20.4f}"
                                        for n in table.columns))
        table.to_csv(stamped(f"STEP9_{metric}_comparison_table.csv"))

    # ---- pick the best model ------------------------------------------------
    # Selecting on mean AUC alone is not safe here. Two models can finish
    # within a ten-thousandth of each other, which is far inside the
    # uncertainty of the estimate, and whichever happens to be ahead at the
    # fourth decimal place would then decide which model is explained in Step
    # 10 and deployed in Step 12.
    #
    # The tie tolerance is therefore taken from the data itself: the median
    # half-width of the bootstrap confidence intervals already computed for
    # the leading model. Any model whose mean AUC falls inside that band is
    # treated as statistically indistinguishable from the leader, and the tie
    # is broken on mean Matthews correlation coefficient - the metric Step 6d
    # designated as primary for rare classes, and the one that actually
    # discriminates between models on imbalanced data.
    mean_aucs = auc_table.mean()
    leader = str(mean_aucs.idxmax())

    leader_rows = results[(results["Model"] == leader) &
                          (results["Threshold_Type"] == "calibrated_F1")]
    half_widths = ((leader_rows["AUC_CI95_upper"] -
                    leader_rows["AUC_CI95_lower"]) / 2).dropna()
    tie_tolerance = float(half_widths.median()) if len(half_widths) else 0.01

    mcc_table = results[results["Threshold_Type"] == "calibrated_F1"].pivot(
        index="GHS_Column", columns="Model", values="MCC").reindex(
        GHS_LABEL_COLUMNS)
    mean_mcc = mcc_table.mean()

    contenders = mean_aucs[mean_aucs >= mean_aucs.max() - tie_tolerance]
    if len(contenders) > 1:
        best_model_name = str(mean_mcc[contenders.index].idxmax())
        print(f"\nMODEL SELECTION")
        print("-" * 78)
        print(f"Highest mean AUC : {leader} ({mean_aucs.max():.4f})")
        print(f"Tie tolerance    : {tie_tolerance:.4f} "
              f"(median bootstrap CI half-width for {leader})")
        print(f"Indistinguishable: "
              f"{', '.join(f'{m} ({mean_aucs[m]:.4f})' for m in contenders.index)}")
        print(f"Tie broken on mean MCC: "
              f"{', '.join(f'{m} ({mean_mcc[m]:.4f})' for m in contenders.index)}")
        if best_model_name != leader:
            log_issue("9d", f"{leader} had the highest mean AUC "
                            f"({mean_aucs.max():.4f}) but only by "
                            f"{mean_aucs.max() - mean_aucs[best_model_name]:.4f}, "
                            f"far inside the bootstrap uncertainty of "
                            f"+/-{tie_tolerance:.4f}. {best_model_name} was "
                            f"selected instead on mean MCC "
                            f"({mean_mcc[best_model_name]:.4f} vs "
                            f"{mean_mcc[leader]:.4f}), which separates the "
                            f"models clearly and is the primary metric for the "
                            f"rare classes.")
        print("-" * 78)
    else:
        best_model_name = leader

    print(f"\nBEST MODEL: {best_model_name} "
          f"(mean AUC {mean_aucs[best_model_name]:.4f}, "
          f"mean MCC {mean_mcc[best_model_name]:.4f})")

    # ---- 9e plots -----------------------------------------------------------
    plot_roc_curves(y_test, test_probabilities, DIR_EVAL)
    plot_pr_curves(y_test, test_probabilities, DIR_EVAL)
    plot_confusion_matrices(y_test, test_probabilities[best_model_name],
                            all_thresholds[best_model_name], best_model_name,
                            DIR_EVAL)

    # ---- model comparison heat map (also used as Figure 6 in Step 13) ------
    fig, ax = plt.subplots(figsize=(9, 7))
    heat_data = auc_table.copy()
    heat_data.index = [f"{c.split('_')[0]}: "
                       f"{GHS_TRUE_MEANING[c].split('(')[0].strip()}"
                       for c in heat_data.index]
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn", center=0.75,
                vmin=0.5, vmax=1.0, linewidths=1, linecolor="white",
                cbar_kws={"label": "AUC-ROC"}, ax=ax,
                annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Model comparison: AUC-ROC by algorithm and GHS hazard class\n"
                 "(scaffold-split test set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("")
    fig.tight_layout()
    heatmap_path = os.path.join(DIR_EVAL, "STEP9_model_comparison_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- save the summary ---------------------------------------------------
    summary = {
        "best_model": best_model_name,
        "model_selection": {
            "highest_mean_auc_model": leader,
            "highest_mean_auc": round(float(mean_aucs.max()), 4),
            "tie_tolerance_from_bootstrap": round(tie_tolerance, 4),
            "statistically_indistinguishable": list(contenders.index),
            "tie_broken_on": ("mean MCC" if len(contenders) > 1
                              else "not required - clear winner on AUC"),
        },
        "mean_auc_per_model": {k: round(float(v), 4) for k, v in mean_aucs.items()},
        "mean_mcc_per_model": {k: round(float(v), 4) for k, v in mean_mcc.items()},
        "auc_per_class_best_model": {
            column: (round(float(auc_table.loc[column, best_model_name]), 4)
                     if pd.notna(auc_table.loc[column, best_model_name]) else None)
            for column in GHS_LABEL_COLUMNS},
        "n_test_compounds": int(len(test_idx)),
        "n_bootstrap_iterations": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(time.time() - total_start, 1),
    }
    with open(stamped("STEP9_evaluation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP9_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 9 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 9 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Scored every model on the held-out scaffold-split test")
    print("                set with seven metrics per hazard class, calibrated the")
    print("                decision thresholds on validation data, computed")
    print("                bootstrap confidence intervals, and drew ROC,")
    print("                precision-recall and confusion-matrix figures.")
    print(f"BEST MODEL    : {best_model_name} "
          f"(mean AUC {mean_aucs.max():.4f} across nine classes)")
    print(f"OUTPUT FILES  : {stamped('STEP9_model_comparison_results.csv')}")
    print(f"                {stamped('STEP9_auc_comparison_table.csv')}")
    print(f"                {stamped('STEP9_calibrated_thresholds.json')}")
    print(f"                {DIR_EVAL}\\STEP9_ROC_curves_GHS*.png (9 files)")
    print(f"                {DIR_EVAL}\\STEP9_PR_curves_GHS*.png (9 files)")
    print(f"                {DIR_EVAL}\\STEP9_confusion_matrix_GHS*.png (9 files)")
    print(f"                {heatmap_path}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print(f"ELAPSED       : {summary['elapsed_seconds'] / 60:.1f} minutes")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    evaluate_models()
