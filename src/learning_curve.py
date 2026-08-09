"""
LEARNING CURVE - HOW MUCH DOES MORE TRAINING DATA ACTUALLY BUY?
===============================================================
The modelling subset was capped at 40,000 compounds by the memory of the
machine this project was built on. That raises a fair question from any
reviewer: would the model have been better with all 243,323?

A learning curve answers it with evidence instead of assertion. The model is
retrained on progressively larger slices of the training set and scored each
time on the SAME held-out test set. If performance has flattened by the time
the full training set is used, then more data would add little, and the subset
stops being a limitation and becomes a demonstrated sufficiency.

XGBoost is used because Step 9 selected it as the best model.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_FEATURES, DIR_SPLITS, DIR_EVAL,
                        DIR_PUB, GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Fractions of the training set to try. Spaced more densely at the low end,
# because that is where a learning curve changes fastest.
FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00]
N_JOBS = 3


def train_and_score(X_train, y_train, X_test, y_test, class_weights):
    """
    Train one XGBoost model per hazard on the given slice and score them.

    Returns the AUC per class on the fixed test set. A class with fewer than
    two positive examples in the slice cannot be trained, and a class with only
    one label present in the test set has no defined AUC; both return NaN so
    that they are visibly absent rather than silently wrong.
    """
    aucs = {}
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_class = y_train[:, class_index].astype(int)
        y_true = y_test[:, class_index].astype(int)

        if y_class.sum() < 2 or len(np.unique(y_true)) < 2:
            aucs[column] = np.nan
            continue

        n_positive = int(y_class.sum())
        n_negative = int(len(y_class) - n_positive)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(n_negative / n_positive) if n_positive else 1.0,
            random_state=RANDOM_SEED, n_jobs=N_JOBS, tree_method="hist",
            eval_metric="aucpr", verbosity=0)
        model.fit(X_train, y_class)
        aucs[column] = float(roc_auc_score(
            y_true, model.predict_proba(X_test)[:, 1]))
    return aucs


def main():
    """Build the learning curve and save the figure and table."""
    started = time.time()
    print("=" * 78)
    print("LEARNING CURVE - does more training data improve the model?")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))

    X_test, y_test = X[test_idx], y[test_idx]
    with open(stamped("STEP6_imbalance_config.json"), encoding="utf-8") as fh:
        class_weights = json.load(fh)["class_weights"]

    print(f"Training pool : {len(train_idx):,} compounds")
    print(f"Test set      : {len(test_idx):,} compounds (fixed throughout)")
    print(f"Fractions     : {FRACTIONS}\n")

    rng = np.random.RandomState(RANDOM_SEED)
    # One fixed shuffle, then take prefixes of it. This makes each larger slice
    # a superset of the smaller ones, which is what a learning curve requires -
    # otherwise the curve would also reflect which compounds happened to be
    # drawn, not just how many.
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)

    rows = []
    print(f"{'n_train':>9}{'mean AUC':>11}{'GHS01':>9}{'GHS05':>9}"
          f"{'GHS07':>9}{'GHS09':>9}{'minutes':>9}")
    print("-" * 66)

    for fraction in FRACTIONS:
        n = max(50, int(round(fraction * len(shuffled))))
        slice_idx = np.sort(shuffled[:n])
        X_train, y_train = X[slice_idx], y[slice_idx]

        slice_start = time.time()
        aucs = train_and_score(X_train, y_train, X_test, y_test, class_weights)
        elapsed = (time.time() - slice_start) / 60

        valid = [v for v in aucs.values() if np.isfinite(v)]
        mean_auc = float(np.mean(valid)) if valid else np.nan

        row = {"fraction": fraction, "n_train": n, "mean_auc": round(mean_auc, 4),
               "minutes": round(elapsed, 2)}
        for column, value in aucs.items():
            row[column] = round(value, 4) if np.isfinite(value) else None
            # How many positives this slice actually contained, which is what
            # limits the rare classes.
            row["n_pos_" + column.split("_")[0]] = int(
                y_train[:, GHS_LABEL_COLUMNS.index(column)].sum())
        rows.append(row)

        def show(code):
            """Format one class's AUC for the console table."""
            column = next(c for c in GHS_LABEL_COLUMNS if c.startswith(code))
            value = aucs[column]
            return f"{value:>9.4f}" if np.isfinite(value) else f"{'n/a':>9}"

        print(f"{n:>9,}{mean_auc:>11.4f}{show('GHS01')}{show('GHS05')}"
              f"{show('GHS07')}{show('GHS09')}{elapsed:>9.2f}")

    print("-" * 66)
    table = pd.DataFrame(rows)
    table.to_csv(stamped("EXTRA_learning_curve.csv"), index=False)

    # ---- interpretation -----------------------------------------------------
    # A single "did the last doubling help?" test is too crude to decide this.
    # Three separate pieces of evidence are reported instead, because they do
    # not all point the same way and the honest answer depends on all three.
    CI_HALF_WIDTH = 0.0139        # bootstrap 95% CI half-width from Step 9

    final = table["mean_auc"].iloc[-1]
    lines = []
    lines.append("LEARNING CURVE INTERPRETATION")
    lines.append("=" * 74)
    lines.append(f"Mean AUC at {int(table['n_train'].iloc[-1]):,} training "
                 f"compounds: {final:.4f}")
    lines.append(f"Bootstrap 95% CI half-width (Step 9)     : "
                 f"+/-{CI_HALF_WIDTH:.4f}")
    lines.append("")

    # ---- evidence 1: is the marginal return still worth anything? ----------
    lines.append("1. MARGINAL RETURN PER 1,000 ADDITIONAL COMPOUNDS")
    lines.append("-" * 74)
    rates = []
    for i in range(1, len(table)):
        d_n = table["n_train"].iloc[i] - table["n_train"].iloc[i - 1]
        d_auc = table["mean_auc"].iloc[i] - table["mean_auc"].iloc[i - 1]
        rate = 1000 * d_auc / d_n
        rates.append(rate)
        lines.append(f"   {table['n_train'].iloc[i-1]:>6,} -> "
                     f"{table['n_train'].iloc[i]:>6,}   {rate:+.5f}")
    decay = rates[0] / rates[-1] if rates[-1] else float("inf")
    lines.append(f"   The marginal return fell by a factor of {decay:.0f} "
                 f"across the curve.")
    lines.append("")

    # ---- evidence 2: which classes have actually stopped improving? --------
    lines.append("2. PER-CLASS STATUS AT THE FINAL STEP")
    lines.append("-" * 74)
    n_plateaued = 0
    for column in GHS_LABEL_COLUMNS:
        if column not in table.columns or table[column].isna().all():
            continue
        step = table[column].iloc[-1] - table[column].iloc[-2]
        # A class counts as still improving only if it moved UPWARD by more
        # than the noise floor. A downward move is noise, not improvement -
        # treating a negative change as "still rising" would be nonsense.
        rising = step > 0.003
        n_plateaued += (not rising)
        lines.append(f"   {column:<24}{table[column].iloc[-1]:>8.4f}"
                     f"{step:>+10.4f}   "
                     f"{'still improving' if rising else 'plateaued'}")
    lines.append(f"   {n_plateaued} of 9 classes have stopped improving.")
    lines.append("")

    # ---- evidence 3: extrapolation, clearly labelled as such ---------------
    lines.append("3. EXTRAPOLATION TO THE FULL DATASET (TREAT WITH CAUTION)")
    lines.append("-" * 74)
    tail = table.tail(4)
    slope, intercept = np.polyfit(np.log(tail["n_train"]), tail["mean_auc"], 1)
    predicted_full = intercept + slope * np.log(194658)
    predicted_gain = predicted_full - final
    lines.append(f"   A log-linear fit to the last four points predicts")
    lines.append(f"   {predicted_full:.4f} at 194,658 training compounds, a gain")
    lines.append(f"   of {predicted_gain:+.4f}.")
    lines.append(f"   This extrapolates roughly six times beyond the measured")
    lines.append(f"   range, so it is an indication and not a result.")
    lines.append("")

    lines.append("CONCLUSION")
    lines.append("-" * 74)
    if predicted_gain > CI_HALF_WIDTH and abs(rates[-1]) < 0.0005:
        lines.append("   The evidence is genuinely mixed and should be reported")
        lines.append("   as such. Returns have all but stopped at the measured")
        lines.append("   scale - the last increment of data changed the mean AUC")
        lines.append(f"   by {table['mean_auc'].iloc[-1] - table['mean_auc'].iloc[-2]:+.4f}, "
                     f"far inside the confidence interval - and")
        lines.append(f"   {n_plateaued} of the 9 classes have flattened. Against that, "
                     f"the")
        lines.append("   extrapolation suggests the full dataset could still add")
        lines.append(f"   about {predicted_gain:+.3f}, which would be larger than the")
        lines.append("   confidence interval.")
        lines.append("")
        lines.append("   The curve narrows the question but does not settle it.")
        lines.append("   The only way to settle it is to train on the full")
        lines.append("   dataset, which the accompanying Colab notebook does.")
    elif predicted_gain <= CI_HALF_WIDTH:
        lines.append("   The curve has flattened and the extrapolated gain is")
        lines.append("   smaller than the confidence interval of the estimate.")
        lines.append("   The subset is sufficient; report this as a demonstrated")
        lines.append("   sufficiency rather than a limitation.")
    else:
        lines.append("   The curve is still rising at the measured scale, so the")
        lines.append("   subset size is a genuine limitation and should be")
        lines.append("   reported as one.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(stamped("EXTRA_learning_curve_interpretation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")

    # ---- figure -------------------------------------------------------------
    sns.set_style("whitegrid")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    ax_left.plot(table["n_train"], table["mean_auc"], "o-", linewidth=2.4,
                 markersize=8, color="#0072B2")
    ax_left.axhline(final, linestyle="--", color="grey", alpha=0.7,
                    label=f"final = {final:.4f}")
    # The bootstrap uncertainty band from Step 9: differences inside this band
    # are not meaningful.
    ax_left.fill_between(table["n_train"], final - 0.0139, final + 0.0139,
                         color="#0072B2", alpha=0.12,
                         label="bootstrap 95% CI half-width (0.0139)")
    ax_left.set_xlabel("Number of training compounds")
    ax_left.set_ylabel("Mean AUC-ROC across nine classes")
    ax_left.set_title("(a) Learning curve, XGBoost on the scaffold-split test set",
                      fontweight="bold", fontsize=12)
    ax_left.legend(loc="lower right", fontsize=10)

    colours = sns.color_palette("husl", 9)
    for index, column in enumerate(GHS_LABEL_COLUMNS):
        if column in table.columns and table[column].notna().any():
            ax_right.plot(table["n_train"], table[column], "o-", linewidth=1.7,
                          markersize=5, color=colours[index],
                          label=f"{column.split('_')[0]} "
                                f"{GHS_TRUE_MEANING[column].split('(')[0].strip()}")
    ax_right.set_xlabel("Number of training compounds")
    ax_right.set_ylabel("AUC-ROC")
    ax_right.set_title("(b) Per-class learning curves", fontweight="bold",
                       fontsize=12)
    ax_right.legend(fontsize=8.5, loc="lower right", ncol=1)

    fig.suptitle("Does more training data improve GHS hazard prediction?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    for folder in (DIR_EVAL, os.path.join(DIR_PUB, "figures")):
        fig.savefig(os.path.join(folder, "Figure9_learning_curve.png"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {stamped('EXTRA_learning_curve.csv')}")
    print(f"Saved: {os.path.join(DIR_PUB, 'figures', 'Figure9_learning_curve.png')}")
    print(f"Total time: {(time.time() - started) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
