"""
CONTROLLED TRAINING-SIZE EXPERIMENT
===================================
Does training on more compounds actually improve GHS hazard prediction?

The first attempt at this question compared the local 40,000-compound run with
a Colab run on all 243,323. That comparison was confounded three ways and
cannot answer it:

  1. the two runs were scored on DIFFERENT test sets, because the scaffold
     split was recomputed on a different pool of compounds;
  2. the local model used tuned hyperparameters and the Colab model did not;
  3. the rare classes had six times higher prevalence in the 40,000 subset,
     making its test set easier for exactly those classes.

This script removes all three. One fixed test set, one fixed set of
hyperparameters, and the only thing that changes is how many compounds the
model is trained on. Every training slice is a superset of the smaller ones,
so the curve reflects quantity rather than which compounds were drawn.

Memory note: the full training slice is 194,658 x ~800 float32, close to what
this machine can hold. XGBoost's QuantileDMatrix is used because it compresses
the data into histogram bins up front and frees the dense copy, and each
condition is released before the next begins.

Author : Sareer Ahmad
"""

import os
import gc
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
from ghs_config import (FIGURE_DPI, SERIES_HUE, ACCENT_HUE, CONTEXT_GREY,
                        GHS_TRUE_MEANING as _MEANING)
from ghs_config import (RANDOM_SEED, DIR_EVAL, DIR_PUB, GHS_LABEL_COLUMNS,
                        GHS_TRUE_MEANING, seed_everything, stamped)

seed_everything()

import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import VarianceThreshold

COLAB = r"D:\GHS_Project\colab_results"

# Training-set sizes to compare. 32,000 matches the local run exactly, so the
# first point is directly comparable with the published result.
CONDITIONS = [32000, 64000, 128000, 194658]

# Fixed for every condition - these are the Step 7 defaults, not the tuned
# settings, so that no condition gets an unfair advantage.
PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "seed": RANDOM_SEED,
    "nthread": 3,
    "verbosity": 0,
}
N_ROUNDS = 200


def make_figure(table, first):
    """
    Draw Figure 10 from the results table.

    Separated from main() so the figure can be redrawn from the saved
    CSV without re-running the experiment, which takes hours.
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(table["n_train"], table["mean_auc"], "o-", linewidth=2.4,
             markersize=9, color=ACCENT_HUE)
    ax1.axhline(first["mean_auc"], linestyle="--", color="grey", alpha=0.8,
                label=f"at 32,000 = {first['mean_auc']:.4f}")
    ax1.fill_between(table["n_train"], first["mean_auc"] - 0.0139,
                     first["mean_auc"] + 0.0139, color=ACCENT_HUE, alpha=0.12,
                     label="bootstrap 95% CI (+/-0.0139)")
    ax1.set_xlabel("Number of training compounds")
    ax1.set_ylabel("Mean AUC-ROC across nine classes")
    ax1.set_title("(a) Controlled comparison\nsame test set, same "
                  "hyperparameters", fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right", fontsize=10)

    # Nine generated hues are not distinguishable, so the mass is drawn in grey
    # and only the best and worst classes are coloured and labelled - the two
    # the caption discusses. See the same treatment in learning_curve.py.
    plotted = [c for c in GHS_LABEL_COLUMNS
               if c in table.columns and table[c].notna().any()]
    finals = {c: table[c].dropna().iloc[-1] for c in plotted}
    best = max(finals, key=finals.get)
    worst = min(finals, key=finals.get)

    for column in plotted:
        highlight = column in (best, worst)
        ax2.plot(table["n_train"], table[column], "o-",
                 linewidth=2.4 if highlight else 1.3,
                 markersize=6 if highlight else 3.5,
                 color=(SERIES_HUE if column == best
                        else ACCENT_HUE if column == worst else CONTEXT_GREY),
                 zorder=3 if highlight else 1)

    for column, colour in ((best, SERIES_HUE), (worst, ACCENT_HUE)):
        ax2.annotate(f"{column.split('_')[0]} "
                     f"{GHS_TRUE_MEANING[column].split('(')[0].strip()}",
                     xy=(table["n_train"].iloc[-1], finals[column]),
                     xytext=(-8, 8 if column == best else -16),
                     textcoords="offset points", ha="right", fontsize=10,
                     fontweight="bold", color=colour, zorder=4)

    ax2.plot([], [], color=CONTEXT_GREY, linewidth=1.3,
             label=f"other {len(plotted) - 2} classes")
    ax2.legend(fontsize=9.5, loc="lower right")
    ax2.set_xlabel("Number of training compounds")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("(b) Per class, best and worst labelled", fontweight="bold",
                  fontsize=12)

    fig.suptitle("Effect of training-set size, with test set held fixed",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    for folder in (DIR_EVAL, os.path.join(DIR_PUB, "figures")):
        fig.savefig(os.path.join(folder, "Figure10_controlled_size.png"),
                    dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the controlled experiment and save the results and figure."""
    started = time.time()
    print("=" * 78)
    print("CONTROLLED TRAINING-SIZE EXPERIMENT")
    print("=" * 78)

    # ---- load, keeping the big matrix on disk until it is needed -----------
    X_all = np.load(os.path.join(COLAB, "colab_X_full.npy"), mmap_mode="r")
    train_idx = np.load(os.path.join(COLAB, "colab_train_idx.npy"))
    test_idx = np.load(os.path.join(COLAB, "colab_test_idx.npy"))

    labels = pd.read_csv(stamped("STEP3_cleaned_ghs_dataset.csv"),
                         usecols=GHS_LABEL_COLUMNS, low_memory=False)
    y_all = labels.to_numpy().astype(np.int8)
    del labels
    gc.collect()

    if y_all.shape[0] != X_all.shape[0]:
        raise SystemExit(
            f"Row mismatch: descriptors have {X_all.shape[0]:,} rows but the "
            f"cleaned dataset has {y_all.shape[0]:,}. The descriptor matrix "
            f"must come from the same file, in the same order.")

    print(f"Descriptors : {X_all.shape[0]:,} x {X_all.shape[1]:,}")
    print(f"Training pool: {len(train_idx):,}")
    print(f"Test set     : {len(test_idx):,}  (FIXED for every condition)")

    # ---- variance filter, fitted once on the training pool ----------------
    # Fitting it once and reusing it means every condition sees exactly the
    # same feature space. Fitting per condition would let the feature set vary
    # with training size, which is another confound.
    print("\nFitting the variance filter on a sample of the training pool ...")
    rng = np.random.RandomState(RANDOM_SEED)
    sample = np.sort(rng.choice(train_idx, min(40000, len(train_idx)),
                                replace=False))
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(np.asarray(X_all[sample], dtype=np.float32))
    keep_cols = np.where(selector.get_support())[0]
    print(f"   {len(keep_cols):,} of {X_all.shape[1]:,} descriptors retained")
    del sample
    gc.collect()

    # ---- the fixed test set ------------------------------------------------
    X_test = np.asarray(X_all[test_idx][:, keep_cols], dtype=np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = y_all[test_idx]
    dtest = xgb.QuantileDMatrix(X_test)
    print(f"   test matrix: {X_test.nbytes/1e9:.2f} GB")
    del X_test
    gc.collect()

    # One fixed shuffle; every condition takes a prefix, so larger slices
    # contain the smaller ones.
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)

    rows = []
    for n_train in CONDITIONS:
        if n_train > len(shuffled):
            continue
        slice_idx = np.sort(shuffled[:n_train])

        print(f"\n--- training on {n_train:,} compounds ---")
        t0 = time.time()
        X_train = np.asarray(X_all[slice_idx][:, keep_cols], dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = y_all[slice_idx]
        print(f"    matrix {X_train.nbytes/1e9:.2f} GB, "
              f"built in {time.time()-t0:.0f}s")

        result = {"n_train": n_train}
        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            yc = y_train[:, class_index].astype(int)
            yt = y_test[:, class_index].astype(int)
            n_pos = int(yc.sum())
            if n_pos < 2 or len(np.unique(yt)) < 2:
                result[column] = np.nan
                continue

            params = dict(PARAMS)
            params["scale_pos_weight"] = (len(yc) - n_pos) / n_pos
            params["objective"] = "binary:logistic"

            dtrain = xgb.QuantileDMatrix(X_train, label=yc)
            booster = xgb.train(params, dtrain, num_boost_round=N_ROUNDS)
            scores = booster.predict(dtest)
            result[column] = float(roc_auc_score(yt, scores))
            result["ap_" + column.split("_")[0]] = float(
                average_precision_score(yt, scores))
            result["npos_" + column.split("_")[0]] = n_pos
            del dtrain, booster
            gc.collect()
            print(f"    {column:<24} n+={n_pos:>7,}  AUC={result[column]:.4f}")

        valid = [result[c] for c in GHS_LABEL_COLUMNS
                 if isinstance(result.get(c), float) and np.isfinite(result[c])]
        result["mean_auc"] = float(np.mean(valid)) if valid else np.nan
        result["minutes"] = round((time.time() - t0) / 60, 2)
        rows.append(result)
        print(f"    MEAN AUC = {result['mean_auc']:.4f}   "
              f"({result['minutes']:.1f} min)")

        del X_train, y_train
        gc.collect()

    table = pd.DataFrame(rows)
    table.to_csv(stamped("EXTRA_controlled_size_experiment.csv"), index=False)

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULT - same test set, same hyperparameters, only size varies")
    print("=" * 78)
    header = f"{'n_train':>9}{'mean AUC':>11}"
    for column in GHS_LABEL_COLUMNS:
        header += f"{column.split('_')[0]:>9}"
    print(header)
    print("-" * len(header))
    for _, row in table.iterrows():
        line = f"{int(row['n_train']):>9,}{row['mean_auc']:>11.4f}"
        for column in GHS_LABEL_COLUMNS:
            value = row.get(column)
            line += f"{value:>9.4f}" if pd.notna(value) else f"{'n/a':>9}"
        print(line)
    print("-" * len(header))

    first, last = table.iloc[0], table.iloc[-1]
    gain = last["mean_auc"] - first["mean_auc"]
    print(f"\nGoing from {int(first['n_train']):,} to {int(last['n_train']):,} "
          f"training compounds:")
    print(f"   mean AUC {first['mean_auc']:.4f} -> {last['mean_auc']:.4f}  "
          f"({gain:+.4f})")
    print(f"   bootstrap 95% CI half-width for reference: +/-0.0139")
    if gain > 0.0139:
        verdict = ("Training on the full dataset gives a real improvement, "
                   "larger than the uncertainty of the estimate.")
    elif gain < -0.0139:
        verdict = ("Training on the full dataset makes performance WORSE by "
                   "more than the uncertainty - investigate before reporting.")
    else:
        verdict = ("The difference is inside the confidence interval. The "
                   "40,000-compound subset is sufficient; the memory "
                   "constraint did not limit the published results.")
    print(f"\nVERDICT: {verdict}")

    per_class = []
    for column in GHS_LABEL_COLUMNS:
        if pd.notna(first.get(column)) and pd.notna(last.get(column)):
            per_class.append((column, first[column], last[column],
                              last[column] - first[column]))
    print(f"\n{'Class':<24}{'@32,000':>10}{'@full':>10}{'change':>10}")
    print("-" * 54)
    for column, a, b, d in per_class:
        print(f"{column:<24}{a:>10.4f}{b:>10.4f}{d:>+10.4f}")
    print("-" * 54)

    # ---- figure -------------------------------------------------------------
    make_figure(table, first)

    with open(stamped("EXTRA_controlled_size_verdict.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("CONTROLLED TRAINING-SIZE EXPERIMENT\n")
        fh.write("=" * 74 + "\n")
        fh.write(f"Test set: {len(test_idx):,} compounds, identical in every "
                 f"condition\nHyperparameters: identical in every condition\n\n")
        fh.write(table[["n_train", "mean_auc", "minutes"]].to_string(index=False))
        fh.write(f"\n\nChange from {int(first['n_train']):,} to "
                 f"{int(last['n_train']):,}: {gain:+.4f}\n")
        fh.write(f"Bootstrap 95% CI half-width: +/-0.0139\n\nVERDICT: {verdict}\n")
        fh.write(f"\n{'Class':<24}{'@32,000':>10}{'@full':>10}{'change':>10}\n")
        for column, a, b, d in per_class:
            fh.write(f"{column:<24}{a:>10.4f}{b:>10.4f}{d:>+10.4f}\n")

    print(f"\nSaved: {stamped('EXTRA_controlled_size_experiment.csv')}")
    print(f"Saved: {stamped('EXTRA_controlled_size_verdict.txt')}")
    print(f"Saved: Figure10_controlled_size.png")
    print(f"Total time: {(time.time() - started)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
