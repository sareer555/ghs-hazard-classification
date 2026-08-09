"""
STEP 7 - MODEL TRAINING
=======================
Three different algorithms are trained on the same data so that their
performance can be compared fairly in Step 9.

7a  Random Forest      - hundreds of decision trees voting together. Robust,
                         needs no feature scaling, and gives feature
                         importances for free.
7b  XGBoost            - gradient boosting, where each new tree is built to
                         correct the mistakes of the trees before it. Usually
                         the strongest performer on tabular chemical data.
7c  Support Vector Machine - finds the widest possible boundary between the
                         two classes after projecting the data into a higher
                         dimensional space with an RBF kernel.

A methodological conflict in the proposal, and how it was resolved
------------------------------------------------------------------
The proposal asks for the Random Forest to be fitted on the SMOTE-balanced
training data AND to be configured with class_weight='balanced'. These are two
different remedies for the same problem. Applying both at once corrects the
imbalance twice over, which pushes the model to over-predict rare hazards and
inflates its false-alarm rate.

Resolution adopted here:
  * PRIMARY model  - class_weight='balanced' on the original training data,
                     which is the exact classifier configuration the proposal
                     specifies and keeps every training molecule real.
  * ABLATION model - the same Random Forest trained per class on the
                     SMOTE-balanced sets from Step 6, with no class weighting.
Both are trained and both are reported, so the effect of the choice is
measured rather than assumed. This turns an ambiguity in the proposal into a
useful ablation experiment for the paper.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_SPLITS, DIR_FEATURES, DIR_MODELS,
                        DIR_LOGS, GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

ISSUE_LOG = []
TIMINGS = {}

# ---------------------------------------------------------------------------
# HARDWARE BUDGETS
# This machine has two physical cores and 7.9 GB of RAM. An RBF support vector
# machine costs roughly O(n^2) in both time and memory, so it cannot be
# trained on tens of thousands of compounds here. These budgets keep every
# model inside the machine's limits; each one that binds is written to the
# issue log so the limitation appears in the paper.
# ---------------------------------------------------------------------------
SVM_MAX_TRAINING_SAMPLES = 8000    # RBF kernel matrix stays about 0.5 GB
SVM_TOP_N_FEATURES = 100           # the fallback named in proposal step 7c
N_JOBS = 3                         # leave one logical core free
RF_MAX_DEPTH = 25                  # see the note in train_random_forest
RF_MEMORY_BUDGET_BYTES = 2.0e9     # ceiling for the whole nine-class forest


def choose_min_samples_leaf(n_train, n_estimators=200,
                            n_classes=len(GHS_LABEL_COLUMNS),
                            budget_bytes=RF_MEMORY_BUDGET_BYTES):
    """
    Pick the smallest leaf size that keeps the whole forest inside the budget.

    A decision tree grown until each leaf holds `min_samples_leaf` compounds
    has roughly `2 * n_train / min_samples_leaf` nodes, and each node costs
    about 80 bytes in scikit-learn. The project trains `n_estimators` trees
    for each of nine hazard classes and holds all of them in memory at once
    inside the MultiOutputClassifier, so:

        total bytes = 2 * n_train / leaf * 80 * n_estimators * n_classes

    Rearranged, this gives the leaf size needed to stay under the budget.
    Guessing a fixed value would either waste accuracy on a small dataset or
    exhaust memory on a large one, so it is computed from the actual data.
    """
    bytes_per_node = 80
    required = (2 * n_train * bytes_per_node * n_estimators * n_classes
                / budget_bytes)
    return max(2, int(np.ceil(required)))


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


def register_pickle_compatibility():
    """
    Make PerLabelClassifier loadable in any script, not just this one.

    Python records the *module* a class came from when pickling an object.
    Step 7 is run as a script, so at that moment PerLabelClassifier belongs to
    the module called "__main__", and that is the name written into the model
    files. When a different script - Step 9, Step 10, the Streamlit app - later
    tries to load one of those files, its own "__main__" is a different module
    and the class cannot be found, so joblib raises
    "Can't get attribute 'PerLabelClassifier'".

    Registering the class under the running script's "__main__" before loading
    gives the unpickler somewhere to find it. Call this once, before any
    joblib.load of a Step 7 model.
    """
    import __main__
    if not hasattr(__main__, "PerLabelClassifier"):
        __main__.PerLabelClassifier = PerLabelClassifier


class PerLabelClassifier:
    """
    Hold nine independent binary classifiers behind a MultiOutputClassifier-like
    interface.

    XGBoost and the SMOTE ablation both need a separate model per hazard, but
    every later step (evaluation, SHAP, the Streamlit app) expects one object
    with `.predict()` and `.predict_proba()`. This small wrapper provides
    exactly that, so nothing downstream has to know the difference.
    """

    def __init__(self, models, label_names, feature_indices=None):
        """
        models         : list of fitted binary classifiers, one per hazard
        label_names    : the nine GHS column names, in the same order
        feature_indices: optional column subset each model was trained on
                         (used by the SVM, which trains on the top features only)
        """
        self.models = models
        self.label_names = label_names
        self.feature_indices = feature_indices

    def _prepare(self, X):
        """Restrict X to the feature subset the models were trained on."""
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X

    def predict(self, X):
        """Return a hard 0/1 prediction for every hazard, shape (n, 9)."""
        X = self._prepare(X)
        return np.column_stack([
            (m.predict(X) if m is not None else np.zeros(X.shape[0], dtype=int))
            for m in self.models])

    def predict_proba(self, X):
        """
        Return one probability array per hazard, matching the format that
        scikit-learn's MultiOutputClassifier uses: a list of nine arrays, each
        of shape (n_samples, 2).
        """
        X = self._prepare(X)
        output = []
        for model in self.models:
            if model is None:
                # A class with no positive training examples always predicts 0.
                output.append(np.column_stack([np.ones(X.shape[0]),
                                               np.zeros(X.shape[0])]))
            else:
                output.append(model.predict_proba(X))
        return output


# ===========================================================================
# 7a - RANDOM FOREST
# ===========================================================================
def train_random_forest(X_train, y_train):
    """
    Train the primary Random Forest: one MultiOutputClassifier wrapping a
    RandomForestClassifier, exactly as the proposal specifies.

    class_weight='balanced' makes the forest treat each class as if it had the
    same number of examples, which is how the imbalance is handled here.
    """
    print("\n[7a] Training Random Forest (MultiOutputClassifier) ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # DOCUMENTED DEVIATION FROM THE PROPOSAL.
    # The proposal specifies max_depth=None, meaning trees are grown until
    # every leaf is pure. A fully grown tree holds roughly two nodes per
    # training sample, and this project trains 200 trees for each of nine
    # hazard classes, so on this training set that would come to several
    # gigabytes more than the machine's 7.9 GB of RAM. The depth is therefore
    # capped and a minimum leaf size imposed. Both changes also reduce
    # overfitting, so the cost in accuracy is expected to be small - but this
    # is a hardware constraint, not a modelling choice, and is reported as such.
    min_leaf = choose_min_samples_leaf(X_train.shape[0])
    estimated_gb = (2 * X_train.shape[0] / min_leaf * 80 * 200
                    * len(GHS_LABEL_COLUMNS) / 1e9)

    rf_base = RandomForestClassifier(
        n_estimators=200,           # 200 trees per hazard, as specified
        max_depth=RF_MAX_DEPTH,     # capped for memory (proposal says None)
        min_samples_leaf=min_leaf,  # sized to fit the memory budget
        class_weight="balanced",    # counteract the class imbalance
        random_state=RANDOM_SEED,   # Rule 5 reproducibility
        n_jobs=N_JOBS,
    )
    rf_model = MultiOutputClassifier(rf_base, n_jobs=1)  # n_jobs=1: the forest
                                                         # itself is already
                                                         # parallel, and nesting
                                                         # would exhaust RAM
    log_issue("7a", f"HARDWARE DEVIATION: max_depth capped at {RF_MAX_DEPTH} "
                    f"and min_samples_leaf set to {min_leaf}, instead of the "
                    f"proposal's max_depth=None with unlimited leaf splitting. "
                    f"Fully grown trees for nine classes would need an estimated "
                    f"{2 * X_train.shape[0] * 80 * 200 * 9 / 1e9:.1f} GB, far "
                    f"beyond this machine's 7.9 GB. The chosen settings bring "
                    f"that to about {estimated_gb:.1f} GB. Both changes also "
                    f"reduce overfitting, so the accuracy cost should be small.")
    print(f"      min_samples_leaf={min_leaf} chosen from the memory budget "
          f"(estimated forest size {estimated_gb:.1f} GB)")

    started = time.time()
    try:
        rf_model.fit(X_train, y_train)
    except MemoryError:
        log_issue("7a", f"Random Forest still ran out of memory at depth "
                        f"{RF_MAX_DEPTH}. FALLBACK: retrying with 100 trees "
                        f"capped at depth 12 and a minimum leaf size of 5, "
                        f"which cuts memory roughly eightfold.")
        gc.collect()
        rf_base = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=N_JOBS)
        rf_model = MultiOutputClassifier(rf_base, n_jobs=1)
        rf_model.fit(X_train, y_train)

    elapsed = time.time() - started
    TIMINGS["RandomForest"] = round(elapsed, 1)
    print(f"      Trained in {elapsed / 60:.2f} minutes "
          f"({len(GHS_LABEL_COLUMNS)} forests x "
          f"{rf_model.estimators_[0].n_estimators} trees)")
    return rf_model


def train_random_forest_smote_ablation(balanced_dir, feature_count):
    """
    ABLATION: the same Random Forest, but trained per class on the
    SMOTE-balanced sets from Step 6 and with no class weighting.

    Running this alongside the primary model measures whether synthetic
    oversampling actually beats simple class weighting on this dataset - a
    question the proposal leaves open.

    The balanced sets are loaded from disk one class at a time and released
    immediately afterwards; holding all nine would exhaust this machine's RAM.
    """
    print("\n[7a-ablation] Training Random Forest on the SMOTE-balanced sets ...")
    from sklearn.ensemble import RandomForestClassifier

    # Did SMOTE actually run? At full-dataset scale the oversampled matrices
    # exceed the memory budget for every class, so Step 6 falls back to class
    # weighting and the "balanced" sets are simply the original training data.
    # In that case this is not a SMOTE ablation at all - it is a
    # no-class-weighting ablation - and calling it SMOTE would misreport what
    # was measured.
    smote_actually_applied = False
    try:
        report = pd.read_csv(stamped("STEP6_smote_report.csv"))
        smote_actually_applied = bool(
            report["Method"].str.contains("SMOTE|ADASYN|RandomOverSampler",
                                          case=False, na=False).any())
    except Exception:
        pass
    if not smote_actually_applied:
        log_issue("7a-ablation", "Step 6 applied no oversampling to any class - "
                                 "the projected matrices exceeded the memory "
                                 "budget at this dataset size. This ablation "
                                 "therefore measures the effect of removing "
                                 "class weighting, NOT the effect of SMOTE. "
                                 "The SMOTE comparison is only available at the "
                                 "40,000-compound scale, where it was measured.")

    models = []
    started = time.time()
    for column in GHS_LABEL_COLUMNS:
        X_balanced = np.load(os.path.join(balanced_dir, f"X_{column}.npy"))
        y_balanced = np.load(os.path.join(balanced_dir, f"y_{column}.npy"))
        if y_balanced.sum() == 0:
            log_issue("7a-ablation", f"{column}: no positives - model skipped.")
            models.append(None)
            continue
        # The oversampled sets are larger than the original training set, so
        # the leaf size is recomputed for each one. The budget is divided by
        # nine here because these models are trained and saved one at a time.
        min_leaf = choose_min_samples_leaf(
            len(y_balanced), n_classes=1,
            budget_bytes=RF_MEMORY_BUDGET_BYTES / len(GHS_LABEL_COLUMNS))
        model = RandomForestClassifier(
            n_estimators=200, max_depth=RF_MAX_DEPTH, min_samples_leaf=min_leaf,
            class_weight=None,          # deliberately absent: SMOTE already
                                        # balanced the data
            random_state=RANDOM_SEED, n_jobs=N_JOBS)
        model.fit(X_balanced, y_balanced)
        models.append(model)
        print(f"      {column:<22} trained on {len(y_balanced):,} rows "
              f"({int(y_balanced.sum()):,} positive)")
        del X_balanced, y_balanced      # free before loading the next class
        gc.collect()

    elapsed = time.time() - started

    # The ablation's NAME must describe what was actually done, not what was
    # intended. When SMOTE is skipped - which happens at full dataset size,
    # because the oversampled matrices exceed the memory budget - these models
    # are trained on unmodified data with class weighting switched off, so the
    # ablation measures the effect of REMOVING CLASS WEIGHTING. Calling it a
    # SMOTE ablation in that case would misdescribe the experiment in every
    # results table and supplementary file, which a reader would reasonably
    # read as misrepresentation rather than a naming slip.
    #
    # The name and the filename are therefore derived here from what happened,
    # and written to a metadata file that every downstream step reads instead
    # of hard-coding a label of its own.
    if smote_actually_applied:
        name = "RandomForest_SMOTE"
        filename = "STEP7_rf_smote_ablation.pkl"
        measures = ("the effect of SMOTE oversampling relative to class "
                    "weighting")
    else:
        name = "RandomForest_NoClassWeight"
        filename = "STEP7_rf_noclassweight_ablation.pkl"
        measures = ("the effect of REMOVING class weighting. SMOTE was "
                    "requested but skipped for every class because the "
                    "oversampled matrices exceeded the memory budget at this "
                    "dataset size, so no synthetic examples were generated")

    TIMINGS[name + "_ablation"] = round(elapsed, 1)
    metadata = {
        "ablation_model_name": name,
        "ablation_model_file": filename,
        "smote_actually_applied": bool(smote_actually_applied),
        "what_this_ablation_measures": measures,
        "primary_randomforest_uses": "class_weight='balanced'",
        "ablation_randomforest_uses": "class_weight=None",
    }
    with open(os.path.join(DIR_MODELS, "STEP7_ablation_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"      Trained in {elapsed / 60:.2f} minutes")
    print(f"      Ablation named : {name}")
    print(f"      It measures    : {measures}")
    return PerLabelClassifier(models, GHS_LABEL_COLUMNS), name, filename


# ===========================================================================
# 7b - XGBOOST
# ===========================================================================
def train_xgboost(X_train, y_train, class_weights):
    """
    Train nine separate XGBoost classifiers, one per hazard.

    XGBoost does not handle multi-output problems naturally, so the proposal
    correctly asks for one model per label. `scale_pos_weight` is set to the
    number of negatives divided by the number of positives, which tells the
    booster how much more a positive example matters.

    FALLBACK: LightGBM, if XGBoost cannot be trained.
    """
    print("\n[7b] Training XGBoost - one model per hazard class ...")
    models = []
    started = time.time()
    used_lightgbm = False

    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_class = y_train[:, class_index].astype(int)
        n_positive = int(y_class.sum())
        if n_positive == 0:
            log_issue("7b", f"{column}: no positive training examples - "
                            f"model skipped.")
            models.append(None)
            continue

        scale_pos_weight = class_weights.get(column, 1.0)

        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,   # the imbalance correction
                random_state=RANDOM_SEED,            # Rule 5
                n_jobs=N_JOBS,
                eval_metric="aucpr",   # area under precision-recall: the right
                                       # metric to optimise when positives are rare
                tree_method="hist",    # histogram splitting: much lower memory
                verbosity=0,
            )
            model.fit(X_train, y_class)
        except Exception as exc:
            log_issue("7b", f"{column}: XGBoost failed ({type(exc).__name__}: "
                            f"{exc}). FALLBACK: LightGBM.")
            try:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    is_unbalance=True,      # LightGBM's own imbalance handling
                    random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=-1)
                model.fit(X_train, y_class)
                used_lightgbm = True
            except Exception as exc2:
                log_issue("7b", f"{column}: LightGBM failed too ({exc2}). "
                                f"No model for this class.")
                models.append(None)
                continue

        models.append(model)
        print(f"      {column:<22} n+={n_positive:>7,}  "
              f"scale_pos_weight={scale_pos_weight:>8.2f}")
        gc.collect()

    elapsed = time.time() - started
    TIMINGS["XGBoost"] = round(elapsed, 1)
    print(f"      Trained in {elapsed / 60:.2f} minutes"
          f"{' (LightGBM fallback used for at least one class)' if used_lightgbm else ''}")
    return PerLabelClassifier(models, GHS_LABEL_COLUMNS)


# ===========================================================================
# 7c - SUPPORT VECTOR MACHINE
# ===========================================================================
def train_svm(X_train, y_train, rf_model):
    """
    Train the SVM, applying the fallbacks the hardware makes necessary.

    An RBF support vector machine needs to hold an n x n kernel matrix, so its
    cost grows with the SQUARE of the number of training compounds. On the
    full training set that matrix would be far larger than this machine's
    memory, and training would take days.

    Two reductions are therefore applied, both documented:
      1. the proposal's own step 7c fallback - keep only the 100 most important
         features, ranked by the Random Forest trained in 7a;
      2. an additional stratified subsample of the training compounds.

    Feature scaling is applied inside the pipeline, before the SVM, because an
    RBF kernel compares distances and would otherwise be dominated by
    descriptors that happen to have large numeric ranges (molecular weight
    runs to the hundreds, whereas fingerprint bits are 0 or 1).
    """
    print("\n[7c] Training Support Vector Machine ...")
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputClassifier

    n_train = X_train.shape[0]

    # ---- fallback 1: reduce to the top 100 features -----------------------
    importances = np.zeros(X_train.shape[1])
    for estimator in rf_model.estimators_:
        importances += estimator.feature_importances_
    top_features = np.argsort(importances)[::-1][:SVM_TOP_N_FEATURES]
    top_features = np.sort(top_features)
    log_issue("7c", f"FALLBACK APPLIED (proposal step 7c): the SVM is trained on "
                    f"the top {SVM_TOP_N_FEATURES} features by Random Forest "
                    f"importance instead of all {X_train.shape[1]:,}, because "
                    f"full-feature RBF training would exceed 30 minutes on this "
                    f"2-core machine.")

    # ---- fallback 2: subsample the training compounds ---------------------
    if n_train > SVM_MAX_TRAINING_SAMPLES:
        log_issue("7c", f"ADDITIONAL HARDWARE FALLBACK: an RBF kernel matrix for "
                        f"{n_train:,} compounds would need about "
                        f"{n_train ** 2 * 8 / 1e9:.1f} GB, more than this machine "
                        f"has. The SVM is trained on a stratified subsample of "
                        f"{SVM_MAX_TRAINING_SAMPLES:,} compounds. This is a "
                        f"hardware limitation and is reported in the Methods "
                        f"section; the SVM's scores are therefore not strictly "
                        f"comparable with those of RF and XGBoost.")
        rng = np.random.RandomState(RANDOM_SEED)
        # Keep every compound carrying a rare hazard, then fill at random, so
        # the subsample still contains examples of all nine classes.
        rare_mask = y_train[:, [i for i, c in enumerate(GHS_LABEL_COLUMNS)
                                if y_train[:, i].sum() < 1000]].sum(axis=1) > 0 \
            if any(y_train[:, i].sum() < 1000 for i in range(y_train.shape[1])) \
            else np.zeros(n_train, dtype=bool)
        must_keep = np.where(rare_mask)[0]
        pool = np.where(~rare_mask)[0]
        n_fill = max(0, SVM_MAX_TRAINING_SAMPLES - len(must_keep))
        chosen = np.concatenate([
            must_keep[:SVM_MAX_TRAINING_SAMPLES],
            rng.choice(pool, size=min(n_fill, len(pool)), replace=False)])
        chosen = np.sort(chosen)
        X_svm, y_svm = X_train[chosen][:, top_features], y_train[chosen]
    else:
        X_svm, y_svm = X_train[:, top_features], y_train

    print(f"      SVM training data: {X_svm.shape[0]:,} compounds "
          f"x {X_svm.shape[1]} features")

    svm_base = SVC(
        kernel="rbf",
        probability=True,          # needed for ROC curves and SHAP in Step 10
        class_weight="balanced",   # the imbalance correction for the SVM
        random_state=RANDOM_SEED,
        cache_size=500,            # MB of kernel cache; larger is faster
    )
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),   # MUST come before the SVM
        ("svm", svm_base),
    ])
    svm_model = MultiOutputClassifier(svm_pipeline, n_jobs=1)

    started = time.time()
    svm_model.fit(X_svm, y_svm)
    elapsed = time.time() - started
    TIMINGS["SVM"] = round(elapsed, 1)
    print(f"      Trained in {elapsed / 60:.2f} minutes")

    # Wrap so that later steps can pass the full feature matrix and the
    # wrapper will silently select the same 100 columns.
    wrapped = PerLabelClassifier(
        [est for est in svm_model.estimators_], GHS_LABEL_COLUMNS,
        feature_indices=top_features)
    return wrapped, top_features, svm_model


# ===========================================================================
# MAIN
# ===========================================================================
def main(run_ablation=True):
    """Train all three models, plus the ablation, and save everything."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 7 - MODEL TRAINING")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy"))
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    X_train, y_train = X[train_idx], y[train_idx].astype(int)

    with open(stamped("STEP6_imbalance_config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    class_weights = config["class_weights"]

    # Remembered separately because X_train is freed before the ablation runs.
    n_train_compounds, n_features = X_train.shape
    print(f"Training set: {n_train_compounds:,} compounds x "
          f"{n_features:,} features")
    print(f"Memory held by the training matrix: "
          f"{X_train.nbytes / 1e6:.0f} MB (float32)")

    # ---- 7a ---------------------------------------------------------------
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, os.path.join(DIR_MODELS, "STEP7_rf_model.pkl"),
                compress=3)
    print(f"      Saved: {os.path.join(DIR_MODELS, 'STEP7_rf_model.pkl')}")
    gc.collect()

    # ---- 7b ---------------------------------------------------------------
    xgb_models = train_xgboost(X_train, y_train, class_weights)
    joblib.dump(xgb_models, os.path.join(DIR_MODELS, "STEP7_xgb_models.pkl"),
                compress=3)
    print(f"      Saved: {os.path.join(DIR_MODELS, 'STEP7_xgb_models.pkl')}")
    gc.collect()

    # ---- 7c ---------------------------------------------------------------
    svm_model, svm_features, svm_raw = train_svm(X_train, y_train, rf_model)
    joblib.dump(svm_model, os.path.join(DIR_MODELS, "STEP7_svm_model.pkl"),
                compress=3)
    np.save(os.path.join(DIR_MODELS, "STEP7_svm_feature_indices.npy"), svm_features)
    print(f"      Saved: {os.path.join(DIR_MODELS, 'STEP7_svm_model.pkl')}")
    del svm_raw
    gc.collect()

    # ---- 7a ablation ------------------------------------------------------
    if run_ablation:
        try:
            balanced_dir = os.path.join(DIR_FEATURES, "STEP6_balanced")
            # Free the primary training matrix first - the ablation loads its
            # own, larger, oversampled copies.
            del X_train
            gc.collect()
            rf_ablation, ablation_name, ablation_file = \
                train_random_forest_smote_ablation(balanced_dir, None)
            joblib.dump(rf_ablation, os.path.join(DIR_MODELS, ablation_file),
                        compress=3)
            print(f"      Saved: {os.path.join(DIR_MODELS, ablation_file)}")
            # Remove a stale file left by a previous run under the other name,
            # so that no downstream step can pick up the wrong one.
            for other in ("STEP7_rf_smote_ablation.pkl",
                          "STEP7_rf_noclassweight_ablation.pkl"):
                path = os.path.join(DIR_MODELS, other)
                if other != ablation_file and os.path.exists(path):
                    os.remove(path)
                    print(f"      Removed stale {other}")
            del rf_ablation
            gc.collect()
        except Exception as exc:
            log_issue("7a-ablation", f"SMOTE ablation could not be trained "
                                     f"({type(exc).__name__}: {exc}). The primary "
                                     f"models are unaffected.")

    # ---- report -----------------------------------------------------------
    total_elapsed = time.time() - total_start
    TIMINGS["TOTAL"] = round(total_elapsed, 1)

    with open(stamped("STEP7_training_times.json"), "w", encoding="utf-8") as fh:
        json.dump({"seconds": TIMINGS,
                   "n_train_compounds": int(n_train_compounds),
                   "n_features": int(n_features),
                   "svm_training_samples": int(min(SVM_MAX_TRAINING_SAMPLES,
                                                   n_train_compounds)),
                   "svm_n_features": SVM_TOP_N_FEATURES,
                   "random_seed": RANDOM_SEED}, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP7_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 7 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 7 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Trained a Random Forest, nine XGBoost models and an")
    print("                SVM on the scaffold-split training set, plus a")
    print("                SMOTE ablation of the Random Forest.")
    print("\nTRAINING TIME PER ALGORITHM")
    print("-" * 60)
    for name, seconds in TIMINGS.items():
        print(f"   {name:<32}{seconds / 60:>8.2f} minutes")
    print("-" * 60)
    print(f"OUTPUT FILES  : {os.path.join(DIR_MODELS, 'STEP7_rf_model.pkl')}")
    print(f"                {os.path.join(DIR_MODELS, 'STEP7_xgb_models.pkl')}")
    print(f"                {os.path.join(DIR_MODELS, 'STEP7_svm_model.pkl')}")
    print(f"                {stamped('STEP7_training_times.json')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print("=" * 78)


if __name__ == "__main__":
    main()
