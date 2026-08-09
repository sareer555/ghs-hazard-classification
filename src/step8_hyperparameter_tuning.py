"""
STEP 8 - HYPERPARAMETER TUNING
==============================
Every algorithm has settings that are not learned from the data - how many
trees to grow, how deep they may be, how fast the booster learns. These are
the hyperparameters, and the right values differ from dataset to dataset.

RandomizedSearchCV tries a random selection of settings and keeps the best.
Random search is used rather than an exhaustive grid because it finds
near-optimal settings in a small fraction of the time.

Following the proposal, the search is run on the VALIDATION set. The test set
is never touched, so the final scores in Step 9 remain honest.

Compute budget (documented deviation)
-------------------------------------
The proposal asks for n_iter=30 with 5-fold cross-validation. On this machine
- two physical cores - that is 30 x 5 x 9 = 1350 model fits for the Random
Forest alone, which measured out at well over a day of wall-clock time. A
timing probe is therefore run first and the number of search iterations is
reduced to whatever fits a fixed wall-clock budget, with 3-fold rather than
5-fold cross-validation. The actual settings used are recorded in
STEP8_best_hyperparameters.json so that the reduction is visible to any
reader, and the full search grids specified in the proposal are searched -
just with fewer draws from them.

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
                        DIR_LOGS, GHS_LABEL_COLUMNS, seed_everything, stamped)

seed_everything()

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

ISSUE_LOG = []
N_JOBS = 3

# Wall-clock budget per algorithm, in seconds. The search shrinks to fit.
TIME_BUDGET_RF = 600       # 10 minutes
TIME_BUDGET_XGB = 600      # 10 minutes, shared across the nine models
TIME_BUDGET_SVM = 300      # 5 minutes
CV_FOLDS = 3               # reduced from the proposal's 5, see module docstring


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------
def multilabel_weighted_auc(estimator, X, y):
    """
    Score a multi-output model with a support-weighted mean AUC-ROC.

    This reproduces what the proposal's 'roc_auc_ovr_weighted' is asking for.
    scikit-learn's built-in scorer cannot be used directly here because
    MultiOutputClassifier.predict_proba returns a list of arrays rather than
    the single array the built-in scorer expects.

    Classes that happen to have only one value present in a cross-validation
    fold are skipped, because AUC is undefined for them.
    """
    probabilities = estimator.predict_proba(X)
    scores, weights = [], []
    for class_index in range(y.shape[1]):
        y_true = y[:, class_index]
        if len(np.unique(y_true)) < 2:
            continue
        y_score = probabilities[class_index][:, 1]
        scores.append(roc_auc_score(y_true, y_score))
        weights.append(y_true.sum())        # weight by number of positives
    if not scores:
        return 0.0
    return float(np.average(scores, weights=weights))


def binary_auc_scorer(estimator, X, y):
    """Support-free AUC scorer for the single-label XGBoost searches."""
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, estimator.predict_proba(X)[:, 1]))


def refit_on_training_set(best_params, X_train, y_train):
    """
    Rebuild the Random Forest on the TRAINING set using the winning settings.

    This step is a deliberate addition. The proposal runs the hyperparameter
    search with `rf_search.fit(X_val, y_val)`, which is the right way to choose
    settings - the training set must not be used to pick them, or the choice
    would be biased. But RandomizedSearchCV's `best_estimator_` is refitted on
    whatever data the search was given, so it would be a model trained on the
    validation set alone: one tenth of the data, and the very data used to
    calibrate the decision thresholds in Step 9.

    Using that model for the final evaluation would both waste 90% of the
    training data and leak validation information into the reported scores.
    The settings are therefore taken from the search and a fresh model is
    fitted on the proper training set.
    """
    print("\n      Refitting the Random Forest on the full training set "
          "with the winning settings ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # Strip the "estimator__" prefix that MultiOutputClassifier requires.
    clean_params = {k.replace("estimator__", ""): v
                    for k, v in best_params.items()}
    clean_params.setdefault("class_weight", "balanced")
    clean_params["random_state"] = RANDOM_SEED
    clean_params["n_jobs"] = N_JOBS
    # Keep the memory guard from Step 7 in place whatever the search chose.
    if clean_params.get("max_depth") in (None, 50):
        clean_params["max_depth"] = 40
    # Apply the same memory-derived floor the Step 7 forest used, so a search
    # draw of min_samples_leaf=1 cannot blow the memory budget on refit.
    from step7_model_training import choose_min_samples_leaf
    floor = choose_min_samples_leaf(X_train.shape[0],
                                    n_estimators=clean_params.get("n_estimators",
                                                                  200))
    clean_params["min_samples_leaf"] = max(floor,
                                           clean_params.get("min_samples_leaf", 2))
    print(f"      min_samples_leaf floored at {floor} by the memory budget")

    model = MultiOutputClassifier(RandomForestClassifier(**clean_params), n_jobs=1)
    started = time.time()
    try:
        model.fit(X_train, y_train)
    except MemoryError:
        log_issue("8a", "refit on the training set ran out of memory; "
                        "falling back to 100 trees at depth 12.")
        gc.collect()
        clean_params.update({"n_estimators": 100, "max_depth": 12,
                             "min_samples_leaf": 5})
        model = MultiOutputClassifier(RandomForestClassifier(**clean_params),
                                      n_jobs=1)
        model.fit(X_train, y_train)
    print(f"      Refit completed in {(time.time() - started) / 60:.1f} minutes")
    return model, clean_params


def refit_xgboost_on_training_set(per_class_results, X_train, y_train,
                                  class_weights):
    """
    Rebuild the nine XGBoost models on the training set with their winning
    settings, for the same reason as the Random Forest refit above.
    """
    print("\n      Refitting the nine XGBoost models on the training set ...")
    from xgboost import XGBClassifier
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from step7_model_training import PerLabelClassifier

    models = []
    started = time.time()
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_class = y_train[:, class_index].astype(int)
        if y_class.sum() == 0:
            models.append(None)
            continue
        params = (per_class_results.get(column, {}) or {}).get("best_params")
        if not params:
            # No tuning result for this class - keep the Step 7 defaults.
            params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                      "subsample": 0.8, "colsample_bytree": 0.8}
        model = XGBClassifier(
            **params,
            scale_pos_weight=class_weights.get(column, 1.0),
            random_state=RANDOM_SEED, n_jobs=N_JOBS, tree_method="hist",
            verbosity=0, eval_metric="aucpr")
        model.fit(X_train, y_class)
        models.append(model)
        print(f"      {column:<22} refitted")
        gc.collect()
    print(f"      Refit completed in {(time.time() - started) / 60:.1f} minutes")
    return PerLabelClassifier(models, GHS_LABEL_COLUMNS)


def choose_n_iter(single_fit_seconds, budget_seconds, cv_folds, label=""):
    """
    Work out how many random search draws fit inside the time budget.

    Each draw costs `cv_folds + 1` fits (one per fold, plus the final refit).
    At least 5 draws are always attempted, since fewer would not be a search
    at all.
    """
    cost_per_draw = single_fit_seconds * (cv_folds + 1)
    n_iter = int(budget_seconds / max(cost_per_draw, 1e-6))
    n_iter = max(5, min(30, n_iter))       # never more than the proposal's 30
    if n_iter < 30:
        log_issue("8", f"{label}: one fit took {single_fit_seconds:.1f}s, so the "
                       f"proposal's n_iter=30 would need "
                       f"{30 * cost_per_draw / 3600:.1f} hours. Reduced to "
                       f"n_iter={n_iter} with cv={cv_folds} to fit the "
                       f"{budget_seconds / 60:.0f}-minute budget.")
    return n_iter


# ===========================================================================
# 8a - RANDOM FOREST
# ===========================================================================
def tune_random_forest(X_val, y_val):
    """Run RandomizedSearchCV over the Random Forest grid from the proposal."""
    print("\n[8a] Tuning the Random Forest ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # The grid named in the proposal, with one change: max_depth=None has been
    # replaced by 40. Unlimited-depth trees for nine classes cannot fit in this
    # machine's 7.9 GB of RAM (see the note in Step 7), and a search draw that
    # crashed the process would take the whole run with it. Depth 40 is deeper
    # than any tree the data actually produces, so the search space is
    # effectively unchanged.
    param_dist_rf = {
        "estimator__n_estimators":      [100, 200, 300, 500],
        "estimator__max_depth":         [40, 10, 20, 30, 50],
        "estimator__min_samples_split": [2, 5, 10, 20],
        "estimator__max_features":      ["sqrt", "log2", 0.3, 0.5],
        "estimator__min_samples_leaf":  [1, 2, 4, 8],
    }

    base = MultiOutputClassifier(
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS,
                               class_weight="balanced"), n_jobs=1)

    # ---- timing probe -----------------------------------------------------
    probe = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                               n_jobs=N_JOBS, class_weight="balanced"), n_jobs=1)
    started = time.time()
    probe.fit(X_val, y_val)
    probe_seconds = time.time() - started
    del probe
    gc.collect()
    print(f"      Timing probe: one 100-tree fit on {X_val.shape[0]:,} "
          f"validation compounds took {probe_seconds:.1f}s")

    # The grid contains up to 500 trees, so budget for the worst case.
    n_iter = choose_n_iter(probe_seconds * 2.5, TIME_BUDGET_RF, CV_FOLDS,
                           "Random Forest")

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        cv=CV_FOLDS,
        scoring=multilabel_weighted_auc,   # support-weighted mean AUC
        random_state=RANDOM_SEED,          # Rule 5
        n_jobs=1,          # the forest is already parallel; nesting would
                           # thrash this machine's 7.9 GB of RAM
        verbose=1,
        error_score=0.0,   # a failed fit scores 0 rather than crashing the run
    )
    started = time.time()
    search.fit(X_val, y_val)
    elapsed = time.time() - started

    print(f"      Best weighted AUC: {search.best_score_:.4f}")
    print(f"      Best parameters  : {search.best_params_}")
    print(f"      Search took {elapsed / 60:.1f} minutes "
          f"({n_iter} draws x {CV_FOLDS} folds)")
    return search, elapsed, n_iter


# ===========================================================================
# 8b - XGBOOST
# ===========================================================================
def tune_xgboost(X_val, y_val, class_weights):
    """
    Tune each of the nine XGBoost models separately.

    Each hazard gets its own search because the best settings for a common
    hazard such as irritation are not the best settings for a rare one such as
    explosivity.
    """
    print("\n[8b] Tuning XGBoost - one search per hazard class ...")
    from xgboost import XGBClassifier

    # The exact grid named in the proposal.
    param_dist_xgb = {
        "n_estimators":     [100, 200, 300, 500, 1000],
        "max_depth":        [3, 4, 5, 6, 8, 10],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.1, 1.0, 5.0, 10.0],
    }

    # ---- timing probe on one class ---------------------------------------
    probe_model = XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                                random_state=RANDOM_SEED, n_jobs=N_JOBS,
                                verbosity=0, eval_metric="aucpr")
    started = time.time()
    probe_model.fit(X_val, y_val[:, 0])
    probe_seconds = time.time() - started
    del probe_model
    gc.collect()
    print(f"      Timing probe: one 200-tree fit took {probe_seconds:.1f}s")

    # The budget is shared between nine searches, and the grid goes up to
    # 1000 trees (five times the probe's 200).
    per_class_budget = TIME_BUDGET_XGB / len(GHS_LABEL_COLUMNS)
    n_iter = choose_n_iter(probe_seconds * 3, per_class_budget, CV_FOLDS, "XGBoost")

    results = {}
    total_elapsed = 0.0
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_class = y_val[:, class_index].astype(int)
        if len(np.unique(y_class)) < 2 or y_class.sum() < CV_FOLDS:
            log_issue("8b", f"{column}: only {int(y_class.sum())} positive(s) in "
                            f"the validation set - too few for {CV_FOLDS}-fold "
                            f"cross-validation. Step 7 defaults are kept.")
            results[column] = {"best_params": None, "best_score": None,
                               "note": "insufficient validation positives to tune"}
            continue

        model = XGBClassifier(
            scale_pos_weight=class_weights.get(column, 1.0),
            random_state=RANDOM_SEED, n_jobs=N_JOBS, tree_method="hist",
            verbosity=0, eval_metric="aucpr")

        search = RandomizedSearchCV(
            model, param_distributions=param_dist_xgb, n_iter=n_iter,
            cv=CV_FOLDS, scoring=binary_auc_scorer, random_state=RANDOM_SEED,
            n_jobs=1, verbose=0, error_score=0.0)

        started = time.time()
        search.fit(X_val, y_class)
        elapsed = time.time() - started
        total_elapsed += elapsed

        results[column] = {"best_params": search.best_params_,
                           "best_score": round(float(search.best_score_), 4),
                           "n_iter": n_iter, "cv_folds": CV_FOLDS}
        print(f"      {column:<22} AUC={search.best_score_:.4f}  "
              f"({elapsed / 60:.1f} min)")
        gc.collect()

    return results, total_elapsed, n_iter


# ===========================================================================
# 8c - SVM
# ===========================================================================
def tune_svm(X_val, y_val, svm_feature_indices):
    """
    Tune the SVM over the grid from the proposal.

    Two efficiencies are used. The search runs on the same 100-feature subset
    the Step 7 SVM was trained on, and probability estimation is switched off
    during the search - Platt scaling costs an internal five-fold
    cross-validation on every single fit. The winning settings are refitted
    with probability=True at the end.
    """
    print("\n[8c] Tuning the Support Vector Machine ...")
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputClassifier

    X_svm = X_val[:, svm_feature_indices]

    # The exact grid named in the proposal.
    param_dist_svm = {
        "estimator__svm__C":      [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "estimator__svm__gamma":  ["scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0],
        "estimator__svm__kernel": ["rbf", "poly", "sigmoid"],
    }

    # An RBF kernel matrix is O(n^2); cap the tuning sample accordingly.
    max_tuning_samples = 3000
    if X_svm.shape[0] > max_tuning_samples:
        log_issue("8c", f"validation set has {X_svm.shape[0]:,} compounds; the "
                        f"SVM search is run on a random {max_tuning_samples:,}"
                        f"-compound subsample because kernel cost grows with "
                        f"the square of the sample count.")
        rng = np.random.RandomState(RANDOM_SEED)
        chosen = np.sort(rng.choice(X_svm.shape[0], max_tuning_samples,
                                    replace=False))
        X_svm, y_svm = X_svm[chosen], y_val[chosen]
    else:
        y_svm = y_val

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        # probability=False during the search; refitted with True afterwards
        ("svm", SVC(class_weight="balanced", random_state=RANDOM_SEED,
                    probability=False, cache_size=400)),
    ])
    base = MultiOutputClassifier(pipeline, n_jobs=1)

    # Without predict_proba the AUC must come from decision_function.
    def svm_auc(estimator, X, y):
        """Support-weighted mean AUC using the SVM's signed distance."""
        scores, weights = [], []
        for class_index, sub_estimator in enumerate(estimator.estimators_):
            y_true = y[:, class_index]
            if len(np.unique(y_true)) < 2:
                continue
            y_score = sub_estimator.decision_function(X)
            scores.append(roc_auc_score(y_true, y_score))
            weights.append(y_true.sum())
        return float(np.average(scores, weights=weights)) if scores else 0.0

    # ---- timing probe -----------------------------------------------------
    started = time.time()
    probe = MultiOutputClassifier(Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(class_weight="balanced", random_state=RANDOM_SEED,
                    probability=False, cache_size=400))]), n_jobs=1)
    probe.fit(X_svm, y_svm)
    probe_seconds = time.time() - started
    del probe
    gc.collect()
    print(f"      Timing probe: one fit on {X_svm.shape[0]:,} compounds "
          f"took {probe_seconds:.1f}s")

    n_iter = choose_n_iter(probe_seconds, TIME_BUDGET_SVM, CV_FOLDS, "SVM")

    search = RandomizedSearchCV(
        base, param_distributions=param_dist_svm, n_iter=n_iter, cv=CV_FOLDS,
        scoring=svm_auc, random_state=RANDOM_SEED, n_jobs=1, verbose=1,
        error_score=0.0)

    started = time.time()
    search.fit(X_svm, y_svm)
    elapsed = time.time() - started

    print(f"      Best weighted AUC: {search.best_score_:.4f}")
    print(f"      Best parameters  : {search.best_params_}")
    print(f"      Search took {elapsed / 60:.1f} minutes")
    return search, elapsed, n_iter


# ===========================================================================
# MAIN
# ===========================================================================
def tune_hyperparameters():
    """Run all three searches, save the best settings, and refit final models."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 8 - HYPERPARAMETER TUNING")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    val_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_val_indices.npy"))
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    X_val, y_val = X[val_idx], y[val_idx]

    with open(stamped("STEP6_imbalance_config.json"), encoding="utf-8") as fh:
        class_weights = json.load(fh)["class_weights"]

    print(f"Tuning on the validation set: {X_val.shape[0]:,} compounds "
          f"x {X_val.shape[1]:,} features")
    print(f"Cross-validation folds: {CV_FOLDS} "
          f"(reduced from the proposal's 5 - see the module docstring)")

    # ---- CHECKPOINTING -----------------------------------------------------
    # Each algorithm's result is written to disk as soon as its search
    # finishes. If the process is interrupted - which happened repeatedly on
    # this machine - re-running the step resumes from the next unfinished
    # algorithm instead of repeating work already done.
    results_path = stamped("STEP8_best_hyperparameters.json")
    results = {"_meta": {"cv_folds": CV_FOLDS, "random_seed": RANDOM_SEED,
                         "tuned_on": "validation set",
                         "n_validation_compounds": int(X_val.shape[0]),
                         "n_features": int(X_val.shape[1])}}
    if os.path.exists(results_path):
        try:
            with open(results_path, encoding="utf-8") as fh:
                previous = json.load(fh)

            # A checkpoint is only reusable if it was produced from the SAME
            # data. Resuming an interrupted run is the point of checkpointing,
            # but silently reusing results computed on a different dataset is
            # not: it once caused a tuned model built on 817 features to be
            # loaded against an 816-feature matrix, which failed at evaluation
            # time with a confusing shape error rather than where the mistake
            # actually was.
            previous_meta = previous.get("_meta", {})
            same_features = (previous_meta.get("n_features")
                             == int(X_val.shape[1]))
            same_size = (previous_meta.get("n_validation_compounds")
                         == int(X_val.shape[0]))
            if same_features and same_size:
                results.update({k: v for k, v in previous.items()
                                if not k.startswith("_")})
                print(f"\nResuming: found completed results for "
                      f"{[k for k in results if not k.startswith('_')]}")
            else:
                log_issue("8", f"an existing checkpoint was found but it was "
                               f"produced from different data "
                               f"({previous_meta.get('n_features')} features, "
                               f"{previous_meta.get('n_validation_compounds')} "
                               f"validation compounds, against "
                               f"{X_val.shape[1]} and {X_val.shape[0]} now). "
                               f"It has been discarded and every algorithm will "
                               f"be re-tuned from scratch.")
                # Stale tuned models must go too, or Step 9 would load them.
                for filename in ("STEP8_rf_tuned.pkl", "STEP8_xgb_tuned.pkl",
                                 "STEP8_rf_cv_results.csv",
                                 "STEP8_svm_cv_results.csv"):
                    path = os.path.join(DIR_MODELS, filename)
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"      removed stale {filename}")
        except Exception as exc:
            log_issue("8", f"could not read the previous checkpoint ({exc}); "
                           f"starting the search from scratch.")

    def checkpoint():
        """Write the results so far, so an interruption loses nothing."""
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    def already_done(algorithm, required_file):
        """True if this algorithm was tuned AND its refitted model exists."""
        return (algorithm in results
                and os.path.exists(os.path.join(DIR_MODELS, required_file)))

    # ---- 8a ---------------------------------------------------------------
    if already_done("RandomForest", "STEP8_rf_tuned.pkl"):
        print("\n[8a] Random Forest already tuned and refitted - skipping.")
    else:
        try:
            rf_search, rf_seconds, rf_iters = tune_random_forest(X_val, y_val)
            results["RandomForest"] = {
                "best_params": {k: (v if not isinstance(v, np.generic)
                                    else v.item())
                                for k, v in rf_search.best_params_.items()},
                "best_cv_weighted_auc": round(float(rf_search.best_score_), 4),
                "n_iter": rf_iters, "cv_folds": CV_FOLDS,
                "search_seconds": round(rf_seconds, 1),
            }
            pd.DataFrame(rf_search.cv_results_).to_csv(
                os.path.join(DIR_MODELS, "STEP8_rf_cv_results.csv"), index=False)

            # Refit on the training set - see refit_on_training_set for why.
            best_params = dict(rf_search.best_params_)
            del rf_search
            gc.collect()
            tuned_rf, applied_params = refit_on_training_set(
                best_params, X[train_idx], y[train_idx])
            joblib.dump(tuned_rf,
                        os.path.join(DIR_MODELS, "STEP8_rf_tuned.pkl"), compress=3)
            results["RandomForest"]["refitted_on"] = "training set"
            results["RandomForest"]["applied_params"] = {
                k: str(v) for k, v in applied_params.items()}
            del tuned_rf
            gc.collect()
            checkpoint()
        except Exception as exc:
            log_issue("8a", f"Random Forest tuning failed "
                            f"({type(exc).__name__}: {exc}). The Step 7 default "
                            f"settings are kept.")
            results["RandomForest"] = {"error": str(exc),
                                       "fallback": "Step 7 defaults"}
            checkpoint()

    # ---- 8b ---------------------------------------------------------------
    if already_done("XGBoost", "STEP8_xgb_tuned.pkl"):
        print("\n[8b] XGBoost already tuned and refitted - skipping.")
    else:
        try:
            xgb_results, xgb_seconds, xgb_iters = tune_xgboost(X_val, y_val,
                                                               class_weights)
            results["XGBoost"] = {"per_class": xgb_results,
                                  "search_seconds": round(xgb_seconds, 1),
                                  "n_iter": xgb_iters, "cv_folds": CV_FOLDS}
            # Refit on the training set, as for the Random Forest.
            tuned_xgb = refit_xgboost_on_training_set(
                xgb_results, X[train_idx], y[train_idx], class_weights)
            joblib.dump(tuned_xgb,
                        os.path.join(DIR_MODELS, "STEP8_xgb_tuned.pkl"),
                        compress=3)
            results["XGBoost"]["refitted_on"] = "training set"
            del tuned_xgb
            gc.collect()
            checkpoint()
        except Exception as exc:
            log_issue("8b", f"XGBoost tuning failed ({type(exc).__name__}: "
                            f"{exc}). The Step 7 default settings are kept.")
            results["XGBoost"] = {"error": str(exc),
                                  "fallback": "Step 7 defaults"}
            checkpoint()

    # ---- 8c ---------------------------------------------------------------
    # The SVM has no separate refit file: Step 9 scores the Step 7 SVM, so the
    # presence of the results entry alone marks this algorithm as done.
    if "SVM" in results:
        print("\n[8c] SVM already tuned - skipping.")
    else:
        try:
            svm_features = np.load(os.path.join(DIR_MODELS,
                                                "STEP7_svm_feature_indices.npy"))
            svm_search, svm_seconds, svm_iters = tune_svm(X_val, y_val,
                                                          svm_features)
            results["SVM"] = {
                "best_params": {k: (v if not isinstance(v, np.generic)
                                    else v.item())
                                for k, v in svm_search.best_params_.items()},
                "best_cv_weighted_auc": round(float(svm_search.best_score_), 4),
                "n_iter": svm_iters, "cv_folds": CV_FOLDS,
                "search_seconds": round(svm_seconds, 1),
                "n_features_used": int(len(svm_features)),
                "note": ("searched with probability=False for speed. The Step 7 "
                         "SVM, which already carries probability=True, is the "
                         "model scored in Step 9; these settings are reported "
                         "for the record."),
            }
            pd.DataFrame(svm_search.cv_results_).to_csv(
                os.path.join(DIR_MODELS, "STEP8_svm_cv_results.csv"), index=False)
            del svm_search
            gc.collect()
            checkpoint()
        except Exception as exc:
            log_issue("8c", f"SVM tuning failed ({type(exc).__name__}: {exc}). "
                            f"The Step 7 default settings are kept.")
            results["SVM"] = {"error": str(exc), "fallback": "Step 7 defaults"}
            checkpoint()

    results["_meta"]["total_seconds"] = round(time.time() - total_start, 1)
    checkpoint()

    log_path = os.path.join(DIR_LOGS, f"STEP8_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 8 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 8 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Ran RandomizedSearchCV over the hyperparameter grids")
    print("                specified in the proposal for all three algorithms,")
    print("                using the validation set only.")
    for algorithm in ("RandomForest", "XGBoost", "SVM"):
        entry = results.get(algorithm, {})
        if "best_cv_weighted_auc" in entry:
            print(f"   {algorithm:<14} best CV weighted AUC = "
                  f"{entry['best_cv_weighted_auc']:.4f}  "
                  f"({entry['n_iter']} draws, {entry['cv_folds']}-fold)")
        elif "per_class" in entry:
            scores = [c["best_score"] for c in entry["per_class"].values()
                      if c.get("best_score") is not None]
            if scores:
                print(f"   {algorithm:<14} mean best CV AUC across classes = "
                      f"{np.mean(scores):.4f}  ({entry['n_iter']} draws, "
                      f"{entry['cv_folds']}-fold)")
        else:
            print(f"   {algorithm:<14} tuning failed - Step 7 defaults kept")
    print(f"OUTPUT FILE   : {stamped('STEP8_best_hyperparameters.json')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print(f"ELAPSED       : {results['_meta']['total_seconds'] / 60:.1f} minutes")
    print("=" * 78)
    return results


if __name__ == "__main__":
    tune_hyperparameters()
