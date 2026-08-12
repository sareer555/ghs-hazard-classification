"""
APPLICABILITY DOMAIN - WHERE CAN THIS MODEL BE TRUSTED?
=======================================================
A model trained on one region of chemical space does not stop producing
confident numbers when it is handed something from outside that region. It just
stops being right, silently. Reporting a mean AUC without saying where the
model applies invites a reader to use it everywhere.

This analysis came out of testing the deployed application on five compounds
chosen by hand: water, butane, atorvastatin, sodium cyanide and chlorine. Water
was returned as corrosive, acutely toxic and a serious health hazard. Water is
none of those things. That is not a rounding error, and it needed explaining.

The explanation is molecule size. Half the training compounds have around
fourteen heavy atoms and almost none have fewer than six, so the model has
barely seen the small-molecule regime. It has, however, learned something real
about it: small molecules genuinely are more hazardous on average. What it does
wrong is over-apply that, treating smallness as near-proof of toxicity - which
is exactly the error water triggers.

This script measures the effect: for each hazard class it compares how often the
model flags a compound with how often compounds of that size actually carry the
pictogram, inside and outside the domain.

Outputs
    EXTRA_applicability_domain.csv    per-class rates by size band
    EXTRA_applicability_domain.json   the figures the manuscript quotes

Author : Sareer Ahmad
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (DIR_FEATURES, DIR_SPLITS, DIR_MODELS,
                        GHS_LABEL_COLUMNS, GHS_TRUE_MEANING, stamped,
                        seed_everything)

warnings.filterwarnings("ignore")
seed_everything()

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Below this many heavy atoms the model is treated as out of its domain. Chosen
# from the training distribution rather than picked: see report_domain_bound().
DOMAIN_MIN_HEAVY_ATOMS = 6

# Size bands for the per-class table. The first two are the out-of-domain
# region, split so that the effect can be seen strengthening as molecules get
# smaller.
BANDS = [(1, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)]


def heavy_atom_counts(smiles):
    """Return the heavy-atom count for each SMILES, or -1 if it will not parse."""
    counts = np.empty(len(smiles), dtype=int)
    for i, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(text))
        counts[i] = molecule.GetNumHeavyAtoms() if molecule else -1
    return counts


def load_predictions():
    """
    Score the best model on the held-out test split.

    Returns the predicted probabilities, the true labels, the calibrated
    decision thresholds and the heavy-atom count per compound.
    """
    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    index = pd.read_csv(os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"))

    model = joblib.load(os.path.join(DIR_MODELS, "STEP8_xgb_tuned.pkl"))
    probabilities = np.column_stack([
        (p[:, 1] if p.shape[1] > 1 else p[:, 0])
        for p in model.predict_proba(X[test_idx])])

    with open(stamped("STEP9_calibrated_thresholds.json"), encoding="utf-8") as fh:
        thresholds = json.load(fh)["XGBoost"]

    return {
        "probabilities": probabilities,
        "y_test": y[test_idx],
        "thresholds": {c: thresholds[c]["threshold_f1"]
                       for c in GHS_LABEL_COLUMNS},
        "heavy_test": heavy_atom_counts(
            index.iloc[test_idx]["SMILES"].astype(str).values),
        "heavy_train": heavy_atom_counts(
            index.iloc[train_idx]["SMILES"].astype(str).values),
    }


def report_domain_bound(heavy_train):
    """
    Describe the training distribution and justify where the domain ends.

    The bound is not arbitrary: it is set where the training data runs out.
    """
    valid = heavy_train[heavy_train > 0]
    return {
        "n_training_compounds": int(valid.size),
        "median_heavy_atoms": float(np.median(valid)),
        "percentile_1": float(np.percentile(valid, 1)),
        "percentile_5": float(np.percentile(valid, 5)),
        "pct_below_domain": float(100 * (valid < DOMAIN_MIN_HEAVY_ATOMS).mean()),
        "domain_min_heavy_atoms": DOMAIN_MIN_HEAVY_ATOMS,
    }


def per_class_table(data):
    """
    Build the per-class comparison of flagged rate against true rate.

    The comparison is the point: a model that flags a class far more often than
    compounds of that size actually carry it is not detecting the hazard, it is
    guessing from size.
    """
    rows = []
    heavy = data["heavy_test"]
    for low, high in BANDS:
        mask = (heavy >= low) & (heavy <= high)
        if not mask.any():
            continue
        label = f"{low}-{high}" if high < 10_000 else f"{low}+"
        for i, column in enumerate(GHS_LABEL_COLUMNS):
            flagged = (data["probabilities"][mask, i]
                       > data["thresholds"][column]).mean()
            actual = data["y_test"][mask, i].mean()
            rows.append({
                "Size_Band_Heavy_Atoms": label,
                "In_Domain": low >= DOMAIN_MIN_HEAVY_ATOMS,
                "N_Compounds": int(mask.sum()),
                "GHS_Column": column,
                "Pictogram_Code": column.split("_")[0],
                "Meaning": GHS_TRUE_MEANING[column],
                "Pct_Flagged_By_Model": round(100 * flagged, 1),
                "Pct_Actually_Positive": round(100 * actual, 1),
                "Over_Prediction_Factor": (round(flagged / actual, 2)
                                           if actual > 0 else None),
                "Mean_Predicted_Probability": round(
                    float(data["probabilities"][mask, i].mean()), 3),
            })
    return pd.DataFrame(rows)


def main():
    """Measure the applicability domain and write the results."""
    print("=" * 78)
    print("APPLICABILITY DOMAIN ANALYSIS")
    print("=" * 78)

    data = load_predictions()
    bound = report_domain_bound(data["heavy_train"])

    print(f"\nTraining set molecule size")
    print(f"   median heavy atoms          : {bound['median_heavy_atoms']:.0f}")
    print(f"   1st percentile              : {bound['percentile_1']:.0f}")
    print(f"   below {DOMAIN_MIN_HEAVY_ATOMS} heavy atoms          : "
          f"{bound['pct_below_domain']:.2f}% of training compounds")

    table = per_class_table(data)
    table.to_csv(stamped("EXTRA_applicability_domain.csv"), index=False)

    # The out-of-domain summary the manuscript quotes.
    heavy = data["heavy_test"]
    out = (heavy > 0) & (heavy < DOMAIN_MIN_HEAVY_ATOMS)
    inside = heavy >= DOMAIN_MIN_HEAVY_ATOMS

    worst = []
    print(f"\nOut of domain (< {DOMAIN_MIN_HEAVY_ATOMS} heavy atoms, "
          f"n = {int(out.sum())}) versus in domain (n = {int(inside.sum())})")
    print(f"   {'class':<24}{'flagged':>9}{'actual':>9}{'factor':>9}"
          f"{'   in-domain flagged/actual'}")
    for i, column in enumerate(GHS_LABEL_COLUMNS):
        threshold = data["thresholds"][column]
        f_out = 100 * (data["probabilities"][out, i] > threshold).mean()
        a_out = 100 * data["y_test"][out, i].mean()
        f_in = 100 * (data["probabilities"][inside, i] > threshold).mean()
        a_in = 100 * data["y_test"][inside, i].mean()
        factor = f_out / a_out if a_out > 0 else float("inf")
        print(f"   {column:<24}{f_out:8.1f}%{a_out:8.1f}%{factor:9.1f}"
              f"      {f_in:5.1f}% / {a_in:.1f}%")
        if factor >= 2.0 and f_out >= 50:
            worst.append({"column": column,
                          "pictogram": column.split("_")[0],
                          "pct_flagged": round(f_out, 1),
                          "pct_actual": round(a_out, 1),
                          "factor": round(factor, 1)})

    summary = {
        "training_distribution": bound,
        "n_test_out_of_domain": int(out.sum()),
        "n_test_in_domain": int(inside.sum()),
        "pct_test_out_of_domain": round(100 * out.mean(), 2),
        "over_predicting_classes": worst,
        "worked_example": {
            "compound": "water",
            "heavy_atoms": 1,
            "flagged_by_model": ["GHS05_Corrosive", "GHS06_AcuteToxicity",
                                 "GHS08_HealthHazard"],
            "actual_pictograms": [],
        },
    }
    with open(stamped("EXTRA_applicability_domain.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n   {stamped('EXTRA_applicability_domain.csv')}")
    print(f"   {stamped('EXTRA_applicability_domain.json')}")
    print(f"\nClasses over-predicted outside the domain: "
          f"{', '.join(w['pictogram'] for w in worst) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
