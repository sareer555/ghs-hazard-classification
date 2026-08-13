"""
REBUILD TABLE S4B FROM CACHED SHAP VALUES
==========================================
build_interpretation_tables() in step10_shap_analysis.py computes the same
per-class SHAP summary either way; only the "Matches_Chemical_Expectation"
verdict changed - a MACCS or Morgan substructure key can never appear on the
per-class list of expected BULK physicochemical descriptors, so checking a
substructure key against that list always failed regardless of whether the
substructure is textbook-correct chemistry for the hazard, which is what
happened to MACCS_70, the descriptor the abstract cites as the clearest
example of the model recovering known chemistry.

Recomputing that verdict does not require recomputing SHAP itself, which took
most of a 40-minute budget on this machine. STEP10_shap_values.npz already
holds the full per-compound, per-class SHAP array from that run; this script
loads it back and re-runs only the table-building step.

Author : Sareer Ahmad
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import DIR_FEATURES, DIR_SHAP, GHS_LABEL_COLUMNS, stamped


def main():
    """Reload the cached SHAP values and rebuild the interpretation table."""
    print("=" * 78)
    print("REBUILDING TABLE S4B FROM CACHED SHAP VALUES")
    print("=" * 78)

    from step10_shap_analysis import build_interpretation_tables

    npz_path = os.path.join(DIR_SHAP, "STEP10_shap_values.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found - run step10_shap_analysis.py "
                         f"in full instead.")
    cached = np.load(npz_path)
    shap_by_class = [cached[f"shap_{c}"] for c in GHS_LABEL_COLUMNS]
    explain_idx = cached["explained_indices"]

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    X_explain = X[explain_idx]

    with open(stamped("STEP4_feature_names.txt"), encoding="utf-8") as fh:
        feature_names = [line.strip() for line in fh
                         if line.strip() and not line.startswith("#")]

    compound_index = pd.read_csv(
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"), low_memory=False)
    smiles_column = ("CanonicalSMILES_RDKit"
                     if "CanonicalSMILES_RDKit" in compound_index.columns
                     else "SMILES")

    print(f"\nLoaded cached SHAP values for {X_explain.shape[0]:,} compounds "
         f"x {len(feature_names):,} features")

    _, interpretation_table = build_interpretation_tables(
        shap_by_class, feature_names, X_explain,
        compound_index[smiles_column].tolist())

    out_path = stamped("STEP10_SHAP_chemical_interpretation.csv")
    interpretation_table.to_csv(out_path, index=False)

    before_after = interpretation_table["Matches_Chemical_Expectation"].value_counts()
    print("\nNew verdict counts:")
    for verdict, count in before_after.items():
        print(f"   {count:>3}  {verdict}")

    print(f"\n   {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
