"""
BUILD THE ZENODO DATA ARCHIVE
=============================
The GitHub repository holds the code, the figures and the manuscript, and
Zenodo archives that automatically from a release. What GitHub cannot hold is
the data: the curated datasets and the larger trained models exceed its file
size limit.

This script bundles those into a single archive for upload as a second Zenodo
record, together with a README explaining what each file is.

What is deliberately NOT included: roughly 8 GB of .npy intermediate arrays.
Those are byte-for-byte reproducible by re-running Steps 4 and 6, so archiving
them would cost a reader a very long download to obtain something the code
regenerates in minutes.

Author : Sareer Ahmad
"""

import os
import sys
import json
import zipfile
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import PROJECT_ROOT, GHS_LABEL_COLUMNS, stamped

OUT = os.path.join(PROJECT_ROOT, "zenodo_data_archive")
os.makedirs(OUT, exist_ok=True)

# (path relative to project root, description for the README)
CONTENTS = [
    ("STEP2_raw_ghs_dataset.csv",
     "Raw dataset: every compound PubChem holds a GHS classification for, with "
     "SMILES, formula, InChIKey, CAS where available, the nine binary hazard "
     "labels, and which regulatory sources contributed."),
    ("STEP2_ghs_label_schema.csv",
     "Authoritative meaning of each of the nine label columns, with the "
     "correspondence to the names used in the original study design."),
    ("STEP3_cleaned_ghs_dataset.csv",
     "Cleaned dataset used for all modelling: SMILES validated with RDKit, "
     "duplicates removed by InChIKey, disagreements between regulatory sources "
     "resolved by majority vote. This is the file to start from."),
    ("STEP3_class_distribution_table.csv",
     "Counts, percentages and imbalance ratios per hazard class."),
    ("STEP4_feature_matrix.csv",
     "The 816 molecular descriptors retained after variance filtering, one row "
     "per compound."),
    ("STEP4_label_matrix.csv",
     "The nine binary hazard labels, aligned row-for-row with the feature "
     "matrix."),
    ("STEP4_feature_names.txt",
     "Descriptor names, in feature-matrix column order."),
    ("features/STEP4_compound_index.csv",
     "Compound identity for each feature-matrix row: CID, name, SMILES, "
     "InChIKey, formula."),
    ("STEP5_split_class_distribution.csv",
     "Per-class positive counts in the training, validation and test splits."),
    ("models/STEP8_rf_tuned.pkl",
     "Tuned Random Forest, refitted on the training split."),
    ("models/STEP7_rf_model.pkl",
     "Random Forest with balanced class weighting (Step 7 defaults)."),
    ("models/STEP7_rf_noclassweight_ablation.pkl",
     "Ablation: the same Random Forest without class weighting. Note that "
     "SMOTE was NOT applied at full dataset scale - see the Methods."),
    ("models/STEP7_svm_model.pkl",
     "Support vector machine, trained on 100 features and a subsampled "
     "training set; see the Limitations."),
    ("models/STEP8_xgb_tuned.pkl",
     "Tuned XGBoost - the best model, and the one the application uses. Also "
     "in the GitHub repository."),
    ("STEP9_model_comparison_results.csv",
     "Full evaluation: every model, class, metric and decision threshold, with "
     "bootstrap confidence intervals."),
    ("STEP9_calibrated_thresholds.json",
     "Decision thresholds fitted on validation data, per model and class."),
    ("STEP10_mean_SHAP_values.csv",
     "Mean absolute SHAP value per descriptor per hazard class."),
    ("STEP10_SHAP_chemical_interpretation.csv",
     "Top five descriptors per class with chemical interpretation."),
    ("STEP11_malaysia_validation_results.csv",
     "Predictions and reference labels for the Malaysian industrial chemicals "
     "and the Johor 2019 incident compounds."),
]


def build_readme(included, skipped):
    """Write the README that accompanies the archive."""
    clean = json.load(open(stamped("STEP3_cleaning_summary.json"), encoding="utf-8"))
    ev = json.load(open(stamped("STEP9_evaluation_summary.json"), encoding="utf-8"))
    split = json.load(open(stamped("STEP5_split_metadata.json"), encoding="utf-8"))

    lines = []
    lines.append("=" * 78)
    lines.append("DATA ARCHIVE")
    lines.append("Interpretable Machine Learning for Predicting GHS Chemical")
    lines.append("Hazard Classifications")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Sareer Ahmad")
    lines.append("Federal Directorate of Education, Islamabad, Pakistan")
    lines.append("ORCID: https://orcid.org/0009-0003-2580-091X")
    lines.append(f"Archive created: {datetime.now():%d %B %Y}")
    lines.append("")
    lines.append("Analysis code: https://github.com/sareer555/ghs-hazard-classification")
    lines.append("")
    lines.append("-" * 78)
    lines.append("WHAT THIS IS")
    lines.append("-" * 78)
    lines.append("")
    lines.append(f"GHS hazard classifications for {clean['final_cleaned_compounds']:,} "
                 f"unique chemical compounds, harvested from PubChem and")
    lines.append("reconciled across five independent regulatory sources: the European")
    lines.append("Chemicals Agency, Regulation (EC) No 1272/2008, the Hazardous Substances")
    lines.append("Data Bank, NITE-CMC and the Hazardous Chemical Information System of Safe")
    lines.append("Work Australia. Also included are the computed molecular descriptors, the")
    lines.append("trained models, and every result table in the paper.")
    lines.append("")
    lines.append(f"Best model: {ev['best_model']}, mean AUC-ROC "
                 f"{ev['mean_auc_per_model'][ev['best_model']]:.3f} across the nine hazard")
    lines.append(f"classes, on a Bemis-Murcko scaffold split "
                 f"({split['n_train']:,} train / {split['n_val']:,} validation /")
    lines.append(f"{split['n_test']:,} test) in which no chemical skeleton is shared between")
    lines.append("splits.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("THE NINE LABEL COLUMNS")
    lines.append("-" * 78)
    lines.append("")
    lines.append("Names follow official United Nations GHS pictogram numbering.")
    lines.append("")
    for c in GHS_LABEL_COLUMNS:
        lines.append(f"  {c}")
    lines.append("")
    lines.append("A value of 1 means the compound carries that pictogram, 0 that it does")
    lines.append("not. Note that 0 means 'not assigned', which is not the same as")
    lines.append("'demonstrated to be safe' - a compound may simply not have been assessed")
    lines.append("for that hazard.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("FILES")
    lines.append("-" * 78)
    for path, desc in included:
        size = os.path.getsize(os.path.join(PROJECT_ROOT, path)) / 1e6
        lines.append("")
        lines.append(f"{os.path.basename(path)}  ({size:,.1f} MB)")
        for chunk in [desc[i:i + 74] for i in range(0, len(desc), 74)]:
            lines.append(f"    {chunk}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("WHAT IS NOT HERE, AND WHY")
    lines.append("-" * 78)
    lines.append("")
    lines.append("About 8 GB of intermediate NumPy arrays are omitted. They are")
    lines.append("reproducible exactly by re-running Steps 4 and 6 of the pipeline, which")
    lines.append("takes minutes; downloading them would take considerably longer.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("REUSE")
    lines.append("-" * 78)
    lines.append("")
    lines.append("The code is MIT licensed. The underlying classifications originate with")
    lines.append("the regulatory bodies named above and remain subject to their terms.")
    lines.append("PubChem data are in the public domain.")
    lines.append("")
    lines.append("DISCLAIMER. These models are computational screening tools. They do not")
    lines.append("replace laboratory testing or regulatory assessment under Malaysia's")
    lines.append("CLASS Regulations 2013 or equivalent legislation elsewhere. Predictions")
    lines.append("must not be the sole basis for any decision affecting human safety.")
    lines.append("")
    if skipped:
        lines.append("-" * 78)
        lines.append("NOTE: the following expected files were absent when the archive was")
        lines.append("built:")
        for s in skipped:
            lines.append(f"  {s}")
        lines.append("")
    return "\n".join(lines)


def main():
    """Build the archive."""
    print("=" * 78)
    print("BUILDING THE ZENODO DATA ARCHIVE")
    print("=" * 78)

    included, skipped = [], []
    for path, desc in CONTENTS:
        if os.path.exists(os.path.join(PROJECT_ROOT, path)):
            included.append((path, desc))
        else:
            skipped.append(path)
            print(f"   missing (skipped): {path}")

    readme = build_readme(included, skipped)
    readme_path = os.path.join(OUT, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(readme)

    archive = os.path.join(OUT, "GHS_hazard_classification_data.zip")
    raw = 0
    print(f"\nCompressing {len(included)} files ...")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        zf.write(readme_path, "README.txt")
        for path, _ in included:
            full = os.path.join(PROJECT_ROOT, path)
            raw += os.path.getsize(full)
            zf.write(full, os.path.basename(path))
            print(f"   {os.path.basename(path)}")

    size = os.path.getsize(archive)
    print(f"\n   raw        : {raw/1e6:,.0f} MB")
    print(f"   compressed : {size/1e6:,.0f} MB  "
          f"({100*(1-size/raw):.0f}% smaller)")
    print(f"\n   {archive}")
    print(f"   {readme_path}")
    print("\nUpload the .zip to Zenodo as a Dataset record. The README is")
    print("included inside it and is also written separately so you can paste")
    print("its text into the Zenodo description box.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
