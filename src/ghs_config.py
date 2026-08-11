"""
SHARED PROJECT CONFIGURATION
============================
Every step of the GHS hazard-classification project imports this module so that
the random seed, folder layout, and - most importantly - the definition of the
nine GHS label columns are identical everywhere.

Author : Sareer Ahmad
Project: Interpretable Machine Learning for Predicting GHS Chemical Hazard
         Classifications
"""

import os
import random
from datetime import datetime

# ---------------------------------------------------------------------------
# RULE 5 - REPRODUCIBILITY
# One seed, used by Python, NumPy, scikit-learn, XGBoost and every sampler.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Date stamp appended to output filenames (Rule 3 naming convention:
# STEP[number]_[description]_[YYYYMMDD].ext)
TODAY = datetime.now().strftime("%Y%m%d")

# ---------------------------------------------------------------------------
# FOLDER LAYOUT
# ---------------------------------------------------------------------------
# The project root is worked out from where this file sits, rather than being
# typed in as a fixed path. This file lives in src/, one level below the root,
# so going up one directory from src/ gives the root wherever the project has
# been copied to - the original Windows machine, a Linux server, or a hosted
# deployment such as Streamlit Community Cloud. On the machine the analysis was
# run on this still evaluates to D:\GHS_Project exactly as before.
# Set the GHS_PROJECT_ROOT environment variable to override it.
PROJECT_ROOT = os.environ.get(
    "GHS_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DIR_CLEAN = os.path.join(PROJECT_ROOT, "data", "cleaned")
DIR_SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
DIR_FEATURES = os.path.join(PROJECT_ROOT, "features")
DIR_MODELS = os.path.join(PROJECT_ROOT, "models")
DIR_EVAL = os.path.join(PROJECT_ROOT, "evaluation")
DIR_SHAP = os.path.join(PROJECT_ROOT, "shap_analysis")
DIR_MALAYSIA = os.path.join(PROJECT_ROOT, "malaysia_validation")
DIR_INTERFACE = os.path.join(PROJECT_ROOT, "interface")
DIR_PUB = os.path.join(PROJECT_ROOT, "publication_materials")
DIR_PUB_FIGS = os.path.join(PROJECT_ROOT, "publication_figures")
DIR_LOGS = os.path.join(PROJECT_ROOT, "logs")

for _d in (DIR_RAW, DIR_CLEAN, DIR_SPLITS, DIR_FEATURES, DIR_MODELS, DIR_EVAL,
           DIR_SHAP, DIR_MALAYSIA, DIR_INTERFACE, DIR_PUB, DIR_PUB_FIGS, DIR_LOGS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# THE NINE GHS LABEL COLUMNS
# ---------------------------------------------------------------------------
# These names follow the official United Nations GHS pictogram numbering
# (10th revised edition), which is also what PubChem, ECHA and Malaysia's DOSH
# use:
#
#     GHS01 = Exploding bomb       -> Explosive
#     GHS02 = Flame                -> Flammable
#     GHS03 = Flame over circle    -> Oxidiser
#     GHS04 = Gas cylinder         -> Compressed gas
#     GHS05 = Corrosion            -> Corrosive
#     GHS06 = Skull and crossbones -> Acute toxicity
#     GHS07 = Exclamation mark     -> Irritant / harmful
#     GHS08 = Health hazard        -> Serious health hazard
#     GHS09 = Environment          -> Environmental hazard
#
# CORRECTION APPLIED TO THE ORIGINAL STUDY DESIGN
# The research proposal named three of these columns GHS07_HealthHazard,
# GHS08_Environmental and GHS09_Irritant, which rotates the descriptive
# suffixes relative to the official scheme. The underlying data were always
# bound to the numeric pictogram code and were therefore always correct; only
# the three labels were wrong. They have now been renamed to match the UN
# definitions:
#
#     GHS07_HealthHazard  -> GHS07_Irritant
#     GHS08_Environmental -> GHS08_HealthHazard
#     GHS09_Irritant      -> GHS09_Environmental
#
# The correspondence between the original and corrected names is recorded in
# STEP2_ghs_label_schema.csv so that anyone holding an earlier copy of the
# outputs can map between the two.
# ---------------------------------------------------------------------------
GHS_LABEL_COLUMNS = [
    "GHS01_Explosive",
    "GHS02_Flammable",
    "GHS03_Oxidising",
    "GHS04_CompressedGas",
    "GHS05_Corrosive",
    "GHS06_AcuteToxicity",
    "GHS07_Irritant",
    "GHS08_HealthHazard",
    "GHS09_Environmental",
]

# Short pictogram code -> the column it fills. The code remains the
# authoritative binding between PubChem's data and this project's columns.
PICTOGRAM_CODE_TO_COLUMN = {
    "GHS01": "GHS01_Explosive",
    "GHS02": "GHS02_Flammable",
    "GHS03": "GHS03_Oxidising",
    "GHS04": "GHS04_CompressedGas",
    "GHS05": "GHS05_Corrosive",
    "GHS06": "GHS06_AcuteToxicity",
    "GHS07": "GHS07_Irritant",
    "GHS08": "GHS08_HealthHazard",
    "GHS09": "GHS09_Environmental",
}

# The full UN/PubChem meaning of each column, including the pictogram it
# depicts. Used for every human-readable plot title, report and table.
GHS_TRUE_MEANING = {
    "GHS01_Explosive":     "Explosive (exploding bomb)",
    "GHS02_Flammable":     "Flammable (flame)",
    "GHS03_Oxidising":     "Oxidiser (flame over circle)",
    "GHS04_CompressedGas": "Compressed gas (gas cylinder)",
    "GHS05_Corrosive":     "Corrosive (corrosion)",
    "GHS06_AcuteToxicity": "Acute toxicity (skull and crossbones)",
    "GHS07_Irritant":      "Irritant / harmful (exclamation mark)",
    "GHS08_HealthHazard":  "Serious health hazard (health hazard)",
    "GHS09_Environmental": "Environmental hazard (environment)",
}

# The name each column carried in the original research proposal, kept so that
# older output files can still be mapped onto the corrected schema.
ORIGINAL_PROPOSAL_NAME = {
    "GHS01_Explosive":     "GHS01_Explosive",
    "GHS02_Flammable":     "GHS02_Flammable",
    "GHS03_Oxidising":     "GHS03_Oxidising",
    "GHS04_CompressedGas": "GHS04_CompressedGas",
    "GHS05_Corrosive":     "GHS05_Corrosive",
    "GHS06_AcuteToxicity": "GHS06_AcuteToxicity",
    "GHS07_Irritant":      "GHS07_HealthHazard",
    "GHS08_HealthHazard":  "GHS08_Environmental",
    "GHS09_Environmental": "GHS09_Irritant",
}

# Old name -> new name, for migrating files written before the correction.
RENAME_MAP = {old: new for new, old in ORIGINAL_PROPOSAL_NAME.items()
              if old != new}

# The name PubChem puts in the 'Extra' field of each pictogram icon.
PICTOGRAM_EXTRA_TO_CODE = {
    "Explosive":            "GHS01",
    "Flammable":            "GHS02",
    "Oxidizer":             "GHS03",
    "Oxidiser":             "GHS03",
    "Compressed Gas":       "GHS04",
    "Corrosive":            "GHS05",
    "Acute Toxic":          "GHS06",
    "Irritant":             "GHS07",
    "Health Hazard":        "GHS08",
    "Environmental Hazard": "GHS09",
}

# Retained for backward compatibility with code written before the rename.
# Every column now already carries its recommended name, so this maps each
# column to itself.
GHS_RECOMMENDED_NAME = {column: column for column in GHS_LABEL_COLUMNS}


def get_ablation_identity():
    """
    Return the name and file of the Random Forest ablation, as it actually ran.

    Step 7 records what the ablation really measured, because that depends on
    whether SMOTE could be applied at the dataset size in use. At full size the
    oversampled matrices exceed the memory budget, SMOTE is skipped, and the
    ablation instead measures the removal of class weighting.

    Every downstream step calls this rather than hard-coding a label. A
    hard-coded "RandomForest_SMOTE" once survived into the results tables and
    the model-selection summary of a run in which no synthetic example was
    ever generated - a discrepancy a reader would reasonably read as
    misrepresentation.

    Returns
    -------
    (name, filename, metadata_dict)
    """
    import json
    path = os.path.join(DIR_MODELS, "STEP7_ablation_metadata.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return (meta.get("ablation_model_name", "RandomForest_Ablation"),
                meta.get("ablation_model_file", ""), meta)

    # No metadata: fall back on whichever model file is present, and never
    # claim SMOTE was used without evidence that it was.
    for filename, name in (
            ("STEP7_rf_noclassweight_ablation.pkl", "RandomForest_NoClassWeight"),
            ("STEP7_rf_smote_ablation.pkl", "RandomForest_SMOTE")):
        if os.path.exists(os.path.join(DIR_MODELS, filename)):
            return name, filename, {}
    return "RandomForest_Ablation", "", {}


def seed_everything(seed=RANDOM_SEED):
    """
    Fix the random seed in every library that has one.

    Called at the top of every step script. Without this, results would change
    slightly on each run and the study would not be reproducible.
    """
    random.seed(seed)                        # Python standard library
    os.environ["PYTHONHASHSEED"] = str(seed)  # deterministic string hashing
    try:
        import numpy as np
        np.random.seed(seed)                  # NumPy legacy global generator
    except ImportError:
        pass
    return seed


def stamped(name):
    """
    Build a Rule-3 compliant output path inside the project root.

    Example: stamped("STEP2_raw_ghs_dataset.csv") ->
             D:\\GHS_Project\\STEP2_raw_ghs_dataset.csv
    """
    return os.path.join(PROJECT_ROOT, name)
