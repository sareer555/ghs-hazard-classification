"""
COLUMN-NAME MIGRATION
=====================
Renames the three mislabelled GHS columns in every artefact already written to
disk, so that the whole project uses the official United Nations pictogram
names:

    GHS07_HealthHazard  -> GHS07_Irritant
    GHS08_Environmental -> GHS08_HealthHazard
    GHS09_Irritant      -> GHS09_Environmental

Only the LABELS change. The data underneath was always bound to the numeric
pictogram code and was already correct, so no value in any file is altered and
no model needs retraining - the label matrix is a plain array whose column
order is unchanged.

Files that Steps 9 to 13 regenerate from scratch are left alone; re-running
those steps picks up the new names automatically. This script handles the
stored datasets that nothing else rewrites.

Author : Sareer Ahmad
"""

import os
import re
import sys
import csv
import json
import glob
import shutil

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (PROJECT_ROOT, DIR_RAW, DIR_CLEAN, DIR_FEATURES,
                        DIR_SPLITS, DIR_MALAYSIA, GHS_LABEL_COLUMNS,
                        GHS_TRUE_MEANING, ORIGINAL_PROPOSAL_NAME, RENAME_MAP,
                        stamped)

CHANGED, SKIPPED = [], []


def rename_csv_columns(path, chunksize=None):
    """
    Rewrite one CSV with the three columns renamed.

    Large files are streamed in chunks so that a 78 MB dataset never has to be
    held in memory twice on a machine with 7.9 GB of RAM.
    """
    if not os.path.exists(path):
        SKIPPED.append((path, "not found"))
        return False

    # The old names appear in two different places depending on the file: as
    # column HEADERS in the datasets, and as row VALUES in the summary tables
    # (which have a "GHS_Column" column listing them). Both have to be checked
    # - looking only at headers silently skips every summary table.
    VALUE_COLUMNS = ("GHS_Column", "column_name_as_in_proposal", "column_name",
                     "Class", "GHS_Class", "Feature_Class")

    header = pd.read_csv(path, nrows=0)
    header_hits = [c for c in header.columns
                   if any(c == old or c.endswith("_" + old) or c.startswith(old)
                          for old in RENAME_MAP)]

    value_columns_present = [c for c in header.columns if c in VALUE_COLUMNS]
    value_hits = False
    if value_columns_present:
        sample = pd.read_csv(path, usecols=value_columns_present, nrows=5000)
        value_hits = bool(sample.isin(list(RENAME_MAP)).any().any())

    if not header_hits and not value_hits:
        SKIPPED.append((path, "no affected columns or values"))
        return False
    hits = header_hits or value_columns_present

    def new_name(column):
        """Rename a column, preserving any PRED_/PROB_ prefix or _union suffix."""
        for old, new in RENAME_MAP.items():
            if column == old:
                return new
            for prefix in ("PRED_", "PROB_"):
                if column == prefix + old:
                    return prefix + new
            if column == old + "_union":
                return new + "_union"
        return column

    temporary = path + ".migrating"
    if chunksize:
        first = True
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = chunk.rename(columns=new_name)
            for value_column in VALUE_COLUMNS:
                if value_column in chunk.columns:
                    chunk[value_column] = chunk[value_column].replace(RENAME_MAP)
            chunk.to_csv(temporary, index=False, mode="w" if first else "a",
                         header=first)
            first = False
    else:
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.rename(columns=new_name)
        # Some tables store the column name as a VALUE, not a header.
        for value_column in VALUE_COLUMNS:
            if value_column in frame.columns:
                frame[value_column] = frame[value_column].replace(RENAME_MAP)
        frame.to_csv(temporary, index=False)

    os.replace(temporary, path)
    CHANGED.append((path, f"{len(hits)} column(s)"))
    return True


def rename_json_keys(path):
    """Rewrite a JSON file, renaming the three keys wherever they appear."""
    if not os.path.exists(path):
        SKIPPED.append((path, "not found"))
        return False

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    def walk(node):
        """Recursively rename dictionary keys and string values."""
        if isinstance(node, dict):
            return {RENAME_MAP.get(k, k): walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            return RENAME_MAP.get(node, node)
        return node

    before = json.dumps(data)
    data = walk(data)
    after = json.dumps(data)
    if before == after:
        SKIPPED.append((path, "no affected keys"))
        return False

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    CHANGED.append((path, "JSON keys"))
    return True


def rename_balanced_arrays():
    """Rename the per-class SMOTE files, whose names embed the column name."""
    folder = os.path.join(DIR_FEATURES, "STEP6_balanced")
    if not os.path.isdir(folder):
        SKIPPED.append((folder, "not found"))
        return 0
    count = 0
    for old, new in RENAME_MAP.items():
        for prefix in ("X_", "y_"):
            source = os.path.join(folder, f"{prefix}{old}.npy")
            target = os.path.join(folder, f"{prefix}{new}.npy")
            if os.path.exists(source):
                os.replace(source, target)
                count += 1
    if count:
        CHANGED.append((folder, f"{count} array file(s) renamed"))
    return count


def rewrite_label_schema():
    """
    Rewrite the label-schema file to document the correction.

    It now records the original proposal name alongside the corrected one, so
    that anyone holding an earlier copy of the outputs can map between them.
    """
    path = stamped("STEP2_ghs_label_schema.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column_name", "pictogram_code",
                         "authoritative_UN_PubChem_meaning",
                         "name_in_original_proposal", "was_renamed"])
        for column in GHS_LABEL_COLUMNS:
            original = ORIGINAL_PROPOSAL_NAME[column]
            writer.writerow([column, column.split("_")[0],
                             GHS_TRUE_MEANING[column], original,
                             "YES" if original != column else "no"])
    CHANGED.append((path, "rewritten with the rename history"))


def main():
    """Migrate every stored artefact to the corrected column names."""
    print("=" * 78)
    print("COLUMN-NAME MIGRATION")
    print("=" * 78)
    for old, new in RENAME_MAP.items():
        print(f"   {old:<22} -> {new}")
    print("\nOnly labels change; no data value is altered.\n")

    # ---- large datasets, streamed ------------------------------------------
    print("Large datasets (streamed in chunks):")
    for path, chunks in [
            (stamped("STEP2_raw_ghs_dataset.csv"), 50000),
            (stamped("STEP3_cleaned_ghs_dataset.csv"), 50000),
            (stamped("STEP3_modelling_subset.csv"), 50000),
            (stamped("STEP4_label_matrix.csv"), 50000)]:
        rename_csv_columns(path, chunksize=chunks)
        print(f"   {os.path.basename(path)}")

    for pattern, chunks in [
            (os.path.join(DIR_RAW, "STEP2_per_source_records_*.csv"), 50000),
            (os.path.join(DIR_RAW, "STEP2_raw_ghs_dataset_*.csv"), 50000),
            (os.path.join(DIR_CLEAN, "STEP3_cleaned_ghs_dataset_*.csv"), 50000)]:
        for path in glob.glob(pattern):
            rename_csv_columns(path, chunksize=chunks)
            print(f"   {os.path.basename(path)}")

    # ---- small tables -------------------------------------------------------
    print("\nSmall tables:")
    small = [
        stamped("STEP3_class_distribution_table.csv"),
        stamped("STEP3_subset_prevalence_shift.csv"),
        stamped("STEP5_split_class_distribution.csv"),
        stamped("STEP6_smote_report.csv"),
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"),
        os.path.join(DIR_CLEAN, "STEP3_class_distribution_table.csv"),
        os.path.join(DIR_CLEAN, "STEP3_subset_prevalence_shift.csv"),
        os.path.join(DIR_SPLITS, "STEP5_split_class_distribution.csv"),
        os.path.join(DIR_FEATURES, "STEP6_smote_report.csv"),
    ]
    small += glob.glob(os.path.join(DIR_MALAYSIA, "*.csv"))
    small += glob.glob(os.path.join(PROJECT_ROOT, "STEP9_*.csv"))
    small += glob.glob(os.path.join(PROJECT_ROOT, "STEP10_*.csv"))
    small += glob.glob(os.path.join(PROJECT_ROOT, "STEP11_*.csv"))
    for path in small:
        if rename_csv_columns(path):
            print(f"   {os.path.basename(path)}")

    # ---- JSON configuration -------------------------------------------------
    print("\nJSON files:")
    for path in [stamped("STEP6_imbalance_config.json"),
                 stamped("STEP9_calibrated_thresholds.json"),
                 stamped("STEP9_evaluation_summary.json"),
                 stamped("STEP8_best_hyperparameters.json"),
                 stamped("STEP10_shap_summary.json")]:
        if rename_json_keys(path):
            print(f"   {os.path.basename(path)}")

    # ---- SMOTE arrays and the schema file ----------------------------------
    print("\nOther artefacts:")
    n = rename_balanced_arrays()
    print(f"   per-class SMOTE arrays: {n} renamed")
    rewrite_label_schema()
    print(f"   STEP2_ghs_label_schema.csv rewritten")

    print("\n" + "=" * 78)
    print(f"MIGRATION COMPLETE: {len(CHANGED)} file(s) updated, "
          f"{len(SKIPPED)} unaffected or absent")
    print("=" * 78)
    print("\nNow re-run Steps 9, 10, 11 and 13 so that every derived table,")
    print("figure and report is regenerated with the corrected names:")
    print("    python run_pipeline.py 9 99")
    return 0


if __name__ == "__main__":
    sys.exit(main())
