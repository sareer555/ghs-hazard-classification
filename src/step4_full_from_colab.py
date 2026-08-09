"""
STEP 4 (FULL DATASET) - BUILD THE FEATURE MATRIX FOR ALL 243,323 COMPOUNDS
=========================================================================
Step 4 normally computes descriptors from SMILES. That work has already been
done for the complete cleaned dataset on Google Colab, and the resulting
matrix was brought back as colab_X_full.npy. This script reuses it instead of
spending another half hour recomputing an identical result.

The descriptors are the same 1218 as the local Step 4 - 19 physicochemical,
1024 Morgan (ECFP4), 167 MACCS and 8 topological - computed by the same code,
in the same order, from the same cleaned dataset. What remains to be done here
is the imputation and variance filtering, and writing the outputs in the
layout the rest of the pipeline expects.

Memory: the full matrix is 1.19 GB, which is a large fraction of this
machine's free RAM. It is therefore memory-mapped rather than loaded, and
every transformation is done in chunks.

Author : Sareer Ahmad
"""

import os
import gc
import sys
import json
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_FEATURES, DIR_LOGS,
                        GHS_LABEL_COLUMNS, seed_everything, stamped)
from step4_descriptors import build_feature_names

seed_everything()

COLAB_DIR = r"D:\GHS_Project\colab_results"
CHUNK = 20000          # rows processed at a time


def main():
    """Build the full-dataset feature matrix from the cached descriptors."""
    started = time.time()
    print("=" * 78)
    print("STEP 4 (FULL DATASET) - feature matrix for all 243,323 compounds")
    print("=" * 78)

    source = os.path.join(COLAB_DIR, "colab_X_full.npy")
    if not os.path.exists(source):
        raise SystemExit(
            f"Cached descriptors not found at {source}.\n"
            f"Either run the Colab notebook and place its output there, or run "
            f"src/step4_descriptors.py to compute them locally (about 30 min).")

    X = np.load(source, mmap_mode="r")
    df = pd.read_csv(stamped("STEP3_cleaned_ghs_dataset.csv"), low_memory=False)
    print(f"Descriptors : {X.shape[0]:,} x {X.shape[1]:,} (memory-mapped)")
    print(f"Cleaned data: {len(df):,} compounds")

    if X.shape[0] != len(df):
        raise SystemExit(
            f"Row mismatch: {X.shape[0]:,} descriptor rows against "
            f"{len(df):,} compounds. The descriptor matrix must have been "
            f"built from this exact cleaned dataset, in the same order.")

    feature_names = build_feature_names()
    if len(feature_names) != X.shape[1]:
        raise SystemExit(
            f"Expected {len(feature_names)} descriptors but the matrix has "
            f"{X.shape[1]}. The cached matrix was built with different "
            f"settings and cannot be reused.")

    # ---- 4f: column medians, computed in chunks ---------------------------
    print("\n[4f] Scanning for missing values ...")
    n_nan_total = 0
    nan_columns = np.zeros(X.shape[1], dtype=bool)
    for start in range(0, X.shape[0], CHUNK):
        block = np.asarray(X[start:start + CHUNK])
        mask = np.isnan(block)
        n_nan_total += int(mask.sum())
        nan_columns |= mask.any(axis=0)
        del block, mask
    print(f"      {n_nan_total:,} missing values across "
          f"{int(nan_columns.sum())} descriptor column(s)")

    medians = np.zeros(X.shape[1], dtype=np.float32)
    if n_nan_total:
        # Only the affected columns need a median, and each is read once.
        for column in np.where(nan_columns)[0]:
            values = np.asarray(X[:, column])
            medians[column] = np.nanmedian(values[np.isfinite(values)]) \
                if np.isfinite(values).any() else 0.0
            del values
        print(f"      medians computed for the affected columns")

    # ---- 4e: variance, computed in one pass -------------------------------
    print("\n[4e] Computing per-descriptor variance ...")
    total = np.zeros(X.shape[1], dtype=np.float64)
    total_sq = np.zeros(X.shape[1], dtype=np.float64)
    n_rows = X.shape[0]
    for start in range(0, n_rows, CHUNK):
        block = np.asarray(X[start:start + CHUNK], dtype=np.float64)
        if n_nan_total:
            inds = np.where(np.isnan(block))
            block[inds] = medians[inds[1]]
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        total += block.sum(axis=0)
        total_sq += (block ** 2).sum(axis=0)
        del block
    mean = total / n_rows
    variance = total_sq / n_rows - mean ** 2
    keep = variance > 0.01
    kept_names = [n for n, k in zip(feature_names, keep) if k]
    keep_cols = np.where(keep)[0]
    print(f"      {int((~keep).sum()):,} of {X.shape[1]:,} descriptors removed")
    print(f"      {len(kept_names):,} retained")

    # ---- write the filtered matrix in chunks ------------------------------
    print("\n[4] Writing the filtered feature matrix ...")
    out_path = os.path.join(DIR_FEATURES, "STEP4_X.npy")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n_rows, len(keep_cols)))
    for start in range(0, n_rows, CHUNK):
        block = np.asarray(X[start:start + CHUNK], dtype=np.float32)
        if n_nan_total:
            inds = np.where(np.isnan(block))
            block[inds] = medians[inds[1]]
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        out[start:start + CHUNK] = block[:, keep_cols]
        del block
    out.flush()
    del out, X
    gc.collect()
    print(f"      {out_path}")

    # ---- labels and index --------------------------------------------------
    labels = df[GHS_LABEL_COLUMNS].astype(np.int8)
    np.save(os.path.join(DIR_FEATURES, "STEP4_y.npy"), labels.to_numpy())

    with open(stamped("STEP4_feature_names.txt"), "w", encoding="utf-8") as fh:
        fh.write("# Feature names retained after VarianceThreshold(0.01)\n")
        fh.write(f"# {len(kept_names)} features, in matrix column order\n")
        for name in kept_names:
            fh.write(name + "\n")

    index_columns = [c for c in ["CID", "Name", "SMILES",
                                 "CanonicalSMILES_RDKit", "InChIKey_RDKit",
                                 "MolecularFormula", "CAS"] if c in df.columns]
    df[index_columns + GHS_LABEL_COLUMNS].to_csv(
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"), index=False)

    label_frame = labels.copy()
    label_frame.insert(0, "CID", df["CID"].values)
    label_frame.to_csv(stamped("STEP4_label_matrix.csv"), index=False)

    # The full feature matrix as CSV would be roughly 2 GB and is far slower
    # to read than the .npy every later step actually uses. It is written
    # compressed instead, which keeps the deliverable without the bulk.
    print("\n[4] Writing the feature matrix as compressed CSV ...")
    X_final = np.load(out_path, mmap_mode="r")
    csv_path = stamped("STEP4_feature_matrix.csv.gz")
    first = True
    for start in range(0, n_rows, CHUNK):
        block = pd.DataFrame(np.asarray(X_final[start:start + CHUNK]),
                             columns=kept_names)
        block.insert(0, "CID", df["CID"].values[start:start + CHUNK])
        block.to_csv(csv_path, index=False, float_format="%.5g",
                     mode="w" if first else "a", header=first,
                     compression="gzip")
        first = False
        del block
    print(f"      {csv_path}")

    metadata = {
        "n_compounds": int(n_rows),
        "n_features_computed": int(len(feature_names)),
        "n_features_after_variance_filter": int(len(kept_names)),
        "n_features_removed": int((~keep).sum()),
        "n_missing_values_imputed": int(n_nan_total),
        "variance_threshold": 0.01,
        "descriptor_source": ("reused from the Google Colab run "
                              "(colab_X_full.npy); identical code and settings "
                              "to src/step4_descriptors.py"),
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with open(stamped("STEP4_descriptor_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n" + "=" * 78)
    print("STEP 4 (FULL DATASET) PROGRESS REPORT")
    print("=" * 78)
    print(f"FEATURE MATRIX: {n_rows:,} compounds x {len(kept_names):,} features")
    print(f"                (from {len(feature_names):,} computed; "
          f"{int((~keep).sum()):,} removed by variance filter)")
    print(f"NaN IMPUTED   : {n_nan_total:,}")
    print(f"ELAPSED       : {metadata['elapsed_seconds']/60:.1f} minutes")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
