"""
STEP 4 - MOLECULAR DESCRIPTOR COMPUTATION
=========================================
A machine-learning model cannot read a chemical structure directly. Each
molecule must first be turned into a fixed-length list of numbers - its
"descriptors". This step computes three complementary families of them.

4a  Physicochemical descriptors (19 numbers)
      Bulk properties a chemist would recognise: molecular weight, lipophilicity
      (LogP), polar surface area, hydrogen-bond donors and acceptors, ring
      counts and so on. These carry most of the interpretable signal, which
      matters because Step 10 must explain the model's reasoning in chemical
      terms.

4b  Structural fingerprints (1024 + 167 bits)
      Morgan/ECFP4 fingerprints record which circular atom environments are
      present, and MACCS keys record the presence of 166 predefined
      substructures such as "contains a nitro group". These capture the
      functional-group information that drives most GHS classifications.

4c  Topological descriptors (8 numbers)
      Chi connectivity and Kappa shape indices describe how the atoms are
      wired together and how branched or linear the skeleton is.

4d  All three families are concatenated into one feature matrix.
4f  Missing values are replaced by the median of their column.
4e  Features that barely vary across the dataset are removed.

Note on the order of 4e and 4f
------------------------------
The proposal lists variance filtering (4e) before missing-value imputation
(4f). They are performed in the opposite order here for a purely numerical
reason: a column containing even one NaN has an undefined variance, so
filtering first would silently discard usable descriptors. Imputing first and
then filtering achieves exactly what the proposal intends.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, PROJECT_ROOT, DIR_FEATURES, DIR_LOGS,
                        GHS_LABEL_COLUMNS, seed_everything, stamped)

seed_everything()
warnings.filterwarnings("ignore", category=RuntimeWarning)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, GraphDescriptors
RDLogger.DisableLog("rdApp.*")

ISSUE_LOG = []

# ---------------------------------------------------------------------------
# DESCRIPTOR DEFINITIONS
# ---------------------------------------------------------------------------
# 4a - the nineteen physicochemical descriptors named in the proposal.
# Each entry is (output column name, the name of the RDKit function).
#
# IMPORTANT: the FUNCTION NAMES are stored here as plain strings, not the
# function objects themselves. RDKit's descriptor functions are compiled C++
# objects (Boost.Python.function) which Python cannot pickle, and descriptor
# calculation is spread over several worker processes - so anything stored at
# module level has to survive being pickled and sent to a worker. The actual
# functions are looked up by name inside each worker instead.
PHYSCHEM_DESCRIPTORS = [
    ("MolWt",              "MolWt"),              # average molecular mass
    ("ExactMolWt",         "ExactMolWt"),         # monoisotopic mass
    ("MolLogP",            "MolLogP"),            # lipophilicity (Crippen)
    ("TPSA",               "TPSA"),               # topological polar surface area
    ("NumHDonors",         "NumHDonors"),         # H-bond donors
    ("NumHAcceptors",      "NumHAcceptors"),      # H-bond acceptors
    ("NumRotatableBonds",  "NumRotatableBonds"),  # molecular flexibility
    ("NumAromaticRings",   "NumAromaticRings"),
    ("NumSaturatedRings",  "NumSaturatedRings"),
    ("NumAliphaticRings",  "NumAliphaticRings"),
    ("RingCount",          "RingCount"),
    ("FractionCSP3",       "FractionCSP3"),       # fraction of sp3 carbons
    ("HeavyAtomCount",     "HeavyAtomCount"),     # non-hydrogen atoms
    ("NumHeteroatoms",     "NumHeteroatoms"),     # atoms that are not C or H
    ("NOCount",            "NOCount"),            # nitrogen + oxygen count
    ("NHOHCount",          "NHOHCount"),          # NH and OH count
    ("LabuteASA",          "LabuteASA"),          # approximate surface area
    ("BalabanJ",           "BalabanJ"),           # topological connectivity index
    ("BertzCT",            "BertzCT"),            # structural complexity
]

# 4c - the eight topological descriptors named in the proposal.
TOPOLOGICAL_DESCRIPTORS = [
    ("Chi0",   "Chi0"),    # connectivity indices of increasing order
    ("Chi1",   "Chi1"),
    ("Chi2n",  "Chi2n"),
    ("Chi3n",  "Chi3n"),
    ("Chi4n",  "Chi4n"),
    ("Kappa1", "Kappa1"),  # molecular shape indices
    ("Kappa2", "Kappa2"),
    ("Kappa3", "Kappa3"),
]

# Filled in on first use, separately inside each worker process.
_RESOLVED_FUNCTIONS = None


def _resolve_descriptor_functions():
    """
    Look up the RDKit descriptor functions by name, once per process.

    Returns two lists of callables, in the same order as the tables above.
    The result is cached in a module-level variable so the lookup cost is
    paid only once per worker rather than once per molecule.
    """
    global _RESOLVED_FUNCTIONS
    if _RESOLVED_FUNCTIONS is None:
        physchem = [getattr(Descriptors, attribute)
                    for _, attribute in PHYSCHEM_DESCRIPTORS]
        topological = [getattr(GraphDescriptors, attribute)
                       for _, attribute in TOPOLOGICAL_DESCRIPTORS]
        _RESOLVED_FUNCTIONS = (physchem, topological)
    return _RESOLVED_FUNCTIONS

MORGAN_BITS = 1024      # ECFP4 fingerprint length
MORGAN_RADIUS = 2       # radius 2 == ECFP4
MACCS_BITS = 167        # RDKit returns 167 bits (bit 0 is unused padding)

N_PHYSCHEM = len(PHYSCHEM_DESCRIPTORS)
N_TOPO = len(TOPOLOGICAL_DESCRIPTORS)
N_FEATURES_TOTAL = N_PHYSCHEM + MORGAN_BITS + MACCS_BITS + N_TOPO


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


def build_feature_names():
    """Return the name of every column of the feature matrix, in order."""
    names = [name for name, _ in PHYSCHEM_DESCRIPTORS]
    names += [f"Morgan_{i}" for i in range(MORGAN_BITS)]
    names += [f"MACCS_{i}" for i in range(MACCS_BITS)]
    names += [name for name, _ in TOPOLOGICAL_DESCRIPTORS]
    return names


def descriptors_for_one_molecule(smiles):
    """
    Turn a single SMILES string into one row of the feature matrix.

    Any descriptor that cannot be computed for a particular molecule (a rare
    but real occurrence - BalabanJ is undefined for disconnected structures,
    for instance) becomes NaN and is repaired later by median imputation.

    Returns
    -------
    numpy.ndarray of float32, length N_FEATURES_TOTAL, or None if the molecule
    could not be parsed at all.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    physchem_functions, topological_functions = _resolve_descriptor_functions()

    row = np.full(N_FEATURES_TOTAL, np.nan, dtype=np.float32)
    position = 0

    # ---- 4a physicochemical ------------------------------------------------
    for function in physchem_functions:
        try:
            value = function(mol)
            # Some descriptors can return infinity for pathological structures;
            # infinity would break the scaler in Step 7, so it is treated as
            # missing and imputed.
            row[position] = value if np.isfinite(value) else np.nan
        except Exception:
            row[position] = np.nan
        position += 1

    # ---- 4b Morgan fingerprint (ECFP4) -------------------------------------
    try:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
        # ToBitString gives a string like "0101...". Reading those characters
        # as raw bytes and subtracting 48 (the ASCII code of "0") is the
        # fastest way to turn them into a numeric 0/1 array.
        row[position:position + MORGAN_BITS] = np.frombuffer(
            fingerprint.ToBitString().encode(), dtype=np.uint8) - 48
    except Exception:
        row[position:position + MORGAN_BITS] = 0   # absent bits are genuinely 0
    position += MORGAN_BITS

    # ---- 4b MACCS keys -----------------------------------------------------
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        row[position:position + MACCS_BITS] = np.frombuffer(
            maccs.ToBitString().encode(), dtype=np.uint8) - 48
    except Exception:
        row[position:position + MACCS_BITS] = 0
    position += MACCS_BITS

    # ---- 4c topological ----------------------------------------------------
    for function in topological_functions:
        try:
            value = function(mol)
            row[position] = value if np.isfinite(value) else np.nan
        except Exception:
            row[position] = np.nan
        position += 1

    return row


def _descriptor_chunk(smiles_list):
    """Compute descriptors for a block of molecules inside one worker process."""
    return [descriptors_for_one_molecule(s) for s in smiles_list]


def compute_descriptors(smiles_series, n_jobs=3, chunk_size=500):
    """
    Compute the full feature matrix for a list of SMILES strings.

    Work is split across worker processes because descriptor calculation is
    CPU-bound. Only three of the four logical cores are used, leaving one free
    so the machine stays responsive.

    Returns
    -------
    (X, failed_indices)
        X               : float32 array, shape (n_molecules, N_FEATURES_TOTAL)
        failed_indices  : positions of molecules RDKit could not parse
    """
    smiles_list = list(smiles_series)
    n_molecules = len(smiles_list)
    print(f"\n[4a-4d] Computing {N_FEATURES_TOTAL} descriptors for "
          f"{n_molecules:,} molecules on {n_jobs} worker processes ...")
    print(f"        {N_PHYSCHEM} physicochemical + {MORGAN_BITS} Morgan(ECFP4) "
          f"+ {MACCS_BITS} MACCS + {N_TOPO} topological")

    chunks = [smiles_list[i:i + chunk_size]
              for i in range(0, n_molecules, chunk_size)]

    started = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_descriptor_chunk)(chunk)
        for chunk in tqdm(chunks, desc="        descriptor chunks",
                          unit="chunk", ncols=78)
    )
    elapsed = time.time() - started

    # Flatten the per-chunk lists back into one array.
    X = np.zeros((n_molecules, N_FEATURES_TOTAL), dtype=np.float32)
    failed_indices = []
    position = 0
    for chunk_result in results:
        for row in chunk_result:
            if row is None:
                failed_indices.append(position)
                X[position, :] = np.nan
            else:
                X[position, :] = row
            position += 1

    print(f"        Done in {elapsed / 60:.1f} min "
          f"({1000 * elapsed / max(n_molecules, 1):.1f} ms per molecule)")
    if failed_indices:
        log_issue("4a-4d", f"{len(failed_indices):,} molecule(s) could not be "
                           f"parsed at descriptor stage and were dropped.")
    return X, failed_indices


def compute_descriptors_mordred_fallback(smiles_series):
    """
    FALLBACK for Step 4 if RDKit descriptor calculation fails wholesale.

    Mordred computes over 1800 descriptors and copes with some molecules RDKit
    struggles with. It is only attempted if the primary route fails, because
    it is roughly ten times slower.
    """
    log_issue("4-FALLBACK", "RDKit descriptor computation failed - "
                            "attempting the mordred library.")
    try:
        from mordred import Calculator, descriptors as mordred_descriptors
        calculator = Calculator(mordred_descriptors, ignore_3D=True)
        molecules = [Chem.MolFromSmiles(s) for s in smiles_series]
        frame = calculator.pandas([m for m in molecules if m is not None])
        return frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32), \
            list(frame.columns)
    except Exception as exc:
        log_issue("4-FALLBACK", f"mordred unavailable or failed: {exc}")
        return None, None


def compute_descriptors_padel_fallback(smiles_series):
    """
    FINAL FALLBACK for Step 4 - PaDEL-Descriptor through the padelpy wrapper.

    PaDEL runs as an external Java program and therefore does not depend on
    RDKit at all. Reached only if both RDKit and mordred fail.
    """
    log_issue("4-FALLBACK", "mordred also failed - attempting PaDEL/padelpy.")
    try:
        from padelpy import from_smiles
        rows = [from_smiles(s) for s in smiles_series]
        frame = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
        return frame.to_numpy(dtype=np.float32), list(frame.columns)
    except Exception as exc:
        log_issue("4-FALLBACK", f"padelpy unavailable or failed: {exc}")
        return None, None


def main(input_csv=None):
    """Run the whole of Step 4 and save the feature and label matrices."""
    start_time = time.time()
    # The full cleaned dataset is now the modelling set. The 40,000-compound
    # subset was used while the project was memory-bound; a controlled
    # experiment (src/controlled_size_experiment.py) later showed that training
    # on all 243,323 compounds improves the mean AUC by +0.055, four times the
    # bootstrap confidence interval, so the subset is no longer used.
    input_csv = input_csv or stamped("STEP3_cleaned_ghs_dataset.csv")

    print("=" * 78)
    print("STEP 4 - MOLECULAR DESCRIPTOR COMPUTATION")
    print("=" * 78)
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df):,} compounds from {input_csv}")

    # Prefer RDKit's own canonical SMILES, written in Step 3 - it is already
    # known to parse cleanly.
    smiles_column = ("CanonicalSMILES_RDKit"
                     if "CanonicalSMILES_RDKit" in df.columns else "SMILES")
    print(f"Using SMILES column: {smiles_column}")

    # ---- 4a-4d compute ----------------------------------------------------
    feature_names = build_feature_names()
    X, failed_indices = compute_descriptors(df[smiles_column])

    if X is None or len(failed_indices) == len(df):
        X, feature_names = compute_descriptors_mordred_fallback(df[smiles_column])
        if X is None:
            X, feature_names = compute_descriptors_padel_fallback(df[smiles_column])
        if X is None:
            raise RuntimeError("Step 4 failed: RDKit, mordred and PaDEL all "
                               "failed to produce descriptors.")

    # Drop molecules that failed completely, keeping labels aligned with rows.
    if failed_indices:
        keep_mask = np.ones(len(df), dtype=bool)
        keep_mask[failed_indices] = False
        X = X[keep_mask]
        df = df[keep_mask].reset_index(drop=True)

    print(f"\n        Raw feature matrix: {X.shape[0]:,} compounds "
          f"x {X.shape[1]:,} descriptors")

    # ---- 4f impute missing values ----------------------------------------
    print("\n[4f] Imputing missing descriptor values with the column median ...")
    n_nan = int(np.isnan(X).sum())
    nan_columns = np.where(np.isnan(X).any(axis=0))[0]
    if n_nan:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X = imputer.fit_transform(X).astype(np.float32)
        print(f"      {n_nan:,} missing values imputed across "
              f"{len(nan_columns)} descriptor column(s):")
        for index in nan_columns[:15]:
            print(f"         - {feature_names[index]}")
        if len(nan_columns) > 15:
            print(f"         ... and {len(nan_columns) - 15} more")
        import joblib
        joblib.dump(imputer, os.path.join(DIR_FEATURES, "STEP4_imputer.pkl"))
    else:
        print("      No missing values found - imputation not required.")

    # Guard against any residual non-finite value.
    n_infinite = int((~np.isfinite(X)).sum())
    if n_infinite:
        log_issue("4f", f"{n_infinite:,} non-finite value(s) remained after "
                        f"imputation and were set to 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 4e variance threshold -------------------------------------------
    print("\n[4e] Removing near-constant descriptors (VarianceThreshold=0.01) ...")
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_filtered = selector.fit_transform(X).astype(np.float32)
    kept_mask = selector.get_support()
    kept_names = [name for name, keep in zip(feature_names, kept_mask) if keep]
    n_removed = int((~kept_mask).sum())

    # Show which descriptor families lost the most columns - useful context
    # for the paper, because most Morgan bits are expected to be near-empty.
    removed_names = [name for name, keep in zip(feature_names, kept_mask) if not keep]
    family_counts = {"physicochemical": 0, "Morgan": 0, "MACCS": 0, "topological": 0}
    for name in removed_names:
        if name.startswith("Morgan_"):
            family_counts["Morgan"] += 1
        elif name.startswith("MACCS_"):
            family_counts["MACCS"] += 1
        elif name in dict(TOPOLOGICAL_DESCRIPTORS):
            family_counts["topological"] += 1
        else:
            family_counts["physicochemical"] += 1

    print(f"      Removed {n_removed:,} of {len(feature_names):,} descriptors:")
    for family, count in family_counts.items():
        print(f"         {family:<18}: {count:,} removed")
    print(f"      Retained {X_filtered.shape[1]:,} descriptors.")

    # ---- save --------------------------------------------------------------
    print("\n[4] Saving outputs ...")
    labels = df[GHS_LABEL_COLUMNS].astype(np.int8)

    # NumPy binaries: fast to load and compact - used by every later step.
    np.save(os.path.join(DIR_FEATURES, "STEP4_X.npy"), X_filtered)
    np.save(os.path.join(DIR_FEATURES, "STEP4_y.npy"), labels.to_numpy())

    # CSV copies, as the proposal requires. Written with float_format to keep
    # the file size reasonable - full float32 precision is preserved in the
    # .npy files above.
    feature_frame = pd.DataFrame(X_filtered, columns=kept_names)
    feature_frame.insert(0, "CID", df["CID"].values)
    feature_frame.to_csv(stamped("STEP4_feature_matrix.csv"),
                         index=False, float_format="%.5g")

    label_frame = labels.copy()
    label_frame.insert(0, "CID", df["CID"].values)
    label_frame.to_csv(stamped("STEP4_label_matrix.csv"), index=False)

    with open(stamped("STEP4_feature_names.txt"), "w", encoding="utf-8") as fh:
        fh.write("# Feature names retained after VarianceThreshold(0.01)\n")
        fh.write(f"# {len(kept_names)} features, in matrix column order\n")
        for name in kept_names:
            fh.write(name + "\n")

    # The compound index keeps CIDs, names and structures aligned with the
    # feature-matrix rows, so Steps 5, 10 and 11 can look compounds back up.
    index_columns = [c for c in ["CID", "Name", "SMILES", "CanonicalSMILES_RDKit",
                                 "InChIKey_RDKit", "MolecularFormula", "CAS"]
                     if c in df.columns]
    df[index_columns + GHS_LABEL_COLUMNS].to_csv(
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"), index=False)

    metadata = {
        "n_compounds": int(X_filtered.shape[0]),
        "n_features_computed": int(len(feature_names)),
        "n_features_after_variance_filter": int(X_filtered.shape[1]),
        "n_features_removed": n_removed,
        "removed_by_family": family_counts,
        "n_missing_values_imputed": n_nan,
        "n_molecules_failed": len(failed_indices),
        "variance_threshold": 0.01,
        "morgan_radius": MORGAN_RADIUS,
        "morgan_bits": MORGAN_BITS,
        "maccs_bits": MACCS_BITS,
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(stamped("STEP4_descriptor_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP4_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 4 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else
                 "No issues. RDKit succeeded; mordred and PaDEL fallbacks "
                 "were implemented but not needed.\n")

    print("\n" + "=" * 78)
    print("STEP 4 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Computed 19 physicochemical, 1024 Morgan (ECFP4),")
    print("                167 MACCS and 8 topological descriptors for every")
    print("                compound, imputed missing values, and removed")
    print("                near-constant descriptors.")
    print(f"FEATURE MATRIX: {X_filtered.shape[0]:,} compounds "
          f"x {X_filtered.shape[1]:,} features")
    print(f"                (from {len(feature_names):,} computed; "
          f"{n_removed:,} removed by variance filter)")
    print(f"NaN IMPUTED   : {n_nan:,} values")
    print(f"LABEL MATRIX  : {labels.shape[0]:,} x {labels.shape[1]} "
          f"(the nine GHS classes)")
    print(f"OUTPUT FILES  : {stamped('STEP4_feature_matrix.csv')}")
    print(f"                {stamped('STEP4_label_matrix.csv')}")
    print(f"                {stamped('STEP4_feature_names.txt')}")
    print(f"                {os.path.join(DIR_FEATURES, 'STEP4_X.npy')}")
    print(f"                {os.path.join(DIR_FEATURES, 'STEP4_y.npy')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged")
    print(f"ELAPSED       : {metadata['elapsed_seconds'] / 60:.1f} minutes")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
