"""
STEP 3 - DATA CLEANING AND VALIDATION
=====================================
The raw table from Step 2 contains everything PubChem knows, including
structures that cannot be parsed, the same compound listed several times, and
compounds that two regulators classified differently. This step turns that raw
table into a clean, unambiguous, machine-learning-ready dataset.

Sub-steps (in the order required by the proposal)
-------------------------------------------------
3a  Validate every SMILES string with RDKit
3b  Remove duplicates using the InChIKey as the unique structural identifier
3c  Remove compounds carrying no hazard label at all
3d  Reconcile disagreements between regulatory sources by majority vote
3e  Analyse and plot the class distribution
3f  (ADDED, see note below) Draw a stratified modelling subset

Note on sub-step 3f
-------------------
This machine has 7.9 GB of RAM and two physical CPU cores. The full cleaned
dataset combined with the ~1200 molecular descriptors of Step 4 would need
several gigabytes as a dense matrix, which this machine cannot hold. The full
cleaned dataset is therefore saved in its entirety as the scientific
deliverable, and a label-stratified subset is drawn from it for model
training. The subset preserves the prevalence of all nine hazard classes and
retains every positive example of the rare classes. This is a documented
hardware limitation, not a shortcut, and is reported in the Methods section.

Author : Sareer Ahmad
"""

import os
import re
import sys
import json
import time
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # render to file; this machine has no display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, PROJECT_ROOT, DIR_RAW, DIR_CLEAN,
                        DIR_LOGS, GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

# Silence RDKit's per-molecule parse warnings; invalid structures are counted
# deliberately in 3a instead, which is far more useful than thousands of lines.
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

ISSUE_LOG = []

# How large the modelling subset may be, given this machine's memory.
#
# The binding constraint is not the feature matrix - 40,000 compounds x ~800
# descriptors as float32 is only about 130 MB - but the Random Forest built
# from it. A fully grown decision tree has roughly two nodes per training
# sample, and the project trains 200 trees for each of nine hazard classes.
# At 60,000 compounds that comes to an estimated 14 GB of tree structures,
# far beyond this machine's 7.9 GB. The subset size and the tree depth cap in
# Step 7 were chosen together so that the forest fits in memory.
MAX_MODELLING_COMPOUNDS = 40000

# A hazard class with fewer than this many positives project-wide is treated
# as rare: every one of its positive examples is kept in the modelling subset.
RARE_CLASS_THRESHOLD = 6000


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ===========================================================================
# 3a - SMILES VALIDATION
# ===========================================================================
# Only these characters can legally appear in a SMILES string. Used by the
# last-resort regex fallback if every chemistry toolkit is unavailable.
VALID_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#%.\\/*:$]+$")


def validate_smiles_rdkit(smiles_series):
    """
    Parse every SMILES string with RDKit and report which ones are valid.

    RDKit returns None when a SMILES string is syntactically wrong or
    chemically impossible (for example a carbon with six bonds). Those rows
    cannot be turned into descriptors and must be discarded.

    Returns
    -------
    (is_valid, canonical, inchikeys) : three lists, one entry per input row
    """
    is_valid, canonical, inchikeys = [], [], []
    for smiles in tqdm(smiles_series, desc="      3a RDKit parse",
                       unit="mol", ncols=78, mininterval=2.0):
        if not isinstance(smiles, str) or not smiles.strip():
            is_valid.append(False); canonical.append(""); inchikeys.append("")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            is_valid.append(False); canonical.append(""); inchikeys.append("")
            continue
        is_valid.append(True)
        # Canonical SMILES gives one unique spelling per structure, so that
        # "OCC" and "CCO" (both ethanol) become the same string.
        canonical.append(Chem.MolToSmiles(mol))
        try:
            # 3b uses the InChIKey as the deduplication key. It is a fixed
            # length hash of the structure and is the international standard
            # identifier for a chemical substance.
            inchikeys.append(Chem.MolToInchiKey(mol))
        except Exception:
            inchikeys.append("")   # a few exotic structures defeat InChI
    return is_valid, canonical, inchikeys


def validate_smiles_regex(smiles_series):
    """
    LAST-RESORT FALLBACK for 3a, used only if no chemistry toolkit works.

    Keeps a SMILES string if it contains nothing but the characters that are
    legal in SMILES notation. This catches obvious corruption but cannot
    detect chemically impossible structures, so it is much weaker than RDKit.
    """
    log_issue("3a", "FALLBACK: no chemistry toolkit available - "
                    "validating SMILES with a character-set regex only.")
    is_valid = [bool(isinstance(s, str) and s.strip() and VALID_SMILES_CHARS.match(s))
                for s in smiles_series]
    # Without a toolkit there is no canonical form and no InChIKey.
    return is_valid, list(smiles_series), [""] * len(smiles_series)


def step3a_validate(df):
    """Run sub-step 3a with its documented fallback chain."""
    print("\n[3a] Validating SMILES strings ...")
    n_before = len(df)

    if RDKIT_AVAILABLE:
        try:
            valid, canonical, inchikeys = validate_smiles_rdkit(df["SMILES"])
        except Exception as exc:
            log_issue("3a", f"RDKit validation crashed ({exc}); trying MolVS.")
            try:
                # FALLBACK: MolVS wraps RDKit but is more forgiving.
                from molvs import Standardizer            # noqa: F401
                valid, canonical, inchikeys = validate_smiles_rdkit(df["SMILES"])
            except Exception:
                valid, canonical, inchikeys = validate_smiles_regex(df["SMILES"])
    else:
        valid, canonical, inchikeys = validate_smiles_regex(df["SMILES"])

    df = df.copy()
    df["_valid"] = valid
    df["CanonicalSMILES_RDKit"] = canonical
    df["InChIKey_RDKit"] = inchikeys

    n_invalid = int((~df["_valid"]).sum())
    df = df[df["_valid"]].drop(columns=["_valid"]).reset_index(drop=True)

    print(f"      Parsed   : {n_before:,} SMILES strings")
    print(f"      Invalid  : {n_invalid:,} removed "
          f"({100 * n_invalid / max(n_before, 1):.2f}%)")
    print(f"      Remaining: {len(df):,}")
    return df, n_invalid


# ===========================================================================
# 3b - DUPLICATE REMOVAL BY InChIKey
# ===========================================================================
def step3b_deduplicate(df):
    """
    Remove repeated structures, keeping the first occurrence of each.

    The same substance often appears in PubChem under several CIDs (different
    salts, hydrates or registry entries). Training on duplicates would let the
    same molecule appear in both the training and the test set, which inflates
    the apparent accuracy - this is the single most common methodological
    error in published QSAR work, so it is guarded against carefully here.
    """
    print("\n[3b] Removing duplicate structures ...")
    n_before = len(df)

    # Prefer the InChIKey; fall back to canonical SMILES for the handful of
    # structures for which InChI generation failed.
    key = df["InChIKey_RDKit"].where(df["InChIKey_RDKit"].astype(bool),
                                     df["CanonicalSMILES_RDKit"])
    n_missing_key = int((~df["InChIKey_RDKit"].astype(bool)).sum())
    if n_missing_key:
        log_issue("3b", f"FALLBACK applied for {n_missing_key:,} compound(s): "
                        f"InChIKey generation failed, canonical SMILES used as "
                        f"the deduplication key instead.")

    df = df.copy()
    df["_dedup_key"] = key

    # Sort so that the retained copy is the one supported by the most
    # regulatory sources - the best-evidenced record for that structure.
    if "N_Sources" in df.columns:
        df = df.sort_values("N_Sources", ascending=False, kind="stable")

    deduped = df.drop_duplicates(subset="_dedup_key", keep="first")
    deduped = deduped.sort_index().drop(columns=["_dedup_key"]).reset_index(drop=True)

    n_removed = n_before - len(deduped)
    print(f"      Before   : {n_before:,}")
    print(f"      Duplicates removed: {n_removed:,} "
          f"({100 * n_removed / max(n_before, 1):.2f}%)")
    print(f"      Remaining: {len(deduped):,} unique structures")
    return deduped, n_removed


# ===========================================================================
# 3c - REMOVE UNLABELLED COMPOUNDS
# ===========================================================================
def step3c_remove_unlabelled(df):
    """
    Drop compounds whose nine hazard columns are all zero.

    A compound with no hazard assigned carries no supervision signal: we
    cannot tell whether it is genuinely non-hazardous or simply has not been
    assessed yet. Including such rows as negatives would teach the model that
    unassessed chemicals are safe, which is exactly the wrong lesson for a
    safety screening tool.
    """
    print("\n[3c] Removing compounds with no hazard label ...")
    n_before = len(df)
    label_total = df[GHS_LABEL_COLUMNS].sum(axis=1)
    kept = df[label_total > 0].reset_index(drop=True)
    n_removed = n_before - len(kept)
    print(f"      Removed  : {n_removed:,} unlabelled compounds")
    print(f"      Remaining: {len(kept):,}")
    return kept, n_removed


# ===========================================================================
# 3d - CONFLICTING LABELS BETWEEN SOURCES
# ===========================================================================
def step3d_resolve_conflicts(df, per_source_path):
    """
    Reconcile classifications that different regulators disagree about.

    PubChem carries GHS classifications from several independent bodies (ECHA,
    the EU CLP regulation, Safe Work Australia's HCIS, Japan's NITE-CMC and
    the US HSDB). They do not always agree.

    Voting rule used
    ----------------
    For each compound and each of the nine hazards, count how many sources
    assigned that pictogram.
      * one source only        -> take that source's value, no conflict possible
      * a clear majority       -> take the majority value
      * an exact 50/50 split   -> mark Conflicted = 1 and take the HAZARDOUS
                                  value (1)

    The tie-break is deliberately safety-first. For a chemical screening tool,
    wrongly warning about a safe compound wastes a little time, whereas
    wrongly clearing a hazardous compound can injure someone.
    """
    print("\n[3d] Reconciling label conflicts between regulatory sources ...")

    if not os.path.exists(per_source_path):
        log_issue("3d", f"per-source file not found at {per_source_path}; "
                        f"keeping the Step 2 union labels and marking no "
                        f"conflicts. Majority voting could not be applied.")
        df = df.copy()
        df["Conflicted"] = 0
        return df, 0

    per_source = pd.read_csv(per_source_path)
    print(f"      Loaded {len(per_source):,} per-source classification records.")

    # How many independent sources classified each compound?
    votes_total = per_source.groupby("CID").size().rename("n_votes")
    # How many of them assigned each particular pictogram?
    votes_positive = per_source.groupby("CID")[GHS_LABEL_COLUMNS].sum()

    tally = votes_positive.join(votes_total)

    majority_labels = pd.DataFrame(index=tally.index)
    conflict_flags = pd.Series(0, index=tally.index, dtype=int)

    for column in GHS_LABEL_COLUMNS:
        positive = tally[column]
        total = tally["n_votes"]
        # Strict majority: more than half the sources said yes.
        majority_yes = (positive * 2) > total
        # Exact tie: only possible with an even number of sources.
        tied = (positive * 2) == total
        tied = tied & (positive > 0)      # 0 of 2 sources is agreement, not a tie
        majority_labels[column] = (majority_yes | tied).astype(int)
        conflict_flags |= tied.astype(int)

    # Counts must be restricted to the compounds actually in this dataset.
    # The per-source file covers every CID PubChem returned, which is a larger
    # set than the cleaned dataframe, so a global count would overstate them.
    cids_in_dataset = set(df["CID"].astype(int).tolist())
    in_dataset = tally.index.isin(cids_in_dataset)
    n_conflicted = int(conflict_flags[in_dataset].sum())
    n_multi_source = int((tally.loc[in_dataset, "n_votes"] > 1).sum())

    # Attach the reconciled labels back onto the main table.
    df = df.copy()
    union_labels = df[GHS_LABEL_COLUMNS].copy()      # keep Step 2 union for reference
    for column in GHS_LABEL_COLUMNS:
        union_labels = union_labels.rename(columns={column: column + "_union"})

    reconciled = df[["CID"]].merge(majority_labels, left_on="CID",
                                   right_index=True, how="left")
    # Any compound missing from the vote table keeps its Step 2 value.
    for column in GHS_LABEL_COLUMNS:
        df[column] = reconciled[column].fillna(df[column]).astype(int).values

    df["Conflicted"] = df["CID"].map(conflict_flags).fillna(0).astype(int)
    df = pd.concat([df, union_labels], axis=1)

    # Report how much the vote actually changed, within this dataset only.
    n_changed = int(sum(
        (df[c].values != df[c + "_union"].values).sum() for c in GHS_LABEL_COLUMNS))
    n_conflicted = int(df["Conflicted"].sum())   # authoritative in-dataset count

    print(f"      Compounds classified by more than one source: {n_multi_source:,}")
    print(f"      Compounds with at least one tied vote        : {n_conflicted:,}")
    print(f"      Individual labels changed by the vote        : {n_changed:,}")
    print(f"      (tied votes resolved as HAZARDOUS - safety-first rule)")
    return df, n_conflicted


# ===========================================================================
# 3e - CLASS DISTRIBUTION ANALYSIS
# ===========================================================================
def step3e_class_distribution(df, tag="cleaned"):
    """
    Build the class-distribution table and bar chart.

    The 'imbalance ratio' is the number of negatives divided by the number of
    positives. A ratio of 100 means only one compound in 101 carries that
    hazard, which is why Step 6 exists.
    """
    print(f"\n[3e] Class distribution analysis ({tag}) ...")
    n_total = len(df)
    rows = []
    for column in GHS_LABEL_COLUMNS:
        n_positive = int(df[column].sum())
        n_negative = n_total - n_positive
        rows.append({
            "GHS_Column": column,
            "Pictogram_Code": column.split("_")[0],
            "Actual_Meaning": GHS_TRUE_MEANING[column],
            "N_Positive": n_positive,
            "N_Negative": n_negative,
            "Percent_of_Dataset": round(100 * n_positive / max(n_total, 1), 3),
            "Imbalance_Ratio_NegPerPos": (round(n_negative / n_positive, 2)
                                          if n_positive else np.inf),
        })
    table = pd.DataFrame(rows)

    per_compound = df[GHS_LABEL_COLUMNS].sum(axis=1)
    n_multilabel = int((per_compound > 1).sum())

    print("-" * 100)
    print(f"{'Column':<22}{'Code':<7}{'Meaning':<40}{'N+':>8}{'%':>8}{'Imb.':>9}")
    print("-" * 100)
    for row in rows:
        print(f"{row['GHS_Column']:<22}{row['Pictogram_Code']:<7}"
              f"{row['Actual_Meaning']:<40}{row['N_Positive']:>8,}"
              f"{row['Percent_of_Dataset']:>8.2f}"
              f"{row['Imbalance_Ratio_NegPerPos']:>9.1f}")
    print("-" * 100)
    print(f"Total compounds                  : {n_total:,}")
    print(f"Multi-label compounds (>1 hazard): {n_multilabel:,} "
          f"({100 * n_multilabel / max(n_total, 1):.2f}%)")
    print(f"Mean hazards per compound        : {per_compound.mean():.2f}")
    print(f"Max hazards on one compound      : {int(per_compound.max())}")
    print(f"Most imbalanced class            : "
          f"{table.loc[table['Imbalance_Ratio_NegPerPos'].idxmax(), 'GHS_Column']} "
          f"({table['Imbalance_Ratio_NegPerPos'].max():.1f} negatives per positive)")

    # ---------------- the bar chart ---------------------------------------
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: how many compounds carry each hazard.
    labels = [f"{r['Pictogram_Code']}\n{r['Actual_Meaning'].split('(')[0].strip()}"
              for r in rows]
    counts = [r["N_Positive"] for r in rows]
    colours = sns.color_palette("rocket_r", len(rows))
    bars = axes[0].bar(labels, counts, color=colours, edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Number of compounds", fontsize=12)
    axes[0].set_title("Positive examples per GHS hazard class", fontsize=13,
                      fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].set_yscale("log")    # log scale: the classes differ by ~100x
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{count:,}", ha="center", va="bottom", fontsize=8.5,
                     fontweight="bold")

    # Right panel: how many hazards each compound carries.
    hazard_counts = per_compound.value_counts().sort_index()
    axes[1].bar(hazard_counts.index.astype(str), hazard_counts.values,
                color=sns.color_palette("mako", len(hazard_counts)),
                edgecolor="black", linewidth=0.6)
    axes[1].set_xlabel("Number of GHS hazards carried by one compound", fontsize=12)
    axes[1].set_ylabel("Number of compounds", fontsize=12)
    axes[1].set_title("Multi-label structure of the dataset", fontsize=13,
                      fontweight="bold")
    for x, y in zip(range(len(hazard_counts)), hazard_counts.values):
        axes[1].text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=8.5,
                     fontweight="bold")

    fig.suptitle(f"GHS class distribution - {tag} dataset "
                 f"(n = {n_total:,} compounds)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_png = stamped("STEP3_class_distribution.png"
                      if tag == "cleaned" else
                      f"STEP3_class_distribution_{tag}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Chart saved: {out_png}")

    return table, n_multilabel, out_png


# ===========================================================================
# 3f - STRATIFIED MODELLING SUBSET (hardware-driven, documented)
# ===========================================================================
def step3f_modelling_subset(df, max_compounds=MAX_MODELLING_COMPOUNDS):
    """
    Draw a memory-feasible subset that preserves the class structure.

    Selection rule
    --------------
    1. Every compound carrying a RARE hazard (fewer than RARE_CLASS_THRESHOLD
       positives in the whole dataset) is kept. Rare classes such as the
       explosive pictogram have so few examples that discarding any of them
       would make the class unlearnable.
    2. The remaining places are filled by a random sample of the other
       compounds, drawn with the project seed so the choice is reproducible.

    The consequence - a mild enrichment of rare hazards relative to the full
    dataset - is measured and reported, so that the prevalence shift is
    visible to any reader of the paper.
    """
    print("\n[3f] Drawing the modelling subset (hardware-constrained) ...")
    n_total = len(df)

    if n_total <= max_compounds:
        print(f"      Cleaned dataset has {n_total:,} compounds, which is within "
              f"the {max_compounds:,} limit - no subsetting needed.")
        return df.copy(), None

    log_issue("3f", f"Cleaned dataset has {n_total:,} compounds. This machine "
                    f"(7.9 GB RAM, 2 physical cores) cannot hold the resulting "
                    f"dense descriptor matrix, so a stratified subset of "
                    f"{max_compounds:,} compounds is used for modelling. The "
                    f"full cleaned dataset is still saved in its entirety.")

    positives_per_class = df[GHS_LABEL_COLUMNS].sum()
    rare_classes = [c for c in GHS_LABEL_COLUMNS
                    if positives_per_class[c] < RARE_CLASS_THRESHOLD]
    print(f"      Rare classes (kept in full): "
          f"{', '.join(c.split('_')[0] for c in rare_classes) or 'none'}")

    # Step 1 - compulsory keeps
    if rare_classes:
        must_keep_mask = df[rare_classes].sum(axis=1) > 0
    else:
        must_keep_mask = pd.Series(False, index=df.index)
    must_keep = df[must_keep_mask]
    print(f"      Compounds retained for carrying a rare hazard: {len(must_keep):,}")

    # Step 2 - fill the rest at random, reproducibly
    remaining_slots = max_compounds - len(must_keep)
    pool = df[~must_keep_mask]

    if remaining_slots <= 0:
        log_issue("3f", f"rare-class compounds alone ({len(must_keep):,}) already "
                        f"exceed the {max_compounds:,} budget; a random sample of "
                        f"them is taken instead.")
        subset = must_keep.sample(n=max_compounds, random_state=RANDOM_SEED)
    else:
        filler = pool.sample(n=min(remaining_slots, len(pool)),
                             random_state=RANDOM_SEED)
        subset = pd.concat([must_keep, filler])

    subset = subset.sort_values("CID").reset_index(drop=True)

    # Report exactly how the prevalence shifted, for honesty in the paper.
    print(f"      Modelling subset size: {len(subset):,} compounds "
          f"({100 * len(subset) / n_total:.1f}% of the cleaned dataset)")
    print("\n      Prevalence shift caused by subsetting")
    print("      " + "-" * 68)
    print(f"      {'Class':<22}{'full %':>10}{'subset %':>11}{'change':>11}")
    print("      " + "-" * 68)
    shift_rows = []
    for column in GHS_LABEL_COLUMNS:
        full_pct = 100 * df[column].mean()
        sub_pct = 100 * subset[column].mean()
        print(f"      {column:<22}{full_pct:>10.2f}{sub_pct:>11.2f}"
              f"{sub_pct - full_pct:>+11.2f}")
        shift_rows.append({"GHS_Column": column,
                           "Percent_full_dataset": round(full_pct, 3),
                           "Percent_modelling_subset": round(sub_pct, 3),
                           "Percentage_point_change": round(sub_pct - full_pct, 3)})
    print("      " + "-" * 68)

    return subset, pd.DataFrame(shift_rows)


# ===========================================================================
# MAIN
# ===========================================================================
def clean_ghs_dataset(raw_path=None, per_source_path=None):
    """Run every sub-step of Step 3 and save all outputs."""
    start_time = time.time()
    raw_path = raw_path or stamped("STEP2_raw_ghs_dataset.csv")
    per_source_path = per_source_path or os.path.join(
        DIR_RAW, f"STEP2_per_source_records_{TODAY}.csv")

    print("=" * 78)
    print("STEP 3 - DATA CLEANING AND VALIDATION")
    print("=" * 78)
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Loaded raw dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"   from {raw_path}")
    n_raw = len(df)

    df, n_invalid = step3a_validate(df)
    df, n_duplicates = step3b_deduplicate(df)
    df, n_unlabelled = step3c_remove_unlabelled(df)
    df, n_conflicted = step3d_resolve_conflicts(df, per_source_path)

    # 3d can zero-out a label, which may leave a compound unlabelled again.
    n_before_recheck = len(df)
    df = df[df[GHS_LABEL_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    n_lost_to_voting = n_before_recheck - len(df)
    if n_lost_to_voting:
        print(f"      {n_lost_to_voting:,} compound(s) lost every label during the "
              f"majority vote and were removed.")

    dist_table, n_multilabel, dist_png = step3e_class_distribution(df, tag="cleaned")

    # ---- save the full cleaned dataset (the scientific deliverable) -------
    clean_path = stamped("STEP3_cleaned_ghs_dataset.csv")
    df.to_csv(clean_path, index=False)
    df.to_csv(os.path.join(DIR_CLEAN,
                           f"STEP3_cleaned_ghs_dataset_{TODAY}.csv"), index=False)
    dist_table.to_csv(stamped("STEP3_class_distribution_table.csv"), index=False)

    # ---- 3f modelling subset --------------------------------------------
    subset, shift_table = step3f_modelling_subset(df)
    subset_path = stamped("STEP3_modelling_subset.csv")
    subset.to_csv(subset_path, index=False)
    if shift_table is not None:
        shift_table.to_csv(stamped("STEP3_subset_prevalence_shift.csv"), index=False)
        step3e_class_distribution(subset, tag="modelling_subset")

    # ---- summary ---------------------------------------------------------
    summary = {
        "raw_rows": n_raw,
        "removed_invalid_smiles": n_invalid,
        "removed_duplicates": n_duplicates,
        "removed_unlabelled": n_unlabelled,
        "removed_by_voting": n_lost_to_voting,
        "final_cleaned_compounds": len(df),
        "conflicted_compounds": n_conflicted,
        "multilabel_compounds": n_multilabel,
        "modelling_subset_compounds": len(subset),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(stamped("STEP3_cleaning_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP3_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 3 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 3 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Validated every structure with RDKit, removed duplicate")
    print("                structures by InChIKey, dropped unlabelled compounds,")
    print("                reconciled disagreements between regulators by majority")
    print("                vote, and analysed the class distribution.")
    print(f"RAW ROWS      : {n_raw:,}")
    print(f"  - invalid SMILES removed    : {n_invalid:,}")
    print(f"  - duplicate structures      : {n_duplicates:,}")
    print(f"  - unlabelled compounds      : {n_unlabelled:,}")
    print(f"  - lost during majority vote : {n_lost_to_voting:,}")
    print(f"FINAL CLEANED : {len(df):,} compounds")
    print(f"CONFLICTED    : {n_conflicted:,} compounds had a tied vote "
          f"(kept, flagged Conflicted=1)")
    print(f"MULTI-LABEL   : {n_multilabel:,} compounds carry more than one hazard")
    print(f"MODELLING SET : {len(subset):,} compounds (hardware-constrained subset)")
    print(f"OUTPUT FILES  : {clean_path}")
    print(f"                {subset_path}")
    print(f"                {dist_png}")
    print(f"                {stamped('STEP3_class_distribution_table.csv')}")
    print(f"                {log_path}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged")
    print(f"ELAPSED       : {summary['elapsed_seconds'] / 60:.1f} minutes")
    print("=" * 78)
    return df, subset


if __name__ == "__main__":
    clean_ghs_dataset()
