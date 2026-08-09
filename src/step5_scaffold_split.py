"""
STEP 5 - SCAFFOLD-BASED DATASET SPLITTING
=========================================
Why not a simple random split?
------------------------------
Chemical datasets contain families of closely related molecules. If a random
split puts one member of a family in the training set and its near-twin in the
test set, the model appears far more accurate than it really is - it has
effectively seen the test compound already. A model validated that way would
fail the moment it met a genuinely new chemical, which is exactly the
situation a hazard-screening tool must handle.

Bemis-Murcko scaffold splitting prevents this. The "scaffold" of a molecule is
its ring system plus the linkers joining those rings, with all the decorating
side chains stripped away. Compounds sharing a scaffold are forced into the
same split, so the test set contains only chemical skeletons the model has
never encountered.

Reference: Bemis, G. W.; Murcko, M. A. J. Med. Chem. 1996, 39 (15), 2887-2893.

Handling of acyclic molecules (documented deviation)
----------------------------------------------------
A molecule with no rings - hexane, methanol, acetic acid - has an empty
Murcko scaffold. The textbook implementation places every acyclic compound in
one enormous group, which for this dataset would shove a large fraction of all
industrial solvents into a single split and leave the other splits with no
acyclic chemistry at all. Each acyclic molecule is therefore treated as its
own scaffold group. The number of compounds affected is reported below.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time
import collections
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, DIR_SPLITS, DIR_FEATURES, DIR_LOGS,
                        GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

ISSUE_LOG = []

# Split proportions required by the proposal.
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ===========================================================================
# 5a - COMPUTE THE BEMIS-MURCKO SCAFFOLD
# ===========================================================================
def compute_scaffold(smiles):
    """
    Return the Bemis-Murcko scaffold of one molecule as a SMILES string.

    An empty result means the molecule has no rings. In that case the
    molecule's own canonical SMILES is returned instead, so that it forms a
    scaffold group of its own (see the module docstring).

    Returns
    -------
    (scaffold_smiles, is_acyclic)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
    except Exception:
        return None, False

    if not scaffold_smiles:            # no rings at all
        return Chem.MolToSmiles(mol), True
    return scaffold_smiles, False


def compute_all_scaffolds(smiles_series):
    """Compute scaffolds for the whole dataset with a progress bar."""
    print("\n[5a] Computing Bemis-Murcko scaffolds ...")
    scaffolds, acyclic_flags = [], []
    for smiles in tqdm(smiles_series, desc="      scaffolds", unit="mol",
                       ncols=78, mininterval=2.0):
        scaffold, is_acyclic = compute_scaffold(smiles)
        scaffolds.append(scaffold)
        acyclic_flags.append(is_acyclic)

    n_failed = sum(1 for s in scaffolds if s is None)
    n_acyclic = sum(acyclic_flags)
    if n_failed:
        log_issue("5a", f"{n_failed:,} molecule(s) gave no scaffold; each was "
                        f"given a unique placeholder group so it stays in the "
                        f"dataset without leaking across splits.")
        scaffolds = [s if s is not None else f"__unparsed_{i}__"
                     for i, s in enumerate(scaffolds)]

    print(f"      {len(set(scaffolds)):,} distinct scaffolds "
          f"for {len(scaffolds):,} compounds")
    print(f"      {n_acyclic:,} compounds are acyclic "
          f"({100 * n_acyclic / max(len(scaffolds), 1):.1f}%) and each forms "
          f"its own group")
    return scaffolds, n_acyclic


# ===========================================================================
# 5b + 5c - GROUP BY SCAFFOLD AND ASSIGN TO SPLITS
# ===========================================================================
def scaffold_split(scaffolds, y, train_fraction=TRAIN_FRACTION,
                   val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """
    Assign whole scaffold groups to the training, validation and test sets,
    balancing both the overall split sizes and the nine hazard classes.

    Why this is not a simple quota fill
    -----------------------------------
    Two earlier versions of this function both failed, in instructive ways.

    The first filled the training set to 80%, then validation to 10%, then
    put everything else in test. Because groups are processed largest first,
    a single huge scaffold group met after the training quota was full could
    not fit in validation either and fell through to test - producing an
    80/3/17 split instead of 80/10/10.

    The second assigned each group to whichever split was furthest below its
    quota by count. The overall ratios came out exactly right, but the rare
    classes were still starved: only 31% of compressed gases reached the
    training set instead of 80%. The reason is that ring-bearing scaffolds
    form the large groups and fill the training quota early, after which the
    three splits have equal remaining capacity and the single-compound groups
    are shared out roughly one-third each. Every acyclic molecule is a
    single-compound group, and the rare hazard classes are overwhelmingly
    small acyclic molecules - compressed gases are methane, nitrogen,
    hydrogen sulfide and the like. The splitting algorithm was quietly
    depriving the model of the examples it had fewest of.

    What this version does
    ----------------------
    Each group is scored against every split on how far the assignment would
    push that split past its target - measured both on overall size AND on
    each hazard class the group contains - and goes to the split with the
    lowest worst-case overshoot. This is the group-wise form of iterative
    stratification for multi-label data, and it balances the classes and the
    split sizes at the same time.

    Groups carrying rare-class compounds are placed first, while all three
    splits still have room, since those are the assignments that matter most.

    Returns
    -------
    (train_indices, val_indices, test_indices) : three numpy integer arrays
    """
    print("\n[5b-5c] Grouping by scaffold and assigning to splits ...")

    # Map each scaffold to the row numbers of the compounds that share it.
    groups = collections.defaultdict(list)
    for row_index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(row_index)

    group_list = list(groups.values())
    n_total = len(scaffolds)

    # Shuffle the groups reproducibly, as required by sub-step 5c.
    rng = np.random.RandomState(seed)
    rng.shuffle(group_list)

    test_fraction = 1.0 - train_fraction - val_fraction
    fractions = {"train": train_fraction, "val": val_fraction,
                 "test": test_fraction}

    y = np.asarray(y)
    n_classes = y.shape[1]

    # How many positives of each class each group carries, and how rare the
    # rarest class it touches is. Groups holding rare-class compounds are
    # placed first, while every split still has room for them.
    group_positives = [y[g].sum(axis=0) for g in group_list]
    class_totals = y.sum(axis=0)
    rarity = []
    for positives in group_positives:
        present = np.where(positives > 0)[0]
        # Smaller number = touches a rarer class = place this group earlier.
        rarity.append(class_totals[present].min() if len(present)
                      else class_totals.max() + 1)

    order = sorted(range(len(group_list)),
                   key=lambda i: (rarity[i], -len(group_list[i])))

    # Targets: overall size, and the number of positives of each class that
    # each split should ideally receive.
    target_size = {s: fractions[s] * n_total for s in fractions}
    target_class = {s: fractions[s] * class_totals for s in fractions}

    count_size = {s: 0 for s in fractions}
    count_class = {s: np.zeros(n_classes) for s in fractions}
    buckets = {s: [] for s in fractions}

    for i in order:
        group, positives = group_list[i], group_positives[i]
        best_split, best_score = None, None
        for split in ("train", "val", "test"):
            # Worst-case overshoot if this group went here: the largest
            # fraction-of-target across overall size and every class the
            # group actually contains. Lower is better.
            score = (count_size[split] + len(group)) / max(target_size[split], 1)
            for c in np.where(positives > 0)[0]:
                score = max(score, (count_class[split][c] + positives[c])
                            / max(target_class[split][c], 1e-9))
            if best_score is None or score < best_score:
                best_score, best_split = score, split
        buckets[best_split].extend(group)
        count_size[best_split] += len(group)
        count_class[best_split] += positives

    train_indices = buckets["train"]
    val_indices = buckets["val"]
    test_indices = buckets["test"]

    print(f"      scaffold groups : {len(group_list):,}")
    print(f"      train {len(train_indices):,} ({100 * len(train_indices) / n_total:.1f}%) | "
          f"val {len(val_indices):,} ({100 * len(val_indices) / n_total:.1f}%) | "
          f"test {len(test_indices):,} ({100 * len(test_indices) / n_total:.1f}%)")

    return (np.array(sorted(train_indices)), np.array(sorted(val_indices)),
            np.array(sorted(test_indices)))


def repair_empty_classes(y, scaffolds, train_idx, val_idx, test_idx):
    """
    Guarantee that every hazard class has positive examples in every split.

    A purely random shuffle can, by bad luck, send all the examples of a rare
    class into one split. A class with no positives in the training set cannot
    be learned, and a class with none in the test set cannot be evaluated.

    The repair moves whole scaffold groups - never individual compounds - from
    the over-supplied split to the starved one, so the no-shared-scaffold
    guarantee is preserved.
    """
    print("\n[5d] Checking that every class has positives in every split ...")

    # scaffold -> which split it currently sits in
    split_of_row = {}
    for row in train_idx:
        split_of_row[row] = "train"
    for row in val_idx:
        split_of_row[row] = "val"
    for row in test_idx:
        split_of_row[row] = "test"

    groups = collections.defaultdict(list)
    for row_index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(row_index)

    splits = {"train": set(train_idx.tolist()),
              "val": set(val_idx.tolist()),
              "test": set(test_idx.tolist())}

    n_repairs = 0
    # Minimum positives we insist on before declaring a split usable.
    MIN_POSITIVES = {"train": 5, "val": 1, "test": 2}

    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        for split_name, minimum in MIN_POSITIVES.items():
            current = int(y[sorted(splits[split_name]), class_index].sum())
            if current >= minimum:
                continue

            # Find donor groups: scaffold groups that contain positives of this
            # class and currently live in a split that can spare them.
            donors = []
            for scaffold, rows in groups.items():
                n_positive = int(y[rows, class_index].sum())
                if n_positive == 0:
                    continue
                home = split_of_row.get(rows[0])
                if home == split_name:
                    continue
                # Do not empty the donor split of this class.
                donor_total = int(y[sorted(splits[home]), class_index].sum())
                if donor_total - n_positive < MIN_POSITIVES.get(home, 1):
                    continue
                donors.append((len(rows), n_positive, scaffold, rows, home))

            # Prefer small groups carrying many positives - the cheapest move.
            donors.sort(key=lambda d: (d[0] / max(d[1], 1)))

            for _, n_positive, scaffold, rows, home in donors:
                if current >= minimum:
                    break
                for row in rows:
                    splits[home].discard(row)
                    splits[split_name].add(row)
                    split_of_row[row] = split_name
                current = int(y[sorted(splits[split_name]), class_index].sum())
                n_repairs += 1
                log_issue("5d", f"moved scaffold group '{scaffold[:40]}' "
                                f"({len(rows)} compounds) from {home} to "
                                f"{split_name} so that {column} has at least "
                                f"{minimum} positive example(s) there.")

            if current < minimum:
                log_issue("5d", f"{column} still has only {current} positive(s) "
                                f"in the {split_name} split - no scaffold group "
                                f"could be moved without starving another split. "
                                f"Metrics for this class in {split_name} must be "
                                f"read with caution.")

    if n_repairs == 0:
        print("      All nine classes already had positives in all three "
              "splits - no repair needed.")
    else:
        print(f"      {n_repairs} scaffold group(s) relocated to guarantee "
              f"class coverage.")

    return (np.array(sorted(splits["train"])), np.array(sorted(splits["val"])),
            np.array(sorted(splits["test"])))


# ===========================================================================
# 5d - VERIFY SPLIT QUALITY
# ===========================================================================
def verify_split(scaffolds, y, train_idx, val_idx, test_idx):
    """
    Prove that the split is sound, and print the class distribution per split.

    Three checks are run:
      1. No compound appears in two splits.
      2. No scaffold appears in two splits (the whole point of the exercise).
      3. Every one of the nine hazard classes has positive examples in both
         the training and the test set.
    """
    print("\n[5d] Verifying split quality ...")
    all_ok = True

    # ---- check 1: no compound in two splits -------------------------------
    sets = [set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())]
    overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
    if overlap:
        log_issue("5d", f"CHECK FAILED: {len(overlap)} compound(s) appear in "
                        f"more than one split.")
        all_ok = False
    else:
        print("      [PASS] no compound appears in more than one split")

    total_assigned = len(train_idx) + len(val_idx) + len(test_idx)
    if total_assigned != len(scaffolds):
        log_issue("5d", f"CHECK FAILED: {total_assigned:,} compounds assigned "
                        f"but the dataset has {len(scaffolds):,}.")
        all_ok = False
    else:
        print(f"      [PASS] all {len(scaffolds):,} compounds assigned exactly once")

    # ---- check 2: no scaffold shared between splits ------------------------
    scaffolds = np.asarray(scaffolds, dtype=object)
    train_scaffolds = set(scaffolds[train_idx].tolist())
    val_scaffolds = set(scaffolds[val_idx].tolist())
    test_scaffolds = set(scaffolds[test_idx].tolist())
    shared = ((train_scaffolds & val_scaffolds) | (train_scaffolds & test_scaffolds)
              | (val_scaffolds & test_scaffolds))
    if shared:
        log_issue("5d", f"CHECK FAILED: {len(shared)} scaffold(s) appear in more "
                        f"than one split - this would leak information.")
        all_ok = False
    else:
        print(f"      [PASS] no scaffold is shared between splits "
              f"(train {len(train_scaffolds):,} / val {len(val_scaffolds):,} / "
              f"test {len(test_scaffolds):,} distinct scaffolds)")

    # ---- check 3 + the per-split class distribution table ------------------
    print("\n      Class distribution per split")
    print("      " + "-" * 92)
    print(f"      {'Class':<22}{'meaning':<32}"
          f"{'train n+':>10}{'val n+':>9}{'test n+':>9}{'test %':>9}")
    print("      " + "-" * 92)

    rows = []
    starved = []
    train_share_target = 100 * len(train_idx) / max(
        len(train_idx) + len(val_idx) + len(test_idx), 1)
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        n_train = int(y[train_idx, class_index].sum())
        n_val = int(y[val_idx, class_index].sum())
        n_test = int(y[test_idx, class_index].sum())
        n_class_total = n_train + n_val + n_test
        # What share of this class's positives actually reached training? It
        # should be close to the overall training share. A class far below it
        # is being starved by the splitting algorithm, which is a bug in the
        # split - not a property of the data - and it silently cripples the
        # rare classes if nobody checks.
        train_share = 100 * n_train / max(n_class_total, 1)
        if n_class_total >= 20 and train_share < train_share_target - 15:
            starved.append((column, train_share, n_train, n_class_total))
        meaning = GHS_TRUE_MEANING[column].split("(")[0].strip()
        print(f"      {column:<22}{meaning:<32}{n_train:>10,}{n_val:>9,}"
              f"{n_test:>9,}{100 * n_test / max(len(test_idx), 1):>9.2f}")
        rows.append({
            "GHS_Column": column, "Meaning": GHS_TRUE_MEANING[column],
            "Train_Positives": n_train, "Val_Positives": n_val,
            "Test_Positives": n_test,
            "Train_Percent": round(100 * n_train / max(len(train_idx), 1), 3),
            "Val_Percent": round(100 * n_val / max(len(val_idx), 1), 3),
            "Test_Percent": round(100 * n_test / max(len(test_idx), 1), 3),
        })
        if n_train == 0 or n_test == 0:
            log_issue("5d", f"CHECK FAILED: {column} has "
                            f"{'no training' if n_train == 0 else 'no test'} "
                            f"positives.")
            all_ok = False
    print("      " + "-" * 92)

    # ---- check 4: is any class being starved of training data? -------------
    print(f"\n      Training share per class "
          f"(overall training share is {train_share_target:.1f}%)")
    print("      " + "-" * 68)
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        n_train = int(y[train_idx, class_index].sum())
        n_total_class = n_train + int(y[val_idx, class_index].sum()) + \
            int(y[test_idx, class_index].sum())
        share = 100 * n_train / max(n_total_class, 1)
        flag = "  <-- STARVED" if any(s[0] == column for s in starved) else ""
        print(f"      {column:<24}{n_train:>8,} of {n_total_class:>8,}"
              f"{share:>9.1f}%{flag}")
    print("      " + "-" * 68)

    if starved:
        # Deliberately a WARNING, not a failure. An earlier version treated
        # starvation as fatal, which sent the run to the stratified fallback -
        # and that fallback abandons scaffold grouping altogether, allowing
        # near-identical molecules into both training and test. Trading the
        # entire validity of the evaluation for a better class balance is a bad
        # bargain. The imbalance is reported loudly and the scaffold split is
        # kept.
        for column, share, n_train, n_total_class in starved:
            log_issue("5d", f"WARNING: only {share:.0f}% of {column} positives "
                            f"reached the training set ({n_train:,} of "
                            f"{n_total_class:,}), against an overall training "
                            f"share of {train_share_target:.0f}%. This class is "
                            f"under-represented in training and its performance "
                            f"will suffer. The scaffold split is KEPT - "
                            f"switching to a non-scaffold split would allow "
                            f"structural leakage, which is worse.")
    else:
        print("      [PASS] no class is starved of training data")

    if all_ok:
        print("\n      ALL SPLIT-QUALITY CHECKS PASSED")
    return pd.DataFrame(rows), all_ok


# ===========================================================================
# FALLBACKS
# ===========================================================================
def fallback_multilabel_stratified_split(y, seed=RANDOM_SEED):
    """
    FALLBACK if scaffold splitting cannot be performed.

    Uses iterative stratification, which keeps the proportion of every one of
    the nine labels roughly equal across the splits. It does not prevent
    scaffold leakage, so any result obtained this way is optimistic and must
    be labelled as such in the paper.
    """
    log_issue("5-FALLBACK", "Scaffold splitting failed - falling back to "
                            "MultilabelStratifiedShuffleSplit. NOTE: this does "
                            "NOT prevent scaffold leakage; results would be "
                            "optimistic and must be reported as such.")
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    n_samples = y.shape[0]
    all_indices = np.arange(n_samples)

    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                             random_state=seed)
    train_idx, holdout_idx = next(first.split(all_indices, y))

    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                              random_state=seed)
    val_rel, test_rel = next(second.split(holdout_idx, y[holdout_idx]))
    return (np.sort(train_idx), np.sort(holdout_idx[val_rel]),
            np.sort(holdout_idx[test_rel]))


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    """Run the whole of Step 5 and save the three index files."""
    start_time = time.time()
    print("=" * 78)
    print("STEP 5 - SCAFFOLD-BASED DATASET SPLITTING")
    print("=" * 78)

    compound_index = pd.read_csv(
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"), low_memory=False)
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy"))
    print(f"Loaded {len(compound_index):,} compounds and a "
          f"{y.shape[0]:,} x {y.shape[1]} label matrix")

    smiles_column = ("CanonicalSMILES_RDKit"
                     if "CanonicalSMILES_RDKit" in compound_index.columns
                     else "SMILES")

    # ---- 5a ----------------------------------------------------------------
    try:
        scaffolds, n_acyclic = compute_all_scaffolds(compound_index[smiles_column])
        used_fallback = False
    except Exception as exc:
        log_issue("5a", f"scaffold computation crashed: {exc}")
        scaffolds, n_acyclic, used_fallback = None, 0, True

    # ---- 5b, 5c ------------------------------------------------------------
    if scaffolds is not None:
        train_idx, val_idx, test_idx = scaffold_split(scaffolds, y)
        train_idx, val_idx, test_idx = repair_empty_classes(
            y, scaffolds, train_idx, val_idx, test_idx)
    else:
        train_idx, val_idx, test_idx = fallback_multilabel_stratified_split(y)
        scaffolds = [f"__fallback_{i}__" for i in range(len(compound_index))]
        used_fallback = True

    # ---- 5d ----------------------------------------------------------------
    split_table, all_ok = verify_split(scaffolds, y, train_idx, val_idx, test_idx)

    if not all_ok and not used_fallback:
        log_issue("5d", "scaffold split failed its quality checks - switching "
                        "to the stratified fallback split.")
        train_idx, val_idx, test_idx = fallback_multilabel_stratified_split(y)
        split_table, all_ok = verify_split(
            [f"__fallback_{i}__" for i in range(len(compound_index))],
            y, train_idx, val_idx, test_idx)
        used_fallback = True

    # ---- save --------------------------------------------------------------
    np.save(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"), train_idx)
    np.save(os.path.join(DIR_SPLITS, "STEP5_val_indices.npy"), val_idx)
    np.save(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"), test_idx)
    # Also at the project root, as the proposal's file list specifies.
    np.save(stamped("STEP5_train_indices.npy"), train_idx)
    np.save(stamped("STEP5_val_indices.npy"), val_idx)
    np.save(stamped("STEP5_test_indices.npy"), test_idx)

    split_table.to_csv(stamped("STEP5_split_class_distribution.csv"), index=False)

    # Label every compound with the split it landed in. This is done by direct
    # array assignment rather than by testing membership per compound: an
    # earlier version rebuilt the index sets inside the loop, which is fine for
    # a few thousand compounds but becomes about 5 x 10^10 operations at full
    # dataset size and effectively never finishes.
    split_labels = np.empty(len(compound_index), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    scaffold_frame = pd.DataFrame({
        "CID": compound_index["CID"].values,
        "Scaffold": scaffolds,
        "Split": split_labels,
    })
    scaffold_frame.to_csv(os.path.join(DIR_SPLITS,
                                       f"STEP5_scaffold_assignments_{TODAY}.csv"),
                          index=False)

    metadata = {
        "n_compounds": int(len(compound_index)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_fraction": round(len(train_idx) / len(compound_index), 4),
        "val_fraction": round(len(val_idx) / len(compound_index), 4),
        "test_fraction": round(len(test_idx) / len(compound_index), 4),
        "n_distinct_scaffolds": int(len(set(scaffolds))),
        "n_acyclic_compounds": int(n_acyclic),
        "split_method": ("MultilabelStratifiedShuffleSplit (FALLBACK)"
                         if used_fallback else "Bemis-Murcko scaffold split"),
        "all_checks_passed": bool(all_ok),
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(stamped("STEP5_split_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP5_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 5 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 5 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Computed the Bemis-Murcko scaffold of every compound,")
    print("                grouped compounds by scaffold, and split whole groups")
    print("                into training, validation and test sets so that no")
    print("                chemical skeleton is ever seen in two splits.")
    print(f"METHOD        : {metadata['split_method']}")
    print(f"SPLIT SIZES   : train {len(train_idx):,} "
          f"({100 * metadata['train_fraction']:.1f}%) | "
          f"val {len(val_idx):,} ({100 * metadata['val_fraction']:.1f}%) | "
          f"test {len(test_idx):,} ({100 * metadata['test_fraction']:.1f}%)")
    print(f"SCAFFOLDS     : {metadata['n_distinct_scaffolds']:,} distinct "
          f"({n_acyclic:,} acyclic compounds each forming their own group)")
    print(f"CHECKS        : {'ALL PASSED' if all_ok else 'SEE ISSUE LOG'}")
    print(f"OUTPUT FILES  : {stamped('STEP5_train_indices.npy')}")
    print(f"                {stamped('STEP5_val_indices.npy')}")
    print(f"                {stamped('STEP5_test_indices.npy')}")
    print(f"                {stamped('STEP5_split_class_distribution.csv')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged")
    print(f"ELAPSED       : {metadata['elapsed_seconds']:.1f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
