"""
STEP 10 - SHAP INTERPRETABILITY ANALYSIS
========================================
A model that says "this chemical is corrosive" but cannot say why is of little
use to a safety officer, and no journal referee will accept it. SHAP (SHapley
Additive exPlanations) solves this by borrowing an idea from cooperative game
theory: it treats each molecular descriptor as a "player" and works out how
much each one contributed to the final prediction.

The key property is that the contributions add up exactly to the prediction,
so nothing is hidden. A positive SHAP value pushes the molecule towards
"hazardous"; a negative one pushes it towards "safe".

10a  Compute SHAP values for the best model from Step 9.
10b  Bar plot of the twenty most important descriptors per hazard.
10c  Beeswarm plot showing direction as well as magnitude.
10d  Waterfall plots explaining five individual compounds.
10e  A chemical interpretation of the top features - what a chemist should
     take from them.
10f  Mean absolute SHAP value per descriptor per hazard.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (get_ablation_identity, RANDOM_SEED, TODAY, DIR_SPLITS, DIR_FEATURES, DIR_MODELS,
                        DIR_SHAP, DIR_LOGS, GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        seed_everything, stamped)

seed_everything()

# The ablation's name reflects what it actually measured; see
# get_ablation_identity() in ghs_config.py.
_ABL_NAME, _ABL_FILE, _ABL_META = get_ablation_identity()
warnings.filterwarnings("ignore")

import shap
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
RDLogger.DisableLog("rdApp.*")

ISSUE_LOG = []

# The proposal's documented fallback: if SHAP exhausts memory, explain a random
# sample of 500 test compounds instead of all of them. This machine has 7.9 GB,
# so the sample is used from the outset and the limitation is reported.
SHAP_SAMPLE_SIZE = 500

# Wall-clock ceiling for the whole SHAP computation. If the measured cost of
# the winning model would exceed this, the sample is cut down to fit.
SHAP_TIME_BUDGET_SECONDS = 2400   # 40 minutes


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ===========================================================================
# CHEMICAL KNOWLEDGE BASE FOR SUB-STEP 10e
# ===========================================================================
# What each physicochemical and topological descriptor means in plain
# chemical language. Used to write the interpretation table.
DESCRIPTOR_MEANING = {
    "MolWt": "average molecular mass",
    "ExactMolWt": "monoisotopic molecular mass",
    "MolLogP": "lipophilicity - how strongly the compound prefers oil to water",
    "TPSA": "topological polar surface area - the size of the polar region",
    "NumHDonors": "number of hydrogen-bond donors (OH, NH groups)",
    "NumHAcceptors": "number of hydrogen-bond acceptors (O, N lone pairs)",
    "NumRotatableBonds": "molecular flexibility",
    "NumAromaticRings": "number of aromatic rings",
    "NumSaturatedRings": "number of fully saturated rings",
    "NumAliphaticRings": "number of non-aromatic rings",
    "RingCount": "total number of rings",
    "FractionCSP3": "fraction of carbons that are sp3 - three-dimensionality",
    "HeavyAtomCount": "number of non-hydrogen atoms - molecular size",
    "NumHeteroatoms": "number of atoms that are neither carbon nor hydrogen",
    "NOCount": "combined nitrogen and oxygen count",
    "NHOHCount": "number of NH and OH groups",
    "LabuteASA": "approximate molecular surface area",
    "BalabanJ": "topological connectivity index - branching pattern",
    "BertzCT": "structural complexity",
    "Chi0": "zeroth-order connectivity index - atom count weighted by degree",
    "Chi1": "first-order connectivity index - bond-level branching",
    "Chi2n": "second-order connectivity index",
    "Chi3n": "third-order connectivity index",
    "Chi4n": "fourth-order connectivity index",
    "Kappa1": "first-order shape index - overall size",
    "Kappa2": "second-order shape index - how linear or star-like the skeleton is",
    "Kappa3": "third-order shape index - centrality of branching",
}

# For each hazard, the chemistry a referee would expect the model to have
# learned. Used to write a specific interpretation for each top feature.
HAZARD_CHEMISTRY = {
    "GHS01_Explosive": {
        "summary": ("Explosivity comes from functional groups that store "
                    "chemical energy and can release it without external "
                    "oxygen - nitro, nitrate ester, azide, peroxide and "
                    "strained-ring systems. A high nitrogen and oxygen count "
                    "relative to carbon is the classic signature."),
        "expected": ["NOCount", "NumHeteroatoms", "MolWt", "BertzCT",
                     "FractionCSP3"],
    },
    "GHS02_Flammable": {
        "summary": ("Flammability tracks volatility and hydrocarbon content. "
                    "Small, light, lipophilic molecules with few polar groups "
                    "have low flash points; high polarity and hydrogen bonding "
                    "raise the boiling point and reduce flammability."),
        "expected": ["MolWt", "MolLogP", "TPSA", "HeavyAtomCount",
                     "NumHeteroatoms", "FractionCSP3"],
    },
    "GHS03_Oxidising": {
        "summary": ("Oxidisers carry oxygen in a high formal oxidation state - "
                    "peroxides, perchlorates, nitrates, chlorates, permanganates. "
                    "A high oxygen-to-carbon ratio combined with low carbon "
                    "content is the dominant signal."),
        "expected": ["NOCount", "NumHeteroatoms", "MolWt", "HeavyAtomCount"],
    },
    "GHS04_CompressedGas": {
        "summary": ("The compressed-gas pictogram is assigned on physical state "
                    "rather than reactivity, so it is predicted almost entirely "
                    "from molecular size. Very low molecular weight and very few "
                    "heavy atoms are what a gas looks like to a descriptor "
                    "calculation."),
        "expected": ["MolWt", "HeavyAtomCount", "ExactMolWt", "LabuteASA",
                     "Kappa1"],
    },
    "GHS05_Corrosive": {
        "summary": ("Corrosivity is driven by extremes of pH and by strongly "
                    "electrophilic groups: mineral acids, acid halides, "
                    "anhydrides, quaternary hydroxides and amines. Ionisable "
                    "groups and hydrogen-bond donors dominate."),
        "expected": ["NHOHCount", "TPSA", "NumHDonors", "NumHeteroatoms",
                     "MolLogP"],
    },
    "GHS06_AcuteToxicity": {
        "summary": ("Acute toxicity requires a molecule to reach a biological "
                    "target quickly. Moderate lipophilicity aids membrane "
                    "crossing, while specific toxicophores - organophosphates, "
                    "cyanides, heavy-metal centres, alkaloid scaffolds - drive "
                    "potency. Fingerprint bits usually outrank bulk properties "
                    "for this class."),
        "expected": ["MolLogP", "TPSA", "NumHeteroatoms", "MolWt",
                     "NumAromaticRings"],
    },
    "GHS07_Irritant": {
        "summary": ("The exclamation-mark pictogram covers skin, eye and "
                    "respiratory irritation and lower-grade acute toxicity. It "
                    "is the commonest and most chemically diverse class, so the "
                    "model relies on a broad mix of polarity, reactivity and "
                    "substructure features rather than any single driver."),
        "expected": ["TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
                     "NHOHCount"],
    },
    "GHS08_HealthHazard": {
        "summary": ("The health-hazard pictogram flags carcinogenicity, "
                    "mutagenicity, reproductive toxicity, respiratory "
                    "sensitisation and aspiration hazard. Planar aromatic "
                    "systems capable of DNA intercalation, aromatic amines and "
                    "halogenated aromatics are the classic structural alerts, so "
                    "aromatic ring count and low sp3 fraction matter."),
        "expected": ["NumAromaticRings", "FractionCSP3", "MolLogP", "BertzCT",
                     "NumHeteroatoms"],
    },
    "GHS09_Environmental": {
        "summary": ("The environmental pictogram is assigned on aquatic "
                    "toxicity and persistence. High lipophilicity drives "
                    "bioaccumulation, and halogenated aromatics resist "
                    "degradation, so LogP together with halogen-bearing "
                    "fingerprint bits are the strongest predictors."),
        "expected": ["MolLogP", "NumAromaticRings", "NumHeteroatoms",
                     "FractionCSP3", "MolWt"],
    },
}


def describe_feature(feature_name, maccs_smarts, morgan_examples):
    """
    Turn a feature name into a sentence a chemist can read.

    Fingerprint bits are meaningless as bare numbers, so MACCS bits are
    described by their defining SMARTS pattern and Morgan bits by an example
    of the atom environment that switches them on.
    """
    if feature_name in DESCRIPTOR_MEANING:
        return DESCRIPTOR_MEANING[feature_name]
    if feature_name.startswith("MACCS_"):
        bit = int(feature_name.split("_")[1])
        smarts = maccs_smarts.get(bit)
        if smarts:
            return f"MACCS substructure key {bit}: matches SMARTS pattern {smarts}"
        return f"MACCS substructure key {bit}"
    if feature_name.startswith("Morgan_"):
        bit = int(feature_name.split("_")[1])
        example = morgan_examples.get(bit)
        if example:
            return (f"ECFP4 circular substructure bit {bit}; an example "
                    f"environment that sets it is {example}")
        return f"ECFP4 circular substructure bit {bit} (radius 2 atom environment)"
    return feature_name


def build_maccs_smarts_table():
    """Read the SMARTS definition of every MACCS key out of RDKit."""
    table = {}
    try:
        for bit, (smarts, _count) in MACCSkeys.smartsPatts.items():
            table[int(bit)] = smarts
    except Exception as exc:
        log_issue("10e", f"could not read the MACCS SMARTS definitions: {exc}")
    return table


def find_morgan_bit_examples(smiles_list, wanted_bits, max_molecules=2000):
    """
    Find, for each Morgan bit of interest, a real substructure that sets it.

    RDKit can report which atom and radius switched on each fingerprint bit.
    Converting that atom environment back to a SMILES fragment gives a
    concrete, chemically meaningful description of an otherwise opaque
    fingerprint index.
    """
    examples = {}
    wanted = set(int(b) for b in wanted_bits)
    for smiles in smiles_list[:max_molecules]:
        if not wanted:
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bit_info = {}
        try:
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024,
                                                  bitInfo=bit_info)
        except Exception:
            continue
        for bit in list(wanted):
            if bit not in bit_info:
                continue
            atom_index, radius = bit_info[bit][0]
            try:
                if radius == 0:
                    fragment = mol.GetAtomWithIdx(atom_index).GetSymbol()
                else:
                    environment = Chem.FindAtomEnvironmentOfRadiusN(
                        mol, radius, atom_index)
                    atom_map = {}
                    submol = Chem.PathToSubmol(mol, environment, atomMap=atom_map)
                    fragment = Chem.MolToSmiles(submol)
                if fragment:
                    examples[bit] = fragment
                    wanted.discard(bit)
            except Exception:
                continue
    return examples


# ===========================================================================
# 10a - COMPUTE SHAP VALUES
# ===========================================================================
def compute_shap_values(model, model_name, X_background, X_explain, feature_names):
    """
    Compute SHAP values for every hazard class.

    TreeExplainer is used for Random Forest and XGBoost. It is exact and fast
    because it exploits the tree structure directly. KernelExplainer is the
    model-agnostic fallback used for the SVM; it is far slower, so it runs on
    a k-means summary of the background data as the proposal specifies.

    Returns
    -------
    (shap_by_class, base_values, X_explain)
        shap_by_class[i] has shape (n_explained, n_features). X_explain is
        returned because the adaptive sizing below may shrink it, and every
        caller must then use the same reduced set or the shapes will not match.
    """
    print(f"\n[10a] Computing SHAP values for {model_name} ...")
    started = time.time()

    is_tree_model = model_name in ("RandomForest", "XGBoost", _ABL_NAME)

    shap_by_class, base_values = [], []

    if is_tree_model:
        # Both wrapper types keep the nine underlying binary models in a list.
        sub_models = (model.models if hasattr(model, "models")
                      else list(model.estimators_))

        # ---- adaptive sizing -------------------------------------------
        # TreeExplainer's cost grows as trees x leaves x depth-squared. For a
        # deep Random Forest that can mean days of computation, whereas for
        # the shallow trees XGBoost builds it is seconds. Rather than guess,
        # the cost is measured on five compounds and the sample size is cut to
        # whatever fits the time budget.
        first_model = next((m for m in sub_models if m is not None), None)
        if first_model is not None and X_explain.shape[0] > 25:
            try:
                probe_start = time.time()
                shap.TreeExplainer(first_model).shap_values(
                    X_explain[:5], check_additivity=False)
                seconds_per_compound = (time.time() - probe_start) / 5
                projected = (seconds_per_compound * X_explain.shape[0]
                             * len(GHS_LABEL_COLUMNS))
                print(f"      Timing probe: {seconds_per_compound:.2f} s per "
                      f"compound per class; explaining all "
                      f"{X_explain.shape[0]} compounds across nine classes "
                      f"would take {projected / 60:.0f} minutes.")
                if projected > SHAP_TIME_BUDGET_SECONDS:
                    affordable = max(25, int(SHAP_TIME_BUDGET_SECONDS /
                                             (seconds_per_compound *
                                              len(GHS_LABEL_COLUMNS))))
                    affordable = min(affordable, X_explain.shape[0])
                    log_issue("10a", f"FALLBACK APPLIED: explaining "
                                     f"{X_explain.shape[0]} compounds would take "
                                     f"{projected / 3600:.1f} hours with this "
                                     f"model's tree depth. The sample is reduced "
                                     f"to {affordable} compounds to fit the "
                                     f"{SHAP_TIME_BUDGET_SECONDS // 60}-minute "
                                     f"budget. This limitation is reported in "
                                     f"the Methods section.")
                    X_explain = X_explain[:affordable]
            except Exception as exc:
                log_issue("10a", f"SHAP timing probe failed ({exc}); "
                                 f"proceeding with the full sample.")

        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            sub_model = sub_models[class_index]
            if sub_model is None:
                shap_by_class.append(np.zeros((X_explain.shape[0],
                                               X_explain.shape[1])))
                base_values.append(0.0)
                log_issue("10a", f"{column}: no trained model - SHAP values "
                                 f"are all zero for this class.")
                continue
            try:
                explainer = shap.TreeExplainer(sub_model)
                values = explainer.shap_values(X_explain, check_additivity=False)
                expected = explainer.expected_value

                # Random Forest returns one array per class; XGBoost returns a
                # single array for the positive class. Normalise both to the
                # positive-class contribution.
                values = np.asarray(values)
                if values.ndim == 3:            # (n, features, 2)
                    values = values[:, :, 1]
                    expected = (expected[1] if np.ndim(expected) > 0 else expected)
                elif isinstance(expected, (list, np.ndarray)) and np.ndim(expected) > 0:
                    expected = expected[-1]

                shap_by_class.append(values)
                base_values.append(float(np.ravel(expected)[0]))
                print(f"      {column:<22} SHAP array {values.shape}")
            except Exception as exc:
                log_issue("10a", f"{column}: TreeExplainer failed ({exc}); "
                                 f"zeros used for this class.")
                shap_by_class.append(np.zeros((X_explain.shape[0],
                                               X_explain.shape[1])))
                base_values.append(0.0)
    else:
        # ---- SVM: KernelExplainer on a k-means background ------------------
        log_issue("10a", "best model is the SVM, so the slow model-agnostic "
                         "KernelExplainer is required. A 50-cluster k-means "
                         "summary of the training data is used as the "
                         "background distribution, as the proposal specifies.")
        background = shap.kmeans(X_background, 50)
        sub_models = (model.models if hasattr(model, "models")
                      else list(model.estimators_))
        feature_indices = getattr(model, "feature_indices", None)
        X_for_svm = (X_explain[:, feature_indices] if feature_indices is not None
                     else X_explain)

        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            sub_model = sub_models[class_index]
            try:
                explainer = shap.KernelExplainer(
                    lambda data, m=sub_model: m.predict_proba(data)[:, 1],
                    background)
                values = explainer.shap_values(X_for_svm, nsamples=100,
                                               silent=True)
                # Expand back to the full feature width so all classes align.
                if feature_indices is not None:
                    full = np.zeros((X_explain.shape[0], X_explain.shape[1]))
                    full[:, feature_indices] = values
                    values = full
                shap_by_class.append(values)
                base_values.append(float(explainer.expected_value))
                print(f"      {column:<22} SHAP array {values.shape}")
            except Exception as exc:
                log_issue("10a", f"{column}: KernelExplainer failed ({exc}).")
                shap_by_class.append(np.zeros((X_explain.shape[0],
                                               X_explain.shape[1])))
                base_values.append(0.0)

    print(f"      Completed in {(time.time() - started) / 60:.1f} minutes")
    return shap_by_class, base_values, X_explain


# ===========================================================================
# 10b + 10c - SUMMARY AND BEESWARM PLOTS
# ===========================================================================
def plot_shap_summaries(shap_by_class, X_explain, feature_names, output_dir):
    """Draw the bar and beeswarm summary plots for all nine hazards."""
    print("\n[10b + 10c] Drawing SHAP summary and beeswarm plots ...")
    bar_paths, beeswarm_paths = [], []

    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        code = column.split("_")[0]
        values = shap_by_class[class_index]
        if not np.any(values):
            log_issue("10b", f"{column}: all SHAP values are zero - plots "
                             f"skipped for this class.")
            continue

        # ---- 10b bar plot: which descriptors matter most ------------------
        plt.figure(figsize=(9, 7))
        shap.summary_plot(values, X_explain, feature_names=feature_names,
                          max_display=20, plot_type="bar", show=False)
        plt.title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                  f"Top 20 descriptors by mean absolute SHAP value",
                  fontsize=12, fontweight="bold")
        plt.xlabel("Mean |SHAP value| (average impact on the prediction)",
                   fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir, f"STEP10_SHAP_summary_{code}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close("all")
        bar_paths.append(path)

        # ---- 10c beeswarm: direction as well as magnitude -----------------
        plt.figure(figsize=(9, 7))
        shap.summary_plot(values, X_explain, feature_names=feature_names,
                          max_display=20, show=False)
        plt.title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                  f"SHAP beeswarm - red = high descriptor value, "
                  f"blue = low; right = pushes towards hazardous",
                  fontsize=11, fontweight="bold")
        plt.xlabel("SHAP value (impact on the model's output)", fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir, f"STEP10_SHAP_beeswarm_{code}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close("all")
        beeswarm_paths.append(path)
        print(f"      {column} done")

    print(f"      {len(bar_paths)} bar plots, {len(beeswarm_paths)} beeswarm plots")
    return bar_paths, beeswarm_paths


# ===========================================================================
# 10d - WATERFALL PLOTS FOR INDIVIDUAL COMPOUNDS
# ===========================================================================
def plot_waterfalls(shap_by_class, base_values, X_explain, feature_names,
                    probabilities, compound_info, output_dir):
    """
    Explain five individual compounds in detail.

    The five are chosen to span the full range of model confidence: the
    compound the model is most certain is hazardous, the one it is most
    certain is safe, and three borderline cases near the decision boundary.
    Borderline cases are the most informative, because they show what tips
    the balance.
    """
    print("\n[10d] Drawing SHAP waterfall plots for five representative "
          "compounds ...")
    paths = []

    # The most common hazard class is the most informative one to explain.
    class_index = int(np.argmax([np.abs(v).sum() for v in shap_by_class]))
    column = GHS_LABEL_COLUMNS[class_index]
    code = column.split("_")[0]
    scores = probabilities[:, class_index]

    order = np.argsort(scores)
    selections = [
        ("1_most_confidently_hazardous", int(order[-1])),
        ("2_borderline_high", int(order[int(0.60 * len(order))])),
        ("3_borderline_middle", int(order[len(order) // 2])),
        ("4_borderline_low", int(order[int(0.40 * len(order))])),
        ("5_least_confidently_hazardous", int(order[0])),
    ]

    for label, row_index in selections:
        try:
            name = compound_info.iloc[row_index].get("Name", "unknown")
            cid = compound_info.iloc[row_index].get("CID", "?")
            explanation = shap.Explanation(
                values=shap_by_class[class_index][row_index],
                base_values=base_values[class_index],
                data=X_explain[row_index],
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 7))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.title(f"{code}: {GHS_TRUE_MEANING[column]}\n"
                      f"{str(name)[:60]} (CID {cid}) - "
                      f"predicted probability {scores[row_index]:.3f}",
                      fontsize=11, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(output_dir,
                                f"STEP10_SHAP_waterfall_compound{label}_{code}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close("all")
            paths.append(path)
            print(f"      {label:<34} {str(name)[:40]:<42} p={scores[row_index]:.3f}")
        except Exception as exc:
            log_issue("10d", f"waterfall plot for {label} failed: {exc}")
            plt.close("all")
    return paths, column


# ===========================================================================
# 10e + 10f - INTERPRETATION AND MEAN SHAP TABLES
# ===========================================================================
def build_interpretation_tables(shap_by_class, feature_names, X_explain,
                                smiles_list):
    """
    Produce the two tables that make the model's reasoning readable.

    10f is the raw numbers: the mean absolute SHAP value of every descriptor
    for every hazard.
    10e is the chemistry: the top five descriptors per hazard, what each one
    means, whether it pushes towards or away from the hazard, and whether the
    model's reasoning matches what a chemist would expect.
    """
    print("\n[10f] Building the mean absolute SHAP value table ...")
    mean_abs = {}
    signed_mean = {}
    direction_corr = {}
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        values = shap_by_class[class_index]
        mean_abs[column] = np.abs(values).mean(axis=0)
        signed_mean[column] = values.mean(axis=0)

        # ---- which way does the descriptor actually push? ------------------
        # The mean signed SHAP value is NOT the answer. For a rare hazard,
        # almost every compound in the sample is a negative example, so nearly
        # every descriptor has a negative mean signed value regardless of the
        # chemistry - which would make the interpretation table say that a
        # higher value pushes towards "safe" even for descriptors that do the
        # opposite.
        #
        # The correct measure is how the SHAP value moves as the descriptor
        # value moves: the correlation between the two across the sample. A
        # positive correlation means a higher descriptor value pushes the
        # prediction towards hazardous. This is exactly what the colour axis
        # of a SHAP beeswarm plot shows.
        correlations = np.zeros(values.shape[1])
        for feature_index in range(values.shape[1]):
            feature_values = X_explain[:, feature_index]
            shap_values = values[:, feature_index]
            # A constant descriptor or a constant SHAP column has no direction.
            if feature_values.std() > 1e-12 and shap_values.std() > 1e-12:
                correlations[feature_index] = np.corrcoef(feature_values,
                                                          shap_values)[0, 1]
        direction_corr[column] = correlations

    mean_shap_table = pd.DataFrame(mean_abs, index=feature_names)
    mean_shap_table.index.name = "Feature"
    mean_shap_table["Mean_across_all_classes"] = mean_shap_table.mean(axis=1)
    mean_shap_table = mean_shap_table.sort_values("Mean_across_all_classes",
                                                  ascending=False)

    print("\n      Top 15 descriptors overall, by mean |SHAP| across all classes")
    print("      " + "-" * 70)
    for rank, (feature, row) in enumerate(mean_shap_table.head(15).iterrows(), 1):
        print(f"      {rank:>3}. {feature:<24} "
              f"{row['Mean_across_all_classes']:.6f}")
    print("      " + "-" * 70)

    # ---- 10e chemical interpretation ---------------------------------------
    print("\n[10e] Building the chemical interpretation table ...")
    maccs_smarts = build_maccs_smarts_table()

    # Work out which Morgan bits appear in any class's top five, then find a
    # real substructure example for each one.
    top_morgan_bits = set()
    for column in GHS_LABEL_COLUMNS:
        order = np.argsort(mean_abs[column])[::-1][:5]
        for index in order:
            name = feature_names[index]
            if name.startswith("Morgan_"):
                top_morgan_bits.add(int(name.split("_")[1]))
    morgan_examples = (find_morgan_bit_examples(smiles_list, top_morgan_bits)
                       if top_morgan_bits else {})

    rows = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        chemistry = HAZARD_CHEMISTRY[column]
        order = np.argsort(mean_abs[column])[::-1][:5]
        for rank, feature_index in enumerate(order, 1):
            feature = feature_names[feature_index]
            correlation = direction_corr[column][feature_index]
            if correlation > 0.05:
                direction = ("positive - a HIGHER value of this descriptor "
                             "pushes the prediction towards HAZARDOUS")
            elif correlation < -0.05:
                direction = ("negative - a HIGHER value of this descriptor "
                             "pushes the prediction towards SAFE")
            else:
                direction = ("non-monotonic - the descriptor matters, but its "
                             "effect is not a simple increase or decrease")
            expected = ("yes - this is a descriptor a chemist would expect to "
                        "matter for this hazard"
                        if feature in chemistry["expected"] else
                        "not among the classically expected bulk descriptors; "
                        "likely acting as a structural marker")
            rows.append({
                "GHS_Column": column,
                "Pictogram_Code": column.split("_")[0],
                "Hazard_Meaning": GHS_TRUE_MEANING[column],
                "Rank": rank,
                "Feature": feature,
                "Mean_Abs_SHAP": round(float(mean_abs[column][feature_index]), 6),
                "Mean_Signed_SHAP": round(float(signed_mean[column][feature_index]), 6),
                "Value_SHAP_Correlation": round(float(correlation), 4),
                "SHAP_Direction": direction,
                "What_The_Descriptor_Measures": describe_feature(
                    feature, maccs_smarts, morgan_examples),
                "Chemical_Interpretation": chemistry["summary"],
                "Matches_Chemical_Expectation": expected,
            })

    interpretation_table = pd.DataFrame(rows)

    print("\n      Top descriptor for each hazard class")
    print("      " + "-" * 92)
    for column in GHS_LABEL_COLUMNS:
        top = interpretation_table[
            (interpretation_table["GHS_Column"] == column) &
            (interpretation_table["Rank"] == 1)]
        if len(top):
            row = top.iloc[0]
            correlation = row["Value_SHAP_Correlation"]
            sign = "+" if correlation > 0.05 else "-" if correlation < -0.05 else "~"
            print(f"      {column:<22} {row['Feature']:<18} ({sign}) "
                  f"{str(row['What_The_Descriptor_Measures'])[:44]}")
    print("      " + "-" * 92)

    return mean_shap_table, interpretation_table


# ===========================================================================
# MAIN
# ===========================================================================
def run_shap_analysis():
    """Run the whole of Step 10 and save every plot and table."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 10 - SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 78)

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    train_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_train_indices.npy"))
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))

    with open(stamped("STEP4_feature_names.txt"), encoding="utf-8") as fh:
        feature_names = [line.strip() for line in fh
                         if line.strip() and not line.startswith("#")]

    with open(stamped("STEP9_evaluation_summary.json"), encoding="utf-8") as fh:
        best_model_name = json.load(fh)["best_model"]
    print(f"Best model from Step 9: {best_model_name}")

    # Step 7 pickled its wrapper class from __main__; register it before load.
    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()

    # Must match the file Step 9 actually scored, or the explanation would
    # describe a different model from the one whose results are reported.
    candidates = {
        "RandomForest": ["STEP8_rf_tuned.pkl", "STEP7_rf_model.pkl"],
        "XGBoost": ["STEP8_xgb_tuned.pkl", "STEP7_xgb_models.pkl"],
        "SVM": ["STEP7_svm_model.pkl"],
        _ABL_NAME: [_ABL_FILE] if _ABL_FILE else [],
    }[best_model_name]
    model_file = next((f for f in candidates
                       if os.path.exists(os.path.join(DIR_MODELS, f))),
                      candidates[-1])
    path = os.path.join(DIR_MODELS, model_file)
    if not os.path.exists(path):
        path = os.path.join(DIR_MODELS, "STEP7_rf_model.pkl")
        log_issue("10a", f"{model_file} not found; using STEP7_rf_model.pkl.")
    model = joblib.load(path)

    # ---- choose which test compounds to explain ---------------------------
    compound_index = pd.read_csv(
        os.path.join(DIR_FEATURES, "STEP4_compound_index.csv"), low_memory=False)

    if len(test_idx) > SHAP_SAMPLE_SIZE:
        log_issue("10a", f"FALLBACK APPLIED (proposal step 10a): SHAP values are "
                         f"computed on a random {SHAP_SAMPLE_SIZE}-compound sample "
                         f"of the {len(test_idx):,}-compound test set. Explaining "
                         f"every test compound would exceed this machine's 7.9 GB "
                         f"of RAM. This limitation is reported in the Methods "
                         f"section.")
        rng = np.random.RandomState(RANDOM_SEED)
        chosen = np.sort(rng.choice(len(test_idx), SHAP_SAMPLE_SIZE, replace=False))
        explain_idx = test_idx[chosen]
    else:
        explain_idx = test_idx

    X_explain = X[explain_idx]
    info_explain = compound_index.iloc[explain_idx].reset_index(drop=True)
    print(f"Explaining {X_explain.shape[0]:,} test compounds "
          f"x {X_explain.shape[1]:,} features")

    # A small background sample is enough for the SVM's KernelExplainer.
    rng = np.random.RandomState(RANDOM_SEED)
    background_idx = train_idx[rng.choice(len(train_idx),
                                          min(500, len(train_idx)), replace=False)]
    X_background = X[background_idx]

    # ---- 10a ---------------------------------------------------------------
    shap_by_class, base_values, X_explain = compute_shap_values(
        model, best_model_name, X_background, X_explain, feature_names)

    # The adaptive time budget may have shrunk the sample; keep the compound
    # index and the saved indices aligned with whatever was actually explained.
    if X_explain.shape[0] < len(explain_idx):
        explain_idx = explain_idx[:X_explain.shape[0]]
        info_explain = info_explain.iloc[:X_explain.shape[0]].reset_index(drop=True)
        print(f"      Sample reduced to {X_explain.shape[0]} compounds "
              f"by the SHAP time budget.")

    np.savez_compressed(
        os.path.join(DIR_SHAP, "STEP10_shap_values.npz"),
        **{f"shap_{c}": v for c, v in zip(GHS_LABEL_COLUMNS, shap_by_class)},
        base_values=np.array(base_values),
        explained_indices=explain_idx)

    # ---- 10b, 10c ----------------------------------------------------------
    bar_paths, beeswarm_paths = plot_shap_summaries(
        shap_by_class, X_explain, feature_names, DIR_SHAP)

    # ---- 10d ---------------------------------------------------------------
    probabilities = np.column_stack([
        (p[:, 1] if p.shape[1] > 1 else p[:, 0])
        for p in model.predict_proba(X_explain)])
    waterfall_paths, waterfall_class = plot_waterfalls(
        shap_by_class, base_values, X_explain, feature_names, probabilities,
        info_explain, DIR_SHAP)

    # ---- 10e, 10f ----------------------------------------------------------
    smiles_column = ("CanonicalSMILES_RDKit"
                     if "CanonicalSMILES_RDKit" in compound_index.columns
                     else "SMILES")
    mean_shap_table, interpretation_table = build_interpretation_tables(
        shap_by_class, feature_names, X_explain,
        compound_index[smiles_column].tolist())

    mean_shap_table.to_csv(stamped("STEP10_mean_SHAP_values.csv"))
    interpretation_table.to_csv(stamped("STEP10_SHAP_chemical_interpretation.csv"),
                                index=False)

    # A wider top-20 table is needed for supplementary Table S4 in Step 13.
    rows = []
    maccs_smarts = build_maccs_smarts_table()
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        values = np.abs(shap_by_class[class_index]).mean(axis=0)
        signed = shap_by_class[class_index].mean(axis=0)
        for rank, feature_index in enumerate(np.argsort(values)[::-1][:20], 1):
            # Same direction measure as the interpretation table above.
            feature_values = X_explain[:, feature_index]
            shap_values = shap_by_class[class_index][:, feature_index]
            correlation = (np.corrcoef(feature_values, shap_values)[0, 1]
                           if feature_values.std() > 1e-12
                           and shap_values.std() > 1e-12 else 0.0)
            rows.append({
                "GHS_Column": column,
                "Hazard_Meaning": GHS_TRUE_MEANING[column],
                "Rank": rank,
                "Feature": feature_names[feature_index],
                "Mean_Abs_SHAP": round(float(values[feature_index]), 6),
                "Mean_Signed_SHAP": round(float(signed[feature_index]), 6),
                "Value_SHAP_Correlation": round(float(correlation), 4),
                "Direction": ("higher value -> more hazardous" if correlation > 0.05
                              else "higher value -> less hazardous"
                              if correlation < -0.05 else "non-monotonic"),
                "Description": describe_feature(feature_names[feature_index],
                                                maccs_smarts, {}),
            })
    pd.DataFrame(rows).to_csv(stamped("STEP10_top20_SHAP_features_per_class.csv"),
                              index=False)

    summary = {
        "best_model_explained": best_model_name,
        "n_compounds_explained": int(X_explain.shape[0]),
        "n_features": int(X_explain.shape[1]),
        "shap_sample_used": bool(len(test_idx) > SHAP_SAMPLE_SIZE),
        "top3_features_overall": mean_shap_table.head(3).index.tolist(),
        "waterfall_class": waterfall_class,
        "n_bar_plots": len(bar_paths),
        "n_beeswarm_plots": len(beeswarm_paths),
        "n_waterfall_plots": len(waterfall_paths),
        "random_seed": RANDOM_SEED,
        "elapsed_seconds": round(time.time() - total_start, 1),
    }
    with open(stamped("STEP10_shap_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log_path = os.path.join(DIR_LOGS, f"STEP10_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 10 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    print("\n" + "=" * 78)
    print("STEP 10 PROGRESS REPORT")
    print("=" * 78)
    print(f"WHAT WAS DONE : Computed SHAP values for {best_model_name}, drew bar,")
    print("                beeswarm and waterfall plots, and wrote the chemical")
    print("                interpretation of the most influential descriptors.")
    print(f"EXPLAINED     : {X_explain.shape[0]:,} test compounds "
          f"x {X_explain.shape[1]:,} descriptors")
    print(f"TOP 3 OVERALL : {', '.join(summary['top3_features_overall'])}")
    print(f"OUTPUT FILES  : {DIR_SHAP}\\STEP10_SHAP_summary_GHS*.png "
          f"({len(bar_paths)} files)")
    print(f"                {DIR_SHAP}\\STEP10_SHAP_beeswarm_GHS*.png "
          f"({len(beeswarm_paths)} files)")
    print(f"                {DIR_SHAP}\\STEP10_SHAP_waterfall_*.png "
          f"({len(waterfall_paths)} files)")
    print(f"                {stamped('STEP10_SHAP_chemical_interpretation.csv')}")
    print(f"                {stamped('STEP10_mean_SHAP_values.csv')}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print(f"ELAPSED       : {summary['elapsed_seconds'] / 60:.1f} minutes")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    run_shap_analysis()
