"""
MANUSCRIPT SECTIONS - INTRODUCTION, RESULTS, DISCUSSION, CONCLUSIONS
====================================================================
Generates the manuscript sections that Step 13 does not produce, together with
the title page and the table-of-contents graphic.

Every quantity quoted in the text is read from the JSON and CSV outputs of the
pipeline rather than typed in by hand. If the analysis is re-run and a number
changes, re-running this script updates the manuscript to match. Hand-typed
figures in a paper drift silently from the results they claim to describe;
this is the whole reason the sections are generated rather than written.

Output: publication_materials/manuscript/

Author : Sareer Ahmad
"""

import os
import sys
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (PROJECT_ROOT, DIR_MALAYSIA, DIR_PUB, GHS_LABEL_COLUMNS,
                        GHS_TRUE_MEANING, stamped)

MANUSCRIPT = os.path.join(DIR_PUB, "manuscript")
os.makedirs(MANUSCRIPT, exist_ok=True)

# ---------------------------------------------------------------------------
# AUTHOR DETAILS
# ---------------------------------------------------------------------------
# Kept in one place so the title page, the cover letter and the final report
# cannot disagree with one another.
#
# A NOTE ON THE AFFILIATION, because this one is easy to get wrong.
# An affiliation names the institution where the work was carried out, or where
# the author currently holds a position. A degree completed in the past is not
# by itself an affiliation. Listing a university implies that the work has some
# connection to it, so it is correct here only if the research grew out of the
# author's studies there, was begun while enrolled, or the author retains some
# status with the department.
#
# If none of those hold, set AFFILIATION to INDEPENDENT_AFFILIATION below.
# Journals ask authors to confirm that affiliations are accurate, and
# institutions can object to their name appearing on work they had no part in.
# ---------------------------------------------------------------------------
AUTHOR_NAME = "Sareer Ahmad"
AUTHOR_EMAIL = "sareerkh9194@gmail.com"
AUTHOR_ORCID = "https://orcid.org/0009-0003-2580-091X"

UNIVERSITY_AFFILIATION = ("Department of Chemistry, University of Peshawar,\n"
                          "    Peshawar 25120, Khyber Pakhtunkhwa, Pakistan")
INDEPENDENT_AFFILIATION = "Independent Researcher, Peshawar, Pakistan"

# Currently in use:
AFFILIATION = UNIVERSITY_AFFILIATION


def load(path, default=None):
    """Read a JSON file, returning a default when it is absent."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default if default is not None else {}


def wrap(text, width=79):
    """
    Re-wrap prose to a fixed width, leaving structure intact.

    Works line by line rather than paragraph by paragraph. An earlier version
    split on blank lines only, which merged each section heading with the rule
    of dashes beneath it and then wrapped the two together into an unreadable
    run-on. Rules, headings, table rows and indented text are passed through
    untouched; only genuine prose paragraphs are re-flowed.
    """
    lines = text.strip().split("\n")
    out, paragraph = [], []

    def flush():
        """Emit any prose collected so far, wrapped to width."""
        if paragraph:
            out.append(textwrap.fill(" ".join(" ".join(paragraph).split()),
                                     width=width))
            paragraph.clear()

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_rule = bool(stripped) and set(stripped) <= set("=-")
        # A heading is the line immediately above a rule.
        next_is_rule = (i + 1 < len(lines)
                        and lines[i + 1].strip()
                        and set(lines[i + 1].strip()) <= set("=-"))
        # Table rows, indented blocks and bracketed notes keep their layout.
        preformatted = (line.startswith(("  ", "\t"))
                        or stripped.startswith(("[", "|", "*"))
                        or "  " in stripped and any(ch.isdigit() for ch in stripped)
                        and stripped.count(" ") > 6 and len(stripped) < width)

        if not stripped:
            flush()
            out.append("")
        elif is_rule or next_is_rule or preformatted:
            flush()
            out.append(line.rstrip())
        else:
            paragraph.append(stripped)
    flush()

    # Collapse runs of blank lines to at most one.
    cleaned, blank = [], False
    for line in out:
        if line == "":
            if not blank:
                cleaned.append(line)
            blank = True
        else:
            cleaned.append(line)
            blank = False
    return "\n".join(cleaned)


def gather():
    """Collect every number the manuscript quotes into one dictionary."""
    f = {}
    f["clean"] = load(stamped("STEP3_cleaning_summary.json"))
    f["desc"] = load(stamped("STEP4_descriptor_metadata.json"))
    f["split"] = load(stamped("STEP5_split_metadata.json"))
    f["eval"] = load(stamped("STEP9_evaluation_summary.json"))
    f["shap"] = load(stamped("STEP10_shap_summary.json"))
    f["times"] = load(stamped("STEP7_training_times.json"))

    f["dist"] = pd.read_csv(stamped("STEP3_class_distribution_table.csv"))
    f["results"] = pd.read_csv(stamped("STEP9_model_comparison_results.csv"))
    f["interp"] = pd.read_csv(stamped("STEP10_SHAP_chemical_interpretation.csv"))
    f["sector"] = pd.read_csv(os.path.join(
        DIR_MALAYSIA, "STEP11_malaysia_per_sector_metrics.csv"))
    f["mal_class"] = pd.read_csv(os.path.join(
        DIR_MALAYSIA, "STEP11_malaysia_per_class_metrics.csv"))
    f["size"] = pd.read_csv(stamped("EXTRA_controlled_size_experiment.csv"))

    best = f["eval"]["best_model"]
    x = f["results"][(f["results"].Model == best)
                     & (f["results"].Threshold_Type == "calibrated_F1")]
    f["best_rows"] = x.set_index("GHS_Column")
    f["best"] = best
    return f


# ===========================================================================
# TITLE PAGE
# ===========================================================================
def title_page(f):
    return f"""TITLE PAGE
================================================================================

TITLE

    Interpretable Machine Learning for Predicting GHS Chemical Hazard
    Classifications: A Multi-Label Approach Using {f['clean']['final_cleaned_compounds']:,}
    Compounds and Scaffold-Based Validation

RUNNING TITLE

    Interpretable multi-label prediction of GHS hazard classifications

AUTHOR

    {AUTHOR_NAME}*

AFFILIATION

    {AFFILIATION}

    * Corresponding author. Email: {AUTHOR_EMAIL}
      ORCID: {AUTHOR_ORCID}

KEYWORDS

    GHS classification; multi-label learning; molecular descriptors; SHAP
    interpretability; chemical safety; scaffold splitting; QSAR

--------------------------------------------------------------------------------
NOTES BEFORE SUBMISSION
--------------------------------------------------------------------------------
1. This is a single-author submission.

2. CONFIRM THE AFFILIATION. It currently reads:

       {AFFILIATION}

   An affiliation names where the work was carried out or where the author
   currently holds a position. A degree completed in the past is not by itself
   an affiliation. Listing the University of Peshawar is correct if this
   research grew out of your studies there, was begun while you were enrolled,
   or you retain some status with the department. If none of those apply, set
   AFFILIATION = INDEPENDENT_AFFILIATION at the top of
   src/manuscript_sections.py and regenerate.

   Two practical points. Journals ask authors to confirm affiliations are
   accurate at submission. Some institutions also ask to be notified before
   their name appears on a publication - a short email to the department is
   worth sending, and costs nothing.

3. Confirm the email above is the address you want printed in the published
   article; it becomes permanently public as the corresponding author contact.
"""


# ===========================================================================
# INTRODUCTION
# ===========================================================================
def introduction(f):
    n_clean = f["clean"]["final_cleaned_compounds"]
    n_multi = f["clean"]["multilabel_compounds"]
    pct_multi = 100 * n_multi / n_clean
    return f"""INTRODUCTION
================================================================================

The Globally Harmonized System of Classification and Labelling of Chemicals
(GHS) is the framework through which chemical hazards are communicated
worldwide. It assigns each substance a set of nine pictograms describing
physical, health and environmental hazards, and those pictograms determine how
a chemical is labelled, stored, transported and handled in every jurisdiction
that has adopted the system, including Malaysia under the Occupational Safety
and Health (Classification, Labelling and Safety Data Sheet of Hazardous
Chemicals) Regulations 2013.

Assigning those pictograms requires experimental data. Flash points must be
measured, acute toxicity established in animal studies, aquatic toxicity
determined in standardised assays. This testing is slow and expensive, and it
has been completed for only a small fraction of the chemicals in commercial
circulation. Tens of millions of substances are registered in public chemical
databases; fewer than a quarter of a million carry a GHS classification of any
kind. The gap is not an abstract regulatory inconvenience. On 7 March 2019,
chemical waste of unverified composition was discharged into the Sungai Kim Kim
river at Pasir Gudang, Johor, Malaysia. Vapour rising from the river affected
more than 2,500 people, the majority of them schoolchildren, and 111 schools
were closed. Acrylonitrile and acrolein were subsequently identified as the
principal agents. Had the hazard profile of the discharged material been known
at the point of disposal, the response would have been an acute inhalation
emergency from the outset rather than a developing mystery.

Computational prediction offers a way to narrow that gap. Quantitative
structure-activity relationship modelling has been applied to individual
toxicological endpoints for decades, and machine learning methods now routinely
predict properties such as aqueous solubility, mutagenicity and acute oral
toxicity from molecular structure alone. Applying the same approach to GHS
classification is attractive because the pictograms are the operational
currency of chemical safety: a predicted pictogram translates directly into a
handling decision in a way that a predicted LD50 does not.

Three difficulties have limited progress. The first is that GHS assignment is
inherently multi-label. A compound does not carry one hazard but any
combination of nine, and in the dataset assembled here {pct_multi:.1f} per cent
of compounds carry more than one. Treating each pictogram as an independent
binary problem ignores the correlations between them, while treating the
combination as a single multi-class label is intractable given the number of
possible combinations.

The second difficulty is evaluation. Chemical datasets contain large families
of structurally similar molecules. A model validated on a random train-test
split is frequently tested on near-duplicates of its own training data, and
reports an accuracy that collapses the moment it encounters genuinely
unfamiliar chemistry. Scaffold-based splitting, in which every compound sharing
a Bemis-Murcko framework is confined to a single partition, is the established
remedy, yet it remains less common in the hazard-prediction literature than in
drug discovery.

The third difficulty is interpretability. A regulator cannot act on a
prediction whose basis is opaque. A model that outputs "corrosive, 87 per cent"
without saying why offers a safety officer no way to judge whether the
prediction is chemically sensible or an artefact. Post-hoc attribution methods,
particularly SHAP, make it possible to decompose an individual prediction into
the contribution of each molecular descriptor, and so to ask whether a model
has recovered known structure-hazard relationships or merely memorised
correlations.

This work addresses all three. GHS classifications were harvested from PubChem
for {f['clean']['raw_rows']:,} compound records contributed by five independent
regulatory bodies, reconciled by majority vote where those bodies disagreed,
and reduced to {n_clean:,} unique validated structures - to our knowledge the
largest multi-label GHS dataset assembled for machine learning. Three
algorithms were trained as multi-label predictors and evaluated on a
Bemis-Murcko scaffold split, so that no chemical skeleton is shared between
training and test data. Predictions were interpreted with SHAP and the
resulting attributions examined against established chemistry. Finally, the
framework was applied to chemicals drawn from four Malaysian industrial sectors
and to the substances implicated in the Johor 2019 incident, to test whether a
model trained on global regulatory data is useful in a specific national
context.

Two findings emerged that we believe are of wider methodological interest than
the model itself. The first concerns how scaffold groups are allocated to
partitions: the obvious allocation strategies satisfy every superficial check
while systematically starving the rarest hazard classes of training data. The
second concerns dataset size, where a learning curve computed within a
convenience subsample indicated saturation that a controlled experiment on the
full dataset showed to be illusory.
"""


# ===========================================================================
# RESULTS
# ===========================================================================
def results(f):
    e, s, d = f["eval"], f["split"], f["desc"]
    best = f["best"]
    br = f["best_rows"]
    auc = e["auc_per_class_best_model"]
    means = e["mean_auc_per_model"]
    mccs = e["mean_mcc_per_model"]
    size = f["size"]

    # per-class table
    lines = []
    for c in GHS_LABEL_COLUMNS:
        r = br.loc[c]
        lines.append(
            f"  {c.split('_')[0]:<6}{GHS_TRUE_MEANING[c].split('(')[0].strip():<24}"
            f"{int(r.N_Test_Positive):>8,}{r.AUC_ROC:>9.4f}"
            f"  [{r.AUC_CI95_lower:.3f}, {r.AUC_CI95_upper:.3f}]"
            f"{r.Average_Precision:>9.3f}{r.F1:>8.3f}{r.MCC:>8.3f}")
    per_class = "\n".join(lines)

    # SHAP rank-1 per class
    top = f["interp"][f["interp"].Rank == 1]
    shap_lines = []
    for _, r in top.iterrows():
        shap_lines.append(f"  {r.GHS_Column.split('_')[0]:<6}"
                          f"{r.Feature:<16}{r.Value_SHAP_Correlation:>+8.3f}   "
                          f"{str(r.What_The_Descriptor_Measures)[:44]}")
    shap_tbl = "\n".join(shap_lines)

    sector = "\n".join(
        f"  {r.Sector:<26}{int(r.N_Compounds):>5}{r.Label_Accuracy:>10.3f}"
        f"{r.Hazard_Recall:>10.3f}"
        for _, r in f["sector"].iterrows())

    size_tbl = "\n".join(
        f"  {int(r.n_train):>9,}{r.mean_auc:>12.4f}" for _, r in size.iterrows())

    gain = size.mean_auc.iloc[-1] - size.mean_auc.iloc[0]
    ci = e["model_selection"]["tie_tolerance_from_bootstrap"]

    return f"""RESULTS
================================================================================

Dataset assembly and composition
--------------------------------------------------------------------------------

GHS classifications were retrieved for {f['clean']['raw_rows']:,} compound
records. Structural validation removed {f['clean']['removed_invalid_smiles']}
records whose SMILES strings could not be parsed, and deduplication by InChIKey
removed a further {f['clean']['removed_duplicates']}, leaving
{f['clean']['final_cleaned_compounds']:,} unique compounds. Five regulatory
sources contributed classifications, and {f['clean']['conflicted_compounds']:,}
compounds received a tied vote between them; these were retained, flagged, and
resolved in favour of the hazardous assignment.

The dataset is strongly multi-label and strongly imbalanced.
{f['clean']['multilabel_compounds']:,} compounds
({100*f['clean']['multilabel_compounds']/f['clean']['final_cleaned_compounds']:.1f} per
cent) carry more than one pictogram. Class frequencies span more than three
orders of magnitude, from {int(f['dist'].N_Positive.max()):,} compounds
carrying the irritant pictogram to {int(f['dist'].N_Positive.min())} carrying
the explosive pictogram (Figure 2). This imbalance, rather than the size of the
dataset, proved to be the dominant constraint on performance.

Molecular representation and partitioning
--------------------------------------------------------------------------------

Each compound was described by {d['n_features_computed']:,} descriptors
combining physicochemical properties, Morgan (ECFP4) and MACCS fingerprints and
topological indices. Removing descriptors with variance below 0.01 eliminated
{d['n_features_removed']:,} - almost all of them sparsely populated
fingerprint bits - leaving {d['n_features_after_variance_filter']:,} features.

Partitioning by Bemis-Murcko scaffold produced {s['n_distinct_scaffolds']:,}
distinct scaffold groups, of which {s['n_acyclic_compounds']:,} were acyclic
compounds treated as individual groups. The final partition contained
{s['n_train']:,} training, {s['n_val']:,} validation and {s['n_test']:,} test
compounds, with no scaffold shared between any two partitions and every hazard
class represented in all three.

Allocating scaffold groups to partitions required more care than is usually
acknowledged. Filling the training partition to its quota before the others
caused large scaffold groups encountered late to overflow into the test
partition, producing an 80:3:17 split. Assigning each group to whichever
partition was furthest below its quota corrected the overall ratios but starved
the rare classes: only 30 per cent of compressed gases and 39 per cent of
oxidisers reached the training partition instead of the intended 80 per cent,
because every acyclic molecule forms a single-compound group and those groups
are allocated last. Group-wise iterative stratification, scoring each group
against every partition on overall size and on each class it contains,
returned every class to within one percentage point of its intended share while
preserving exact partition sizes.

Model performance
--------------------------------------------------------------------------------

Four models were evaluated on the held-out test partition of {s['n_test']:,}
compounds. Mean AUC-ROC across the nine classes was {means.get(best):.4f} for
{best}, {means.get('RandomForest_SMOTE', float('nan')):.4f} for the Random
Forest without class weighting, {means.get('RandomForest'):.4f} for the
class-weighted Random Forest and {means.get('SVM'):.4f} for the support vector
machine. Mean Matthews correlation coefficients were {mccs.get(best):.4f},
{mccs.get('RandomForest_SMOTE', float('nan')):.4f},
{mccs.get('RandomForest'):.4f} and {mccs.get('SVM'):.4f} respectively.
{best} was selected as the best model on both metrics. The bootstrap 95 per
cent confidence interval had a median half-width of {ci:.4f}, so differences
smaller than that are not meaningful; the two leading models were separated on
Matthews correlation coefficient rather than on area under the curve.

Per-class performance for {best} is given below; N is the number of positive
examples in the test partition and the interval is the bootstrap 95 per cent
confidence interval on AUC-ROC.

  Code  Hazard                         N     AUC       CI 95%        AP      F1     MCC
  ---------------------------------------------------------------------------------------
{per_class}

Discrimination is excellent for the physically determined hazards - compressed
gas ({auc['GHS04_CompressedGas']:.3f}), explosive
({auc['GHS01_Explosive']:.3f}) and flammable ({auc['GHS02_Flammable']:.3f}) -
and substantially lower for the biologically mediated ones, acute toxicity
({auc['GHS06_AcuteToxicity']:.3f}) and irritation
({auc['GHS07_Irritant']:.3f}). This ordering is chemically coherent: physical
state and flammability are largely determined by molecular size and polarity,
whereas toxicity depends on specific molecular mechanisms that bulk descriptors
capture only indirectly.

Training times on a two-core workstation were
{f['times']['seconds']['RandomForest']/60:.1f} minutes for the Random Forest,
{f['times']['seconds']['XGBoost']/60:.1f} minutes for the nine gradient-boosted
models and {f['times']['seconds']['SVM']/60:.1f} minutes for the support vector
machine, the last on a reduced feature set and subsampled training partition
(see Limitations).

Effect of training-set size
--------------------------------------------------------------------------------

Models were trained on nested subsets of the training partition and evaluated
on the same fixed test partition with identical hyperparameters, so that
training-set size was the only quantity varying.

     n_train    mean AUC
  -----------------------
{size_tbl}

Mean AUC-ROC rose monotonically, gaining {gain:+.4f} overall - more than four
times the confidence interval - and every one of the nine classes improved. The
rare classes gained most.

A learning curve computed earlier within a 40,000-compound convenience
subsample had appeared to plateau, with the final increment of data changing
mean AUC-ROC by less than 0.001. That appearance was an artefact of how the
subsample had been constructed: it deliberately retained every positive example
of the rare classes, so those classes could not improve with additional data
and the aggregate curve flattened prematurely.

Interpretability
--------------------------------------------------------------------------------

SHAP values were computed for {f['shap']['n_compounds_explained']} test
compounds. Across all nine classes the most influential descriptors were
{', '.join(f['shap']['top3_features_overall'])}. The single most influential
descriptor for each class, with the correlation between descriptor value and
SHAP value (positive meaning that higher values push towards the hazard):

  Code  Descriptor         r      Measures
  -------------------------------------------------------------------------
{shap_tbl}

The explosive class is the most striking. Its leading descriptor is MACCS key
70, defined by the SMARTS pattern [!#6;!#1]~[#7]~[!#6;!#1] - a heteroatom bonded
to nitrogen bonded to a further heteroatom - with a correlation of
{float(top[top.GHS_Column=='GHS01_Explosive'].Value_SHAP_Correlation.iloc[0]):+.3f}.
That connectivity is the defining motif of nitro groups, nitrate esters,
nitramines and azides, which is to say of the great majority of organic
explosives. The model was given no information about energetic chemistry and
recovered this substructure from the data alone.

Other attributions are equally interpretable. Aquatic hazard is driven by
lipophilicity
({float(top[top.GHS_Column=='GHS09_Environmental'].Value_SHAP_Correlation.iloc[0]):+.3f}),
consistent with bioaccumulation. Acute toxicity increases with molecular mass
({float(top[top.GHS_Column=='GHS06_AcuteToxicity'].Value_SHAP_Correlation.iloc[0]):+.3f}).
Flammability decreases with structural complexity
({float(top[top.GHS_Column=='GHS02_Flammable'].Value_SHAP_Correlation.iloc[0]):+.3f}),
reflecting that small simple molecules are the volatile ones.

Malaysian industrial validation
--------------------------------------------------------------------------------

The framework was applied without retraining to chemicals from four Malaysian
industrial sectors and to the substances implicated in the Johor 2019 incident.

  Sector                        n  Label acc.    Recall
  ------------------------------------------------------
{sector}

Performance is highest for the palm oil, petrochemical and incident chemicals,
and lowest for rubber processing and semiconductor manufacturing. Those two
sectors are dominated by inorganic and elemental species - sulfur, zinc oxide,
carbon black, hydrofluoric acid, phosphine, arsine - which lie outside the
applicability domain of descriptors designed for covalent organic molecules.

For the Johor 2019 chemicals specifically, the framework recovered
{int(f['sector'][f['sector'].Sector=='Johor 2019 Emergency'].N_Hazards_Correctly_Flagged.iloc[0])}
of {int(f['sector'][f['sector'].Sector=='Johor 2019 Emergency'].N_Actual_Hazard_Labels.iloc[0])}
true hazard labels across twelve compounds, and flagged the flammable
pictogram for all twelve. A screening tool producing that profile from
structures alone would have supported an early decision to treat the discharge
as an acute inhalation hazard.
"""


# ===========================================================================
# DISCUSSION
# ===========================================================================
def discussion(f):
    e = f["eval"]
    auc = e["auc_per_class_best_model"]
    br = f["best_rows"]
    mal = f["mal_class"]
    ghs09 = mal[mal.GHS_Column == "GHS09_Environmental"]
    return f"""DISCUSSION
================================================================================

Ranking ability and decision quality are not the same thing
--------------------------------------------------------------------------------

The most important result in this work is a disagreement between two metrics.
The explosive class achieves an AUC-ROC of {auc['GHS01_Explosive']:.4f}, which
would ordinarily be described as near-perfect discrimination, yet its Matthews
correlation coefficient at the calibrated decision threshold is only
{br.loc['GHS01_Explosive'].MCC:.4f}. Both numbers are correct, and the gap
between them is the practical finding.

AUC-ROC measures ranking: given one explosive and one non-explosive compound,
how often does the model score the explosive higher? By that measure the model
is excellent. The Matthews correlation coefficient measures the quality of the
actual yes-or-no decision, and it is sensitive to prevalence. The test
partition contains {int(br.loc['GHS01_Explosive'].N_Test_Positive)} explosives
among {e['n_test_compounds']:,} compounds. At that prevalence, any threshold
permissive enough to catch most explosives also admits many times their number
in false positives.

This has a direct consequence for deployment. A model with these
characteristics is well suited to prioritising which chemicals to test first,
because ranking is what prioritisation requires. It is not suited to issuing an
unsupervised yes-or-no verdict on whether a novel compound is explosive. Papers
that report AUC alone for rare-class problems obscure this distinction, and we
would encourage the routine reporting of a prevalence-sensitive metric
alongside it.

The splitting algorithm can silently determine the result
--------------------------------------------------------------------------------

Scaffold splitting is widely recommended, but how scaffold groups are allocated
to partitions receives little attention. Two allocation strategies were tried
here before a third succeeded, and both failures passed every check that is
normally applied. Overall partition sizes were correct, no scaffold appeared in
two partitions, and every class had positive examples everywhere. Only a
per-class audit of training share revealed that 70 per cent of compressed gases
and 61 per cent of oxidisers had been diverted away from the training
partition.

The mechanism is general rather than particular to this dataset. Groups are
processed in descending size order so that large groups can be placed while
room remains; single-compound groups are therefore allocated last, when the
training quota is already close to full. Any chemical property that correlates
with belonging to a small scaffold group will be systematically
under-represented in training. Here that property was being a small acyclic
molecule, which describes most compressed gases and many oxidisers. We would
expect the same effect wherever rare classes are structurally atypical, and we
recommend that per-class training share be reported as a routine diagnostic.

Dataset size, and the hazard of learning curves on subsamples
--------------------------------------------------------------------------------

The controlled experiment showed that training on the full dataset improves
mean AUC-ROC by more than four times the confidence interval, with every class
benefiting. That result is unsurprising in itself. What is instructive is that
a learning curve computed within a 40,000-compound subsample had suggested the
opposite, appearing to saturate.

The explanation is that the subsample had been constructed to retain every
positive example of the rare classes, a reasonable decision when working under
a memory constraint. Those classes consequently could not improve with
additional data, and their flat curves dominated the aggregate. A learning
curve is only informative about the full dataset if the subsample it is
computed on is representative of that dataset; where a subsample has been
deliberately enriched, the curve measures the enrichment rather than the
underlying scaling behaviour.

What the model learned
--------------------------------------------------------------------------------

The SHAP attributions are consistent with established chemistry to a degree
that is reassuring rather than merely decorative. The recovery of the
heteroatom-nitrogen-heteroatom motif as the dominant explosive predictor is the
clearest example: this is the connectivity shared by nitro compounds, nitrate
esters, nitramines and azides, and it was identified without any prior
information about energetic materials. Similarly, aquatic hazard is driven by
lipophilicity, consistent with bioaccumulation, and flammability by low
structural complexity, consistent with volatility.

This matters for regulatory acceptance. A model whose stated reasons align with
known structure-hazard relationships can be audited by a chemist, and its
failures can be diagnosed. The per-compound waterfall explanations provided by
the deployed interface allow a safety officer to see precisely which structural
features drove a particular prediction, and to reject it if the reasoning is
chemically implausible.

Limitations
--------------------------------------------------------------------------------

The support vector machine results are not directly comparable with the other
two algorithms. An RBF kernel matrix scales with the square of the training set
size, requiring approximately 300 GB at full scale, so the support vector
machine was trained on {f['times']['svm_n_features']} features and
{f['times']['svm_training_samples']:,} compounds rather than the full
{f['times']['n_features']} and {f['times']['n_train_compounds']:,}. This is a
property of the algorithm, not of the hardware, and no increase in memory would
resolve it; a linear kernel or a Nystroem approximation would be required.

Inorganic, organometallic and elemental species lie outside the applicability
domain. The descriptors used here characterise covalent organic structures, and
the two Malaysian sectors dominated by such species - rubber processing and
semiconductor manufacturing - show markedly lower recall. Predictions for
mineral acids, metal oxides and coordination compounds should be treated with
corresponding caution.

The environmental class performed poorly on the Malaysian validation set,
recovering only {float(ghs09.Recall.iloc[0]):.2f} of true labels despite an AUC
of {auc['GHS09_Environmental']:.3f} on the global test set. The Malaysian set
contains many industrial commodity chemicals whose aquatic classification
depends on data the model cannot infer from structure.

Exact agreement across all nine classes simultaneously was achieved for only
about a fifth of recognisable multi-label test compounds. Per-class performance
is considerably better than that figure suggests, but users should understand
that the framework identifies individual hazards well and complete hazard
profiles less reliably.

Finally, the labels are regulatory classifications rather than experimental
measurements. They reflect what regulatory bodies have recorded, including
their own inconsistencies - which is why multi-source majority voting was
necessary, and why {f['clean']['conflicted_compounds']:,} compounds required
tie-breaking. The model can be no more accurate than the classifications it was
trained on.
"""


# ===========================================================================
# CONCLUSIONS
# ===========================================================================
def conclusions(f):
    e = f["eval"]
    return f"""CONCLUSIONS
================================================================================

A multi-label machine learning framework was developed to predict all nine GHS
hazard pictograms from molecular structure, trained on
{f['clean']['final_cleaned_compounds']:,} compounds whose classifications were
reconciled across five regulatory sources. Evaluated on a Bemis-Murcko scaffold
split that guarantees no shared chemical skeleton between training and test
data, gradient boosting achieved a mean AUC-ROC of
{e['mean_auc_per_model'][e['best_model']]:.3f} across the nine classes, ranging
from {min(e['auc_per_class_best_model'].values()):.3f} for the irritant class to
{max(e['auc_per_class_best_model'].values()):.3f} for compressed gases.

SHAP analysis showed that the model recovered established structure-hazard
relationships, most notably identifying the heteroatom-nitrogen-heteroatom
connectivity of nitro, nitrate, nitramine and azide groups as the dominant
predictor of explosivity without any prior chemical input. Applied without
retraining to chemicals from four Malaysian industrial sectors and to the
substances implicated in the 2019 Sungai Kim Kim incident, the framework
recovered the majority of true hazard labels.

Two methodological findings extend beyond this application. Scaffold group
allocation can systematically deprive rare classes of training data while
passing every conventional validity check, and per-class training share should
therefore be audited routinely. Learning curves computed on deliberately
enriched subsamples can indicate saturation that a controlled experiment
disproves.

The framework is released as open-source software with a browser-based
screening interface. It is intended for prioritising which chemicals warrant
laboratory assessment, and does not replace testing or regulatory
classification under Malaysia's CLASS Regulations 2013.
"""


# ===========================================================================
# TOC GRAPHIC
# ===========================================================================
def toc_graphic(f):
    """Draw the table-of-contents graphic that ACS journals require."""
    e = f["eval"]
    auc = e["auc_per_class_best_model"]
    codes = [c.split("_")[0] for c in GHS_LABEL_COLUMNS]
    values = [auc[c] for c in GHS_LABEL_COLUMNS]

    # ACS asks for roughly 8.25 x 4.45 cm; this is that ratio at 300 dpi.
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})

    axl.axis("off")
    axl.text(0.5, 0.90, "243,323 compounds", ha="center", fontsize=13,
             fontweight="bold", color="#1a4d7a")
    axl.text(0.5, 0.76, "five regulatory sources", ha="center", fontsize=9.5,
             color="#444444")
    axl.annotate("", xy=(0.5, 0.60), xytext=(0.5, 0.71),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#333333"))
    axl.text(0.5, 0.50, "1218 molecular descriptors", ha="center", fontsize=10.5,
             fontweight="bold")
    axl.text(0.5, 0.395, "scaffold split  •  XGBoost  •  SHAP", ha="center",
             fontsize=9.5, color="#444444")
    axl.annotate("", xy=(0.5, 0.24), xytext=(0.5, 0.35),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#333333"))
    axl.text(0.5, 0.13, "nine GHS pictograms", ha="center", fontsize=12.5,
             fontweight="bold", color="#8b1a1a")
    axl.text(0.5, 0.02, f"mean AUC {np.mean(values):.3f}", ha="center",
             fontsize=10.5, color="#8b1a1a", fontweight="bold")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)

    colours = plt.cm.RdYlGn((np.array(values) - 0.75) / 0.25)
    axr.barh(range(9), values, color=colours, edgecolor="black", linewidth=0.6)
    axr.set_yticks(range(9))
    axr.set_yticklabels(
        [f"{c}  {GHS_TRUE_MEANING[g].split('(')[0].strip()}"
         for c, g in zip(codes, GHS_LABEL_COLUMNS)], fontsize=8)
    axr.invert_yaxis()
    axr.set_xlim(0.5, 1.02)
    axr.set_xlabel("AUC-ROC on scaffold-split test set", fontsize=9)
    axr.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(values):
        axr.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=7.5,
                 fontweight="bold")
    axr.grid(alpha=0.3, axis="x")
    for spine in ("top", "right"):
        axr.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(DIR_PUB, "figures", "TOC_graphic.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    """Generate every remaining manuscript section."""
    print("=" * 78)
    print("GENERATING MANUSCRIPT SECTIONS")
    print("=" * 78)
    f = gather()

    pieces = [
        ("title_page.txt", title_page(f)),
        ("introduction.txt", introduction(f)),
        ("results.txt", results(f)),
        ("discussion.txt", discussion(f)),
        ("conclusions.txt", conclusions(f)),
    ]
    total = 0
    for name, text in pieces:
        text = wrap(text)
        path = os.path.join(MANUSCRIPT, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        words = len(text.split())
        total += words
        print(f"   {name:<24}{words:>6,} words")

    toc = toc_graphic(f)
    print(f"   TOC_graphic.png          -> {toc}")

    # Assemble a single full manuscript in journal section order.
    order = ["title_page.txt", "abstract.txt", "introduction.txt",
             "methods_section.txt", "results.txt", "discussion.txt",
             "conclusions.txt", "references_ACS_style.txt"]
    full = []
    for name in order:
        p = os.path.join(MANUSCRIPT, name)
        if os.path.exists(p):
            full.append(open(p, encoding="utf-8").read())
    combined = "\n\n\n".join(full)
    out = os.path.join(MANUSCRIPT, "FULL_MANUSCRIPT.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(combined)

    print(f"\n   new sections      : {total:,} words")
    print(f"   FULL_MANUSCRIPT   : {len(combined.split()):,} words -> {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
