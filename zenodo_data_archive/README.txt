==============================================================================
DATA ARCHIVE
Interpretable Machine Learning for Predicting GHS Chemical
Hazard Classifications
==============================================================================

Sareer Ahmad
Department of Chemistry, University of Peshawar
ORCID: https://orcid.org/0009-0003-2580-091X
Archive created: 10 August 2026

Analysis code: https://github.com/sareer555/ghs-hazard-classification

------------------------------------------------------------------------------
WHAT THIS IS
------------------------------------------------------------------------------

GHS hazard classifications for 243,323 unique chemical compounds, harvested from PubChem and
reconciled across five independent regulatory sources: the European
Chemicals Agency, Regulation (EC) No 1272/2008, the Hazardous Substances
Data Bank, NITE-CMC and the Hazardous Chemical Information System of Safe
Work Australia. Also included are the computed molecular descriptors, the
trained models, and every result table in the paper.

Best model: XGBoost, mean AUC-ROC 0.915 across the nine hazard
classes, on a Bemis-Murcko scaffold split (194,619 train / 24,352 validation /
24,352 test) in which no chemical skeleton is shared between
splits.

------------------------------------------------------------------------------
THE NINE LABEL COLUMNS
------------------------------------------------------------------------------

Names follow official United Nations GHS pictogram numbering.

  GHS01_Explosive
  GHS02_Flammable
  GHS03_Oxidising
  GHS04_CompressedGas
  GHS05_Corrosive
  GHS06_AcuteToxicity
  GHS07_Irritant
  GHS08_HealthHazard
  GHS09_Environmental

A value of 1 means the compound carries that pictogram, 0 that it does
not. Note that 0 means 'not assigned', which is not the same as
'demonstrated to be safe' - a compound may simply not have been assessed
for that hazard.

------------------------------------------------------------------------------
FILES
------------------------------------------------------------------------------

STEP2_raw_ghs_dataset.csv  (61.2 MB)
    Raw dataset: every compound PubChem holds a GHS classification for, with S
    MILES, formula, InChIKey, CAS where available, the nine binary hazard labe
    ls, and which regulatory sources contributed.

STEP2_ghs_label_schema.csv  (0.0 MB)
    Authoritative meaning of each of the nine label columns, with the correspo
    ndence to the names used in the original study design.

STEP3_cleaned_ghs_dataset.csv  (80.3 MB)
    Cleaned dataset used for all modelling: SMILES validated with RDKit, dupli
    cates removed by InChIKey, disagreements between regulatory sources resolv
    ed by majority vote. This is the file to start from.

STEP3_class_distribution_table.csv  (0.0 MB)
    Counts, percentages and imbalance ratios per hazard class.

STEP4_feature_matrix.csv  (68.8 MB)
    The 816 molecular descriptors retained after variance filtering, one row p
    er compound.

STEP4_label_matrix.csv  (6.7 MB)
    The nine binary hazard labels, aligned row-for-row with the feature matrix
    .

STEP4_feature_names.txt  (0.0 MB)
    Descriptor names, in feature-matrix column order.

STEP4_compound_index.csv  (45.4 MB)
    Compound identity for each feature-matrix row: CID, name, SMILES, InChIKey
    , formula.

STEP5_split_class_distribution.csv  (0.0 MB)
    Per-class positive counts in the training, validation and test splits.

STEP8_rf_tuned.pkl  (94.8 MB)
    Tuned Random Forest, refitted on the training split.

STEP7_rf_model.pkl  (108.0 MB)
    Random Forest with balanced class weighting (Step 7 defaults).

STEP7_rf_noclassweight_ablation.pkl  (102.3 MB)
    Ablation: the same Random Forest without class weighting. Note that SMOTE 
    was NOT applied at full dataset scale - see the Methods.

STEP7_svm_model.pkl  (3.4 MB)
    Support vector machine, trained on 100 features and a subsampled training 
    set; see the Limitations.

STEP8_xgb_tuned.pkl  (5.8 MB)
    Tuned XGBoost - the best model, and the one the application uses. Also in 
    the GitHub repository.

STEP9_model_comparison_results.csv  (0.0 MB)
    Full evaluation: every model, class, metric and decision threshold, with b
    ootstrap confidence intervals.

STEP9_calibrated_thresholds.json  (0.0 MB)
    Decision thresholds fitted on validation data, per model and class.

STEP10_mean_SHAP_values.csv  (0.1 MB)
    Mean absolute SHAP value per descriptor per hazard class.

STEP10_SHAP_chemical_interpretation.csv  (0.0 MB)
    Top five descriptors per class with chemical interpretation.

STEP11_malaysia_validation_results.csv  (0.0 MB)
    Predictions and reference labels for the Malaysian industrial chemicals an
    d the Johor 2019 incident compounds.

------------------------------------------------------------------------------
WHAT IS NOT HERE, AND WHY
------------------------------------------------------------------------------

About 8 GB of intermediate NumPy arrays are omitted. They are
reproducible exactly by re-running Steps 4 and 6 of the pipeline, which
takes minutes; downloading them would take considerably longer.

------------------------------------------------------------------------------
REUSE
------------------------------------------------------------------------------

The code is MIT licensed. The underlying classifications originate with
the regulatory bodies named above and remain subject to their terms.
PubChem data are in the public domain.

DISCLAIMER. These models are computational screening tools. They do not
replace laboratory testing or regulatory assessment under Malaysia's
CLASS Regulations 2013 or equivalent legislation elsewhere. Predictions
must not be the sole basis for any decision affecting human safety.
