"""
FINAL DELIVERABLES AUDIT
========================
Checks that every artefact the research proposal asks for actually exists on
disk and is non-empty. Run last, after the whole pipeline.

Author : Sareer Ahmad
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import PROJECT_ROOT, GHS_LABEL_COLUMNS

PRESENT, MISSING = [], []


def need(description, pattern, expected_count=1):
    """Assert that at least `expected_count` files match the pattern."""
    matches = [p for p in glob.glob(os.path.join(PROJECT_ROOT, pattern))
               if os.path.getsize(p) > 0]
    ok = len(matches) >= expected_count
    (PRESENT if ok else MISSING).append(description)
    marker = "OK  " if ok else "MISS"
    detail = (f"{len(matches)} file(s)" if expected_count > 1
              else os.path.basename(matches[0]) if matches else "not found")
    print(f"  [{marker}] {description:<52} {detail}")
    return ok


def main():
    """Audit every step's required outputs."""
    print("=" * 78)
    print("FINAL DELIVERABLES AUDIT")
    print("=" * 78)

    print("\nSTEP 1 - Environment")
    need("environment requirements file", "STEP1_environment_requirements.txt")

    print("\nSTEP 2 - Data collection")
    need("raw GHS dataset", "STEP2_raw_ghs_dataset.csv")
    need("GHS label schema (documents the numbering)", "STEP2_ghs_label_schema.csv")
    need("per-source records (for majority voting)",
         "data/raw/STEP2_per_source_records_*.csv")

    print("\nSTEP 3 - Cleaning")
    need("cleaned dataset", "STEP3_cleaned_ghs_dataset.csv")
    need("class distribution chart", "STEP3_class_distribution.png")
    need("class distribution table", "STEP3_class_distribution_table.csv")
    need("modelling subset", "STEP3_modelling_subset.csv")
    need("subset prevalence shift", "STEP3_subset_prevalence_shift.csv")

    print("\nSTEP 4 - Descriptors")
    need("feature matrix (CSV)", "STEP4_feature_matrix.csv")
    need("label matrix (CSV)", "STEP4_label_matrix.csv")
    need("feature names", "STEP4_feature_names.txt")
    need("feature matrix (NPY)", "features/STEP4_X.npy")
    need("label matrix (NPY)", "features/STEP4_y.npy")

    print("\nSTEP 5 - Scaffold split")
    need("train indices", "STEP5_train_indices.npy")
    need("validation indices", "STEP5_val_indices.npy")
    need("test indices", "STEP5_test_indices.npy")
    need("per-split class distribution", "STEP5_split_class_distribution.csv")
    need("scaffold assignments", "data/splits/STEP5_scaffold_assignments_*.csv")

    print("\nSTEP 6 - Class imbalance")
    need("balanced training data (X)", "STEP6_X_train_balanced.npy")
    need("balanced training data (y)", "STEP6_y_train_balanced.npy")
    need("per-class balanced sets", "features/STEP6_balanced/X_*.npy", 9)
    need("SMOTE report", "STEP6_smote_report.csv")
    need("imbalance config (weights, metrics)", "STEP6_imbalance_config.json")

    print("\nSTEP 7 - Model training")
    need("Random Forest model", "models/STEP7_rf_model.pkl")
    need("XGBoost models", "models/STEP7_xgb_models.pkl")
    need("SVM model", "models/STEP7_svm_model.pkl")
    need("SMOTE ablation model", "models/STEP7_rf_smote_ablation.pkl")
    need("training times", "STEP7_training_times.json")

    print("\nSTEP 8 - Hyperparameter tuning")
    need("best hyperparameters", "STEP8_best_hyperparameters.json")
    need("tuned Random Forest (refit on train)", "models/STEP8_rf_tuned.pkl")
    need("tuned XGBoost (refit on train)", "models/STEP8_xgb_tuned.pkl")

    print("\nSTEP 9 - Evaluation")
    need("full results table", "STEP9_model_comparison_results.csv")
    need("AUC comparison table", "STEP9_auc_comparison_table.csv")
    need("F1 comparison table", "STEP9_F1_comparison_table.csv")
    need("MCC comparison table", "STEP9_MCC_comparison_table.csv")
    need("calibrated thresholds", "STEP9_calibrated_thresholds.json")
    need("evaluation summary", "STEP9_evaluation_summary.json")
    need("ROC curves (one per class)", "evaluation/STEP9_ROC_curves_GHS*.png", 9)
    need("PR curves (one per class)", "evaluation/STEP9_PR_curves_GHS*.png", 9)
    need("confusion matrices (one per class)",
         "evaluation/STEP9_confusion_matrix_GHS*.png", 9)

    print("\nSTEP 10 - SHAP interpretability")
    need("SHAP values archive", "shap_analysis/STEP10_shap_values.npz")
    need("SHAP summary bar plots", "shap_analysis/STEP10_SHAP_summary_GHS*.png", 9)
    need("SHAP beeswarm plots", "shap_analysis/STEP10_SHAP_beeswarm_GHS*.png", 9)
    need("SHAP waterfall plots", "shap_analysis/STEP10_SHAP_waterfall_*.png", 5)
    need("chemical interpretation table",
         "STEP10_SHAP_chemical_interpretation.csv")
    need("mean SHAP values table", "STEP10_mean_SHAP_values.csv")
    need("top-20 SHAP features per class",
         "STEP10_top20_SHAP_features_per_class.csv")

    print("\nSTEP 11 - Malaysian validation")
    need("Malaysian validation results", "STEP11_malaysia_validation_results.csv")
    need("Malaysian validation PDF report",
         "STEP11_malaysia_validation_report.pdf")
    need("per-sector metrics",
         "malaysia_validation/STEP11_malaysia_per_sector_metrics.csv")
    need("Johor 2019 predictions",
         "malaysia_validation/STEP11_johor_2019_predictions.csv")
    need("Malaysian performance figure",
         "malaysia_validation/STEP11_malaysia_performance_comparison.png")

    print("\nSTEP 12 - Interface")
    need("Streamlit application", "app.py")
    need("command-line tool", "predict_ghs.py")
    need("shared prediction engine", "src/ghs_predictor.py")

    print("\nSTEP 13 - Publication materials")
    need("publication figures (300 dpi)",
         "publication_materials/figures/Figure*.png", 8)
    need("figure captions", "publication_materials/figures/figure_captions.txt")
    need("supplementary tables workbook",
         "publication_materials/tables/publication_supplementary_tables.xlsx")
    need("abstract", "publication_materials/manuscript/abstract.txt")
    need("methods section", "publication_materials/manuscript/methods_section.txt")
    need("ACS reference list",
         "publication_materials/manuscript/references_ACS_style.txt")
    need("submission checklist",
         "publication_materials/manuscript/submission_checklist.txt")

    print("\nFINAL")
    need("final project summary report", "FINAL_PROJECT_SUMMARY_REPORT.pdf")
    need("file inventory", "FINAL_file_inventory.csv")
    need("README", "README.md")
    need("per-step issue logs", "logs/STEP*_issue_log_*.txt", 8)

    print("\n" + "=" * 78)
    print(f"DELIVERABLES AUDIT: {len(PRESENT)} present, {len(MISSING)} missing")
    if MISSING:
        print("MISSING:")
        for item in MISSING:
            print(f"   - {item}")
    print("=" * 78)
    return 0 if not MISSING else 1


if __name__ == "__main__":
    sys.exit(main())
