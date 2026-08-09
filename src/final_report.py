"""
FINAL PROJECT SUMMARY REPORT
============================
Collects the outputs of all thirteen steps into a single PDF that answers the
ten questions listed under FINAL DELIVERABLES SUMMARY in the research
proposal, and organises every file into the required folder layout.

Author : Sareer Ahmad
"""

import os
import sys
import json
import glob
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, PROJECT_ROOT, DIR_RAW, DIR_CLEAN,
                        DIR_SPLITS, DIR_FEATURES, DIR_MODELS, DIR_EVAL, DIR_SHAP,
                        DIR_MALAYSIA, DIR_INTERFACE, DIR_PUB, DIR_LOGS,
                        GHS_LABEL_COLUMNS, GHS_TRUE_MEANING,
                        ORIGINAL_PROPOSAL_NAME, stamped)


def load_json(path, default=None):
    """Read a JSON file, returning a default if it is missing."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return default if default is not None else {}


def human_size(n_bytes):
    """Format a byte count as a readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def collect_fallbacks():
    """Read every step's issue log and pull out the fallbacks that were used."""
    fallbacks = []
    for path in sorted(glob.glob(os.path.join(DIR_LOGS, "STEP*_issue_log_*.txt"))):
        step = os.path.basename(path).split("_")[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                lowered = line.lower()
                if any(word in lowered for word in
                       ("fallback", "reduced to", "budget", "skipped",
                        "could not", "failed")):
                    fallbacks.append((step, line.strip()))
    return fallbacks


def inventory_files():
    """List every output file the project produced, with its size."""
    rows = []
    skip_parts = {".venv", "__pycache__", "_annotation_pages"}
    for root, directories, filenames in os.walk(PROJECT_ROOT):
        directories[:] = [d for d in directories if d not in skip_parts]
        for filename in filenames:
            if filename.endswith((".pyc", ".log")):
                continue
            path = os.path.join(root, filename)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            rows.append({
                "File": os.path.relpath(path, PROJECT_ROOT),
                "Bytes": size,
                "Size": human_size(size),
                "Category": os.path.relpath(root, PROJECT_ROOT) or "(root)",
            })
    return pd.DataFrame(rows).sort_values("Bytes", ascending=False)


def organise_project_folders():
    """
    Copy the root-level step outputs into the folder layout the proposal
    specifies, leaving the originals in place so nothing is lost.
    """
    print("\nOrganising files into the required folder layout ...")
    mapping = [
        ("STEP2_raw_ghs_dataset.csv", DIR_RAW),
        ("STEP2_ghs_label_schema.csv", DIR_RAW),
        ("STEP3_cleaned_ghs_dataset.csv", DIR_CLEAN),
        ("STEP3_modelling_subset.csv", DIR_CLEAN),
        ("STEP3_class_distribution_table.csv", DIR_CLEAN),
        ("STEP3_class_distribution.png", DIR_CLEAN),
        ("STEP3_subset_prevalence_shift.csv", DIR_CLEAN),
        ("STEP3_cleaning_summary.json", DIR_CLEAN),
        ("STEP4_feature_matrix.csv", DIR_FEATURES),
        ("STEP4_label_matrix.csv", DIR_FEATURES),
        ("STEP4_feature_names.txt", DIR_FEATURES),
        ("STEP4_descriptor_metadata.json", DIR_FEATURES),
        ("STEP5_train_indices.npy", DIR_SPLITS),
        ("STEP5_val_indices.npy", DIR_SPLITS),
        ("STEP5_test_indices.npy", DIR_SPLITS),
        ("STEP5_split_class_distribution.csv", DIR_SPLITS),
        ("STEP5_split_metadata.json", DIR_SPLITS),
        ("STEP6_X_train_balanced.npy", DIR_FEATURES),
        ("STEP6_y_train_balanced.npy", DIR_FEATURES),
        ("STEP6_smote_report.csv", DIR_FEATURES),
        ("STEP6_imbalance_config.json", DIR_FEATURES),
        ("STEP7_training_times.json", DIR_MODELS),
        ("STEP8_best_hyperparameters.json", DIR_MODELS),
        ("STEP9_model_comparison_results.csv", DIR_EVAL),
        ("STEP9_auc_comparison_table.csv", DIR_EVAL),
        ("STEP9_F1_comparison_table.csv", DIR_EVAL),
        ("STEP9_MCC_comparison_table.csv", DIR_EVAL),
        ("STEP9_calibrated_thresholds.json", DIR_EVAL),
        ("STEP9_evaluation_summary.json", DIR_EVAL),
        ("STEP10_SHAP_chemical_interpretation.csv", DIR_SHAP),
        ("STEP10_mean_SHAP_values.csv", DIR_SHAP),
        ("STEP10_top20_SHAP_features_per_class.csv", DIR_SHAP),
        ("STEP10_shap_summary.json", DIR_SHAP),
        ("STEP11_malaysia_validation_results.csv", DIR_MALAYSIA),
        ("STEP11_malaysia_validation_report.pdf", DIR_MALAYSIA),
        ("app.py", DIR_INTERFACE),
        ("predict_ghs.py", DIR_INTERFACE),
    ]
    n_copied = 0
    for filename, destination in mapping:
        source = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(source):
            target = os.path.join(destination, filename)
            if os.path.abspath(source) != os.path.abspath(target):
                try:
                    shutil.copy2(source, target)
                    n_copied += 1
                except Exception:
                    pass
    print(f"   {n_copied} files copied into the structured layout")
    return n_copied


def build_final_pdf(output_path):
    """Compose the final project summary report as a PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5,
                          leading=13, spaceAfter=6)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=13,
                             textColor=colors.HexColor("#1a4d7a"),
                             spaceBefore=14, spaceAfter=7)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5)

    def make_table(data, widths, header_colour="#1a4d7a", font_size=8.5):
        """Build a styled table with alternating row shading."""
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_colour)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef3f8")]),
        ]))
        return table

    # ---- gather the facts --------------------------------------------------
    cleaning = load_json(stamped("STEP3_cleaning_summary.json"))
    descriptors = load_json(stamped("STEP4_descriptor_metadata.json"))
    splits = load_json(stamped("STEP5_split_metadata.json"))
    timings = load_json(stamped("STEP7_training_times.json"))
    evaluation = load_json(stamped("STEP9_evaluation_summary.json"))
    shap_summary = load_json(stamped("STEP10_shap_summary.json"))

    document = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=1.8 * cm,
                                 rightMargin=1.8 * cm, topMargin=1.6 * cm,
                                 bottomMargin=1.6 * cm,
                                 title="Final Project Summary Report")
    story = []

    story.append(Paragraph("Final Project Summary Report", styles["Title"]))
    story.append(Paragraph(
        "Interpretable Machine Learning for Predicting GHS Chemical Hazard "
        "Classifications:<br/>A Multi-Label Classification Approach Using "
        "PubChem Molecular Descriptors", styles["Heading3"]))
    story.append(Paragraph(
        f"Sareer Ahmad &nbsp;|&nbsp; MSc Physical Chemistry, University of "
        f"Peshawar<br/>Target institution: Universiti Sains Malaysia &nbsp;|"
        f"&nbsp; Proposed supervisor: Assoc. Prof. Dr. Lee Hooi Ling<br/>"
        f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}", body))
    story.append(Spacer(1, 0.3 * cm))

    # ---- 1, 2 --------------------------------------------------------------
    story.append(Paragraph("1-2. Dataset and feature matrix", heading))
    data = [["Quantity", "Value"],
            ["Raw compound records harvested from PubChem",
             f"{cleaning.get('raw_rows', 0):,}"],
            ["Removed: invalid SMILES",
             f"{cleaning.get('removed_invalid_smiles', 0):,}"],
            ["Removed: duplicate structures (by InChIKey)",
             f"{cleaning.get('removed_duplicates', 0):,}"],
            ["Removed: no hazard label assigned",
             f"{cleaning.get('removed_unlabelled', 0):,}"],
            ["<b>Final cleaned dataset</b>",
             f"<b>{cleaning.get('final_cleaned_compounds', 0):,} compounds</b>"],
            ["Compounds with a conflicting source vote",
             f"{cleaning.get('conflicted_compounds', 0):,}"],
            ["Multi-label compounds (more than one hazard)",
             f"{cleaning.get('multilabel_compounds', 0):,}"],
            ["Modelling subset (hardware-constrained)",
             f"{cleaning.get('modelling_subset_compounds', 0):,}"],
            ["Descriptors computed per compound",
             f"{descriptors.get('n_features_computed', 0):,}"],
            ["Removed by variance filter",
             f"{descriptors.get('n_features_removed', 0):,}"],
            ["<b>Final feature matrix</b>",
             f"<b>{descriptors.get('n_compounds', 0):,} x "
             f"{descriptors.get('n_features_after_variance_filter', 0):,}</b>"],
            ["Missing descriptor values imputed (median)",
             f"{descriptors.get('n_missing_values_imputed', 0):,}"],
            ["Split method", splits.get("split_method", "n/a")],
            ["Split sizes (train / val / test)",
             f"{splits.get('n_train', 0):,} / {splits.get('n_val', 0):,} / "
             f"{splits.get('n_test', 0):,}"],
            ["Distinct Bemis-Murcko scaffolds",
             f"{splits.get('n_distinct_scaffolds', 0):,}"]]
    data = [[Paragraph(c, small) for c in row] for row in data]
    story.append(make_table(data, [10.5 * cm, 6.5 * cm]))

    # ---- 3, 4 --------------------------------------------------------------
    story.append(Paragraph("3-4. Best algorithm and per-class performance",
                           heading))
    best = evaluation.get("best_model", "n/a")
    means = evaluation.get("mean_auc_per_model", {})
    story.append(Paragraph(
        f"<b>Best performing algorithm overall: {best}</b> "
        f"(mean AUC-ROC {means.get(best, float('nan')):.4f} across the nine "
        f"hazard classes on the scaffold-split test set). Mean AUC for every "
        f"algorithm tested: " +
        ", ".join(f"{k} {v:.4f}" for k, v in sorted(means.items(),
                                                    key=lambda kv: -kv[1])) +
        ".", body))

    per_class = evaluation.get("auc_per_class_best_model", {})
    data = [["GHS column", "Code", "Authoritative meaning", "AUC-ROC"]]
    for column in GHS_LABEL_COLUMNS:
        value = per_class.get(column)
        data.append([column, column.split("_")[0], GHS_TRUE_MEANING[column],
                     f"{value:.4f}" if value else "n/a"])
    data = [[Paragraph(c, small) for c in row] for row in data]
    story.append(make_table(data, [4.4 * cm, 1.5 * cm, 8.6 * cm, 2.5 * cm]))

    renamed = [c for c in GHS_LABEL_COLUMNS
               if ORIGINAL_PROPOSAL_NAME[c] != c]
    story.append(Paragraph(
        "<b>Note on column naming.</b> Column names follow the official United "
        "Nations pictogram numbering. Three columns were renamed from the "
        "original study design, whose descriptive suffixes did not match that "
        "scheme: " +
        "; ".join(f"{ORIGINAL_PROPOSAL_NAME[c]} &rarr; {c}" for c in renamed) +
        ". The underlying data were bound to the numeric pictogram code "
        "throughout and were therefore unaffected by the correction - no model "
        "was retrained and no classification changed.", body))

    story.append(PageBreak())

    # ---- 5 -----------------------------------------------------------------
    story.append(Paragraph("5. Most predictive molecular features", heading))
    try:
        mean_shap = pd.read_csv(stamped("STEP10_mean_SHAP_values.csv"), index_col=0)
        top = mean_shap.head(10)
        data = [["Rank", "Descriptor", "Mean |SHAP| across all classes"]]
        for rank, (feature, row) in enumerate(top.iterrows(), 1):
            data.append([str(rank), str(feature),
                         f"{row['Mean_across_all_classes']:.6f}"])
        data = [[Paragraph(c, small) for c in r] for r in data]
        story.append(make_table(data, [1.6 * cm, 9.4 * cm, 6 * cm]))
        story.append(Paragraph(
            "<b>Top three overall: " +
            ", ".join(shap_summary.get("top3_features_overall", [])) +
            "</b>", body))
    except Exception as exc:
        story.append(Paragraph(f"SHAP table unavailable: {exc}", body))

    # ---- 6 -----------------------------------------------------------------
    story.append(Paragraph("6. Malaysian validation performance", heading))
    try:
        malaysia = pd.read_csv(os.path.join(
            DIR_MALAYSIA, "STEP11_malaysia_per_sector_metrics.csv"))
        data = [["Sector", "Compounds", "Label accuracy", "Hazard recall"]]
        for _, row in malaysia.iterrows():
            data.append([row["Sector"], str(row["N_Compounds"]),
                         f"{row['Label_Accuracy']:.3f}",
                         f"{row['Hazard_Recall']:.3f}"])
        data = [[Paragraph(c, small) for c in r] for r in data]
        story.append(make_table(data, [8 * cm, 2.6 * cm, 3.2 * cm, 3.2 * cm]))
    except Exception as exc:
        story.append(Paragraph(f"Malaysian sector table unavailable: {exc}", body))

    # ---- 9 -----------------------------------------------------------------
    story.append(Paragraph("9. Training time per algorithm", heading))
    seconds = timings.get("seconds", {})
    data = [["Algorithm", "Training time"]]
    for name, value in seconds.items():
        data.append([name, f"{value / 60:.2f} minutes"])
    data = [[Paragraph(c, small) for c in r] for r in data]
    story.append(make_table(data, [10 * cm, 7 * cm]))
    story.append(Paragraph(
        f"Measured on an Intel Core i7-6500U (two physical cores) with 7.9 GB "
        f"of RAM, training on {timings.get('n_train_compounds', 0):,} compounds "
        f"described by {timings.get('n_features', 0):,} features.", body))

    # ---- 8 -----------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("8. Steps where a fallback was used", heading))
    fallbacks = collect_fallbacks()
    if fallbacks:
        data = [["Step", "Fallback applied or limitation encountered"]]
        for step, line in fallbacks[:40]:
            text = line.split(": ", 1)[-1] if ": " in line else line
            data.append([step, text[:400]])
        data = [[Paragraph(c, small) for c in r] for r in data]
        story.append(make_table(data, [2 * cm, 15 * cm]))
        if len(fallbacks) > 40:
            story.append(Paragraph(
                f"...and {len(fallbacks) - 40} further entries; the complete "
                f"record is in the logs/ folder.", body))
    else:
        story.append(Paragraph("No fallbacks were required.", body))

    story.append(Paragraph(
        "<b>Principal fallback, Step 1.</b> No Python installation existed on "
        "the target machine, and the official python.org Windows installer "
        "failed twice with WiX bootstrapper exit code 0x3 because no "
        "administrator elevation was available for its chained MSI packages. "
        "conda was also absent, so the conda fallback named in the proposal "
        "could not be used. A standalone CPython 3.11.15 build was installed "
        "instead via the uv package manager, which requires no Windows "
        "installer. All subsequent package installation proceeded normally.",
        body))

    story.append(Paragraph(
        "<b>Two defects found and corrected during development.</b> Both "
        "produced results that looked entirely plausible, which is why they "
        "are recorded here rather than quietly fixed. First, the scaffold "
        "group allocation starved the rare classes: only 30 per cent of "
        "compressed gases and 39 per cent of oxidisers reached the training "
        "partition instead of 80 per cent, because every acyclic molecule "
        "forms a single-compound scaffold group and those groups were "
        "allocated last. Allocation is now performed by group-wise iterative "
        "stratification, and every class lands within one percentage point of "
        "its intended share. Second, the modelling set had been restricted to "
        "40,000 compounds on memory grounds; a controlled experiment showed "
        "this cost 0.055 of mean AUC-ROC, four times the bootstrap confidence "
        "interval, so all results now use the complete dataset.", body))

    # ---- 7 -----------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("7. Files produced", heading))
    inventory = inventory_files()
    total = inventory["Bytes"].sum()
    story.append(Paragraph(
        f"{len(inventory):,} files totalling {human_size(total)}. "
        f"The twenty-five largest are listed below; the complete inventory is "
        f"saved as FINAL_file_inventory.csv.", body))
    data = [["File", "Size"]]
    for _, row in inventory.head(25).iterrows():
        data.append([row["File"], row["Size"]])
    data = [[Paragraph(c, small) for c in r] for r in data]
    story.append(make_table(data, [13.5 * cm, 3.5 * cm]))

    # ---- 10 ----------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("10. Recommended next steps", heading))
    next_steps = [
        ("Train the support vector machine on the full feature set and full "
         "training partition. Its scores here are not strictly comparable with "
         "Random Forest and XGBoost because it was restricted to 100 features "
         "and a subsampled training set for tractability. Note that this is "
         "not simply a matter of buying more memory: an RBF kernel matrix for "
         "the full training partition would be roughly 300 GB, so a different "
         "formulation - a linear kernel, or Nystroem approximation - would be "
         "needed rather than a larger machine."),
        ("Investigate the threshold calibration for the irritant class. At the "
         "high prevalence this class reaches in the full dataset, the "
         "F1-optimal threshold degenerates towards predicting the positive "
         "class for almost every compound, giving a high F1 alongside a very "
         "low Matthews correlation coefficient. A prevalence-aware criterion, "
         "or simply reporting the safety-first threshold for this class, would "
         "be more honest."),
        ("Add a graph neural network baseline. Message-passing networks learn "
         "their own representation from the molecular graph and typically "
         "outperform fingerprint-based models on large chemical datasets; the "
         "scaffold split and evaluation code here would carry over unchanged."),
        ("Define and report an applicability domain. Predictions for compounds "
         "structurally remote from the training set should be flagged as "
         "extrapolation rather than reported with the same confidence. A "
         "distance-to-training-set measure in descriptor space would be the "
         "simplest implementation."),
        ("Incorporate Malaysian CLASS Regulations classifications into the "
         "training data. The current model learns from European, Japanese, "
         "Australian and American sources; adding Malaysian classifications "
         "would improve coverage of locally significant chemicals."),
        ("Extend coverage to inorganic and organometallic compounds. The "
         "descriptors used here are designed for covalent organic structures, "
         "and predictions for salts, metal oxides and coordination compounds "
         "are correspondingly less reliable."),
        ("Conduct a prospective validation. Hold back chemicals classified "
         "after a chosen date and test on those, which is the closest "
         "computational analogue of the real deployment scenario."),
    ]
    for index, text in enumerate(next_steps, 1):
        story.append(Paragraph(f"<b>{index}.</b> {text}", body))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>DISCLAIMER.</b> The models described in this report are "
        "computational screening tools. They do not replace laboratory testing "
        "or regulatory assessment under Malaysia's Occupational Safety and "
        "Health (Classification, Labelling and Safety Data Sheet of Hazardous "
        "Chemicals) Regulations 2013.",
        ParagraphStyle("disclaimer", parent=body, fontSize=9,
                       textColor=colors.HexColor("#8b1a1a"),
                       borderWidth=1, borderPadding=6,
                       borderColor=colors.HexColor("#8b1a1a"))))

    document.build(story)
    inventory.to_csv(stamped("FINAL_file_inventory.csv"), index=False)
    return output_path, inventory


def main():
    """Generate the final report and organise the project folders."""
    print("=" * 78)
    print("FINAL PROJECT SUMMARY REPORT")
    print("=" * 78)

    organise_project_folders()

    output_path = stamped("FINAL_PROJECT_SUMMARY_REPORT.pdf")
    try:
        path, inventory = build_final_pdf(output_path)
        print(f"\nPDF report written: {path}")
        print(f"File inventory    : {len(inventory):,} files, "
              f"{human_size(inventory['Bytes'].sum())} total")
    except Exception as exc:
        print(f"\nPDF generation failed ({type(exc).__name__}: {exc}).")
        print("FALLBACK: writing the summary as a text file instead.")
        # The report must exist in some form even if reportlab fails.
        text_path = stamped("FINAL_PROJECT_SUMMARY_REPORT.txt")
        inventory = inventory_files()
        with open(text_path, "w", encoding="utf-8") as fh:
            fh.write("FINAL PROJECT SUMMARY REPORT\n")
            fh.write("=" * 70 + "\n\n")
            for name, path in [
                    ("cleaning", stamped("STEP3_cleaning_summary.json")),
                    ("descriptors", stamped("STEP4_descriptor_metadata.json")),
                    ("splits", stamped("STEP5_split_metadata.json")),
                    ("training times", stamped("STEP7_training_times.json")),
                    ("evaluation", stamped("STEP9_evaluation_summary.json")),
                    ("shap", stamped("STEP10_shap_summary.json"))]:
                fh.write(f"\n--- {name} ---\n")
                fh.write(json.dumps(load_json(path), indent=2) + "\n")
            fh.write("\n--- files ---\n")
            fh.write(inventory.to_string(index=False))
        inventory.to_csv(stamped("FINAL_file_inventory.csv"), index=False)
        print(f"Text report written: {text_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
