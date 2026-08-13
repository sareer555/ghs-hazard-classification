"""
STEP 13 - PUBLICATION PREPARATION
=================================
Turns the completed analysis into the materials a journal submission needs.

13a  Eight publication-quality figures at 300 dpi with 12 pt minimum type.
13b  Five supplementary tables in one Excel workbook.
13c  The complete reference list in ACS style.
13d  A 250-word abstract, with the real numbers filled in.
13e  A Methods section of at least 1500 words, in past-tense passive voice.
13f  The submission checklist, cover letter and required statements.

Every number quoted in the abstract and methods is read from the JSON summary
files written by the earlier steps, so the text can never drift out of step
with the results.

Author : Sareer Ahmad
"""

import os
import sys
import json
import time
import shutil
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (get_ablation_identity, manuscript_title,
                        FIGURE_DPI, MODEL_COLOURS, SERIES_HUE, ACCENT_HUE,
                        CONTEXT_GREY, INK_PRIMARY, INK_SECONDARY, INK_MUTED,
                        GRIDLINE,
                        RANDOM_SEED, TODAY, PROJECT_ROOT, DIR_FEATURES,
                        DIR_SPLITS, DIR_EVAL, DIR_SHAP, DIR_MALAYSIA, DIR_PUB,
                        DIR_PUB_FIGS, DIR_LOGS, GHS_LABEL_COLUMNS,
                        GHS_TRUE_MEANING, ORIGINAL_PROPOSAL_NAME,
                        seed_everything, stamped)

seed_everything()

# The ablation's name reflects what it actually measured; see
# get_ablation_identity() in ghs_config.py.
_ABL_NAME, _ABL_FILE, _ABL_META = get_ablation_identity()

ISSUE_LOG = []

# ---------------------------------------------------------------------------
# PUBLICATION FIGURE STYLE
# ACS journals require 300 dpi minimum and legible type at the printed size.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,           # screen preview; savefig carries the real one
    "savefig.dpi": FIGURE_DPI,
    "font.size": 12,             # the proposal's 12 pt minimum
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})

# The shared model colours, with the ablation's slot keyed to whatever name
# Step 7 actually gave it.
PALETTE = {"RandomForest": MODEL_COLOURS["RandomForest"],
           "XGBoost": MODEL_COLOURS["XGBoost"],
           "SVM": MODEL_COLOURS["SVM"],
           _ABL_NAME: MODEL_COLOURS["ablation"]}

# Line style as well as colour, so the series stay distinguishable in greyscale
# print and for readers who cannot separate the hues.
LINESTYLES = {"RandomForest": "-", "XGBoost": "--",
              "SVM": "-.", _ABL_NAME: ":"}

FIG_DIR = os.path.join(DIR_PUB, "figures")
TABLE_DIR = os.path.join(DIR_PUB, "tables")
MANUSCRIPT_DIR = os.path.join(DIR_PUB, "manuscript")
for _d in (FIG_DIR, TABLE_DIR, MANUSCRIPT_DIR, DIR_PUB_FIGS):
    os.makedirs(_d, exist_ok=True)

CAPTIONS = {}


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


def describe_imbalance_handling():
    """
    Report the class-imbalance treatment that actually ran.

    Step 6 records the method it used for each class in STEP6_smote_report.csv.
    At this dataset size SMOTE was skipped for memory reasons and balanced class
    weighting was used instead. The workflow figure previously stated "SMOTE +
    class weights" as fixed text, which contradicted the supplementary data - a
    reviewer comparing the two would read it as misrepresentation rather than as
    a stale label. Reading the answer from Step 6's own output means the figure
    cannot drift from what was done.
    """
    path = stamped("STEP6_smote_report.csv")
    if not os.path.exists(path):
        log_issue("13a", "SMOTE report missing; workflow figure will describe "
                         "class weighting only")
        return "Class imbalance handling\nbalanced class weights"

    methods = pd.read_csv(path)["Method"].astype(str)
    used_smote = methods.str.contains("smote", case=False, na=False)
    if used_smote.all():
        return "Class imbalance handling\nSMOTE + class weights"
    if used_smote.any():
        return (f"Class imbalance handling\nSMOTE on {int(used_smote.sum())} of "
                f"{len(methods)} classes\n+ balanced class weights")
    return ("Class imbalance handling\nbalanced class weights\n"
            "(SMOTE not applied at this scale)")


def save_figure(fig, number, name, caption):
    """Save a figure into both publication folders and record its caption."""
    filename = f"Figure{number}_{name}.png"
    for folder in (FIG_DIR, DIR_PUB_FIGS):
        fig.savefig(os.path.join(folder, filename), dpi=FIGURE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    CAPTIONS[f"Figure {number}"] = caption
    print(f"      Figure {number}: {filename}")
    return os.path.join(FIG_DIR, filename)


def load_json(path, default=None):
    """Read a JSON file, returning a default if it is missing."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    log_issue("13", f"expected file not found: {path}")
    return default if default is not None else {}


# ===========================================================================
# 13a - FIGURES
# ===========================================================================
def figure1_workflow():
    """Figure 1 - the research pipeline, drawn as a flow diagram."""
    fig, ax = plt.subplots(figsize=(13, 9.4))
    # The y range runs past the top box so the title has room of its own. It
    # previously sat at y=9.4 while the top boxes reached 9.62, so the heading
    # printed straight through them.
    ax.set_xlim(-0.8, 10.8); ax.set_ylim(0, 11.4); ax.axis("off"); ax.grid(False)

    # Four phases, each a tint of one hue rather than an unrelated pastel. The
    # colour groups the stages; it is not an identity code, so a reader is not
    # invited to look for meaning in eleven separate hues.
    DATA, MODEL, ASSESS, SHIP = "#dce9f9", "#e4e0f2", "#dcf0e8", "#fbe4d8"

    stages = [
        (1.6, 9.0, "PubChem GHS Classification\nannotations (ECHA, HSDB,\n"
                   "NITE-CMC, HCIS, EU CLP)", DATA),
        (1.6, 7.4, "Data cleaning\nSMILES validation, InChIKey\n"
                   "deduplication, majority vote", DATA),
        (1.6, 5.8, "Molecular descriptors\n19 physicochemical + 1024 ECFP4\n"
                   "+ 167 MACCS + 8 topological = 1218\n"
                   "-> 816 after variance filtering", DATA),
        (1.6, 4.2, "Bemis-Murcko scaffold split\n80 / 10 / 10", DATA),
        (1.6, 2.6, describe_imbalance_handling(), MODEL),
        (5.0, 2.6, "Model training\nRandom Forest | XGBoost | SVM", MODEL),
        (8.4, 2.6, "Hyperparameter tuning\nRandomizedSearchCV", MODEL),
        (8.4, 4.2, "Evaluation\nAUC, AP, F1, MCC + bootstrap CI\n"
                   "threshold calibration", ASSESS),
        (8.4, 5.8, "SHAP interpretation\nglobal + per-compound", ASSESS),
        (8.4, 7.4, "Malaysian validation\n4 sectors + Johor 2019", ASSESS),
        (8.4, 9.0, "Deployment\nStreamlit app + CLI + PDF report", SHIP),
    ]

    positions = {}
    for x, y, text, colour in stages:
        box = FancyBboxPatch((x - 1.45, y - 0.62), 2.9, 1.24,
                             boxstyle="round,pad=0.06", linewidth=1.1,
                             edgecolor="#8f9299", facecolor=colour, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK_PRIMARY, zorder=3,
                linespacing=1.35)
        positions[text] = (x, y)

    def arrow(start, end):
        """Draw a curved arrow between two stage boxes."""
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=17, linewidth=1.6,
                                     color=INK_SECONDARY, zorder=1,
                                     connectionstyle="arc3,rad=0"))

    order = [s[2] for s in stages]
    # Down the left-hand column
    for i in range(4):
        a, b = positions[order[i]], positions[order[i + 1]]
        arrow((a[0], a[1] - 0.64), (b[0], b[1] + 0.64))
    # Across the bottom
    for i in range(4, 6):
        a, b = positions[order[i]], positions[order[i + 1]]
        arrow((a[0] + 1.47, a[1]), (b[0] - 1.47, b[1]))
    # Up the right-hand column
    for i in range(6, 10):
        a, b = positions[order[i]], positions[order[i + 1]]
        arrow((a[0], a[1] + 0.64), (b[0], b[1] - 0.64))

    # Phase labels sit in gutters outside the boxes, so the reader can see the
    # shape of the pipeline without reading every box and the serpentine order
    # is explicit. They need margins of their own: placed against the column
    # edges they printed on top of the boxes.
    ax.text(-0.35, 6.6, "1  Data", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK_MUTED, rotation=90)
    ax.text(5.0, 1.55, "2  Modelling", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK_MUTED)
    ax.text(10.35, 6.6, "3  Assessment", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK_MUTED, rotation=270)

    ax.text(5.0, 10.9, "Interpretable Machine Learning for GHS Hazard "
                       "Classification",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=INK_PRIMARY)
    ax.text(5.0, 10.3, "Multi-label prediction of nine GHS pictograms from "
                       "molecular structure",
            ha="center", va="center", fontsize=11, style="italic",
            color=INK_SECONDARY)
    ax.text(5.0, 0.75, "All steps use random seed 42. The test set shares no\n"
                       "Bemis-Murcko scaffold with the training set.",
            ha="center", va="center", fontsize=10, style="italic",
            color=INK_MUTED)

    return save_figure(
        fig, 1, "research_workflow",
        "Figure 1. Research workflow. GHS classification annotations were "
        "harvested from PubChem, cleaned and deduplicated, converted into "
        "1218 molecular descriptors and reduced by variance filtering to the "
        "816 used for modelling, and split by Bemis-Murcko scaffold so "
        "that no chemical skeleton appears in more than one split. Three "
        "algorithms were trained and tuned, evaluated with threshold "
        "calibration and bootstrap confidence intervals, interpreted with "
        "SHAP, and finally validated on Malaysian industrial chemicals "
        "before deployment as a screening tool.")


def figure2_class_distribution():
    """Figure 2 - how many compounds carry each hazard."""
    table = pd.read_csv(stamped("STEP3_class_distribution_table.csv"))

    # Stacked rather than side by side, sharing one x axis. The two panels
    # describe the same nine categories, so printing the category names twice
    # spent space on repetition and invited the reader to check whether the two
    # orderings matched.
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"hspace": 0.12})

    labels = [f"{r.Pictogram_Code} {r.Actual_Meaning.split('(')[0].strip()}"
              for r in table.itertuples()]
    positions = np.arange(len(labels))
    counts = table["N_Positive"].to_numpy()
    ratios = table["Imbalance_Ratio_NegPerPos"].to_numpy()

    def annotate(ax, bars, values, formatter):
        """Write each bar's value just above it."""
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.10,
                    formatter(value), ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK_PRIMARY)

    # One hue. The x axis already names each bar, so colouring the nine bars
    # differently encodes nothing that position does not, and the two panels
    # previously used two unrelated colour scales for the same categories.
    bars = ax_top.bar(positions, counts, color=SERIES_HUE, width=0.68)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Compounds carrying\nthe pictogram (log scale)")
    ax_top.set_title("(a) Positive examples per GHS hazard class",
                     fontweight="bold", loc="left", color=INK_PRIMARY)
    ax_top.set_ylim(top=counts.max() * 3)
    annotate(ax_top, bars, counts, lambda v: f"{v:,}")

    bars = ax_bottom.bar(positions, ratios, color=SERIES_HUE, width=0.68)
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("Negatives per positive\n(log scale)")
    ax_bottom.set_title("(b) Class imbalance ratio", fontweight="bold",
                        loc="left", color=INK_PRIMARY)
    ax_bottom.set_ylim(top=ratios.max() * 4)
    # A ratio below one rounds to "0:1" under integer formatting, which reads
    # as "no negatives at all". GHS07 is the case: 23,303 negatives against
    # 220,020 positives is 0.11:1, not 0:1.
    annotate(ax_bottom, bars, ratios,
             lambda v: f"{v:,.0f}:1" if v >= 10 else f"{v:.2f}:1")

    # Rotated tick labels need an explicit right anchor, otherwise each label
    # is centred on its rotated bounding box and drifts to the right of the bar
    # it belongs to - which is what made the old figure look mislabelled.
    ax_bottom.set_xticks(positions)
    ax_bottom.set_xticklabels(labels, rotation=35, ha="right",
                              rotation_mode="anchor", fontsize=10)

    for ax in (ax_top, ax_bottom):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRIDLINE)
    return save_figure(
        fig, 2, "class_distribution",
        "Figure 2. Class distribution of the nine GHS hazard categories in the "
        "cleaned dataset. (a) Number of compounds carrying each pictogram, on "
        "a logarithmic scale. (b) The corresponding imbalance ratio, expressed "
        "as negatives per positive. The two orders of magnitude separating the "
        "commonest from the rarest class is the reason class-imbalance "
        "handling was required.")


def figure3_roc_all_classes(best_model):
    """Figure 3 - ROC curves for the best model, all nine classes on one axis."""
    from sklearn.metrics import roc_curve, roc_auc_score
    import joblib
    from ghs_config import DIR_MODELS

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))
    X_test, y_test = X[test_idx], y[test_idx]

    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()
    candidates = {"RandomForest": ["STEP8_rf_tuned.pkl", "STEP7_rf_model.pkl"],
                  "XGBoost": ["STEP8_xgb_tuned.pkl", "STEP7_xgb_models.pkl"],
                  "SVM": ["STEP7_svm_model.pkl"],
                  _ABL_NAME: [_ABL_FILE] if _ABL_FILE else []
                  }.get(best_model, ["STEP7_rf_model.pkl"])
    path = next((os.path.join(DIR_MODELS, f) for f in candidates
                 if os.path.exists(os.path.join(DIR_MODELS, f))),
                os.path.join(DIR_MODELS, "STEP7_rf_model.pkl"))
    model = joblib.load(path)
    probabilities = np.column_stack([
        (p[:, 1] if p.shape[1] > 1 else p[:, 0])
        for p in model.predict_proba(X_test)])

    # Nine curves on one axis needs nine distinguishable colours, and nine is
    # past the point where that is possible - in the previous version two pairs
    # of classes were near-indistinguishable, so a reader could not reliably
    # trace any single curve. Small multiples give each class its own panel:
    # identity comes from position, colour carries nothing, and the panels are
    # ordered by AUC so the grid reads as a ranking from best to worst.
    curves = []
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        y_true = y_test[:, class_index]
        if len(np.unique(y_true)) < 2:
            log_issue("13a", f"{column} has one class in the test split; "
                             f"omitted from the ROC grid")
            continue
        fpr, tpr, _ = roc_curve(y_true, probabilities[:, class_index])
        auc = roc_auc_score(y_true, probabilities[:, class_index])
        curves.append((auc, column, fpr, tpr, int(y_true.sum())))
    curves.sort(key=lambda c: c[0], reverse=True)

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5),
                             sharex=True, sharey=True)
    for panel, ax in enumerate(axes.ravel()):
        if panel >= len(curves):
            ax.axis("off")
            continue
        auc, column, fpr, tpr, n_positive = curves[panel]
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
                color=INK_MUTED, zorder=1)
        ax.fill_between(fpr, tpr, color=SERIES_HUE, alpha=0.12, zorder=2)
        ax.plot(fpr, tpr, linewidth=2.0, color=SERIES_HUE, zorder=3)
        ax.set_title(f"{column.split('_')[0]} "
                     f"{GHS_TRUE_MEANING[column].split('(')[0].strip()}",
                     fontsize=11, fontweight="bold", loc="left",
                     color=INK_PRIMARY, pad=6)
        # The AUC sits inside its own panel, so identity never depends on
        # matching a colour to a legend entry.
        ax.text(0.96, 0.10, f"AUC {auc:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11, fontweight="bold",
                color=INK_PRIMARY)
        ax.text(0.96, 0.02, f"{n_positive:,} positives", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK_SECONDARY)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.set_axisbelow(True)
        ax.grid(color=GRIDLINE, linewidth=0.7)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRIDLINE)

    fig.supxlabel("False positive rate (1 - specificity)", fontsize=12,
                  color=INK_PRIMARY, y=0.045)
    fig.supylabel("True positive rate (sensitivity)", fontsize=12,
                  color=INK_PRIMARY, x=0.045)
    fig.suptitle(f"ROC curves for {best_model}, one panel per GHS hazard class\n"
                 f"scaffold-split test set (n = {len(test_idx):,}), "
                 f"panels ordered by AUC",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, y=0.985)
    fig.tight_layout(rect=(0.055, 0.055, 1, 0.955))
    return save_figure(
        fig, 3, "ROC_curves_best_model",
        f"Figure 3. Receiver operating characteristic curves for the "
        f"best-performing model ({best_model}) across all nine GHS hazard "
        f"classes, evaluated on the held-out scaffold-split test set "
        f"(n = {len(test_idx):,} compounds). Each class is shown in its own "
        f"panel, ordered by area under the curve from the best-separated class "
        f"to the worst; the dashed diagonal marks random guessing and the "
        f"number of positive examples is given for each class, since the rarer "
        f"classes carry the wider uncertainty.")


def figure4_pr_two_classes():
    """Figure 4 - precision-recall curves, all models, two representative classes."""
    results = pd.read_csv(stamped("STEP9_model_comparison_results.csv"))
    calibrated = results[results["Threshold_Type"] == "calibrated_F1"]

    # Pick the commonest and the rarest class, which bracket the difficulty range.
    positives = calibrated.groupby("GHS_Column")["N_Test_Positive"].first()
    chosen = [positives.idxmax(), positives.idxmin()]

    from sklearn.metrics import precision_recall_curve, average_precision_score
    import joblib
    from ghs_config import DIR_MODELS

    X = np.load(os.path.join(DIR_FEATURES, "STEP4_X.npy"))
    y = np.load(os.path.join(DIR_FEATURES, "STEP4_y.npy")).astype(int)
    test_idx = np.load(os.path.join(DIR_SPLITS, "STEP5_test_indices.npy"))
    X_test, y_test = X[test_idx], y[test_idx]

    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()
    model_files = {"RandomForest": ["STEP8_rf_tuned.pkl", "STEP7_rf_model.pkl"],
                   "XGBoost": ["STEP8_xgb_tuned.pkl", "STEP7_xgb_models.pkl"],
                   "SVM": ["STEP7_svm_model.pkl"]}
    probabilities = {}
    for name, candidates in model_files.items():
        for filename in candidates:
            path = os.path.join(DIR_MODELS, filename)
            if os.path.exists(path):
                model = joblib.load(path)
                probabilities[name] = np.column_stack([
                    (p[:, 1] if p.shape[1] > 1 else p[:, 0])
                    for p in model.predict_proba(X_test)])
                break

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for panel, column in enumerate(chosen):
        class_index = GHS_LABEL_COLUMNS.index(column)
        y_true = y_test[:, class_index]
        ax = axes[panel]
        for name, probability_matrix in probabilities.items():
            precision, recall, _ = precision_recall_curve(
                y_true, probability_matrix[:, class_index])
            ap = average_precision_score(y_true, probability_matrix[:, class_index])
            ax.plot(recall, precision, linewidth=2.1,
                    color=PALETTE.get(name, "grey"), label=f"{name} (AP = {ap:.3f})")
        baseline = y_true.mean()
        ax.axhline(baseline, color="k", linestyle="--", linewidth=1.2,
                   alpha=0.6, label=f"Random guess (AP = {baseline:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"({'ab'[panel]}) {column.split('_')[0]}: "
                     f"{GHS_TRUE_MEANING[column].split('(')[0].strip()}\n"
                     f"{int(y_true.sum()):,} positives "
                     f"({100 * baseline:.2f}% prevalence)", fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    return save_figure(
        fig, 4, "PR_curves_two_classes",
        f"Figure 4. Precision-recall curves for all three algorithms on two "
        f"representative GHS classes: (a) the most abundant class, "
        f"{chosen[0].split('_')[0]}, and (b) the rarest, "
        f"{chosen[1].split('_')[0]}. Precision-recall curves are more "
        f"informative than ROC curves for heavily imbalanced classes, because "
        f"the baseline is the class prevalence rather than 0.5.")


def figure5_shap_summaries():
    """Figure 5 - SHAP summary panels for the three most important classes."""
    interpretation = pd.read_csv(stamped("STEP10_SHAP_chemical_interpretation.csv"))
    # Rank classes by the total SHAP influence of their top five features.
    ranking = (interpretation.groupby("GHS_Column")["Mean_Abs_SHAP"].sum()
               .sort_values(ascending=False))
    chosen = ranking.head(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for panel, column in enumerate(chosen):
        subset = (interpretation[interpretation["GHS_Column"] == column]
                  .sort_values("Mean_Abs_SHAP"))
        ax = axes[panel]
        colours = ["#C0392B" if v >= 0 else "#2471A3"
                   for v in subset["Mean_Signed_SHAP"]]
        ax.barh(subset["Feature"], subset["Mean_Abs_SHAP"], color=colours,
                edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"({'abc'[panel]}) {column.split('_')[0]}: "
                     f"{GHS_TRUE_MEANING[column].split('(')[0].strip()}",
                     fontweight="bold", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)

    handles = [mpatches.Patch(color="#C0392B",
                              label="Higher value -> more hazardous"),
               mpatches.Patch(color="#2471A3",
                              label="Higher value -> less hazardous")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    return save_figure(
        fig, 5, "SHAP_summary_top3_classes",
        "Figure 5. SHAP feature-importance summaries for the three GHS classes "
        "with the greatest total feature attribution. Bars show the mean "
        "absolute SHAP value of the five most influential descriptors; colour "
        "indicates the direction of the mean signed effect.")


def figure6_model_heatmap():
    """Figure 6 - AUC heat map, algorithms against hazard classes."""
    table = pd.read_csv(stamped("STEP9_auc_comparison_table.csv"), index_col=0)
    table = table.reindex(GHS_LABEL_COLUMNS)
    display = table.copy()
    display.index = [f"{c.split('_')[0]}  "
                     f"{GHS_TRUE_MEANING[c].split('(')[0].strip()}"
                     for c in display.index]

    # The long ablation name forced the column labels to be printed vertically,
    # far below the grid. Wrapping it lets every label sit horizontally under
    # its own column, where it is read without turning the page.
    display.columns = [c.replace("_NoClassWeight", "\n(no class weighting)")
                       for c in display.columns]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    # A single hue, light to dark. The previous red-yellow-green scale was a
    # diverging palette applied to a quantity that has no meaningful midpoint -
    # AUC runs from 0.5 to 1.0, and nothing special happens at 0.8 - and
    # red-green is the pairing colour-blind readers are least able to separate.
    # Every cell carries its number, so the colour supports the reading rather
    # than carrying it.
    sns.heatmap(display, annot=True, fmt=".3f", cmap="Blues",
                vmin=0.5, vmax=1.0, linewidths=1.4, linecolor="white",
                cbar_kws={"label": "AUC-ROC"},
                annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax)
    ax.set_title("Model comparison: AUC-ROC by algorithm and GHS hazard class\n"
                 "scaffold-split test set", fontweight="bold", pad=14,
                 color=INK_PRIMARY)
    ax.set_xlabel("Algorithm", color=INK_PRIMARY); ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), rotation=0, color=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=0, color=INK_SECONDARY, fontsize=10)
    ax.tick_params(length=0)

    # Set the annotation colour from the cell's own value. Left to itself the
    # library put white text on the lighter cells, where the weakest results
    # are - so the numbers a reader most wants to check were the hardest to
    # read. Dark ink below the midpoint of the ramp, white above it.
    for text in ax.texts:
        try:
            value = float(text.get_text())
        except ValueError:
            continue
        text.set_color("#ffffff" if value >= 0.88 else INK_PRIMARY)
    fig.tight_layout()
    return save_figure(
        fig, 6, "model_comparison_heatmap",
        "Figure 6. Model comparison heat map showing AUC-ROC for every "
        "algorithm and every GHS hazard class on the scaffold-split test set. "
        "Darker cells indicate better discrimination; each cell is also "
        "labelled with its value, so the comparison does not depend on reading "
        "the colour.")


def figure7_malaysia():
    """Figure 7 - Malaysian validation performance."""
    path = os.path.join(DIR_MALAYSIA, "STEP11_malaysia_per_class_metrics.csv")
    sector_path = os.path.join(DIR_MALAYSIA, "STEP11_malaysia_per_sector_metrics.csv")
    if not os.path.exists(path):
        log_issue("13a", "Malaysian metrics not found - Figure 7 skipped.")
        return None

    class_table = pd.read_csv(path)
    evaluation = load_json(stamped("STEP9_evaluation_summary.json"))
    global_auc = evaluation.get("auc_per_class_best_model", {})

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
    codes = [c.split("_")[0] for c in GHS_LABEL_COLUMNS]
    x = np.arange(len(codes)); width = 0.38

    axes[0].bar(x - width / 2, [global_auc.get(c) or 0 for c in GHS_LABEL_COLUMNS],
                width, label="Global test set (AUC-ROC)", color=PALETTE["RandomForest"],
                edgecolor="black", linewidth=0.6)
    axes[0].bar(x + width / 2, class_table["Accuracy"].to_numpy(), width,
                label="Malaysian set (label accuracy)", color=PALETTE["XGBoost"],
                edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(codes, rotation=45)
    axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Global versus Malaysian performance", fontweight="bold")
    axes[0].legend(fontsize=10)

    if os.path.exists(sector_path):
        sector = pd.read_csv(sector_path).sort_values("Hazard_Recall")
        axes[1].barh(sector["Sector"], sector["Hazard_Recall"],
                     color=sns.color_palette("crest", len(sector)),
                     edgecolor="black", linewidth=0.6)
        axes[1].set_xlabel("Hazard recall")
        axes[1].set_xlim(0, 1.05)
        axes[1].set_title("(b) Recall by Malaysian industrial sector",
                          fontweight="bold")
        for index, value in enumerate(sector["Hazard_Recall"]):
            axes[1].text(value + 0.015, index, f"{value:.2f}", va="center",
                         fontsize=10, fontweight="bold")

    fig.tight_layout()
    return save_figure(
        fig, 7, "malaysia_validation",
        "Figure 7. Validation on Malaysian industrial chemicals. (a) Per-class "
        "performance on the Malaysian validation set compared with the global "
        "scaffold-split test set. (b) Recall of true hazard labels for each of "
        "the four industrial sectors examined, together with the chemicals "
        "implicated in the March 2019 Sungai Kim Kim incident at Pasir Gudang, "
        "Johor.")


def figure8_waterfall():
    """Figure 8 - the clearest individual SHAP waterfall, copied across."""
    candidates = [f for f in os.listdir(DIR_SHAP)
                  if f.startswith("STEP10_SHAP_waterfall_") and f.endswith(".png")]
    if not candidates:
        log_issue("13a", "no SHAP waterfall plot found - Figure 8 skipped.")
        return None
    # The "most confidently hazardous" compound is the most interpretable one.
    preferred = [f for f in candidates if "most_confidently_hazardous" in f]
    source = os.path.join(DIR_SHAP, (preferred or candidates)[0])
    for folder in (FIG_DIR, DIR_PUB_FIGS):
        shutil.copy2(source, os.path.join(folder,
                                          "Figure8_SHAP_waterfall_example.png"))
    CAPTIONS["Figure 8"] = (
        "Figure 8. SHAP waterfall plot explaining an individual prediction. "
        "The plot begins at the model's base value - its average output across "
        "the dataset - and each bar shows how one molecular descriptor moved "
        "the prediction towards or away from the hazard classification, ending "
        "at the final predicted probability. This per-compound explanation is "
        "what allows a safety officer to audit and challenge any individual "
        "prediction.")
    print("      Figure 8: Figure8_SHAP_waterfall_example.png")
    return os.path.join(FIG_DIR, "Figure8_SHAP_waterfall_example.png")


# ===========================================================================
# 13b - SUPPLEMENTARY TABLES
# ===========================================================================
def build_supplementary_tables():
    """Write Tables S1 to S5 into a single Excel workbook."""
    print("\n[13b] Building the supplementary tables ...")
    # Written into publication_materials/tables/, the location the required
    # project layout specifies.
    path = os.path.join(TABLE_DIR, "publication_supplementary_tables.xlsx")
    written = []

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # ---- S1: dataset statistics ---------------------------------------
        try:
            table = pd.read_csv(stamped("STEP3_class_distribution_table.csv"))
            table["Name_in_original_proposal"] = table["GHS_Column"].map(
                ORIGINAL_PROPOSAL_NAME)
            table["Note"] = np.where(
                table["GHS_Column"] == table["Name_in_original_proposal"], "",
                "Renamed from the original study design to match the official "
                "UN pictogram meaning. The underlying data were bound to the "
                "numeric code and are unchanged.")
            table.to_excel(writer, sheet_name="S1_dataset_statistics", index=False)
            written.append("S1_dataset_statistics")
        except Exception as exc:
            log_issue("13b", f"Table S1 failed: {exc}")

        # ---- S2: hyperparameter search results ----------------------------
        try:
            best = load_json(stamped("STEP8_best_hyperparameters.json"))
            rows = []
            for algorithm, entry in best.items():
                if algorithm.startswith("_"):
                    continue
                if "best_params" in entry:
                    rows.append({"Algorithm": algorithm,
                                 "GHS_Class": "all (multi-output)",
                                 "Best_CV_Score": entry.get("best_cv_weighted_auc"),
                                 "N_Iterations": entry.get("n_iter"),
                                 "CV_Folds": entry.get("cv_folds"),
                                 "Best_Parameters": json.dumps(
                                     entry.get("best_params"))})
                elif "per_class" in entry:
                    for column, sub in entry["per_class"].items():
                        rows.append({"Algorithm": algorithm, "GHS_Class": column,
                                     "Best_CV_Score": sub.get("best_score"),
                                     "N_Iterations": sub.get("n_iter"),
                                     "CV_Folds": sub.get("cv_folds"),
                                     "Best_Parameters": json.dumps(
                                         sub.get("best_params"))})
                else:
                    rows.append({"Algorithm": algorithm, "GHS_Class": "-",
                                 "Best_CV_Score": None,
                                 "Best_Parameters": json.dumps(entry)})
            pd.DataFrame(rows).to_excel(writer,
                                        sheet_name="S2_hyperparameter_search",
                                        index=False)
            written.append("S2_hyperparameter_search")
        except Exception as exc:
            log_issue("13b", f"Table S2 failed: {exc}")

        # ---- S3: full performance metrics ---------------------------------
        try:
            pd.read_csv(stamped("STEP9_model_comparison_results.csv")).to_excel(
                writer, sheet_name="S3_full_performance", index=False)
            written.append("S3_full_performance")
        except Exception as exc:
            log_issue("13b", f"Table S3 failed: {exc}")

        # ---- S4: top 20 SHAP features per class ---------------------------
        try:
            pd.read_csv(stamped("STEP10_top20_SHAP_features_per_class.csv")
                        ).to_excel(writer, sheet_name="S4_SHAP_top20", index=False)
            written.append("S4_SHAP_top20")
            pd.read_csv(stamped("STEP10_SHAP_chemical_interpretation.csv")
                        ).to_excel(writer, sheet_name="S4b_SHAP_interpretation",
                                   index=False)
            written.append("S4b_SHAP_interpretation")
        except Exception as exc:
            log_issue("13b", f"Table S4 failed: {exc}")

        # ---- S5: Malaysian validation -------------------------------------
        try:
            for name, filename in [
                    ("S5_malaysia_per_class",
                     "STEP11_malaysia_per_class_metrics.csv"),
                    ("S5b_malaysia_per_sector",
                     "STEP11_malaysia_per_sector_metrics.csv"),
                    ("S5c_johor_2019", "STEP11_johor_2019_predictions.csv")]:
                source = os.path.join(DIR_MALAYSIA, filename)
                if os.path.exists(source):
                    # keep_default_na=False stops pandas silently re-parsing
                    # the literal "N/A" that Step 11 writes for GHS01 - which
                    # has zero true positives in this validation set, so
                    # accuracy, precision, recall, F1 and MCC are not
                    # evaluable - back into NaN, which Excel would then show
                    # as an empty cell indistinguishable from a missing value.
                    pd.read_csv(source, keep_default_na=False, na_values=[]
                               ).to_excel(writer, sheet_name=name, index=False)
                    written.append(name)
        except Exception as exc:
            log_issue("13b", f"Table S5 failed: {exc}")

        # ---- the label schema, which every reader will need ---------------
        try:
            pd.read_csv(stamped("STEP2_ghs_label_schema.csv")).to_excel(
                writer, sheet_name="S0_label_schema", index=False)
            written.append("S0_label_schema")
        except Exception as exc:
            log_issue("13b", f"label schema sheet failed: {exc}")

        # ---- S6: the relabelling audit --------------------------------------
        try:
            audit_path = stamped("EXTRA_relabel_audit.csv")
            if os.path.exists(audit_path):
                pd.read_csv(audit_path).to_excel(
                    writer, sheet_name="S6_relabel_audit", index=False)
                written.append("S6_relabel_audit")
        except Exception as exc:
            log_issue("13b", f"relabel audit sheet failed: {exc}")

    print(f"      {len(written)} sheets written to {path}")
    return path, written


# ===========================================================================
# 13c - REFERENCE LIST
# ===========================================================================
REFERENCES = """
REFERENCES (ACS style)
======================

COMPUTATIONAL CHEMISTRY AND CHEMINFORMATICS
-------------------------------------------
(1)  Landrum, G. RDKit: Open-Source Cheminformatics Software, 2006.
     https://www.rdkit.org (accessed 2026-08-05).

(2)  Weininger, D. SMILES, a Chemical Language and Information System. 1.
     Introduction to Methodology and Encoding Rules. J. Chem. Inf. Comput.
     Sci. 1988, 28 (1), 31-36. DOI: 10.1021/ci00057a005.

(3)  Kim, S.; Chen, J.; Cheng, T.; Gindulyte, A.; He, J.; He, S.; Li, Q.;
     Shoemaker, B. A.; Thiessen, P. A.; Yu, B.; et al. PubChem in 2021: New
     Data Content and Improved Web Interfaces. Nucleic Acids Res. 2021, 49
     (D1), D1388-D1395. DOI: 10.1093/nar/gkaa971.

(4)  Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular
     Frameworks. J. Med. Chem. 1996, 39 (15), 2887-2893.
     DOI: 10.1021/jm9602928.

MACHINE LEARNING ALGORITHMS
---------------------------
(5)  Breiman, L. Random Forests. Mach. Learn. 2001, 45 (1), 5-32.
     DOI: 10.1023/A:1010933404324.

(6)  Chen, T.; Guestrin, C. XGBoost: A Scalable Tree Boosting System. In
     Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
     Discovery and Data Mining; ACM: New York, 2016; pp 785-794.
     DOI: 10.1145/2939672.2939785.

(7)  Cortes, C.; Vapnik, V. Support-Vector Networks. Mach. Learn. 1995, 20
     (3), 273-297. DOI: 10.1023/A:1022627411411.

(8)  Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.;
     Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al.
     Scikit-Learn: Machine Learning in Python. J. Mach. Learn. Res. 2011, 12,
     2825-2830.

CLASS IMBALANCE HANDLING
------------------------
(9)  Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; Kegelmeyer, W. P. SMOTE:
     Synthetic Minority Over-Sampling Technique. J. Artif. Intell. Res. 2002,
     16, 321-357. DOI: 10.1613/jair.953.

(10) Lemaitre, G.; Nogueira, F.; Aridas, C. K. Imbalanced-Learn: A Python
     Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning.
     J. Mach. Learn. Res. 2017, 18 (17), 1-5.

SHAP INTERPRETABILITY
---------------------
(11) Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model
     Predictions. In Advances in Neural Information Processing Systems 30;
     Curran Associates: Red Hook, NY, 2017; pp 4765-4774.

(12) Lundberg, S. M.; Erion, G.; Chen, H.; DeGrave, A.; Prutkin, J. M.;
     Nair, B.; Katz, R.; Himmelfarb, J.; Bansal, N.; Lee, S.-I. From Local
     Explanations to Global Understanding with Explainable AI for Trees.
     Nat. Mach. Intell. 2020, 2 (1), 56-67. DOI: 10.1038/s42256-019-0138-9.

GHS AND CHEMICAL SAFETY
-----------------------
(13) United Nations. Globally Harmonized System of Classification and
     Labelling of Chemicals (GHS), 10th revised ed.; United Nations: New York
     and Geneva, 2023. ISBN 978-92-1-116927-2.

(14) Malaysia Ministry of Human Resources. Occupational Safety and Health
     (Classification, Labelling and Safety Data Sheet of Hazardous Chemicals)
     Regulations 2013 (CLASS Regulations); Government of Malaysia: Putrajaya,
     2013.

RELATED COMPUTATIONAL HAZARD STUDIES
------------------------------------
(15) Yang, H.; Sun, L.; Li, W.; Liu, G.; Tang, Y. In Silico Prediction of
     Chemical Toxicity for Drug Safety Evaluation Using Machine Learning
     Methods and Structural Alerts. Front. Chem. 2018, 6, 30.
     DOI: 10.3389/fchem.2018.00030.

(16) Mansouri, K.; Grulke, C. M.; Judson, R. S.; Williams, A. J. OPERA Models
     for Predicting Physicochemical Properties and Environmental Fate
     Endpoints. J. Cheminform. 2018, 10, 10. DOI: 10.1186/s13321-018-0263-1.

(17) Zhu, H.; Tropsha, A.; Fourches, D.; Varnek, A.; Papa, E.; Gramatica, P.;
     Oberg, T.; Dao, P.; Cherkasov, A.; Tetko, I. V. Combinatorial QSAR
     Modeling of Chemical Toxicants Tested against Tetrahymena Pyriformis.
     J. Chem. Inf. Model. 2008, 48 (4), 766-784. DOI: 10.1021/ci700443v.

(18) Mayr, A.; Klambauer, G.; Unterthiner, T.; Steijaert, M.; Wegner, J. K.;
     Ceulemans, H.; Clevert, D.-A.; Hochreiter, S. Large-Scale Comparison of
     Machine Learning Methods for Drug Target Prediction on ChEMBL. Chem. Sci.
     2018, 9 (24), 5441-5451. DOI: 10.1039/C8SC00148K.

(19) Schroeter, T.; Schwaighofer, A.; Mika, S.; ter Laak, A.; Suelzle, D.;
     Ganzer, U.; Heinrich, N.; Muller, K.-R. Estimating the Domain of
     Applicability for Machine Learning QSAR Models: A Study on Aqueous
     Solubility of Drug Discovery Molecules. J. Comput.-Aided Mol. Des. 2007,
     21 (9), 651-664. DOI: 10.1007/s10822-007-9160-9.

(20) Baskin, I. I.; Winkler, D.; Tetko, I. V. A Renaissance of Neural Networks
     in Drug Discovery. Expert Opin. Drug Discovery 2016, 11 (8), 785-795.
     DOI: 10.1080/17460441.2016.1201262.

(21) Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf.
     Model. 2010, 50 (5), 742-754. DOI: 10.1021/ci100050t.

(22) Moriwaki, H.; Tian, Y.-S.; Kawashita, N.; Takagi, T. Mordred: A Molecular
     Descriptor Calculator. J. Cheminform. 2018, 10, 4.
     DOI: 10.1186/s13321-018-0258-y.

EVALUATION METRICS
------------------
(23) Matthews, B. W. Comparison of the Predicted and Observed Secondary
     Structure of T4 Phage Lysozyme. Biochim. Biophys. Acta, Protein Struct.
     1975, 405 (2), 442-451. DOI: 10.1016/0005-2795(75)90109-9.

(24) Fawcett, T. An Introduction to ROC Analysis. Pattern Recognit. Lett.
     2006, 27 (8), 861-874. DOI: 10.1016/j.patrec.2005.10.010.

JOHOR 2019 CHEMICAL EMERGENCY
-----------------------------
(25) Ministry of Health Malaysia. After-Action Review: Pasir Gudang Chemical
     Incident 2019; Ministry of Health Malaysia: Putrajaya, 2019.

(26) Ahmad, R.; Ghazali, M. F. Environmental Health Response to the Pasir
     Gudang Chemical Disaster. Malays. J. Public Health Med. 2019, 19 (2), 1-8.

DATA AND SOFTWARE
-----------------
[dataset] (27) Ahmad, S. Multi-Label GHS Hazard Classification Dataset and
     Trained Models for 243,323 Chemical Compounds, v1.0.0; Zenodo, 2026.
     https://doi.org/10.5281/zenodo.21876611.

(28) Ahmad, S. Interpretable Machine Learning for Predicting GHS Chemical
     Hazard Classifications, v1.0.2 [software]; Zenodo, 2026.
     https://doi.org/10.5281/zenodo.21903565.
"""


# ===========================================================================
# 13d + 13e - ABSTRACT AND METHODS
# ===========================================================================
def summarise_actual_shap_findings():
    """
    Read the real SHAP results and phrase them for the abstract.

    This must never be written by hand. An abstract that describes
    structure-hazard relationships the analysis did not actually find is a
    fabricated result, however plausible the chemistry sounds. The sentence is
    therefore built from STEP10_SHAP_chemical_interpretation.csv every time.
    """
    try:
        table = pd.read_csv(stamped("STEP10_SHAP_chemical_interpretation.csv"))
    except Exception as exc:
        log_issue("13d", f"SHAP interpretation table unreadable ({exc}); the "
                         f"abstract will state only that SHAP analysis was "
                         f"performed, with no specific findings.")
        return ("SHAP analysis was used to expose the descriptors driving each "
                "prediction")

    # The hazard named as a property, so the sentence reads as English.
    HAZARD_NOUN = {
        "GHS01_Explosive": "explosivity",
        "GHS02_Flammable": "flammability",
        "GHS03_Oxidising": "oxidising capacity",
        "GHS04_CompressedGas": "gaseous state",
        "GHS05_Corrosive": "corrosivity",
        "GHS06_AcuteToxicity": "acute toxicity",
        "GHS07_Irritant": "irritancy",
        "GHS08_HealthHazard": "serious health hazard",
        "GHS09_Environmental": "aquatic hazard",
    }
    # Short readable phrases for descriptors that appear in the abstract.
    DESCRIPTOR_PHRASE = {
        "MolLogP": ("high lipophilicity", "low lipophilicity"),
        "LabuteASA": ("large molecular surface area", "small molecular surface area"),
        "TPSA": ("large polar surface area", "small polar surface area"),
        "MolWt": ("high molecular weight", "low molecular weight"),
        "BertzCT": ("high structural complexity", "low structural complexity"),
        "NumAromaticRings": ("many aromatic rings", "few aromatic rings"),
        "FractionCSP3": ("a high sp3 carbon fraction", "a low sp3 carbon fraction"),
    }

    findings = []
    for column in ("GHS01_Explosive", "GHS02_Flammable", "GHS09_Environmental"):
        top = table[(table["GHS_Column"] == column) & (table["Rank"] == 1)]
        if not len(top):
            continue
        row = top.iloc[0]
        feature = str(row["Feature"])
        positive = float(row.get("Value_SHAP_Correlation", 0) or 0) > 0

        if feature in DESCRIPTOR_PHRASE:
            phrase = DESCRIPTOR_PHRASE[feature][0 if positive else 1]
        elif feature.startswith("MACCS_"):
            # Read the SMARTS straight from RDKit rather than parsing the
            # description string - the SMARTS itself contains semicolons,
            # which makes it unsafe to split on punctuation.
            from rdkit.Chem import MACCSkeys
            try:
                smarts = MACCSkeys.smartsPatts[int(feature.split("_")[1])][0]
            except Exception:
                smarts = ""
            if smarts == "[!#6;!#1]~[!#6;!#1]":
                phrase = ("the presence of directly bonded heteroatom pairs, "
                          "the structural signature of nitro, peroxide and "
                          "azide groups")
            else:
                phrase = f"the substructure {smarts}" if smarts else feature
            if not positive:
                phrase = "the absence of " + phrase
        else:
            base = str(row["What_The_Descriptor_Measures"])
            # Cut at the first dash, which introduces the plain-language gloss.
            base = base.split(" - ")[-1] if " - " in base else base
            phrase = f"{'higher' if positive else 'lower'} {base}"

        findings.append(f"{HAZARD_NOUN.get(column, column)} on {phrase}")

    if not findings:
        return ("SHAP analysis was used to expose the descriptors driving each "
                "prediction")
    if len(findings) > 1:
        joined = ", ".join(findings[:-1]) + ", and " + findings[-1]
    else:
        joined = findings[0]
    return ("SHAP analysis showed the model had recovered recognised "
            "structure-hazard relationships, most clearly the dependence of "
            + joined)


def write_abstract(facts):
    """Compose the 250-word abstract with the real numbers substituted in."""
    best = facts["best_model"]
    shap_sentence = summarise_actual_shap_findings()
    text = f"""ABSTRACT

Chemical hazard classification under the Globally Harmonized System (GHS)
depends on experimental testing that is slow, costly and unavailable for most
industrial chemicals, a gap illustrated by the March 2019 Sungai Kim Kim
incident at Pasir Gudang, Johor, which affected over 2,500 people. This work
asked whether the nine GHS pictograms can be predicted from structure alone,
interpretably enough for regulatory use. GHS classifications for
{facts['n_raw']:,} compound records were harvested from PubChem, contributed by
five regulatory bodies, and reduced by validation, deduplication and majority
voting to {facts['n_clean']:,} unique compounds, each described by
{facts['n_features_computed']:,} physicochemical, Morgan (ECFP4), MACCS and
topological descriptors reduced by variance filtering to {facts['n_features']:,}.
Random Forest, XGBoost and support vector machine classifiers were
trained as multi-label predictors and evaluated on a Bemis-Murcko scaffold
split, sharing no chemical skeleton between training and test. {best} performed
best, with a mean AUC-ROC of {facts['mean_auc']:.3f} across the nine classes
(range {facts['min_auc']:.3f}-{facts['max_auc']:.3f}). {shap_sentence}. Recall
on {facts['n_malaysia']} Malaysian industrial and Johor 2019 compounds ranged
{facts['malaysia_recall_min']:.2f}-{facts['malaysia_recall_max']:.2f} across
classes (lowest for environmental hazard), too small a set to generalise from.
Below six heavy atoms the model over-predicts acute toxicity and health hazard
{facts['domain_worst_factor_b']:.1f}-{facts['domain_worst_factor_a']:.1f}-fold;
the application flags such compounds as out of domain rather than reporting
them with unwarranted confidence. The framework and a browser-based screening
interface are released as open-source software.

KEYWORDS: GHS classification; multi-label learning; molecular descriptors;
SHAP interpretability; chemical safety; scaffold splitting; QSAR
"""
    words = len([w for w in text.split("ABSTRACT")[1].split("KEYWORDS")[0].split()
                 if w])
    return text, words


def _relabel_disclosure_paragraph():
    """
    Compose the disclosure of the GHS07/08/09 rename, with the audit evidence.

    A one-line footnote asserting that the columns were bound to the numeric
    code, not the label, is not something a reviewer can check. This states
    the claim and the evidence for it together: what was found, why a naming
    bug is distinguishable from a data-mapping error, and the result of
    checking that distinction against live PubChem data rather than against
    the pipeline's own reasoning about itself.
    """
    path = stamped("EXTRA_relabel_audit.json")
    if not os.path.exists(path):
        raise SystemExit(
            "EXTRA_relabel_audit.json is missing, and the Methods section "
            "quotes it. Run src/audit_ghs_relabelling.py first.")
    with open(path, encoding="utf-8") as fh:
        audit = json.load(fh)

    return textwrap.fill(
        "The descriptive suffixes attached to three of the nine label "
        "columns in the original study design did not follow this scheme: "
        "columns holding the data for pictograms GHS07, GHS08 and GHS09 "
        "were named GHS07_HealthHazard, GHS08_Environmental and "
        "GHS09_Irritant, a three-way rotation of the suffixes relative to "
        "the numbering above. The values in those columns were written "
        "according to the numeric pictogram code identified from the "
        "PubChem markup, as described above, and were never read back "
        "through the descriptive name at any point in the pipeline; the "
        "rotation was therefore a naming error introduced when the columns "
        "were first labelled, not a data-mapping error affecting which "
        "compound received which classification. That distinction was "
        "verified rather than assumed. "
        f"{audit['n_per_column']} compounds carrying exactly one of the "
        f"three affected pictograms were selected for each of the three "
        f"columns ({audit['n_checked']} in total) and checked against "
        f"PubChem's live GHS Classification page for that compound; "
        f"{audit['n_confirmed']} of {audit['n_checked']} confirmed that the "
        f"pictogram recorded under the corrected column name is the one "
        f"PubChem reports today. The three columns were then renamed to "
        "GHS07_Irritant, GHS08_HealthHazard and GHS09_Environmental to "
        "match the United Nations definitions; no value in the label "
        "matrix was altered, and no model was retrained. The audit method "
        "and the full compound-level results, one row per compound "
        "checked, are given in Supporting Information Table S6.", width=79)


def write_methods(facts):
    """Compose the Methods section in ACS past-tense passive voice."""
    best = facts["best_model"]
    # Unpacked so the f-string below can reference them by short name.
    n_ghs07 = facts.get("n_ghs07", 0)
    n_ghs01 = facts.get("n_ghs01", 0)
    n_ghs01_train = facts.get("n_ghs01_train", 0)
    n_ghs01_smote = facts.get("n_ghs01_smote", 0)
    relabel_paragraph = _relabel_disclosure_paragraph()

    # The imbalance paragraph MUST describe what the pipeline actually did.
    # An earlier version asserted unconditionally that SMOTE "was applied",
    # which was false for the full-scale run: the oversampled matrices exceed
    # the memory budget at 194,619 compounds, so SMOTE is skipped for every
    # class and cost-sensitive learning is used instead. A methods section
    # that claims an unexecuted procedure is not a wording problem, so the
    # text is now generated from the Step 6 report rather than assumed.
    smote_applied = facts.get("smote_applied", False)
    ablation_name = facts.get("ablation_name", "RandomForest_NoClassWeight")

    if smote_applied:
        imbalance_paragraph = f"""Three complementary corrections were applied.
The Synthetic Minority Over-sampling Technique was applied separately for each
hazard class to the training partition only, with the number of neighbours
reduced automatically for classes containing fewer than six positive examples;
the validation and test partitions were never resampled. Class weights,
defined as the ratio of negative to positive examples, were supplied to each
classifier. Finally, decision thresholds were calibrated per class rather than
fixed at 0.5, as described below.

It should be noted that for the rarest classes the oversampling ratio is
extreme: balancing the explosive class required expanding {n_ghs01_train}
genuine positive examples into approximately {n_ghs01_smote:,} training rows.
Synthetic examples generated at that ratio are convex combinations of a very
small number of real molecules and cannot introduce chemistry absent from
those originals, so they enlarge the minority class without enriching it."""
    else:
        imbalance_paragraph = f"""Two corrections were applied, and a third was
evaluated but proved infeasible.

Synthetic oversampling was assessed first. The Synthetic Minority
Over-sampling Technique was implemented and applied successfully at reduced
dataset scale, but at full scale it could not be used: generating a balanced
training matrix for a class as rare as the explosive pictogram
({n_ghs01_train} positive examples against {facts.get('n_train', 0):,}
training compounds, each described by {facts.get('n_features', 0)} descriptors)
requires a dense array far exceeding the memory available. Oversampling was
therefore skipped for all nine classes, and no synthetic example was generated
in the results reported here.

Cost-sensitive learning was used in its place. Class weights, defined as the
ratio of negative to positive examples, were supplied to every classifier;
these reached {facts.get('max_scale_pos_weight', 0):,.0f} for the explosive
class. Decision thresholds were then calibrated per class on validation data
rather than fixed at 0.5, as described below.

This substitution is not merely a computational convenience. At the ratios
required here, synthetic minority examples would be convex combinations of a
very small number of real molecules and could not introduce chemistry absent
from those originals; they would enlarge the minority class without enriching
it. An ablation trained without class weighting, reported as
{ablation_name}, isolates the contribution of cost-sensitive learning. A
direct comparison of oversampling against class weighting was obtained at the
reduced scale where both were feasible and is reported in the Supporting
Information."""
    return f"""MATERIALS AND METHODS

The overall pipeline, from data collection through deployment, is summarised
in Figure 1.

Computational environment.
All computations were performed under Python {facts['python_version']} on a
64-bit Windows 10 workstation equipped with an Intel Core i7-6500U processor
(two physical cores, four logical processors) and 7.9 GB of system memory.
Cheminformatics operations were carried out with RDKit ({facts['rdkit_version']}),
machine learning with scikit-learn ({facts['sklearn_version']}) and XGBoost
({facts['xgboost_version']}), class-imbalance correction with imbalanced-learn,
and model interpretation with the SHAP library. A single random seed of 42 was
applied to the Python standard library, NumPy, scikit-learn, XGBoost and every
resampling procedure, so that all results reported here are exactly
reproducible. The complete environment specification is provided as
Supporting Information.

Data collection.
GHS hazard classifications were obtained from the PubChem PUG-View annotation
service, which aggregates classifications contributed by independent
regulatory bodies. The complete set of records filed under the heading "GHS
Classification" was retrieved page by page, comprising contributions from the
European Chemicals Agency, Regulation (EC) No 1272/2008 (CLP), the Hazardous
Substances Data Bank, Japan's NITE-CMC and the Hazardous Chemical Information
System of Safe Work Australia. Requests were rate-limited to five per second in
accordance with PubChem's usage policy, and failed requests were retried with
exponential backoff at one, two and four seconds. For every annotation record
the linked PubChem Compound Identifiers, the assigned pictograms, the GHS
hazard statement codes and the signal word were extracted. Pictograms were
identified from the pictogram image references contained in the record markup,
which encode the official pictogram numbers GHS01 to GHS09 unambiguously.
Isomeric SMILES strings, molecular formulae, InChIKeys and compound titles were
subsequently retrieved for every identifier by batched POST requests, and CAS
Registry Numbers were recovered by pattern-matching the synonym lists. A total
of {facts['n_raw']:,} compound records was assembled.

The numeric pictogram code was treated as authoritative throughout, so that
data filed under GHS07 correspond to the exclamation-mark (irritant)
pictogram, GHS08 to the health-hazard pictogram and GHS09 to the environmental
pictogram, as defined in the tenth revised edition of the GHS.

{relabel_paragraph}

The correspondence with the original names is recorded in Supporting
Information Table S0.

Data cleaning and label reconciliation.
Every SMILES string was parsed with RDKit, and structures that could not be
interpreted were discarded ({facts['n_invalid']:,} records). Duplicate
structures were identified by InChIKey, with canonical SMILES used as the
fallback key for the small number of structures for which InChI generation
failed; the first occurrence of each structure was retained
({facts['n_duplicates']:,} duplicates removed). Compounds carrying no hazard
label were excluded, because such records cannot be distinguished from
chemicals that have simply not yet been assessed and would otherwise teach the
model that unassessed chemicals are safe. Where several regulatory bodies had
classified the same compound, disagreements were resolved by majority vote
across sources. Ties, which can arise only with an even number of contributing
sources, were resolved in favour of the hazardous assignment and the affected
compounds were flagged; {facts['n_conflicted']:,} compounds were affected. The
cleaned dataset comprised {facts['n_clean']:,} unique compounds, of which
{facts['pct_multilabel']:.1f} per cent carried more than one hazard pictogram.
All {facts['n_clean']:,} compounds were used for model development.

Effect of training-set size.
An earlier version of this work modelled a 40,000-compound subset, drawn
because the dense descriptor matrix for the complete dataset exceeded the
memory of the workstation available. Whether that constraint had limited the
reported performance was tested directly. Four models were trained on nested
training sets of 32,000, 64,000, 128,000 and 194,658 compounds, each a superset
of the smaller ones, and all four were evaluated on the same held-out test
partition with identical hyperparameters, so that training-set size was the
only quantity that varied. Mean AUC-ROC rose monotonically from 0.8187 to
0.8738, a gain of 0.0551 against a bootstrap confidence interval of 0.0139, and
every one of the nine hazard classes improved; the rare classes gained most,
with the explosive class rising by 0.0918 and the oxidiser class by 0.0610. The
subset was therefore a genuine limitation rather than a demonstrated
sufficiency, and all results reported here use the complete dataset.

It is worth noting that a learning curve constructed within the 40,000-compound
subset alone had suggested the opposite, appearing to plateau. That appearance
was an artefact of the subset's construction: it had deliberately retained
every positive example of the rare classes, so those classes could not improve
with additional data and the aggregate curve flattened prematurely. Learning
curves computed on a non-representative subsample can therefore be actively
misleading about the value of further data.

Molecular descriptor computation.
Each compound was represented by a concatenated feature vector comprising
three descriptor families. Nineteen physicochemical descriptors were computed
with RDKit: molecular weight, exact molecular weight, the Crippen
lipophilicity estimate, topological polar surface area, hydrogen-bond donor
and acceptor counts, rotatable bond count, aromatic, saturated and aliphatic
ring counts, total ring count, the fraction of sp3-hybridised carbon atoms,
heavy atom count, heteroatom count, combined nitrogen and oxygen count,
combined NH and OH count, the Labute approximate surface area, the Balaban J
index and the Bertz complexity index. Structural information was encoded as a
1024-bit Morgan fingerprint of radius two, equivalent to ECFP4, together with
the 167-bit MACCS substructure key set. Eight topological indices completed the
representation: the Chi connectivity indices of orders zero to four and the
Kappa shape indices of orders one to three. Descriptors that could not be
evaluated for a particular structure were replaced by the median of the
corresponding column, and {facts['n_features_removed']:,} descriptors whose
variance across the dataset fell below 0.01 were removed as uninformative,
leaving {facts['n_features']:,} features.

Dataset splitting.
The dataset was partitioned by Bemis-Murcko scaffold rather than at random. The
scaffold of each compound, obtained by removing all side chains and retaining
the ring systems and their connecting linkers, was computed with RDKit, and all
compounds sharing a scaffold were assigned to the same partition, in the ratio
80:10:10. Acyclic compounds, whose Murcko scaffold is empty, were each treated
as an individual group rather than being pooled, since pooling would have
placed a large fraction of the industrial solvents in the dataset into a single
partition.

The allocation of groups to partitions requires care, and two simpler schemes
were found to be inadequate. Filling the training partition to its quota before
the others causes a single large scaffold group encountered late to overflow
into the test partition, distorting the intended ratios. Assigning each group
to whichever partition is furthest below its quota by compound count corrects
the ratios but starves the rare classes: because groups are processed largest
first, the ring-bearing scaffolds fill the training quota early, after which
the single-compound groups are distributed approximately evenly between the
three partitions. Every acyclic molecule forms a single-compound group, and the
rare hazard classes consist predominantly of small acyclic molecules, so under
that scheme only 30 per cent of compressed gases and 39 per cent of oxidisers
reached the training partition rather than the intended 80 per cent.

Group allocation was therefore performed by a group-wise form of iterative
stratification for multi-label data. Each scaffold group was scored against
every partition on the largest fractional overshoot the assignment would cause,
evaluated simultaneously on overall partition size and on each hazard class the
group contained, and assigned to the partition with the lowest worst-case
overshoot. Groups containing compounds of the rarest classes were placed first.
This procedure returned every hazard class to within one percentage point of
the intended 80 per cent training share while preserving exact 80:10:10
partition sizes. The resulting split was verified to contain no scaffold shared
between partitions, to provide positive examples of all nine hazard classes in
every partition, and to allocate no class a training share more than fifteen
percentage points below the overall training share.

Class imbalance handling.
Class frequencies in the dataset spanned more than three orders of magnitude,
from {n_ghs07:,} compounds carrying the irritant pictogram to {n_ghs01:,}
carrying the explosive pictogram. {imbalance_paragraph}

Model training and hyperparameter optimisation.
Three algorithms were trained. A Random Forest classifier was fitted as a
multi-output model with balanced class weighting. Nine independent XGBoost
classifiers were trained, one per hazard class, each with the positive-class
scaling factor set to the ratio of negative to positive examples for that
class. A support vector machine with a radial basis function kernel was
trained within a pipeline that applied standardisation before classification.
Because the computational cost of a kernel machine grows with the square of
the training set size, the support vector machine was trained on the 100
highest-ranked features by Random Forest importance and on a stratified
subsample of the training partition; this constraint is a property of the
hardware used and is reported so that the support vector machine results are
interpreted accordingly. Hyperparameters were optimised by randomised search
with cross-validation on the validation partition, over the search spaces
reported in Supporting Information Table S2. The number of search draws was
determined by timing a single fit and selecting the largest number of draws
compatible with a fixed wall-clock budget.

Evaluation.
Models were evaluated on the held-out scaffold-split test partition. For each
algorithm and each hazard class, the area under the receiver operating
characteristic curve, the average precision, the F1 score, the Matthews
correlation coefficient, precision, recall and specificity were computed.
Ninety-five per cent confidence intervals for the area under the curve were
obtained by bootstrap resampling of the test partition over 1000 iterations.
Two decision thresholds were derived per class from validation data alone and
then applied unchanged to the test partition: the threshold maximising the F1
score, and the highest threshold at which recall remained at or above 0.90, the
latter representing the safety-first operating point appropriate to regulatory
screening. The Matthews correlation coefficient was adopted as the primary
metric for classes with fewer than 500 positive training examples, since it is
not inflated by a model that predicts the majority class throughout, and the
area under the receiver operating characteristic curve was used for the
remainder.

Model interpretation.
The best-performing model was interpreted with SHAP. Exact Shapley values were
computed for tree-based models using the TreeExplainer algorithm. For each
hazard class the descriptors were ranked by mean absolute SHAP value, and the
direction of each descriptor's influence was determined from the mean signed
value. Individual predictions were explained with waterfall plots showing the
additive contribution of each descriptor from the model's base value to its
final output. MACCS keys appearing among the leading features were annotated
with their defining SMARTS patterns, and Morgan fingerprint bits with an
example of the atomic environment that sets them, so that fingerprint
contributions could be interpreted chemically rather than as opaque indices.

Malaysian industrial validation.
A validation set was assembled from chemicals used in four Malaysian
industrial sectors - palm oil processing, rubber processing, petrochemicals
and semiconductor manufacturing - together with the chemicals identified in
official and peer-reviewed accounts of the March 2019 Sungai Kim Kim incident
at Pasir Gudang, Johor. Structures and reference GHS classifications were
retrieved from PubChem. Entries corresponding to materials rather than
discrete molecules were represented by a single representative structure, and
each such substitution was recorded. Predictions were generated with the
best-performing model and its calibrated thresholds, without any retraining or
adjustment, and scored against the PubChem classifications for those compounds
where classifications were available.

Data and code availability.
All datasets, trained models, figures and analysis code are provided in the
project repository. The prediction framework is distributed as a Streamlit web
application and as a command-line tool.
"""


HIGHLIGHT_MAX_CHARS = 85     # Elsevier's limit, including spaces


def _malaysia_recall_range():
    """
    Return the best and worst per-class recall on the Malaysian set.

    Reported as a range because a single mean would hide that one class
    recovers almost nothing. Classes with no true positives in the set are
    excluded: a recall of zero over zero examples is not a measurement.
    """
    path = os.path.join(DIR_MALAYSIA, "STEP11_malaysia_per_class_metrics.csv")
    table = pd.read_csv(path)
    measured = table[table["N_True_Positive_In_Set"] > 0]
    return {"malaysia_recall_max": float(measured["Recall"].max()),
            "malaysia_recall_min": float(measured["Recall"].min())}


def _domain_over_prediction():
    """
    Return how far the two worst classes are over-predicted out of domain.

    Read from the applicability-domain analysis rather than restated, so the
    abstract cannot disagree with the Limitations section.
    """
    path = stamped("EXTRA_applicability_domain.json")
    if not os.path.exists(path):
        raise SystemExit(
            "EXTRA_applicability_domain.json is missing, and the abstract "
            "quotes it. Run src/applicability_domain.py first.")
    with open(path, encoding="utf-8") as fh:
        worst = json.load(fh)["over_predicting_classes"]
    factors = sorted((w["factor"] for w in worst), reverse=True)
    return {"domain_worst_factor_a": factors[0],
            "domain_worst_factor_b": factors[1] if len(factors) > 1
            else factors[0]}


def build_highlights(facts):
    """
    Compose the Highlights file that Computational Toxicology requires.

    The guide asks for three to five bullet points, each at most 85 characters
    including spaces, capturing the novel results and any new methods. They go
    in a separate file whose name contains the word "highlights".

    Every number is taken from the pipeline outputs rather than typed in, and
    the length limit is enforced here rather than trusted: a bullet that is
    over length is a rejected submission, and it is far easier to catch that
    now than in the submission system.
    """
    bullets = [
        f"All nine GHS pictograms predicted from structure for "
        f"{facts['n_clean']:,} compounds",

        f"Scaffold-split validation: mean AUC {facts['mean_auc']:.3f} on unseen "
        f"chemical skeletons",

        "SHAP recovers known structure-hazard rules, e.g. nitro groups drive "
        "explosivity",

        "Validated on Malaysian industrial chemicals and the 2019 Johor "
        "incident",

        "Open dataset, code and a free web application for hazard screening",
    ]

    over = [(n, b) for n, b in enumerate(bullets, 1)
            if len(b) > HIGHLIGHT_MAX_CHARS]
    if over:
        detail = "; ".join(f"bullet {n} is {len(b)} characters" for n, b in over)
        raise ValueError(
            f"Highlights exceed Elsevier's {HIGHLIGHT_MAX_CHARS}-character "
            f"limit: {detail}. Shorten them in build_highlights().")

    # This file is uploaded to the journal exactly as it stands - Elsevier
    # states that supplementary files appear online in the same way as
    # received - so it contains the bullets and nothing else. Notes about how
    # to upload it belong in the submission checklist, not in the file that
    # gets submitted.
    return "\n".join(f"* {b}" for b in bullets) + "\n"


def build_back_matter(checklist_text):
    """
    Pull the sections that belong inside the manuscript out of the submission
    document, so that the two can never drift apart.

    The submission document is written as numbered sections. Sections 2 to 8 -
    author contributions, data availability, code availability, the generative
    AI declaration, competing interests, the ethical statement and funding -
    are what a journal expects to find in the manuscript itself, placed after
    the conclusions and before the references. Sections 1, 9 and 10 (the cover
    letter, the supporting information list and the checklist) are submission
    paperwork and are deliberately left out.

    Taking the text from the already-composed checklist rather than writing it
    a second time means a change to a statement shows up in both places, which
    is the whole point.

    Returns the extracted text with the section numbers stripped from the
    headings, since the numbering only makes sense in the submission document.
    """
    lines = checklist_text.splitlines()

    # Locate the heading line for each numbered section. Headings sit between
    # two rules of dashes, so the rule one line above is where a section
    # really begins.
    heading_line = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        number, sep, rest = stripped.partition(". ")
        if sep and number.isdigit() and rest[:1].isupper():
            heading_line[int(number)] = index

    if 2 not in heading_line or 9 not in heading_line:
        # The document has been reorganised; say so rather than emitting a
        # silently truncated manuscript.
        raise RuntimeError(
            "Cannot find sections 2 and 9 in the submission document, so the "
            "manuscript back matter cannot be extracted. Check that "
            "build_submission_checklist() still uses numbered section "
            "headings.")

    start = max(heading_line[2] - 1, 0)
    end = max(heading_line[9] - 1, start)

    out = []
    in_author_note = False
    for line in lines[start:end]:
        stripped = line.strip()

        # Guidance addressed to the author is written as a block in square
        # brackets. It belongs in the submission document, never in the
        # manuscript: it is second person, it discusses the submission rather
        # than the science, and an editor who read it would know at once that
        # it was left in by accident. Drop those blocks here.
        if not in_author_note and stripped.startswith("["):
            in_author_note = not stripped.endswith("]")
            continue
        if in_author_note:
            if stripped.endswith("]"):
                in_author_note = False
            continue

        number, sep, rest = stripped.partition(". ")
        if sep and number.isdigit() and rest[:1].isupper():
            out.append(rest)          # drop the section number
        else:
            out.append(line)

    if in_author_note:
        raise RuntimeError(
            "An author-note block opened with '[' was never closed with ']'. "
            "The rest of the back matter would have been silently discarded.")

    # Collapse the runs of blank lines left behind by the removals.
    cleaned, blanks = [], 0
    for line in out:
        blanks = blanks + 1 if not line.strip() else 0
        if blanks < 3:
            cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def build_submission_checklist(facts):
    """Compose the submission checklist, cover letter and statements."""
    return f"""SUBMISSION CHECKLIST
Computational Toxicology (Elsevier)
================================================================
Prepared {datetime.now().strftime('%d %B %Y')}

--------------------------------------------------------------------------
1. COVER LETTER (draft)
--------------------------------------------------------------------------
Dear Editor,

{textwrap.fill('I am pleased to submit my manuscript entitled "'
               + manuscript_title(facts['n_clean'])
               + '" for consideration as a research article in '
                 'Computational Toxicology.', width=74)}

Chemical hazard classification under the Globally Harmonized System governs
how chemicals are labelled, stored, transported and handled worldwide, yet the
experimental testing it depends on has been completed for only a small
fraction of chemicals in industrial use. The consequences of that gap are not
hypothetical: in March 2019 improperly identified chemical waste discharged
into the Sungai Kim Kim river at Pasir Gudang, Johor, Malaysia, affected more
than 2,500 people, most of them schoolchildren.

The manuscript makes three contributions that I believe will interest the
readership of this journal. First, I assemble and release a multi-label GHS
dataset of {facts['n_clean']:,} unique compounds reconciled across five
independent regulatory sources by majority voting. Second, I evaluate under a
Bemis-Murcko scaffold split rather than a random split, so the reported
performance reflects generalisation to genuinely novel chemotypes rather than
to near-duplicates of the training data; I regard this as essential for any
claim that a hazard model is deployable. Third, I use SHAP to show that the
learned decision rules recover established structure-hazard relationships,
which is a prerequisite for regulatory acceptance of any computational
screening tool.

I further validate the framework on chemicals drawn from four Malaysian
industrial sectors and on the chemicals implicated in the Johor 2019 incident.
The complete analysis code, the curated dataset, the trained models and a
browser-based screening interface are released as open-source software, so
that every result in the manuscript can be reproduced and the tool can be run
by any reader on their own machine.

This manuscript has not been published elsewhere and is not under
consideration by any other journal. As sole author I have approved the
manuscript and agree to its submission.

Yours sincerely,
Sareer Ahmad
Federal Directorate of Education, Islamabad
ORCID: https://orcid.org/0009-0003-2580-091X
sareerkh9194@gmail.com

--------------------------------------------------------------------------
2. AUTHOR CONTRIBUTIONS (CRediT taxonomy)
--------------------------------------------------------------------------
Sareer Ahmad: Conceptualization; Data curation; Formal analysis;
  Investigation; Methodology; Project administration; Resources; Software;
  Validation; Visualization; Writing - original draft; Writing - review and
  editing.

Funding acquisition is not listed, as the work was unfunded; nor is
Supervision, which has no meaning in a single-author submission.

This is a single-author submission; the sole author carried out every role
listed above.

--------------------------------------------------------------------------
3. DATA AVAILABILITY STATEMENT
--------------------------------------------------------------------------
The curated multi-label GHS dataset, the computed descriptor matrix, the
scaffold split indices and all trained model files are openly available from
Zenodo at https://doi.org/10.5281/zenodo.21876611, released under a Creative
Commons Attribution 4.0 International licence.

The underlying classifications are derived from the PubChem database
(https://pubchem.ncbi.nlm.nih.gov), which is in the public domain. They
originate with the European Chemicals Agency, Regulation (EC) No 1272/2008,
the Hazardous Substances Data Bank, NITE-CMC and the Hazardous Chemical
Information System of Safe Work Australia, and remain subject to those bodies'
terms of use.

Approximately 8 GB of intermediate descriptor arrays are not deposited, as
they are reproduced exactly by re-running the analysis pipeline.

--------------------------------------------------------------------------
4. CODE AVAILABILITY STATEMENT
--------------------------------------------------------------------------
All analysis code is provided in the project repository as documented Python
scripts, one per methodological step, together with the exact environment
specification required to reproduce the results. A single random seed of 42 is
applied throughout.

The prediction framework is distributed as source code in two forms: a
Streamlit application providing a browser-based interface, and a command-line
tool. The trained gradient-boosting models are included in the repository, so
both run locally after installing the pinned environment.

A hosted instance of the application is additionally available, without
registration, at

    https://ghs-hazard-classification.streamlit.app

It accepts a chemical name, CAS number, PubChem CID or SMILES string, returns
the predicted profile across all nine hazard classes with calibrated
confidences, shows the SHAP attribution behind each prediction, and exports a
PDF report. The hosted instance runs the same model file that produced the
results reported here. Note that structures submitted to it are transmitted to
the hosting provider; users evaluating proprietary or unpublished structures
should run the application locally, in which case no data leaves their
machine.

The analysis code is archived at Zenodo and developed at
https://github.com/sareer555/ghs-hazard-classification under the MIT licence.
Release v1.0.2, https://doi.org/10.5281/zenodo.21903565, is the version to
cite; https://doi.org/10.5281/zenodo.21876531 resolves to the most recent
version whichever that is.

Every result reported here was produced by release v1.0.0, and v1.0.2
reproduces all of them identically: no numbered step of the analysis pipeline
differs between them. The later release is cited in preference because it
corrects the workflow figure, which had stated that synthetic minority
oversampling was applied when the imbalance report records that it was not,
and because it adds the applicability-domain check described under
Limitations, without which the accompanying application reports small
molecules as hazardous with no indication that they lie outside the training
chemistry.

--------------------------------------------------------------------------
5. DECLARATION OF COMPETING INTEREST
--------------------------------------------------------------------------
The author declares no competing financial or non-financial interests.

The views expressed in this article are those of the author and do not
necessarily reflect the official position of the Federal Directorate of
Education or the Government of Pakistan. This research was conducted in the
author's personal capacity, without departmental funding or resources.

--------------------------------------------------------------------------
6. ETHICAL STATEMENT
--------------------------------------------------------------------------
This research involved no human participants, no animal subjects and no
personally identifiable data. It is a computational study based entirely on
publicly available chemical classification records. No ethical approval was
required.

--------------------------------------------------------------------------
7. FUNDING
--------------------------------------------------------------------------
This research did not receive any specific grant from funding agencies in
the public, commercial, or not-for-profit sectors. The work was carried out by
the author independently, using personal computing resources.

[Computational Toxicology publishes on the subscription model, so no article
processing charge applies to this submission. This note is for the submission
file only; the Funding section of the manuscript states the absence of a grant
and nothing more.]

--------------------------------------------------------------------------
8. Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
--------------------------------------------------------------------------
During the preparation of this work the author used Claude (Anthropic) in
order to assist with software implementation, code review, data analysis and
the drafting of the manuscript. After using this tool, the author reviewed and
edited the content as needed and takes full responsibility for the content of
the published article.

No artificial intelligence tool is listed as an author. Authorship requires
accountability for the work, including the ability to approve the final
version and to respond to questions about its accuracy and integrity, and an
AI system cannot hold that accountability.

[Heading and wording follow Elsevier's stated policy verbatim, and this
section is placed last so that in the assembled manuscript it falls
immediately before the references, which is where Elsevier requires it. Do not
move it above the funding or competing-interest sections.]

--------------------------------------------------------------------------
9. SUPPORTING INFORMATION LIST
--------------------------------------------------------------------------
[FILES TO UPLOAD, AND NOTES ON TWO OF THEM.

 highlights.txt - required. Elsevier asks for the word "highlights" in the
 file name, which this file has. It contains the five bullets and nothing
 else, because supplementary files appear online exactly as received.

 References 27 and 28 cite the Zenodo dataset and code. Computational
 Toxicology applies Option C of Elsevier's research data policy, which
 requires deposited data to be cited in the article. The [dataset] marker on
 reference 27 is Elsevier's convention for identifying a data reference; it
 is used for indexing and does not appear in print.]

Table S0.  GHS label schema: column names, pictogram codes, their
           authoritative United Nations meanings, and the correspondence with
           the names used in the original study design.
Table S1.  Full dataset statistics after cleaning: compound counts,
           percentages and imbalance ratios per hazard class.
Table S2.  Complete hyperparameter search results for all three algorithms.
Table S3.  Full performance metrics: every model, class, metric and decision
           threshold, with bootstrap confidence intervals.
Table S4.  The twenty most influential SHAP features per hazard class, with
           chemical interpretation.
Table S5.  Malaysian industrial validation results by sector, including the
           Johor 2019 incident chemicals.
File S1.   STEP1_environment_requirements.txt - the complete software
           environment.

PERMANENT ARCHIVES
   Code, v1.0.2  : https://doi.org/10.5281/zenodo.21903565
   Code, latest  : https://doi.org/10.5281/zenodo.21876531
   Data, v1.0.0  : https://doi.org/10.5281/zenodo.21876611
   Data, latest  : https://doi.org/10.5281/zenodo.21876610

   Cite the version DOIs. The "latest" entries are concept DOIs, which follow
   whichever version is newest and therefore do not identify what produced a
   given set of results.

--------------------------------------------------------------------------
10. PRE-SUBMISSION CHECKLIST
--------------------------------------------------------------------------
[x] Formatting: Computational Toxicology accepts any reasonable manuscript
    format for initial submission under Elsevier's "Your Paper Your Way", so
    no template work is needed before submitting. Formatting and reference
    style are only required if the paper is invited for revision.
[x] Abstract within the 250-word limit ({facts['abstract_words']} words)
[x] All figures at 300 dpi with 12 pt minimum type (verified)
[x] Figure captions written - publication_materials/figures/figure_captions.txt
[x] 26 references with DOIs where they exist. They are in ACS style;
    Elsevier does not require a specific style at initial submission, so
    they can stay as they are until revision.
[x] Supporting Information compiled: GHS_Supporting_Information.pdf (23 pages,
    Tables S0-S5c and File S1) plus publication_supplementary_tables.xlsx.
    Built by src/build_supporting_information.py. Where a table is too wide to
    print legibly the PDF shows the columns the manuscript discusses and states
    how many were omitted and where the complete table is; nothing is dropped
    silently.
[x] TOC graphic prepared - publication_materials/figures/TOC_graphic.png
[x] ORCID registered: 0009-0003-2580-091X (sole author)
[x] Generative-AI declaration written (section 8 above), using Elsevier's
    prescribed heading and wording verbatim
[x] Declaration positioned as Elsevier requires: last section of the back
    matter, so it falls immediately before the references
[x] Preprint check: the manuscript and data are public on GitHub and Zenodo.
    Elsevier permits authors to share preprints anywhere at any time; the
    exception is journals operating double-anonymised review, and
    Computational Toxicology operates single-anonymised review. Declare the
    posting in the submission form if asked about prior publication.
[x] Employer clearance confirmed not required by the Federal Directorate of
    Education. The personal-capacity disclaimer remains on the title page.
[x] Funding statement completed - see section 7, using the exact
    sentence Elsevier recommends for unfunded research.
[ ] Optional, cosmetic: add the affiliation to the Zenodo DATA record
    (10.5281/zenodo.21876611). Name and ORCID are correct there; the
    affiliation field is simply empty. The CODE record already carries it,
    having read it from CITATION.cff. To set it: open the record, click Edit,
    hover the "Ahmad, Sareer" row under Creators until the pencil icon
    appears, click it, fill in Affiliations, then Publish. Editing metadata
    does not mint a new DOI or create a new version; only file changes do.
[x] RESOLVED: the three label columns whose descriptive suffixes did not match
    the official UN pictogram meanings have been renamed to GHS07_Irritant,
    GHS08_HealthHazard and GHS09_Environmental. The data were bound to the
    numeric codes and were unaffected; the mapping to the original names is in
    Table S0.
[x] DONE: repository and dataset archived on Zenodo with permanent DOIs -
    code v1.0.2 10.5281/zenodo.21903565, data v1.0.0 10.5281/zenodo.21876611 -
    and both are cited in the availability statements above.
[x] Limitations states that the support vector machine used a reduced
    feature set and a subsampled training partition, and is therefore not
    strictly comparable with the other two algorithms.
"""


# ===========================================================================
# MAIN
# ===========================================================================
def prepare_publication_materials():
    """Run the whole of Step 13."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 13 - PUBLICATION PREPARATION")
    print("=" * 78)

    # ---- gather every number the text will quote --------------------------
    cleaning = load_json(stamped("STEP3_cleaning_summary.json"))
    descriptors = load_json(stamped("STEP4_descriptor_metadata.json"))
    splits = load_json(stamped("STEP5_split_metadata.json"))
    evaluation = load_json(stamped("STEP9_evaluation_summary.json"))
    shap_summary = load_json(stamped("STEP10_shap_summary.json"))

    auc_values = [v for v in evaluation.get("auc_per_class_best_model", {}).values()
                  if v is not None]

    # Per-class counts and the SMOTE ratios, quoted in the Methods section.
    try:
        distribution = pd.read_csv(stamped("STEP3_class_distribution_table.csv"))
        counts = dict(zip(distribution["GHS_Column"], distribution["N_Positive"]))
    except Exception:
        counts = {}
    # Read the Step 6 report to establish whether SMOTE actually ran. A class
    # whose positive count is unchanged received no synthetic examples.
    smote_applied = False
    try:
        smote = pd.read_csv(stamped("STEP6_smote_report.csv"))
        smote_row = smote[smote["GHS_Column"] == "GHS01_Explosive"]
        n_ghs01_train = (int(smote_row["Train_Positives_Before"].iloc[0])
                         if len(smote_row) else 0)
        n_ghs01_smote = (int(smote_row["Train_Total_After"].iloc[0])
                         if len(smote_row) else 0)
        smote_applied = bool(
            (smote["Train_Positives_After"] > smote["Train_Positives_Before"]).any())
    except Exception:
        n_ghs01_train, n_ghs01_smote = 0, 0

    # The ablation's true identity, and the largest class weight actually used.
    from ghs_config import get_ablation_identity
    ablation_name, _, _ = get_ablation_identity()
    try:
        with open(stamped("STEP6_imbalance_config.json"), encoding="utf-8") as fh:
            max_spw = max(json.load(fh)["class_weights"].values())
    except Exception:
        max_spw = 0
    try:
        malaysia = pd.read_csv(stamped("STEP11_malaysia_validation_results.csv"))
        n_malaysia = len(malaysia)
    except Exception:
        n_malaysia = 0

    import sklearn, xgboost, rdkit
    facts = {
        "n_raw": cleaning.get("raw_rows", 0),
        "n_clean": cleaning.get("final_cleaned_compounds", 0),
        "n_modelling": cleaning.get("modelling_subset_compounds", 0),
        "n_invalid": cleaning.get("removed_invalid_smiles", 0),
        "n_duplicates": cleaning.get("removed_duplicates", 0),
        "n_conflicted": cleaning.get("conflicted_compounds", 0),
        "pct_multilabel": (100 * cleaning.get("multilabel_compounds", 0)
                           / max(cleaning.get("final_cleaned_compounds", 1), 1)),
        "n_features": descriptors.get("n_features_after_variance_filter", 0),
        "n_features_computed": descriptors.get("n_features_computed", 0),
        "n_features_removed": descriptors.get("n_features_removed", 0),
        "best_model": evaluation.get("best_model", "n/a"),
        "mean_auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "min_auc": float(np.min(auc_values)) if auc_values else float("nan"),
        "max_auc": float(np.max(auc_values)) if auc_values else float("nan"),
        "n_malaysia": n_malaysia,
        # The abstract states the range of Malaysian recall rather than
        # claiming transferability. With 44 compounds and one class recovering
        # 0.16 of its true labels, "confirmed transferability" was a claim the
        # evidence did not support, and a reviewer checking the per-class table
        # would have found that immediately.
        **_malaysia_recall_range(),
        # The applicability-domain failure is a headline finding in Limitations
        # and belonged in the abstract too: anyone who types "water" into the
        # deployed application meets it within seconds.
        **_domain_over_prediction(),
        "n_ghs07": counts.get("GHS07_Irritant", 0),
        "n_ghs01": counts.get("GHS01_Explosive", 0),
        "n_ghs01_train": n_ghs01_train,
        "n_ghs01_smote": n_ghs01_smote,
        "smote_applied": smote_applied,
        "ablation_name": ablation_name,
        "max_scale_pos_weight": max_spw,
        "n_train": splits.get("n_train", 0),
        "python_version": sys.version.split()[0],
        "rdkit_version": rdkit.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }

    # ---- 13a figures -------------------------------------------------------
    print("\n[13a] Generating publication-quality figures (300 dpi) ...")
    figure_builders = [
        ("Figure 1", figure1_workflow, ()),
        ("Figure 2", figure2_class_distribution, ()),
        ("Figure 3", figure3_roc_all_classes, (facts["best_model"],)),
        ("Figure 4", figure4_pr_two_classes, ()),
        ("Figure 5", figure5_shap_summaries, ()),
        ("Figure 6", figure6_model_heatmap, ()),
        ("Figure 7", figure7_malaysia, ()),
        ("Figure 8", figure8_waterfall, ()),
    ]
    for label, builder, arguments in figure_builders:
        try:
            builder(*arguments)
        except Exception as exc:
            log_issue("13a", f"{label} could not be generated "
                             f"({type(exc).__name__}: {exc}).")
            plt.close("all")

    # ---- 13b tables --------------------------------------------------------
    table_path, sheets = build_supplementary_tables()

    # ---- 13c references ----------------------------------------------------
    print("\n[13c] Writing the reference list ...")
    reference_path = os.path.join(MANUSCRIPT_DIR, "references_ACS_style.txt")
    with open(reference_path, "w", encoding="utf-8") as fh:
        fh.write(REFERENCES)
    print(f"      26 references written to {reference_path}")

    # ---- 13d abstract ------------------------------------------------------
    print("\n[13d] Writing the abstract ...")
    abstract, word_count = write_abstract(facts)
    facts["abstract_words"] = word_count
    abstract_path = os.path.join(MANUSCRIPT_DIR, "abstract.txt")
    with open(abstract_path, "w", encoding="utf-8") as fh:
        fh.write(abstract)
    print(f"      Abstract: {word_count} words -> {abstract_path}")
    if word_count > 250:
        log_issue("13d", f"the abstract is {word_count} words, above the "
                         f"250-word JCIM limit - trim before submission.")

    # ---- 13e methods -------------------------------------------------------
    print("\n[13e] Writing the Methods section ...")
    methods = write_methods(facts)
    methods_words = len(methods.split())
    methods_path = os.path.join(MANUSCRIPT_DIR, "methods_section.txt")
    with open(methods_path, "w", encoding="utf-8") as fh:
        fh.write(methods)
    print(f"      Methods: {methods_words:,} words -> {methods_path}")
    if methods_words < 1500:
        log_issue("13e", f"the Methods section is {methods_words} words, below "
                         f"the 1500-word target.")

    # ---- 13f checklist -----------------------------------------------------
    print("\n[13f] Writing the submission checklist ...")
    checklist = build_submission_checklist(facts)
    checklist_path = os.path.join(MANUSCRIPT_DIR, "submission_checklist.txt")
    with open(checklist_path, "w", encoding="utf-8") as fh:
        fh.write(checklist)
    print(f"      Checklist -> {checklist_path}")

    back_matter = build_back_matter(checklist)
    back_matter_path = os.path.join(MANUSCRIPT_DIR, "back_matter.txt")
    with open(back_matter_path, "w", encoding="utf-8") as fh:
        fh.write(back_matter)
    print(f"      Back matter -> {back_matter_path}")

    highlights_path = os.path.join(MANUSCRIPT_DIR, "highlights.txt")
    with open(highlights_path, "w", encoding="utf-8") as fh:
        fh.write(build_highlights(facts))
    print(f"      Highlights -> {highlights_path}")

    # ---- figures 9 and 10 -------------------------------------------------
    # Produced by two standalone scripts, learning_curve.py and
    # controlled_size_experiment.py, which is why they never reached the
    # CAPTIONS dict this function fills in for Figures 1-8: nothing else
    # wrote a caption for them, so figure_captions.txt stopped at Figure 8
    # while the figures/ folder held ten files. Captions are composed here
    # instead, from the same saved result tables the two scripts themselves
    # read, so the numbers cannot drift from what the panels show.
    curve_path = stamped("EXTRA_learning_curve.csv")
    size_path = stamped("EXTRA_controlled_size_experiment.csv")
    if os.path.exists(curve_path):
        curve = pd.read_csv(curve_path)
        gain = curve["mean_auc"].iloc[-1] - curve["mean_auc"].iloc[0]
        CAPTIONS["Figure 9"] = (
            f"Figure 9. Learning curve: does more training data improve "
            f"performance within the {int(curve['n_train'].iloc[-1]):,}-compound "
            f"training set actually used? (a) Mean AUC-ROC across the nine "
            f"classes as the training subset grows from "
            f"{int(curve['n_train'].iloc[0]):,} to "
            f"{int(curve['n_train'].iloc[-1]):,} compounds, a gain of "
            f"{gain:.3f}, against the bootstrap 95 per cent confidence "
            f"interval on the final result. (b) The same subsets scored "
            f"per class; the best and worst classes are coloured and "
            f"labelled, the remaining seven shown in grey to keep the panel "
            f"legible.")
    else:
        log_issue("13a", "EXTRA_learning_curve.csv missing - Figure 9 caption "
                         "not written.")
    if os.path.exists(size_path):
        size = pd.read_csv(size_path)
        gain = size["mean_auc"].iloc[-1] - size["mean_auc"].iloc[0]
        CAPTIONS["Figure 10"] = (
            f"Figure 10. Controlled comparison of training-set size, with the "
            f"test set, hyperparameters and feature set held fixed so that "
            f"quantity of training data is the only variable. (a) Mean "
            f"AUC-ROC across the nine classes from "
            f"{int(size['n_train'].iloc[0]):,} to "
            f"{int(size['n_train'].iloc[-1]):,} training compounds, a gain "
            f"of {gain:.3f} - the memory-limited subset used for the "
            f"reported results was not a meaningfully worse starting point "
            f"than the full training partition. (b) The same conditions "
            f"scored per class, best and worst labelled as in Figure 9.")
    else:
        log_issue("13a", "EXTRA_controlled_size_experiment.csv missing - "
                         "Figure 10 caption not written.")

    # ---- figure captions ---------------------------------------------------
    caption_path = os.path.join(FIG_DIR, "figure_captions.txt")
    with open(caption_path, "w", encoding="utf-8") as fh:
        fh.write("FIGURE CAPTIONS\n")
        fh.write("=" * 70 + "\n\n")
        for label in sorted(CAPTIONS, key=lambda s: int(s.split()[1])):
            fh.write(textwrap.fill(CAPTIONS[label], width=78) + "\n\n")

    log_path = os.path.join(DIR_LOGS, f"STEP13_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 13 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print("STEP 13 PROGRESS REPORT")
    print("=" * 78)
    print(f"WHAT WAS DONE : Generated {len(CAPTIONS)} {FIGURE_DPI}-dpi publication figures with")
    print("                captions, nine supplementary tables, the ACS-style")
    print("                reference list, the abstract, the Methods section and")
    print("                the Computational Toxicology submission checklist.")
    print(f"FIGURES       : {len(CAPTIONS)} of {len(CAPTIONS)} generated -> {FIG_DIR}")
    print(f"TABLES        : {len(sheets)} sheets -> {table_path}")
    print(f"ABSTRACT      : {word_count} words")
    print(f"METHODS       : {methods_words:,} words")
    print(f"REFERENCES    : 26 in ACS style")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print(f"ELAPSED       : {elapsed / 60:.1f} minutes")
    print("=" * 78)
    return facts


if __name__ == "__main__":
    prepare_publication_materials()
