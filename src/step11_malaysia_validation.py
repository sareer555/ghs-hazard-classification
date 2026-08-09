"""
STEP 11 - MALAYSIAN INDUSTRIAL VALIDATION
=========================================
A model that scores well on a global benchmark is not automatically useful in
Malaysia. This step tests the framework on the chemicals that Malaysian
workers actually handle, drawn from the four industrial sectors named in the
proposal, and on the chemicals implicated in the Sungai Kim Kim incident at
Pasir Gudang, Johor, in March 2019 - the event that motivates this research.

11a  Compile the Malaysian chemical list and fetch structures from PubChem.
11b  Apply the best trained model and score it.
11c  Analyse the Johor 2019 emergency chemicals specifically.
11d  Write a PDF report aimed at DOSH (the Department of Occupational Safety
     and Health, Malaysia).

Author : Sareer Ahmad
"""

import os
import re
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (get_ablation_identity, RANDOM_SEED, TODAY, DIR_FEATURES, DIR_MODELS, DIR_SPLITS,
                        DIR_MALAYSIA, DIR_LOGS, GHS_LABEL_COLUMNS,
                        GHS_TRUE_MEANING, PICTOGRAM_CODE_TO_COLUMN,
                        PICTOGRAM_EXTRA_TO_CODE, seed_everything, stamped)

seed_everything()

# The ablation's name reflects what it actually measured; see
# get_ablation_identity() in ghs_config.py.
_ABL_NAME, _ABL_FILE, _ABL_META = get_ablation_identity()

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_descriptors import descriptors_for_one_molecule, build_feature_names

ISSUE_LOG = []
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GHS-Hazard-ML-Research/1.0 (academic)"})
RATE_LIMIT_DELAY = 0.2


def log_issue(step, message):
    """Record an issue and its resolution for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


# ===========================================================================
# 11a - THE MALAYSIAN CHEMICAL LIST
# ===========================================================================
# Four sectors, as specified in the proposal. Some entries in the proposal are
# materials rather than single molecules - "bleaching earth" is a clay,
# "carbon black" is elemental carbon, "fatty acids" is a family. A single
# representative molecule is substituted for each of those and the substitution
# is recorded in the results table so that no reader is misled.
MALAYSIAN_CHEMICALS = {
    "Palm Oil Processing": [
        ("hexane", None),
        ("phosphoric acid", None),
        ("bleaching earth", "montmorillonite"),   # substitute: the clay mineral
        ("sodium hydroxide", None),
        ("citric acid", None),
        ("hydrogen peroxide", None),
        ("fatty acids", "palmitic acid"),         # substitute: the dominant
                                                  # fatty acid in palm oil
        ("glycerol", None),
    ],
    "Rubber Processing": [
        ("sulfur", None),
        ("zinc oxide", None),
        ("mercaptobenzothiazole", "2-mercaptobenzothiazole"),
        ("ammonia", None),
        ("acetic acid", None),
        ("carbon black", "graphite"),             # substitute: elemental carbon
        ("tetramethylthiuram disulfide", None),
    ],
    "Petrochemicals": [
        ("benzene", None),
        ("toluene", None),
        ("xylene", "p-xylene"),                   # substitute: a defined isomer
        ("ethylene", None),
        ("propylene", None),
        ("methanol", None),
        ("ethanol", None),
        ("acetone", None),
        ("cyclohexane", None),
    ],
    "Semiconductor (Penang)": [
        ("hydrofluoric acid", None),
        ("sulfuric acid", None),
        ("hydrogen peroxide", None),
        ("acetone", None),
        ("isopropanol", None),
        ("phosphine", None),
        ("arsine", None),
        ("tetramethylammonium hydroxide", None),
    ],
}

# ---------------------------------------------------------------------------
# 11c - SUNGAI KIM KIM, PASIR GUDANG, JOHOR - MARCH 2019
# ---------------------------------------------------------------------------
# On 7 March 2019 chemical waste was illegally discharged into the Sungai Kim
# Kim river at Pasir Gudang. Vapour released from the river affected more than
# 2,500 people, most of them schoolchildren, and 111 schools were closed.
#
# The chemicals below are those named in the Ministry of Health Malaysia
# after-action review and in the peer-reviewed accounts of the incident. The
# waste was a mixture of marine-fuel and industrial solvent residues, so
# published constituent lists differ in detail; every compound named in more
# than one account is included here. Acrylonitrile and acrolein were
# identified as the principal agents responsible for the acute symptoms.
#
# Each entry is (name as reported, PubChem lookup name or None, role). A
# lookup name is supplied where the reported name is not a single compound:
# "xylene" is a mixture of three isomers and PubChem returns no record for it,
# so the para isomer is used as the representative structure.
JOHOR_2019_CHEMICALS = [
    ("acrylonitrile",    None,       "principal agent identified by the Ministry of Health"),
    ("acrolein",         None,       "principal agent; a severe respiratory irritant"),
    ("benzene",          None,       "volatile aromatic detected in air and water samples"),
    ("toluene",          None,       "volatile aromatic detected in air samples"),
    ("xylene",           "p-xylene", "volatile aromatic detected in air samples"),
    ("ethylbenzene",     None,       "volatile aromatic detected in air samples"),
    ("hydrogen sulfide", None,       "released from the anaerobic river sediment"),
    ("methane",          None,       "released from the anaerobic river sediment"),
    ("methyl mercaptan", None,       "odorous sulfur compound reported at the site"),
    ("d-limonene",       None,       "solvent component reported in the waste mixture"),
    ("hydrogen chloride", None,      "acid gas reported in the waste mixture"),
    ("styrene",          None,       "monomer reported among the solvent residues"),
]


BACKOFF_SCHEDULE = [1, 2, 4, 8]


def pubchem_get(url, params=None, timeout=60):
    """
    Fetch one PubChem URL, retrying when the server is merely busy.

    PubChem answers HTTP 503 when it is throttling or temporarily overloaded.
    That is a transient condition, not a missing compound, so it must be
    retried rather than treated as "not found" - on the first run of this step
    six chemicals were lost to 503s, including hydrogen sulfide, one of the
    agents released at Sungai Kim Kim.

    Returns the response, or None if every attempt failed.
    """
    for wait in [0] + BACKOFF_SCHEDULE:
        if wait:
            time.sleep(wait)          # 1 s, then 2, then 4, then 8
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
        except Exception:
            continue
        # 404 is a real answer: PubChem has no such compound. Do not retry it.
        if response.status_code == 404:
            return response
        if response.status_code in (429, 503, 504):
            continue                  # busy - wait and try again
        time.sleep(RATE_LIMIT_DELAY)  # be polite before the next call
        return response
    return None


def fetch_compound_from_pubchem(name):
    """
    Look up one chemical by name and return its structure and GHS labels.

    Two requests are made: one for the structure, and one for the compound's
    GHS classification, which serves as the ground truth this step scores
    against. Both go through pubchem_get, so a busy server is retried rather
    than being mistaken for an unknown chemical.
    """
    result = {"Query_Name": name, "CID": None, "SMILES": None,
              "MolecularFormula": None, "PubChem_Name": None,
              "Found": False, "GHS_Found": False}
    for column in GHS_LABEL_COLUMNS:
        result[column] = 0

    # ---- structure ---------------------------------------------------------
    try:
        response = pubchem_get(
            f"{PUBCHEM_BASE}/pug/compound/name/{requests.utils.quote(name)}"
            f"/property/SMILES,MolecularFormula,Title/JSON")
        if response is None:
            log_issue("11a", f"'{name}': PubChem stayed unavailable through "
                             f"{len(BACKOFF_SCHEDULE) + 1} attempts with "
                             f"exponential backoff - compound not resolved.")
            return result
        if response.status_code != 200:
            log_issue("11a", f"'{name}': PubChem returned HTTP "
                             f"{response.status_code} - compound not resolved.")
            return result
        properties = response.json()["PropertyTable"]["Properties"][0]
        result.update({
            "CID": properties.get("CID"),
            "SMILES": properties.get("SMILES") or properties.get("ConnectivitySMILES"),
            "MolecularFormula": properties.get("MolecularFormula"),
            "PubChem_Name": properties.get("Title"),
            "Found": True,
        })
    except Exception as exc:
        log_issue("11a", f"'{name}': structure lookup failed ({exc}).")
        return result

    # ---- GHS ground truth --------------------------------------------------
    try:
        response = pubchem_get(
            f"{PUBCHEM_BASE}/pug_view/data/compound/{result['CID']}/JSON",
            params={"heading": "GHS Classification"}, timeout=90)
        if response is not None and response.status_code == 200:
            # The pictogram codes appear as image URLs anywhere in the record,
            # so a regular expression over the raw text is the most robust way
            # to find them without walking PubChem's deeply nested JSON.
            codes = set(re.findall(r"/(GHS0[1-9])\.svg", response.text))
            if codes:
                result["GHS_Found"] = True
                for code in codes:
                    column = PICTOGRAM_CODE_TO_COLUMN.get(code)
                    if column:
                        result[column] = 1
            else:
                log_issue("11a", f"'{name}' (CID {result['CID']}): no GHS "
                                 f"classification in PubChem - it can be "
                                 f"predicted but not scored.")
    except Exception as exc:
        log_issue("11a", f"'{name}': GHS lookup failed ({exc}).")

    return result


def compile_malaysian_dataset():
    """Build the full Malaysian validation table from PubChem."""
    print("\n[11a] Compiling the Malaysian industrial chemical list ...")
    rows = []

    for sector, chemicals in MALAYSIAN_CHEMICALS.items():
        print(f"\n      --- {sector} ---")
        for requested_name, substitute in chemicals:
            lookup_name = substitute or requested_name
            record = fetch_compound_from_pubchem(lookup_name)
            record["Sector"] = sector
            record["Requested_Name"] = requested_name
            record["Substitution_Note"] = (
                f"'{requested_name}' is a material rather than a single "
                f"molecule; '{substitute}' used as a representative structure"
                if substitute else "")
            record["Incident_Role"] = ""
            rows.append(record)
            status = "found" if record["Found"] else "NOT FOUND"
            ghs = "with GHS" if record["GHS_Found"] else "no GHS labels"
            print(f"      {requested_name:<32} {status:<10} {ghs}")

    print(f"\n      --- Johor 2019 (Sungai Kim Kim, Pasir Gudang) ---")
    for name, substitute, role in JOHOR_2019_CHEMICALS:
        record = fetch_compound_from_pubchem(substitute or name)
        record["Sector"] = "Johor 2019 Emergency"
        record["Requested_Name"] = name
        record["Substitution_Note"] = (
            f"'{name}' is an isomer mixture rather than a single compound; "
            f"'{substitute}' used as a representative structure"
            if substitute else "")
        record["Incident_Role"] = role
        rows.append(record)
        status = "found" if record["Found"] else "NOT FOUND"
        print(f"      {name:<32} {status:<10} {role[:40]}")

    frame = pd.DataFrame(rows)
    n_found = int(frame["Found"].sum())
    n_ghs = int(frame["GHS_Found"].sum())
    print(f"\n      {n_found}/{len(frame)} compounds resolved to a structure")
    print(f"      {n_ghs}/{len(frame)} have GHS ground truth in PubChem")
    return frame


# ===========================================================================
# 11b - APPLY THE TRAINED MODEL
# ===========================================================================
def predict_malaysian_compounds(frame, model, thresholds):
    """
    Compute descriptors for the Malaysian compounds and predict their hazards.

    The same descriptor code and the same calibrated thresholds used on the
    global test set are applied here, so the comparison is fair.
    """
    print("\n[11b] Computing descriptors and applying the trained model ...")

    with open(stamped("STEP4_feature_names.txt"), encoding="utf-8") as fh:
        kept_names = [line.strip() for line in fh
                      if line.strip() and not line.startswith("#")]
    all_names = build_feature_names()
    # Map the retained feature names back to their position in the full
    # descriptor vector, so the same columns are selected as in training.
    keep_positions = [all_names.index(name) for name in kept_names]

    valid_rows, descriptor_rows = [], []
    for position, row in frame.iterrows():
        if not row["Found"] or not row["SMILES"]:
            continue
        descriptors = descriptors_for_one_molecule(row["SMILES"])
        if descriptors is None:
            log_issue("11b", f"'{row['Requested_Name']}': RDKit could not parse "
                             f"the SMILES returned by PubChem - excluded.")
            continue
        # Descriptors that are NaN for this molecule are set to zero; the
        # median imputer from Step 4 was fitted on the training set only, and
        # a handful of single molecules do not justify refitting it.
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        descriptor_rows.append(descriptors[keep_positions])
        valid_rows.append(position)

    if not descriptor_rows:
        raise RuntimeError("Step 11 failed: no Malaysian compound produced "
                           "usable descriptors.")

    X_malaysia = np.vstack(descriptor_rows).astype(np.float32)
    print(f"      {X_malaysia.shape[0]} compounds x {X_malaysia.shape[1]} features")

    probabilities = np.column_stack([
        (p[:, 1] if p.shape[1] > 1 else p[:, 0])
        for p in model.predict_proba(X_malaysia)])

    results = frame.loc[valid_rows].copy().reset_index(drop=True)
    for class_index, column in enumerate(GHS_LABEL_COLUMNS):
        threshold = thresholds[column]["threshold_f1"]
        results[f"PRED_{column}"] = (probabilities[:, class_index] >= threshold
                                     ).astype(int)
        results[f"PROB_{column}"] = np.round(probabilities[:, class_index], 4)

    return results, probabilities


def score_malaysian_performance(results):
    """
    Score the predictions against PubChem's GHS labels.

    Only compounds that actually have a GHS classification in PubChem can be
    scored; the rest are predicted but excluded from the metrics, and that is
    stated explicitly.
    """
    print("\n[11b] Scoring against PubChem GHS ground truth ...")
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, matthews_corrcoef)

    scorable = results[results["GHS_Found"]]
    print(f"      {len(scorable)} of {len(results)} compounds have ground truth")

    if len(scorable) == 0:
        log_issue("11b", "no Malaysian compound has GHS ground truth in "
                         "PubChem - per-class metrics cannot be computed.")
        return pd.DataFrame(), {}

    rows = []
    for column in GHS_LABEL_COLUMNS:
        y_true = scorable[column].to_numpy()
        y_predicted = scorable[f"PRED_{column}"].to_numpy()
        rows.append({
            "GHS_Column": column,
            "Pictogram_Code": column.split("_")[0],
            "Meaning": GHS_TRUE_MEANING[column],
            "N_True_Positive_In_Set": int(y_true.sum()),
            "Accuracy": round(float(accuracy_score(y_true, y_predicted)), 4),
            "Precision": round(float(precision_score(y_true, y_predicted,
                                                     zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_predicted,
                                               zero_division=0)), 4),
            "F1": round(float(f1_score(y_true, y_predicted, zero_division=0)), 4),
            "MCC": round(float(matthews_corrcoef(y_true, y_predicted))
                         if len(np.unique(y_true)) > 1 else np.nan, 4),
        })
    table = pd.DataFrame(rows)

    print("-" * 96)
    print(f"{'Class':<22}{'meaning':<26}{'n+':>5}{'Acc':>8}{'Prec':>8}"
          f"{'Rec':>8}{'F1':>8}{'MCC':>8}")
    print("-" * 96)
    for row in rows:
        mcc = f"{row['MCC']:>8.3f}" if pd.notna(row["MCC"]) else f"{'n/a':>8}"
        print(f"{row['GHS_Column']:<22}"
              f"{row['Meaning'].split('(')[0].strip()[:24]:<26}"
              f"{row['N_True_Positive_In_Set']:>5}{row['Accuracy']:>8.3f}"
              f"{row['Precision']:>8.3f}{row['Recall']:>8.3f}"
              f"{row['F1']:>8.3f}{mcc}")
    print("-" * 96)

    # Per-sector performance, which is what a sector regulator will look at.
    sector_rows = []
    for sector, group in scorable.groupby("Sector"):
        correct = total = 0
        for column in GHS_LABEL_COLUMNS:
            correct += int((group[column].to_numpy() ==
                            group[f"PRED_{column}"].to_numpy()).sum())
            total += len(group)
        # A "hazard hit" is a real hazard the model successfully flagged.
        true_positive = sum(int(((group[c] == 1) &
                                 (group[f"PRED_{c}"] == 1)).sum())
                            for c in GHS_LABEL_COLUMNS)
        actual_positive = sum(int((group[c] == 1).sum())
                              for c in GHS_LABEL_COLUMNS)
        sector_rows.append({
            "Sector": sector,
            "N_Compounds": len(group),
            "Label_Accuracy": round(correct / max(total, 1), 4),
            "Hazard_Recall": round(true_positive / max(actual_positive, 1), 4),
            "N_Actual_Hazard_Labels": actual_positive,
            "N_Hazards_Correctly_Flagged": true_positive,
        })
    sector_table = pd.DataFrame(sector_rows)

    print("\n      Performance by sector")
    print("-" * 96)
    print(f"{'Sector':<28}{'n':>5}{'label accuracy':>18}{'hazard recall':>17}"
          f"{'hazards found':>18}")
    print("-" * 96)
    for row in sector_rows:
        # Built separately: nesting the same quote inside an f-string is not
        # valid in Python 3.11.
        found_ratio = (f"{row['N_Hazards_Correctly_Flagged']}"
                       f"/{row['N_Actual_Hazard_Labels']}")
        print(f"{row['Sector']:<28}{row['N_Compounds']:>5}"
              f"{row['Label_Accuracy']:>18.3f}{row['Hazard_Recall']:>17.3f}"
              f"{found_ratio:>18}")
    print("-" * 96)

    return table, sector_table


# ===========================================================================
# 11c - JOHOR 2019 ANALYSIS
# ===========================================================================
def analyse_johor_2019(results):
    """
    Report which hazards the framework would have flagged for the Sungai Kim
    Kim chemicals.

    This is the question the whole project is built around: if this tool had
    existed in March 2019, what would it have said about the waste being
    tipped into that river?
    """
    print("\n[11c] Johor 2019 (Sungai Kim Kim) analysis ...")
    johor = results[results["Sector"] == "Johor 2019 Emergency"].copy()
    if johor.empty:
        log_issue("11c", "no Johor 2019 compound could be resolved.")
        return johor, {}

    print(f"      {len(johor)} incident chemicals analysed\n")
    print("-" * 108)
    print(f"{'Compound':<20}{'role':<40}{'hazards the model would have flagged'}")
    print("-" * 108)

    flagged_counts = {column: 0 for column in GHS_LABEL_COLUMNS}
    for _, row in johor.iterrows():
        flagged = [c.split("_")[0] for c in GHS_LABEL_COLUMNS
                   if row[f"PRED_{c}"] == 1]
        for column in GHS_LABEL_COLUMNS:
            flagged_counts[column] += int(row[f"PRED_{column}"])
        print(f"{str(row['Requested_Name'])[:19]:<20}"
              f"{str(row['Incident_Role'])[:39]:<40}"
              f"{', '.join(flagged) if flagged else 'none above threshold'}")
    print("-" * 108)

    print("\n      Hazard classes flagged across the incident mixture")
    print("-" * 80)
    for column in GHS_LABEL_COLUMNS:
        count = flagged_counts[column]
        if count:
            bar = "#" * count
            print(f"      {column:<22}{count:>3}/{len(johor)}  {bar}")
    print("-" * 80)

    # How many of the flags the model raised are confirmed by PubChem.
    scorable = johor[johor["GHS_Found"]]
    agreement = {}
    if len(scorable):
        for column in GHS_LABEL_COLUMNS:
            actual = int(scorable[column].sum())
            caught = int(((scorable[column] == 1) &
                          (scorable[f"PRED_{column}"] == 1)).sum())
            if actual:
                agreement[column] = f"{caught}/{actual}"
        print(f"\n      Of the incident chemicals with GHS labels in PubChem "
              f"({len(scorable)} compounds),")
        print(f"      the model recovered these real hazards: ")
        for column, ratio in agreement.items():
            print(f"         {column:<22} {ratio}")

    return johor, flagged_counts


# ===========================================================================
# 11d - PDF REPORT
# ===========================================================================
def generate_pdf_report(results, class_table, sector_table, johor,
                        flagged_counts, global_summary, output_path):
    """
    Write the Malaysian validation report as a PDF aimed at DOSH.

    reportlab's platypus engine is used, which lays out flowable paragraphs
    and tables and handles page breaks automatically.
    """
    print("\n[11d] Generating the Malaysian validation PDF report ...")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=17,
                                 spaceAfter=14)
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"],
                                   fontSize=13, spaceBefore=14, spaceAfter=7,
                                   textColor=colors.HexColor("#1a4d7a"))
    body_style = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5,
                                leading=13.5, spaceAfter=7)

    document = SimpleDocTemplate(output_path, pagesize=A4,
                                 leftMargin=2 * cm, rightMargin=2 * cm,
                                 topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                                 title="Malaysian Industrial Validation Report")
    story = []

    story.append(Paragraph("Malaysian Industrial Validation Report", title_style))
    story.append(Paragraph(
        "Interpretable Machine Learning for Predicting GHS Chemical Hazard "
        "Classifications", styles["Heading3"]))
    story.append(Paragraph(
        f"Sareer Ahmad, MSc Physical Chemistry, University of Peshawar<br/>"
        f"Prepared for consideration by the Department of Occupational Safety "
        f"and Health (DOSH), Malaysia<br/>"
        f"Generated {datetime.now().strftime('%d %B %Y')}", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # ---- 1. purpose --------------------------------------------------------
    story.append(Paragraph("1. Purpose and scope", heading_style))
    story.append(Paragraph(
        "This report tests whether a machine-learning model trained on global "
        "GHS classification data performs adequately on the chemicals actually "
        "handled by Malaysian industry. Four sectors were examined - palm oil "
        "processing, rubber processing, petrochemicals and semiconductor "
        "manufacturing in Penang - together with the chemicals implicated in "
        "the Sungai Kim Kim incident at Pasir Gudang, Johor, in March 2019.",
        body_style))
    story.append(Paragraph(
        f"The best-performing model, <b>{global_summary.get('best_model', 'n/a')}"
        f"</b>, was selected on a scaffold-split global test set of "
        f"{global_summary.get('n_test_compounds', 0):,} compounds, where it "
        f"achieved a mean AUC-ROC of "
        f"{global_summary.get('mean_auc_per_model', {}).get(global_summary.get('best_model'), float('nan')):.3f} "
        f"across the nine GHS hazard classes. The same model, the same "
        f"descriptors and the same calibrated decision thresholds were applied "
        f"unchanged here.", body_style))

    # ---- 2. dataset --------------------------------------------------------
    story.append(Paragraph("2. The Malaysian validation set", heading_style))
    sector_counts = results.groupby("Sector").size()
    data = [["Sector", "Compounds resolved", "With GHS ground truth"]]
    for sector in sector_counts.index:
        group = results[results["Sector"] == sector]
        data.append([sector, str(len(group)), str(int(group["GHS_Found"].sum()))])
    table = Table(data, colWidths=[8 * cm, 4.2 * cm, 4.5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))

    substitutions = results[results["Substitution_Note"] != ""]
    if len(substitutions):
        story.append(Paragraph(
            "<b>Substitutions.</b> Three entries in the proposal name materials "
            "rather than single molecules. A representative structure was used "
            "for each, and this is recorded so that no reader is misled: " +
            "; ".join(f"{r['Requested_Name']} &rarr; {r['PubChem_Name']}"
                      for _, r in substitutions.iterrows()) + ".", body_style))

    # ---- 3. performance ----------------------------------------------------
    story.append(Paragraph("3. Performance on Malaysian chemicals", heading_style))
    if len(class_table):
        data = [["GHS class", "Meaning", "n+", "Acc.", "Prec.", "Recall", "F1"]]
        for _, row in class_table.iterrows():
            data.append([row["Pictogram_Code"],
                         row["Meaning"].split("(")[0].strip()[:26],
                         str(row["N_True_Positive_In_Set"]),
                         f"{row['Accuracy']:.2f}", f"{row['Precision']:.2f}",
                         f"{row['Recall']:.2f}", f"{row['F1']:.2f}"])
        table = Table(data, colWidths=[1.7 * cm, 6.1 * cm, 1.1 * cm, 1.5 * cm,
                                       1.6 * cm, 1.7 * cm, 1.5 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef3f8")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.25 * cm))

    if len(sector_table):
        story.append(Paragraph("<b>By sector.</b>", body_style))
        data = [["Sector", "n", "Label accuracy", "Hazard recall"]]
        for _, row in sector_table.iterrows():
            data.append([row["Sector"], str(row["N_Compounds"]),
                         f"{row['Label_Accuracy']:.3f}",
                         f"{row['Hazard_Recall']:.3f}"])
        table = Table(data, colWidths=[7.5 * cm, 1.5 * cm, 3.8 * cm, 3.5 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#eef3f8")]),
        ]))
        story.append(table)

    story.append(PageBreak())

    # ---- 4. Johor ----------------------------------------------------------
    story.append(Paragraph("4. Sungai Kim Kim, Pasir Gudang, March 2019",
                           heading_style))
    story.append(Paragraph(
        "On 7 March 2019 chemical waste was illegally discharged into the "
        "Sungai Kim Kim river at Pasir Gudang, Johor. Vapour rising from the "
        "river affected more than 2,500 people, the majority of them "
        "schoolchildren, and 111 schools were closed. Acrylonitrile and "
        "acrolein were identified by the Ministry of Health as the principal "
        "agents. The table below shows what this framework would have "
        "predicted for each of the chemicals named in the official and "
        "peer-reviewed accounts of the incident.", body_style))

    if len(johor):
        data = [["Compound", "Role in the incident", "Hazards flagged"]]
        for _, row in johor.iterrows():
            flagged = [c.split("_")[0] for c in GHS_LABEL_COLUMNS
                       if row[f"PRED_{c}"] == 1]
            data.append([str(row["Requested_Name"])[:22],
                         str(row["Incident_Role"])[:44],
                         ", ".join(flagged) if flagged else "none"])
        table = Table(data, colWidths=[3.4 * cm, 7.6 * cm, 5.7 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8b1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8eeee")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.25 * cm))

        top_flagged = sorted(flagged_counts.items(), key=lambda kv: -kv[1])[:4]
        story.append(Paragraph(
            "<b>Interpretation.</b> Across the incident mixture the framework "
            "most frequently flagged " +
            ", ".join(f"{c.split('_')[0]} ({GHS_TRUE_MEANING[c].split('(')[0].strip()}, "
                      f"{n}/{len(johor)} compounds)"
                      for c, n in top_flagged if n) +
            ". A screening tool producing this profile from nothing but the "
            "chemical structures would have supported an early decision to "
            "treat the discharge as an acute inhalation hazard rather than as "
            "ordinary waste.", body_style))

    # ---- 5. recommendations ------------------------------------------------
    story.append(Paragraph("5. Recommendations for DOSH adoption", heading_style))
    recommendations = [
        ("Use as a triage tool, not a replacement for testing. The framework "
         "assigns a hazard profile from structure alone in under a second, "
         "which makes it suitable for prioritising which of thousands of "
         "registered chemicals should be tested first. It cannot substitute "
         "for laboratory determination under the CLASS Regulations 2013."),
        ("Operate at the safety-first threshold. Each hazard class has a "
         "calibrated threshold that catches at least 90% of true hazards, at "
         "the cost of more false alarms. For regulatory screening this is the "
         "correct operating point: a false alarm costs an unnecessary test, "
         "whereas a missed hazard can cost lives."),
        ("Treat low-confidence predictions as requiring testing. Any compound "
         "whose predicted probability falls between 0.4 and 0.6 for a hazard "
         "class should be flagged for laboratory assessment rather than "
         "accepted either way."),
        ("Extend the training data with Malaysian regulatory sources. The "
         "model is trained on ECHA, HSDB, NITE-CMC and Safe Work Australia "
         "data. Incorporating Malaysian CLASS Regulations classifications "
         "would improve coverage of locally significant chemicals, "
         "particularly palm-oil and rubber-processing auxiliaries."),
        ("Inorganic and organometallic compounds are the known weak point. "
         "The descriptors used here are designed for organic molecules. "
         "Predictions for simple inorganic salts, metal oxides and mineral "
         "acids should be treated with particular caution."),
    ]
    for index, text in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{index}.</b> {text}", body_style))

    # ---- 6. limitations and disclaimer -------------------------------------
    story.append(Paragraph("6. Limitations", heading_style))
    uncertain = []
    for _, row in results.iterrows():
        probabilities = [row[f"PROB_{c}"] for c in GHS_LABEL_COLUMNS]
        if any(0.4 <= p <= 0.6 for p in probabilities):
            uncertain.append(str(row["Requested_Name"]))
    story.append(Paragraph(
        f"<b>Compounds with uncertain predictions.</b> "
        f"{len(uncertain)} of {len(results)} compounds produced at least one "
        f"hazard probability in the ambiguous 0.4-0.6 band"
        + (": " + ", ".join(uncertain[:18]) +
           (" and others." if len(uncertain) > 18 else ".")
           if uncertain else "."), body_style))
    story.append(Paragraph(
        "<b>Structural coverage.</b> The molecular descriptors used here "
        "describe covalent organic structures. Ionic solids, elemental "
        "materials and coordination compounds are represented only "
        "approximately, so several entries in the palm-oil and rubber sectors "
        "(sodium hydroxide, zinc oxide, sulfur, graphite) sit outside the "
        "model's reliable applicability domain.", body_style))
    story.append(Paragraph(
        "<b>Ground truth.</b> Performance can only be measured for compounds "
        "that already carry a GHS classification in PubChem. Compounds "
        "without one were predicted but could not be scored.", body_style))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>DISCLAIMER.</b> This prediction is a computational screening tool "
        "and does not replace laboratory testing or regulatory assessment "
        "under Malaysia's Occupational Safety and Health (Classification, "
        "Labelling and Safety Data Sheet of Hazardous Chemicals) Regulations "
        "2013 (CLASS Regulations).",
        ParagraphStyle("disclaimer", parent=body_style, fontSize=9,
                       textColor=colors.HexColor("#8b1a1a"),
                       borderPadding=6, borderWidth=1,
                       borderColor=colors.HexColor("#8b1a1a"))))

    document.build(story)
    print(f"      PDF written: {output_path}")
    return output_path


# ===========================================================================
# MAIN
# ===========================================================================
def malaysia_validation():
    """Run the whole of Step 11."""
    total_start = time.time()
    print("=" * 78)
    print("STEP 11 - MALAYSIAN INDUSTRIAL VALIDATION")
    print("=" * 78)

    with open(stamped("STEP9_evaluation_summary.json"), encoding="utf-8") as fh:
        global_summary = json.load(fh)
    best_model_name = global_summary["best_model"]

    with open(stamped("STEP9_calibrated_thresholds.json"), encoding="utf-8") as fh:
        thresholds = json.load(fh)[best_model_name]

    # Step 7 pickled its wrapper class from __main__; register it before load.
    from step7_model_training import register_pickle_compatibility
    register_pickle_compatibility()

    # Must match the file Step 9 scored, so the Malaysian results describe the
    # same model as the reported global performance.
    candidates = {
        "RandomForest": ["STEP8_rf_tuned.pkl", "STEP7_rf_model.pkl"],
        "XGBoost": ["STEP8_xgb_tuned.pkl", "STEP7_xgb_models.pkl"],
        "SVM": ["STEP7_svm_model.pkl"],
        _ABL_NAME: [_ABL_FILE] if _ABL_FILE else [],
    }[best_model_name]
    path = next((os.path.join(DIR_MODELS, f) for f in candidates
                 if os.path.exists(os.path.join(DIR_MODELS, f))),
                os.path.join(DIR_MODELS, "STEP7_rf_model.pkl"))
    model = joblib.load(path)
    print(f"Using best model from Step 9: {best_model_name}")

    # ---- 11a ---------------------------------------------------------------
    frame = compile_malaysian_dataset()
    frame.to_csv(os.path.join(DIR_MALAYSIA,
                              f"STEP11_malaysian_chemicals_raw_{TODAY}.csv"),
                 index=False)

    # ---- 11b ---------------------------------------------------------------
    results, probabilities = predict_malaysian_compounds(frame, model, thresholds)
    class_table, sector_table = score_malaysian_performance(results)

    results.to_csv(stamped("STEP11_malaysia_validation_results.csv"), index=False)
    if len(class_table):
        class_table.to_csv(os.path.join(DIR_MALAYSIA,
                                        "STEP11_malaysia_per_class_metrics.csv"),
                           index=False)
    if len(sector_table):
        sector_table.to_csv(os.path.join(DIR_MALAYSIA,
                                         "STEP11_malaysia_per_sector_metrics.csv"),
                            index=False)

    # ---- 11c ---------------------------------------------------------------
    johor, flagged_counts = analyse_johor_2019(results)
    if len(johor):
        johor.to_csv(os.path.join(DIR_MALAYSIA,
                                  "STEP11_johor_2019_predictions.csv"), index=False)

    # ---- comparison figure -------------------------------------------------
    if len(class_table):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        global_auc = global_summary.get("auc_per_class_best_model", {})
        codes = [c.split("_")[0] for c in GHS_LABEL_COLUMNS]

        x = np.arange(len(codes))
        width = 0.38
        axes[0].bar(x - width / 2,
                    [global_auc.get(c) or 0 for c in GHS_LABEL_COLUMNS],
                    width, label="Global test set (AUC-ROC)",
                    color="#1f77b4", edgecolor="black", linewidth=0.5)
        axes[0].bar(x + width / 2, class_table["Accuracy"].to_numpy(), width,
                    label="Malaysian set (label accuracy)",
                    color="#ff7f0e", edgecolor="black", linewidth=0.5)
        axes[0].set_xticks(x); axes[0].set_xticklabels(codes, rotation=45)
        axes[0].set_ylabel("Score", fontsize=12)
        axes[0].set_title("Global test set vs Malaysian validation set",
                          fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")
        axes[0].set_ylim(0, 1.05)

        if len(sector_table):
            sector_sorted = sector_table.sort_values("Hazard_Recall")
            axes[1].barh(sector_sorted["Sector"], sector_sorted["Hazard_Recall"],
                         color=sns.color_palette("crest", len(sector_sorted)),
                         edgecolor="black", linewidth=0.5)
            axes[1].set_xlabel("Hazard recall (fraction of real hazards flagged)",
                               fontsize=11)
            axes[1].set_title("Coverage by Malaysian industrial sector",
                              fontsize=12, fontweight="bold")
            axes[1].grid(alpha=0.3, axis="x"); axes[1].set_xlim(0, 1.05)

        fig.tight_layout()
        figure_path = os.path.join(DIR_MALAYSIA,
                                   "STEP11_malaysia_performance_comparison.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n      Figure saved: {figure_path}")

    # ---- 11d ---------------------------------------------------------------
    pdf_path = stamped("STEP11_malaysia_validation_report.pdf")
    try:
        generate_pdf_report(results, class_table, sector_table, johor,
                            flagged_counts, global_summary, pdf_path)
    except Exception as exc:
        log_issue("11d", f"PDF generation failed ({type(exc).__name__}: {exc}). "
                         f"FALLBACK: writing the same content as a text report.")
        pdf_path = stamped("STEP11_malaysia_validation_report.txt")
        with open(pdf_path, "w", encoding="utf-8") as fh:
            fh.write("MALAYSIAN INDUSTRIAL VALIDATION REPORT\n")
            fh.write("=" * 70 + "\n\n")
            fh.write(class_table.to_string() + "\n\n")
            fh.write(sector_table.to_string() + "\n\n")
            fh.write(johor.to_string() + "\n")

    log_path = os.path.join(DIR_LOGS, f"STEP11_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 11 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(ISSUE_LOG) if ISSUE_LOG else "No issues encountered.\n")

    elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print("STEP 11 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Compiled chemicals from four Malaysian industrial")
    print("                sectors plus the Johor 2019 incident, predicted their")
    print("                GHS hazards with the best model, scored the")
    print("                predictions, and wrote a DOSH-facing PDF report.")
    print(f"COMPOUNDS     : {len(results)} predicted, "
          f"{int(results['GHS_Found'].sum())} with ground truth to score against")
    if len(class_table):
        print(f"MEAN ACCURACY : {class_table['Accuracy'].mean():.3f} "
              f"across the nine hazard classes")
        print(f"MEAN RECALL   : {class_table['Recall'].mean():.3f}")
    print(f"JOHOR 2019    : {len(johor)} incident chemicals analysed")
    print(f"OUTPUT FILES  : {stamped('STEP11_malaysia_validation_results.csv')}")
    print(f"                {pdf_path}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged (see {log_path})")
    print(f"ELAPSED       : {elapsed / 60:.1f} minutes")
    print("=" * 78)


if __name__ == "__main__":
    malaysia_validation()
