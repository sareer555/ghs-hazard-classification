"""
SHARED PREDICTION ENGINE
========================
The logic used by both the Streamlit web application (app.py) and the
command-line tool (predict_ghs.py). Keeping it in one place means the two
interfaces can never drift apart and give different answers for the same
chemical.

What it does
------------
  * resolve a chemical name, CAS number or SMILES string to a structure
  * compute the same descriptors used to train the model
  * apply the trained model and its calibrated thresholds
  * explain the prediction with SHAP
  * draw the molecule
  * write a PDF report

Author : Sareer Ahmad
"""

import os
import re
import io
import sys
import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (get_ablation_identity, PROJECT_ROOT, DIR_MODELS, DIR_FEATURES,
                        GHS_LABEL_COLUMNS, GHS_TRUE_MEANING, stamped)

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

from step4_descriptors import descriptors_for_one_molecule, build_feature_names

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")

# The name and file of the Step 7 ablation model, read from the metadata Step 7
# wrote rather than assumed, so that the name always matches the resampling that
# actually ran. Looked up once here, at import time, because it only reads a
# small JSON file and every GHSPredictor needs the same answer.
_ABL_NAME, _ABL_FILE, _ = get_ablation_identity()

DISCLAIMER = (
    "This prediction is a computational screening tool and does not replace "
    "laboratory testing or regulatory assessment under Malaysia's CLASS "
    "Regulations 2013."
)

# Molecules beyond this size are rejected: descriptor calculation on very
# large structures (polymers, proteins) is slow and the model was never
# trained on anything like them.
MAX_HEAVY_ATOMS = 200


class GHSPredictor:
    """
    Loads the trained model once and answers prediction requests.

    Loading the model takes a few seconds, so the object is created once and
    reused for every chemical the user submits.
    """

    def __init__(self, model_name=None):
        """Load the best model, its thresholds and the feature definitions."""
        # Which model won in Step 9?
        summary_path = stamped("STEP9_evaluation_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
            self.model_name = model_name or summary["best_model"]
            self.global_auc = summary.get("auc_per_class_best_model", {})
        else:
            self.model_name = model_name or "RandomForest"
            self.global_auc = {}

        candidates = {
            "RandomForest": ["STEP8_rf_tuned.pkl", "STEP7_rf_model.pkl"],
            "XGBoost": ["STEP8_xgb_tuned.pkl", "STEP7_xgb_models.pkl"],
            "SVM": ["STEP7_svm_model.pkl"],
            _ABL_NAME: [_ABL_FILE] if _ABL_FILE else [],
        }.get(self.model_name, ["STEP7_rf_model.pkl"])

        # Step 7 pickled its wrapper class from __main__, so it has to be
        # registered before those model files can be read from any other
        # script - including this one, the Streamlit app and the CLI.
        from step7_model_training import register_pickle_compatibility
        register_pickle_compatibility()

        self.model = None
        for filename in candidates:
            path = os.path.join(DIR_MODELS, filename)
            if os.path.exists(path):
                self.model = joblib.load(path)
                self.model_file = filename
                break
        if self.model is None:
            raise FileNotFoundError(
                f"No trained model found in {DIR_MODELS}. Run Steps 7-9 first.")

        # Calibrated decision thresholds from Step 9.
        threshold_path = stamped("STEP9_calibrated_thresholds.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, encoding="utf-8") as fh:
                self.thresholds = json.load(fh).get(self.model_name, {})
        else:
            self.thresholds = {}

        # Which descriptor columns survived the Step 4 variance filter.
        with open(stamped("STEP4_feature_names.txt"), encoding="utf-8") as fh:
            self.feature_names = [line.strip() for line in fh
                                  if line.strip() and not line.startswith("#")]
        all_names = build_feature_names()
        self.keep_positions = [all_names.index(n) for n in self.feature_names]

        self._explainers = {}   # SHAP explainers, built lazily and cached

    # -----------------------------------------------------------------------
    # 12a - INPUT RESOLUTION
    # -----------------------------------------------------------------------
    def resolve_input(self, user_input, input_type="auto"):
        """
        Turn whatever the user typed into a SMILES string.

        input_type may be "name", "cas", "smiles" or "auto". In auto mode the
        input is recognised by its shape: a CAS number matches the pattern
        digits-digits-digit, anything RDKit can parse is treated as SMILES,
        and everything else is looked up in PubChem as a name.

        Returns
        -------
        dict with keys: ok, smiles, name, formula, cid, source, error
        """
        result = {"ok": False, "smiles": None, "name": None, "formula": None,
                  "cid": None, "source": None, "error": None}
        user_input = (user_input or "").strip()
        if not user_input:
            result["error"] = "No input was provided."
            return result

        # ---- decide what kind of input this is ----------------------------
        if input_type == "auto":
            if CAS_PATTERN.match(user_input):
                input_type = "cas"
            elif Chem.MolFromSmiles(user_input) is not None:
                input_type = "smiles"
            else:
                input_type = "name"

        # ---- SMILES entered directly --------------------------------------
        if input_type == "smiles":
            mol = Chem.MolFromSmiles(user_input)
            if mol is None:
                result["error"] = (
                    f"'{user_input}' is not a valid SMILES string. RDKit could "
                    f"not parse it. Check for unbalanced brackets or "
                    f"unrecognised element symbols.")
                return result
            result.update({"ok": True, "smiles": Chem.MolToSmiles(mol),
                           "name": "(structure entered directly)",
                           "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                           "source": "user-supplied SMILES"})
            return result

        # ---- name or CAS: ask PubChem -------------------------------------
        route = "name"      # PubChem resolves CAS numbers through the name route
        try:
            response = requests.get(
                f"{PUBCHEM_BASE}/compound/{route}/"
                f"{requests.utils.quote(user_input)}"
                f"/property/SMILES,MolecularFormula,Title/JSON",
                timeout=30,
                headers={"User-Agent": "GHS-Hazard-Screening-Tool/1.0"})
        except requests.exceptions.Timeout:
            result["error"] = ("PubChem did not respond within 30 seconds. "
                               "Check your internet connection and try again, "
                               "or paste the SMILES string directly.")
            return result
        except requests.exceptions.RequestException as exc:
            result["error"] = (f"Could not reach PubChem ({type(exc).__name__}). "
                               f"Check your internet connection, or paste the "
                               f"SMILES string directly.")
            return result

        if response.status_code == 404:
            label = "CAS number" if input_type == "cas" else "chemical name"
            result["error"] = (f"PubChem has no record for the {label} "
                               f"'{user_input}'. Check the spelling, try a "
                               f"synonym, or paste the SMILES string directly.")
            return result
        if response.status_code != 200:
            result["error"] = (f"PubChem returned HTTP {response.status_code}. "
                               f"The service may be busy - please try again.")
            return result

        try:
            properties = response.json()["PropertyTable"]["Properties"][0]
        except (KeyError, IndexError, ValueError):
            result["error"] = "PubChem returned a response that could not be read."
            return result

        smiles = properties.get("SMILES") or properties.get("ConnectivitySMILES")
        if not smiles:
            result["error"] = (f"PubChem found '{user_input}' but holds no "
                               f"structure for it. This happens with mixtures "
                               f"and undefined materials.")
            return result

        result.update({"ok": True, "smiles": smiles,
                       "name": properties.get("Title", user_input),
                       "formula": properties.get("MolecularFormula"),
                       "cid": properties.get("CID"),
                       "source": f"PubChem CID {properties.get('CID')}"})
        return result

    # -----------------------------------------------------------------------
    # 12b - PREDICTION
    # -----------------------------------------------------------------------
    def predict(self, smiles, compute_shap=True):
        """
        Predict the nine GHS hazards for one structure.

        Returns
        -------
        dict with the hazard table, the descriptor vector, SHAP contributions
        and any error message.
        """
        output = {"ok": False, "error": None, "hazards": [], "shap": None,
                  "properties": {}}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            output["error"] = "RDKit could not parse this structure."
            return output

        # ---- size guard ---------------------------------------------------
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy > MAX_HEAVY_ATOMS:
            output["error"] = (
                f"This molecule has {n_heavy} heavy atoms, more than the "
                f"{MAX_HEAVY_ATOMS} limit. Descriptor calculation would be slow "
                f"and the model was never trained on structures this large, so "
                f"any prediction would be unreliable.")
            return output
        if n_heavy == 0:
            output["error"] = "This structure contains no heavy atoms."
            return output

        # ---- descriptors ---------------------------------------------------
        try:
            descriptors = descriptors_for_one_molecule(smiles)
            if descriptors is None:
                output["error"] = "Descriptor calculation failed for this molecule."
                return output
            descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0,
                                        neginf=0.0)
            X = descriptors[self.keep_positions].reshape(1, -1).astype(np.float32)
        except Exception as exc:
            output["error"] = f"Descriptor calculation failed: {exc}"
            return output

        # ---- model ---------------------------------------------------------
        try:
            probability_arrays = self.model.predict_proba(X)
            probabilities = np.array([
                (p[0, 1] if p.shape[1] > 1 else p[0, 0])
                for p in probability_arrays])
        except Exception as exc:
            output["error"] = f"The model could not score this molecule: {exc}"
            return output

        # ---- assemble the hazard table -------------------------------------
        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            probability = float(probabilities[class_index])
            threshold = float(self.thresholds.get(column, {}).get(
                "threshold_f1", 0.5))
            flagged = probability >= threshold

            # The colour must agree with the flag. Colouring by raw probability
            # instead produced badges that contradicted themselves: the
            # irritant class has a calibrated threshold near 0.05, because it
            # is present in most of the training data, so a compound scoring
            # 0.124 was correctly FLAGGED yet shown in green as though it were
            # safe. A safety officer reading that badge would draw exactly the
            # wrong conclusion.
            #
            # The raw probability is therefore rescaled so that each class's
            # own decision threshold sits at 0.5. The published bands - above
            # 0.7 high, 0.5 to 0.7 moderate, below 0.5 low - then apply to
            # every class consistently, and the colour can never disagree with
            # the flag. The unscaled probability is still reported alongside,
            # since that is what the model actually output.
            if threshold <= 0.0:
                calibrated = probability
            elif probability >= threshold:
                # At the threshold this is 0.5; at probability 1.0 it is 1.0.
                calibrated = 0.5 + 0.5 * (probability - threshold) / max(
                    1.0 - threshold, 1e-9)
            else:
                # At the threshold this is 0.5; at probability 0 it is 0.
                calibrated = 0.5 * probability / threshold

            if calibrated > 0.7:
                level, colour = "HIGH", "red"
            elif calibrated >= 0.5:
                level, colour = "MODERATE", "orange"
            elif calibrated >= 0.3:
                # Below the threshold but not comfortably so - worth a second
                # look rather than a clean green.
                level, colour = "BORDERLINE", "orange"
            else:
                level, colour = "LOW", "green"

            output["hazards"].append({
                "column": column,
                "code": column.split("_")[0],
                "meaning": GHS_TRUE_MEANING[column],
                "probability": round(probability, 4),
                "percent": round(100 * probability, 1),
                "calibrated_confidence": round(calibrated, 4),
                "calibrated_percent": round(100 * calibrated, 1),
                "threshold": round(threshold, 4),
                "threshold_percent": round(100 * threshold, 1),
                "predicted": int(flagged),
                "level": level,
                "colour": colour,
                "test_set_auc": self.global_auc.get(column),
            })

        # ---- readable physicochemical properties ---------------------------
        output["properties"] = {
            "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 2),
            "LogP (lipophilicity)": round(Descriptors.MolLogP(mol), 2),
            "Topological polar surface area (A^2)": round(Descriptors.TPSA(mol), 2),
            "Hydrogen-bond donors": int(Descriptors.NumHDonors(mol)),
            "Hydrogen-bond acceptors": int(Descriptors.NumHAcceptors(mol)),
            "Rotatable bonds": int(Descriptors.NumRotatableBonds(mol)),
            "Aromatic rings": int(Descriptors.NumAromaticRings(mol)),
            "Heavy atoms": int(n_heavy),
            "Molecular formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        }

        # ---- SHAP explanation ----------------------------------------------
        if compute_shap:
            try:
                output["shap"] = self._explain(X)
            except Exception as exc:
                # An explanation failure must never block the prediction.
                output["shap"] = None
                output["shap_error"] = str(exc)

        output["ok"] = True
        return output

    def _explain(self, X):
        """
        Compute the SHAP contributions of each descriptor, per hazard class.

        The explainer objects are cached because building one is far slower
        than using it.
        """
        import shap
        sub_models = (self.model.models if hasattr(self.model, "models")
                      else list(self.model.estimators_))
        explanations = {}
        for class_index, column in enumerate(GHS_LABEL_COLUMNS):
            sub_model = sub_models[class_index]
            if sub_model is None:
                continue
            if class_index not in self._explainers:
                self._explainers[class_index] = shap.TreeExplainer(sub_model)
            values = self._explainers[class_index].shap_values(
                X, check_additivity=False)
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, 1]
            explanations[column] = values[0]
        return explanations

    def top_shap_features(self, shap_values, n=10):
        """
        Rank descriptors by their total influence across all nine hazards.

        Returns a list of (feature name, signed contribution) pairs, largest
        absolute contribution first.
        """
        if not shap_values:
            return []
        total = np.zeros(len(self.feature_names))
        signed = np.zeros(len(self.feature_names))
        for values in shap_values.values():
            total += np.abs(values)
            signed += values
        order = np.argsort(total)[::-1][:n]
        return [(self.feature_names[i], float(signed[i]), float(total[i]))
                for i in order]


def draw_molecule(smiles, size=(420, 340)):
    """Render a molecule as a PNG image and return the raw bytes."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)          # lay the atoms out on a 2D grid
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def build_pdf_report(resolved, prediction, predictor, output_path=None):
    """
    Write the hazard profile as a PDF, including the mandatory disclaimer.

    Returns the PDF as raw bytes so that Streamlit can offer it for download
    without writing it to disk first.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image)

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5,
                          leading=13)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=12.5,
                             textColor=colors.HexColor("#1a4d7a"),
                             spaceBefore=12, spaceAfter=6)

    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm,
                                 rightMargin=2 * cm, topMargin=1.8 * cm,
                                 bottomMargin=1.8 * cm,
                                 title="GHS Hazard Screening Report")
    story = [Paragraph("GHS Hazard Screening Report", styles["Title"])]
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; "
        f"Model: {predictor.model_name}", body))
    story.append(Spacer(1, 0.3 * cm))

    # ---- identity ----------------------------------------------------------
    story.append(Paragraph("Chemical identity", heading))
    identity = [["Name", str(resolved.get("name", "unknown"))],
                ["Molecular formula", str(resolved.get("formula", "unknown"))],
                ["SMILES", str(resolved.get("smiles", ""))[:88]],
                ["Source", str(resolved.get("source", ""))]]
    table = Table(identity, colWidths=[4 * cm, 12.5 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)

    # ---- structure image ---------------------------------------------------
    try:
        png = draw_molecule(resolved["smiles"], size=(360, 280))
        if png:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(io.BytesIO(png), width=8 * cm, height=6.2 * cm))
    except Exception:
        pass    # a missing picture must not stop the report being produced

    # ---- hazard profile ----------------------------------------------------
    story.append(Paragraph("Predicted GHS hazard profile", heading))
    # Both the calibrated confidence and the raw probability are shown, with
    # the threshold that separates them, so the table cannot appear to
    # contradict itself the way a bare probability beside a FLAGGED label does.
    # The test-set AUC is kept alongside because it tells the reader how much
    # to trust each individual row - the classes are not equally predictable.
    #
    # The pictogram description in brackets is dropped here to make room; the
    # GHS code in the first column identifies the pictogram unambiguously.
    data = [["Code", "Hazard", "Conf.", "Raw", "Thresh.", "AUC", "Flagged"]]
    row_styles = []
    for index, hazard in enumerate(prediction["hazards"], start=1):
        short_name = hazard["meaning"].split("(")[0].strip()
        data.append([hazard["code"], short_name,
                     f"{hazard['calibrated_percent']:.0f}%",
                     f"{hazard['percent']:.1f}%",
                     f"{hazard['threshold_percent']:.1f}%",
                     (f"{hazard['test_set_auc']:.3f}"
                      if hazard["test_set_auc"] else "n/a"),
                     "YES" if hazard["predicted"] else "no"])
        fill = {"red": "#f7d4d4", "orange": "#fbe6cc",
                "green": "#dff0d8"}[hazard["colour"]]
        row_styles.append(("BACKGROUND", (0, index), (-1, index),
                           colors.HexColor(fill)))
    table = Table(data, colWidths=[1.4 * cm, 5.5 * cm, 1.9 * cm, 1.9 * cm,
                                   2.1 * cm, 1.9 * cm, 1.9 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ] + row_styles))
    story.append(table)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "<i><b>Conf.</b> is calibrated so that 50% is this class's own "
        "decision threshold; a hazard is flagged when the <b>Raw</b> model "
        "probability reaches that <b>Thresh.</b> Each class has its own "
        "threshold, fitted on validation data, so raw probabilities are not "
        "comparable between classes. <b>AUC</b> is the model's area under the "
        "ROC curve for that class on the held-out test set - it indicates how "
        "much weight to place on the row, and the classes are not equally "
        "predictable.</i>",
        ParagraphStyle("note", parent=body, fontSize=8,
                       textColor=colors.HexColor("#555555"))))

    # ---- physicochemical properties ---------------------------------------
    story.append(Paragraph("Calculated properties", heading))
    data = [[k, str(v)] for k, v in prediction["properties"].items()]
    table = Table(data, colWidths=[8.5 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
    ]))
    story.append(table)

    # ---- SHAP explanation --------------------------------------------------
    if prediction.get("shap"):
        story.append(Paragraph("Why the model reached this conclusion", heading))
        story.append(Paragraph(
            "The ten molecular descriptors below had the greatest influence on "
            "this prediction. A positive value pushed the molecule towards "
            "being classified as hazardous; a negative value pushed it towards "
            "safe.", body))
        data = [["Descriptor", "Net effect", "Total influence"]]
        for name, signed, total in predictor.top_shap_features(
                prediction["shap"], n=10):
            data.append([name, f"{signed:+.4f}", f"{total:.4f}"])
        table = Table(data, colWidths=[8.5 * cm, 4 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story.append(table)

    # ---- disclaimer --------------------------------------------------------
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        f"<b>DISCLAIMER.</b> {DISCLAIMER}",
        ParagraphStyle("disclaimer", parent=body, fontSize=9,
                       textColor=colors.HexColor("#8b1a1a"),
                       borderWidth=1, borderPadding=6,
                       borderColor=colors.HexColor("#8b1a1a"))))

    document.build(story)
    pdf_bytes = buffer.getvalue()
    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(pdf_bytes)
    return pdf_bytes
