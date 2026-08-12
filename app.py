"""
STEP 12 - STREAMLIT WEB APPLICATION
===================================
A point-and-click interface to the GHS hazard classifier.

Run it with:
    streamlit run app.py

The user types a chemical name, a CAS number or a SMILES string; the app
resolves it to a structure, computes descriptors, applies the trained model,
explains the answer with SHAP and offers a downloadable PDF report.

Author : Sareer Ahmad
"""

import os
import sys
import io
import time

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the project's own modules importable. The location of src/ is worked out
# by looking beside this file first and then one directory up, so the app starts
# correctly whether it is launched from the repository root or from the copy in
# interface/ - a hosted deployment may be pointed at either one.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_APP_DIR, os.path.dirname(_APP_DIR)):
    _SRC_DIR = os.path.join(_candidate, "src")
    if os.path.isfile(os.path.join(_SRC_DIR, "ghs_config.py")):
        sys.path.insert(0, _SRC_DIR)
        break
else:
    raise RuntimeError(
        "Could not find the project's src/ directory beside app.py or one "
        "level above it. Start the app from the repository root with: "
        "streamlit run app.py")

from ghs_config import GHS_LABEL_COLUMNS, GHS_TRUE_MEANING
from ghs_predictor import GHSPredictor, draw_molecule, build_pdf_report, DISCLAIMER

st.set_page_config(page_title="GHS Chemical Hazard Screening",
                   page_icon="⚗️", layout="wide")


@st.cache_resource
def load_predictor():
    """
    Load the trained model once and keep it in memory.

    Streamlit re-runs this whole script on every interaction, so without the
    cache the model would be reloaded from disk on every keystroke.
    """
    return GHSPredictor()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("⚗️ GHS Chemical Hazard Screening")
st.markdown(
    "Predicts the nine **Globally Harmonised System** hazard pictograms for "
    "any chemical structure, and explains each prediction in terms of the "
    "molecular features that drove it.")

try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(f"**The trained model could not be loaded.**\n\n{exc}\n\n"
             f"Run the pipeline (Steps 1-9) before starting this application.")
    st.stop()

with st.sidebar:
    st.header("About this tool")
    st.markdown(f"""
**Model in use:** `{predictor.model_name}`

**Training data:** GHS classifications from PubChem, contributed by ECHA, the
EU CLP regulation, Japan's NITE-CMC, Safe Work Australia's HCIS and the US
HSDB.

**Validation:** Bemis-Murcko scaffold split, so the test compounds share no
chemical skeleton with the training set.

**Descriptors:** 19 physicochemical + 1024 Morgan (ECFP4) + 167 MACCS keys +
8 topological.
""")
    st.divider()
    st.subheader("Confidence colours")
    st.markdown("""
- 🔴 **red** - above 70%, hazard predicted with confidence
- 🟠 **orange** - 30-70%, close to the decision boundary
- 🟢 **green** - below 30%, no hazard predicted
""")
    st.caption(
        "The percentage shown is **calibrated confidence**, where 50% is that "
        "class's own decision threshold. Each of the nine classes has a "
        "different threshold, fitted on validation data — the irritant class "
        "sits near 5%, because most training compounds carry it. Raw model "
        "probabilities cannot be compared across classes, so they are shown "
        "underneath each badge rather than as the headline number.")
    st.divider()
    st.caption(f"⚠️ {DISCLAIMER}")

    st.divider()
    st.subheader("Pictogram numbering")
    st.caption(
        "Hazard names follow the official United Nations GHS numbering: "
        "GHS07 is the exclamation mark (irritant / harmful), GHS08 is the "
        "serious health hazard, and GHS09 is the environmental hazard.")


# ---------------------------------------------------------------------------
# 12a - INPUT SECTION
# ---------------------------------------------------------------------------
st.header("1. Enter a chemical")

input_mode = st.radio(
    "How would you like to identify the chemical?",
    ["Chemical name", "CAS number", "SMILES string"],
    horizontal=True,
    help="A name or CAS number is looked up in PubChem. A SMILES string is "
         "used directly and needs no internet connection.")

placeholders = {"Chemical name": "e.g. benzene, acrylonitrile, sodium hydroxide",
                "CAS number": "e.g. 71-43-2",
                "SMILES string": "e.g. C1=CC=CC=C1"}
type_map = {"Chemical name": "name", "CAS number": "cas",
            "SMILES string": "smiles"}

column_input, column_examples = st.columns([3, 1])
with column_input:
    user_input = st.text_input("Chemical identifier",
                               placeholder=placeholders[input_mode],
                               label_visibility="collapsed")
with column_examples:
    submitted = st.button("Predict hazards", type="primary",
                          use_container_width=True)

st.caption("**Try one of the Johor 2019 incident chemicals:** "
           "acrylonitrile · acrolein · benzene · toluene · hydrogen sulfide")

# ---------------------------------------------------------------------------
# 12b + 12c - PROCESSING AND RESULTS
# ---------------------------------------------------------------------------
if submitted and user_input.strip():

    # ---- 12e error handling: input resolution -----------------------------
    with st.spinner("Resolving the chemical structure ..."):
        resolved = predictor.resolve_input(user_input, type_map[input_mode])

    if not resolved["ok"]:
        st.error(f"**Could not identify this chemical.**\n\n{resolved['error']}")
        st.info("**What to try next:** check the spelling, use a different "
                "synonym, or switch to *SMILES string* and paste the structure "
                "directly. SMILES input works without an internet connection.")
        st.stop()

    st.success(f"Resolved to **{resolved['name']}** ({resolved['formula']})")

    # ---- run the model -----------------------------------------------------
    with st.spinner("Computing descriptors, predicting and explaining ..."):
        started = time.time()
        prediction = predictor.predict(resolved["smiles"], compute_shap=True)
        elapsed = time.time() - started

    if not prediction["ok"]:
        st.error(f"**Prediction failed.**\n\n{prediction['error']}")
        st.stop()

    st.caption(f"Completed in {elapsed:.2f} seconds")

    # ---- 12c results display ----------------------------------------------
    st.header("2. Results")
    column_structure, column_hazards = st.columns([1, 2])

    with column_structure:
        st.subheader("Structure")
        try:
            png = draw_molecule(resolved["smiles"])
            if png:
                st.image(png, use_container_width=True)
            else:
                st.info("The structure could not be drawn.")
        except Exception:
            st.info("The structure could not be drawn.")

        st.markdown(f"**Name:** {resolved['name']}")
        st.markdown(f"**Formula:** {resolved['formula']}")
        if resolved.get("cid"):
            st.markdown(f"**PubChem CID:** [{resolved['cid']}]"
                        f"(https://pubchem.ncbi.nlm.nih.gov/compound/"
                        f"{resolved['cid']})")
        st.code(resolved["smiles"], language=None)

    with column_hazards:
        st.subheader("GHS hazard profile")

        # Shown before the profile, not after it, so it cannot be read as a
        # footnote to a result the user has already believed.
        if prediction.get("domain_warning"):
            st.error("⚠️ " + prediction["domain_warning"])

        flagged = [h for h in prediction["hazards"] if h["predicted"]]
        if flagged:
            st.warning(f"**{len(flagged)} of 9 hazard classes flagged:** "
                       + ", ".join(h["code"] for h in flagged))
        else:
            st.info("No hazard class exceeded its decision threshold. This is "
                    "not a guarantee of safety - see the disclaimer.")

        # Nine badges in a 3x3 grid. The headline number is the calibrated
        # confidence, not the raw probability, so that the colour, the number
        # and the FLAGGED label always tell the same story. Each class has its
        # own decision threshold - the irritant class sits near 0.05 - so a raw
        # probability cannot be compared across classes or against a fixed 50%.
        for row_start in range(0, 9, 3):
            columns = st.columns(3)
            for offset, hazard in enumerate(
                    prediction["hazards"][row_start:row_start + 3]):
                with columns[offset]:
                    icon = {"red": "🔴", "orange": "🟠", "green": "🟢"}[
                        hazard["colour"]]
                    auc_text = (f"\nTest-set AUC: {hazard['test_set_auc']:.3f}"
                                if hazard["test_set_auc"] else "")
                    st.metric(
                        label=f"{icon} {hazard['code']}",
                        value=f"{hazard['calibrated_percent']:.0f}%",
                        delta=("FLAGGED" if hazard["predicted"] else "not flagged"),
                        delta_color=("inverse" if hazard["predicted"] else "off"),
                        help=(f"{hazard['meaning']}\n\n"
                              f"Calibrated confidence: "
                              f"{hazard['calibrated_percent']:.0f}% "
                              f"(50% = this class's decision threshold)\n"
                              f"Raw model probability: {hazard['percent']:.1f}%\n"
                              f"Decision threshold: "
                              f"{hazard['threshold_percent']:.1f}%"
                              f"{auc_text}"))
                    st.caption(f"{hazard['meaning']}  \n"
                               f"raw {hazard['percent']:.1f}% · "
                               f"threshold {hazard['threshold_percent']:.1f}%")

    # ---- full confidence table --------------------------------------------
    st.subheader("Confidence per hazard class")
    hazard_frame = pd.DataFrame([{
        "Code": h["code"],
        "Hazard": h["meaning"],
        "Confidence (%)": h["percent"],
        "Threshold": h["threshold"],
        "Flagged": "YES" if h["predicted"] else "no",
        "Model AUC on test set": (round(h["test_set_auc"], 3)
                                  if h["test_set_auc"] else None),
    } for h in prediction["hazards"]])
    st.dataframe(hazard_frame, use_container_width=True, hide_index=True)

    # ---- calculated properties --------------------------------------------
    st.subheader("Calculated molecular properties")
    property_columns = st.columns(3)
    for index, (key, value) in enumerate(prediction["properties"].items()):
        with property_columns[index % 3]:
            st.markdown(f"**{key}:** {value}")

    # ---- SHAP explanation --------------------------------------------------
    st.header("3. Why the model reached this conclusion")
    if prediction.get("shap"):
        top_features = predictor.top_shap_features(prediction["shap"], n=10)
        names = [f[0] for f in top_features][::-1]
        values = [f[1] for f in top_features][::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        colours = ["#c0392b" if v > 0 else "#2471a3" for v in values]
        ax.barh(names, values, color=colours, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Net SHAP contribution "
                      "(right = pushes towards hazardous, left = towards safe)",
                      fontsize=10)
        ax.set_title("The ten molecular features that most influenced this "
                     "prediction", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SHAP values come from cooperative game theory: each descriptor is "
            "treated as a player and given credit for its exact share of the "
            "final prediction. Red bars pushed this molecule towards being "
            "classified as hazardous, blue bars pushed it towards safe.")
    else:
        st.info("A SHAP explanation could not be generated for this molecule. "
                "The prediction above is unaffected."
                + (f" ({prediction.get('shap_error')})"
                   if prediction.get("shap_error") else ""))

    # ---- 12d downloadable PDF ---------------------------------------------
    st.header("4. Download the report")
    try:
        pdf_bytes = build_pdf_report(resolved, prediction, predictor)
        safe_name = "".join(ch for ch in str(resolved["name"])
                            if ch.isalnum() or ch in " -_")[:40].strip()
        st.download_button(
            "📄 Download the full PDF hazard report",
            data=pdf_bytes,
            file_name=f"GHS_hazard_report_{safe_name or 'compound'}.pdf",
            mime="application/pdf",
            type="primary")
    except Exception as exc:
        st.error(f"The PDF report could not be generated: {exc}")
        # FALLBACK: offer the same content as a CSV so the user is never
        # left without a downloadable record.
        st.download_button("⬇️ Download the results as CSV instead",
                           data=hazard_frame.to_csv(index=False),
                           file_name="GHS_hazard_report.csv", mime="text/csv")

    st.divider()
    st.error(f"⚠️ **DISCLAIMER.** {DISCLAIMER}")

elif submitted:
    st.warning("Please type a chemical name, CAS number or SMILES string first.")
