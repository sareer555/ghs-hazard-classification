"""
STEP 12 VERIFICATION
====================
Exercises every path through the prediction interface, including the error
paths, so that the Streamlit application and the command-line tool are known
to work rather than merely known to import.

Run after Step 9 has produced the calibrated thresholds.

Author : Sareer Ahmad
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import GHS_LABEL_COLUMNS, GHS_TRUE_MEANING
from ghs_predictor import GHSPredictor, draw_molecule, build_pdf_report

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    """Record one test result."""
    (PASSED if condition else FAILED).append(name)
    print(f"   [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


def pdf_contains(pdf_bytes, phrase):
    """
    Search the visible text of a PDF for a phrase.

    None of the words on the page appear in the raw file bytes. reportlab
    encodes each content stream twice: it compresses the text with zlib and
    then ASCII85-encodes the result, so a stream begins with characters like
    'Gb"0WHX+kZrs*'. Both layers have to be undone before the text is
    readable. Skipping this makes a perfectly correct PDF look as though its
    text is missing.

    Word spacing is applied with separate PDF operators, so the phrase is
    matched word by word rather than as one string.
    """
    import re as _re
    import zlib
    import base64

    words = [w.encode() for w in phrase.split()]
    if all(w in pdf_bytes for w in words):    # an uncompressed PDF
        return True

    for match in _re.finditer(rb"stream\r?\n(.*?)endstream", pdf_bytes,
                              _re.DOTALL):
        chunk = match.group(1).strip()
        for decoder in (
                lambda b: zlib.decompress(base64.a85decode(b, adobe=True)),
                lambda b: zlib.decompress(b),
                lambda b: base64.a85decode(b, adobe=True)):
            try:
                text = decoder(chunk)
            except Exception:
                continue
            if all(w in text for w in words):
                return True
            break        # this decoder worked; no need to try the others
    return False


def main():
    """Run every interface check and print a summary."""
    print("=" * 78)
    print("STEP 12 - INTERFACE VERIFICATION")
    print("=" * 78)

    # ---- the predictor must load at all -----------------------------------
    print("\n1. Loading the predictor")
    started = time.time()
    predictor = GHSPredictor()
    check("model loads", predictor.model is not None,
          f"{predictor.model_name} from {predictor.model_file} "
          f"in {time.time() - started:.1f}s")
    check("thresholds loaded", len(predictor.thresholds) == 9,
          f"{len(predictor.thresholds)} classes")
    check("feature names loaded", len(predictor.feature_names) > 0,
          f"{len(predictor.feature_names)} features")

    # ---- 12a: the three input routes ---------------------------------------
    print("\n2. Input resolution (the three routes in proposal step 12a)")
    for label, value, mode in [
            ("Option A - chemical name", "benzene", "name"),
            ("Option B - CAS number", "71-43-2", "cas"),
            ("Option C - SMILES string", "C1=CC=CC=C1", "smiles"),
            ("auto-detect a name", "acrylonitrile", "auto"),
            ("auto-detect a SMILES", "CCO", "auto")]:
        resolved = predictor.resolve_input(value, mode)
        check(label, resolved["ok"],
              f"-> {resolved.get('name')} ({resolved.get('formula')})"
              if resolved["ok"] else resolved.get("error", "")[:70])

    # ---- 12e: error handling ------------------------------------------------
    print("\n3. Error handling (proposal step 12e)")
    for label, value, mode in [
            ("invalid SMILES rejected", "this-is-not-a-molecule((", "smiles"),
            ("unknown name rejected", "zzqqxxnotachemical123", "name"),
            ("empty input rejected", "", "auto")]:
        resolved = predictor.resolve_input(value, mode)
        check(label, (not resolved["ok"]) and bool(resolved["error"]),
              (resolved.get("error") or "")[:70])

    # a molecule too large for the model to handle responsibly
    huge = "C" * 250
    prediction = predictor.predict(huge, compute_shap=False)
    check("oversized molecule rejected",
          (not prediction["ok"]) and "heavy atoms" in (prediction["error"] or ""),
          (prediction.get("error") or "")[:70])

    # ---- 12b + 12c: full prediction ----------------------------------------
    print("\n4. Full prediction pipeline (proposal steps 12b and 12c)")
    resolved = predictor.resolve_input("benzene", "name")
    started = time.time()
    prediction = predictor.predict(resolved["smiles"], compute_shap=True)
    elapsed = time.time() - started

    check("prediction succeeds", prediction["ok"], f"{elapsed:.2f}s")
    check("nine hazard classes returned", len(prediction["hazards"]) == 9)
    check("probabilities are valid",
          all(0.0 <= h["probability"] <= 1.0 for h in prediction["hazards"]))
    check("colour coding assigned",
          all(h["colour"] in ("red", "orange", "green")
              for h in prediction["hazards"]))
    check("molecular properties computed", len(prediction["properties"]) >= 8)

    # The badge colour and the FLAGGED label must never contradict each other.
    # They did: colour came from the raw probability while the flag came from
    # the class's calibrated threshold, so the irritant class - whose threshold
    # is near 5% - showed a green badge labelled FLAGGED.
    contradictions = [
        h for h in prediction["hazards"]
        if (h["predicted"] and h["colour"] == "green")
        or (not h["predicted"] and h["colour"] == "red")]
    check("colour never contradicts the flag", not contradictions,
          "; ".join(f"{h['code']} {h['colour']}/"
                    f"{'FLAGGED' if h['predicted'] else 'not flagged'}"
                    for h in contradictions) or "all nine consistent")

    # Calibrated confidence must cross 50% exactly when the hazard is flagged.
    misaligned = [
        h for h in prediction["hazards"]
        if bool(h["calibrated_confidence"] >= 0.5) != bool(h["predicted"])]
    check("calibrated confidence agrees with the decision", not misaligned,
          "; ".join(f"{h['code']} {h['calibrated_percent']}%"
                    for h in misaligned) or "all nine aligned")
    check("SHAP explanation produced", prediction.get("shap") is not None,
          f"{len(prediction.get('shap') or {})} classes explained")

    top = predictor.top_shap_features(prediction.get("shap") or {}, n=10)
    check("top ten SHAP features returned", len(top) == 10,
          ", ".join(f[0] for f in top[:4]) + " ...")

    print("\n   Benzene hazard profile (a known flammable carcinogen):")
    print(f"      {'':<6}{'hazard':<40}{'conf.':>7}{'raw':>8}{'thresh':>8}")
    for hazard in prediction["hazards"]:
        marker = "FLAGGED" if hazard["predicted"] else ""
        print(f"      {hazard['code']:<6}{hazard['meaning']:<40}"
              f"{hazard['calibrated_percent']:>6.0f}%{hazard['percent']:>7.1f}%"
              f"{hazard['threshold_percent']:>7.1f}%  {marker}")

    # ---- structure drawing --------------------------------------------------
    print("\n5. Structure rendering")
    png = draw_molecule(resolved["smiles"])
    check("molecule renders to PNG", png is not None and len(png) > 1000,
          f"{len(png) if png else 0} bytes")

    # ---- 12d: PDF report ----------------------------------------------------
    print("\n6. PDF report generation (proposal step 12d)")
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "interface", "verification_report.pdf")
    pdf_bytes = build_pdf_report(resolved, prediction, predictor, pdf_path)
    check("PDF generated", pdf_bytes is not None and len(pdf_bytes) > 5000,
          f"{len(pdf_bytes):,} bytes -> {pdf_path}")
    check("PDF contains the required disclaimer",
          pdf_contains(pdf_bytes, "CLASS Regulations 2013"))

    # ---- the Johor 2019 chemicals -------------------------------------------
    print("\n7. Johor 2019 incident chemicals end to end")
    for name in ["acrylonitrile", "acrolein", "toluene"]:
        resolved = predictor.resolve_input(name, "name")
        if not resolved["ok"]:
            check(f"{name}", False, resolved["error"][:60])
            continue
        prediction = predictor.predict(resolved["smiles"], compute_shap=False)
        flagged = [h["code"] for h in prediction["hazards"] if h["predicted"]]
        check(f"{name}", prediction["ok"],
              f"flags: {', '.join(flagged) if flagged else 'none'}")

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"INTERFACE VERIFICATION: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED CHECKS:")
        for name in FAILED:
            print(f"   - {name}")
    print("=" * 78)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
