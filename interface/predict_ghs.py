"""
STEP 12 - COMMAND-LINE INTERFACE
================================
The documented fallback for the Streamlit application: a plain command-line
tool that needs no web browser and no graphical display.

Usage
-----
    python predict_ghs.py "C1=CC=CC=C1"          # a SMILES string
    python predict_ghs.py --name benzene         # a chemical name
    python predict_ghs.py --cas 71-43-2          # a CAS number
    python predict_ghs.py --name acrolein --pdf report.pdf
    python predict_ghs.py --batch chemicals.txt --csv results.csv

Author : Sareer Ahmad
"""

import os
import sys
import csv
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ghs_config import GHS_LABEL_COLUMNS, GHS_TRUE_MEANING
from ghs_predictor import GHSPredictor, build_pdf_report, DISCLAIMER

# Plain-text bars used instead of colour, so the output is readable when
# redirected to a file or viewed in a terminal without colour support.
LEVEL_MARK = {"HIGH": "[!!]", "MODERATE": "[! ]", "BORDERLINE": "[! ]",
              "LOW": "[  ]"}


def print_report(resolved, prediction, predictor):
    """Print one compound's complete hazard profile as formatted text."""
    print()
    print("=" * 78)
    print("GHS CHEMICAL HAZARD SCREENING REPORT")
    print("=" * 78)
    print(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model     : {predictor.model_name}")
    print("-" * 78)
    print(f"Name      : {resolved.get('name')}")
    print(f"Formula   : {resolved.get('formula')}")
    print(f"SMILES    : {resolved.get('smiles')}")
    print(f"Source    : {resolved.get('source')}")
    print("-" * 78)

    print("\nPREDICTED GHS HAZARD PROFILE")
    print("-" * 78)
    print(f"{'':<5}{'Code':<6}{'Hazard':<34}{'Conf.':>8}{'raw':>8}"
          f"{'thresh':>8}{'Flagged':>8}")
    print("-" * 78)
    for hazard in prediction["hazards"]:
        mark = LEVEL_MARK[hazard["level"]]
        flagged = "YES" if hazard["predicted"] else "no"
        print(f"{mark:<5}{hazard['code']:<6}{hazard['meaning']:<34}"
              f"{hazard['calibrated_percent']:>7.0f}%{hazard['percent']:>7.1f}%"
              f"{hazard['threshold_percent']:>7.1f}%{flagged:>8}")
    print("-" * 78)
    print("  [!!] above 70%   [! ] 30-70%   [  ] below 30%")
    print("  'Conf.' is calibrated so that 50% is this class's own decision")
    print("  threshold; 'raw' is the unscaled model probability. Each class has")
    print("  a different threshold, so raw values are not comparable across")
    print("  classes.")

    flagged = [h["code"] for h in prediction["hazards"] if h["predicted"]]
    print(f"\nSUMMARY: {len(flagged)} of 9 hazard classes flagged"
          + (f" - {', '.join(flagged)}" if flagged else ""))

    print("\nCALCULATED MOLECULAR PROPERTIES")
    print("-" * 78)
    for key, value in prediction["properties"].items():
        print(f"   {key:<44}{value}")

    if prediction.get("shap"):
        print("\nWHY THE MODEL REACHED THIS CONCLUSION")
        print("-" * 78)
        print("The ten most influential molecular descriptors. A positive net")
        print("effect pushed this molecule towards 'hazardous'.")
        print("-" * 78)
        print(f"{'Descriptor':<30}{'net effect':>14}{'total influence':>18}")
        print("-" * 78)
        for name, signed, total in predictor.top_shap_features(
                prediction["shap"], n=10):
            print(f"{name:<30}{signed:>+14.5f}{total:>18.5f}")
        print("-" * 78)

    print()
    print("!" * 78)
    print("DISCLAIMER")
    # Wrap the disclaimer to the terminal width by hand.
    words, line = DISCLAIMER.split(), ""
    for word in words:
        if len(line) + len(word) + 1 > 76:
            print(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    print(line)
    print("!" * 78)
    print()


def process_one(predictor, user_input, input_type):
    """Resolve and predict one chemical, printing errors clearly."""
    resolved = predictor.resolve_input(user_input, input_type)
    if not resolved["ok"]:
        print(f"\nERROR: {resolved['error']}", file=sys.stderr)
        print("Try a different synonym, or pass the SMILES string directly.",
              file=sys.stderr)
        return None, None

    prediction = predictor.predict(resolved["smiles"], compute_shap=True)
    if not prediction["ok"]:
        print(f"\nERROR: {prediction['error']}", file=sys.stderr)
        return resolved, None
    return resolved, prediction


def main():
    """Parse the command line and run the requested prediction."""
    parser = argparse.ArgumentParser(
        description="Predict GHS hazard classifications from chemical structure.",
        epilog="If no flag is given, the positional argument is auto-detected "
               "as a SMILES string, CAS number or chemical name.")
    parser.add_argument("query", nargs="?",
                        help="SMILES string, chemical name or CAS number")
    parser.add_argument("--name", help="look the chemical up by name")
    parser.add_argument("--cas", help="look the chemical up by CAS number")
    parser.add_argument("--smiles", help="use this SMILES string directly")
    parser.add_argument("--batch", help="a text file with one chemical per line")
    parser.add_argument("--csv", help="write the results to this CSV file")
    parser.add_argument("--pdf", help="write a PDF report to this path "
                                      "(single compound only)")
    parser.add_argument("--no-shap", action="store_true",
                        help="skip the SHAP explanation to run faster")
    arguments = parser.parse_args()

    # ---- work out what was asked for --------------------------------------
    if arguments.name:
        queries, input_type = [arguments.name], "name"
    elif arguments.cas:
        queries, input_type = [arguments.cas], "cas"
    elif arguments.smiles:
        queries, input_type = [arguments.smiles], "smiles"
    elif arguments.batch:
        if not os.path.exists(arguments.batch):
            print(f"ERROR: batch file '{arguments.batch}' does not exist.",
                  file=sys.stderr)
            return 1
        with open(arguments.batch, encoding="utf-8") as fh:
            queries = [line.strip() for line in fh
                       if line.strip() and not line.startswith("#")]
        input_type = "auto"
        print(f"Batch mode: {len(queries)} chemicals to process")
    elif arguments.query:
        queries, input_type = [arguments.query], "auto"
    else:
        parser.print_help()
        return 1

    # ---- load the model ---------------------------------------------------
    try:
        predictor = GHSPredictor()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Run the pipeline (Steps 1-9) before using this tool.",
              file=sys.stderr)
        return 1

    # ---- process ----------------------------------------------------------
    csv_rows = []
    for query in queries:
        resolved, prediction = process_one(predictor, query, input_type)
        if prediction is None:
            csv_rows.append({"Query": query, "Resolved": "FAILED"})
            continue

        if len(queries) == 1:
            print_report(resolved, prediction, predictor)
        else:
            flagged = [h["code"] for h in prediction["hazards"] if h["predicted"]]
            print(f"{query:<30} {str(resolved['name'])[:26]:<28} "
                  f"{', '.join(flagged) if flagged else 'none flagged'}")

        row = {"Query": query, "Resolved": resolved["name"],
               "Formula": resolved["formula"], "SMILES": resolved["smiles"]}
        for hazard in prediction["hazards"]:
            row[f"PROB_{hazard['column']}"] = hazard["probability"]
            row[f"PRED_{hazard['column']}"] = hazard["predicted"]
        csv_rows.append(row)

        if arguments.pdf and len(queries) == 1:
            try:
                build_pdf_report(resolved, prediction, predictor, arguments.pdf)
                print(f"PDF report written to: {arguments.pdf}")
            except Exception as exc:
                print(f"WARNING: the PDF could not be written ({exc}). "
                      f"The text report above is complete.", file=sys.stderr)

    # ---- CSV output --------------------------------------------------------
    if arguments.csv and csv_rows:
        fieldnames = sorted({key for row in csv_rows for key in row})
        # Keep the identity columns at the front for readability.
        for key in ("SMILES", "Formula", "Resolved", "Query"):
            if key in fieldnames:
                fieldnames.remove(key)
                fieldnames.insert(0, key)
        with open(arguments.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults written to: {arguments.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
