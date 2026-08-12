"""
BUILD THE SUPPORTING INFORMATION PDF
====================================
Computational Toxicology asks for supplementary material to be uploaded as
files that "appear online in the exact same way as received" - they are not
typeset, checked or reformatted by the production team. So this has to be
readable as it leaves here.

The document collects Tables S0-S5 and File S1 into one PDF. The accompanying
Excel workbook carries the same tables in a form a reader can compute with; the
PDF is for reading, and where a table is too wide to be read on a page it shows
the columns that carry the argument and says plainly that the complete table is
in the workbook. Silently truncating a table would be worse than not printing
it at all.

Output: publication_materials/supporting_information/
        GHS_Supporting_Information.pdf

Author : Sareer Ahmad
"""

import os
import sys
import warnings
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Preformatted)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (DIR_PUB, PROJECT_ROOT, manuscript_title, stamped,
                        INK_PRIMARY, INK_SECONDARY, SERIES_HUE)

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(DIR_PUB, "supporting_information")
os.makedirs(OUT_DIR, exist_ok=True)
WORKBOOK = os.path.join(DIR_PUB, "tables",
                        "publication_supplementary_tables.xlsx")

PAGE = landscape(A4)
USABLE_WIDTH = PAGE[0] - 2.4 * cm

# Each entry: sheet in the workbook, the table's number in the paper, its
# caption, and the columns that must appear if the table has to be narrowed.
# The priority columns are the ones the manuscript actually discusses.
SHEETS = [
    ("S0_label_schema", "S0",
     "GHS label schema: column names, pictogram codes, their authoritative "
     "United Nations meanings, and the correspondence with the names used in "
     "the original study design.", None),
    ("S1_dataset_statistics", "S1",
     "Full dataset statistics after cleaning: compound counts, percentages "
     "and imbalance ratios per hazard class.", None),
    ("S2_hyperparameter_search", "S2",
     "Complete hyperparameter search results for all three algorithms.", None),
    ("S3_full_performance", "S3",
     "Full performance metrics: every model, class, metric and decision "
     "threshold, with bootstrap confidence intervals.",
     ["Model", "Pictogram_Code", "Threshold_Type", "Threshold_Value",
      "N_Test_Positive", "AUC_ROC", "AUC_CI95_lower", "AUC_CI95_upper",
      "Average_Precision", "F1", "MCC", "Precision", "Recall"]),
    ("S4_SHAP_top20", "S4",
     "The twenty most influential SHAP features per hazard class.",
     ["GHS_Column", "Rank", "Feature", "Mean_Abs_SHAP",
      "Mean_Signed_SHAP"]),
    ("S4b_SHAP_interpretation", "S4b",
     "Chemical interpretation of the five most influential features per "
     "hazard class.",
     ["Pictogram_Code", "Rank", "Feature", "Mean_Abs_SHAP", "SHAP_Direction",
      "What_The_Descriptor_Measures", "Chemical_Interpretation",
      "Matches_Chemical_Expectation"]),
    ("S5_malaysia_per_class", "S5",
     "Malaysian industrial validation, per hazard class.", None),
    ("S5b_malaysia_per_sector", "S5b",
     "Malaysian industrial validation, per sector.", None),
    ("S5c_johor_2019", "S5c",
     "The chemicals implicated in the 2019 Sungai Kim Kim incident at Pasir "
     "Gudang, Johor, with predicted and reference hazard profiles.",
     ["Query_Name", "PubChem_Name", "CID", "MolecularFormula", "Sector",
      "Incident_Role", "Substitution_Note"]),
]


def styles():
    """Build the paragraph styles the document uses."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SITitle", parent=base["Title"], fontSize=17, leading=21,
            textColor=colors.HexColor(INK_PRIMARY), spaceAfter=10),
        "subtitle": ParagraphStyle(
            "SISubtitle", parent=base["Normal"], fontSize=11.5, leading=16,
            alignment=TA_CENTER, textColor=colors.HexColor(INK_SECONDARY)),
        "heading": ParagraphStyle(
            "SIHeading", parent=base["Heading2"], fontSize=13, leading=16,
            textColor=colors.HexColor(SERIES_HUE), spaceBefore=2, spaceAfter=4),
        "caption": ParagraphStyle(
            "SICaption", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor(INK_SECONDARY), spaceAfter=8),
        "body": ParagraphStyle(
            "SIBody", parent=base["Normal"], fontSize=10, leading=14,
            textColor=colors.HexColor(INK_PRIMARY), spaceAfter=6),
        "note": ParagraphStyle(
            "SINote", parent=base["Normal"], fontSize=8.5, leading=11,
            textColor=colors.HexColor(INK_SECONDARY), spaceBefore=4),
    }


def shorten(value, limit=34):
    """
    Trim a cell so a wide table still fits, marking anything cut.

    The marker is three full stops rather than a single ellipsis character:
    reportlab's built-in fonts do not carry every Unicode glyph, and a missing
    one prints as a solid black box.
    """
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def select_columns(frame, priority):
    """
    Choose the columns to print, keeping the table readable.

    Returns the frame to print and a note explaining any omission, or an empty
    note when the whole table fits. Columns are never dropped silently: the
    note names how many were left out and points at the workbook.
    """
    if priority:
        keep = [c for c in priority if c in frame.columns]
        if keep and len(keep) < len(frame.columns):
            omitted = len(frame.columns) - len(keep)
            return frame[keep], (
                f"This table has {len(frame.columns)} columns in full; "
                f"{len(keep)} are shown here and {omitted} are omitted so the "
                f"table stays legible. The complete table, with every column, "
                f"is sheet '{{sheet}}' of the accompanying Excel workbook.")
        if keep:
            return frame[keep], ""
    return frame, ""


def build_table(frame, doc_styles):
    """Lay out one dataframe as a reportlab Table."""
    header = [Paragraph(f"<b>{str(c).replace('_', ' ')}</b>",
                        ParagraphStyle("th", fontSize=7.5, leading=9,
                                       textColor=colors.white))
              for c in frame.columns]
    rows = [header]
    for record in frame.itertuples(index=False):
        rows.append([Paragraph(shorten(v),
                               ParagraphStyle("td", fontSize=7, leading=8.6))
                     for v in record])

    column_width = USABLE_WIDTH / max(len(frame.columns), 1)
    table = Table(rows, colWidths=[column_width] * len(frame.columns),
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SERIES_HUE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9c8c2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6f9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def main():
    """Assemble the Supporting Information PDF."""
    print("=" * 78)
    print("BUILDING THE SUPPORTING INFORMATION PDF")
    print("=" * 78)

    if not os.path.exists(WORKBOOK):
        raise SystemExit(f"Workbook not found: {WORKBOOK}\nRun "
                         f"step13_publication.py first.")

    s = styles()
    workbook = pd.ExcelFile(WORKBOOK)
    clean = pd.read_json(stamped("STEP3_cleaning_summary.json"), typ="series")
    n_compounds = int(clean["final_cleaned_compounds"])

    story = []

    # ---- title page --------------------------------------------------------
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("Supporting Information", s["title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(manuscript_title(n_compounds), s["subtitle"]))
    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph(
        "Sareer Ahmad<br/>Federal Directorate of Education, Islamabad 44000, "
        "Pakistan<br/>ORCID 0009-0003-2580-091X", s["subtitle"]))
    story.append(Spacer(1, 1.0 * cm))

    contents = [["Item", "Contents"]]
    for _, number, caption, _ in SHEETS:
        contents.append([f"Table {number}", caption])
    contents.append(["File S1", "Complete software environment "
                                "specification, with pinned versions."])
    toc = Table(
        [[Paragraph(f"<b>{a}</b>", s["caption"]) if i == 0
          else Paragraph(a, s["caption"]),
          Paragraph(f"<b>{b}</b>", s["caption"]) if i == 0
          else Paragraph(b, s["caption"])]
         for i, (a, b) in enumerate(contents)],
        colWidths=[3.2 * cm, USABLE_WIDTH - 3.2 * cm])
    toc.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(SERIES_HUE)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(toc)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "The tables below are also provided as an Excel workbook, "
        "<i>publication_supplementary_tables.xlsx</i>, in which every table "
        "appears in full. The underlying dataset, descriptors and trained "
        "models are archived at "
        "<font color='#2a78d6'>https://doi.org/10.5281/zenodo.21876611</font> "
        "and the analysis code at "
        "<font color='#2a78d6'>https://doi.org/10.5281/zenodo.21903565</font>.",
        s["caption"]))
    story.append(PageBreak())

    # ---- tables ------------------------------------------------------------
    for sheet, number, caption, priority in SHEETS:
        if sheet not in workbook.sheet_names:
            print(f"   ! sheet missing, skipped: {sheet}")
            continue
        frame = workbook.parse(sheet)
        printed, note = select_columns(frame, priority)

        story.append(Paragraph(f"Table {number}", s["heading"]))
        story.append(Paragraph(caption, s["caption"]))
        story.append(build_table(printed, s))
        if note:
            story.append(Paragraph(note.format(sheet=sheet), s["note"]))
        story.append(Paragraph(
            f"{len(frame):,} rows. Source sheet: {sheet}.", s["note"]))
        story.append(PageBreak())
        print(f"   Table {number:<4} {len(printed.columns)} of "
              f"{len(frame.columns)} columns, {len(frame):,} rows")

    # ---- File S1 -----------------------------------------------------------
    env_path = stamped("STEP1_environment_requirements.txt")
    if os.path.exists(env_path):
        story.append(Paragraph("File S1", s["heading"]))
        story.append(Paragraph(
            "Complete software environment. Every version is pinned, and a "
            "single random seed of 42 is applied throughout, so the analysis "
            "reproduces exactly.", s["caption"]))
        with open(env_path, encoding="utf-8") as fh:
            text = fh.read().replace("\t", "    ")
        story.append(Preformatted(
            text, ParagraphStyle("mono", fontName="Courier", fontSize=7.6,
                                 leading=9.4,
                                 textColor=colors.HexColor(INK_PRIMARY))))
        print(f"   File S1   {len(text.splitlines())} lines")
    else:
        print("   ! environment file missing; File S1 omitted")

    out = os.path.join(OUT_DIR, "GHS_Supporting_Information.pdf")
    SimpleDocTemplate(
        out, pagesize=PAGE,
        leftMargin=1.2 * cm, rightMargin=1.2 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title="Supporting Information - GHS hazard classification",
        author="Sareer Ahmad").build(story)

    size = os.path.getsize(out) / 1024
    print(f"\n   {out}")
    print(f"   {size:,.0f} KB")
    print(f"\nUpload alongside the workbook at "
          f"{os.path.relpath(WORKBOOK, PROJECT_ROOT)}")
    print(f"Generated {datetime.now():%d %B %Y}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
