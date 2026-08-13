"""
ASSEMBLE THE SUBMISSION FOLDER
==============================
Collects everything Computational Toxicology asks for into one folder, in
upload order, and leaves everything else behind.

The distinction that matters is between the deliverables and the working
documents. publication_materials/manuscript/ holds both: the manuscript and the
highlights go to the journal, while the submission checklist and the author
notes are working papers that must not. Rather than trusting anyone to
remember which is which at the upload screen, this builds a folder that
contains only the former.

Files are numbered in the order the submission form asks for them, so the
folder reads top to bottom as the upload sequence.

Output: SUBMISSION/

Author : Sareer Ahmad
"""

import os
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import PROJECT_ROOT, DIR_PUB

OUT = os.path.join(PROJECT_ROOT, "SUBMISSION")
MANUSCRIPT = os.path.join(DIR_PUB, "manuscript")
FIGURES = os.path.join(DIR_PUB, "figures")
TABLES = os.path.join(DIR_PUB, "tables")
SUPPORTING = os.path.join(DIR_PUB, "supporting_information")

# (destination name, source path, what it is, where it goes in the form)
ITEMS = [
    ("01_MANUSCRIPT.docx",
     os.path.join(MANUSCRIPT, "GHS_manuscript.docx"),
     "The manuscript. A4, single column, double spaced, line numbered.",
     "Upload as 'Manuscript'."),
    ("02_HIGHLIGHTS.txt",
     os.path.join(MANUSCRIPT, "highlights.txt"),
     "Five bullet points, each within the 85-character limit.",
     "Upload as 'Highlights'. Required - the file name must contain the word "
     "'highlights', which it does."),
    ("03_COVER_LETTER.txt",
     None,                                   # extracted, not copied
     "Cover letter addressed to the editor.",
     "Paste into the 'Cover Letter' box, or upload if the form prefers a file."),
    ("04_GRAPHICAL_ABSTRACT.png",
     os.path.join(FIGURES, "TOC_graphic.png"),
     "Graphical abstract.",
     "Upload as 'Graphical Abstract'. Optional but encouraged."),
    ("05_SUPPORTING_INFORMATION.pdf",
     os.path.join(SUPPORTING, "GHS_Supporting_Information.pdf"),
     "Tables S0-S5c and File S1, 23 pages.",
     "Upload as 'Supplementary Material'."),
    ("06_SUPPLEMENTARY_TABLES.xlsx",
     os.path.join(TABLES, "publication_supplementary_tables.xlsx"),
     "The same tables in full, as a workbook.",
     "Upload as 'Supplementary Material'."),
    ("07_FIGURE_CAPTIONS.txt",
     os.path.join(FIGURES, "figure_captions.txt"),
     "Captions for all ten figures.",
     "Some forms ask for these separately; otherwise they are already in the "
     "manuscript."),
]


def extract_cover_letter():
    """
    Pull the cover letter out of the submission checklist.

    The checklist is a working document that is not submitted, but the cover
    letter inside it is. Taking it from there rather than keeping a second copy
    means the two cannot disagree.
    """
    path = os.path.join(MANUSCRIPT, "submission_checklist.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    start = text.find("Dear Editor,")
    end = text.find("2. AUTHOR CONTRIBUTIONS")
    if start == -1 or end == -1 or end < start:
        return None

    letter = text[start:end]
    # Drop the rule of dashes that separated the checklist's sections.
    lines = [l for l in letter.rstrip().split("\n")
             if not (l.strip() and set(l.strip()) <= set("-"))]
    return "\n".join(lines).rstrip() + "\n"


def main():
    """Build the submission folder and print the upload order."""
    print("=" * 78)
    print("ASSEMBLING THE SUBMISSION FOLDER")
    print("=" * 78)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    manifest, missing = [], []

    for name, source, description, where in ITEMS:
        target = os.path.join(OUT, name)
        if source is None:                       # the cover letter
            letter = extract_cover_letter()
            if letter is None:
                missing.append((name, "could not be extracted from the "
                                      "submission checklist"))
                continue
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(letter)
        elif os.path.exists(source):
            shutil.copy2(source, target)
        else:
            missing.append((name, source))
            continue
        manifest.append((name, os.path.getsize(target), description, where))

    # Figures, numbered so they sort in the order they appear in the paper.
    figure_rows = []
    for number in range(1, 11):
        matches = [f for f in os.listdir(FIGURES)
                   if re.match(rf"Figure{number}_.*\.png$", f)]
        if not matches:
            missing.append((f"Figure {number}", "not found"))
            continue
        source = os.path.join(FIGURES, matches[0])
        name = f"Figure{number:02d}_{matches[0].split('_', 1)[1]}"
        target = os.path.join(OUT, "figures", name)
        shutil.copy2(source, target)
        figure_rows.append((name, os.path.getsize(target)))

    # ---- the folder's own README ------------------------------------------
    lines = ["SUBMISSION PACKAGE",
             "Computational Toxicology (Elsevier)",
             f"Assembled {datetime.now():%d %B %Y}",
             "=" * 74, "",
             "Submit at https://submit.elsevier.com/COMTOX",
             "",
             "Upload in this order. Everything in this folder is submitted;",
             "nothing that belongs only to you has been copied in.", ""]
    for i, (name, size, description, where) in enumerate(manifest, 1):
        lines += [f"{name}   ({size/1024:,.0f} KB)",
                  f"    {description}",
                  f"    {where}", ""]
    lines += [f"figures/   ({len(figure_rows)} files)",
              "    Figures 1-10 at 500 dpi, one file each.",
              "    Upload each as a separate 'Figure' item, in order.", ""]
    for name, size in figure_rows:
        lines.append(f"      {name:<44}{size/1024:>8,.0f} KB")
    lines += ["",
              "-" * 74,
              "TWO THINGS THIS FOLDER CANNOT CONTAIN",
              "-" * 74,
              "",
              "1. The declaration of competing interests. Generate it at",
              "   https://declarations.elsevier.com, choose 'I have nothing to",
              "   declare', and upload the .docx the site produces.",
              "",
              "2. Your Editorial Manager account and the form itself.",
              "",
              "-" * 74,
              "DELIBERATELY NOT INCLUDED",
              "-" * 74,
              "",
              "submission_checklist.txt and AUTHOR_NOTES_do_not_submit.txt are",
              "working documents. They contain the cover letter draft, notes on",
              "the affiliation and the outstanding-task list. They stay in",
              "publication_materials/manuscript/ and are not submitted.",
              ""]
    with open(os.path.join(OUT, "README_UPLOAD_ORDER.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    for name, size, _, _ in manifest:
        print(f"   {name:<34}{size/1024:>9,.0f} KB")
    print(f"   {'figures/ (10 files)':<34}"
          f"{sum(s for _, s in figure_rows)/1024:>9,.0f} KB")

    if missing:
        print("\n   MISSING:")
        for name, why in missing:
            print(f"      {name}: {why}")
    else:
        print("\n   Every expected file was found.")

    print(f"\n   {OUT}")
    print("   README_UPLOAD_ORDER.txt lists the upload order.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
