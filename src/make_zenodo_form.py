"""
ZENODO UPLOAD FORM - copy-paste values for every field
======================================================
Writes a plain text file listing exactly what to type or paste into each box
on the Zenodo "New upload" form, so nothing has to be composed on the spot.

Author : Sareer Ahmad
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import PROJECT_ROOT, stamped

OUT = os.path.join(PROJECT_ROOT, "zenodo_data_archive")
os.makedirs(OUT, exist_ok=True)


def main():
    clean = json.load(open(stamped("STEP3_cleaning_summary.json"), encoding="utf-8"))
    ev = json.load(open(stamped("STEP9_evaluation_summary.json"), encoding="utf-8"))
    split = json.load(open(stamped("STEP5_split_metadata.json"), encoding="utf-8"))
    desc = json.load(open(stamped("STEP4_descriptor_metadata.json"), encoding="utf-8"))

    n = clean["final_cleaned_compounds"]
    best = ev["best_model"]
    auc = ev["mean_auc_per_model"][best]

    archive = os.path.join(OUT, "GHS_hazard_classification_data.zip")
    size_mb = os.path.getsize(archive) / 1e6 if os.path.exists(archive) else 0

    text = f"""================================================================================
ZENODO UPLOAD - PART 2, THE DATA RECORD
Copy each block into the matching box on the Zenodo form.
================================================================================

WHERE TO START
--------------------------------------------------------------------------------
Go to  https://zenodo.org/uploads/new
(or: zenodo.org -> the "+" / "New upload" button at the top right)

Zenodo will ask you to pick a community - you can SKIP that, it is optional.


STEP 1 - THE FILE
--------------------------------------------------------------------------------
Drag this one file into the upload box:

    {archive}
    ({size_mb:,.0f} MB - the upload will take a while; leave the tab open)

That is the only file to upload. Everything else is inside it.


STEP 2 - "Resource type"
--------------------------------------------------------------------------------
Choose:  Dataset


STEP 3 - "Title"
--------------------------------------------------------------------------------
Multi-label GHS hazard classification dataset and trained models for {n:,} chemical compounds


STEP 4 - "Creators"
--------------------------------------------------------------------------------
Click "Add creator" and fill in:

    Family name:   Ahmad
    Given names:   Sareer
    ORCID:         0009-0003-2580-091X
    Affiliation:   Federal Directorate of Education, Islamabad, Pakistan

(Typing the ORCID may auto-fill the name - check it reads "Sareer Ahmad".)


STEP 5 - "Description"
--------------------------------------------------------------------------------
Paste everything between the lines:

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This record contains the curated dataset, computed molecular descriptors and
trained models supporting the study "Interpretable Machine Learning for
Predicting GHS Chemical Hazard Classifications".

GHS hazard classifications were harvested from PubChem for {n:,} unique
chemical compounds and reconciled across five independent regulatory sources:
the European Chemicals Agency, Regulation (EC) No 1272/2008, the Hazardous
Substances Data Bank, NITE-CMC, and the Hazardous Chemical Information System
of Safe Work Australia. Where sources disagreed, labels were resolved by
majority vote.

Each compound is described by {desc['n_features_computed']:,} molecular
descriptors combining physicochemical properties, Morgan (ECFP4) and MACCS
fingerprints and topological indices, reduced to
{desc['n_features_after_variance_filter']:,} after variance filtering.

Models were evaluated on a Bemis-Murcko scaffold split
({split['n_train']:,} training / {split['n_val']:,} validation /
{split['n_test']:,} test) in which no chemical skeleton is shared between
partitions. {best} performed best, with a mean AUC-ROC of {auc:.3f} across the
nine GHS hazard classes.

CONTENTS
- Raw and cleaned datasets with SMILES, InChIKey, CAS and the nine binary
  hazard labels
- The molecular descriptor matrix and aligned label matrix
- Trained Random Forest, XGBoost and support vector machine models
- Full evaluation results with bootstrap confidence intervals
- SHAP interpretability tables
- Validation results for Malaysian industrial chemicals and the compounds
  implicated in the 2019 Sungai Kim Kim incident at Pasir Gudang, Johor

A README inside the archive describes every file.

Approximately 8 GB of intermediate NumPy arrays are deliberately excluded, as
they are reproducible exactly by re-running the analysis pipeline.

ANALYSIS CODE
https://github.com/sareer555/ghs-hazard-classification

LIVE APPLICATION
https://ghs-hazard-classification.streamlit.app

DISCLAIMER
These models are computational screening tools. They do not replace laboratory
testing or regulatory assessment under Malaysia's Occupational Safety and
Health (Classification, Labelling and Safety Data Sheet of Hazardous Chemicals)
Regulations 2013, or equivalent legislation elsewhere. Predictions must not be
used as the sole basis for any decision affecting human safety.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


STEP 6 - "License"
--------------------------------------------------------------------------------
Choose:  Creative Commons Attribution 4.0 International  (CC-BY-4.0)

This lets others reuse the data provided they cite you.


STEP 7 - "Keywords" (add one at a time)
--------------------------------------------------------------------------------
    GHS classification
    chemical hazard
    multi-label classification
    molecular descriptors
    QSAR
    SHAP
    cheminformatics
    chemical safety
    PubChem


STEP 8 - "Related works"   *** THIS IS THE IMPORTANT ONE ***
--------------------------------------------------------------------------------
This is what links your data record to your code record.

Click "Add related work", then:

    Relation:    Is supplement to
    Identifier:  <paste the DOI you got in Part 1 - the code DOI>
    Resource type: Software

Your Part 1 DOI looks like:  10.5281/zenodo.XXXXXXX
Find it on your Zenodo dashboard, or on the GitHub repository page as a badge.


STEP 9 - PUBLISH
--------------------------------------------------------------------------------
Click "Save draft" first if you want to come back to it.

When you are ready, click "Publish".

  A published Zenodo record CANNOT be deleted. Check the fields before
  publishing. You can add new versions later, but not withdraw the record -
  which is exactly why journals trust DOIs.

After publishing you get a second DOI, for the data. You then have two:

    Code DOI  (Part 1)  -> cite in the Code Availability statement
    Data DOI  (Part 2)  -> cite in the Data Availability statement

Send me both and I will insert them into the manuscript.


================================================================================
COMMON CONFUSIONS
================================================================================

"Which DOI do I cite - it gave me two?"
    Each record has a CONCEPT DOI (always resolves to the newest version) and a
    VERSION DOI (frozen to one upload). Cite the VERSION DOI in the paper, so a
    reviewer sees exactly what produced your numbers.

"Do I need a community?"
    No. Communities are curated collections. Skip it.

"It says my upload failed / timed out."
    {size_mb:,.0f} MB on a slow connection can exceed the browser timeout. Save
    the draft first, then retry the file upload - Zenodo resumes drafts.

"Should the data record be a separate record at all?"
    Yes. Code and data have different licences, different update cycles and
    different audiences. Linking them with "Is supplement to" is the standard
    arrangement.

Generated {datetime.now():%d %B %Y, %H:%M}
"""

    path = os.path.join(OUT, "ZENODO_FORM_part2.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nSaved to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
