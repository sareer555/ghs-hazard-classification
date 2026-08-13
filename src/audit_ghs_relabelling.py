"""
AUDIT THE GHS07/08/09 RELABELLING AGAINST LIVE PUBCHEM DATA
=============================================================
The columns now named GHS07_Irritant, GHS08_HealthHazard and
GHS09_Environmental were originally named GHS07_HealthHazard,
GHS08_Environmental and GHS09_Irritant - a three-way rotation of the
descriptive suffixes relative to the official United Nations numbering. The
rename (src/migrate_column_names.py) changed only the column headers; the
argument for why the underlying data needed no change is that every value was
always bound to the numeric pictogram code, and the label matrix is a plain
array whose column order never moved.

That argument is correct, but a reviewer cannot verify it from a code comment.
This script checks it directly: it selects compounds from the cleaned dataset
that carry exactly one of the three renamed pictograms, queries PubChem's own
live GHS Classification page for that compound by CID, and confirms the
pictogram PubChem reports today matches the one recorded under the corrected
column name. Agreement is the actual evidence that the rename was a naming
correction and not a data-mapping error; disagreement would mean the argument
above is wrong and the labels need real correction, not just renaming.

Fifteen compounds are checked: five for each of the three renamed columns,
chosen to carry only that one of the three so the test isolates each column
rather than being satisfied by a compound that happens to carry more than one.

Output
    EXTRA_relabel_audit.csv    one row per compound, PubChem's answer and
                                whether it matches
    EXTRA_relabel_audit.json   the pass count the Methods section quotes

Author : Sareer Ahmad
"""

import os
import re
import sys
import json
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (GHS_LABEL_COLUMNS, PICTOGRAM_CODE_TO_COLUMN,
                        RANDOM_SEED, stamped, seed_everything)

seed_everything()

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
BACKOFF_SCHEDULE = [1, 2, 4, 8]
RATE_LIMIT_DELAY = 0.25

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "GHS-hazard-classification-audit/1.0 "
                  "(academic research; Sareer Ahmad; sareerkh9194@gmail.com)"
})

# The three columns under review, and the code a live PubChem lookup must
# report to confirm each one.
RENAMED_COLUMNS = {
    "GHS07_Irritant": "GHS07",
    "GHS08_HealthHazard": "GHS08",
    "GHS09_Environmental": "GHS09",
}
N_PER_COLUMN = 5


def pubchem_get(url, params=None, timeout=60):
    """
    Fetch one PubChem URL, retrying on a transient failure.

    Identical policy to the one used for the Malaysian validation in Step 11:
    503/429/504 are PubChem being busy and are retried with backoff; 404 is a
    real answer and is returned as-is.
    """
    for wait in [0] + BACKOFF_SCHEDULE:
        if wait:
            time.sleep(wait)
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
        except Exception:
            continue
        if response.status_code == 404:
            return response
        if response.status_code in (429, 503, 504):
            continue
        time.sleep(RATE_LIMIT_DELAY)
        return response
    return None


def select_candidates(dataset_path):
    """
    Choose five compounds per renamed column, each carrying only that one of
    the three renamed pictograms, with a usable name and CID.

    Restricting to compounds with exactly one of the three isolates each
    column: a compound carrying two of them would pass the check even if one
    of the three were mapped incorrectly, as long as the other one happened to
    be right.
    """
    df = pd.read_csv(dataset_path, low_memory=False,
                     usecols=["CID", "Name"] + GHS_LABEL_COLUMNS)
    df = df[df["Name"].notna() & (df["Name"].astype(str).str.len() > 0)
            & df["CID"].notna()]

    rng = np.random.RandomState(RANDOM_SEED)
    chosen = []
    for column in RENAMED_COLUMNS:
        others = [c for c in RENAMED_COLUMNS if c != column]
        mask = df[column] == 1
        for other in others:
            mask &= df[other] == 0
        pool = df[mask]
        if len(pool) < N_PER_COLUMN:
            raise SystemExit(
                f"Only {len(pool)} candidates for {column} carrying no other "
                f"renamed pictogram - need {N_PER_COLUMN}.")
        sample = pool.sample(N_PER_COLUMN, random_state=rng)
        for _, row in sample.iterrows():
            chosen.append({"column": column,
                           "expected_code": RENAMED_COLUMNS[column],
                           "cid": int(row["CID"]), "name": row["Name"]})
    return chosen


def live_pictograms(cid):
    """
    Return the set of GHS pictogram codes PubChem reports today for one CID.

    Uses the same endpoint and the same image-URL regular expression as the
    Malaysian validation, so this audit and the reported validation results
    are reading PubChem the same way.
    """
    response = pubchem_get(
        f"{PUBCHEM_BASE}/pug_view/data/compound/{cid}/JSON",
        params={"heading": "GHS Classification"}, timeout=90)
    if response is None or response.status_code != 200:
        return None
    return set(re.findall(r"/(GHS0[1-9])\.svg", response.text))


def main():
    """Run the audit and write the results."""
    print("=" * 78)
    print("AUDITING THE GHS07/08/09 RELABELLING AGAINST LIVE PUBCHEM DATA")
    print("=" * 78)

    candidates = select_candidates(stamped("STEP3_cleaned_ghs_dataset.csv"))
    print(f"\n{len(candidates)} compounds selected, "
         f"{N_PER_COLUMN} per renamed column\n")

    rows = []
    print(f"{'Column':<22}{'CID':>9}  {'Name':<32}{'Expect':>7}"
         f"{'Live':>7}  Result")
    print("-" * 92)
    for item in candidates:
        codes = live_pictograms(item["cid"])
        if codes is None:
            outcome = "LOOKUP FAILED"
        elif item["expected_code"] in codes:
            outcome = "CONFIRMED"
        else:
            outcome = "MISMATCH"
        rows.append({**item, "live_pictograms": ",".join(sorted(codes or [])),
                    "outcome": outcome})
        name = str(item["name"])[:30]
        print(f"{item['column']:<22}{item['cid']:>9}  {name:<32}"
             f"{item['expected_code']:>7}  "
             f"{','.join(sorted(codes or [])) or '(none)':<8}{outcome}")

    table = pd.DataFrame(rows)
    table.to_csv(stamped("EXTRA_relabel_audit.csv"), index=False)

    confirmed = int((table["outcome"] == "CONFIRMED").sum())
    failed = table[table["outcome"] != "CONFIRMED"]

    summary = {
        "n_checked": len(table),
        "n_confirmed": confirmed,
        "n_per_column": N_PER_COLUMN,
        "columns_audited": list(RENAMED_COLUMNS),
        "all_confirmed": bool(confirmed == len(table)),
        "random_seed": RANDOM_SEED,
    }
    with open(stamped("EXTRA_relabel_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("-" * 92)
    print(f"\n{confirmed} of {len(table)} confirmed against live PubChem data.")
    if len(failed):
        print("\nNOT confirmed:")
        print(failed[["column", "cid", "name", "outcome"]].to_string(index=False))
        print("\nThis means the relabelling needs to be investigated further "
             "before the Methods paragraph can claim it was naming-only.")
    else:
        print("\nEvery compound checked out: the pictogram recorded under the "
             "corrected column name is the one PubChem's live GHS "
             "Classification page reports for that compound today, for all "
             "three renamed columns. This supports the claim that the bug was "
             "naming-only.")

    print(f"\n   {stamped('EXTRA_relabel_audit.csv')}")
    print(f"   {stamped('EXTRA_relabel_audit.json')}")
    return 0 if confirmed == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
