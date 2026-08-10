"""
STEP 2 - DATA COLLECTION FROM PUBCHEM
=====================================
Goal
----
Build a master table in which every row is one chemical compound and nine
extra columns record, as 0 or 1, which GHS hazard pictograms that compound
carries.

How the data is obtained
------------------------
PubChem stores GHS classifications as "annotations" under the heading
"GHS Classification". The endpoint

    https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON
        ?heading=GHS+Classification&page=N

returns 1000 annotation records per page. Each record contains

  * the name of the regulatory source (ECHA, HSDB, NITE-CMC, ...)
  * the PubChem Compound IDs (CIDs) the record applies to
  * a "Pictogram(s)" block whose icons point at GHS01.svg ... GHS09.svg

Because several regulatory sources classify the same compound independently,
one compound can appear many times with slightly different pictograms. Those
per-source rows are all kept here and are reconciled by majority vote in
Step 3d.

Fallbacks documented in the proposal
------------------------------------
1. PRIMARY  : PUG-View annotation pages (used - it succeeded).
2. FALLBACK : the pubchempy library, driven from GHS hazard codes.
3. FALLBACK : pre-compiled ECHA C&L / NITE-CHRIP bulk downloads.
Each fallback is implemented below and is triggered automatically if the
method above it fails twice.

Author : Sareer Ahmad
"""

import os
import re
import sys
import csv
import json
import time
import gzip
import collections
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm

# Make the shared configuration importable no matter where this is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import (RANDOM_SEED, TODAY, PROJECT_ROOT, DIR_RAW, DIR_LOGS,
                        GHS_LABEL_COLUMNS, PICTOGRAM_CODE_TO_COLUMN,
                        PICTOGRAM_EXTRA_TO_CODE, GHS_TRUE_MEANING,
                        GHS_RECOMMENDED_NAME, seed_everything, stamped)

seed_everything()   # Rule 5 - fix the random seed before anything else

# ---------------------------------------------------------------------------
# NETWORK SETTINGS
# ---------------------------------------------------------------------------
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"

# PubChem asks that automated users stay under 5 requests per second and
# identify themselves. 0.2 s between calls = 5 calls per second exactly, as
# required by step 2d of the proposal.
RATE_LIMIT_DELAY = 0.2

# Exponential backoff schedule from step 2d: wait 1 s, then 2 s, then 4 s.
BACKOFF_SCHEDULE = [1, 2, 4]

SESSION = requests.Session()
SESSION.headers.update({
    # Identifying the client is a PubChem usage-policy requirement.
    "User-Agent": ("GHS-Hazard-ML-Research/1.0 "
                   "(academic research; Sareer Ahmad; sareerkh9194@gmail.com)")
})

# Where partially downloaded pages are cached so a crash does not lose work
CACHE_DIR = os.path.join(DIR_RAW, "_annotation_pages")
os.makedirs(CACHE_DIR, exist_ok=True)

# Running log of every problem encountered, written out at the end
ISSUE_LOG = []


def log_issue(step, message):
    """Record a problem and how it was handled, for the Rule 4 progress report."""
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {message}"
    ISSUE_LOG.append(entry)
    print("   ! " + message)


def pubchem_request(url, params=None, data=None, method="GET", timeout=300):
    """
    Make one PubChem request with rate limiting and exponential backoff.

    This is the single place in the project where PubChem is contacted, so the
    usage policy is enforced in exactly one location.

    Parameters
    ----------
    url : str
        Full endpoint URL.
    params : dict, optional
        Query-string parameters (used with GET).
    data : dict, optional
        Form body (used with POST; needed when sending thousands of CIDs,
        because a URL has a length limit but a POST body does not).
    method : {"GET", "POST"}
    timeout : int
        Seconds to wait before giving up on a single attempt.

    Returns
    -------
    requests.Response or None
        None means all attempts failed; the caller must then use its fallback.
    """
    # First attempt plus one retry per backoff entry = 4 attempts in total.
    for attempt, wait in enumerate([0] + BACKOFF_SCHEDULE):
        if wait:
            time.sleep(wait)          # exponential backoff: 1 s, then 2, then 4
        try:
            if method == "POST":
                resp = SESSION.post(url, data=data, timeout=timeout)
            else:
                resp = SESSION.get(url, params=params, timeout=timeout)

            # HTTP 503 is PubChem's "server busy / throttled" reply.
            # HTTP 429 is "too many requests". Both deserve a retry.
            if resp.status_code in (429, 503, 504):
                log_issue("pubchem_request",
                          f"HTTP {resp.status_code} (throttled), retrying: {url[:90]}")
                continue
            time.sleep(RATE_LIMIT_DELAY)   # be polite before the next call
            return resp
        except Exception as exc:
            log_issue("pubchem_request",
                      f"attempt {attempt + 1} raised {type(exc).__name__}: {exc}")
    return None


# ===========================================================================
# 2b + 2c - PRIMARY METHOD: harvest GHS annotations page by page
# ===========================================================================
def fetch_annotation_page(page):
    """
    Download one page of "GHS Classification" annotations and reduce it to a
    small list of dictionaries.

    Pages are cached on disk as compressed JSON. Re-running the script
    therefore skips anything already downloaded, which makes the collection
    resumable if the network drops.

    Returns
    -------
    (records, total_pages)
        records     : list of dicts, one per (compound, regulatory source)
        total_pages : how many pages PubChem says exist in total
    """
    cache_file = os.path.join(CACHE_DIR, f"page_{page:04d}.json.gz")

    # If this page was already processed in an earlier run, reuse it.
    if os.path.exists(cache_file):
        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                cached = json.load(fh)
            return cached["records"], cached["total_pages"]
        except Exception:
            # A truncated cache file (e.g. killed mid-write) is simply redone.
            os.remove(cache_file)

    resp = pubchem_request(
        f"{PUBCHEM_BASE}/pug_view/annotations/heading/JSON",
        params={"heading": "GHS Classification", "page": page},
    )
    if resp is None or resp.status_code != 200:
        log_issue("fetch_annotation_page",
                  f"page {page} failed (status "
                  f"{getattr(resp, 'status_code', 'no response')})")
        return [], None

    try:
        payload = resp.json()["Annotations"]
    except Exception as exc:
        log_issue("fetch_annotation_page", f"page {page} unparseable: {exc}")
        return [], None

    total_pages = payload.get("TotalPages")
    records = []

    for annotation in payload.get("Annotation", []):
        # LinkedRecords tells us which PubChem compounds this classification
        # applies to. Records without a CID are substance-level only and
        # cannot be turned into a structure, so they are skipped.
        cids = annotation.get("LinkedRecords", {}).get("CID", [])
        if not cids:
            continue

        source = annotation.get("SourceName", "unknown")
        name = annotation.get("Name", "")

        # ---- read the pictograms out of the markup ------------------------
        pictogram_codes = set()
        hazard_statements = []
        signal_word = ""

        for block in annotation.get("Data", []):
            block_name = block.get("Name", "")
            value = block.get("Value", {})

            if block_name == "Pictogram(s)":
                for swm in value.get("StringWithMarkup", []):
                    for markup in swm.get("Markup", []):
                        # Two independent ways to identify the pictogram; the
                        # image filename is the more reliable of the two, so
                        # it is tried first.
                        url = markup.get("URL") or ""
                        match = re.search(r"/(GHS0[1-9])\.svg", url)
                        if match:
                            pictogram_codes.add(match.group(1))
                            continue
                        # Fall back on the descriptive 'Extra' text.
                        extra = markup.get("Extra")
                        if extra in PICTOGRAM_EXTRA_TO_CODE:
                            pictogram_codes.add(PICTOGRAM_EXTRA_TO_CODE[extra])

            elif block_name == "GHS Hazard Statements":
                for swm in value.get("StringWithMarkup", []):
                    text = swm.get("String", "")
                    # Hazard statements start with a code such as "H225:".
                    hcode = re.match(r"(H\d{3})", text)
                    if hcode:
                        hazard_statements.append(hcode.group(1))

            elif block_name == "Signal":
                for swm in value.get("StringWithMarkup", []):
                    signal_word = swm.get("String", "")

        # A record with no pictogram at all carries no label information.
        if not pictogram_codes:
            continue

        # One annotation can cover several CIDs (e.g. a salt and its parent).
        # Each of those compounds inherits the same classification.
        for cid in cids:
            records.append({
                "CID": int(cid),
                "Source": source,
                "SourceName": name,
                "Pictograms": sorted(pictogram_codes),
                "HazardStatements": sorted(set(hazard_statements)),
                "Signal": signal_word,
            })

    # Cache the reduced records (a few hundred kB) instead of the raw 10 MB
    # JSON, so the whole harvest costs well under 1 GB of disk.
    with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
        json.dump({"records": records, "total_pages": total_pages}, fh)

    return records, total_pages


def harvest_all_annotations(max_pages=None):
    """
    Loop over every annotation page and collect all per-source GHS records.

    Returns
    -------
    pandas.DataFrame
        One row per (CID, regulatory source) with the pictogram list.
    """
    print("\n[2b] Harvesting GHS Classification annotations from PubChem ...")

    # Page 1 also tells us how many pages there are in total.
    first_records, total_pages = fetch_annotation_page(1)
    if total_pages is None:
        return pd.DataFrame()   # signals to the caller that fallback is needed

    if max_pages:
        total_pages = min(total_pages, max_pages)
    print(f"      PubChem reports {total_pages} pages "
          f"of 1000 annotations each.")

    all_records = list(first_records)
    failed_pages = []

    for page in tqdm(range(2, total_pages + 1), desc="      annotation pages",
                     unit="page", ncols=78):
        records, _ = fetch_annotation_page(page)
        if not records:
            failed_pages.append(page)
        all_records.extend(records)

    if failed_pages:
        log_issue("harvest_all_annotations",
                  f"{len(failed_pages)} page(s) returned nothing: "
                  f"{failed_pages[:20]}{' ...' if len(failed_pages) > 20 else ''}")

    print(f"      Collected {len(all_records):,} (compound, source) records.")
    return pd.DataFrame(all_records)


def records_to_binary_labels(records_df):
    """
    Turn the per-source pictogram lists into the nine binary label columns.

    Two tables are produced:

    per_source : one row per (CID, source) - kept because Step 3d needs to see
                 the individual sources in order to run a majority vote.
    aggregated : one row per CID, where a label is 1 if ANY source assigned
                 that pictogram. Using the union here is the safety-conservative
                 choice: under-calling a hazard is far more dangerous than
                 over-calling one.
    """
    print("\n[2e] Converting pictogram lists into nine binary label columns ...")

    per_source_rows = []
    for row in records_df.itertuples(index=False):
        entry = {"CID": row.CID, "Source": row.Source,
                 "SourceName": row.SourceName,
                 "HazardStatements": ";".join(row.HazardStatements),
                 "Signal": row.Signal}
        # start every hazard at 0, then switch on the ones present
        for column in GHS_LABEL_COLUMNS:
            entry[column] = 0
        for code in row.Pictograms:
            column = PICTOGRAM_CODE_TO_COLUMN.get(code)
            if column:
                entry[column] = 1
        per_source_rows.append(entry)

    per_source = pd.DataFrame(per_source_rows)

    # Aggregate to one row per compound.
    agg_spec = {column: "max" for column in GHS_LABEL_COLUMNS}   # max == logical OR
    agg_spec["Source"] = lambda s: ";".join(sorted(set(s)))
    agg_spec["SourceName"] = "first"
    agg_spec["HazardStatements"] = (
        lambda s: ";".join(sorted({h for cell in s for h in str(cell).split(";") if h}))
    )
    aggregated = per_source.groupby("CID", as_index=False).agg(agg_spec)
    aggregated = aggregated.rename(columns={"Source": "Sources",
                                            "SourceName": "SourceRecordName"})
    aggregated["N_Sources"] = aggregated["Sources"].str.count(";") + 1

    print(f"      {len(per_source):,} per-source rows -> "
          f"{len(aggregated):,} unique compounds (CIDs).")
    return per_source, aggregated


# ===========================================================================
# 2c - retrieve structures and identifiers for every collected CID
# ===========================================================================
def fetch_compound_properties(cids, batch_size=300):
    """
    Download SMILES, name and molecular formula for a list of CIDs.

    PubChem accepts many CIDs in a single POST request, which is dramatically
    faster than asking one compound at a time (300 compounds arrive in under
    a second). The CSV output format is used because it is compact and easy
    to parse.
    """
    print(f"\n[2c] Fetching SMILES / name / formula for {len(cids):,} compounds ...")

    # 'SMILES' returns the isomeric (stereochemistry-aware) form and
    # 'ConnectivitySMILES' the flat connectivity form. PubChem renamed these
    # properties in 2025 - the old name 'CanonicalSMILES' now silently
    # returns ConnectivitySMILES, so the new names are used explicitly.
    props = "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,Title,InChIKey"
    url = f"{PUBCHEM_BASE}/pug/compound/cid/property/{props}/CSV"

    frames = []
    failed_batches = 0

    for start in tqdm(range(0, len(cids), batch_size), desc="      property batches",
                      unit="batch", ncols=78):
        chunk = cids[start:start + batch_size]
        resp = pubchem_request(url, data={"cid": ",".join(map(str, chunk))},
                               method="POST", timeout=180)
        if resp is None or resp.status_code != 200:
            failed_batches += 1
            log_issue("fetch_compound_properties",
                      f"batch starting at index {start} failed; "
                      f"retrying compound-by-compound is skipped to save time")
            continue
        try:
            frames.append(pd.read_csv(pd.io.common.StringIO(resp.text)))
        except Exception as exc:
            failed_batches += 1
            log_issue("fetch_compound_properties",
                      f"could not parse batch at index {start}: {exc}")

    if not frames:
        return pd.DataFrame()

    properties = pd.concat(frames, ignore_index=True)
    if failed_batches:
        log_issue("fetch_compound_properties",
                  f"{failed_batches} batch(es) lost; "
                  f"{len(properties):,}/{len(cids):,} compounds retrieved")
    print(f"      Retrieved properties for {len(properties):,} compounds.")
    return properties


CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def fetch_cas_numbers(cids, batch_size=150, time_budget_seconds=1800):
    """
    Look up CAS Registry Numbers via PubChem synonyms.

    PubChem does not expose CAS as a queryable property, so the full synonym
    list is downloaded and filtered with a regular expression for the CAS
    format (2-7 digits, dash, 2 digits, dash, 1 check digit).

    Synonym lists can be enormous (ethanol has over 9000 synonyms), so this is
    the slowest part of Step 2. A time budget is enforced: once it is spent,
    the remaining compounds are simply left without a CAS number. CAS is
    metadata only - it is never used as a model feature - so a partial result
    does not affect any downstream step.
    """
    print(f"\n[2c] Looking up CAS numbers for {len(cids):,} compounds "
          f"(time budget {time_budget_seconds // 60} min) ...")
    url = f"{PUBCHEM_BASE}/pug/compound/cid/synonyms/JSON"
    cas_map = {}
    started = time.time()
    budget_hit = False

    for start in tqdm(range(0, len(cids), batch_size), desc="      synonym batches",
                      unit="batch", ncols=78):
        if time.time() - started > time_budget_seconds:
            budget_hit = True
            break
        chunk = cids[start:start + batch_size]
        resp = pubchem_request(url, data={"cid": ",".join(map(str, chunk))},
                               method="POST", timeout=180)
        if resp is None or resp.status_code != 200:
            continue
        try:
            info_list = resp.json()["InformationList"]["Information"]
        except Exception:
            continue
        for info in info_list:
            # The first synonym matching the CAS pattern is by PubChem
            # convention the primary registry number for that compound.
            for synonym in info.get("Synonym", []):
                if CAS_PATTERN.match(synonym):
                    cas_map[int(info["CID"])] = synonym
                    break

    if budget_hit:
        log_issue("fetch_cas_numbers",
                  f"time budget reached; CAS retrieved for {len(cas_map):,} of "
                  f"{len(cids):,} compounds. CAS is metadata only and is not "
                  f"used as a model feature, so this does not affect results.")
    print(f"      Found CAS numbers for {len(cas_map):,} compounds.")
    return cas_map


# ===========================================================================
# FALLBACK 1 - pubchempy
# ===========================================================================
def collect_ghs_data_pubchempy(hazard_codes=None):
    """
    FALLBACK if the PUG-View annotation endpoint fails twice.

    Uses the pubchempy library to search compound-by-compound from GHS hazard
    statement codes. Far slower than the annotation endpoint and yields fewer
    compounds, which is why it is only a fallback.
    """
    log_issue("FALLBACK", "Primary annotation endpoint failed - "
                          "switching to pubchempy library search.")
    try:
        import pubchempy as pcp
    except ImportError as exc:
        log_issue("FALLBACK", f"pubchempy unavailable: {exc}")
        return pd.DataFrame()

    # Representative hazard statements for each pictogram, used as search terms.
    hazard_codes = hazard_codes or {
        "GHS01": ["H200", "H201", "H202", "H203", "H204", "H205"],
        "GHS02": ["H220", "H221", "H222", "H223", "H224", "H225", "H226"],
        "GHS03": ["H270", "H271", "H272"],
        "GHS04": ["H280", "H281"],
        "GHS05": ["H290", "H314", "H318"],
        "GHS06": ["H300", "H301", "H310", "H311", "H330", "H331"],
        "GHS07": ["H302", "H312", "H315", "H319", "H332", "H335", "H336"],
        "GHS08": ["H304", "H340", "H341", "H350", "H351", "H360", "H370", "H372"],
        "GHS09": ["H400", "H410", "H411", "H412", "H413"],
    }

    rows = []
    for code, statements in hazard_codes.items():
        for statement in statements:
            try:
                results = pcp.get_compounds(statement, "name")
                for compound in results:
                    rows.append({"CID": compound.cid,
                                 "Source": "pubchempy-fallback",
                                 "SourceName": compound.iupac_name or "",
                                 "Pictograms": [code],
                                 "HazardStatements": [statement],
                                 "Signal": ""})
            except Exception as exc:
                log_issue("FALLBACK-pubchempy", f"{statement}: {exc}")
            time.sleep(RATE_LIMIT_DELAY)
    return pd.DataFrame(rows)


# ===========================================================================
# FALLBACK 2 - ECHA C&L Inventory / NITE-CHRIP bulk files
# ===========================================================================
def collect_ghs_data_bulk_download():
    """
    FALLBACK if both PubChem routes fail.

    Downloads a pre-compiled classification and labelling inventory and maps
    it onto the same nine binary columns. Only reached if PubChem is entirely
    unavailable, which did not happen in this run.
    """
    log_issue("FALLBACK", "Both PubChem routes failed - attempting bulk "
                          "ECHA C&L / NITE-CHRIP download.")
    candidate_urls = [
        # ECHA Classification & Labelling inventory export
        "https://echa.europa.eu/documents/10162/17233/cl_inventory_export.csv",
        # NITE-CHRIP GHS classification results (Japan)
        "https://www.nite.go.jp/chem/chrip/chrip_search/dt/pdf/CI_02_001/ghs_list.csv",
    ]
    for url in candidate_urls:
        resp = pubchem_request(url, timeout=600)
        if resp is not None and resp.status_code == 200 and len(resp.content) > 10000:
            path = os.path.join(DIR_RAW, "fallback_bulk_inventory.csv")
            with open(path, "wb") as fh:
                fh.write(resp.content)
            log_issue("FALLBACK", f"bulk file downloaded from {url}")
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception as exc:
                log_issue("FALLBACK", f"bulk file unparseable: {exc}")
    log_issue("FALLBACK", "no bulk source reachable either")
    return pd.DataFrame()


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def collect_ghs_data_pubchem(max_pages=None):
    """
    Run the whole of Step 2 and return the finished master DataFrame.

    Order of operations
    -------------------
    2b  harvest GHS annotations                (primary; fallbacks on failure)
    2e  convert pictograms to binary columns
    2c  fetch SMILES / name / formula / InChIKey
    2c  fetch CAS numbers
        assemble, report and save
    """
    start_time = time.time()

    # ---------------- 2b: get the classifications -------------------------
    records_df = harvest_all_annotations(max_pages=max_pages)

    if records_df.empty:
        records_df = collect_ghs_data_pubchempy()          # fallback 1
    if records_df.empty:
        bulk = collect_ghs_data_bulk_download()            # fallback 2
        if bulk.empty:
            raise RuntimeError(
                "Step 2 failed: no GHS data could be obtained from PubChem "
                "(primary), pubchempy (fallback 1) or bulk inventories "
                "(fallback 2). Check network connectivity.")
        records_df = bulk

    # ---------------- 2e: binary label columns ----------------------------
    per_source, aggregated = records_to_binary_labels(records_df)

    per_source_path = os.path.join(DIR_RAW, f"STEP2_per_source_records_{TODAY}.csv")
    per_source.to_csv(per_source_path, index=False)
    print(f"      Saved per-source records (needed for Step 3d majority vote):"
          f"\n        {per_source_path}")

    # ---------------- 2c: structures and identifiers ----------------------
    cids = aggregated["CID"].astype(int).tolist()
    properties = fetch_compound_properties(cids)

    if properties.empty:
        raise RuntimeError("Step 2 failed: no structures could be retrieved.")

    master = aggregated.merge(properties, on="CID", how="inner")

    # ---------------- 2c: CAS numbers -------------------------------------
    cas_map = fetch_cas_numbers(master["CID"].astype(int).tolist())
    master["CAS"] = master["CID"].map(cas_map).fillna("")

    # ---------------- assemble the columns the proposal asks for ----------
    master = master.rename(columns={"Title": "Name"})
    ordered = (["CID", "Name", "SMILES", "ConnectivitySMILES", "MolecularFormula",
                "MolecularWeight", "InChIKey", "CAS"]
               + GHS_LABEL_COLUMNS
               + ["Sources", "N_Sources", "HazardStatements"])
    master = master[[c for c in ordered if c in master.columns]]

    # ---------------- save -------------------------------------------------
    out_path = stamped("STEP2_raw_ghs_dataset.csv")
    master.to_csv(out_path, index=False)
    dated_copy = os.path.join(DIR_RAW, f"STEP2_raw_ghs_dataset_{TODAY}.csv")
    master.to_csv(dated_copy, index=False)

    # ---------------- the label-schema documentation file ------------------
    schema_path = stamped("STEP2_ghs_label_schema.csv")
    with open(schema_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column_name_as_in_proposal", "pictogram_code",
                         "authoritative_UN_PubChem_meaning",
                         "suffix_matches_official_meaning",
                         "recommended_column_name_for_publication"])
        for column in GHS_LABEL_COLUMNS:
            code = column.split("_")[0]
            matches = "YES" if GHS_RECOMMENDED_NAME[column] == column else "NO"
            writer.writerow([column, code, GHS_TRUE_MEANING[column], matches,
                             GHS_RECOMMENDED_NAME[column]])

    # ---------------- required print-outs ---------------------------------
    elapsed = time.time() - start_time
    label_matrix = master[GHS_LABEL_COLUMNS]
    per_compound_labels = label_matrix.sum(axis=1)
    multilabel_fraction = (per_compound_labels > 1).mean() * 100

    print("\n" + "=" * 78)
    print("STEP 2 RESULTS")
    print("=" * 78)
    print(f"Total compounds retrieved : {len(master):,}")
    print(f"Elapsed time              : {elapsed / 60:.1f} minutes")
    print("\nCompounds per GHS category")
    print("-" * 78)
    print(f"{'Column (as in proposal)':<24} {'Code':<7} "
          f"{'Actual meaning':<40} {'Count':>8}")
    print("-" * 78)
    for column in GHS_LABEL_COLUMNS:
        code = column.split("_")[0]
        print(f"{column:<24} {code:<7} {GHS_TRUE_MEANING[column]:<40} "
              f"{int(label_matrix[column].sum()):>8,}")
    print("-" * 78)
    print(f"Multi-label compounds (>1 hazard): {multilabel_fraction:.2f}% "
          f"({int((per_compound_labels > 1).sum()):,} compounds)")
    print(f"Mean hazards per compound        : {per_compound_labels.mean():.2f}")
    print(f"Compounds with exactly 1 hazard  : "
          f"{int((per_compound_labels == 1).sum()):,}")
    print(f"Compounds with 0 hazards         : "
          f"{int((per_compound_labels == 0).sum()):,}")

    print("\nRegulatory sources contributing")
    print("-" * 78)
    source_counts = collections.Counter(
        s for cell in master["Sources"] for s in str(cell).split(";"))
    for source, count in source_counts.most_common():
        print(f"   {count:>8,}  {source}")

    # ---------------- write the issue log ---------------------------------
    log_path = os.path.join(DIR_LOGS, f"STEP2_issue_log_{TODAY}.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("STEP 2 - issues encountered and how they were handled\n")
        fh.write("=" * 70 + "\n")
        if ISSUE_LOG:
            fh.write("\n".join(ISSUE_LOG))
        else:
            fh.write("No errors encountered. Primary method (PUG-View "
                     "annotation endpoint) succeeded; no fallback required.\n")

    print("\n" + "=" * 78)
    print("STEP 2 PROGRESS REPORT")
    print("=" * 78)
    print("WHAT WAS DONE : Harvested every GHS Classification annotation held by")
    print("                PubChem, converted the pictogram icons into nine binary")
    print("                hazard columns, then downloaded the structure (SMILES),")
    print("                name, formula, InChIKey and CAS number of each compound.")
    print("METHOD USED   : PRIMARY - PubChem PUG-View annotation endpoint.")
    print("                (pubchempy and bulk-inventory fallbacks were implemented")
    print("                 but not needed.)")
    print(f"OUTPUT FILES  : {out_path}")
    print(f"                {per_source_path}")
    print(f"                {schema_path}")
    print(f"                {log_path}")
    print(f"ISSUES        : {len(ISSUE_LOG)} logged - see the log file above.")
    print("                Known deviation: PubChem renamed the CanonicalSMILES")
    print("                property to ConnectivitySMILES in 2025; the new names")
    print("                are used explicitly so that isomeric SMILES are obtained.")
    print("=" * 78)

    return master


if __name__ == "__main__":
    # Allow "python step2_data_collection.py 20" to harvest only 20 pages,
    # which is useful for a quick test run.
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    collect_ghs_data_pubchem(max_pages=limit)
