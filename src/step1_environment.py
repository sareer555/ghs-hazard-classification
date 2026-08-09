"""
STEP 1 - ENVIRONMENT SETUP AND VERIFICATION
===========================================
Project : Interpretable Machine Learning for Predicting GHS Chemical Hazard
          Classifications: A Multi-Label Classification Approach Using
          PubChem Molecular Descriptors
Author  : Sareer Ahmad (MSc Physical Chemistry, University of Peshawar)

Purpose of this script
----------------------
Before any chemistry or machine-learning work can start we must prove that
every software library the project depends on is actually installed and can
be imported. This script imports each library one at a time, records its
version number, and writes a requirements file that lets anyone reproduce
this exact environment.

Nothing here touches the data - this is purely a health check.
"""

import sys              # gives access to the Python interpreter version
import platform         # tells us which operating system we are running on
import subprocess       # lets us call `pip freeze` to capture exact versions
import random           # Python's built-in random number generator
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# RULE 5 - REPRODUCIBILITY
# A "random seed" fixes the starting point of every random number generator so
# that re-running this project produces identical results. 42 is used
# throughout the entire project, in every library that accepts a seed.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)          # seeds Python's own random module
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)  # makes string hashing deterministic

# Project root folder - every output file is written relative to this
PROJECT_ROOT = r"D:\GHS_Project"

# ---------------------------------------------------------------------------
# PINNED DEPENDENCIES
# ---------------------------------------------------------------------------
# Packages that must NOT simply be upgraded to the newest release. Each entry
# records the exact version required and why, so that nobody rebuilding this
# environment "helpfully" removes the pin and reintroduces a fixed bug.
# ---------------------------------------------------------------------------
PINNED_DEPENDENCIES = [
    {
        "package": "starlette",
        "version": "0.52.1",
        "constraint": "starlette<1.0",
        "reason": (
            "Streamlit 1.61.0 declares 'starlette<2,>=0.46.0', but that range "
            "is too permissive. Starlette 1.4.0 added a required keyword-only "
            "argument 'thread_minimum_size' to GZipResponder.__init__, which "
            "Streamlit's own gzip middleware subclasses without passing. With "
            "Starlette >= 1.0 installed, every HTTP response from the web "
            "application fails with HTTP 500 and the page never loads. "
            "Installing with an unconstrained resolver picks the newest "
            "Starlette and reproduces the fault, so the pin is required."),
        "symptom": ("TypeError: GZipResponder.__init__() missing 1 required "
                    "keyword-only argument: 'thread_minimum_size'"),
        "verified_working": "streamlit 1.61.0 + starlette 0.52.1",
    },
]

# Timestamp used in output filenames (Rule 3 naming convention)
TODAY = datetime.now().strftime("%Y%m%d")


def check_library(import_name, friendly_name=None, version_attr="__version__"):
    """
    Try to import one library and return its version number.

    Parameters
    ----------
    import_name : str
        The name used in an `import` statement, e.g. "sklearn".
    friendly_name : str, optional
        The name used when installing with pip, e.g. "scikit-learn".
        Only differs from import_name for a few packages.
    version_attr : str
        The attribute holding the version string. Almost every library
        uses "__version__", but a few use something else.

    Returns
    -------
    (status, version) : tuple of str
        status is "OK" or "FAILED"; version is the version string or the
        error message explaining why the import did not work.
    """
    if friendly_name is None:
        friendly_name = import_name
    try:
        # __import__ imports a module whose name is only known at run time
        module = __import__(import_name)
        # getattr reads the version attribute; "unknown" if the library
        # does not expose one (rare, but harmless)
        version = getattr(module, version_attr, "version-attribute-not-exposed")
        return "OK", str(version)
    except Exception as exc:  # catch *any* import problem, not just ImportError
        return "FAILED", f"{type(exc).__name__}: {exc}"


def main():
    """Run every environment check and write the requirements file."""

    print("=" * 78)
    print("STEP 1 - ENVIRONMENT SETUP AND VERIFICATION")
    print("=" * 78)
    print(f"Run date            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version      : {sys.version.split()[0]}")
    print(f"Python executable   : {sys.executable}")
    print(f"Operating system    : {platform.platform()}")
    print(f"CPU architecture    : {platform.machine()}")
    print(f"CPU cores available : {os.cpu_count()}")
    print(f"Global random seed  : {RANDOM_SEED}")
    print("-" * 78)

    # Each tuple is (import name, pip name). The pip name is what a user would
    # type in `pip install ...`; it sometimes differs from the import name.
    required_libraries = [
        ("requests",   "requests"),
        ("pandas",     "pandas"),
        ("numpy",      "numpy"),
        ("rdkit",      "rdkit"),
        ("sklearn",    "scikit-learn"),
        ("xgboost",    "xgboost"),
        ("imblearn",   "imbalanced-learn"),
        ("shap",       "shap"),
        ("matplotlib", "matplotlib"),
        ("seaborn",    "seaborn"),
        ("scipy",      "scipy"),
        ("joblib",     "joblib"),
        ("streamlit",  "streamlit"),
        ("reportlab",  "reportlab"),
        ("tqdm",       "tqdm"),
        ("pubchempy",  "pubchempy"),
    ]

    # Fallback libraries named in the research proposal. These are not
    # strictly required, but having them installed means that if a primary
    # method fails we can switch immediately without stopping to install.
    fallback_libraries = [
        ("lightgbm", "lightgbm"),        # fallback if XGBoost fails (Step 7)
        ("iterstrat", "iterative-stratification"),  # fallback split (Step 5)
        ("openpyxl", "openpyxl"),        # needed to write .xlsx tables (Step 13)
        ("starlette", "starlette"),      # pinned - see PINNED_DEPENDENCIES
    ]

    results = []   # collects (pip_name, status, version) for the summary table

    print("\nREQUIRED LIBRARIES")
    print("-" * 78)
    print(f"{'Library':<24} {'Status':<8} {'Version'}")
    print("-" * 78)
    for import_name, pip_name in required_libraries:
        status, version = check_library(import_name)
        results.append((pip_name, status, version))
        print(f"{pip_name:<24} {status:<8} {version}")

    print("\nFALLBACK / SUPPORT LIBRARIES")
    print("-" * 78)
    print(f"{'Library':<24} {'Status':<8} {'Version'}")
    print("-" * 78)
    for import_name, pip_name in fallback_libraries:
        status, version = check_library(import_name)
        results.append((pip_name, status, version))
        print(f"{pip_name:<24} {status:<8} {version}")

    # -----------------------------------------------------------------------
    # FUNCTIONAL SMOKE TESTS
    # Importing a library is not the same as it working correctly. RDKit in
    # particular has compiled C++ parts that can import but then fail. We
    # therefore run one tiny real calculation per critical library.
    # -----------------------------------------------------------------------
    print("\nFUNCTIONAL SMOKE TESTS (import alone is not proof of a working install)")
    print("-" * 78)
    smoke_results = []

    # Test 1: RDKit can parse a SMILES string and compute a property.
    # Aspirin is used because its structure and molecular weight are well known.
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, MACCSkeys
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")   # silence RDKit's very chatty warnings
        aspirin = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        mw = Descriptors.MolWt(aspirin)                 # ~180.16 g/mol
        inchikey = Chem.MolToInchiKey(aspirin)          # unique structure hash
        fp = AllChem.GetMorganFingerprintAsBitVect(aspirin, radius=2, nBits=1024)
        maccs = MACCSkeys.GenMACCSKeys(aspirin)
        scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(aspirin))
        msg = (f"MolWt={mw:.2f}, InChIKey={inchikey}, "
               f"MorganBits={fp.GetNumBits()}, MACCSBits={maccs.GetNumBits()}, "
               f"Scaffold={scaf}")
        smoke_results.append(("RDKit parse+descriptors+FP+scaffold", "OK", msg))
    except Exception as exc:
        smoke_results.append(("RDKit parse+descriptors+FP+scaffold", "FAILED", str(exc)))

    # Test 2: scikit-learn can actually fit a model.
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        rng = np.random.RandomState(RANDOM_SEED)   # seeded for reproducibility
        X_toy = rng.rand(40, 6)                    # 40 fake compounds, 6 features
        y_toy = (rng.rand(40, 3) > 0.5).astype(int)  # 3 fake hazard labels
        clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED))
        clf.fit(X_toy, y_toy)
        smoke_results.append(("scikit-learn MultiOutputClassifier fit", "OK",
                              f"predicted shape {clf.predict(X_toy).shape}"))
    except Exception as exc:
        smoke_results.append(("scikit-learn MultiOutputClassifier fit", "FAILED", str(exc)))

    # Test 3: XGBoost can fit a binary classifier.
    try:
        import numpy as np
        from xgboost import XGBClassifier
        rng = np.random.RandomState(RANDOM_SEED)
        X_toy = rng.rand(40, 6)
        y_toy = (rng.rand(40) > 0.5).astype(int)
        xgb = XGBClassifier(n_estimators=10, random_state=RANDOM_SEED,
                            eval_metric="logloss")
        xgb.fit(X_toy, y_toy)
        smoke_results.append(("XGBoost fit", "OK", "binary classifier trained"))
    except Exception as exc:
        smoke_results.append(("XGBoost fit", "FAILED", str(exc)))

    # Test 4: SMOTE (imbalanced-learn) can oversample a minority class.
    try:
        import numpy as np
        from imblearn.over_sampling import SMOTE
        rng = np.random.RandomState(RANDOM_SEED)
        X_toy = rng.rand(60, 6)
        # deliberately imbalanced: only 10 positives out of 60
        y_toy = np.array([1] * 10 + [0] * 50)
        X_res, y_res = SMOTE(random_state=RANDOM_SEED, k_neighbors=5
                             ).fit_resample(X_toy, y_toy)
        smoke_results.append(("imbalanced-learn SMOTE", "OK",
                              f"{len(y_toy)} -> {len(y_res)} samples after oversampling"))
    except Exception as exc:
        smoke_results.append(("imbalanced-learn SMOTE", "FAILED", str(exc)))

    # Test 5: SHAP TreeExplainer works on a tree model.
    try:
        import numpy as np
        import shap
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.RandomState(RANDOM_SEED)
        X_toy = rng.rand(40, 6)
        y_toy = (rng.rand(40) > 0.5).astype(int)
        rf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
        rf.fit(X_toy, y_toy)
        sv = shap.TreeExplainer(rf).shap_values(X_toy[:5])
        smoke_results.append(("SHAP TreeExplainer", "OK",
                              f"shap values array shape {np.array(sv).shape}"))
    except Exception as exc:
        smoke_results.append(("SHAP TreeExplainer", "FAILED", str(exc)))

    # Test 6: matplotlib can render a figure without a screen.
    # "Agg" is a non-interactive backend that writes straight to a PNG file.
    # This is essential because this project runs without a graphical desktop.
    try:
        import matplotlib
        matplotlib.use("Agg")          # must be set before importing pyplot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        test_png = os.path.join(PROJECT_ROOT, "logs", "_matplotlib_smoketest.png")
        fig.savefig(test_png, dpi=100)
        plt.close(fig)
        ok = os.path.exists(test_png)
        smoke_results.append(("matplotlib Agg PNG render", "OK" if ok else "FAILED",
                              f"wrote {test_png}"))
    except Exception as exc:
        smoke_results.append(("matplotlib Agg PNG render", "FAILED", str(exc)))

    # Test 7: the Streamlit web stack can actually serve a response.
    # Importing streamlit is not enough. The failure this guards against sits
    # in the HTTP middleware, so it only appears when a page is requested -
    # the server starts, reports itself healthy, and then returns HTTP 500 for
    # every request. Checking the constructor signature catches it here, at
    # setup time, instead of leaving the researcher with a blank browser tab.
    try:
        import inspect
        import streamlit                                     # noqa: F401
        from starlette.middleware.gzip import GZipResponder
        parameters = inspect.signature(GZipResponder.__init__).parameters
        if "thread_minimum_size" in parameters:
            import starlette
            smoke_results.append((
                "Streamlit web stack compatibility", "FAILED",
                f"starlette {starlette.__version__} is incompatible with "
                f"streamlit {streamlit.__version__}; run "
                f"'uv pip install \"starlette<1.0\"'"))
        else:
            import starlette
            smoke_results.append((
                "Streamlit web stack compatibility", "OK",
                f"streamlit {streamlit.__version__} + "
                f"starlette {starlette.__version__}"))
    except Exception as exc:
        smoke_results.append(("Streamlit web stack compatibility", "FAILED",
                              str(exc)))

    # Test 8: reportlab can build a PDF (needed for Steps 11 and 13).
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        test_pdf = os.path.join(PROJECT_ROOT, "logs", "_reportlab_smoketest.pdf")
        c = canvas.Canvas(test_pdf, pagesize=A4)
        c.drawString(72, 750, "GHS project reportlab smoke test")
        c.save()
        ok = os.path.exists(test_pdf)
        smoke_results.append(("reportlab PDF generation", "OK" if ok else "FAILED",
                              f"wrote {test_pdf}"))
    except Exception as exc:
        smoke_results.append(("reportlab PDF generation", "FAILED", str(exc)))

    print(f"{'Test':<40} {'Status':<8} {'Detail'}")
    print("-" * 78)
    for name, status, detail in smoke_results:
        print(f"{name:<40} {status:<8} {detail}")

    # -----------------------------------------------------------------------
    # WRITE THE REQUIREMENTS FILE
    # `pip freeze` lists every installed package with its exact version. Saving
    # this makes the environment fully reproducible by another researcher.
    # -----------------------------------------------------------------------
    req_path = os.path.join(PROJECT_ROOT, "STEP1_environment_requirements.txt")
    try:
        frozen = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=180
        ).stdout
    except Exception as exc:
        frozen = f"# pip freeze failed: {exc}\n"

    with open(req_path, "w", encoding="utf-8") as fh:
        fh.write("# " + "=" * 74 + "\n")
        fh.write("# STEP1_environment_requirements.txt\n")
        fh.write("# GHS Chemical Hazard Classification project - Sareer Ahmad\n")
        fh.write(f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# Python   : {sys.version.split()[0]} ({sys.executable})\n")
        fh.write(f"# Platform : {platform.platform()}\n")
        fh.write(f"# Random seed used project-wide: {RANDOM_SEED}\n")
        fh.write("#\n")
        fh.write("# INSTALLATION NOTE (documented deviation, see Rule 2):\n")
        fh.write("#   The official python.org Windows installer (python-3.11.9-amd64.exe)\n")
        fh.write("#   failed twice with WiX bootstrapper exit code 0x3 on this machine\n")
        fh.write("#   (no administrator elevation available for the chained MSI packages).\n")
        fh.write("#   FALLBACK APPLIED: the 'uv' package manager (Astral) was used to\n")
        fh.write("#   install a standalone CPython 3.11.15 build, which requires no\n")
        fh.write("#   Windows installer at all. All packages were then installed with\n")
        fh.write("#   'uv pip install', which is a drop-in replacement for pip.\n")
        fh.write("#   conda was not available on this machine, so the conda fallback\n")
        fh.write("#   named in the proposal could not be used.\n")
        fh.write("# " + "=" * 74 + "\n\n")

        # ---- pinned dependencies, with the reason for each pin -------------
        fh.write("# " + "=" * 74 + "\n")
        fh.write("# PINNED DEPENDENCIES - DO NOT UPGRADE WITHOUT READING THIS\n")
        fh.write("# " + "=" * 74 + "\n")
        for pin in PINNED_DEPENDENCIES:
            fh.write(f"#\n# {pin['package']}=={pin['version']}\n")
            fh.write(f"#   WHY: ")
            # Wrap the explanation to keep the file readable.
            words, line = pin["reason"].split(), ""
            for word in words:
                if len(line) + len(word) + 1 > 68:
                    fh.write(line + "\n#        ")
                    line = word
                else:
                    line = f"{line} {word}".strip()
            fh.write(line + "\n")
            fh.write(f"#   SYMPTOM IF UNPINNED: {pin['symptom']}\n")
            fh.write(f"#   VERIFIED WORKING: {pin['verified_working']}\n")
        fh.write("#\n")
        for pin in PINNED_DEPENDENCIES:
            fh.write(f"{pin['constraint']}\n")
        fh.write("# " + "=" * 74 + "\n\n")

        fh.write("# --- Directly required by the research proposal ---\n")
        for pip_name, status, version in results:
            if status == "OK":
                fh.write(f"{pip_name}=={version}\n")
            else:
                fh.write(f"# {pip_name}  -> IMPORT FAILED: {version}\n")
        fh.write("\n\n# --- Complete frozen environment (pip freeze) ---\n")
        fh.write(frozen)

    print("\n" + "=" * 78)
    print("STEP 1 PROGRESS REPORT")
    print("=" * 78)
    n_required_ok = sum(1 for _, s, _ in results[:len(required_libraries)] if s == "OK")
    n_fallback_ok = sum(1 for _, s, _ in results[len(required_libraries):] if s == "OK")
    n_smoke_ok = sum(1 for _, s, _ in smoke_results if s == "OK")
    print(f"WHAT WAS DONE   : Installed and verified the full Python environment.")
    print(f"REQUIRED LIBS   : {n_required_ok}/{len(required_libraries)} imported successfully")
    print(f"FALLBACK LIBS   : {n_fallback_ok}/{len(fallback_libraries)} imported successfully")
    print(f"SMOKE TESTS     : {n_smoke_ok}/{len(smoke_results)} passed")
    print(f"OUTPUT FILE     : {req_path}")
    failed = [r for r in results if r[1] == "FAILED"] + \
             [s for s in smoke_results if s[1] == "FAILED"]
    if failed:
        print(f"ISSUES          : {len(failed)} problem(s) - see list above")
    else:
        print("ISSUES          : Python itself was absent from this machine and the")
        print("                  official installer failed twice (exit 0x3, no admin");
        print("                  elevation). Resolved with the documented fallback:")
        print("                  standalone CPython 3.11.15 installed via 'uv'.")
        print("                  All libraries then installed and verified cleanly.")
    print("=" * 78)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
