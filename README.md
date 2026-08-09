# Interpretable Machine Learning for Predicting GHS Chemical Hazard Classifications

A multi-label classification framework that predicts all nine GHS hazard
pictograms from molecular structure alone, and explains every prediction in
chemical terms.

**Researcher:** Sareer Ahmad, MSc Physical Chemistry, University of Peshawar
**Target institution:** Universiti Sains Malaysia (USM)
**Proposed supervisor:** Assoc. Prof. Dr. Lee Hooi Ling
**Target journals:** *Journal of Cheminformatics* (recommended),
*Journal of Chemical Information and Modeling*, *Journal of Hazardous Materials*
**Repository:** <https://github.com/sareer555/ghs-hazard-classification>

> **Note on what is and is not in this repository.** The code, models needed to
> run the application, figures, tables and manuscript are all here (~27 MB).
> The large intermediate arrays and datasets — about 8 GB — are excluded by
> `.gitignore` and are archived separately on Zenodo. Re-running the pipeline
> regenerates every one of them from PubChem. See *Reproducing from scratch*
> below.

---

## Quick start

The environment is already built at `.venv`. To use the trained model:

```bash
.venv\Scripts\python.exe predict_ghs.py --name acrylonitrile
```

To launch the web interface, then open <http://localhost:8501>:

```bash
.venv\Scripts\streamlit.exe run app.py
```

> **If you rebuild the environment from scratch, read the Streamlit note under
> "Documented deviations" first.** Two issues stop the web app from starting on
> a fresh install, and both are already fixed in this copy.

To re-run the whole analysis:

```bash
.venv\Scripts\python.exe run_pipeline.py
```

---

## What this project does

GHS hazard classification governs how chemicals are labelled, stored and
handled worldwide, but the experimental testing it depends on has been done
for only a small fraction of chemicals in industrial use. In March 2019
improperly identified chemical waste discharged into the Sungai Kim Kim river
at Pasir Gudang, Johor, affected more than 2,500 people, most of them
schoolchildren.

This project asks whether the nine GHS pictograms can be predicted from
structure alone, and whether those predictions can be made transparent enough
for a regulator to act on.

---

## The nine hazard classes

Column names follow the official United Nations GHS pictogram numbering (10th
revised edition), which is also what PubChem, ECHA and Malaysia's DOSH use.

| Column name | Code | Pictogram | Meaning |
|---|---|---|---|
| `GHS01_Explosive` | GHS01 | Exploding bomb | Explosive |
| `GHS02_Flammable` | GHS02 | Flame | Flammable |
| `GHS03_Oxidising` | GHS03 | Flame over circle | Oxidiser |
| `GHS04_CompressedGas` | GHS04 | Gas cylinder | Compressed gas |
| `GHS05_Corrosive` | GHS05 | Corrosion | Corrosive |
| `GHS06_AcuteToxicity` | GHS06 | Skull and crossbones | Acute toxicity |
| `GHS07_Irritant` | GHS07 | Exclamation mark | Irritant / harmful |
| `GHS08_HealthHazard` | GHS08 | Health hazard | Serious health hazard |
| `GHS09_Environmental` | GHS09 | Environment | Environmental hazard |

### Correction applied to the original study design

The research proposal named three of these columns `GHS07_HealthHazard`,
`GHS08_Environmental` and `GHS09_Irritant`, which rotates the descriptive
suffixes relative to the UN scheme. **The data were always bound to the
numeric pictogram code and were therefore always correct — only the three
labels were wrong.** They have been renamed:

| Original proposal name | Corrected name |
|---|---|
| `GHS07_HealthHazard` | `GHS07_Irritant` |
| `GHS08_Environmental` | `GHS08_HealthHazard` |
| `GHS09_Irritant` | `GHS09_Environmental` |

Because the label matrix is a plain array whose column *order* never changed,
no model was retrained and no prediction changed. The rename was applied by
`src/migrate_column_names.py` and the mapping is preserved in
`STEP2_ghs_label_schema.csv` so that anyone holding an earlier copy of the
outputs can translate between the two.

---

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/step1_environment.py` | Environment setup and verification |
| 2 | `src/step2_data_collection.py` | Harvest GHS annotations from PubChem |
| 3 | `src/step3_data_cleaning.py` | Validate, deduplicate, reconcile labels |
| 4 | `src/step4_descriptors.py` | Compute 1218 molecular descriptors |
| 5 | `src/step5_scaffold_split.py` | Bemis-Murcko scaffold split (80/10/10) |
| 6 | `src/step6_imbalance.py` | SMOTE, class weights, metric selection |
| 7 | `src/step7_model_training.py` | Random Forest, XGBoost, SVM |
| 8 | `src/step8_hyperparameter_tuning.py` | RandomizedSearchCV + refit on train |
| 9 | `src/step9_evaluation.py` | Metrics, thresholds, bootstrap CIs, plots |
| 10 | `src/step10_shap_analysis.py` | SHAP global and per-compound explanation |
| 11 | `src/step11_malaysia_validation.py` | Malaysian sectors + Johor 2019 |
| 12 | `app.py`, `predict_ghs.py` | Web app and command-line tool |
| 13 | `src/step13_publication.py` | Figures, tables, abstract, methods |

### Supporting analyses

| Script | What it does |
|---|---|
| `run_full_pipeline.py` | Runs Steps 4–13 on the full 243,323-compound dataset |
| `src/controlled_size_experiment.py` | Does more training data help? Same test set, same hyperparameters, only size varies |
| `src/learning_curve.py` | Learning curve within a fixed dataset |
| `src/step4_full_from_colab.py` | Builds the full feature matrix from cached Colab descriptors |
| `src/migrate_column_names.py` | One-off rename of the three mislabelled GHS columns |
| `src/verify_interface.py` | 25 end-to-end checks on the app and CLI |
| `src/verify_deliverables.py` | Audits that every required output exists |
| `GHS_full_dataset_colab.ipynb` | Google Colab notebook for a full-dataset run |

---

## Descriptors

1218 features per compound, reduced by a variance filter:

- **19 physicochemical** — molecular weight, LogP, TPSA, H-bond donors and
  acceptors, ring counts, sp3 fraction, Labute ASA, Balaban J, Bertz CT
- **1024 Morgan / ECFP4** — circular substructure fingerprint, radius 2
- **167 MACCS keys** — predefined substructure patterns
- **8 topological** — Chi connectivity and Kappa shape indices

---

## Why the scaffold split matters

Chemical datasets are full of near-duplicate molecules. A random split puts one
member of a family in training and its twin in test, so the model appears far
more accurate than it is. Bemis-Murcko scaffold splitting forces every compound
sharing a chemical skeleton into the same split, so the reported scores reflect
generalisation to genuinely new chemotypes.

Acyclic compounds have an empty Murcko scaffold. Rather than pooling them all
into one enormous group — which would put most industrial solvents into a
single split — each is treated as its own group.

### How groups are allocated, and two ways of getting it wrong

Allocating scaffold groups to splits is harder than it looks, and two simpler
schemes were tried and rejected. Both are documented in
`src/step5_scaffold_split.py` because both fail silently.

**Fill training first, then validation, then test.** A single large scaffold
group met after the training quota is full cannot fit in the 10% validation
quota either, so it falls through to test. This produced an **80/3/17** split
instead of 80/10/10.

**Give each group to whichever split is furthest below its quota.** The overall
ratios come out exactly right, so this looks correct — but it starves the rare
classes. Groups are processed largest first, so ring-bearing scaffolds fill the
training quota early; after that the three splits have equal remaining capacity
and single-compound groups are shared out roughly a third each. *Every* acyclic
molecule is a single-compound group, and the rare classes are overwhelmingly
small acyclic molecules — compressed gases are methane, nitrogen, hydrogen
sulfide. The result:

| Class | Training share (should be 80%) |
|---|---|
| GHS04 Compressed gas | **30%** |
| GHS03 Oxidiser | **39%** |
| GHS02 Flammable | 54% |
| GHS07 Irritant | 81% ✓ |

The splitting algorithm was depriving the model of the examples it had fewest
of, while every visible check passed.

**What is used now:** a group-wise form of iterative stratification for
multi-label data. Each group is scored against every split on the worst
fractional overshoot the assignment would cause — measured on overall size
*and* on each hazard class the group contains — and goes to the split with the
lowest worst case. Groups holding rare-class compounds are placed first. Every
class now lands within one percentage point of 80%, with exact 80/10/10 split
sizes.

Step 5 now also *checks* for this: any class whose training share falls more
than 15 points below the overall share is reported as starved. That check is a
**warning, not a failure** — an earlier version treated it as fatal, which sent
the run to a non-scaffold fallback split and silently destroyed the leakage
guarantee. Trading the validity of the whole evaluation for a better class
balance is never the right trade.

---

## Documented deviations from the proposal

Every one of these is recorded in the step logs under `logs/` and reproduced in
`FINAL_PROJECT_SUMMARY_REPORT.pdf`.

1. **Python installation (Step 1).** No Python existed on the machine. The
   official python.org installer failed twice (WiX exit `0x3`, no admin
   elevation for the chained MSIs) and conda was unavailable, so the proposal's
   conda fallback could not be used. A standalone CPython 3.11.15 was installed
   with `uv`, which needs no Windows installer.

2. **PubChem API change (Step 2).** PubChem renamed `CanonicalSMILES` to
   `ConnectivitySMILES` in 2025. The new property names are requested
   explicitly so that isomeric SMILES are obtained.

3. **CAS coverage (Step 2).** CAS numbers come from synonym lists, which are
   slow to download. A 30-minute budget was applied, yielding CAS for 65% of
   compounds. CAS is metadata only and is never a model feature.

3b. **Structure-download losses (Step 2).** Seven batched property requests
   failed and were not retried compound-by-compound, costing 2,100 of the
   245,807 annotated compounds (0.85%). The loss is random with respect to
   chemistry and does not bias any class.

4. **Pictogram numbering (Step 2).** Three column names in the original study
   design did not match the UN pictogram scheme. The data were bound to the
   numeric code and were unaffected; the labels have since been corrected.
   See the table above.

5. **Modelling subset (Step 3f) — RESOLVED, no longer applies.** Earlier
   versions modelled a 40,000-compound subset because of the 7.9 GB memory
   limit. A controlled experiment (`src/controlled_size_experiment.py`) showed
   this was a real limitation, not a harmless one: mean AUC rises from 0.8187
   to 0.8738 (**+0.0551**, four times the confidence interval) on going to the
   full 194,658-compound training set, and every class improves. **All results
   now use the complete 243,323-compound dataset.**

   Worth knowing: a learning curve computed *within* the subset appeared to
   plateau and suggested the opposite. That was an artefact — the subset had
   deliberately retained every rare-class positive, so those classes could not
   improve and the aggregate curve flattened prematurely. A learning curve on a
   non-representative subsample can actively mislead.

6. **Imputation before variance filtering (Step 4).** The proposal lists 4e
   before 4f. A column containing a NaN has undefined variance, so imputing
   first is required for the filter to work as intended.

7. **Random Forest depth (Step 7).** `max_depth=None` for 200 trees × 9 classes
   would need several GB more than the machine has. Depth is capped and a
   minimum leaf size imposed.

8. **SMOTE versus class weights (Step 7).** The proposal specifies both
   SMOTE-balanced input *and* `class_weight='balanced'`, which corrects the
   imbalance twice. Class weighting is used for the primary models and SMOTE is
   run as a separate ablation, so the effect is measured rather than assumed.

9. **SVM scale (Step 7).** An RBF kernel costs O(n²). The SVM uses the top 100
   features (the proposal's own fallback) *and* a subsampled training set. Its
   scores are therefore not strictly comparable with the other two algorithms.

10. **Tuning budget (Step 8).** `n_iter=30` with 5-fold CV was measured at well
    over a day. A timing probe sets the largest `n_iter` that fits a fixed
    wall-clock budget, with 3-fold CV.

11. **Refit after tuning (Step 8).** `RandomizedSearchCV.best_estimator_` is
    fitted on whatever data the search saw — the validation set. Using it as
    the final model would waste 90% of the training data and leak validation
    information into the reported scores. The winning *settings* are taken from
    the search and a fresh model is fitted on the training set.

12. **SHAP sample (Step 10).** SHAP values are computed on a 500-compound
    random sample of the test set, which is the proposal's own documented
    fallback.

13. **Streamlit will not start on a fresh install (Step 12).** Two separate
    faults, both fixed here, both of which will recur on any clean rebuild:

    **(a) First-run email prompt.** Streamlit asks for an email address on the
    terminal before it starts its web server. Launched in the background the
    process looks alive but never opens a port, so the app appears to hang.
    Fixed by `~/.streamlit/credentials.toml` containing an empty email, and by
    `headless = true` in `.streamlit/config.toml`.

    **(b) Streamlit/Starlette incompatibility — `starlette` is pinned.**
    Streamlit 1.61.0 declares `starlette<2,>=0.46.0`, but that range is too
    permissive. Starlette 1.4.0 added a required keyword-only argument
    `thread_minimum_size` to `GZipResponder.__init__`, which Streamlit's own
    gzip middleware subclasses without passing. The server starts and reports
    itself healthy, then returns **HTTP 500 for every request**:

    ```
    TypeError: GZipResponder.__init__() missing 1 required
    keyword-only argument: 'thread_minimum_size'
    ```

    An unconstrained `pip install streamlit` picks the newest Starlette and
    reproduces the fault. The constraint `starlette<1.0` is recorded in
    `STEP1_environment_requirements.txt`, and Step 1's smoke tests now check
    the `GZipResponder` signature so a broken combination is caught at setup
    rather than at first page load:

    ```bash
    .venv\Scripts\python.exe src\step1_environment.py
    ```

    `.streamlit/config.toml` also sets `gatherUsageStats = false` — this is a
    chemical safety tool, and no information about the chemicals a user
    submits should leave the machine.

---

## Results

**Dataset:** 243,323 unique compounds after cleaning; 40,000-compound
label-stratified modelling subset; 40,000 × 817 feature matrix; scaffold split
32,000 / 4,000 / 4,000 with zero scaffold leakage.

**Best model: XGBoost**, mean AUC-ROC **0.890** across the nine classes.

| Class | Meaning | AUC-ROC | MCC |
|---|---|---|---|
| `GHS04_CompressedGas` | Compressed gas | 0.998 | 0.842 |
| `GHS03_Oxidising` | Oxidiser | 0.997 | 0.828 |
| `GHS01_Explosive` | Explosive | 0.973 | 0.534 |
| `GHS02_Flammable` | Flammable | 0.956 | 0.698 |
| `GHS05_Corrosive` | Corrosive | 0.868 | 0.521 |
| `GHS08_HealthHazard` | Serious health hazard | 0.838 | 0.381 |
| `GHS09_Environmental` | Environmental hazard | 0.831 | 0.347 |
| `GHS06_AcuteToxicity` | Acute toxicity | 0.783 | 0.272 |
| `GHS07_Irritant` | Irritant / harmful | 0.767 | 0.228 |

Physically determined hazards (gas, oxidiser, explosive, flammable) are
predicted almost perfectly from structure. Biologically mediated ones (acute
toxicity, irritation) are hardest, because they depend on mechanism rather
than bulk molecular properties.

**Model selection.** Three of the four models were statistically
indistinguishable on AUC (bootstrap CI half-width 0.0139). RandomForest+SMOTE
had the highest mean AUC by 0.0002, which is meaningless; XGBoost was selected
on mean MCC (0.517 vs 0.500), the metric designated for the rare classes.

**SMOTE versus class weighting** — a question the proposal left open. SMOTE
clearly beat plain class weighting for the Random Forest (MCC 0.500 vs 0.432),
but neither beat gradient boosting.

**Interpretability.** The strongest single finding is that MACCS key 124,
`[!#6;!#1]~[!#6;!#1]` — two directly bonded heteroatoms — is the top predictor
for *both* explosives (r = +0.96) and oxidisers (r = +0.97). That is the
structural signature of nitro, nitrate, peroxide and azide groups, recovered
by the model without being told.

**Malaysian validation** (43 compounds): hazard recall 0.900 palm oil, 0.873
Johor 2019 incident, 0.871 petrochemicals, 0.667 rubber and semiconductor. On
the Johor chemicals the framework recovered 10/10 flammable, 5/5 acute
toxicity and 10/10 serious-health-hazard labels, but **0/3 corrosive** — it
missed hydrogen chloride. Inorganic acids sit outside the applicability domain
of these descriptors, and the weakest sectors (rubber, semiconductor) are the
most inorganic-heavy. This is a real boundary of the method, not a tuning
problem.

## Reproducibility

Random seed 42 is applied to Python's `random`, NumPy, scikit-learn, XGBoost
and every resampling procedure. The exact environment is pinned in
`STEP1_environment_requirements.txt`. Steps 3 to 10 and 13 are deterministic:
re-running them on the saved data reproduces every number exactly.

**One exception.** Step 11 queries PubChem live for the structures and
reference classifications of the Malaysian and Johor chemicals, so its results
depend on PubChem's content and availability at the moment it runs. Between
two runs the mean accuracy moved from 0.831 to 0.835 for this reason. To make
Step 11 exactly reproducible, work from the saved snapshot
`malaysia_validation/STEP11_malaysian_chemicals_raw_*.csv` instead of
re-querying.

---

## Disclaimer

These models are computational screening tools. They do not replace laboratory
testing or regulatory assessment under Malaysia's Occupational Safety and
Health (Classification, Labelling and Safety Data Sheet of Hazardous Chemicals)
Regulations 2013.
