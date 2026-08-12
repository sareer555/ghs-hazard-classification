# v1.0.2 — Applicability domain warning and figure corrections

This release corrects a factual error in the published workflow figure and adds
an applicability-domain check to the prediction tool. **Anyone citing or
re-running v1.0.1 should use this release instead.** No analysis step, model or
reported result changes: every number in the manuscript is reproduced exactly by
this release and by v1.0.1.

## Corrections

- **Figure 1 stated that SMOTE was applied. It was not.** `STEP6_smote_report.csv`
  records `none (memory budget — class weights used)` for all nine hazard
  classes: at this dataset size SMOTE was skipped and balanced class weighting
  used instead. The figure carried the claim as fixed text and so contradicted
  the supplementary data. It now reads the method from Step 6's own output and
  cannot drift again.
- **Figure 2's rotated axis labels were not anchored**, so each label was centred
  on its rotated bounding box and drifted to the right of the bar it named.
- **Figure 2 formatted the class-imbalance ratio as an integer**, printing
  GHS07's ratio of 0.106 as `0:1` — readable as "no negative examples at all".
  It now reads `0.11:1`.
- **Figure 1's title and subtitle overlapped the top row of boxes.**

## New

- **Applicability domain check.** The model is unreliable for very small
  molecules: the median training compound has 14 heavy atoms and only 1.1% have
  fewer than six. On the 260 test compounds below that bound it assigns acute
  toxicity to 84% (true rate 27%) and serious health hazard to 80% (true rate
  33%). Water is returned as corrosive, acutely toxic and a serious health
  hazard, and carries none of those pictograms.

  The predictor now flags any structure below six heavy atoms as outside the
  domain, and the warning appears before the hazard profile in the web
  application, the command-line tool and the exported PDF report. The profile is
  still produced rather than withheld — the failure mode is over-prediction, and
  suppressing a genuine hazard would be the worse error — but it is presented
  without the appearance of confidence.

  `src/applicability_domain.py` measures this and writes
  `EXTRA_applicability_domain.csv` and `.json`. The manuscript's Limitations
  section quotes those outputs rather than hard-coded figures.
- `src/build_supporting_information.py` compiles the Supporting Information into
  a single 23-page PDF. Where a table is too wide to print legibly it shows the
  columns the manuscript discusses and states how many were omitted and which
  workbook sheet holds the complete table.
- `src/redraw_figures_9_10.py` rebuilds the learning-curve and training-size
  figures from their saved result tables, without retraining.

## Changed

- **Figure 3** replaces nine overlaid ROC curves with a 3×3 grid of small
  multiples ordered by AUC. Nine distinguishable colours do not exist; two pairs
  of classes were previously indistinguishable. Each panel also reports its
  positive count, which makes visible that the two highest AUCs rest on 28 and
  18 positive examples.
- **Figure 6** replaces a red–yellow–green diverging scale with a single-hue
  sequential ramp. AUC has no meaningful midpoint, and red–green is the pairing
  colour-blind readers can least separate. Cell values now set their own text
  colour, so the weakest results are no longer white text on a pale ground.
- **Model colours** were selected by running the palette through colour-vision
  checks rather than by eye. The previous set separated the support vector
  machine and the ablation by ΔE 7.6 under deuteranopia, inside the band where
  colour alone cannot distinguish two series; the new set is 11.0 at worst and
  every colour clears 3:1 contrast on white paper.
- **Figures 9 and 10** draw the nine per-class curves in grey and colour only the
  best and worst classes, labelled where they end.
- Figure style moved to `src/ghs_config.py`. It was duplicated across three
  scripts, so a model could change colour between figures in the same paper.
- Figure resolution raised to 500 dpi, Elsevier's requirement for combination
  line/halftone artwork.
- Publication materials retargeted from the *Journal of Chemical Information and
  Modeling* to *Computational Toxicology*: numbered sections, a Highlights file,
  the generative-AI declaration positioned and worded per Elsevier's policy, and
  the deposited dataset and code cited in the reference list as that journal's
  research-data policy requires.
- Author-facing notes that had been assembled into the manuscript were moved out
  of it.

## Reproducing

```bash
git clone https://github.com/sareer555/ghs-hazard-classification
cd ghs-hazard-classification
pip install -r requirements.txt
python src/applicability_domain.py
```

The trained model is included in the repository. The curated dataset and the
larger model files are archived separately at
https://doi.org/10.5281/zenodo.21876611.
