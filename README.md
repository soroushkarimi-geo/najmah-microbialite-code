# Najmah microbialite fabric classification

Code for Paper 1: classification of laminated (stromatolite, MF-B, Class 1) and
clotted (thrombolite, MF-E, Class 2) microbialite fabric in thin-section images of
the Najmah Formation (Middle-Upper Jurassic, Dudan/Dawan and Kezi sections, NW Iran),
and a test of the resulting lamination index for stratigraphic order.

- 49 specimens, 1,651 PPL/XPL images (Classes 1 and 2)
- 404 features per image: 12 GLCM, 384 DINOv2-small, 8 lamination descriptors
- leave-one-specimen-out (LOSO) cross-validation, every specimen held out once

Authors: Soroush Karimi (IASBS), Dr. Bayet-Goll, Dr. Daraei, Dr. Azizi.

## Install

Tested with Python 3.14.5 on macOS (Apple silicon).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`torch`, `torchvision` and `transformers` are only needed to rebuild the features
from the images (`extract_embeddings.py`).

## Data

The data are not in this repository. They are available from the authors on
reasonable request. What you need depends on what you want to run:

| You want to | You need |
|---|---|
| Reproduce the paper's tables and figures (except the log drawing in Fig. 7) | 1. the two feature tables |
| Draw Fig. 7 | 1. and 3. the Dudan measured-log PDF |
| Rebuild the features from the images, run the video scripts or `binocular/` | 1. and 2. the thin-section image archive |

All paths below are relative to the repository root, and every command is run from
there. `data/`, `out/`, `results/`, `models/` and `Najmeh Formation/` are ignored by
git, so nothing you place in them can be committed by accident.

### 1. Feature tables

Put the two CSV files in `data/`:

```
najmah-microbialite-code/
  data/
    image_metadata_auto_label.csv          every image, labeled or not: 1,973 rows, 61 specimens
    image_metadata_reliable_classes.csv    MF-B and MF-E only: 1,651 rows, 49 specimens
```

Both tables have one row per image and the same 410 columns:

| Column(s) | Content |
|---|---|
| `image_id` | `<sample_code>_<PPL or XPL>_<NN>`, e.g. `SD_102_PPL_01` |
| `sample_code` | specimen, e.g. `SD_102`, `SD_102B`, `SK_84` (SD = Dudan, SK = Kezi) |
| `section` | `Dudan` or `Kezi` |
| `image_type` | `PPL` or `XPL` |
| `source_path` | image file relative to the folder that holds `Najmeh Formation/`, e.g. `Najmeh Formation/Dudan Section/SD_100/PPL/<file>.jpg` |
| `label` | `MF-B`, `MF-C`, `MF-D`, `MF-E`, `MF-F`, `MF-G` or `unlabeled` |
| `glcm_*` (12) | GLCM texture, e.g. `glcm_contrast_d1` |
| `dino_000` ... `dino_383` (384) | DINOv2-small embedding |
| `lam_*` (8) | lamination descriptors, e.g. `lam_gabor_anisotropy` |

Check that the tables are in place (expected output: `(1973, 410) (1651, 410) 49`):

```bash
python -c "import pandas as pd; a = pd.read_csv('data/image_metadata_auto_label.csv'); b = pd.read_csv('data/image_metadata_reliable_classes.csv'); print(a.shape, b.shape, b.sample_code.nunique())"
```

This is all that the commands in [Paper tables and figures](#paper-tables-and-figures) need.

### 2. Thin-section image archive (optional)

Place the archive at the repository root, or link it there:

```bash
ln -s "/path/to/archive/Najmeh Formation" "Najmeh Formation"
```

The folder layout and names must be exactly as follows:

```
Najmeh Formation/
  Dudan Section/
    SD_87/
      PPL/*.jpg
      XPL/*.jpg
      Bino/*.jpg           binocular images (only for binocular/; also spelled Binocullar)
    SD_102 B/              spaces are removed, so this specimen is SD_102B
    ...
  Kezi Section/
    SK_83/
      PPL/*.jpg
      XPL/*.jpg
    ...
```

- The specimen folder name is the sample code. Use an underscore (`SD_87`); the
  labels in `auto_label.py` are matched after replacing `_` by `-`.
- Spaces in folder names are removed, so `SD_102 B` and `SD_102B` give the same
  specimen and the same features. The distributed tables were built with the folder
  named `SD_102 B`, and their 24 `source_path` values for SD_102B use that spelling.
- Image files can be `.jpg`, `.jpeg`, `.png`, `.tif` or `.tiff`. Their names can be
  anything (the archive uses Persian-digit dates).
- `image_id` numbers the files of each `PPL`/`XPL` folder in sorted order (`_01`,
  `_02`, ...). Adding, removing or renaming images changes the numbering, and the
  rebuilt table will then no longer match the distributed one row for row.

With the archive at the repository root, `source_path` in the tables points at the
right files, and `--base-dir .` works for the extraction scripts below. The first
run of `extract_embeddings.py` downloads the DINOv2-small weights
(`facebook/dinov2-small`) from Hugging Face, so it needs internet access once; the
model runs on the CPU.

### 3. Dudan measured log (Fig. 7 only)

`plot_doudan_column_figure.py` crops the log and its legend from the vector PDF of
the Dudan measured section, passed with `--log-pdf /path/to/doudan_log.pdf`. The crop
coordinates in the script are for that specific drawing, so another PDF will not
line up.

### Outputs and run order

The paper commands below write everything to `out/`. A few scripts read the output
of another one, so run these first:

1. `train_xgboost_cv.py` writes `out/models/` and `out/tables/cv_per_sample_predictions.csv`,
   which `score_against_reliable_model.py`, `light_mode_and_site_check.py` and
   `plot_doudan_column_figure.py` read.
2. `compare_models.py` writes the tables for `plot_model_confusion_matrices.py` (Fig. 5).
3. `score_against_reliable_model.py` writes `out/tables/small_class_triage.csv` for Fig. 7.

### Scripts with fixed input paths

The paper scripts take their paths as options. Some scripts in `exploratory/` and
`binocular/` read fixed paths under `results/` and `models/` instead:

| Script | Reads |
|---|---|
| `exploratory/sequence_stratigraphy_synthesis.py` | `results/features/image_metadata_auto_label.csv` |
| `exploratory/build_3d_stratigraphic_model.py` | `results/tables/sequence_stratigraphy/stratigraphic_log.csv` (from the script above) |
| `exploratory/blind_validation_check.py` (defaults) | `results/features/image_metadata_auto_label.csv`, `models/reliable_classes/` (both can be set with `--input-csv`, `--models-dir`) |
| `exploratory/visualize_decision_video.py`, `visualize_feature_math_video.py` | `results/features/image_metadata_reliable_classes.csv`, `models/xgb_microfacies.json`, `models/label_encoder.pkl`, and the images |
| `binocular/feasibility_check_lamination.py` | `results/features/image_metadata_auto_label.csv` and `Najmeh Formation/` (with the `Bino` folders) |

To run them, copy the tables and the trained model to those places once:

```bash
mkdir -p results/features models
cp data/image_metadata_auto_label.csv data/image_metadata_reliable_classes.csv results/features/
cp out/models/xgb_microfacies.json out/models/label_encoder.pkl models/    # after train_xgboost_cv.py
```

### Settings

- `config.py`: XGBoost hyperparameters (`XGBOOST_PARAMS`, shared by every model in the
  project), the microfacies list and the facies belts.
- `auto_label.py`: the microfacies label of every specimen (`SEED_LABELS`).
- Random seeds are fixed in the scripts (42; 0 for `stratigraphic_power_analysis.py`).
  Changing any of these settings changes the results relative to the paper.

## Rebuild the feature table (needs the images)

With the image archive at the repository root (section 2 above):

```bash
python extract_embeddings.py --base-dir . --output-dir results/features
python extract_texture_features.py --base-dir . --features-dir results/features
python extract_lamination_features.py --base-dir . --features-csv results/features/combined_features.csv
python auto_label.py --input-csv results/features/combined_features.csv --output-csv data/image_metadata_auto_label.csv
python select_reliable_classes.py --input-csv data/image_metadata_auto_label.csv --output-csv data/image_metadata_reliable_classes.csv
```

## Paper tables and figures

All commands run from the repository root and write to `out/`.

| Paper | What | Command |
|---|---|---|
| Sect. 5.1 | Main classifier, 42/49 specimens | `python train_xgboost_cv.py --input-csv data/image_metadata_reliable_classes.csv --models-dir out/models --tables-dir out/tables` |
| Table 2 | 4 models, accuracy and macro F1 | `python compare_models.py --input-csv data/image_metadata_reliable_classes.csv --tables-dir out/tables` |
| Fig. 5 | Confusion matrices of the 4 models | `python plot_model_confusion_matrices.py --tables-dir out/tables --figures-dir out/figures` (after `compare_models.py`) |
| Table 2, 3 | Specimens correct, 95% Wilson interval | `python specimen_level_rerun.py --input-csv data/image_metadata_reliable_classes.csv --output-dir out/specimen_level` |
| Table 3 | Feature family ablation | `python feature_family_ablation.py --input-csv data/image_metadata_reliable_classes.csv --output-csv out/tables/feature_family_ablation.csv` |
| Sect. 5.2 | Fused vs DINOv2 only, McNemar / permutation | `python fusion_vs_dinov2_significance_test.py --input-csv data/image_metadata_reliable_classes.csv --output-csv out/tables/fusion_vs_dinov2_significance_test.csv` |
| Fig. 3, 4 | SHAP and F-value importance | `python interpretability_analysis.py --input-csv data/image_metadata_auto_label.csv --tables-dir out/tables --figures-dir out/figures` |
| Fig. 6 | First two principal components, all images | `python plot_pca_scatter.py --input-csv data/image_metadata_auto_label.csv --figures-dir out/figures` |
| Sect. 5.3 | Fold-wise check of the selected features | `python foldwise_interpretability_check.py --input-csv data/image_metadata_auto_label.csv --output-csv out/tables/foldwise_interpretability_check.csv` |
| Table 4 | Power of the Lomb-Scargle test | `python stratigraphic_power_analysis.py data/image_metadata_reliable_classes.csv --output-csv out/tables/power_table.csv` |
| Table 5 | Runs test, Lomb-Scargle | same command as Table 4 |
| Table 5 | Turning point, coarse and fine | `python shallowing_index_cyclicity_analysis.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity` |
| Table 5 | Mean shift at SB5, SB6 | `python shallowing_index_changepoint_test.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity` |
| Fig. 7 | Dudan measured log with class, model output and lamination index | `python plot_doudan_column_figure.py --log-pdf /path/to/doudan_log.pdf` (after `train_xgboost_cv.py` and `score_against_reliable_model.py`; the log PDF is not distributed) |
| Sect. 4.3, 5.1 | Accuracy by light mode; site-confound AUC | `python light_mode_and_site_check.py` (after `train_xgboost_cv.py`) |
| Fig. 8 | Kezi section, class and lamination index | `python plot_section_lamination_index.py --input-csv data/image_metadata_auto_label.csv --section SK --figures-dir out/figures` |
| Fig. 10 | Cumulative explained variance, first 50 components | `python plot_pca_explained_variance.py --input-csv data/image_metadata_auto_label.csv --figures-dir out/figures` |
| Sect. 5.5 | Triage of the other microfacies | `python score_against_reliable_model.py --input-csv data/image_metadata_auto_label.csv --models-dir out/models --output-csv out/tables/small_class_triage.csv` (after `train_xgboost_cv.py`) |
| - | Dudan profile on a thickness axis, trend per T-R sequence | `python sequence_stratigraphic_profile.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity --figures-dir out/figures` |

Run times on an Apple M-series laptop: about 2.5 min for `train_xgboost_cv.py`,
about 10 min for `specimen_level_rerun.py`, a few seconds for the stratigraphy scripts.

## Repository layout

- Top level: the scripts that produce Paper 1's tables and figures (table above).
- `exploratory/`: earlier and supporting analyses from the same project, kept for transparency. Run from the repository root with `PYTHONPATH=. python exploratory/<script>.py`.
- `binocular/`: feasibility check of binocular images for the lamination features.

## Reproduced results (macOS, 2026-10-20)

| | Paper | This code |
|---|---|---|
| Specimens correct, XGBoost fused | 42/49 | 42/49 |
| Misclassified specimens | SD-88, SD-89, SD-96, SD-99, SD-102, SD-109, SK-85 | same |
| Image accuracy, XGBoost fused | 0.779 | 0.785 |
| Image accuracy, L1 logistic regression / random forest / kNN | 0.783 / 0.765 / 0.711 | 0.783 / 0.764 / 0.711 |
| Fig. 5, gradient-boosted trees panel (`compare_models.py`) | 77.9% | 78.6% |
| Specimens correct, LR / RF / kNN | 42 / 40 / 39 | 42 / 40 / 39 |
| Ablation, specimens correct GLCM / lamination / DINOv2 / fused | 22 / 38 / 42 / 42 | 22 / 38 / 42 / 42 |
| Tables 4 and 5 | | identical |
| Features in the top 30 of both SHAP and F-value | 10 | 9 |
| Fold-wise validation, Gabor anisotropy / 45 deg energy | 42/49 / 0/49 | 42/49 / 0/49 |

The XGBoost image accuracies are 0.2-0.8 points higher than in the paper, and the
SHAP / F-value overlap is 9 instead of 10; the paper was run on Windows with an older
XGBoost. The other models and every specimen-level count agree.

## Tests

```bash
pytest
```

Synthetic-data tests in `tests/`: LOSO never puts a specimen in both the training and
the test fold, every image is predicted once, the runs test gives 22.8 expected runs
for 20 and 24 specimens, turning points, the mean-shift test, the thickness
calibration and the SD_102 / SD_102B order.

## Notes

- `SD_102` and `SD_102B` have the same sample number. The stratigraphy scripts sort them
  by code too, so SD_102 (lower) always comes first. Without this the fine
  turning-point count can change between platforms (31 vs 29).
- Sample number is used as stratigraphic position (field numbering follows the measured
  section). Heights in metres come from the 39-point thickness-sample table of the
  Dudan section.

## License

MIT, see `LICENSE`.
