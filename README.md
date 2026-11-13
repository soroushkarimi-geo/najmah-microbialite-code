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

The data are not in this repository; they are available from the authors on reasonable request.
Place the two feature tables here before running the scripts (this folder is ignored by git):

```
data/
  image_metadata_auto_label.csv          all labeled + unlabeled images (1,973 rows)
  image_metadata_reliable_classes.csv    MF-B and MF-E only (1,651 rows, 49 specimens)
```

One row per image: `image_id`, `sample_code`, `section`, `image_type`, `source_path`
(relative to the image root), `label`, then the features `glcm_*`, `dino_000`...`dino_383`, `lam_*`.

To rebuild the feature table,
put them like this:

```
<image_root>/Najmeh Formation/Dudan Section/SD_87/PPL/*.jpg
<image_root>/Najmeh Formation/Dudan Section/SD_87/XPL/*.jpg
<image_root>/Najmeh Formation/Kezi Section/SK_83/PPL/*.jpg
...
```

## Rebuild the feature table (needs the images)

```bash
python extract_embeddings.py --base-dir /path/to/image_root --output-dir results/features
python extract_texture_features.py --base-dir /path/to/image_root --features-dir results/features
python extract_lamination_features.py --base-dir /path/to/image_root --features-csv results/features/combined_features.csv
python auto_label.py --input-csv results/features/combined_features.csv --output-csv data/image_metadata_auto_label.csv
python select_reliable_classes.py --input-csv data/image_metadata_auto_label.csv --output-csv data/image_metadata_reliable_classes.csv
```

Labels per specimen are in `auto_label.py` (`SEED_LABELS`), model settings in `config.py`.

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
