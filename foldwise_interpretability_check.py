"""Fold-wise check of the SHAP + F-value feature selection.

interpretability_analysis.py ranks the features once on all data. Here the same
ranking is repeated with each MF-B / MF-E specimen held out in turn (49 folds),
training on all labeled images of every class minus that specimen. A lamination
feature counts as validated in a fold when it is
  (a) in the overlap of the top 30 by SHAP and the top 30 by F-value, and
  (b) in the top 10 per-class SHAP features of MF-B or MF-E.

Result: lam_gabor_anisotropy validates in 42/49 folds, lam_gabor_energy_45 in 0/49
and the other lamination features in at most 1/49, so the lamination index S uses
anisotropy only.

    python foldwise_interpretability_check.py --input-csv data/image_metadata_auto_label.csv --output-csv out/tables/foldwise_interpretability_check.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import Config
from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
TOP_N_FAMILY = 30
TOP_N_PER_CLASS = 10
# all 8 lamination features
TARGET_FEATURES = [
    "lam_coherence_mean", "lam_coherence_std",
    "lam_gabor_energy_0", "lam_gabor_energy_45", "lam_gabor_energy_90", "lam_gabor_energy_135",
    "lam_gabor_anisotropy", "lam_fft_peak_strength",
]
HELD_OUT_CLASSES = {"MF-B", "MF-E"}


def run(input_csv, output_csv):
    """Repeat the SHAP + F-value feature check with each specimen held out.

    For every MF-B or MF-E specimen, an XGBoost model (``Config`` settings,
    balanced weights) is trained on all labeled images of all classes except
    that specimen. Within the fold, SHAP and F-value rankings are computed on
    the training images. A lamination feature is "validated" in the fold if
    it is in the overlap of the top 30 by SHAP and the top 30 by F-value and
    in the top 10 per-class SHAP features of MF-B or MF-E.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        output_csv: Path of the per-fold table. For each lamination feature
            it has ``<feature>_in_top30_overlap``, ``<feature>_in_class_top10``,
            ``<feature>_validated``, ``<feature>_shap_rank`` and
            ``<feature>_fvalue_rank`` (1 = most important).
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    feature_cols = get_feature_columns(df)

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder().fit(present_classes)

    be_specimens = sorted(
        labeled_df.loc[labeled_df["label"].isin(HELD_OUT_CLASSES), "sample_code"].unique()
    )
    print(f"SHAP + F-value check over {len(be_specimens)} folds (MF-B/MF-E specimens held out)...\n")

    rows = []
    for i, held_out in enumerate(be_specimens, start=1):
        train_fold = labeled_df[labeled_df["sample_code"] != held_out]

        X = train_fold[feature_cols]
        y = label_encoder.transform(train_fold["label"])
        sample_weight = compute_sample_weight("balanced", y)

        params = dict(Config.XGBOOST_PARAMS)
        params["num_class"] = len(label_encoder.classes_)
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, sample_weight=sample_weight)

        # SHAP, global and per class
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)
        else:
            per_class = np.abs(shap_values).mean(axis=0)
            if per_class.ndim == 1:
                per_class = per_class[:, np.newaxis]
        per_class_df = pd.DataFrame(per_class, index=X.columns, columns=label_encoder.classes_)
        shap_global = per_class_df.mean(axis=1).sort_values(ascending=False)

        # F-value
        f_values, _ = f_classif(X, y)
        f_ranking = pd.Series(f_values, index=X.columns).sort_values(ascending=False)

        shap_top30 = set(shap_global.head(TOP_N_FAMILY).index)
        f_top30 = set(f_ranking.head(TOP_N_FAMILY).index)
        overlap_top30 = shap_top30 & f_top30

        row = {"fold": i, "held_out_specimen": held_out}
        for feat in TARGET_FEATURES:
            in_overlap = feat in overlap_top30
            per_class_top10 = set()
            for cls in HELD_OUT_CLASSES:
                if cls in per_class_df.columns:
                    top10 = set(per_class_df[cls].sort_values(ascending=False).head(TOP_N_PER_CLASS).index)
                    per_class_top10 |= top10
            in_per_class_top10 = feat in per_class_top10
            row[f"{feat}_in_top30_overlap"] = in_overlap
            row[f"{feat}_in_class_top10"] = in_per_class_top10
            row[f"{feat}_validated"] = in_overlap and in_per_class_top10
            row[f"{feat}_shap_rank"] = int(shap_global.index.get_loc(feat)) + 1
            row[f"{feat}_fvalue_rank"] = int(f_ranking.index.get_loc(feat)) + 1
        rows.append(row)

        validated_now = [f.replace("lam_", "") for f in TARGET_FEATURES if row[f"{f}_validated"]]
        print(f"  [{i}/{len(be_specimens)}] held out {held_out}: "
              f"validated = {validated_now if validated_now else 'none'}")

    results_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print(f"\n=== {len(results_df)} folds ===")
    summary_rows = []
    for feat in TARGET_FEATURES:
        n_validated = results_df[f"{feat}_validated"].sum()
        n_total = len(results_df)
        pct = 100 * n_validated / n_total
        summary_rows.append((feat, n_validated, pct))
        print(f"  {feat}: validated in {n_validated}/{n_total} folds ({pct:.1f}%)")
        print(f"    SHAP rank range: {results_df[f'{feat}_shap_rank'].min()}-{results_df[f'{feat}_shap_rank'].max()} "
              f"(median {results_df[f'{feat}_shap_rank'].median():.0f})")
        print(f"    F-value rank range: {results_df[f'{feat}_fvalue_rank'].min()}-{results_df[f'{feat}_fvalue_rank'].max()} "
              f"(median {results_df[f'{feat}_fvalue_rank'].median():.0f})")

    summary_rows.sort(key=lambda r: -r[1])
    print("\nBy number of folds validated:")
    for feat, n_validated, pct in summary_rows:
        print(f"  {feat}: {n_validated}/{len(results_df)} ({pct:.1f}%)")
    print(f"\nSaved to {output_csv}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results/tables/foldwise_interpretability_check.csv")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
