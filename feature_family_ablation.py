"""Feature family ablation (Table 3): the same LOSO XGBoost run on GLCM only,
DINOv2 only, lamination only and all three fused.

    python feature_family_ablation.py --input-csv data/image_metadata_reliable_classes.csv --output-csv out/tables/feature_family_ablation.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from config import Config
from train_xgboost_cv import run_loso_cv

UNLABELED = "unlabeled"

FEATURE_FAMILIES = {
    "GLCM only": lambda cols: [c for c in cols if c.startswith("glcm_")],
    "DINOv2 only": lambda cols: [c for c in cols if c.startswith("dino_")],
    "Lamination only": lambda cols: [c for c in cols if c.startswith("lam_")],
    "Fused (all)": lambda cols: [c for c in cols if c.startswith(("glcm_", "dino_", "lam_"))],
}


def run(input_csv, output_csv):
    """LOSO XGBoost on each feature family of ``FEATURE_FAMILIES`` (Table 3).

    Every family uses the same folds and hyperparameters
    (``train_xgboost_cv.run_loso_cv`` with ``Config.XGBOOST_PARAMS``), so
    only the feature set differs between rows.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        output_csv: Path of the results table. One row per family with
            ``feature_family``, ``n_features``, image-level ``accuracy``,
            ``macro_f1``, ``MF-B_f1`` and ``MF-E_f1``.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder()
    label_encoder.fit(present_classes)

    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(present_classes)

    rows = []
    for family_name, selector in FEATURE_FAMILIES.items():
        feature_cols = selector(list(df.columns))
        print(f"\n=== {family_name}: {len(feature_cols)} features ===")
        predictions_df = run_loso_cv(labeled_df, feature_cols, label_encoder, params)

        y_true = label_encoder.transform(predictions_df["label"])
        y_pred = label_encoder.transform(predictions_df["predicted_label"])
        report = classification_report(
            y_true, y_pred, labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_, output_dict=True, zero_division=0,
        )
        accuracy = report["accuracy"]
        macro_f1 = report["macro avg"]["f1-score"]
        mfb_f1 = report.get("MF-B", {}).get("f1-score", float("nan"))
        mfe_f1 = report.get("MF-E", {}).get("f1-score", float("nan"))
        print(f"  accuracy={accuracy:.3f}  macro F1={macro_f1:.3f}  MF-B F1={mfb_f1:.3f}  MF-E F1={mfe_f1:.3f}")

        rows.append({
            "feature_family": family_name,
            "n_features": len(feature_cols),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "MF-B_f1": mfb_f1,
            "MF-E_f1": mfe_f1,
        })

    results_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nSaved to {output_csv}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results/tables/feature_family_ablation.csv")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
