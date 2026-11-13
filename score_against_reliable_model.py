"""Triage: score the small classes (MF-C, MF-D, MF-F, MF-G) and unlabeled images
with the MF-B / MF-E model, one row per specimen (mean class probability of its images).

It never changes a label. A high MF-B or MF-E probability on a specimen labeled
something else marks it for petrographic review. Probabilities near 0.5 across a
small class are weak evidence that the class is distinct from both.

The model is the one train_xgboost_cv.py saves when run on
data/image_metadata_reliable_classes.csv.

    python score_against_reliable_model.py --input-csv data/image_metadata_auto_label.csv --models-dir out/models --output-csv out/tables/small_class_triage.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
RELIABLE_CLASSES = ("MF-B", "MF-E")


def score(input_csv, models_dir, output_csv):
    """Score every specimen outside MF-B / MF-E with the saved MF-B / MF-E model.

    Each image gets the model's class probabilities; these are averaged per
    specimen. Specimens are sorted by their highest probability, so the
    ones most similar to MF-B or MF-E come first. Labels are never changed.

    Args:
        input_csv: Labeled feature table (all images).
        models_dir: Folder (``pathlib.Path``) with ``xgb_microfacies.json``
            and ``label_encoder.pkl`` from ``train_xgboost_cv.py``.
        output_csv: Path of the triage table. One row per specimen with
            ``sample_code``, ``label``, ``prob_MF-B``, ``prob_MF-E``,
            ``max_reliable_prob`` and ``closest_reliable_class``.
    """
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)

    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    model = xgb.XGBClassifier()
    model.load_model(str(models_dir / "xgb_microfacies.json"))

    to_score = df[~df["label"].isin(RELIABLE_CLASSES)].copy()
    # predict_proba gives probabilities even though the objective is multi:softmax
    probs = model.predict_proba(to_score[feature_cols])

    prob_cols = [f"prob_{c}" for c in label_encoder.classes_]
    for i, c in enumerate(prob_cols):
        to_score[c] = probs[:, i]

    per_specimen = (
        to_score.groupby(["sample_code", "label"])[prob_cols]
        .mean()
        .reset_index()
    )
    per_specimen["max_reliable_prob"] = per_specimen[prob_cols].max(axis=1)
    per_specimen["closest_reliable_class"] = per_specimen[prob_cols].idxmax(axis=1).str.replace("prob_", "")
    per_specimen = per_specimen.sort_values("max_reliable_prob", ascending=False)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    per_specimen.to_csv(output_csv, index=False)

    print(f"Scored {len(per_specimen)} specimens, classes: {sorted(per_specimen['label'].unique())}")
    print("\nTop 15, most MF-B / MF-E like (check these first):")
    print(per_specimen.head(15)[["sample_code", "label", "closest_reliable_class", "max_reliable_prob"]].to_string(index=False))
    print("\nBottom 15, least like either:")
    print(per_specimen.tail(15)[["sample_code", "label", "closest_reliable_class", "max_reliable_prob"]].to_string(index=False))
    print(f"\nSaved to {output_csv}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``models_dir`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models/reliable_classes")
    parser.add_argument("--output-csv", type=str, default="results/tables/reliable_classes/small_class_triage.csv")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``score`` with the given options."""
    args = parse_args()
    score(input_csv=args.input_csv, models_dir=Path(args.models_dir), output_csv=args.output_csv)


if __name__ == "__main__":
    main()
