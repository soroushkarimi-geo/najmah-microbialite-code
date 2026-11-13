"""Add the microfacies label of each specimen to the feature table.

Labels come from the sample-by-sample petrographic classification of the
Najmah Formation thin sections. Specimens not in SEED_LABELS get 'unlabeled'.

    python auto_label.py --input-csv results/features/combined_features.csv --output-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import pandas as pd

# sample code -> microfacies
SEED_LABELS = {
    # MF-A, dolostone / lime mudstone (3 specimens)
    # left out of the classifier: 3 specimens can't be evaluated with leave-one-specimen-out
    # "SD-100": "MF-A", "SD-128": "MF-A", "SD-136": "MF-A",

    # MF-B, stromatolite (dolo)boundstone = Class 1 (25 specimens)
    "SK-83": "MF-B", "SK-84": "MF-B", "SK-85": "MF-B", "SK-86": "MF-B", "SK-87": "MF-B",
    "SD-87": "MF-B", "SD-88": "MF-B", "SD-90": "MF-B", "SD-95": "MF-B", "SD-96": "MF-B",
    "SD-97": "MF-B", "SD-102": "MF-B", "SD-105": "MF-B", "SD-106": "MF-B", "SD-107": "MF-B",
    "SD-109": "MF-B", "SD-111": "MF-B", "SD-113": "MF-B", "SD-114": "MF-B", "SD-120": "MF-B",
    "SD-127": "MF-B", "SD-133": "MF-B", "SD-135": "MF-B", "SD-138": "MF-B", "SD-140": "MF-B",

    # MF-C, bioclastic calcisphere mudstone-wackestone-packstone (3)
    "SD-129": "MF-C", "SD-130": "MF-C", "SD-131": "MF-C",

    # MF-D, peloid-oncoid microbial (dolo)boundstone
    "SD-132": "MF-D",
    "SD-85": "MF-D",

    # MF-E, thrombolite (dolo)boundstone = Class 2 (24 specimens)
    "SD-89": "MF-E", "SD-91": "MF-E", "SD-92": "MF-E", "SD-94": "MF-E", "SD-99": "MF-E",
    "SD-101": "MF-E", "SD-102B": "MF-E", "SD-103": "MF-E", "SD-104": "MF-E", "SD-108": "MF-E",
    "SD-110": "MF-E", "SD-112": "MF-E", "SD-115": "MF-E", "SD-116": "MF-E", "SD-117": "MF-E",
    "SD-118": "MF-E", "SD-119": "MF-E", "SD-121": "MF-E", "SD-122": "MF-E", "SD-123": "MF-E",
    "SD-124": "MF-E", "SD-125": "MF-E",
    "SD-134": "MF-E",
    "SD-139": "MF-E",

    # MF-F, mollusk wackestone (1)
    "SD-98": "MF-F",

    # MF-G, radiolarian wackestone / radiolarite (4)
    "SD-141": "MF-G", "SD-142": "MF-G", "SD-144": "MF-G", "SD-145": "MF-G",
}

UNLABELED = "unlabeled"


def normalize_sample_code(code):
    """Convert a sample code to the form used in ``SEED_LABELS``.

    Image folders use an underscore (``SD_84``) and the section log a hyphen
    (``SD-84``); both map to ``SD-84``.

    Args:
        code: Sample code as found in the feature table.

    Returns:
        The code with underscores replaced by hyphens, in upper case.
    """
    return code.replace("_", "-").upper()


def label_dataset(input_csv, output_csv):
    """Add a ``label`` column to the feature table and save it.

    Each image gets the microfacies of its specimen from ``SEED_LABELS``;
    images of specimens not listed there get ``'unlabeled'``. A summary of the
    label counts is printed.

    Args:
        input_csv: Feature table with a ``sample_code`` column.
        output_csv: Path of the labeled table to write. Parent folders are
            created if needed.

    Raises:
        ValueError: If the input table has no ``sample_code`` column.
    """
    df = pd.read_csv(input_csv)
    if "sample_code" not in df.columns:
        raise ValueError(f"'sample_code' column not found in {input_csv}")

    df["label"] = df["sample_code"].apply(
        lambda c: SEED_LABELS.get(normalize_sample_code(str(c)), UNLABELED)
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    n_labeled = (df["label"] != UNLABELED).sum()
    print(f"Labeled {n_labeled} of {len(df)} rows")
    print(f"{len(df) - n_labeled} rows are '{UNLABELED}'")
    print("\nLabels:")
    print(df[df["label"] != UNLABELED]["label"].value_counts())


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    return parser.parse_args()


def main():
    """Command-line entry point: label the feature table."""
    args = parse_args()
    label_dataset(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
