"""Keep only MF-B (Class 1) and MF-E (Class 2): the 49-specimen, 1651-image table
used for all classification results. Row order is kept.

    python select_reliable_classes.py --input-csv data/image_metadata_auto_label.csv --output-csv out/image_metadata_reliable_classes.csv
"""
import argparse
from pathlib import Path

import pandas as pd

RELIABLE_CLASSES = ("MF-B", "MF-E")


def select(input_csv, output_csv):
    """Write the subset of images labeled MF-B or MF-E.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        output_csv: Path of the filtered table to write. Parent folders are
            created if needed.

    Returns:
        pandas.DataFrame: The filtered table, with a fresh 0..n-1 index and
        the original row order.
    """
    df = pd.read_csv(input_csv)
    out = df[df["label"].isin(RELIABLE_CLASSES)].reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Kept {len(out)} images, {out['sample_code'].nunique()} specimens, saved to {output_csv}")
    print(out.groupby("label")["sample_code"].nunique().rename("specimens").to_string())
    return out


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
    """Command-line entry point: write the MF-B / MF-E table."""
    args = parse_args()
    select(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
