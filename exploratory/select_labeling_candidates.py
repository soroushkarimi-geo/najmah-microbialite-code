"""Suggest which images to label next, by diversity sampling.

The images are clustered in the standardized feature space with k-means, and
the image closest to each cluster centre is proposed as a labeling candidate.
Clusters that contain no labeled image yet come first, because they are the
largest gaps in the labeled set.

This needs no trained classifier, which makes it useful when few specimens
are labeled and some classes have no examples: a classifier can only be
uncertain among the classes it has seen, so confidence-based active learning
cannot point to a class that is missing altogether.

Output (--output-csv): one row per cluster with its size, the representative
image_id, sample_code and section, and the labels already present in the
cluster, sorted with unlabeled clusters first and then by size.

Run from the repository root:
    PYTHONPATH=. python exploratory/select_labeling_candidates.py --features-csv data/image_metadata_auto_label.csv --n-clusters 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata
        (``NON_FEATURE_COLS``).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def select_candidates(features_csv: Path, n_clusters: int, output_csv: Path) -> None:
    """Cluster the images and write one labeling candidate per cluster.

    Features are standardized and clustered with k-means (seed 42, 10
    initializations). The representative of a cluster is the image nearest
    to its centre.

    Args:
        features_csv: Feature table, optionally with a ``label`` column in
            which unlabeled rows are ``'unlabeled'``.
        n_clusters: Number of clusters, i.e. of candidates to review
            (reduced to the number of images if larger).
        output_csv: Path of the candidate list.

    Raises:
        ValueError: If the table has no feature columns.
    """
    df = pd.read_csv(features_csv)
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError(f"No feature columns found in {features_csv} - check the input file.")

    X = StandardScaler().fit_transform(df[feature_cols].values)

    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    df = df.assign(cluster=cluster_ids)

    has_label_col = "label" in df.columns
    rows = []
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_X = X[df["cluster"] == cluster_id]
        distances = np.linalg.norm(cluster_X - cluster_center, axis=1)
        representative_idx = cluster_df.index[distances.argmin()]
        representative = df.loc[representative_idx]

        existing_labels = []
        if has_label_col:
            existing_labels = sorted(
                l for l in cluster_df["label"].unique() if l != "unlabeled"
            )

        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_df),
                "representative_image_id": representative["image_id"],
                "representative_sample_code": representative["sample_code"],
                "representative_section": representative.get("section", ""),
                "already_has_seed_label": bool(existing_labels),
                "existing_labels_in_cluster": ";".join(existing_labels),
            }
        )

    candidates_df = pd.DataFrame(rows).sort_values(
        by=["already_has_seed_label", "cluster_size"], ascending=[True, False]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(output_csv, index=False)

    n_unlabeled_clusters = (~candidates_df["already_has_seed_label"]).sum()
    print(f"Clustered {len(df)} images into {n_clusters} groups.")
    print(f"{n_unlabeled_clusters} of {n_clusters} clusters have NO seed label yet - review these first.")
    print(f"Saved candidate list -> {output_csv}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``features_csv``, ``n_clusters`` and
        ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv", type=str, required=True,
        help="CSV with combined GLCM+DINOv2 features (and optionally a 'label' column)",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=20,
        help="Number of clusters / labeling candidates to generate this round",
    )
    parser.add_argument(
        "--output-csv", type=str, default="results/tables/labeling_candidates.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``select_candidates``."""
    args = parse_args()
    select_candidates(
        features_csv=Path(args.features_csv),
        n_clusters=args.n_clusters,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
