"""Suggest which specimens to label next, by diversity sampling at specimen level.

Clustering individual images (select_labeling_candidates.py) can propose
another image of a specimen that is already labeled. Here the images of each
specimen are first averaged into one feature vector, the specimens are
clustered with k-means, and one specimen per cluster is proposed. Within a
cluster an unlabeled specimen is preferred (the one nearest the cluster
centre); a labeled specimen is proposed only if the cluster has no unlabeled
one. The number of independent specimens, not of images, limits what a
classifier can learn, so this targets the labels that add the most.

Output (--output-csv): one row per cluster with its number of specimens and of
labeled specimens and the representative specimen, sorted with clusters
without any labeled specimen first and then by size.

Run from the repository root:
    PYTHONPATH=. python exploratory/select_sample_candidates.py --features-csv data/image_metadata_auto_label.csv --n-clusters 15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label"}
UNLABELED = "unlabeled"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata
        (``NON_FEATURE_COLS``).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def aggregate_to_sample_level(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Average the images of each specimen into one feature vector.

    Args:
        df: Image-level table with ``sample_code``, ``section``,
            ``image_id``, ``label`` and the feature columns.
        feature_cols: Feature columns to average.

    Returns:
        pandas.DataFrame: One row per specimen with ``sample_code``,
        ``section``, ``image_count``, ``is_labeled`` (True if any image has
        a label other than ``'unlabeled'``) and the mean of each feature.
    """
    agg = df.groupby("sample_code").agg(
        section=("section", "first"),
        image_count=("image_id", "count"),
        is_labeled=("label", lambda s: bool((s != UNLABELED).any())),
    )
    feature_means = df.groupby("sample_code")[feature_cols].mean()
    return agg.join(feature_means).reset_index()


def select_sample_candidates(features_csv: Path, n_clusters: int, output_csv: Path) -> None:
    """Cluster the specimens and write one labeling candidate per cluster.

    Specimen-mean features are standardized and clustered with k-means
    (seed 42, 10 initializations).

    Args:
        features_csv: Feature table with a ``label`` column
            (``'unlabeled'`` where there is no label).
        n_clusters: Number of clusters, i.e. of candidates (reduced to the
            number of specimens if larger).
        output_csv: Path of the candidate list.

    Raises:
        ValueError: If the table has no feature columns.
    """
    df = pd.read_csv(features_csv)
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError(f"No feature columns found in {features_csv} - check the input file.")

    sample_df = aggregate_to_sample_level(df, feature_cols)
    print(f"Aggregated {len(df)} images into {len(sample_df)} unique physical samples.")
    print(f"Currently labeled samples: {sample_df['is_labeled'].sum()} of {len(sample_df)}")
    print("\nSamples per section:")
    print(sample_df.groupby("section")["sample_code"].count())

    X = StandardScaler().fit_transform(sample_df[feature_cols].values)

    n_clusters = min(n_clusters, len(sample_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    sample_df = sample_df.assign(cluster=cluster_ids)

    rows = []
    for cluster_id in range(n_clusters):
        cluster_mask = (sample_df["cluster"] == cluster_id).values
        cluster_df = sample_df[cluster_mask]
        cluster_X = X[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_X - cluster_center, axis=1)

        # Prefer an unlabeled sample as the representative if one exists in
        # this cluster, even if it isn't the single closest-to-centroid -
        # the goal is new labels, not just a description of the cluster.
        is_unlabeled = ~cluster_df["is_labeled"].values
        if is_unlabeled.any():
            unlabeled_X = cluster_X[is_unlabeled]
            unlabeled_distances = np.linalg.norm(unlabeled_X - cluster_center, axis=1)
            representative = cluster_df[is_unlabeled].iloc[unlabeled_distances.argmin()]
        else:
            representative = cluster_df.iloc[distances.argmin()]

        rows.append(
            {
                "cluster_id": cluster_id,
                "samples_in_cluster": len(cluster_df),
                "labeled_samples_in_cluster": int(cluster_df["is_labeled"].sum()),
                "representative_sample_code": representative["sample_code"],
                "representative_section": representative["section"],
                "representative_is_labeled": bool(representative["is_labeled"]),
            }
        )

    candidates_df = pd.DataFrame(rows).sort_values(
        by=["labeled_samples_in_cluster", "samples_in_cluster"], ascending=[True, False]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(output_csv, index=False)

    n_fully_unlabeled = (candidates_df["labeled_samples_in_cluster"] == 0).sum()
    print(f"\n{n_fully_unlabeled} of {n_clusters} clusters have NO labeled sample at all - prioritize these.")
    print(f"Saved candidate list -> {output_csv}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``features_csv``, ``n_clusters`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv", type=str, required=True,
        help="CSV with combined GLCM+DINOv2 features and a 'label' column",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=15,
        help="Number of sample-level clusters / labeling candidates to generate",
    )
    parser.add_argument(
        "--output-csv", type=str, default="results/tables/sample_labeling_candidates.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``select_sample_candidates``."""
    args = parse_args()
    select_sample_candidates(
        features_csv=Path(args.features_csv),
        n_clusters=args.n_clusters,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()