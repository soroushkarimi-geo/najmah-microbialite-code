"""Feasibility check: do the lamination features separate MF-B from MF-E
better on binocular-microscope images than on thin-section images?

The eight lamination features of ``extract_lamination_features.py`` (structure
tensor coherence, Gabor energies and anisotropy, FFT peak strength; same code,
imported) are computed on the binocular images of every MF-B and MF-E specimen
listed in ``auto_label.SEED_LABELS`` and averaged per specimen. The same
features from the thin-section images of the same specimens are averaged in
the same way. For each feature, the separability of the two classes is
measured with the one-way ANOVA F-value (the model-independent measure also
used in ``interpretability_analysis.py``), on both image types.

Binocular images are read from the ``Bino`` or ``Binocullar`` subfolder of each
specimen folder (both spellings occur in the archive). These folders also hold
photographs of the specimen's label tag, which are not filtered out. They can
only add noise to the binocular features, so a higher binocular F-value is a
conservative result, while a lower one should be confirmed on a curated set of
fabric images.

Inputs (not distributed with this repository):
  <repository root>/Najmeh Formation/<Dudan|Kezi> Section/<specimen>/<Bino|Binocullar>/*.jpg
  results/features/image_metadata_auto_label.csv   thin-section feature table

Outputs, written next to this script:
  step1_binocular_lamination_features.csv   specimen-mean binocular features
  step1_separability_comparison.csv         F-values of both image types per feature

Run from the repository root:
    python binocular/feasibility_check_lamination.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extract_lamination_features import compute_lamination_features  # noqa: E402
from auto_label import SEED_LABELS, normalize_sample_code  # noqa: E402

FORMATION_DIR = PROJECT_ROOT / "Najmeh Formation"
SECTION_FOLDERS = {"Dudan Section": "Dudan", "Kezi Section": "Kezi"}
BINOCULAR_FOLDER_NAMES = ("Bino", "Binocullar")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_CLASSES = ("MF-B", "MF-E")

LAM_FEATURE_COLS = [
    "lam_coherence_mean", "lam_coherence_std",
    "lam_gabor_energy_0", "lam_gabor_energy_45", "lam_gabor_energy_90", "lam_gabor_energy_135",
    "lam_gabor_anisotropy", "lam_fft_peak_strength",
]


def find_specimen_dirs() -> dict[str, Path]:
    """Find every specimen folder of the Dudan and Kezi sections.

    Returns:
        dict: Normalized sample code (``auto_label.normalize_sample_code``)
        -> specimen folder path. Missing section folders are skipped.
    """
    specimen_dirs = {}
    for section_folder in SECTION_FOLDERS:
        section_dir = FORMATION_DIR / section_folder
        if not section_dir.is_dir():
            continue
        for sample_dir in section_dir.iterdir():
            if sample_dir.is_dir():
                specimen_dirs[normalize_sample_code(sample_dir.name)] = sample_dir
    return specimen_dirs


def binocular_images_for(specimen_dir: Path) -> list[Path]:
    """List the binocular images of one specimen.

    Args:
        specimen_dir: Specimen folder.

    Returns:
        list of pathlib.Path: The .jpg, .jpeg and .png files in its ``Bino``
        and ``Binocullar`` subfolders (empty if neither exists).
    """
    images = []
    for folder_name in BINOCULAR_FOLDER_NAMES:
        bino_dir = specimen_dir / folder_name
        if bino_dir.is_dir():
            images.extend(p for p in bino_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return images


def compute_binocular_lamination(specimen_dirs: dict[str, Path]) -> pd.DataFrame:
    """Specimen-mean lamination features from the binocular images.

    Covers every MF-B and MF-E specimen of ``SEED_LABELS``. Sample codes are
    matched to folders ignoring hyphens, underscores and case. Specimens
    without a folder or without binocular images, and images that cannot
    be read, are skipped with a message.

    Args:
        specimen_dirs: Output of ``find_specimen_dirs``.

    Returns:
        pandas.DataFrame: One row per specimen with the eight ``lam_``
        features (means over its binocular images), ``sample_code``,
        ``label`` and ``n_binocular_images``.
    """
    rows = []
    for sample_code, label in SEED_LABELS.items():
        if label not in TARGET_CLASSES:
            continue
        norm_code = re.sub(r"[-_]", "", sample_code).upper()
        matching_dir = None
        for dir_code, dir_path in specimen_dirs.items():
            if re.sub(r"[-_]", "", dir_code).upper() == norm_code:
                matching_dir = dir_path
                break
        if matching_dir is None:
            print(f"  (no image folder found for {sample_code}, skipping)")
            continue

        images = binocular_images_for(matching_dir)
        if not images:
            print(f"  (no binocular images found for {sample_code}, skipping)")
            continue

        per_image_feats = []
        for img_path in images:
            try:
                per_image_feats.append(compute_lamination_features(img_path))
            except (OSError, ValueError) as e:
                print(f"  (failed on {img_path.name}: {e})")
        if not per_image_feats:
            continue

        feat_df = pd.DataFrame(per_image_feats)
        mean_feats = feat_df.mean().to_dict()
        mean_feats["sample_code"] = sample_code
        mean_feats["label"] = label
        mean_feats["n_binocular_images"] = len(per_image_feats)
        rows.append(mean_feats)

    return pd.DataFrame(rows)


def anova_f_per_feature(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """One-way ANOVA F-value of MF-B vs MF-E for each feature.

    Args:
        df: Specimen-level table with ``label`` and the feature columns.
        feature_cols: Features to test.

    Returns:
        pandas.Series: F-value per feature, highest first. NaN when either
        class has fewer than two non-missing values.
    """
    results = {}
    for col in feature_cols:
        b_vals = df.loc[df["label"] == "MF-B", col].dropna()
        e_vals = df.loc[df["label"] == "MF-E", col].dropna()
        if len(b_vals) < 2 or len(e_vals) < 2:
            results[col] = np.nan
            continue
        f_stat, _ = f_oneway(b_vals, e_vals)
        results[col] = f_stat
    return pd.Series(results).sort_values(ascending=False)


def main() -> None:
    """Command-line entry point: run the binocular vs thin-section comparison.

    Computes the binocular features, the F-value per feature on both image
    types for the same specimens, and writes the two output tables. Stops
    early if no binocular features could be computed.
    """
    print("=== Step 1 feasibility check: binocular vs thin-section lamination separability ===\n")

    specimen_dirs = find_specimen_dirs()
    print(f"Found {len(specimen_dirs)} specimen folders on disk.\n")

    print("Computing lamination features on BINOCULAR images (confirmed MF-B/MF-E specimens)...")
    binoc_df = compute_binocular_lamination(specimen_dirs)
    print(f"\nBinocular-derived feature table: {len(binoc_df)} specimens "
          f"({(binoc_df['label'] == 'MF-B').sum()} MF-B, {(binoc_df['label'] == 'MF-E').sum()} MF-E)")

    if binoc_df.empty:
        print("No binocular data could be computed - aborting.")
        return

    binoc_df.to_csv(Path(__file__).parent / "step1_binocular_lamination_features.csv", index=False)

    print("\n--- ANOVA F-value per feature, BINOCULAR images ---")
    binoc_f = anova_f_per_feature(binoc_df, LAM_FEATURE_COLS)
    print(binoc_f.round(3))

    # Thin-section comparison, same specimens, from the existing combined feature table
    thin_section_csv = PROJECT_ROOT / "results" / "features" / "image_metadata_auto_label.csv"
    ts_df = pd.read_csv(thin_section_csv)
    ts_df = ts_df[ts_df["sample_code"].isin(binoc_df["sample_code"])]
    ts_per_specimen = ts_df.groupby(["sample_code", "label"])[LAM_FEATURE_COLS].mean().reset_index()

    print(f"\nThin-section comparison table: {len(ts_per_specimen)} specimens "
          f"({(ts_per_specimen['label'] == 'MF-B').sum()} MF-B, {(ts_per_specimen['label'] == 'MF-E').sum()} MF-E)")
    print("\n--- ANOVA F-value per feature, THIN-SECTION images (same specimens) ---")
    ts_f = anova_f_per_feature(ts_per_specimen, LAM_FEATURE_COLS)
    print(ts_f.round(3))

    comparison = pd.DataFrame({"binocular_F": binoc_f, "thin_section_F": ts_f})
    comparison["binocular_higher"] = comparison["binocular_F"] > comparison["thin_section_F"]
    comparison.to_csv(Path(__file__).parent / "step1_separability_comparison.csv")

    print("\n=== Side-by-side comparison ===")
    print(comparison.round(3))
    print(f"\nBinocular F-value higher than thin-section for {comparison['binocular_higher'].sum()} "
          f"of {len(comparison)} features.")


if __name__ == "__main__":
    main()
