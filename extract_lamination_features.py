"""Lamination features (step 3 of feature extraction), added to combined_features.csv.

Laminated fabric (stromatolite, MF-B) has one strong local orientation and a
regular banding; clotted fabric (thrombolite, MF-E) has neither. Images are
resized to 256 x 256 grayscale, then:
  1. structure tensor coherence, mean and std
  2. Gabor energy at 0/45/90/135 deg (frequency 0.15 cycles/px) and the
     anisotropy = coefficient of variation of the 4 energies
  3. 2D FFT peak strength outside the DC centre (radius 5 px)
-> 8 columns, prefixed lam_.

Image paths are found again from the folders (not read from source_path), because
the Persian-digit file names do not always survive a CSV round trip.
The CSV is updated in place; old lam_ columns are replaced. Images that cannot
be opened are skipped with a warning.

    python extract_lamination_features.py --base-dir /path/to/image_root --features-csv results/features/combined_features.csv
"""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import gabor
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SECTION_FOLDERS = {"Dudan Section": "Dudan", "Kezi Section": "Kezi"}
IMAGE_TYPE_FOLDERS = ("PPL", "XPL")

GABOR_ANGLES_DEG = [0, 45, 90, 135]
GABOR_FREQUENCY = 0.15  # cycles/pixel at 256x256
STRUCTURE_TENSOR_SIGMA = 2.0
RESIZE_DIM = (256, 256)  # same size for all images so the frequencies are comparable
FFT_DC_RADIUS = 5  # px around the centre, not used for the peak


class ImageRecord(NamedTuple):
    """One discovered image.

    Attributes:
        image_id: ``<sample_code>_<image_type>_<NN>``, matching the
            ``image_id`` of the feature table.
        source_path: Full path of the image file.
    """

    image_id: str
    source_path: Path


def normalize_sample_code(folder_name):
    """Turn a specimen folder name into a sample code.

    White space is removed, so ``'SD_102 B'`` becomes ``'SD_102B'`` (SD_102B
    is a separate specimen from SD_102).

    Args:
        folder_name: Name of the specimen folder.

    Returns:
        The sample code, without any white space.
    """
    return re.sub(r"\s+", "", folder_name.strip())


def discover_images(base_dir):
    """List one ``ImageRecord`` per PPL/XPL thin-section image.

    Layout: ``<base_dir>/Najmeh Formation/<Dudan|Kezi> Section/<specimen>/<PPL|XPL>/``.
    Records are ordered by section, specimen folder, image type and file
    name; ``image_id`` numbers the files of each folder from 01. The three
    feature-extraction scripts use the same ordering, so their ``image_id``
    values match.

    Args:
        base_dir: Folder (``pathlib.Path``) that contains ``Najmeh Formation``.

    Returns:
        list of ImageRecord: One record per image file (.jpg, .jpeg, .png,
        .tif, .tiff). Sections or folders that are missing are skipped.

    Raises:
        FileNotFoundError: If ``base_dir`` has no ``Najmeh Formation`` folder.
    """
    formation_dir = base_dir / "Najmeh Formation"
    if not formation_dir.is_dir():
        raise FileNotFoundError(f"'Najmeh Formation' folder not found under {base_dir}")

    records = []
    for section_folder, section_label in SECTION_FOLDERS.items():
        section_dir = formation_dir / section_folder
        if not section_dir.is_dir():
            continue
        for sample_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            sample_code = normalize_sample_code(sample_dir.name)
            for image_type in IMAGE_TYPE_FOLDERS:
                type_dir = sample_dir / image_type
                if not type_dir.is_dir():
                    continue
                image_paths = sorted(
                    p for p in type_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                for index, image_path in enumerate(image_paths, start=1):
                    image_id = f"{sample_code}_{image_type}_{index:02d}"
                    records.append(ImageRecord(image_id, image_path))
    return records


def compute_lamination_features(image_path):
    """Eight descriptors of lamination (preferred orientation and banding).

    The image is converted to grayscale, resized to 256 x 256 px so that the
    spatial frequencies are comparable between images, and scaled to [0, 1].

    1. Structure tensor (Gaussian sigma 2 px): local coherence
       (l1 - l2) / (l1 + l2) from the two eigenvalues, 0 for isotropic and 1
       for perfectly oriented texture; mean and standard deviation over the
       image.
    2. Gabor filters at 0, 45, 90 and 135 degrees (frequency 0.15
       cycles/px): mean magnitude of the response at each angle, and the
       anisotropy = coefficient of variation (std / mean) of the four
       energies. Laminated fabric concentrates energy in one direction (high
       anisotropy); clotted fabric spreads it evenly (low anisotropy).
    3. 2D FFT: maximum magnitude divided by the mean magnitude, after
       zeroing a radius of 5 px around the DC centre. Regular banding gives a
       strong isolated peak.

    Args:
        image_path: Path of the image file.

    Returns:
        dict: ``lam_coherence_mean``, ``lam_coherence_std``,
        ``lam_gabor_energy_0``, ``lam_gabor_energy_45``,
        ``lam_gabor_energy_90``, ``lam_gabor_energy_135``,
        ``lam_gabor_anisotropy`` and ``lam_fft_peak_strength``.
    """
    image = Image.open(image_path).convert("L").resize(RESIZE_DIM)
    gray = np.array(image, dtype=np.float64) / 255.0

    # -----------------------------
    # 1. Structure tensor coherence
    Axx, Axy, Ayy = structure_tensor(gray, sigma=STRUCTURE_TENSOR_SIGMA)
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    denom = l1 + l2
    coherence = np.zeros_like(l1)
    valid = denom > 1e-10
    coherence[valid] = (l1[valid] - l2[valid]) / denom[valid]

    # -----------------------------
    # 2. Gabor energy, 4 orientations
    gabor_energies = []
    for angle_deg in GABOR_ANGLES_DEG:
        theta = np.deg2rad(angle_deg)
        filt_real, filt_imag = gabor(gray, frequency=GABOR_FREQUENCY, theta=theta)
        energy = np.sqrt(filt_real**2 + filt_imag**2).mean()
        gabor_energies.append(float(energy))
    energies_array = np.array(gabor_energies)
    # high = energy in one direction (laminated), low = even (clotted)
    gabor_anisotropy = float(energies_array.std() / (energies_array.mean() + 1e-10))

    # -----------------------------
    # 3. FFT peak strength, DC centre removed
    fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    center_y, center_x = fft_magnitude.shape[0] // 2, fft_magnitude.shape[1] // 2
    yy, xx = np.ogrid[: fft_magnitude.shape[0], : fft_magnitude.shape[1]]
    dist_from_center = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    fft_magnitude[dist_from_center < FFT_DC_RADIUS] = 0
    fft_peak_strength = float(fft_magnitude.max() / (fft_magnitude.mean() + 1e-10))

    return {
        "lam_coherence_mean": float(coherence.mean()),
        "lam_coherence_std": float(coherence.std()),
        "lam_gabor_energy_0": gabor_energies[0],
        "lam_gabor_energy_45": gabor_energies[1],
        "lam_gabor_energy_90": gabor_energies[2],
        "lam_gabor_energy_135": gabor_energies[3],
        "lam_gabor_anisotropy": gabor_anisotropy,
        "lam_fft_peak_strength": fft_peak_strength,
    }


def run(base_dir, features_csv):
    """Compute the lamination features of every image and add them to a table.

    Images that cannot be read are skipped and listed. Existing ``lam_``
    columns in the table are replaced, so the script can be run again on the
    same file. Rows without lamination features are kept (with empty values)
    and counted in a warning.

    Args:
        base_dir: Folder (``pathlib.Path``) that contains ``Najmeh Formation``.
        features_csv: Feature table (``pathlib.Path``) with an ``image_id``
            column; it is overwritten with the extended table.

    Raises:
        RuntimeError: If no images are found.
    """
    records = discover_images(base_dir)
    if not records:
        raise RuntimeError("No PPL/XPL images found. Check --base-dir and the folder layout.")

    existing_df = pd.read_csv(features_csv)

    print(f"Lamination features for {len(records)} images...")
    rows = []
    failed = []
    for record in tqdm(records, desc="Lamination features"):
        try:
            features = compute_lamination_features(record.source_path)
        except (OSError, ValueError) as e:
            failed.append((record.image_id, str(e)))
            continue
        features["image_id"] = record.image_id
        rows.append(features)

    if failed:
        print(f"\nWarning: {len(failed)} images could not be read:")
        for image_id, error in failed[:10]:
            print(f"  {image_id}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    lam_df = pd.DataFrame(rows)

    # replace old lam_ columns if the script is run again
    lam_cols = [c for c in lam_df.columns if c != "image_id"]
    existing_df = existing_df.drop(
        columns=[c for c in lam_cols if c in existing_df.columns], errors="ignore"
    )

    merged = existing_df.merge(lam_df, on="image_id", how="left")

    n_missing = merged[lam_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"\nWarning: {n_missing} rows have no lamination features")

    merged.to_csv(features_csv, index=False)
    print(f"\nAdded {len(lam_cols)} lamination columns to {features_csv}")
    print(f"Columns: {merged.shape[1]}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``base_dir`` and ``features_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir", type=str, required=True,
        help="Folder that contains the 'Najmeh Formation' image tree (images are not part of this repository)",
    )
    parser.add_argument(
        "--features-csv", type=str, default="results/features/combined_features.csv",
        help="Path to combined_features.csv - extended in place with lam_ columns",
    )
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(base_dir=Path(args.base_dir), features_csv=Path(args.features_csv))


if __name__ == "__main__":
    main()
