"""GLCM texture features of every PPL / XPL image, joined with the DINOv2
embeddings (step 2 of feature extraction, run after extract_embeddings.py).

GLCM: grayscale quantized to 32 levels, distances 1 and 3 px, angles 0/45/90/135,
six properties averaged over the angles -> 12 features (glcm_<property>_d<distance>).

Output: combined_features.csv in --features-dir, one row per image:
    image_id, sample_code, section, image_type, source_path, glcm_..., dino_000 ... dino_383

    python extract_texture_features.py --base-dir /path/to/image_root --features-dir results/features
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# same image discovery as extract_embeddings.py, so the image_id values match
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SECTION_FOLDERS = {"Dudan Section": "Dudan", "Kezi Section": "Kezi"}
IMAGE_TYPE_FOLDERS = ("PPL", "XPL")

GLCM_DISTANCES = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GRAY_LEVELS = 32


class ImageRecord(NamedTuple):
    """One discovered image and where it came from.

    Attributes:
        image_id: ``<sample_code>_<image_type>_<NN>``, where NN is the 1-based
            position of the file in its sorted folder.
        sample_code: Specimen code from the folder name, e.g. ``SD_102B``.
        section: ``'Dudan'`` or ``'Kezi'``.
        image_type: ``'PPL'`` or ``'XPL'``.
        source_path: Full path of the image file.
    """

    image_id: str
    sample_code: str
    section: str
    image_type: str
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
    import re

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
                    records.append(
                        ImageRecord(image_id, sample_code, section_label, image_type, image_path)
                    )
    return records


def compute_glcm_features(image_path):
    """Gray-level co-occurrence (GLCM) texture features of one image.

    The image is converted to grayscale and quantized to ``GRAY_LEVELS``
    (32) levels. A symmetric, normalized GLCM is computed for each distance in
    ``GLCM_DISTANCES`` (1 and 3 px) at the four angles 0, 45, 90 and 135
    degrees. Each property in ``GLCM_PROPERTIES`` is averaged over the angles,
    which makes the features independent of image orientation.

    Args:
        image_path: Path of the image file.

    Returns:
        dict: 12 features named ``glcm_<property>_d<distance>``, e.g.
        ``glcm_contrast_d1``.
    """
    image = Image.open(image_path).convert("L")
    gray = np.array(image)
    quantized = (gray.astype(np.float64) / 256 * GRAY_LEVELS).astype(np.uint8)
    quantized = np.clip(quantized, 0, GRAY_LEVELS - 1)

    features = {}
    for distance in GLCM_DISTANCES:
        glcm = graycomatrix(
            quantized,
            distances=[distance],
            angles=GLCM_ANGLES,
            levels=GRAY_LEVELS,
            symmetric=True,
            normed=True,
        )
        for prop in GLCM_PROPERTIES:
            values = graycoprops(glcm, prop)[0]  # one value per angle
            features[f"glcm_{prop.lower()}_d{distance}"] = float(values.mean())
    return features


def run(base_dir, features_dir):
    """Compute GLCM features and join them with the DINOv2 embeddings.

    Reads ``embeddings.npy`` and ``embeddings_meta.csv`` (from
    ``extract_embeddings.py``), computes the GLCM features of every image and
    joins the three on ``image_id``. A warning is printed if some images have
    no matching embedding.

    Args:
        base_dir: Folder (``pathlib.Path``) that contains ``Najmeh Formation``.
        features_dir: Folder (``pathlib.Path``) with the embedding files;
            ``combined_features.csv`` is written here.

    Raises:
        FileNotFoundError: If the embedding files are missing.
        RuntimeError: If no images are found.
    """
    embeddings_path = features_dir / "embeddings.npy"
    meta_path = features_dir / "embeddings_meta.csv"
    if not embeddings_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Run extract_embeddings.py first - missing {embeddings_path} or {meta_path}"
        )

    embeddings = np.load(embeddings_path)
    meta_df = pd.read_csv(meta_path)

    records = discover_images(base_dir)
    if not records:
        raise RuntimeError("No PPL/XPL images found. Check --base-dir and the folder layout.")

    print(f"GLCM features for {len(records)} images...")
    texture_rows = []
    for record in tqdm(records, desc="Texture features"):
        row = compute_glcm_features(record.source_path)
        row["image_id"] = record.image_id
        texture_rows.append(row)
    texture_df = pd.DataFrame(texture_rows)

    dino_cols = [f"dino_{i:03d}" for i in range(embeddings.shape[1])]
    dino_df = pd.DataFrame(embeddings, columns=dino_cols)
    dino_df["image_id"] = meta_df["image_id"].values

    combined = meta_df.merge(texture_df, on="image_id", how="inner").merge(
        dino_df, on="image_id", how="inner"
    )

    if len(combined) != len(records):
        print(
            f"Warning: only {len(combined)} of {len(records)} images matched the embeddings, "
            "were images added or removed after extract_embeddings.py?"
        )

    output_path = features_dir / "combined_features.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} rows x {combined.shape[1]} columns to {output_path}")
    print(f"  Texture columns: {len(texture_df.columns) - 1}")
    print(f"  DINOv2 columns: {len(dino_cols)}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``base_dir`` and ``features_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Folder that contains the 'Najmeh Formation' image tree (images are not part of this repository)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="results/features",
        help="Folder with embeddings.npy/embeddings_meta.csv; combined_features.csv is written here too",
    )
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(base_dir=Path(args.base_dir), features_dir=Path(args.features_dir))


if __name__ == "__main__":
    main()
