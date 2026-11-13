"""DINOv2 embeddings of the PPL / XPL thin-section images (step 1 of feature extraction).

Each image goes through the pretrained DINOv2-small model (frozen, CPU only, no
training) and its 384-dim CLS token is saved.

Outputs in --output-dir:
  embeddings.npy       float32, (n_images, 384)
  embeddings_meta.csv  image_id, sample_code, section, image_type, source_path
                       (same row order as embeddings.npy)

Folder layout under --base-dir:
    Najmeh Formation/Dudan Section/SD_87/PPL/*.jpg
    Najmeh Formation/Dudan Section/SD_87/XPL/*.jpg
    Najmeh Formation/Kezi Section/SK_83/PPL/*.jpg
    ...

    python extract_embeddings.py --base-dir /path/to/image_root --output-dir results/features
"""

import argparse
import re
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SECTION_FOLDERS = {"Dudan Section": "Dudan", "Kezi Section": "Kezi"}
IMAGE_TYPE_FOLDERS = ("PPL", "XPL")


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
    return re.sub(r"\s+", "", folder_name.strip())


def find_image_type_dir(sample_dir, image_type):
    """Return the PPL or XPL subfolder of a specimen folder, if it exists.

    Args:
        sample_dir: Specimen folder (``pathlib.Path``).
        image_type: ``'PPL'`` or ``'XPL'``.

    Returns:
        pathlib.Path or None: The subfolder, or None if it is missing.
    """
    candidate = sample_dir / image_type
    return candidate if candidate.is_dir() else None


def discover_images(base_dir):
    """Yield one ``ImageRecord`` per PPL/XPL thin-section image.

    Layout: ``<base_dir>/Najmeh Formation/<Dudan|Kezi> Section/<specimen>/<PPL|XPL>/``.
    Records are ordered by section, specimen folder, image type and file
    name; ``image_id`` numbers the files of each folder from 01. The three
    feature-extraction scripts use the same ordering, so their ``image_id``
    values match.

    Args:
        base_dir: Folder (``pathlib.Path``) that contains ``Najmeh Formation``.

    Yields:
        ImageRecord: One record per image file (.jpg, .jpeg, .png, .tif,
        .tiff). Sections or folders that are missing are skipped.

    Raises:
        FileNotFoundError: If ``base_dir`` has no ``Najmeh Formation`` folder.
    """
    formation_dir = base_dir / "Najmeh Formation"
    if not formation_dir.is_dir():
        raise FileNotFoundError(f"'Najmeh Formation' folder not found under {base_dir}")

    for section_folder, section_label in SECTION_FOLDERS.items():
        section_dir = formation_dir / section_folder
        if not section_dir.is_dir():
            continue

        for sample_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            sample_code = normalize_sample_code(sample_dir.name)

            for image_type in IMAGE_TYPE_FOLDERS:
                type_dir = find_image_type_dir(sample_dir, image_type)
                if type_dir is None:
                    continue

                image_paths = sorted(
                    p for p in type_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                for index, image_path in enumerate(image_paths, start=1):
                    image_id = f"{sample_code}_{image_type}_{index:02d}"
                    yield ImageRecord(
                        image_id=image_id,
                        sample_code=sample_code,
                        section=section_label,
                        image_type=image_type,
                        source_path=image_path,
                    )


def load_model(device):
    """Load the pretrained DINOv2-small model and its image processor.

    The model is frozen: evaluation mode and no gradients. It is used only
    as a fixed feature extractor and is never fine-tuned.

    Args:
        device: ``torch.device`` to place the model on.

    Returns:
        tuple: ``(processor, model)`` from Hugging Face ``transformers``.
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


@torch.no_grad()
def embed_image(image_path, processor, model, device):
    """Compute the DINOv2 embedding of one image.

    The image is converted to RGB and passed through the processor (resize,
    crop and normalization as defined by the model) and the model.

    Args:
        image_path: Path of the image file.
        processor: DINOv2 image processor from ``load_model``.
        model: DINOv2 model from ``load_model``.
        device: ``torch.device`` the model is on.

    Returns:
        numpy.ndarray: The CLS token of the last hidden layer, float32,
        shape (384,).
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy().astype(np.float32)


def run(base_dir, output_dir):
    """Embed every PPL/XPL image and save the embeddings with their metadata.

    Args:
        base_dir: Folder (``pathlib.Path``) that contains ``Najmeh Formation``.
        output_dir: Folder (``pathlib.Path``) for ``embeddings.npy`` and
            ``embeddings_meta.csv``; created if needed. Row i of the CSV
            describes row i of the array.

    Raises:
        RuntimeError: If no images are found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading {MODEL_NAME} (CPU)...")
    processor, model = load_model(device)

    records = list(discover_images(base_dir))
    if not records:
        raise RuntimeError(
            "No PPL/XPL images found. Check --base-dir and the folder layout."
        )
    print(f"Found {len(records)} PPL/XPL images")

    embeddings = np.zeros((len(records), model.config.hidden_size), dtype=np.float32)
    for i, record in enumerate(tqdm(records, desc="Embedding images")):
        embeddings[i] = embed_image(record.source_path, processor, model, device)

    meta_df = pd.DataFrame(
        {
            "image_id": [r.image_id for r in records],
            "sample_code": [r.sample_code for r in records],
            "section": [r.section for r in records],
            "image_type": [r.image_type for r in records],
            "source_path": [str(r.source_path) for r in records],
        }
    )

    np.save(output_dir / "embeddings.npy", embeddings)
    meta_df.to_csv(output_dir / "embeddings_meta.csv", index=False)

    print(f"\nSaved {embeddings.shape[0]} embeddings, dim {embeddings.shape[1]}")
    print(f"  {output_dir / 'embeddings.npy'}")
    print(f"  {output_dir / 'embeddings_meta.csv'}")
    print("\nImages per section and type:")
    print(meta_df.groupby(["section", "image_type"]).size())


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``base_dir`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Folder that contains the 'Najmeh Formation' image tree (images are not part of this repository)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/features",
        help="Where to save embeddings.npy and embeddings_meta.csv",
    )
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(base_dir=Path(args.base_dir), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
