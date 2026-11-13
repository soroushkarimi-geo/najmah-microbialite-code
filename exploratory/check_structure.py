"""Print the folder layout of ``data/raw`` and count the images of a few specimens.

A quick check of an image archive before feature extraction: lists the first
30 entries under ``data/raw`` (folders, image files and other files), then the
number of top-level images and of images in the ``Binocular`` subfolder for the
first five specimen folders. Exits with status 1 if ``data/raw`` is missing.

Run from the repository root:
    python exploratory/check_structure.py
"""
from pathlib import Path

raw_dir = Path("data/raw")
print("Structure of data/raw:")
print("=" * 50)

if not raw_dir.exists():
    print("❌ data/raw directory does not exist!")
    exit(1)

items = sorted(raw_dir.rglob("*"))
for item in items[:30]:  # First 30 items
    rel_path = item.relative_to(raw_dir)
    indent = "  " * (len(rel_path.parts) - 1)
    if item.is_dir():
        marker = "📁"
    elif item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        marker = "🖼️"
    else:
        marker = "📄"
    print(f"{indent}{marker} {rel_path}")

print(f"\nTotal items found: {len(items)}")

# Count images per sample
samples = [d for d in raw_dir.iterdir() if d.is_dir()]
print(f"\nSample directories found: {len(samples)}")
for sample in samples[:5]:
    images = list(sample.glob("*.jpg")) + list(sample.glob("*.png")) + list(sample.glob("*.tif"))
    binocular_dir = sample / "Binocular"
    binocular_count = 0
    if binocular_dir.exists():
        binocular_count = len(list(binocular_dir.glob("*.jpg")) + list(binocular_dir.glob("*.png")) + list(binocular_dir.glob("*.tif")))
    print(f"  {sample.name}: {len(images)} top-level images, {binocular_count} in Binocular/")
