"""Render a video showing how the trained classifier reads one thin-section
image.

Scenes: the greyscale image; GLCM texture; the lamination descriptors (Gabor
responses at four orientations, structure-tensor coherence, FFT magnitude);
the frozen DINOv2 embedding; assembly of the 404-feature vector; the SHAP
contributions of the top 8 features toward the predicted class; and the
predicted class with its probability.

This is a visualization of existing results, not a new analysis. The feature
values, the prediction and the SHAP explanation come from the stored feature
table and the saved model. The GLCM, Gabor, structure-tensor and FFT arrays
are recomputed from the image only for display, with the same settings as
extract_texture_features.py and extract_lamination_features.py.

Inputs:
  results/features/image_metadata_reliable_classes.csv   feature table (its
      source_path column must point to the image files)
  models/xgb_microfacies.json, models/label_encoder.pkl  from train_xgboost_cv.py

Output: one MP4 (needs imageio-ffmpeg) or GIF of about 24 seconds, by default
results/figures/decision_video_<image_id>.<format>.

Run from the repository root:
    PYTHONPATH=. python exploratory/visualize_decision_video.py --image-id SD_102_PPL_01
    PYTHONPATH=. python exploratory/visualize_decision_video.py --image-id SD_101_PPL_01 --format gif
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib import animation
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.feature import graycomatrix
from skimage.filters import gabor
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

# --- constants copied verbatim from extract_texture_features.py /
# extract_lamination_features.py, so the visuals match the paper's actual
# computation, not a re-tuned approximation of it ---
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GRAY_LEVELS = 32

GABOR_ANGLES_DEG = [0, 45, 90, 135]
GABOR_FREQUENCY = 0.15
STRUCTURE_TENSOR_SIGMA = 2.0
RESIZE_DIM = (256, 256)

FEATURES_CSV = Path("results/features/image_metadata_reliable_classes.csv")
MODEL_PATH = Path("models/xgb_microfacies.json")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label", "cluster"}

FPS = 20
FIGSIZE = (12.8, 7.2)  # 16:9 at reasonable DPI
DPI = 100

# (name, seconds) - edit these to pace the video differently.
# ~24s total, giving each scene enough dwell time to actually read; scale
# all values together (same ratios) for a faster or slower cut instead.
PHASES = [
    ("title", 1.5),
    ("raw_image", 1.8),
    ("glcm", 4.0),
    ("lamination", 5.0),
    ("dinov2", 2.5),
    ("assembly", 1.5),
    ("shap", 4.5),
    ("verdict", 3.0),
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Limit a value to an interval.

    Args:
        x: Value to limit.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        float: ``lo`` if x < lo, ``hi`` if x > hi, otherwise x.
    """
    return max(lo, min(hi, x))


def load_row_and_columns(image_id: str) -> tuple[pd.Series, list[str]]:
    """Read the feature row of one image and the model's feature order.

    Args:
        image_id: Image identifier, e.g. ``'SD_102_PPL_01'``.

    Returns:
        tuple: ``(row, feature_cols)``, the image's row as a Series and the
        404 feature column names in the order the model expects.

    Raises:
        ValueError: If the image is not in the feature table.
    """
    df = pd.read_csv(FEATURES_CSV)
    match = df[df["image_id"] == image_id]
    if match.empty:
        available = df["image_id"].head(5).tolist()
        raise ValueError(
            f"image_id {image_id!r} not found in {FEATURES_CSV}. "
            f"Example valid ids: {available}"
        )
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return match.iloc[0], feature_cols


def compute_visual_intermediates(image_path: Path) -> dict:
    """Recompute the intermediate arrays of the hand-crafted features, for display.

    These arrays are only drawn; the prediction uses the stored feature
    values.

    Args:
        image_path: Path of the image file.

    Returns:
        dict: ``gray_full`` (greyscale image), ``quantized`` (32 grey
        levels), ``glcm_avg`` (GLCM at distance 1, averaged over the four
        angles), ``coherence`` (structure-tensor coherence map),
        ``gabor_images`` (Gabor magnitude at 0, 45, 90 and 135 degrees) and
        ``fft_display`` (log FFT magnitude); the last three at 256 x 256 px.
    """
    image = Image.open(image_path).convert("L")
    gray_full = np.array(image)
    quantized = np.clip(
        (gray_full.astype(np.float64) / 256 * GRAY_LEVELS).astype(np.uint8), 0, GRAY_LEVELS - 1
    )
    glcm = graycomatrix(
        quantized, distances=[1], angles=GLCM_ANGLES, levels=GRAY_LEVELS, symmetric=True, normed=True
    )
    glcm_avg = glcm[:, :, 0, :].mean(axis=2)  # averaged over the 4 angles, distance=1, for display only

    small = Image.open(image_path).convert("L").resize(RESIZE_DIM)
    gray = np.array(small, dtype=np.float64) / 255.0

    Axx, Axy, Ayy = structure_tensor(gray, sigma=STRUCTURE_TENSOR_SIGMA)
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    denom = l1 + l2
    coherence = np.zeros_like(l1)
    valid = denom > 1e-10
    coherence[valid] = (l1[valid] - l2[valid]) / denom[valid]

    gabor_images = []
    for angle_deg in GABOR_ANGLES_DEG:
        theta = np.deg2rad(angle_deg)
        filt_real, filt_imag = gabor(gray, frequency=GABOR_FREQUENCY, theta=theta)
        gabor_images.append(np.sqrt(filt_real**2 + filt_imag**2))

    fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    fft_display = np.log1p(fft_magnitude)

    return dict(
        gray_full=gray_full,
        quantized=quantized,
        glcm_avg=glcm_avg,
        coherence=coherence,
        gabor_images=gabor_images,
        fft_display=fft_display,
    )


def compute_prediction_and_shap(row: pd.Series, feature_cols: list[str]) -> dict:
    """Predict one image with the saved model and explain the prediction.

    The booster is switched to ``multi:softprob`` so that it returns class
    probabilities from the same trees. SHAP values (TreeExplainer) are
    taken for the predicted class.

    Args:
        row: Feature row of the image.
        feature_cols: Feature names in model order.

    Returns:
        dict: ``probs`` (per class), ``classes``, ``pred_idx``,
        ``pred_label`` and ``class_shap`` (one SHAP value per feature for
        the predicted class).
    """
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    classes = label_encoder.classes_

    # The saved booster's objective is multi:softmax, which returns only the
    # predicted class index. Switching to multi:softprob for prediction uses
    # the same trained trees but returns per-class probabilities instead.
    booster.set_param({"objective": "multi:softprob"})

    X = row[feature_cols].to_frame().T.astype(float)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    probs = np.asarray(booster.predict(dmat)).reshape(-1, len(classes))[0]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]

    explainer = shap.TreeExplainer(booster)
    raw_shap = explainer.shap_values(X)
    if isinstance(raw_shap, list):
        class_shap = np.asarray(raw_shap[pred_idx])[0]
    elif np.asarray(raw_shap).ndim == 3:
        class_shap = np.asarray(raw_shap)[0, :, pred_idx]
    else:
        class_shap = np.asarray(raw_shap)[0]

    return dict(
        probs=probs,
        classes=classes,
        pred_idx=pred_idx,
        pred_label=pred_label,
        class_shap=class_shap,
    )


def build_context(image_id: str) -> dict:
    """Collect everything the scene functions need for one image.

    Args:
        image_id: Image identifier.

    Returns:
        dict: ``image_id``, ``row``, ``feature_cols``, ``true_label``, the
        8 features with the largest absolute SHAP value (``top_features``,
        ``top_values``), and the entries of ``compute_visual_intermediates``
        and ``compute_prediction_and_shap``.
    """
    row, feature_cols = load_row_and_columns(image_id)
    image_path = Path(row["source_path"])
    visuals = compute_visual_intermediates(image_path)
    pred = compute_prediction_and_shap(row, feature_cols)

    top_n = 8
    order = np.argsort(-np.abs(pred["class_shap"]))[:top_n]
    top_features = [feature_cols[i] for i in order]
    top_values = pred["class_shap"][order]

    ctx = dict(
        image_id=image_id,
        row=row,
        feature_cols=feature_cols,
        true_label=row.get("label"),
        top_features=top_features,
        top_values=top_values,
        **visuals,
        **pred,
    )
    return ctx


# --- one drawing function per scene; each takes the shared figure, the
# progress t in [0, 1] within the scene, and the context dict ---


def draw_title(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the title card with the image identifier.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    alpha = clamp(t / 0.4)
    ax.text(0.5, 0.58, "How the model reads this specimen", fontsize=26, ha="center", alpha=alpha, weight="bold")
    ax.text(0.5, 0.46, ctx["image_id"], fontsize=18, ha="center", alpha=alpha, family="monospace")
    ax.text(
        0.5, 0.35,
        "Feature extraction, embedding, and SHAP interpretability, step by step",
        fontsize=13, ha="center", alpha=alpha, style="italic",
    )


def draw_raw_image(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the greyscale thin-section image (step 1).

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    ax = fig.add_axes((0.22, 0.15, 0.56, 0.72))
    ax.imshow(ctx["gray_full"], cmap="gray")
    ax.axis("off")
    alpha = clamp(t / 0.3)
    fig.text(0.5, 0.90, "Step 1 - Read the raw thin-section image", ha="center", fontsize=16, alpha=alpha, weight="bold")
    fig.text(0.5, 0.06, "Every image is converted to greyscale before any feature is computed.", ha="center", fontsize=11, alpha=alpha)


def draw_glcm(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the GLCM scene (step 2).

    A window sweeps across the quantized image next to the averaged GLCM;
    the image's stored contrast, homogeneity and correlation (distance 1)
    then count up.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    ax1 = fig.add_axes((0.06, 0.15, 0.42, 0.68))
    ax1.imshow(ctx["quantized"], cmap="gray")
    h, w = ctx["quantized"].shape
    win = max(20, w // 6)
    x = (w - win) * clamp(t * 1.15)
    y = h * 0.42
    ax1.add_patch(Rectangle((x, y), win, win, edgecolor="#ffd400", facecolor="none", lw=2.5))
    ax1.axis("off")
    ax1.set_title("32-level quantized image\n(scanned for grey-level pairs)", fontsize=10)

    ax2 = fig.add_axes((0.55, 0.20, 0.40, 0.55))
    ax2.imshow(ctx["glcm_avg"], cmap="magma")
    ax2.set_title("Grey-level co-occurrence matrix\n(averaged over 4 orientations)", fontsize=10)
    ax2.axis("off")

    reveal = clamp((t - 0.55) / 0.4)
    if reveal > 0:
        contrast = ctx["row"]["glcm_contrast_d1"] * reveal
        homogeneity = ctx["row"]["glcm_homogeneity_d1"] * reveal
        correlation = ctx["row"]["glcm_correlation_d1"] * reveal
        fig.text(
            0.5, 0.05,
            f"Contrast = {contrast:.2f}     Homogeneity = {homogeneity:.3f}     Correlation = {correlation:.3f}",
            ha="center", fontsize=13, family="monospace",
        )
    fig.text(0.5, 0.90, "Step 2 - Classical texture (GLCM / Haralick)", ha="center", fontsize=16, weight="bold")


def draw_lamination(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the lamination scene (step 3).

    The four Gabor responses, the coherence map and the FFT magnitude fade
    in one after another, followed by the image's Gabor anisotropy.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    fig.text(0.5, 0.94, "Step 3 - Fabric orientation (Gabor + structure tensor)", ha="center", fontsize=16, weight="bold")
    panels = [
        ("Gabor 0deg", ctx["gabor_images"][0], "viridis"),
        ("Gabor 45deg", ctx["gabor_images"][1], "viridis"),
        ("Gabor 90deg", ctx["gabor_images"][2], "viridis"),
        ("Gabor 135deg", ctx["gabor_images"][3], "viridis"),
        ("Structure-tensor coherence", ctx["coherence"], "plasma"),
        ("2D FFT magnitude (log)", ctx["fft_display"], "inferno"),
    ]
    n = len(panels)
    positions = [
        (0.04, 0.52, 0.28, 0.34), (0.36, 0.52, 0.28, 0.34), (0.68, 0.52, 0.28, 0.34),
        (0.04, 0.14, 0.28, 0.34), (0.36, 0.14, 0.28, 0.34), (0.68, 0.14, 0.28, 0.34),
    ]
    for i, ((name, arr, cmap), rect) in enumerate(zip(panels, positions)):
        appear_t = i / n
        alpha = clamp((t - appear_t) / (1.0 / n))
        if alpha <= 0:
            continue
        ax = fig.add_axes(rect)
        ax.imshow(arr, cmap=cmap, alpha=alpha)
        ax.set_title(name, fontsize=9, alpha=alpha)
        ax.axis("off")

    if t > 0.88:
        anisotropy = ctx["row"]["lam_gabor_anisotropy"]
        fig.text(
            0.5, 0.06,
            f"Gabor anisotropy = {anisotropy:.3f}  "
            "(spread of filter energy across the 4 orientations - the single feature that survives fold-wise validation)",
            ha="center", fontsize=11,
        )


def draw_dinov2(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the DINOv2 scene (step 4): the first 40 of the 384 embedding values.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    fig.text(0.5, 0.90, "Step 4 - Frozen deep visual embedding (DINOv2-small)", ha="center", fontsize=16, weight="bold")

    ax_img = fig.add_axes((0.04, 0.30, 0.18, 0.4))
    ax_img.imshow(ctx["gray_full"], cmap="gray")
    ax_img.axis("off")

    ax_arrow = fig.add_axes((0.24, 0.30, 0.16, 0.4))
    ax_arrow.axis("off")
    alpha = clamp(t / 0.3)
    ax_arrow.annotate("", xy=(0.9, 0.5), xytext=(0.05, 0.5), arrowprops=dict(arrowstyle="->", lw=2, alpha=alpha))
    ax_arrow.text(0.5, 0.62, "frozen ViT\n(no fine-tuning)", ha="center", fontsize=9, alpha=alpha)

    n_show = 40
    reveal_count = int(n_show * clamp((t - 0.25) / 0.6))
    ax_bar = fig.add_axes((0.44, 0.20, 0.52, 0.55))
    if reveal_count > 0:
        cols = [f"dino_{i:03d}" for i in range(reveal_count)]
        vals = ctx["row"][cols].astype(float).values
        ax_bar.bar(range(reveal_count), vals, color="#3b7dd8", width=0.9)
    ax_bar.set_xlim(0, n_show)
    ax_bar.set_title("First 40 of 384 embedding dimensions", fontsize=10)
    ax_bar.set_xlabel("dimension", fontsize=9)

    fig.text(0.5, 0.06, "The embedding is the model's learned summary of the whole image - not built from any named rule.", ha="center", fontsize=11)


def draw_assembly(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the feature-count scene (step 5): 12 GLCM + 384 DINOv2 + 8
    lamination = 404 features.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    fig.text(0.5, 0.90, "Step 5 - Assemble the full feature vector", ha="center", fontsize=16, weight="bold")

    families = [("GLCM / Haralick", 12, 0.20), ("DINOv2 embedding", 384, 0.5), ("Lamination descriptors", 8, 0.80)]
    # The counters reach 12 / 384 / 8 by t = 0.22 and hold them for the rest
    # of the scene; the early threshold leaves a margin for rounding to whole
    # frames (30 frames at 20 fps).
    reveal = clamp(t / 0.22)
    for name, count, x in families:
        shown = int(round(count * reveal))
        ax.text(x, 0.55, f"{shown}", ha="center", fontsize=30, weight="bold", family="monospace")
        ax.text(x, 0.40, name, ha="center", fontsize=12)

    if t > 0.22:
        alpha = clamp((t - 0.22) / 0.08)
        ax.text(0.5, 0.12, "= 404 features total", ha="center", fontsize=22, weight="bold", alpha=alpha, color="#1a7a3c")


def draw_shap(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the SHAP scene (step 6).

    Bars for the 8 features with the largest absolute SHAP value, positive
    (toward the predicted class) in red and negative in blue.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    fig.text(0.5, 0.94, f"Step 6 - Which features pushed toward {ctx['pred_label']}?", ha="center", fontsize=16, weight="bold")
    ax = fig.add_axes((0.32, 0.12, 0.62, 0.75))

    names = ctx["top_features"]
    values = ctx["top_values"]
    n = len(names)
    y_pos = np.arange(n)[::-1]

    reveal_each = 1.0 / n
    for i, (name, val) in enumerate(zip(names, values)):
        appear_t = i * reveal_each
        local = clamp((t - appear_t) / reveal_each)
        if local <= 0:
            continue
        color = "#c0392b" if val > 0 else "#2166ac"
        ax.barh(y_pos[i], val * local, color=color, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (push toward / away from the predicted class)", fontsize=10)


def draw_verdict(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the result: predicted class, its probability, the true label (if
    known) and the probability of every class.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, from 0 to 1.
        ctx: Scene context from ``build_context``.
    """
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    alpha = clamp(t / 0.3)

    confidence = ctx["probs"][ctx["pred_idx"]] * 100
    ax.text(0.5, 0.68, f"Predicted: {ctx['pred_label']}", ha="center", fontsize=28, weight="bold", alpha=alpha)
    ax.text(0.5, 0.56, f"{confidence:.1f}% predicted probability", ha="center", fontsize=15, alpha=alpha)

    true_label = ctx["true_label"]
    if isinstance(true_label, str) and t > 0.35:
        alpha2 = clamp((t - 0.35) / 0.3)
        match = true_label == ctx["pred_label"]
        mark = "correct" if match else "does not match"
        color = "#1a7a3c" if match else "#c0392b"
        ax.text(0.5, 0.44, f"True label: {true_label}  ({mark})", ha="center", fontsize=14, color=color, alpha=alpha2)

    if t > 0.55:
        alpha3 = clamp((t - 0.55) / 0.35)
        ax_bar = fig.add_axes((0.28, 0.10, 0.44, 0.18))
        ax_bar.bar(ctx["classes"], ctx["probs"] * alpha3, color="#3b7dd8")
        ax_bar.tick_params(labelsize=8)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title("Predicted probability by class", fontsize=9)


PHASE_DRAW_FUNCS = {
    "title": draw_title,
    "raw_image": draw_raw_image,
    "glcm": draw_glcm,
    "lamination": draw_lamination,
    "dinov2": draw_dinov2,
    "assembly": draw_assembly,
    "shap": draw_shap,
    "verdict": draw_verdict,
}


def total_frames() -> int:
    """Total number of video frames for all scenes in ``PHASES`` at ``FPS``."""
    return sum(round(seconds * FPS) for _, seconds in PHASES)


def frame_to_phase(frame_idx: int) -> tuple[str, float]:
    """Find the scene of a frame and the progress within that scene.

    Args:
        frame_idx: 0-based frame number.

    Returns:
        tuple: ``(scene_name, t)`` with t from 0 (first frame of the scene)
        to 1 (last frame). Frames past the end map to the last scene, t = 1.
    """
    remaining = frame_idx
    for name, seconds in PHASES:
        n = round(seconds * FPS)
        if remaining < n:
            return name, remaining / max(1, n - 1)
        remaining -= n
    name, seconds = PHASES[-1]
    return name, 1.0


def render(image_id: str, output_path: Path, fmt: str) -> None:
    """Render the video for one image and save it.

    Args:
        image_id: Image identifier.
        output_path: Output file; parent folders are created.
        fmt: ``'mp4'`` (needs imageio-ffmpeg) or ``'gif'``.

    Raises:
        RuntimeError: If MP4 output is requested without imageio-ffmpeg.
    """
    print(f"Loading feature row and model for {image_id} ...")
    ctx = build_context(image_id)
    print(f"Predicted: {ctx['pred_label']}  (true label: {ctx['true_label']})")

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    def update(frame_idx: int):
        """Redraw the figure for one frame."""
        fig.clf()
        phase_name, t = frame_to_phase(frame_idx)
        PHASE_DRAW_FUNCS[phase_name](fig, t, ctx)
        return []

    n_frames = total_frames()
    print(f"Rendering {n_frames} frames at {FPS} fps (~{n_frames / FPS:.1f}s) ...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        anim.save(str(output_path), writer=animation.PillowWriter(fps=FPS))
    else:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError(
                "Saving .mp4 needs imageio-ffmpeg (pip install imageio-ffmpeg), "
                "or pass --format gif to avoid the ffmpeg dependency entirely."
            ) from exc
        anim.save(str(output_path), writer=animation.FFMpegWriter(fps=FPS, bitrate=4000))

    plt.close(fig)
    print(f"Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``image_id``, ``output`` and ``format``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-id", type=str, default="SD_102_PPL_01", help="e.g. SD_102_PPL_01 or SK_84_PPL_01")
    parser.add_argument("--output", type=str, default=None, help="Output path; defaults to results/figures/decision_video_<image_id>.<ext>")
    parser.add_argument("--format", type=str, choices=["mp4", "gif"], default="mp4")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: render the video for the chosen image."""
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else Path("results/figures") / f"decision_video_{args.image_id}.{args.format}"
    )
    render(args.image_id, output_path, args.format)


if __name__ == "__main__":
    main()
