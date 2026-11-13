# -*- coding: utf-8 -*-
"""Short video showing how each feature family describes one image, with
its formula.

Three scenes, for the GLCM texture features, the lamination (fabric
orientation) features and the DINOv2 embedding, after a title card. The
scenes are drawn by the functions of ``visualize_decision_video.py`` on the
same image and the same computed values, with the governing formula added at
the bottom of each scene. Requires ``imageio-ffmpeg``.

Output: results/figures/feature_math_video_<image_id>.mp4 (or --output)

Run from the repository root:
    PYTHONPATH=. python exploratory/visualize_feature_math_video.py --image-id SD_102_PPL_01
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from visualize_decision_video import (
    FIGSIZE,
    build_context,
    clamp,
    draw_dinov2,
    draw_glcm,
    draw_lamination,
)

BUILD = Path(__file__).parent / ".scratch" / "pptx_build"
FPS = 20

PHASES = [
    ("intro", 1.4),
    ("glcm", 4.2),
    ("lamination", 5.2),
    ("dinov2", 3.6),
]

FORMULA_TEXT = {
    "glcm": (
        r"$Contrast=\sum_{i,j}(i-j)^2\,P(i,j)$      "
        r"$Homogeneity=\sum_{i,j}\dfrac{P(i,j)}{1+(i-j)^2}$"
    ),
    "lamination": (
        r"$Coherence=\dfrac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}$      "
        r"$Anisotropy=\dfrac{std(E_{0},E_{45},E_{90},E_{135})}{mean(E_{0},E_{45},E_{90},E_{135})}$"
    ),
    "dinov2": "No hand-written formula — a learned representation from a frozen, pretrained vision transformer",
}


def draw_intro(fig: plt.Figure, t: float, ctx: dict) -> None:
    """Draw the title card.

    Args:
        fig: Figure to draw on (cleared by the caller).
        t: Progress through the scene, 0 to 1; the text fades in.
        ctx: Scene context from ``build_context`` (not used here).
    """
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    alpha = clamp(t / 0.4)
    ax.text(0.5, 0.58, "How the Model Sees Each Feature", fontsize=26, ha="center", alpha=alpha, weight="bold")
    ax.text(0.5, 0.46, "GLCM texture, fabric orientation, and the deep embedding — with the underlying math",
            fontsize=13, ha="center", alpha=alpha, style="italic")


def with_formula(draw_fn, phase_name):
    """Wrap a scene-drawing function so it also shows the scene's formula.

    Args:
        draw_fn: Scene function with signature ``(fig, t, ctx)``.
        phase_name: Key of ``FORMULA_TEXT`` for the formula to show.

    Returns:
        callable: Scene function that draws the scene, then fades in the
        formula along the bottom edge.
    """
    def wrapped(fig, t, ctx):
        """Draw the wrapped scene and its formula."""
        draw_fn(fig, t, ctx)
        alpha = clamp((t - 0.15) / 0.3)
        fig.text(0.5, 0.045, FORMULA_TEXT[phase_name], ha="center", va="bottom", fontsize=13, alpha=alpha,
                  color="#1a1a1a")
    return wrapped


PHASE_DRAW_FUNCS = {
    "intro": draw_intro,
    "glcm": with_formula(draw_glcm, "glcm"),
    "lamination": with_formula(draw_lamination, "lamination"),
    "dinov2": with_formula(draw_dinov2, "dinov2"),
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
    name, _ = PHASES[-1]
    return name, 1.0


def render(image_id: str, output_path: Path) -> None:
    """Render the video for one image and save it as MP4.

    Args:
        image_id: ``image_id`` of the image to show.
        output_path: Path of the MP4 file; parent folders are created.
    """
    print(f"Loading feature row and model for {image_id} ...")
    ctx = build_context(image_id)

    fig = plt.figure(figsize=FIGSIZE, dpi=100)

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
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    anim.save(str(output_path), writer=animation.FFMpegWriter(fps=FPS, bitrate=4000))

    plt.close(fig)
    print(f"Saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``image_id`` and ``output``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=str, default="SD_102_PPL_01")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: render the video for the chosen image."""
    args = parse_args()
    output_path = Path(args.output) if args.output else Path("results/figures") / f"feature_math_video_{args.image_id}.mp4"
    render(args.image_id, output_path)


if __name__ == "__main__":
    main()
