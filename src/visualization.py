"""Visualization utilities for assessment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .assessment.knee_valgus import KneeValgusResult
from .assessment.posture import PostureResult


def plot_knee_valgus_assessment(
    result: KneeValgusResult,
    output_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot knee valgus assessment results.

    Args:
        result: Knee valgus assessment result
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    frames = np.arange(len(result.left_angle))

    # Left knee
    ax1.plot(frames, result.left_angle, "b-", label="Angle", linewidth=2)
    ax1.fill_between(
        frames,
        result.left_angle_lower,
        result.left_angle_upper,
        alpha=0.3,
        label="95% CI",
    )
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Valgus Angle (degrees)")
    ax1.set_title(f"Left Knee (max: {result.left_max_valgus:.1f}°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right knee
    ax2.plot(frames, result.right_angle, "b-", label="Angle", linewidth=2)
    ax2.fill_between(
        frames,
        result.right_angle_lower,
        result.right_angle_upper,
        alpha=0.3,
        label="95% CI",
    )
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Valgus Angle (degrees)")
    ax2.set_title(f"Right Knee (max: {result.right_max_valgus:.1f}°)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_posture_assessment(
    result: PostureResult,
    output_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot posture assessment results.

    Args:
        result: Posture assessment result
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    frames = np.arange(len(result.back_arch_angle))

    # Back arch
    ax1.plot(frames, result.back_arch_angle, "b-", label="Angle", linewidth=2)
    ax1.fill_between(
        frames,
        result.back_arch_lower,
        result.back_arch_upper,
        alpha=0.3,
        label="95% CI",
    )
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Back Arch Angle (degrees)")
    ax1.set_title(f"Low Back Arch (max: {result.max_back_arch:.1f}°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Torso lean
    ax2.plot(frames, result.torso_lean_angle, "b-", label="Angle", linewidth=2)
    ax2.fill_between(
        frames,
        result.torso_lean_lower,
        result.torso_lean_upper,
        alpha=0.3,
        label="95% CI",
    )
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Torso Lean Angle (degrees)")
    ax2.set_title(f"Torso Lean (max: {result.max_torso_lean:.1f}°)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
