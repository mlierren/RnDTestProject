"""
Fast animation generation with parallel processing.

Optimizations:
1. Skeleton-only visualization (no bottom graph)
2. Parallel frame generation using multiprocessing
3. CSV caching for processed results
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import multiprocessing as mp
from functools import partial
import tempfile
import shutil

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from .config import JOINT_NAMES, BONES, JOINT_INDEX


# Bone connections for visualization
BONE_CONNECTIONS = [
    (JOINT_INDEX[p], JOINT_INDEX[c]) for p, c in BONES
]


def save_results_csv(
    positions: np.ndarray,
    output_path: Path,
    bone_lengths: Optional[dict] = None,
) -> None:
    """
    Save processed positions to CSV for fast reloading.

    Format: Each row is a frame, columns are joint_x, joint_y, joint_z for each joint.

    Args:
        positions: (n_frames, n_joints, 3) array
        output_path: Path to save CSV
        bone_lengths: Optional bone lengths dict to save as metadata
    """
    n_frames, n_joints, _ = positions.shape

    # Flatten to (n_frames, n_joints * 3)
    flat_positions = positions.reshape(n_frames, -1)

    # Create header
    headers = []
    for joint in JOINT_NAMES:
        headers.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])

    # Save with header
    np.savetxt(
        output_path,
        flat_positions,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )

    # Save bone lengths as separate file if provided
    if bone_lengths is not None:
        bone_path = output_path.with_suffix(".bones.csv")
        with open(bone_path, "w") as f:
            f.write("parent,child,length\n")
            for (parent, child), length in bone_lengths.items():
                f.write(f"{parent},{child},{length}\n")


def load_results_csv(csv_path: Path) -> tuple[np.ndarray, Optional[dict]]:
    """
    Load processed positions from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        (positions, bone_lengths) tuple
    """
    # Load positions
    flat_positions = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    n_frames = flat_positions.shape[0]
    n_joints = len(JOINT_NAMES)

    positions = flat_positions.reshape(n_frames, n_joints, 3)

    # Load bone lengths if available
    bone_path = csv_path.with_suffix(".bones.csv")
    bone_lengths = None
    if bone_path.exists():
        bone_lengths = {}
        with open(bone_path) as f:
            next(f)  # Skip header
            for line in f:
                parent, child, length = line.strip().split(",")
                bone_lengths[(parent, child)] = float(length)

    return positions, bone_lengths


def _render_single_frame(
    args: tuple,
) -> tuple[int, str]:
    """
    Render a single frame (for parallel processing).

    Args:
        args: (frame_idx, original_pos, filtered_pos, coord_system, temp_dir, figsize, dpi, global_limits)

    Returns:
        (frame_idx, temp_file_path)
    """
    frame_idx, original_pos, filtered_pos, coord_system, temp_dir, figsize, dpi, global_limits = args

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Two side-by-side 3D plots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Map coordinate system
    axis_map = {"x": 0, "y": 1, "z": 2}
    v_idx = axis_map[coord_system["vertical"]]
    l_idx = axis_map[coord_system["lateral"]]
    a_idx = axis_map[coord_system["anterior"]]

    # Plot original skeleton (red)
    _plot_skeleton_3d(ax1, original_pos, v_idx, l_idx, a_idx, color='red', alpha=0.8)
    ax1.set_title(f"Original (Frame {frame_idx})", fontsize=12)

    # Plot filtered skeleton (blue)
    _plot_skeleton_3d(ax2, filtered_pos, v_idx, l_idx, a_idx, color='blue', alpha=0.8)
    ax2.set_title(f"Filtered (Frame {frame_idx})", fontsize=12)

    # Set FIXED axis limits from global limits (prevents flickering)
    _set_fixed_axis_limits(ax1, global_limits)
    _set_fixed_axis_limits(ax2, global_limits)

    plt.tight_layout()

    # Save to temp file
    temp_path = Path(temp_dir) / f"frame_{frame_idx:05d}.png"
    fig.savefig(temp_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return frame_idx, str(temp_path)


def _plot_skeleton_3d(
    ax: Axes3D,
    positions: np.ndarray,  # (n_joints, 3)
    v_idx: int,
    l_idx: int,
    a_idx: int,
    color: str = 'blue',
    alpha: float = 0.8,
) -> None:
    """Plot skeleton on 3D axes."""
    # Plot joints
    ax.scatter(
        positions[:, l_idx],
        positions[:, a_idx],
        positions[:, v_idx],
        c=color,
        s=50,
        alpha=alpha,
    )

    # Plot bones
    for p_idx, c_idx in BONE_CONNECTIONS:
        p_pos = positions[p_idx]
        c_pos = positions[c_idx]
        ax.plot(
            [p_pos[l_idx], c_pos[l_idx]],
            [p_pos[a_idx], c_pos[a_idx]],
            [p_pos[v_idx], c_pos[v_idx]],
            c=color,
            linewidth=2,
            alpha=alpha,
        )


def _compute_global_limits(
    original_positions: np.ndarray,  # (n_frames, n_joints, 3)
    filtered_positions: np.ndarray,  # (n_frames, n_joints, 3)
    coord_system: dict[str, str],
    margin: float = 100.0,
) -> dict:
    """
    Compute global axis limits from ALL frames to prevent flickering.

    Returns:
        Dict with xlim, ylim, zlim tuples
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    v_idx = axis_map[coord_system["vertical"]]
    l_idx = axis_map[coord_system["lateral"]]
    a_idx = axis_map[coord_system["anterior"]]

    # Combine all positions from all frames
    all_positions = np.concatenate([
        original_positions.reshape(-1, 3),
        filtered_positions.reshape(-1, 3),
    ], axis=0)

    # Get global min/max for each axis (ignore NaN values)
    l_min = np.nanmin(all_positions[:, l_idx]) - margin
    l_max = np.nanmax(all_positions[:, l_idx]) + margin
    a_min = np.nanmin(all_positions[:, a_idx]) - margin
    a_max = np.nanmax(all_positions[:, a_idx]) + margin
    v_min = np.nanmin(all_positions[:, v_idx]) - margin
    v_max = np.nanmax(all_positions[:, v_idx]) + margin

    # Make ranges equal for proper aspect ratio
    max_range = max(l_max - l_min, a_max - a_min, v_max - v_min) / 2
    l_mid = (l_max + l_min) / 2
    a_mid = (a_max + a_min) / 2
    v_mid = (v_max + v_min) / 2

    return {
        "xlim": (l_mid - max_range, l_mid + max_range),
        "ylim": (a_mid - max_range, a_mid + max_range),
        "zlim": (v_mid - max_range, v_mid + max_range),
    }


def _set_fixed_axis_limits(ax: Axes3D, limits: dict) -> None:
    """Set fixed axis limits from precomputed global limits."""
    ax.set_xlim(limits["xlim"])
    ax.set_ylim(limits["ylim"])
    ax.set_zlim(limits["zlim"])

    ax.set_xlabel("Lateral")
    ax.set_ylabel("Anterior")
    ax.set_zlabel("Vertical")


def create_fast_animation(
    original_positions: np.ndarray,
    filtered_positions: np.ndarray,
    output_path: Path,
    coord_system: dict[str, str],
    fps: int = 15,
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 100,
    max_frames: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> None:
    """
    Create fast skeleton comparison animation using parallel processing.

    Args:
        original_positions: (n_frames, n_joints, 3) original data
        filtered_positions: (n_frames, n_joints, 3) filtered data
        output_path: Path for output GIF
        coord_system: Coordinate system mapping
        fps: Frames per second
        figsize: Figure size in inches
        dpi: Resolution
        max_frames: Maximum frames to render (None = all)
        n_workers: Number of parallel workers (None = auto)
    """
    n_frames = original_positions.shape[0]

    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"  Rendering {n_frames} frames using {n_workers} workers...")

    # Compute GLOBAL axis limits from ALL frames (prevents flickering)
    global_limits = _compute_global_limits(
        original_positions[:n_frames],
        filtered_positions[:n_frames],
        coord_system,
    )
    print(f"  Fixed axis limits: X={global_limits['xlim']}, Y={global_limits['ylim']}, Z={global_limits['zlim']}")

    # Create temp directory for frame images
    temp_dir = tempfile.mkdtemp(prefix="animation_")

    try:
        # Prepare arguments for each frame (includes global_limits)
        render_args = [
            (
                i,
                original_positions[i],
                filtered_positions[i],
                coord_system,
                temp_dir,
                figsize,
                dpi,
                global_limits,  # Fixed limits for all frames
            )
            for i in range(n_frames)
        ]

        # Parallel rendering
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = list(pool.imap(_render_single_frame, render_args, chunksize=10))
        else:
            # Sequential for debugging
            results = [_render_single_frame(args) for args in render_args]

        # Sort by frame index
        results.sort(key=lambda x: x[0])

        # Combine into GIF
        print(f"  Combining frames into GIF...")
        frames = []
        for _, frame_path in results:
            img = Image.open(frame_path)
            frames.append(img.copy())
            img.close()

        # Save GIF
        duration = int(1000 / fps)  # ms per frame
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

        print(f"  Animation saved: {output_path}")

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@dataclass
class CachedResult:
    """Cached processing result for animation-only mode."""
    original_positions: np.ndarray
    filtered_positions: np.ndarray
    bone_lengths: Optional[dict]
    coord_system: dict[str, str]


def save_cached_result(
    result: CachedResult,
    subject_output_dir: Path,
) -> None:
    """Save cached result to CSV files in subject folder."""
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Save original and filtered positions
    original_path = subject_output_dir / "original.csv"
    filtered_path = subject_output_dir / "filtered.csv"

    save_results_csv(result.original_positions, original_path)
    save_results_csv(result.filtered_positions, filtered_path, result.bone_lengths)

    # Save coordinate system
    coord_path = subject_output_dir / "coord.csv"
    with open(coord_path, "w") as f:
        f.write("axis,direction\n")
        for axis, direction in result.coord_system.items():
            f.write(f"{axis},{direction}\n")


def load_cached_result(
    subject_output_dir: Path,
) -> Optional[CachedResult]:
    """Load cached result from CSV files in subject folder."""
    original_path = subject_output_dir / "original.csv"
    filtered_path = subject_output_dir / "filtered.csv"
    coord_path = subject_output_dir / "coord.csv"

    if not (original_path.exists() and filtered_path.exists() and coord_path.exists()):
        return None

    original_positions, _ = load_results_csv(original_path)
    filtered_positions, bone_lengths = load_results_csv(filtered_path)

    # Load coordinate system
    coord_system = {}
    with open(coord_path) as f:
        next(f)  # Skip header
        for line in f:
            axis, direction = line.strip().split(",")
            coord_system[axis] = direction

    return CachedResult(
        original_positions=original_positions,
        filtered_positions=filtered_positions,
        bone_lengths=bone_lengths,
        coord_system=coord_system,
    )


def has_cached_result(subject_output_dir: Path) -> bool:
    """Check if cached result exists in subject folder."""
    original_path = subject_output_dir / "original.csv"
    filtered_path = subject_output_dir / "filtered.csv"
    coord_path = subject_output_dir / "coord.csv"

    return original_path.exists() and filtered_path.exists() and coord_path.exists()
