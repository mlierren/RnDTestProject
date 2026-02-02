"""Posture assessment for NASM overhead squat (Lateral View)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.preprocessing import FilteredResultAdapter


@dataclass
class PostureResult:
    """Results of posture assessment (lateral view)."""

    # Low back arch (lumbar lordosis angle in degrees)
    # Angle between upper back and lower back vectors
    back_arch_angle: np.ndarray  # (N,)
    back_arch_lower: np.ndarray
    back_arch_upper: np.ndarray

    # Torso lean (forward lean angle in degrees)
    # Positive = leaning forward, Negative = leaning backward
    torso_lean_angle: np.ndarray  # (N,)
    torso_lean_lower: np.ndarray
    torso_lean_upper: np.ndarray

    # Summary statistics
    max_back_arch: float
    mean_back_arch: float
    max_torso_lean: float
    mean_torso_lean: float


def compute_back_arch_angle(
    torso: np.ndarray,
    waist: np.ndarray,
    hip_center: np.ndarray,
    lateral_axis: int,
    anterior_axis: int,
) -> np.ndarray:
    """
    Compute low back arch angle (lumbar lordosis) in sagittal plane.

    Measures the angle between:
    - Vector from waist to torso (upper back direction)
    - Vector from hip_center to waist (lower back direction)

    Args:
        torso: (N, 3) torso positions
        waist: (N, 3) waist positions
        hip_center: (N, 3) hip center positions
        lateral_axis: Index of lateral axis
        anterior_axis: Index of anterior axis

    Returns:
        Angle in degrees. 180 = straight, < 180 = lordosis (arch)
    """
    vertical_axis = 3 - lateral_axis - anterior_axis

    # Project to sagittal plane (anterior + vertical)
    torso_2d = np.column_stack([torso[:, anterior_axis], torso[:, vertical_axis]])
    waist_2d = np.column_stack([waist[:, anterior_axis], waist[:, vertical_axis]])
    hip_2d = np.column_stack([hip_center[:, anterior_axis], hip_center[:, vertical_axis]])

    # Upper back vector (waist to torso)
    upper_vec = torso_2d - waist_2d
    upper_len = np.linalg.norm(upper_vec, axis=1, keepdims=True)
    upper_unit = upper_vec / (upper_len + 1e-8)

    # Lower back vector (hip to waist)
    lower_vec = waist_2d - hip_2d
    lower_len = np.linalg.norm(lower_vec, axis=1, keepdims=True)
    lower_unit = lower_vec / (lower_len + 1e-8)

    # Angle between vectors (0° when aligned = straight back)
    dot_product = np.sum(upper_unit * lower_unit, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_product))

    # angles = 0° means straight back, angles > 0 means lordosis
    return angles


def compute_torso_lean_angle(
    head: np.ndarray,
    torso: np.ndarray,
    waist: np.ndarray,
    lateral_axis: int,
    anterior_axis: int,
) -> np.ndarray:
    """
    Compute torso forward lean angle in sagittal plane.

    Measures the angle of the torso from vertical.

    Args:
        head: (N, 3) head positions
        torso: (N, 3) torso positions
        waist: (N, 3) waist positions
        lateral_axis: Index of lateral axis
        anterior_axis: Index of anterior axis

    Returns:
        Angle in degrees from vertical. Positive = forward lean.
    """
    vertical_axis = 3 - lateral_axis - anterior_axis

    # Project to sagittal plane
    head_2d = np.column_stack([head[:, anterior_axis], head[:, vertical_axis]])
    waist_2d = np.column_stack([waist[:, anterior_axis], waist[:, vertical_axis]])

    # Torso vector (waist to head)
    torso_vec = head_2d - waist_2d

    # Vertical reference (pointing up)
    vertical_ref = np.array([0, 1])

    # Normalize torso vector
    torso_len = np.linalg.norm(torso_vec, axis=1, keepdims=True)
    torso_unit = torso_vec / (torso_len + 1e-8)

    # Angle from vertical
    dot_product = np.sum(torso_unit * vertical_ref, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_product))

    # Determine sign: positive if leaning forward (anterior direction positive)
    forward_component = torso_unit[:, 0]  # Anterior component
    angles = np.where(forward_component > 0, angles, -angles)

    return angles


def assess_posture(
    filtered: "FilteredResultAdapter",
    coord_system: dict[str, str],
) -> PostureResult:
    """
    Assess posture from filtered skeleton data.

    Args:
        filtered: FilteredResultAdapter with positions dict
        coord_system: Coordinate system mapping

    Returns:
        PostureResult with assessment
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    lateral_axis = axis_map[coord_system["lateral"]]
    anterior_axis = axis_map[coord_system["anterior"]]

    # Get positions
    head = filtered.positions["head"]
    torso = filtered.positions["torso"]
    waist = filtered.positions["waist"]
    l_hip = filtered.positions["l_hip"]
    r_hip = filtered.positions["r_hip"]

    # Hip center
    hip_center = (l_hip + r_hip) / 2

    # Find valid frames
    valid_arch = ~(np.isnan(torso).any(axis=1) | np.isnan(waist).any(axis=1) | np.isnan(hip_center).any(axis=1))
    valid_lean = ~(np.isnan(head).any(axis=1) | np.isnan(torso).any(axis=1) | np.isnan(waist).any(axis=1))

    # Compute angles only for valid frames
    back_arch = np.full(len(torso), np.nan)
    torso_lean = np.full(len(head), np.nan)

    if np.any(valid_arch):
        back_arch[valid_arch] = compute_back_arch_angle(
            torso[valid_arch], waist[valid_arch], hip_center[valid_arch], lateral_axis, anterior_axis
        )
    if np.any(valid_lean):
        torso_lean[valid_lean] = compute_torso_lean_angle(
            head[valid_lean], torso[valid_lean], waist[valid_lean], lateral_axis, anterior_axis
        )

    # Uncertainty from bootstrap (if available)
    if filtered.positions_std is not None:
        pos_std = np.mean([
            filtered.positions_std["torso"].mean(),
            filtered.positions_std["waist"].mean(),
        ])
        angle_std = np.degrees(np.arctan2(pos_std, 200))
    else:
        angle_std = 3.0

    back_arch_lower = back_arch - 2 * angle_std
    back_arch_upper = back_arch + 2 * angle_std
    torso_lean_lower = torso_lean - 2 * angle_std
    torso_lean_upper = torso_lean + 2 * angle_std

    # Summary statistics (use nanmax/nanmean to handle NaN)
    max_back_arch = float(np.nanmax(back_arch)) if not np.all(np.isnan(back_arch)) else 0.0
    mean_back_arch = float(np.nanmean(back_arch)) if not np.all(np.isnan(back_arch)) else 0.0
    max_torso_lean = float(np.nanmax(torso_lean)) if not np.all(np.isnan(torso_lean)) else 0.0
    mean_torso_lean = float(np.nanmean(torso_lean)) if not np.all(np.isnan(torso_lean)) else 0.0

    return PostureResult(
        back_arch_angle=back_arch,
        back_arch_lower=back_arch_lower,
        back_arch_upper=back_arch_upper,
        torso_lean_angle=torso_lean,
        torso_lean_lower=torso_lean_lower,
        torso_lean_upper=torso_lean_upper,
        max_back_arch=max_back_arch,
        mean_back_arch=mean_back_arch,
        max_torso_lean=max_torso_lean,
        mean_torso_lean=mean_torso_lean,
    )
