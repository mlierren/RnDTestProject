"""Knee valgus/varus assessment for NASM overhead squat (Anterior View)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.preprocessing import FilteredResultAdapter


@dataclass
class KneeValgusResult:
    """Results of knee valgus assessment."""

    # Angle trajectories (in degrees)
    # Positive = valgus (knees inward), Negative = varus (knees outward)
    left_angle: np.ndarray  # (N,)
    right_angle: np.ndarray  # (N,)

    # Confidence intervals
    left_angle_lower: np.ndarray
    left_angle_upper: np.ndarray
    right_angle_lower: np.ndarray
    right_angle_upper: np.ndarray

    # Summary statistics
    left_max_valgus: float  # Maximum inward deviation
    right_max_valgus: float
    left_max_varus: float  # Maximum outward deviation
    right_max_varus: float


def compute_knee_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    lateral_axis: int,
    anterior_axis: int,
) -> np.ndarray:
    """
    Compute knee deviation angle in the frontal plane (anterior view).

    The angle measures how much the knee deviates from the line connecting
    hip to ankle when viewed from the front.

    Args:
        hip: (N, 3) hip positions
        knee: (N, 3) knee positions
        ankle: (N, 3) ankle positions
        lateral_axis: Index of lateral axis (0=x, 1=y, 2=z)
        anterior_axis: Index of anterior axis

    Returns:
        Angle in degrees. Positive = valgus (inward), Negative = varus (outward)
    """
    # Project to frontal plane (use lateral and vertical axes)
    # Vertical axis is the remaining one
    vertical_axis = 3 - lateral_axis - anterior_axis

    # Extract frontal plane coordinates
    hip_2d = np.column_stack([hip[:, lateral_axis], hip[:, vertical_axis]])
    knee_2d = np.column_stack([knee[:, lateral_axis], knee[:, vertical_axis]])
    ankle_2d = np.column_stack([ankle[:, lateral_axis], ankle[:, vertical_axis]])

    # Vector from hip to ankle (reference line)
    ref_vec = ankle_2d - hip_2d
    ref_len = np.linalg.norm(ref_vec, axis=1, keepdims=True)
    ref_unit = ref_vec / (ref_len + 1e-8)

    # Vector from hip to knee
    knee_vec = knee_2d - hip_2d

    # Project knee onto reference line
    proj_len = np.sum(knee_vec * ref_unit, axis=1, keepdims=True)
    knee_proj = hip_2d + proj_len * ref_unit

    # Perpendicular deviation
    deviation = knee_2d - knee_proj

    # Signed deviation (positive = toward midline = valgus for right leg)
    # For left leg: positive lateral = outward = varus
    # For right leg: positive lateral = inward = valgus
    lateral_dev = deviation[:, 0]  # Lateral component

    # Distance from hip to knee along reference line
    dist_along = proj_len.flatten()

    # Compute angle
    angles = np.degrees(np.arctan2(lateral_dev, dist_along + 1e-8))

    return angles


def assess_knee_valgus(
    filtered: "FilteredResultAdapter",
    coord_system: dict[str, str],
) -> KneeValgusResult:
    """
    Assess knee valgus/varus from filtered skeleton data.

    Args:
        filtered: FilteredResultAdapter with positions dict
        coord_system: Coordinate system mapping

    Returns:
        KneeValgusResult with assessment
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    lateral_axis = axis_map[coord_system["lateral"]]
    anterior_axis = axis_map[coord_system["anterior"]]

    # Get positions
    l_hip = filtered.positions["l_hip"]
    l_knee = filtered.positions["l_knee"]
    l_ankle = filtered.positions["l_ankle"]
    r_hip = filtered.positions["r_hip"]
    r_knee = filtered.positions["r_knee"]
    r_ankle = filtered.positions["r_ankle"]

    # Find valid frames (no NaN in any required joint)
    valid_left = ~(np.isnan(l_hip).any(axis=1) | np.isnan(l_knee).any(axis=1) | np.isnan(l_ankle).any(axis=1))
    valid_right = ~(np.isnan(r_hip).any(axis=1) | np.isnan(r_knee).any(axis=1) | np.isnan(r_ankle).any(axis=1))

    # Compute angles only for valid frames
    left_angle = np.full(len(l_hip), np.nan)
    right_angle = np.full(len(r_hip), np.nan)

    if np.any(valid_left):
        left_angle[valid_left] = compute_knee_angle(
            l_hip[valid_left], l_knee[valid_left], l_ankle[valid_left], lateral_axis, anterior_axis
        )
    if np.any(valid_right):
        right_angle[valid_right] = compute_knee_angle(
            r_hip[valid_right], r_knee[valid_right], r_ankle[valid_right], lateral_axis, anterior_axis
        )

    # For left leg, flip sign so positive = valgus
    left_angle = -left_angle

    # Uncertainty from bootstrap (if available)
    if filtered.positions_std is not None:
        # Estimate angle uncertainty from position uncertainty
        pos_std = np.mean([
            filtered.positions_std["l_knee"].mean(),
            filtered.positions_std["r_knee"].mean(),
        ])
        angle_std = np.degrees(np.arctan2(pos_std, 300))  # Rough estimate
    else:
        angle_std = 3.0  # Default uncertainty

    left_angle_lower = left_angle - 2 * angle_std
    left_angle_upper = left_angle + 2 * angle_std
    right_angle_lower = right_angle - 2 * angle_std
    right_angle_upper = right_angle + 2 * angle_std

    # Summary statistics (use nanmax/nanmin to handle NaN)
    left_max_valgus = float(np.nanmax(left_angle)) if not np.all(np.isnan(left_angle)) else 0.0
    right_max_valgus = float(np.nanmax(right_angle)) if not np.all(np.isnan(right_angle)) else 0.0
    left_max_varus = float(np.nanmin(left_angle)) if not np.all(np.isnan(left_angle)) else 0.0
    right_max_varus = float(np.nanmin(right_angle)) if not np.all(np.isnan(right_angle)) else 0.0

    return KneeValgusResult(
        left_angle=left_angle,
        right_angle=right_angle,
        left_angle_lower=left_angle_lower,
        left_angle_upper=left_angle_upper,
        right_angle_lower=right_angle_lower,
        right_angle_upper=right_angle_upper,
        left_max_valgus=left_max_valgus,
        right_max_valgus=right_max_valgus,
        left_max_varus=left_max_varus,
        right_max_varus=right_max_varus,
    )
