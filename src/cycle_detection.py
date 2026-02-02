"""Squat cycle detection using peak detection."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class SquatCycle:
    """A single squat cycle (descent + ascent)."""

    # Frame indices
    start_frame: int  # Standing position before descent
    bottom_frame: int  # Lowest point of squat
    end_frame: int  # Standing position after ascent

    # Vertical displacement at bottom (relative to start)
    depth: float

    # Duration in frames
    descent_frames: int
    ascent_frames: int

    @property
    def total_frames(self) -> int:
        """Total cycle duration in frames."""
        return self.end_frame - self.start_frame


def detect_squat_cycles(
    positions: dict[str, np.ndarray],
    vertical_axis: int = 1,
    min_depth: float = 100.0,
    min_distance: int = 30,
) -> list[SquatCycle]:
    """
    Detect squat cycles from filtered skeleton data.

    Uses hip center vertical position to detect squat cycles.
    A cycle consists of: standing -> descent -> bottom -> ascent -> standing

    Args:
        positions: Filtered joint positions (joint_name -> (N, 3))
        vertical_axis: Index of vertical axis (default 1 for Y)
        min_depth: Minimum vertical displacement to count as squat (mm)
        min_distance: Minimum frames between peaks

    Returns:
        List of detected SquatCycle objects
    """
    # Use hip center for cycle detection (most stable reference)
    l_hip = positions["l_hip"]
    r_hip = positions["r_hip"]
    hip_center = (l_hip + r_hip) / 2

    # Get vertical position (inverted so peaks = low points = squat bottoms)
    vertical = hip_center[:, vertical_axis]

    # Handle NaN values
    valid_mask = ~np.isnan(vertical)
    if not np.any(valid_mask):
        return []

    # Interpolate NaN values for peak detection
    x = np.arange(len(vertical))
    vertical_interp = np.interp(x, x[valid_mask], vertical[valid_mask])

    # Invert to find minima as peaks (squat bottoms)
    vertical_inv = -vertical_interp

    # Find peaks (squat bottoms)
    # prominence ensures we only detect significant squats
    peaks, properties = find_peaks(
        vertical_inv,
        distance=min_distance,
        prominence=min_depth,
    )

    if len(peaks) == 0:
        return []

    # Find standing positions (local maxima between squat bottoms)
    # These are the start/end points of each cycle
    standing_positions = []

    # Add start of recording if before first peak
    if peaks[0] > min_distance // 2:
        pre_peak_region = vertical_interp[: peaks[0]]
        standing_positions.append(int(np.argmax(pre_peak_region)))

    # Find maxima between consecutive peaks
    for i in range(len(peaks) - 1):
        region_start = peaks[i]
        region_end = peaks[i + 1]
        region = vertical_interp[region_start:region_end]
        local_max_idx = region_start + int(np.argmax(region))
        standing_positions.append(local_max_idx)

    # Add end of recording if after last peak
    if peaks[-1] < len(vertical_interp) - min_distance // 2:
        post_peak_region = vertical_interp[peaks[-1] :]
        standing_positions.append(peaks[-1] + int(np.argmax(post_peak_region)))

    # Create cycles
    cycles = []

    for i, bottom_frame in enumerate(peaks):
        # Find start (standing before this squat)
        start_frame = 0
        for sp in standing_positions:
            if sp < bottom_frame:
                start_frame = sp
            else:
                break

        # Find end (standing after this squat)
        end_frame = len(vertical_interp) - 1
        for sp in standing_positions:
            if sp > bottom_frame:
                end_frame = sp
                break

        # Calculate depth (vertical displacement from start to bottom)
        start_height = vertical_interp[start_frame]
        bottom_height = vertical_interp[bottom_frame]
        depth = start_height - bottom_height

        # Only include if depth meets threshold
        if depth >= min_depth:
            cycle = SquatCycle(
                start_frame=int(start_frame),
                bottom_frame=int(bottom_frame),
                end_frame=int(end_frame),
                depth=float(depth),
                descent_frames=int(bottom_frame - start_frame),
                ascent_frames=int(end_frame - bottom_frame),
            )
            cycles.append(cycle)

    return cycles


def analyze_cycles(
    cycles: list[SquatCycle],
    angle_trajectory: np.ndarray,
) -> dict:
    """
    Analyze angle measurements for each squat cycle.

    Args:
        cycles: List of detected squat cycles
        angle_trajectory: Angle values over time (N,)

    Returns:
        Dictionary with per-cycle and aggregate statistics
    """
    if not cycles:
        return {
            "n_cycles": 0,
            "cycles": [],
            "aggregate": None,
        }

    cycle_stats = []
    for i, cycle in enumerate(cycles):
        # Extract angle values for this cycle
        cycle_angles = angle_trajectory[cycle.start_frame : cycle.end_frame + 1]
        valid_angles = cycle_angles[~np.isnan(cycle_angles)]

        if len(valid_angles) == 0:
            continue

        # At bottom of squat (most relevant for assessment)
        bottom_angle = angle_trajectory[cycle.bottom_frame]

        stats = {
            "cycle_num": i + 1,
            "start_frame": int(cycle.start_frame),
            "bottom_frame": int(cycle.bottom_frame),
            "end_frame": int(cycle.end_frame),
            "depth_mm": float(cycle.depth),
            "max_angle": float(np.nanmax(cycle_angles)),
            "min_angle": float(np.nanmin(cycle_angles)),
            "mean_angle": float(np.nanmean(cycle_angles)),
            "bottom_angle": float(bottom_angle) if not np.isnan(bottom_angle) else None,
        }
        cycle_stats.append(stats)

    # Aggregate statistics across cycles
    if cycle_stats:
        all_max = [s["max_angle"] for s in cycle_stats]
        all_min = [s["min_angle"] for s in cycle_stats]
        all_bottom = [s["bottom_angle"] for s in cycle_stats if s["bottom_angle"] is not None]

        aggregate = {
            "max_angle_mean": float(np.mean(all_max)),
            "max_angle_std": float(np.std(all_max)),
            "min_angle_mean": float(np.mean(all_min)),
            "min_angle_std": float(np.std(all_min)),
            "bottom_angle_mean": float(np.mean(all_bottom)) if all_bottom else None,
            "bottom_angle_std": float(np.std(all_bottom)) if all_bottom else None,
        }
    else:
        aggregate = None

    return {
        "n_cycles": int(len(cycle_stats)),
        "cycles": cycle_stats,
        "aggregate": aggregate,
    }
