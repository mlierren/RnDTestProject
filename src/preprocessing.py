"""Preprocessing module for motion capture data."""

import numpy as np
from scipy import signal
from dataclasses import dataclass

from .config import PreprocessingConfig, JOINT_NAMES


@dataclass
class PreprocessingResult:
    """Result of preprocessing step."""

    # Preprocessed positions: (n_frames, n_joints, 3)
    positions: np.ndarray

    # Valid frame mask: frames without spikes (for data loss weighting)
    valid_mask: np.ndarray

    # Reference bone lengths computed from valid frames
    bone_lengths: dict[tuple[str, str], float]

    # Detected spike locations for debugging
    spike_mask: np.ndarray


def detect_spikes_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect spikes using Median Absolute Deviation (MAD).

    More robust than standard deviation for non-Gaussian noise.

    Args:
        data: 1D array of values
        threshold: Number of MADs for spike detection

    Returns:
        Boolean mask where True indicates a spike
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    # Convert MAD to approximate standard deviation
    # For normal distribution: std = 1.4826 * MAD
    sigma = 1.4826 * mad + 1e-8

    deviation = np.abs(data - median)
    return deviation > threshold * sigma


def interpolate_spikes(positions: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate spike positions from neighboring valid frames.

    Args:
        positions: (n_frames, 3) or (n_frames,) array
        spike_mask: Boolean mask where True indicates spike

    Returns:
        Positions with spikes interpolated
    """
    result = positions.copy()
    spike_indices = np.where(spike_mask)[0]

    if len(spike_indices) == 0:
        return result

    valid_indices = np.where(~spike_mask)[0]
    if len(valid_indices) < 2:
        return result

    # Handle multi-dimensional case
    if positions.ndim == 1:
        result[spike_mask] = np.interp(
            spike_indices, valid_indices, positions[valid_indices]
        )
    else:
        for dim in range(positions.shape[1]):
            result[spike_mask, dim] = np.interp(
                spike_indices, valid_indices, positions[valid_indices, dim]
            )

    return result


def apply_butterworth_filter(
    data: np.ndarray, cutoff_hz: float, fps: float, order: int = 2
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.

    Args:
        data: 1D or 2D array (frames, dims)
        cutoff_hz: Cutoff frequency in Hz
        fps: Sampling rate in frames per second
        order: Filter order

    Returns:
        Filtered data
    """
    nyquist = fps / 2
    normalized_cutoff = cutoff_hz / nyquist

    # Ensure cutoff is valid
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)

    b, a = signal.butter(order, normalized_cutoff, btype="low")

    # Use filtfilt for zero phase distortion
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        result = np.zeros_like(data)
        for dim in range(data.shape[1]):
            result[:, dim] = signal.filtfilt(b, a, data[:, dim])
        return result


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """
    Compute velocity (first derivative) using central differences.

    Args:
        positions: (n_frames, ...) array

    Returns:
        Velocity array with same shape (padded at boundaries)
    """
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / 2
    velocity[0] = positions[1] - positions[0]
    velocity[-1] = positions[-1] - positions[-2]
    return velocity


def preprocess(
    positions: np.ndarray,
    bones: list[tuple[str, str]],
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """
    Apply preprocessing pipeline to motion capture data.

    Pipeline:
    0. Fill NaN values with linear interpolation
    1. Spike detection (MAD-based)
    2. Linear interpolation of spikes
    3. Butterworth low-pass filter

    Args:
        positions: (n_frames, n_joints, 3) array of joint positions
        bones: List of (parent, child) joint name tuples
        config: Preprocessing configuration

    Returns:
        PreprocessingResult with preprocessed data and metadata
    """
    n_frames, n_joints, _ = positions.shape
    result = positions.copy()

    # Step 0: Fill NaN values first
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        for j in range(n_joints):
            for dim in range(3):
                col = result[:, j, dim]
                nan_indices = np.isnan(col)
                if np.any(nan_indices) and not np.all(nan_indices):
                    valid_indices = np.where(~nan_indices)[0]
                    col[nan_indices] = np.interp(
                        np.where(nan_indices)[0],
                        valid_indices,
                        col[valid_indices]
                    )
                    result[:, j, dim] = col

    # Track spikes for all joints
    all_spikes = np.zeros((n_frames, n_joints), dtype=bool)

    # Process each joint
    for j in range(n_joints):
        joint_pos = result[:, j, :]  # (n_frames, 3)

        # Detect spikes on each coordinate
        for dim in range(3):
            spikes = detect_spikes_mad(joint_pos[:, dim], config.spike_threshold)
            all_spikes[:, j] |= spikes

        # Also detect spikes on velocity magnitude
        velocity = compute_velocity(joint_pos)
        speed = np.linalg.norm(velocity, axis=1)
        velocity_spikes = detect_spikes_mad(speed, config.spike_threshold)
        all_spikes[:, j] |= velocity_spikes

    # Interpolate spikes
    for j in range(n_joints):
        if np.any(all_spikes[:, j]):
            result[:, j, :] = interpolate_spikes(result[:, j, :], all_spikes[:, j])

    # Apply Butterworth filter
    for j in range(n_joints):
        result[:, j, :] = apply_butterworth_filter(
            result[:, j, :],
            config.lowpass_cutoff_hz,
            config.fps,
            config.lowpass_order,
        )

    # Compute valid mask (frames without any spikes or NaN values)
    had_nan = np.any(np.any(nan_mask, axis=2), axis=1) if np.any(nan_mask) else np.zeros(n_frames, dtype=bool)
    valid_mask = ~np.any(all_spikes, axis=1) & ~had_nan

    # Compute reference bone lengths from valid frames only
    joint_index = {name: i for i, name in enumerate(JOINT_NAMES)}
    bone_lengths = {}

    for parent, child in bones:
        p_idx = joint_index[parent]
        c_idx = joint_index[child]

        # Use positions after interpolation but before heavy filtering
        # for bone length estimation
        lengths = np.linalg.norm(
            positions[:, c_idx, :] - positions[:, p_idx, :], axis=1
        )

        # Use median of valid frames (robust to outliers)
        if np.any(valid_mask):
            bone_lengths[(parent, child)] = float(np.nanmedian(lengths[valid_mask]))
        else:
            bone_lengths[(parent, child)] = float(np.nanmedian(lengths))

    return PreprocessingResult(
        positions=result,
        valid_mask=valid_mask,
        bone_lengths=bone_lengths,
        spike_mask=all_spikes,
    )


@dataclass
class FilteredResultAdapter:
    """
    Adapter to make unified filter output compatible with assessment functions.

    Provides the same interface (positions dict, positions_std dict) that
    assessment functions expect.
    """

    # Joint positions: joint_name -> (N, 3)
    positions: dict[str, np.ndarray]

    # Optional uncertainty (not used in unified optimizer, but required by interface)
    positions_std: dict[str, np.ndarray] | None = None

    # Reference bone lengths
    bone_lengths: dict[tuple[str, str], float] | None = None


def skeleton_to_array(skeleton) -> np.ndarray:
    """
    Convert SkeletonSequence to numpy array.

    Args:
        skeleton: SkeletonSequence object from src.data_loader

    Returns:
        (n_frames, n_joints, 3) array
    """
    n_frames = skeleton.n_frames
    n_joints = len(JOINT_NAMES)

    positions = np.zeros((n_frames, n_joints, 3))
    for i, joint_name in enumerate(JOINT_NAMES):
        positions[:, i, :] = skeleton.get_joint_positions(joint_name)

    return positions


def array_to_skeleton(positions: np.ndarray, original_skeleton):
    """
    Convert numpy array back to SkeletonSequence.

    Args:
        positions: (n_frames, n_joints, 3) array
        original_skeleton: Original SkeletonSequence to copy metadata from

    Returns:
        New SkeletonSequence with updated positions
    """
    from .data_loader import SkeletonSequence, Joint

    joints = {}
    for i, joint_name in enumerate(JOINT_NAMES):
        joints[joint_name] = Joint(
            x=positions[:, i, 0],
            y=positions[:, i, 1],
            z=positions[:, i, 2],
        )

    return SkeletonSequence(
        timestamps=original_skeleton.timestamps,
        joints=joints,
        subject_id=original_skeleton.subject_id,
    )


def skeleton_to_positions_dict(skeleton) -> dict[str, np.ndarray]:
    """
    Convert SkeletonSequence to positions dict (for cycle detection, assessment).

    Args:
        skeleton: SkeletonSequence object

    Returns:
        Dict mapping joint_name to (n_frames, 3) array
    """
    return {
        joint_name: skeleton.get_joint_positions(joint_name)
        for joint_name in JOINT_NAMES
    }


def create_filtered_result_adapter(skeleton, bone_lengths=None) -> FilteredResultAdapter:
    """
    Create a FilteredResultAdapter from a SkeletonSequence.

    Args:
        skeleton: SkeletonSequence object
        bone_lengths: Optional bone lengths dict

    Returns:
        FilteredResultAdapter compatible with assessment functions
    """
    positions = skeleton_to_positions_dict(skeleton)
    return FilteredResultAdapter(
        positions=positions,
        positions_std=None,
        bone_lengths=bone_lengths,
    )


