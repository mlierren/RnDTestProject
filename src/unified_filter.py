"""
Unified filtering pipeline.

Main entry point combining preprocessing + unified optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import UnifiedFilterConfig, BONES, get_preset_config
from .preprocessing import preprocess, PreprocessingResult
from .optimizer import AdaptiveConfidenceOptimizer, OptimizationResult


@dataclass
class UnifiedFilterResult:
    """Complete result of unified filtering."""

    # Filtered positions: (n_frames, n_joints, 3)
    positions: np.ndarray

    # Preprocessing result
    preprocessing: PreprocessingResult

    # Optimization result
    optimization: OptimizationResult

    # Reference bone lengths used
    bone_lengths: dict[tuple[str, str], float]


def unified_filter(
    positions: np.ndarray,
    config: Optional[UnifiedFilterConfig] = None,
    preset: Optional[str] = None,
) -> UnifiedFilterResult:
    """
    Apply unified filtering to motion capture data.

    Pipeline:
    1. Preprocessing (spike detection, interpolation, low-pass filter)
    2. Unified optimization (all constraints simultaneously)

    Args:
        positions: Raw positions (n_frames, n_joints, 3)
        config: Filter configuration (optional)
        preset: Preset name ("default", "high-noise", etc.)

    Returns:
        UnifiedFilterResult with filtered data and metadata
    """
    # Get configuration
    if config is None:
        if preset is not None:
            config = get_preset_config(preset)
        else:
            config = UnifiedFilterConfig()

    if config.verbose:
        print("=" * 50)
        print("Unified Filter")
        print("=" * 50)
        print(f"  Input shape: {positions.shape}")
        print(f"  Optimizer: {config.optimization.optimizer}")
        print(f"  Adaptive confidence: {config.optimization.use_adaptive_confidence}")

    # Phase 1: Preprocessing
    if config.verbose:
        print("\n[Phase 1] Preprocessing...")

    preprocess_result = preprocess(
        positions,
        bones=BONES,
        config=config.preprocessing,
    )

    if config.verbose:
        n_spikes = preprocess_result.spike_mask.sum()
        n_valid = preprocess_result.valid_mask.sum()
        n_frames = len(positions)
        print(f"  Spikes detected: {n_spikes}")
        print(f"  Valid frames: {n_valid}/{n_frames} ({100*n_valid/n_frames:.1f}%)")

    # Phase 2: Unified Optimization
    if config.verbose:
        print("\n[Phase 2] Unified Optimization...")
        print(f"  Weights: data={config.optimization.weights.data}, "
              f"bone={config.optimization.weights.bone}, "
              f"rom={config.optimization.weights.rom}")

    optimizer = AdaptiveConfidenceOptimizer(
        config.optimization,
        verbose=config.verbose,
        log_interval=config.log_interval,
    )

    opt_result = optimizer.optimize(
        preprocess_result.positions,
        preprocess_result.bone_lengths,
        valid_mask=preprocess_result.valid_mask,
    )

    if config.verbose:
        print("\n[Complete]")
        print(f"  Iterations: {opt_result.n_iterations}")
        print(f"  Converged: {opt_result.converged}")

    return UnifiedFilterResult(
        positions=opt_result.positions,
        preprocessing=preprocess_result,
        optimization=opt_result,
        bone_lengths=preprocess_result.bone_lengths,
    )


def compute_metrics(
    original: np.ndarray,
    filtered: np.ndarray,
    bone_lengths: dict[tuple[str, str], float],
) -> dict[str, float]:
    """
    Compute quality metrics comparing original and filtered data.

    Args:
        original: Original positions (n_frames, n_joints, 3)
        filtered: Filtered positions
        bone_lengths: Reference bone lengths

    Returns:
        Dictionary of metrics
    """
    from .config import JOINT_INDEX

    metrics = {}

    # RMSE (overall deviation from original)
    diff = filtered - original
    rmse = np.sqrt(np.mean(diff ** 2))
    metrics["rmse_mm"] = rmse

    # Bone length consistency (std of bone lengths over time)
    bone_std_total = 0.0
    for (parent, child), ref_length in bone_lengths.items():
        p_idx = JOINT_INDEX[parent]
        c_idx = JOINT_INDEX[child]
        lengths = np.linalg.norm(filtered[:, c_idx] - filtered[:, p_idx], axis=1)
        bone_std_total += np.std(lengths)
    metrics["bone_length_std_mm"] = bone_std_total / len(bone_lengths)

    # Smoothness (mean acceleration magnitude)
    accel = filtered[2:] - 2 * filtered[1:-1] + filtered[:-2]
    accel_mag = np.linalg.norm(accel, axis=2).mean()
    metrics["acceleration_mean"] = accel_mag

    # Original smoothness for comparison
    orig_accel = original[2:] - 2 * original[1:-1] + original[:-2]
    orig_accel_mag = np.linalg.norm(orig_accel, axis=2).mean()
    metrics["acceleration_improvement"] = orig_accel_mag / (accel_mag + 1e-8)

    return metrics
