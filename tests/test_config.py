"""Tests for src.config module."""

import pytest
from src.config import (
    JOINT_NAMES,
    JOINT_INDEX,
    BONES,
    BONE_INDICES,
    ROMConstraint,
    ROM_CONSTRAINTS,
    PreprocessingConfig,
    LossWeights,
    OptimizationConfig,
    UnifiedFilterConfig,
    get_preset_config,
)


class TestSkeletonTopology:
    """Test skeleton topology definitions."""

    def test_joint_names_count(self):
        """Should have 15 joints."""
        assert len(JOINT_NAMES) == 15

    def test_joint_index_mapping(self):
        """Joint index mapping should be correct."""
        assert JOINT_INDEX["head"] == 0
        assert JOINT_INDEX["torso"] == 1
        assert JOINT_INDEX["l_ankle"] == 11
        assert JOINT_INDEX["r_ankle"] == 14

    def test_bones_count(self):
        """Should have 14 bones."""
        assert len(BONES) == 14

    def test_bone_indices_match_bones(self):
        """Bone indices should match bone definitions."""
        assert len(BONE_INDICES) == len(BONES)
        for i, (parent, child) in enumerate(BONES):
            p_idx, c_idx = BONE_INDICES[i]
            assert p_idx == JOINT_INDEX[parent]
            assert c_idx == JOINT_INDEX[child]


class TestROMConstraint:
    """Test ROM constraint definitions."""

    def test_rom_constraints_exist(self):
        """Should have ROM constraints defined."""
        assert len(ROM_CONSTRAINTS) > 0

    def test_rom_constraint_joint_indices(self):
        """ROM constraint should return correct joint indices."""
        knee_constraint = ROM_CONSTRAINTS[0]  # Left knee
        indices = knee_constraint.joint_indices
        assert len(indices) == 3
        assert indices[0] == JOINT_INDEX["l_hip"]
        assert indices[1] == JOINT_INDEX["l_knee"]
        assert indices[2] == JOINT_INDEX["l_ankle"]

    def test_rom_constraint_angles(self):
        """ROM constraints should have valid angle ranges."""
        for constraint in ROM_CONSTRAINTS:
            assert constraint.min_angle < constraint.max_angle
            assert constraint.min_angle >= 0
            assert constraint.max_angle <= 180


class TestPresets:
    """Test configuration presets."""

    def test_default_preset(self):
        """Default preset should use standard values."""
        config = get_preset_config("default")
        assert config.preprocessing.lowpass_cutoff_hz == 3.0
        assert config.optimization.weights.bone == 100.0

    def test_high_noise_preset(self):
        """High-noise preset should have lower cutoff frequency."""
        config = get_preset_config("high-noise")
        assert config.preprocessing.lowpass_cutoff_hz == 2.0
        # Weights should remain the same as default
        assert config.optimization.weights.bone == 100.0

    def test_unknown_preset_returns_default(self):
        """Unknown preset should return default config."""
        config = get_preset_config("unknown")
        assert config.preprocessing.lowpass_cutoff_hz == 3.0


class TestLossWeights:
    """Test loss weight configuration."""

    def test_default_weights(self):
        """Default weights should be set correctly."""
        weights = LossWeights()
        assert weights.data == 1.0
        assert weights.bone == 100.0
        assert weights.rom == 50.0
        assert weights.accel == 10.0
        assert weights.jerk == 5.0
        assert weights.tremor == 20.0
        assert weights.direction == 15.0


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_config(self):
        """Default optimization config should be valid."""
        config = OptimizationConfig()
        assert config.optimizer in ["adam", "sgd"]
        assert config.learning_rate > 0
        assert config.max_iterations == 50000
        assert config.patience > 0
