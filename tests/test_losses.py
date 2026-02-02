"""Tests for src.losses module."""

import pytest
import torch
import numpy as np
from src.losses import (
    compute_data_loss,
    compute_bone_loss,
    compute_angle,
    compute_rom_loss,
    compute_accel_loss,
    compute_jerk_loss,
    compute_tremor_loss,
    compute_direction_loss,
    compute_unified_loss,
)
from src.config import LossWeights, BONE_INDICES


class TestDataLoss:
    """Test data fitting loss."""

    def test_zero_loss_for_identical(self):
        """Data loss should be zero when positions match."""
        x = torch.randn(10, 15, 3)
        loss = compute_data_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_for_different(self):
        """Data loss should be positive when positions differ."""
        x = torch.randn(10, 15, 3)
        x_orig = torch.randn(10, 15, 3)
        loss = compute_data_loss(x, x_orig)
        assert loss.item() > 0

    def test_confidence_weighting(self):
        """Confidence should weight the loss."""
        x = torch.ones(10, 15, 3)
        x_orig = torch.zeros(10, 15, 3)

        # All confidence = 1
        conf_all = torch.ones(10)
        loss_all = compute_data_loss(x, x_orig, conf_all)

        # Half confidence = 0
        conf_half = torch.cat([torch.ones(5), torch.zeros(5)])
        loss_half = compute_data_loss(x, x_orig, conf_half)

        assert loss_half.item() < loss_all.item()


class TestBoneLoss:
    """Test bone length constraint loss."""

    def test_zero_loss_for_correct_lengths(self):
        """Bone loss should be zero when lengths match reference."""
        n_frames = 10
        n_joints = 15

        # Create positions with known bone lengths
        x = torch.zeros(n_frames, n_joints, 3)
        ref_lengths = []

        for p_idx, c_idx in BONE_INDICES:
            # Set child position to create unit length bone
            x[:, c_idx, 0] = x[:, p_idx, 0] + 100.0  # 100mm bone
            ref_lengths.append(100.0)

        bone_lengths = torch.tensor(ref_lengths)
        loss = compute_bone_loss(x, bone_lengths)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_loss_for_wrong_lengths(self):
        """Bone loss should be positive when lengths don't match."""
        x = torch.randn(10, 15, 3) * 100
        bone_lengths = torch.ones(14) * 50.0  # Wrong lengths
        loss = compute_bone_loss(x, bone_lengths)
        assert loss.item() > 0


class TestAngleComputation:
    """Test angle computation."""

    def test_right_angle(self):
        """Should compute 90 degrees correctly."""
        # Create L-shape: vertex at origin, p1 along x, p2 along y
        p1 = torch.tensor([[1.0, 0.0, 0.0]])
        vertex = torch.tensor([[0.0, 0.0, 0.0]])
        p2 = torch.tensor([[0.0, 1.0, 0.0]])

        angle = compute_angle(p1, vertex, p2)
        assert angle.item() == pytest.approx(90.0, abs=0.1)

    def test_straight_angle(self):
        """Should compute 180 degrees for straight line."""
        p1 = torch.tensor([[1.0, 0.0, 0.0]])
        vertex = torch.tensor([[0.0, 0.0, 0.0]])
        p2 = torch.tensor([[-1.0, 0.0, 0.0]])

        angle = compute_angle(p1, vertex, p2)
        assert angle.item() == pytest.approx(180.0, abs=0.1)


class TestAccelLoss:
    """Test acceleration smoothness loss."""

    def test_zero_for_linear_motion(self):
        """Acceleration loss should be zero for linear motion."""
        # Linear motion: positions increase linearly
        n_frames = 20
        x = torch.zeros(n_frames, 15, 3)
        for t in range(n_frames):
            x[t, :, 0] = float(t)  # Linear in x

        loss = compute_accel_loss(x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_jerky_motion(self):
        """Acceleration loss should be positive for jerky motion."""
        x = torch.randn(20, 15, 3)
        loss = compute_accel_loss(x)
        assert loss.item() > 0


class TestTremorLoss:
    """Test tremor (moving average deviation) loss."""

    def test_low_for_smooth_signal(self):
        """Tremor loss should be low for smooth signals."""
        n_frames = 50
        x = torch.zeros(n_frames, 15, 3)
        # Smooth sine wave
        for t in range(n_frames):
            x[t, :, 0] = np.sin(2 * np.pi * t / 20) * 10

        loss = compute_tremor_loss(x, window=5)
        # Should be relatively small for smooth signal
        assert loss.item() < 1000

    def test_high_for_noisy_signal(self):
        """Tremor loss should be high for noisy signals."""
        n_frames = 50
        x = torch.randn(n_frames, 15, 3) * 10  # Random noise

        loss = compute_tremor_loss(x, window=5)
        assert loss.item() > 100


class TestDirectionLoss:
    """Test direction consistency loss."""

    def test_zero_for_consistent_direction(self):
        """Direction loss should be zero for consistent velocity direction."""
        n_frames = 20
        x = torch.zeros(n_frames, 15, 3)
        for t in range(n_frames):
            x[t, :, 0] = float(t)  # Always moving in +x direction

        loss = compute_direction_loss(x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_zigzag_motion(self):
        """Direction loss should be positive for zigzag motion."""
        n_frames = 20
        x = torch.zeros(n_frames, 15, 3)
        for t in range(n_frames):
            # Zigzag: alternating direction
            x[t, :, 0] = float(t) if t % 2 == 0 else float(t - 2)

        loss = compute_direction_loss(x)
        assert loss.item() > 0


class TestUnifiedLoss:
    """Test unified loss computation."""

    def test_returns_total_and_breakdown(self):
        """Unified loss should return total and breakdown dict."""
        x = torch.randn(20, 15, 3)
        x_orig = x.clone()
        bone_lengths = torch.ones(14) * 100.0
        weights = LossWeights()

        total, breakdown = compute_unified_loss(x, x_orig, bone_lengths, weights)

        assert isinstance(total, torch.Tensor)
        assert isinstance(breakdown, dict)
        assert "total" in breakdown
        assert "data" in breakdown
        assert "bone" in breakdown
        assert "tremor" in breakdown
        assert "direction" in breakdown

    def test_total_equals_weighted_sum(self):
        """Total should equal weighted sum of components."""
        x = torch.randn(20, 15, 3)
        x_orig = torch.randn(20, 15, 3)
        bone_lengths = torch.ones(14) * 100.0
        weights = LossWeights()

        total, breakdown = compute_unified_loss(x, x_orig, bone_lengths, weights)

        expected = (
            weights.data * breakdown["data"]
            + weights.bone * breakdown["bone"]
            + weights.rom * breakdown["rom"]
            + weights.accel * breakdown["accel"]
            + weights.jerk * breakdown["jerk"]
            + weights.velocity * breakdown["velocity"]
            + weights.tremor * breakdown["tremor"]
            + weights.direction * breakdown["direction"]
            + weights.spine * breakdown["spine"]
        )

        assert breakdown["total"] == pytest.approx(expected, rel=1e-4)
