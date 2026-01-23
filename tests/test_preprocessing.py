"""Tests for src.preprocessing module."""

import pytest
import numpy as np
from src.preprocessing import (
    detect_spikes_mad,
    interpolate_spikes,
    apply_butterworth_filter,
    preprocess,
)
from src.config import PreprocessingConfig, BONES


class TestSpikeDetection:
    """Test spike detection."""

    def test_no_spikes_in_clean_data(self):
        """Should detect no spikes in clean linear motion."""
        n_frames = 100
        # Create clean 1D data
        data = np.arange(n_frames, dtype=float)

        spike_mask = detect_spikes_mad(data, threshold=3.0)
        assert spike_mask.sum() == 0

    def test_detects_obvious_spike(self):
        """Should detect obvious position spike."""
        n_frames = 100
        data = np.arange(n_frames, dtype=float)

        # Add a huge spike at frame 50
        data[50] = 10000.0

        spike_mask = detect_spikes_mad(data, threshold=3.0)
        assert spike_mask[50] == True

    def test_returns_correct_shape(self):
        """Spike mask should have correct shape."""
        n_frames = 100
        data = np.random.randn(n_frames)
        spike_mask = detect_spikes_mad(data, threshold=3.0)
        assert spike_mask.shape == (n_frames,)
        assert spike_mask.dtype == bool


class TestInterpolation:
    """Test spike interpolation."""

    def test_interpolates_single_spike(self):
        """Should interpolate single spike frame."""
        n_frames = 10
        positions = np.zeros((n_frames, 3))
        for t in range(n_frames):
            positions[t, 0] = float(t) * 10.0

        spike_mask = np.zeros(n_frames, dtype=bool)
        spike_mask[5] = True  # Frame 5 is a spike

        interpolated = interpolate_spikes(positions, spike_mask)

        # Frame 5 should be interpolated between frames 4 and 6
        expected = (positions[4, 0] + positions[6, 0]) / 2
        assert interpolated[5, 0] == pytest.approx(expected, abs=0.1)

    def test_handles_no_spikes(self):
        """Should return same positions when no spikes."""
        positions = np.random.randn(10, 3)
        spike_mask = np.zeros(10, dtype=bool)

        interpolated = interpolate_spikes(positions, spike_mask)
        np.testing.assert_array_equal(interpolated, positions)


class TestButterworthFilter:
    """Test Butterworth low-pass filter."""

    def test_preserves_low_frequency(self):
        """Should preserve low frequency signals."""
        n_frames = 100
        fps = 30.0
        cutoff = 6.0

        # Create low frequency signal (0.5 Hz) - 2D array
        t = np.arange(n_frames) / fps
        data = np.zeros((n_frames, 1))
        data[:, 0] = np.sin(2 * np.pi * 0.5 * t) * 100

        filtered = apply_butterworth_filter(data, cutoff, fps, order=2)

        # Should be similar to original (low freq preserved)
        correlation = np.corrcoef(data[:, 0], filtered[:, 0])[0, 1]
        assert correlation > 0.95

    def test_attenuates_high_frequency(self):
        """Should attenuate high frequency noise."""
        n_frames = 100
        fps = 30.0
        cutoff = 3.0

        # Create signal with high frequency noise
        t = np.arange(n_frames) / fps
        low_freq = np.sin(2 * np.pi * 0.5 * t) * 100
        high_freq = np.sin(2 * np.pi * 10 * t) * 50  # 10 Hz noise

        data = np.zeros((n_frames, 1))
        data[:, 0] = low_freq + high_freq

        filtered = apply_butterworth_filter(data, cutoff, fps, order=2)

        # Filtered should be closer to low_freq than original
        orig_error = np.mean((data[:, 0] - low_freq) ** 2)
        filt_error = np.mean((filtered[:, 0] - low_freq) ** 2)
        assert filt_error < orig_error


class TestPreprocess:
    """Test full preprocessing pipeline."""

    def test_returns_preprocessing_result(self):
        """Should return PreprocessingResult with all fields."""
        positions = np.random.randn(100, 15, 3) * 100
        config = PreprocessingConfig()

        result = preprocess(positions, BONES, config)

        assert hasattr(result, "positions")
        assert hasattr(result, "spike_mask")
        assert hasattr(result, "valid_mask")
        assert hasattr(result, "bone_lengths")

    def test_output_shape_matches_input(self):
        """Output shape should match input."""
        positions = np.random.randn(100, 15, 3) * 100
        config = PreprocessingConfig()

        result = preprocess(positions, BONES, config)

        assert result.positions.shape == positions.shape

    def test_reduces_noise(self):
        """Preprocessing should reduce high-frequency noise."""
        n_frames = 100
        fps = 30.0

        # Create noisy signal
        np.random.seed(42)
        t = np.arange(n_frames) / fps
        clean_signal = np.sin(2 * np.pi * 0.5 * t)[:, None, None]
        clean_signal = np.broadcast_to(clean_signal, (n_frames, 15, 3)).copy() * 100
        # Add high-frequency noise
        noise = np.random.randn(n_frames, 15, 3) * 10
        noisy_signal = clean_signal + noise

        config = PreprocessingConfig(lowpass_cutoff_hz=3.0, fps=fps)
        result = preprocess(noisy_signal, BONES, config)

        # Filtered signal should have less high-frequency content
        # Check by comparing variance of second derivative (acceleration)
        noisy_accel = np.diff(noisy_signal, n=2, axis=0)
        filtered_accel = np.diff(result.positions, n=2, axis=0)

        noisy_var = np.var(noisy_accel)
        filtered_var = np.var(filtered_accel)

        assert filtered_var < noisy_var
