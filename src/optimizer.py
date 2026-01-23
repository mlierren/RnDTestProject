"""
Unified optimizer with adaptive confidence weighting.

Implements the optimization loop with:
- PyTorch autograd for gradient computation
- Adaptive confidence weighting based on residuals (MAD-based)
- Early stopping and learning rate scheduling
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from .config import OptimizationConfig, BONES
from .losses import compute_unified_loss


@dataclass
class OptimizationResult:
    """Result of optimization."""

    # Optimized positions: (n_frames, n_joints, 3)
    positions: np.ndarray

    # Final confidence weights
    confidence: np.ndarray

    # Loss history
    loss_history: list[dict[str, float]]

    # Number of iterations
    n_iterations: int

    # Converged flag
    converged: bool


class AdaptiveConfidenceOptimizer:
    """
    Optimizer with adaptive confidence weighting.

    Key idea: Use residual-based confidence to automatically
    downweight noisy frames during optimization.

    Epoch 0-9:   confidence = all 1.0 (warmup)
    Epoch 10:    Update confidence based on residuals
    Epoch 10-19: Optimize with new confidence
    Epoch 20:    Update confidence again
    ...
    """

    def __init__(
        self,
        config: OptimizationConfig,
        verbose: bool = True,
        log_interval: int = 20,
    ):
        self.config = config
        self.verbose = verbose
        self.log_interval = log_interval

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_confidence(
        self,
        x_current: torch.Tensor,
        x_original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-frame confidence using MAD-based soft confidence.

        Frames with large residuals (far from original) get lower confidence.

        Algorithm:
        1. Compute per-frame residual = ||x_current - x_original||
        2. Estimate robust spread using MAD (Median Absolute Deviation)
        3. Convert to z-score
        4. Apply sigmoid for soft thresholding

        Args:
            x_current: Current positions (n_frames, n_joints, 3)
            x_original: Original positions

        Returns:
            Confidence weights (n_frames,) in [min_conf, 1.0]
        """
        with torch.no_grad():
            # Per-frame residual: sum over joints and xyz
            residual = (x_current - x_original).pow(2).sum(dim=(1, 2)).sqrt()
            residual_np = residual.cpu().numpy()

            # Robust statistics using MAD
            median_res = np.median(residual_np)
            mad = np.median(np.abs(residual_np - median_res))

            # Convert MAD to approximate standard deviation
            sigma = 1.4826 * mad + 1e-6

            # Z-score
            z = (residual_np - median_res) / sigma

            # Soft confidence using sigmoid
            # z=0 -> conf=0.88, z=2 -> conf=0.5, z=4 -> conf=0.12
            confidence_np = 1.0 / (1.0 + np.exp(z - 2.0))

            # Clip to minimum confidence
            confidence_np = np.clip(confidence_np, self.config.min_confidence, 1.0)

            return torch.tensor(confidence_np, device=self.device, dtype=torch.float32)

    def optimize(
        self,
        x_original: np.ndarray,
        bone_lengths: dict[tuple[str, str], float],
        valid_mask: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, dict], None]] = None,
    ) -> OptimizationResult:
        """
        Run unified optimization with adaptive confidence.

        Args:
            x_original: Original positions (n_frames, n_joints, 3)
            bone_lengths: Reference bone lengths {(parent, child): length}
            valid_mask: Optional mask for valid frames (for initial confidence)
            callback: Optional callback(iteration, loss_dict) for logging

        Returns:
            OptimizationResult with optimized positions
        """
        n_frames = x_original.shape[0]

        # Convert to torch tensors
        x_orig_tensor = torch.tensor(
            x_original, device=self.device, dtype=torch.float32
        )
        x = x_orig_tensor.clone().requires_grad_(True)

        # Convert bone lengths to tensor (ordered by BONES)
        bone_length_list = [bone_lengths[bone] for bone in BONES]
        bone_length_tensor = torch.tensor(
            bone_length_list, device=self.device, dtype=torch.float32
        )

        # Initialize confidence
        if valid_mask is not None and self.config.use_adaptive_confidence:
            # Start with higher confidence for valid frames
            confidence = torch.tensor(
                np.where(valid_mask, 1.0, 0.5),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            confidence = torch.ones(n_frames, device=self.device, dtype=torch.float32)

        # Tracking
        loss_history = []
        best_loss = None  # Will be set after first iteration
        best_x = x.detach().clone()
        patience_counter = 0
        converged = False
        nan_recovery_count = 0
        max_nan_recoveries = 5  # Maximum NaN recovery attempts
        current_lr = self.config.learning_rate

        def create_optimizer_and_scheduler(lr):
            """Helper to create fresh optimizer and scheduler."""
            if self.config.optimizer == "adam":
                opt = torch.optim.Adam([x], lr=lr)
            elif self.config.optimizer == "sgd":
                opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.config.max_iterations,
                eta_min=self.config.min_learning_rate,
            )
            return opt, sched

        # Initial optimizer setup
        optimizer, scheduler = create_optimizer_and_scheduler(current_lr)

        # Optimization loop
        for epoch in range(self.config.max_iterations):
            # Update confidence every N epochs after warmup (MAD-based residual detection)
            if (
                self.config.use_adaptive_confidence
                and epoch >= self.config.warmup_epochs
                and epoch % self.config.confidence_update_interval == 0
            ):
                confidence = self.compute_confidence(x.detach(), x_orig_tensor)

                if self.verbose:
                    print(
                        f"  Epoch {epoch}: confidence updated - "
                        f"mean={confidence.mean():.3f}, min={confidence.min():.3f}"
                    )

            # Forward pass
            optimizer.zero_grad()
            loss, breakdown = compute_unified_loss(
                x,
                x_orig_tensor,
                bone_length_tensor,
                self.config.weights,
                confidence if self.config.use_adaptive_confidence else None,
                self.config.max_angular_velocity,
                max_neck_velocity=self.config.max_neck_velocity,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability (prevent NaN from gradient explosion)
            torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)

            # NaN check in loss - recover and continue if possible
            if torch.isnan(loss):
                nan_recovery_count += 1
                if nan_recovery_count > max_nan_recoveries:
                    if self.verbose:
                        print(f"  NaN detected in loss at epoch {epoch}, max recoveries exceeded, stopping")
                    x.data.copy_(best_x)
                    break

                # Recover: revert to best, halve learning rate, add small noise, reset optimizer
                current_lr *= 0.5
                if self.verbose:
                    print(f"  NaN detected in loss at epoch {epoch}, recovering (attempt {nan_recovery_count}/{max_nan_recoveries})")
                    print(f"    Reverting to best, reducing LR to {current_lr:.6f}, adding noise perturbation")
                x.data.copy_(best_x)
                # Add small noise to escape numerically unstable region
                noise_scale = 0.1 * nan_recovery_count  # Increase noise with each attempt
                x.data.add_(torch.randn_like(x) * noise_scale)
                optimizer, scheduler = create_optimizer_and_scheduler(current_lr)
                continue

            # Update
            optimizer.step()
            scheduler.step()  # CosineAnnealingLR: no loss argument needed

            # NaN check in x after update - recover and continue if possible
            if torch.isnan(x).any():
                nan_recovery_count += 1
                if nan_recovery_count > max_nan_recoveries:
                    if self.verbose:
                        print(f"  NaN detected in positions at epoch {epoch}, max recoveries exceeded, stopping")
                    x.data.copy_(best_x)
                    break

                # Recover: revert to best, halve learning rate, add small noise, reset optimizer
                current_lr *= 0.5
                if self.verbose:
                    print(f"  NaN detected in positions at epoch {epoch}, recovering (attempt {nan_recovery_count}/{max_nan_recoveries})")
                    print(f"    Reverting to best, reducing LR to {current_lr:.6f}, adding noise perturbation")
                x.data.copy_(best_x)
                # Add small noise to escape numerically unstable region
                noise_scale = 0.1 * nan_recovery_count  # Increase noise with each attempt
                x.data.add_(torch.randn_like(x) * noise_scale)
                optimizer, scheduler = create_optimizer_and_scheduler(current_lr)
                continue

            # Record history
            loss_history.append(breakdown)

            # Logging
            if self.verbose and epoch % self.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:4d} (lr={current_lr:.4f}): total={breakdown['total']:.1f} | "
                    f"bone={breakdown['bone']:.1f}, tremor={breakdown['tremor']:.1f}, "
                    f"dir={breakdown['direction']:.1f}, spine={breakdown['spine']:.1f}"
                )

            # Callback
            if callback is not None:
                callback(epoch, breakdown)

            # Early stopping check (relative improvement)
            current_loss = breakdown["total"]

            # Initialize best_loss on first iteration
            if best_loss is None:
                best_loss = current_loss
                best_x = x.detach().clone()
                patience_counter = 0
            else:
                improvement_threshold = best_loss * self.config.min_improvement_rate

                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= self.config.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"  No {self.config.min_improvement_rate*100:.2f}% improvement for {self.config.patience} epochs")
                    print(f"  Best loss: {best_loss:.2f}")
                converged = True
                break

        # Use best result
        final_positions = best_x.cpu().numpy()
        final_confidence = confidence.cpu().numpy()

        if self.verbose:
            print(f"Optimization complete: {len(loss_history)} iterations")
            print(f"  Final loss: {loss_history[-1]['total']:.2f}")
            print(f"  Best loss: {best_loss:.2f}")

        return OptimizationResult(
            positions=final_positions,
            confidence=final_confidence,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            converged=converged,
        )
