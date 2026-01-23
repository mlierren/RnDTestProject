"""
Loss functions for unified optimization.

All losses are differentiable and implemented in PyTorch for autograd.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from .config import BONE_INDICES, ROM_CONSTRAINTS, LossWeights


def compute_data_loss(
    x: torch.Tensor,
    x_original: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Data fitting loss: weighted squared difference from original.

    L_data = sum_i(confidence_i * ||x_i - x_orig_i||^2)

    Args:
        x: Current positions (n_frames, n_joints, 3)
        x_original: Original positions (n_frames, n_joints, 3)
        confidence: Per-frame confidence weights (n_frames,), default all 1.0

    Returns:
        Scalar loss value
    """
    diff_sq = (x - x_original).pow(2).sum(dim=(1, 2))  # (n_frames,)

    if confidence is not None:
        # Weighted sum
        loss = (confidence * diff_sq).sum()
    else:
        loss = diff_sq.sum()

    return loss


def compute_bone_loss(
    x: torch.Tensor,
    bone_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Bone length constraint loss: penalize deviation from reference lengths.

    L_bone = sum_bones sum_frames (||child - parent|| - L_ref)^2

    Args:
        x: Current positions (n_frames, n_joints, 3)
        bone_lengths: Reference bone lengths (n_bones,)

    Returns:
        Scalar loss value
    """
    total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for i, (parent_idx, child_idx) in enumerate(BONE_INDICES):
        parent_pos = x[:, parent_idx, :]  # (n_frames, 3)
        child_pos = x[:, child_idx, :]

        # Current bone length
        current_length = torch.norm(child_pos - parent_pos, dim=1)  # (n_frames,)

        # Reference length
        ref_length = bone_lengths[i]

        # Squared error
        total_loss = total_loss + ((current_length - ref_length) ** 2).sum()

    return total_loss


def compute_angle(
    p1: torch.Tensor, vertex: torch.Tensor, p2: torch.Tensor
) -> torch.Tensor:
    """
    Compute angle at vertex between vectors (vertex->p1) and (vertex->p2).

    Args:
        p1: First endpoint (n_frames, 3)
        vertex: Vertex point (n_frames, 3)
        p2: Second endpoint (n_frames, 3)

    Returns:
        Angles in degrees (n_frames,)
    """
    v1 = p1 - vertex
    v2 = p2 - vertex

    # Normalize
    v1_norm = torch.norm(v1, dim=1, keepdim=True).clamp(min=1e-8)
    v2_norm = torch.norm(v2, dim=1, keepdim=True).clamp(min=1e-8)

    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm

    # Dot product -> angle
    cos_angle = (v1_unit * v2_unit).sum(dim=1).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_angle)

    return torch.rad2deg(angle_rad)


def compute_rom_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Range of Motion constraint loss: penalize angles outside ROM.

    L_rom = sum_joints sum_frames [
        ReLU(theta_min - theta)^2 +
        ReLU(theta - theta_max)^2
    ]

    Uses soft constraints (ReLU) so no penalty within ROM.

    Args:
        x: Current positions (n_frames, n_joints, 3)

    Returns:
        Scalar loss value
    """
    total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for constraint in ROM_CONSTRAINTS:
        idx1, idx_vertex, idx2 = constraint.joint_indices

        p1 = x[:, idx1, :]
        vertex = x[:, idx_vertex, :]
        p2 = x[:, idx2, :]

        angle = compute_angle(p1, vertex, p2)  # (n_frames,)

        # Penalize below minimum
        below_min = F.relu(constraint.min_angle - angle)
        total_loss = total_loss + (below_min ** 2).sum()

        # Penalize above maximum
        above_max = F.relu(angle - constraint.max_angle)
        total_loss = total_loss + (above_max ** 2).sum()

    return total_loss


def compute_accel_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Acceleration (2nd derivative) smoothness loss.

    L_accel = sum_{t=2}^{T} ||x_t - 2*x_{t-1} + x_{t-2}||^2

    Minimizing this encourages smooth motion (linear velocity changes).

    Args:
        x: Current positions (n_frames, n_joints, 3)

    Returns:
        Scalar loss value
    """
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Second derivative (acceleration)
    accel = x[2:] - 2 * x[1:-1] + x[:-2]  # (n_frames-2, n_joints, 3)

    return (accel ** 2).sum()


def compute_jerk_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Jerk (3rd derivative) smoothness loss.

    L_jerk = sum_{t=3}^{T} ||x_t - 3*x_{t-1} + 3*x_{t-2} - x_{t-3}||^2

    Minimizing this prevents sudden direction changes.

    Args:
        x: Current positions (n_frames, n_joints, 3)

    Returns:
        Scalar loss value
    """
    if x.shape[0] < 4:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Third derivative (jerk)
    jerk = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]  # (n_frames-3, n_joints, 3)

    return (jerk ** 2).sum()


def compute_tremor_loss(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """
    Tremor loss: penalize deviation from moving average.

    Intuition: Smooth movements are close to their moving average.
               Tremor/jitter deviates from the moving average.

    L_tremor = sum ||x - moving_avg(x)||^2

    Args:
        x: Current positions (n_frames, n_joints, 3)
        window: Moving average window size

    Returns:
        Scalar loss value
    """
    if x.shape[0] < window:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    n_frames, n_joints, _ = x.shape

    # Create averaging kernel
    kernel = torch.ones(window, device=x.device, dtype=x.dtype) / window

    # Reshape for conv1d: (n_joints * 3, 1, n_frames)
    x_reshaped = x.permute(1, 2, 0).reshape(-1, 1, n_frames)

    # Apply moving average via 1D convolution
    # padding='same' equivalent: pad to maintain size
    pad_size = window // 2
    x_padded = F.pad(x_reshaped, (pad_size, window - 1 - pad_size), mode='replicate')
    smoothed = F.conv1d(x_padded, kernel.view(1, 1, -1))

    # Reshape back to (n_frames, n_joints, 3)
    smoothed = smoothed.reshape(n_joints, 3, n_frames).permute(2, 0, 1)

    # Deviation from moving average
    deviation = (x - smoothed).pow(2).sum()

    return deviation


def compute_direction_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Direction consistency loss: penalize rapid velocity direction changes.

    Intuition:
        Normal motion: → → → → (consistent direction)
        Tremor:        → ← → ← (rapid direction reversals)

    When consecutive velocity vectors point in opposite directions (cos < 0),
    apply penalty.

    L_direction = sum ReLU(-cos_similarity)^2

    Args:
        x: Current positions (n_frames, n_joints, 3)

    Returns:
        Scalar loss value
    """
    if x.shape[0] < 3:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Velocity vectors
    velocity = x[1:] - x[:-1]  # (n_frames-1, n_joints, 3)

    # Consecutive velocity pairs
    v1 = velocity[:-1]  # (n_frames-2, n_joints, 3)
    v2 = velocity[1:]   # (n_frames-2, n_joints, 3)

    # Dot product (direction similarity)
    dot_product = (v1 * v2).sum(dim=-1)  # (n_frames-2, n_joints)

    # Magnitudes
    speed1 = torch.norm(v1, dim=-1).clamp(min=1e-8)  # (n_frames-2, n_joints)
    speed2 = torch.norm(v2, dim=-1).clamp(min=1e-8)

    # Cosine similarity: +1 = same direction, -1 = opposite
    cos_sim = dot_product / (speed1 * speed2)

    # Penalize direction reversals (cos < 0)
    # cos = 1 (same direction) → penalty 0
    # cos = -1 (opposite) → penalty max
    direction_penalty = F.relu(-cos_sim).pow(2).sum()

    return direction_penalty


def compute_velocity_loss(
    x: torch.Tensor, max_angular_velocity: float = 15.0
) -> torch.Tensor:
    """
    Angular velocity limit loss: penalize unrealistic angular velocities.

    L_velocity = sum_joints sum_frames ReLU(|dtheta/dt| - omega_max)^2

    Args:
        x: Current positions (n_frames, n_joints, 3)
        max_angular_velocity: Maximum allowed degrees per frame

    Returns:
        Scalar loss value
    """
    total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for constraint in ROM_CONSTRAINTS:
        idx1, idx_vertex, idx2 = constraint.joint_indices

        p1 = x[:, idx1, :]
        vertex = x[:, idx_vertex, :]
        p2 = x[:, idx2, :]

        angles = compute_angle(p1, vertex, p2)  # (n_frames,)

        # Angular velocity (degrees per frame)
        angular_velocity = torch.abs(angles[1:] - angles[:-1])

        # Penalize excessive velocity
        excessive = F.relu(angular_velocity - max_angular_velocity)
        total_loss = total_loss + (excessive ** 2).sum()

    return total_loss


def compute_spine_alignment_loss(
    x: torch.Tensor,
    max_neck_velocity: float = 2.0,
    coord_system: dict = None,
) -> torch.Tensor:
    """
    Spine alignment loss: penalize unrealistic head-torso-waist movement.

    This loss addresses the issue where head and waist move independently,
    causing unrealistic torso lean variations (observed in Subject 4).

    Three components:
    1. Neck angle velocity limit: Prevent rapid neck angle changes
    2. Torso vector direction consistency: Penalize sudden changes in head-waist direction
    3. Sagittal plane torso lean velocity: Directly constrain torso lean angle changes

    Args:
        x: Current positions (n_frames, n_joints, 3)
        max_neck_velocity: Maximum allowed neck angle change per frame (degrees)
        coord_system: Optional coordinate system mapping (default: y=vertical, z=anterior)

    Returns:
        Scalar loss value
    """
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Joint indices: head=0, torso=1, waist=2
    head = x[:, 0, :]   # (n_frames, 3)
    torso = x[:, 1, :]
    waist = x[:, 2, :]

    # Component 1: Neck angle velocity constraint
    # Compute neck angle (head-torso-waist)
    neck_angle = compute_angle(head, torso, waist)  # (n_frames,)

    # Neck angular velocity
    neck_velocity = torch.abs(neck_angle[1:] - neck_angle[:-1])

    # Penalize excessive neck velocity (soft constraint)
    neck_penalty = F.relu(neck_velocity - max_neck_velocity).pow(2).sum()

    # Component 2: Torso vector direction consistency (3D)
    # Vector from waist to head (torso direction)
    torso_vec = head - waist  # (n_frames, 3)

    # Normalize to unit vector
    torso_len = torch.norm(torso_vec, dim=1, keepdim=True).clamp(min=1e-8)
    torso_unit = torso_vec / torso_len

    # Direction change between consecutive frames
    dir_dot = (torso_unit[:-1] * torso_unit[1:]).sum(dim=1)  # (n_frames-1,)

    # Penalize direction changes
    direction_penalty = (1 - dir_dot).pow(2).sum()

    # Component 3: Sagittal plane torso lean velocity constraint
    # Default: y=vertical (axis 1), z=anterior (axis 2)
    vertical_axis = 1
    anterior_axis = 2

    # Extract 2D coordinates in sagittal plane
    head_2d = torch.stack([head[:, anterior_axis], head[:, vertical_axis]], dim=1)
    waist_2d = torch.stack([waist[:, anterior_axis], waist[:, vertical_axis]], dim=1)

    # Torso vector in sagittal plane
    torso_2d = head_2d - waist_2d  # (n_frames, 2)

    # Normalize
    torso_2d_len = torch.norm(torso_2d, dim=1, keepdim=True).clamp(min=1e-8)
    torso_2d_unit = torso_2d / torso_2d_len

    # Vertical reference (pointing up: [0, 1])
    vertical_ref = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)

    # Compute torso lean angle from vertical (using atan2 for signed angle)
    # angle = atan2(anterior_component, vertical_component)
    torso_lean = torch.atan2(torso_2d_unit[:, 0], torso_2d_unit[:, 1])  # radians
    torso_lean_deg = torch.rad2deg(torso_lean)

    # Torso lean angular velocity
    lean_velocity = torch.abs(torso_lean_deg[1:] - torso_lean_deg[:-1])

    # Penalize excessive torso lean velocity (max 0.5 deg/frame)
    max_lean_velocity = 0.5
    lean_penalty = F.relu(lean_velocity - max_lean_velocity).pow(2).sum()

    # Combine components with appropriate scaling
    # neck_penalty: degrees^2
    # direction_penalty: dimensionless (0-4 range per frame)
    # lean_penalty: degrees^2
    total_loss = neck_penalty + 10.0 * direction_penalty + 50.0 * lean_penalty

    return total_loss


def compute_unified_loss(
    x: torch.Tensor,
    x_original: torch.Tensor,
    bone_lengths: torch.Tensor,
    weights: LossWeights,
    confidence: Optional[torch.Tensor] = None,
    max_angular_velocity: float = 15.0,
    tremor_window: int = 5,
    max_neck_velocity: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute unified loss combining all constraints.

    L_total = w_data * L_data + w_bone * L_bone + w_rom * L_rom
            + w_accel * L_accel + w_jerk * L_jerk + w_velocity * L_velocity
            + w_tremor * L_tremor + w_direction * L_direction + w_spine * L_spine

    Args:
        x: Current positions (n_frames, n_joints, 3)
        x_original: Original positions
        bone_lengths: Reference bone lengths
        weights: Loss term weights
        confidence: Per-frame confidence weights
        max_angular_velocity: Max angular velocity limit
        tremor_window: Window size for tremor moving average
        max_neck_velocity: Max neck angle change per frame (degrees)

    Returns:
        Tuple of (total_loss, dict of individual losses)
    """
    # Compute individual losses
    l_data = compute_data_loss(x, x_original, confidence)
    l_bone = compute_bone_loss(x, bone_lengths)
    l_rom = compute_rom_loss(x)
    l_accel = compute_accel_loss(x)
    l_jerk = compute_jerk_loss(x)
    l_velocity = compute_velocity_loss(x, max_angular_velocity)
    l_tremor = compute_tremor_loss(x, tremor_window)
    l_direction = compute_direction_loss(x)
    l_spine = compute_spine_alignment_loss(x, max_neck_velocity)

    # Weighted sum
    total = (
        weights.data * l_data
        + weights.bone * l_bone
        + weights.rom * l_rom
        + weights.accel * l_accel
        + weights.jerk * l_jerk
        + weights.velocity * l_velocity
        + weights.tremor * l_tremor
        + weights.direction * l_direction
        + weights.spine * l_spine
    )

    # Return breakdown for logging
    breakdown = {
        "data": l_data.item(),
        "bone": l_bone.item(),
        "rom": l_rom.item(),
        "accel": l_accel.item(),
        "jerk": l_jerk.item(),
        "velocity": l_velocity.item(),
        "tremor": l_tremor.item(),
        "direction": l_direction.item(),
        "spine": l_spine.item(),
        "total": total.item(),
    }

    return total, breakdown
