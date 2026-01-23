"""Configuration for unified optimization-based motion filtering."""

from dataclasses import dataclass, field
from typing import Literal


# Skeleton topology from existing code
JOINT_NAMES = [
    "head",
    "torso",
    "waist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
]

# Joint name to index mapping
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Bone connections: (parent, child)
BONES = [
    ("head", "torso"),
    ("torso", "waist"),
    ("torso", "l_shoulder"),
    ("torso", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("waist", "l_hip"),
    ("waist", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]

# Bone indices for PyTorch: (parent_idx, child_idx)
BONE_INDICES = [(JOINT_INDEX[p], JOINT_INDEX[c]) for p, c in BONES]


@dataclass
class ROMConstraint:
    """Range of Motion constraint for an angle."""

    # Joints defining the angle: (endpoint1, vertex, endpoint2)
    joints: tuple[str, str, str]
    min_angle: float  # degrees
    max_angle: float  # degrees

    @property
    def joint_indices(self) -> tuple[int, int, int]:
        """Get joint indices for PyTorch."""
        return tuple(JOINT_INDEX[j] for j in self.joints)


# ROM constraints based on anatomical limits
ROM_CONSTRAINTS = [
    # Knee flexion (0=extended, 150=fully flexed)
    ROMConstraint(("l_hip", "l_knee", "l_ankle"), 0.0, 170.0),
    ROMConstraint(("r_hip", "r_knee", "r_ankle"), 0.0, 170.0),

    # Hip flexion (torso-hip-knee angle)
    ROMConstraint(("waist", "l_hip", "l_knee"), 30.0, 180.0),
    ROMConstraint(("waist", "r_hip", "r_knee"), 30.0, 180.0),

    # Elbow flexion
    ROMConstraint(("l_shoulder", "l_elbow", "l_wrist"), 10.0, 180.0),
    ROMConstraint(("r_shoulder", "r_elbow", "r_wrist"), 10.0, 180.0),

    # Shoulder angle (torso-shoulder-elbow)
    ROMConstraint(("torso", "l_shoulder", "l_elbow"), 5.0, 180.0),
    ROMConstraint(("torso", "r_shoulder", "r_elbow"), 5.0, 180.0),

    # Neck angle (head-torso-waist)
    ROMConstraint(("head", "torso", "waist"), 90.0, 180.0),

    # Back arch (torso-waist-hip_center) - using left hip as proxy
    ROMConstraint(("torso", "waist", "l_hip"), 60.0, 180.0),
    ROMConstraint(("torso", "waist", "r_hip"), 60.0, 180.0),
]


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing step."""

    # Spike detection (MAD-based)
    spike_threshold: float = 3.0  # Number of MADs for spike detection

    # Low-pass filter
    lowpass_cutoff_hz: float = 3.0  # Butterworth cutoff frequency (lowered from 6Hz)
    lowpass_order: int = 2

    # Frame rate
    fps: float = 30.0


@dataclass
class LossWeights:
    """Weights for each loss term in unified optimization."""

    # Data fitting (reference weight = 1.0)
    data: float = 1.0

    # Bone length constraint (strong)
    bone: float = 100.0

    # ROM constraint (moderate)
    rom: float = 50.0

    # Acceleration smoothness
    accel: float = 10.0  # increased from 5.0 -> 8.0 -> 10.0

    # Jerk smoothness
    jerk: float = 5.0  # increased from 2.0 -> 4.0 -> 5.0

    # Angular velocity limit
    velocity: float = 10.0

    # Tremor suppression (moving average deviation)
    tremor: float = 20.0

    # Direction consistency (velocity direction reversals)
    direction: float = 15.0

    # Spine alignment (head-torso-waist consistency)
    spine: float = 200.0


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    # Loss weights
    weights: LossWeights = field(default_factory=LossWeights)

    # Optimizer settings
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 0.01  # Initial learning rate (will decay)
    min_learning_rate: float = 0.001  # Minimum learning rate
    max_iterations: int = 50000  # Default: 50000 (early stopping usually triggers earlier)

    # Early stopping
    patience: int = 100  # Epochs without improvement
    min_improvement_rate: float = 0.0001  # 0.01% relative improvement required

    # Adaptive confidence
    use_adaptive_confidence: bool = True
    warmup_epochs: int = 10
    confidence_update_interval: int = 10
    min_confidence: float = 0.01

    # Angular velocity limit (degrees/frame at 30fps)
    max_angular_velocity: float = 15.0

    # Neck angle velocity limit (degrees/frame) - for spine alignment loss
    max_neck_velocity: float = 1.0


@dataclass
class UnifiedFilterConfig:
    """Complete configuration for unified filtering."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Verbosity
    verbose: bool = True
    log_interval: int = 20


# Preset configurations
def get_preset_config(preset: str) -> UnifiedFilterConfig:
    """Get preset configuration."""
    config = UnifiedFilterConfig()

    if preset == "default":
        pass  # Use defaults

    elif preset == "high-noise":
        # For very noisy data (Subject 4, 5)
        # More aggressive low-pass filter only (2Hz instead of 3Hz)
        # Loss weights remain the same - preprocessing handles noise level
        config.preprocessing.lowpass_cutoff_hz = 2.0

    return config
