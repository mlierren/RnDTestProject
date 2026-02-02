"""Data loading and skeleton structure for motion capture data."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import JOINT_NAMES, BONES


class Joint(NamedTuple):
    """3D joint position."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return as (N, 3) array."""
        return np.column_stack([self.x, self.y, self.z])


@dataclass
class SkeletonSequence:
    """A sequence of skeleton poses over time."""

    timestamps: np.ndarray
    joints: dict[str, Joint]
    subject_id: int

    @property
    def n_frames(self) -> int:
        """Number of frames in the sequence."""
        return len(self.timestamps)

    def get_joint_positions(self, joint_name: str) -> np.ndarray:
        """Get joint positions as (N, 3) array."""
        return self.joints[joint_name].as_array()

    def get_bone_lengths(self, parent: str, child: str) -> np.ndarray:
        """Calculate bone lengths over time."""
        p = self.joints[parent].as_array()
        c = self.joints[child].as_array()
        return np.linalg.norm(c - p, axis=1)

    def get_all_bone_lengths(self) -> dict[tuple[str, str], np.ndarray]:
        """Get all bone lengths over time."""
        return {(p, c): self.get_bone_lengths(p, c) for p, c in BONES}

    def get_reference_bone_lengths(self) -> dict[tuple[str, str], float]:
        """Get median bone length for each bone (reference for constraints)."""
        return {
            bone: float(np.nanmedian(lengths))
            for bone, lengths in self.get_all_bone_lengths().items()
        }


def load_subject_data(filepath: Path | str) -> pd.DataFrame:
    """Load motion capture data from Excel file."""
    return pd.read_excel(filepath)


def dataframe_to_skeleton(df: pd.DataFrame, subject_id: int) -> SkeletonSequence:
    """Convert DataFrame to SkeletonSequence."""
    timestamps = df["timestamp"].values

    joints = {}
    for joint_name in JOINT_NAMES:
        joints[joint_name] = Joint(
            x=df[f"{joint_name}_x"].values,
            y=df[f"{joint_name}_y"].values,
            z=df[f"{joint_name}_z"].values,
        )

    return SkeletonSequence(timestamps=timestamps, joints=joints, subject_id=subject_id)


def load_all_subjects(data_dir: Path | str) -> list[SkeletonSequence]:
    """Load all subject data from directory."""
    data_dir = Path(data_dir)
    subjects = []

    for i in range(1, 6):
        filepath = data_dir / f"{i}.xlsx"
        if filepath.exists():
            df = load_subject_data(filepath)
            skeleton = dataframe_to_skeleton(df, subject_id=i)
            subjects.append(skeleton)

    return subjects


def infer_coordinate_system(skeleton: SkeletonSequence) -> dict[str, str]:
    """
    Infer coordinate system axes from skeleton data.

    Returns mapping of semantic axes to data axes:
    - 'vertical': axis perpendicular to ground (typically Y or Z)
    - 'lateral': axis along shoulder line (left-right)
    - 'anterior': axis perpendicular to frontal plane (front-back)
    """
    # Get mean positions for analysis (use nanmean to handle missing data)
    l_ankle = np.nanmean(skeleton.joints["l_ankle"].as_array(), axis=0)
    r_ankle = np.nanmean(skeleton.joints["r_ankle"].as_array(), axis=0)
    l_hip = np.nanmean(skeleton.joints["l_hip"].as_array(), axis=0)
    r_hip = np.nanmean(skeleton.joints["r_hip"].as_array(), axis=0)
    head = np.nanmean(skeleton.joints["head"].as_array(), axis=0)

    # Vertical axis: head is above ankles
    ankle_center = (l_ankle + r_ankle) / 2
    vertical_vec = head - ankle_center
    vertical_axis = int(np.argmax(np.abs(vertical_vec)))

    # Lateral axis: along hip line (left-right)
    lateral_vec = r_hip - l_hip
    # Zero out vertical component
    lateral_vec[vertical_axis] = 0
    lateral_axis = int(np.argmax(np.abs(lateral_vec)))

    # Anterior axis: remaining axis
    anterior_axis = 3 - vertical_axis - lateral_axis

    axis_names = ["x", "y", "z"]
    return {
        "vertical": axis_names[vertical_axis],
        "lateral": axis_names[lateral_axis],
        "anterior": axis_names[anterior_axis],
    }
