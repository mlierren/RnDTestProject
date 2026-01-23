"""
Main entry point for unified filtering pipeline.

Usage:
    uv run python -m src.main --subjects 4 5 --preset high-noise
    uv run python -m src.main --subjects 1 2 3 --preset default
"""

import argparse
from pathlib import Path
import sys
import csv

import numpy as np

# Add parent directory to path for src_old imports (legacy code)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_old.data_loader import load_all_subjects, infer_coordinate_system
from src_old.analysis.cycle_detection import detect_squat_cycles, analyze_cycles
from src_old.assessment.knee_valgus import assess_knee_valgus
from src_old.assessment.posture import assess_posture
from src_old.visualization import (
    plot_knee_valgus_assessment,
    plot_posture_assessment,
)
from .fast_animation import (
    create_fast_animation,
    save_cached_result,
    load_cached_result,
    has_cached_result,
    CachedResult,
)
from .config import get_preset_config, BONES
from .preprocessing import (
    skeleton_to_array,
    array_to_skeleton,
    skeleton_to_positions_dict,
    create_filtered_result_adapter,
)
from .unified_filter import unified_filter, compute_metrics


def export_assessment_csv(
    output_path: Path,
    filtered_positions: np.ndarray,
    knee_result,
    posture_result,
    coord_system: dict[str, str],
):
    """
    Export filtered positions and assessment angles to CSV.

    Args:
        output_path: Path to save the CSV file
        filtered_positions: (n_frames, n_joints, 3) filtered positions
        knee_result: KneeValgusResult from assessment
        posture_result: PostureResult from assessment
        coord_system: Coordinate system mapping
    """
    from .config import JOINT_NAMES

    n_frames = filtered_positions.shape[0]

    # Build header
    header = ["frame"]
    for joint in JOINT_NAMES:
        header.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    header.extend([
        "left_knee_valgus_deg",
        "right_knee_valgus_deg",
        "back_arch_deg",
        "torso_lean_deg",
    ])

    # Build rows
    rows = []
    for frame in range(n_frames):
        row = [frame]
        # Joint positions
        for j in range(len(JOINT_NAMES)):
            row.extend([
                f"{filtered_positions[frame, j, 0]:.4f}",
                f"{filtered_positions[frame, j, 1]:.4f}",
                f"{filtered_positions[frame, j, 2]:.4f}",
            ])
        # Assessment angles
        left_valgus = knee_result.left_angle[frame]
        right_valgus = knee_result.right_angle[frame]
        back_arch = posture_result.back_arch_angle[frame]
        torso_lean = posture_result.torso_lean_angle[frame]

        row.extend([
            f"{left_valgus:.2f}" if not np.isnan(left_valgus) else "",
            f"{right_valgus:.2f}" if not np.isnan(right_valgus) else "",
            f"{back_arch:.2f}" if not np.isnan(back_arch) else "",
            f"{torso_lean:.2f}" if not np.isnan(torso_lean) else "",
        ])
        rows.append(row)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Unified optimization-based motion capture filtering"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing subject data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/assessment",
        help="Output directory for results",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject IDs to process (default: all)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "high-noise"],
        default="default",
        help="Configuration preset",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max iterations",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Generate skeleton animations",
    )
    parser.add_argument(
        "--animation-only",
        action="store_true",
        help="Only generate animation from cached results (skip optimization)",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=None,
        help="Number of frames for animation (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't save/load cached results",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for animation (default: auto)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only apply preprocessing (skip optimization) - for comparison",
    )

    args = parser.parse_args()

    # Load configuration
    config = get_preset_config(args.preset)
    config.verbose = not args.quiet

    if args.max_iterations:
        config.optimization.max_iterations = args.max_iterations

    print(f"Using preset '{args.preset}'")

    # Load data
    print("Loading motion capture data...")
    subjects = load_all_subjects(args.data_dir)
    print(f"Loaded {len(subjects)} subjects")

    # Filter subjects if specified
    if args.subjects:
        subjects = [s for s in subjects if s.subject_id in args.subjects]
        print(f"Processing subjects: {[s.subject_id for s in subjects]}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject
    for skeleton in subjects:
        subject_id = skeleton.subject_id
        subject_output = output_dir / f"subject_{subject_id}"
        subject_output.mkdir(exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*50}")

        # Check for cached results in animation-only mode
        if args.animation_only:
            cached = load_cached_result(subject_output)
            if cached is None:
                print(f"  No cached results found for Subject {subject_id}, skipping...")
                continue

            print(f"  Loaded cached results")
            original_positions = cached.original_positions
            filtered_positions = cached.filtered_positions
            coord_system = cached.coord_system
            bone_lengths = cached.bone_lengths

            # Create animation from cache
            if args.animation or args.animation_only:
                print("Generating 3D skeleton animation (parallel)...")
                create_fast_animation(
                    original_positions,
                    filtered_positions,
                    output_path=subject_output / "skeleton_comparison.gif",
                    coord_system=coord_system,
                    max_frames=args.animation_frames,
                    n_workers=args.n_workers,
                )
            continue

        # Infer coordinate system
        coord_system = infer_coordinate_system(skeleton)
        print(f"Coordinate system: {coord_system}")

        # Convert to array
        original_positions = skeleton_to_array(skeleton)

        # Preprocess-only mode: skip optimization
        if args.preprocess_only:
            from .preprocessing import preprocess
            from .config import BONES

            print("\n[Preprocess-only mode] Skipping optimization...")
            prep_result = preprocess(original_positions, BONES, config.preprocessing)
            preprocessed_positions = prep_result.positions

            print(f"  Spikes detected: {prep_result.spike_mask.sum()}")
            print(f"  Valid frames: {prep_result.valid_mask.sum()}/{len(prep_result.valid_mask)}")

            # Generate animation comparing original vs preprocessed (no optimization)
            if args.animation:
                print("Generating 3D skeleton animation (preprocess only)...")
                create_fast_animation(
                    original_positions,
                    preprocessed_positions,
                    output_path=subject_output / "skeleton_preprocess_only.gif",
                    coord_system=coord_system,
                    max_frames=args.animation_frames,
                    n_workers=args.n_workers,
                )
            continue

        # Apply unified filter (preprocessing + optimization)
        print("\nApplying unified filter...")
        result = unified_filter(original_positions, config=config)

        # Convert back to skeleton
        filtered_skeleton = array_to_skeleton(result.positions, skeleton)

        # Save cached results (unless --no-cache)
        if not args.no_cache:
            cached = CachedResult(
                original_positions=original_positions,
                filtered_positions=result.positions,
                bone_lengths=result.bone_lengths,
                coord_system=coord_system,
            )
            save_cached_result(cached, subject_output)
            print(f"  Results saved to {subject_output}")

        # Compute metrics
        metrics = compute_metrics(
            original_positions,
            result.positions,
            result.bone_lengths,
        )

        print(f"\n{'─'*60}")
        print(f"Results for Subject {subject_id}")
        print(f"{'─'*60}")
        print(f"  RMSE: {metrics['rmse_mm']:.1f} mm")
        print(f"  Bone length std: {metrics['bone_length_std_mm']:.2f} mm")
        print(f"  Acceleration improvement: {metrics['acceleration_improvement']:.1f}x")

        # Detect squat cycles
        print("\nDetecting squat cycles...")
        axis_map = {"x": 0, "y": 1, "z": 2}
        vertical_axis = axis_map[coord_system["vertical"]]
        positions_dict = skeleton_to_positions_dict(filtered_skeleton)
        cycles = detect_squat_cycles(positions_dict, vertical_axis=vertical_axis)
        print(f"  Detected {len(cycles)} squat cycles")

        if len(cycles) == 0:
            print("  Warning: No squat cycles detected, skipping assessment")
            continue

        for i, cycle in enumerate(cycles):
            print(f"    Cycle {i+1}: frames {cycle.start_frame}-{cycle.end_frame}, depth {cycle.depth:.1f}mm")

        # Run assessments
        print("Running assessments...")
        filtered_adapter = create_filtered_result_adapter(
            filtered_skeleton, bone_lengths=result.bone_lengths
        )
        knee_result = assess_knee_valgus(filtered_adapter, coord_system)
        posture_result = assess_posture(filtered_adapter, coord_system)

        # Analyze per-cycle statistics
        left_knee_analysis = analyze_cycles(cycles, knee_result.left_angle)
        right_knee_analysis = analyze_cycles(cycles, knee_result.right_angle)
        back_arch_analysis = analyze_cycles(cycles, posture_result.back_arch_angle)
        torso_lean_analysis = analyze_cycles(cycles, posture_result.torso_lean_angle)

        # Print assessment results
        print(f"\n  ▶ Knee Valgus (positive=inward, negative=outward)")
        ci_width = 3.0  # Approximate CI width for display
        if left_knee_analysis.get("aggregate"):
            agg = left_knee_analysis["aggregate"]
            print(f"    Left:  max {agg['max_angle_mean']:7.1f}° (+/- {agg['max_angle_std']:.1f}°)")
            print(f"           min {agg['min_angle_mean']:7.1f}° (+/- {agg['min_angle_std']:.1f}°)")
        if right_knee_analysis.get("aggregate"):
            agg = right_knee_analysis["aggregate"]
            print(f"    Right: max {agg['max_angle_mean']:7.1f}° (+/- {agg['max_angle_std']:.1f}°)")
            print(f"           min {agg['min_angle_mean']:7.1f}° (+/- {agg['min_angle_std']:.1f}°)")

        if back_arch_analysis.get("aggregate"):
            agg_ba = back_arch_analysis["aggregate"]
            agg_tl = torso_lean_analysis.get("aggregate")
            print(f"\n  ▶ Posture (Lateral View)")
            print(f"    Back Arch:  max {agg_ba['max_angle_mean']:7.1f}° (+/- {agg_ba['max_angle_std']:.1f}°)")
            print(f"                min {agg_ba['min_angle_mean']:7.1f}° (+/- {agg_ba['min_angle_std']:.1f}°)")
            if agg_tl:
                print(f"    Torso Lean: max {agg_tl['max_angle_mean']:7.1f}° (+/- {agg_tl['max_angle_std']:.1f}°)")
                print(f"                min {agg_tl['min_angle_mean']:7.1f}° (+/- {agg_tl['min_angle_std']:.1f}°)")

        # Create visualizations
        print("\nGenerating visualizations...")
        plot_knee_valgus_assessment(knee_result, subject_output / "knee_valgus.png")
        plot_posture_assessment(posture_result, subject_output / "posture.png")

        # Export assessment CSV to subject folder
        assessment_csv_path = subject_output / "assessment.csv"
        export_assessment_csv(
            assessment_csv_path,
            result.positions,
            knee_result,
            posture_result,
            coord_system,
        )
        print(f"  Assessment CSV saved to: {assessment_csv_path}")

        # Animation (if requested)
        if args.animation:
            print("Generating 3D skeleton animation (parallel)...")
            create_fast_animation(
                original_positions,
                result.positions,
                output_path=subject_output / "skeleton_comparison.gif",
                coord_system=coord_system,
                max_frames=args.animation_frames,
                n_workers=args.n_workers,
            )

    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
