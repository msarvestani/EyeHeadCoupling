#!/usr/bin/env python3
"""
Quick test script to demonstrate the random walk chance performance calculation.

This script shows how to use the new functions:
1. compute_movement_statistics() - Extract velocities and turning angles
2. simulate_random_walk_trial() - Generate a random walk trajectory
3. calculate_random_walk_chance_performance() - Compute chance performance by target size
"""

import sys
from pathlib import Path

# Add the Python directory to path
sys.path.append(str(Path(__file__).resolve().parent / 'Clean' / 'Python'))

# Import the analysis module
from analysis.prosaccade_feedback_session import (
    load_feedback_data,
    identify_and_filter_failed_trials,
    extract_trial_trajectories,
    compute_movement_statistics,
    simulate_random_walk_trial,
    calculate_random_walk_chance_performance
)

import numpy as np


def test_random_walk_functions(session_folder: str):
    """Test the random walk chance performance calculation on a session."""

    folder_path = Path(session_folder)

    print("="*80)
    print("RANDOM WALK CHANCE PERFORMANCE TEST")
    print("="*80)
    print(f"\nSession folder: {folder_path}")

    # Load data
    print("\n1. Loading session data...")
    eot_df, eye_df, target_df = load_feedback_data(folder_path)
    print(f"   Loaded {len(eot_df)} trials, {len(eye_df)} eye positions, {len(target_df)} targets")

    # Filter and extract trials
    print("\n2. Extracting trial trajectories...")
    successful_indices = identify_and_filter_failed_trials(target_df, eot_df, eye_df)
    trials = extract_trial_trajectories(eot_df, eye_df, target_df, successful_indices)
    print(f"   Extracted {len(trials)} trial trajectories")

    # Test 1: Compute movement statistics
    print("\n" + "="*80)
    print("TEST 1: Computing Movement Statistics")
    print("="*80)

    velocities, turning_angles = compute_movement_statistics(trials)

    print(f"\nVelocity Distribution:")
    print(f"  N samples: {len(velocities)}")
    print(f"  Mean: {np.mean(velocities):.4f}")
    print(f"  Median: {np.median(velocities):.4f}")
    print(f"  Std: {np.std(velocities):.4f}")
    print(f"  Range: [{np.min(velocities):.4f}, {np.max(velocities):.4f}]")
    print(f"  Q25: {np.percentile(velocities, 25):.4f}")
    print(f"  Q75: {np.percentile(velocities, 75):.4f}")

    print(f"\nTurning Angle Distribution:")
    print(f"  N samples: {len(turning_angles)}")
    print(f"  Mean: {np.mean(turning_angles):.4f} rad ({np.degrees(np.mean(turning_angles)):.2f}°)")
    print(f"  Median: {np.median(turning_angles):.4f} rad ({np.degrees(np.median(turning_angles)):.2f}°)")
    print(f"  Std: {np.std(turning_angles):.4f} rad ({np.degrees(np.std(turning_angles)):.2f}°)")
    print(f"  Range: [{np.min(turning_angles):.4f}, {np.max(turning_angles):.4f}] rad")

    # Test 2: Simulate a single random walk trial
    print("\n" + "="*80)
    print("TEST 2: Simulating Single Random Walk Trial")
    print("="*80)

    # Use first trial with eye data
    test_trial = None
    for trial in trials:
        if trial.get('has_eye_data', False):
            test_trial = trial
            break

    if test_trial is not None:
        start_x = test_trial['eye_x'][0]
        start_y = test_trial['eye_y'][0]
        duration = test_trial['eye_times'][-1] - test_trial['eye_times'][0]

        print(f"\nTrial parameters:")
        print(f"  Start position: ({start_x:.3f}, {start_y:.3f})")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Target: ({test_trial['target_x']:.2f}, {test_trial['target_y']:.2f})")
        print(f"  Target diameter: {test_trial['target_diameter']:.3f}")

        # Generate 5 random walk samples
        print(f"\nGenerating 5 random walk samples...")
        for i in range(5):
            sim_x, sim_y, sim_times = simulate_random_walk_trial(
                start_x, start_y, duration, velocities, turning_angles
            )
            print(f"  Sample {i+1}: {len(sim_x)} positions, "
                  f"final position: ({sim_x[-1]:.3f}, {sim_y[-1]:.3f})")
    else:
        print("No trial with eye data found for testing.")

    # Test 3: Calculate full chance performance
    print("\n" + "="*80)
    print("TEST 3: Calculating Chance Performance (10 simulations per trial)")
    print("="*80)

    # Use smaller number for quick test
    results = calculate_random_walk_chance_performance(
        trials,
        n_simulations=10,  # Small number for quick test
        results_dir=folder_path / 'results'
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOverall chance performance: {100*results['overall_chance']:.2f}%")
    print(f"\nBy target diameter:")
    for diameter in sorted(results['by_diameter'].keys()):
        rate = results['by_diameter'][diameter]
        n_success, n_total = results['by_diameter_counts'][diameter]
        print(f"  {diameter:.3f}: {100*rate:.2f}% ({n_success}/{n_total})")

    print("\n✓ All tests completed successfully!")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_random_walk.py <session_folder>")
        print("\nExample:")
        print("  python test_random_walk.py Clean/Data/Tsh001/Tsh001-250102-145702")
        sys.exit(1)

    session_folder = sys.argv[1]
    test_random_walk_functions(session_folder)
