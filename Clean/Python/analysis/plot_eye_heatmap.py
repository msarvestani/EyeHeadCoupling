"""Simple script to plot a heatmap of eye positions from CSV data.

Usage:
    python plot_eye_heatmap.py --folder <path_to_folder>
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add path to import eyehead module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from eyehead.io import clean_csv


def find_eye_position_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the eye position CSV file in the folder."""
    csv_files = list(folder_path.glob("*ellipse_center_XY_L*.csv"))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No eye position CSV file found in {folder_path}\n"
            f"Looking for files matching pattern: *ellipse_center_XY_L*.csv"
        )
    
    if len(csv_files) > 1:
        print(f"Warning: Found {len(csv_files)} matching files, using first one:")
        for f in csv_files:
            print(f"  - {f.name}")
    
    return csv_files[0]


def find_origin_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the origin of eye coordinate CSV file in the folder."""
    csv_files = list(folder_path.glob("*origin_of_eyecoordinate_L*.csv"))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No origin CSV file found in {folder_path}\n"
            f"Looking for files matching pattern: *origin_of_eyecoordinate_L*.csv"
        )
    
    if len(csv_files) > 1:
        print(f"Warning: Found {len(csv_files)} origin files, using first one:")
        for f in csv_files:
            print(f"  - {f.name}")
    
    return csv_files[0]


def find_vdaxis_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the vdaxis CSV file in the folder."""
    csv_files = list(folder_path.glob("*vdaxis_L*.csv"))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No vdaxis CSV file found in {folder_path}\n"
            f"Looking for files matching pattern: *vdaxis_L*.csv"
        )
    
    if len(csv_files) > 1:
        print(f"Warning: Found {len(csv_files)} vdaxis files, using first one:")
        for f in csv_files:
            print(f"  - {f.name}")
    
    return csv_files[0]


def plot_eye_position_heatmap(folder_path: str, animal_id: str = "Tsh001", 
                              results_dir: str = None):
    """Load eye position CSV from folder and create a heatmap.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files
    animal_id : str
        Animal identifier prefix (default: "Tsh001")
    results_dir : str, optional
        Directory to save the figure (if None, saves in folder/results)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        abs_path = folder_path.resolve()
        raise FileNotFoundError(
            f"Folder not found: {folder_path}\n"
            f"Absolute path tried: {abs_path}\n"
            f"Current working directory: {Path.cwd()}"
        )
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    # Find the eye position CSV file
    csv_path = find_eye_position_csv(folder_path, animal_id)
    
    # Find the origin CSV file (for horizontal centering - left/right corners)
    origin_csv_path = find_origin_csv(folder_path, animal_id)
    
    # Find the vdaxis CSV file (for vertical centering - upper/lower corners)
    vdaxis_csv_path = find_vdaxis_csv(folder_path, animal_id)
    
    # Load origin data for horizontal centering
    print(f"\nLoading origin data from: {origin_csv_path.name}")
    cleaned_origin = clean_csv(str(origin_csv_path))
    origin_data = np.genfromtxt(cleaned_origin, delimiter=",", skip_header=1)
    
    if origin_data.ndim == 1:
        origin_data = origin_data.reshape(1, -1)
    
    print(f"  Origin data shape: {origin_data.shape}")
    
    # Extract left and right corner coordinates
    # Columns: 0=frame, 1=timestamp, 2=left_x, 3=left_y, 4=right_x, 5=right_y
    left_x = origin_data[:, 2]
    left_y = origin_data[:, 3]
    right_x = origin_data[:, 4]
    right_y = origin_data[:, 5]
    
    # Calculate horizontal eye center (midpoint between left and right corners)
    eye_center_x = (left_x + right_x) / 2
    mean_eye_center_x = np.nanmean(eye_center_x)
    
    # Calculate eye width for normalization
    eye_widths = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    valid_widths = eye_widths[~np.isnan(eye_widths)]
    eye_width_pixels = np.mean(valid_widths)
    
    print(f"  Horizontal eye center X: {mean_eye_center_x:.1f} pixels")
    print(f"  Eye width: {eye_width_pixels:.1f} pixels")
    
    # Load vdaxis data for vertical centering
    print(f"\nLoading vdaxis data from: {vdaxis_csv_path.name}")
    cleaned_vdaxis = clean_csv(str(vdaxis_csv_path))
    vdaxis_data = np.genfromtxt(cleaned_vdaxis, delimiter=",", skip_header=1)
    
    if vdaxis_data.ndim == 1:
        vdaxis_data = vdaxis_data.reshape(1, -1)
    
    print(f"  VDaxis data shape: {vdaxis_data.shape}")
    print(f"  First row (first 8 columns): {vdaxis_data[0, :8] if vdaxis_data.shape[1] >= 8 else vdaxis_data[0, :]}")
    
    # Extract upper and lower corner coordinates
    # Assuming columns: 0=frame, 1=timestamp, 2=upper_x, 3=upper_y, 4=lower_x, 5=lower_y
    # (We may need to adjust these based on the actual CSV structure)
    upper_x = vdaxis_data[:, 2]
    upper_y = vdaxis_data[:, 3]
    lower_x = vdaxis_data[:, 4]
    lower_y = vdaxis_data[:, 5]
    
    print(f"\n  Sample upper corner: ({upper_x[0]:.2f}, {upper_y[0]:.2f})")
    print(f"  Sample lower corner: ({lower_x[0]:.2f}, {lower_y[0]:.2f})")
    
    # Calculate vertical eye center (midpoint between upper and lower corners)
    eye_center_y = (upper_y + lower_y) / 2
    mean_eye_center_y = np.nanmean(eye_center_y)
    
    # Calculate eye height for normalization
    eye_heights = np.sqrt((upper_x - lower_x)**2 + (upper_y - lower_y)**2)
    valid_heights = eye_heights[~np.isnan(eye_heights)]
    eye_height_pixels = np.mean(valid_heights)
    
    print(f"  Vertical eye center Y: {mean_eye_center_y:.1f} pixels")
    print(f"  Eye height: {eye_height_pixels:.1f} pixels")
    
    print(f"\nLoading eye position data from: {csv_path.name}")
    
    # Use the existing clean_csv function from eyehead.io
    cleaned = clean_csv(str(csv_path))
    ellipse_center_xy = np.genfromtxt(cleaned, delimiter=",", skip_header=1)
    
    # Extract columns
    eye_x = ellipse_center_xy[:, 2]
    eye_y = ellipse_center_xy[:, 3]
    
    
    # Center the data on the anatomical center of the eye
    eye_x_centered = eye_x - mean_eye_center_x
    eye_y_centered = eye_y - mean_eye_center_y
    
    # Normalize: X by eye width, Y by eye height
    eye_x_normalized = eye_x_centered / eye_width_pixels
    eye_y_normalized = eye_y_centered / eye_height_pixels
    
    
    # Filter out NaN values before plotting
    valid_mask = ~(np.isnan(eye_x_normalized) | np.isnan(eye_y_normalized))
    eye_x_normalized = eye_x_normalized[valid_mask]
    eye_y_normalized = eye_y_normalized[valid_mask]
    
    if len(eye_x_normalized) == 0:
        raise ValueError("No valid data points after normalization")
    
    print(f"  Valid points after filtering NaN: {len(eye_x_normalized)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use adaptive range based on actual data, but with some padding
    x_min, x_max = np.nanmin(eye_x_normalized), np.nanmax(eye_x_normalized)
    y_min, y_max = np.nanmin(eye_y_normalized), np.nanmax(eye_y_normalized)
    
    # Add 20% padding
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2
    
    x_range = [-0.1,0.1]  # Fixed range for better comparison across plots
    y_range = [-0.1, 0.1]  # Fixed range for better comparison across plots
    
    
    # Create 2D histogram/heatmap
    heatmap, xedges, yedges = np.histogram2d(
        eye_x_normalized, eye_y_normalized,
        bins=[80, 80],
        range=[x_range, y_range]
    )
    
    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, origin='lower', extent=extent,
                   cmap='hot', aspect='auto', interpolation='gaussian')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Density')
    
    # Mark the center at (0, 0) - anatomical center of the eye
    ax.plot(0, 0, 'c+', markersize=20, markeredgewidth=3, label='Eye Center')
    
    # Set axis limits
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect('equal')
    
    # Add grid lines at 0
    ax.axhline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('X Position (eye widths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (eye heights)', fontsize=12, fontweight='bold')
    
    # Extract date from folder name if possible
    import re
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', str(folder_path))
    date_str = date_match.group() if date_match else ""
    
    title = f'Eye Position Heatmap (Centered on Eye)\n'
    if animal_id:
        title += f'{animal_id} - '
    if date_str:
        title += date_str
    title += f'\nEye width: {eye_width_pixels:.1f}px, Eye height: {eye_height_pixels:.1f}px'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if results_dir is None:
        results_dir = folder_path / "results"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{animal_id}_" if animal_id else ""
    date_suffix = f"_{date_str}" if date_str else ""
    output_filename = f"{prefix}eye_position_heatmap_normalized{date_suffix}.png"
    output_path = results_dir / output_filename
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved heatmap to: {output_path}")
    
    # Show plot
    plt.show()
    
    return fig


def main():
    """Main function to run from command line."""
    parser = argparse.ArgumentParser(
        description='Plot a heatmap of eye positions from folder containing CSV files'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the folder containing eye position CSV files'
    )
    parser.add_argument(
        '--animal-id',
        type=str,
        default='Tsh001',
        help='Animal identifier (default: Tsh001)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to save the output figure (default: <folder>/results)'
    )
    
    args = parser.parse_args()
    
    plot_eye_position_heatmap(args.folder, args.animal_id, args.results_dir)


if __name__ == '__main__':
    main()
