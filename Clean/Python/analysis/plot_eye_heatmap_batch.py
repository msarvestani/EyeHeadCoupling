"""Batch process multiple sessions and create a single PDF with all heatmaps.

Usage:
    python batch_eye_heatmap.py --parent-folder <path_to_parent_folder>
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add path to import eyehead module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from eyehead.io import clean_csv


def find_eye_position_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the eye position CSV file in the folder."""
    csv_files = list(folder_path.glob("*ellipse_center_XY_L*.csv"))
    if len(csv_files) == 0:
        return None
    return csv_files[0]


def find_origin_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the origin of eye coordinate CSV file in the folder."""
    csv_files = list(folder_path.glob("*origin_of_eyecoordinate_L*.csv"))
    if len(csv_files) == 0:
        return None
    return csv_files[0]


def find_vdaxis_csv(folder_path: Path, animal_id: str = "Tsh001") -> Path:
    """Find the vdaxis CSV file in the folder."""
    csv_files = list(folder_path.glob("*vdaxis_L*.csv"))
    if len(csv_files) == 0:
        return None
    return csv_files[0]


def plot_single_heatmap(csv_path: Path, origin_csv_path: Path, vdaxis_csv_path: Path, 
                       folder_name: str) -> plt.Figure:
    """Create a heatmap figure from CSV files."""
    
    # Load origin data for horizontal centering
    cleaned_origin = clean_csv(str(origin_csv_path))
    origin_data = np.genfromtxt(cleaned_origin, delimiter=",", skip_header=1)
    
    if origin_data.ndim == 1:
        origin_data = origin_data.reshape(1, -1)
    
    # Extract left and right corner coordinates
    left_x = origin_data[:, 2]
    left_y = origin_data[:, 3]
    right_x = origin_data[:, 4]
    right_y = origin_data[:, 5]
    
    # Calculate horizontal eye center
    eye_center_x = (left_x + right_x) / 2
    mean_eye_center_x = np.nanmean(eye_center_x)
    
    # Calculate eye width for normalization
    eye_widths = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
    valid_widths = eye_widths[~np.isnan(eye_widths)]
    eye_width_pixels = np.mean(valid_widths)
    
    # Load vdaxis data for vertical centering
    cleaned_vdaxis = clean_csv(str(vdaxis_csv_path))
    vdaxis_data = np.genfromtxt(cleaned_vdaxis, delimiter=",", skip_header=1)
    
    if vdaxis_data.ndim == 1:
        vdaxis_data = vdaxis_data.reshape(1, -1)
    
    # Extract upper and lower corner coordinates
    upper_x = vdaxis_data[:, 2]
    upper_y = vdaxis_data[:, 3]
    lower_x = vdaxis_data[:, 4]
    lower_y = vdaxis_data[:, 5]
    
    # Calculate vertical eye center
    eye_center_y = (upper_y + lower_y) / 2
    mean_eye_center_y = np.nanmean(eye_center_y)
    
    # Calculate eye height for normalization
    eye_heights = np.sqrt((upper_x - lower_x)**2 + (upper_y - lower_y)**2)
    valid_heights = eye_heights[~np.isnan(eye_heights)]
    eye_height_pixels = np.mean(valid_heights)
    
    print(f"    Eye width: {eye_width_pixels:.1f}px, Eye height: {eye_height_pixels:.1f}px")
    
    # Load eye position data
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
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(eye_x_normalized) | np.isnan(eye_y_normalized))
    eye_x_normalized = eye_x_normalized[valid_mask]
    eye_y_normalized = eye_y_normalized[valid_mask]
    
    if len(eye_x_normalized) == 0:
        raise ValueError("No valid data points after normalization")
    
    print(f"    {len(eye_x_normalized)} valid points")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Fixed range: -0.1 to 0.1 for both axes
    x_range = [-0.06, 0.06]
    y_range = [-0.06, 0.06]
    
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
    plt.colorbar(im, ax=ax, label='Density')
    
    # Mark the center at (0, 0) - anatomical center of the eye
    ax.plot(0, 0, 'c+', markersize=20, markeredgewidth=3, label='Eye Center')
    
    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    
    # Add grid lines at 0
    ax.axhline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('X Position (eye widths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (eye heights)', fontsize=12, fontweight='bold')
    
    # Title with full folder name and eye dimensions
    title = f'Eye Position Heatmap (Centered on Eye)\n{folder_name}\nEye: {eye_width_pixels:.1f}px × {eye_height_pixels:.1f}px'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig


def batch_process_folders(parent_folder: str, animal_id: str = "Tsh001", 
                         output_pdf: str = None):
    """Process all subfolders and create a single PDF with all heatmaps."""
    parent_folder = Path(parent_folder)
    
    if not parent_folder.exists():
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")
    
    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {parent_folder}")
    
    # Find all subfolders that contain required CSV files
    session_folders = []
    
    print(f"Scanning {parent_folder} for session folders...")
    for subfolder in sorted(parent_folder.iterdir()):
        if not subfolder.is_dir():
            continue
        
        csv_file = find_eye_position_csv(subfolder, animal_id)
        origin_file = find_origin_csv(subfolder, animal_id)
        vdaxis_file = find_vdaxis_csv(subfolder, animal_id)
        
        if csv_file is not None and origin_file is not None and vdaxis_file is not None:
            session_folders.append((subfolder, csv_file, origin_file, vdaxis_file))
        else:
            missing = []
            if csv_file is None:
                missing.append("eye position")
            if origin_file is None:
                missing.append("origin")
            if vdaxis_file is None:
                missing.append("vdaxis")
            if missing:
                print(f"  Skipping {subfolder.name}: missing {', '.join(missing)} CSV")
    
    if len(session_folders) == 0:
        print("No session folders with required CSV files found!")
        return
    
    print(f"\nFound {len(session_folders)} session folders with complete data")
    
    # Set output PDF path
    if output_pdf is None:
        output_pdf = parent_folder / "all_eye_heatmaps_normalized.pdf"
    else:
        output_pdf = Path(output_pdf)
    
    print(f"\nGenerating normalized heatmaps and saving to: {output_pdf}")
    
    # Create PDF with all heatmaps
    with PdfPages(output_pdf) as pdf:
        for i, (folder, csv_file, origin_file, vdaxis_file) in enumerate(session_folders, 1):
            print(f"\n[{i}/{len(session_folders)}] Processing: {folder.name}")
            
            try:
                fig = plot_single_heatmap(csv_file, origin_file, vdaxis_file, folder.name)
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Added to PDF")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✓ Successfully created PDF with {len(session_folders)} normalized heatmaps")
    print(f"  Saved to: {output_pdf}")


def main():
    """Main function to run from command line."""
    parser = argparse.ArgumentParser(
        description='Batch process eye position heatmaps and create a single PDF'
    )
    parser.add_argument(
        '--parent-folder',
        type=str,
        required=True,
        help='Path to the parent folder containing session subfolders'
    )
    parser.add_argument(
        '--animal-id',
        type=str,
        default='Tsh001',
        help='Animal identifier (default: Tsh001)'
    )
    parser.add_argument(
        '--output-pdf',
        type=str,
        default=None,
        help='Path to output PDF file (default: <parent-folder>/all_eye_heatmaps_normalized.pdf)'
    )
    
    args = parser.parse_args()
    
    batch_process_folders(args.parent_folder, args.animal_id, args.output_pdf)


if __name__ == '__main__':
    main()
