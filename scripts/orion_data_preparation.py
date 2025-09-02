"""
Data preparation script for rotating and voxelizing 3D mesh files (.off)
from the ModelNet dataset.

This script processes each mesh file by:
1.  Loading the mesh and normalizing it (center and scale to unit cube).
2.  Generating a specified number of evenly-spaced rotations around the Z-axis.
3.  For each rotation:
    a. Voxelizing the mesh into a grid (e.g., 28x28x28).
    b. Padding the grid to a final size (e.g., 32x32x32).
4.  Saving each oriented, voxelized grid as a separate .npy file.
"""
import os
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---

# Defines the number of orientation classes for each object category.
# A value of 1 means only the original orientation is used.
POSE_CONFIG = {
    "modelnet10": {
        'bathtub': 6, 'bed': 12, 'chair': 12, 'desk': 12, 'dresser': 12,
        'monitor': 12, 'night_stand': 12, 'sofa': 12, 'table': 3, 'toilet': 12,
    },
    "modelnet40": {
        'airplane': 12, 'bathtub': 3, 'bed': 12, 'bench': 1, 'bookshelf': 12,
        'bottle': 1, 'bowl': 1, 'car': 12, 'chair': 12, 'cone': 1, 'cup': 1,
        'curtain': 1, 'desk': 12, 'door': 1, 'dresser': 3, 'flower_pot': 1,
        'glass_box': 1, 'guitar': 12, 'keyboard': 1, 'lamp': 1, 'laptop': 12,
        'mantel': 12, 'monitor': 3, 'night_stand': 1, 'person': 1, 'piano': 12,
        'plant': 1, 'radio': 1, 'range_hood': 12, 'sink': 1, 'sofa': 12,
        'stairs': 1, 'stool': 1, 'table': 1, 'tent': 1, 'toilet': 12,
        'tv_stand': 1, 'vase': 1, 'wardrobe': 1, 'xbox': 1
    }
}

def rotate_mesh(mesh: trimesh.Trimesh, angle_rad: float) -> trimesh.Trimesh:
    """Rotates a mesh around the Z-axis by a given angle in radians."""
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle_rad, [0, 0, 1], point=(0, 0, 0)
    )
    return mesh.apply_transform(rotation_matrix)

def voxelize_and_pad(mesh: trimesh.Trimesh, final_size: int, padding: int) -> np.ndarray:
    """Voxelizes a mesh and pads it to a final consistent size."""
    crop_size = final_size - (2 * padding)
    crop_shape = (crop_size, crop_size, crop_size)

    # Voxelize the mesh to the intermediate size
    voxel_grid = mesh.voxelized(pitch=1.0 / crop_size)
    source_matrix = voxel_grid.matrix.astype(np.bool_)

    # Center the source grid in an intermediate matrix of the target crop size
    intermediate_matrix = np.zeros(crop_shape, dtype=np.bool_)
    start_coords = (np.array(crop_shape) - np.array(source_matrix.shape)) // 2
    src_start = np.maximum(0, -start_coords)
    src_end = np.minimum(source_matrix.shape, crop_shape - start_coords)
    dest_start = np.maximum(0, start_coords)
    dest_end = np.minimum(crop_shape, start_coords + source_matrix.shape)
    src_slice = tuple(slice(s, e) for s, e in zip(src_start, src_end))
    dest_slice = tuple(slice(s, e) for s, e in zip(dest_start, dest_end))
    intermediate_matrix[dest_slice] = source_matrix[src_slice]

    # Apply the fixed padding
    target_matrix = np.pad(
        intermediate_matrix,
        pad_width=padding,
        mode='constant',
        constant_values=False
    )
    return target_matrix

def process_mesh_file(
    source_path: Path,
    target_folder: Path,
    num_orientations: int,
    grid_size: int,
    padding: int
):
    """
    Loads a single mesh, generates all oriented and voxelized versions,
    and saves them as .npy files.
    """
    try:
        # 1. Load and normalize the mesh ONCE
        mesh = trimesh.load(source_path, force='mesh')
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(1.0 / mesh.extents.max())
    except Exception as e:
        print(f"\nError loading {source_path.name}: {e}")
        return

    angle_increment_rad = (2 * np.pi) / num_orientations

    for i in range(num_orientations):
        # 2. Create a fresh copy for each rotation to avoid cumulative transforms
        mesh_to_rotate = mesh.copy()

        # 3. Rotate the mesh
        angle_rad = i * angle_increment_rad
        rotated_mesh = rotate_mesh(mesh_to_rotate, angle_rad)

        # 4. Voxelize and pad the rotated mesh
        voxel_grid = voxelize_and_pad(rotated_mesh, grid_size, padding)

        # 5. Save the final voxel grid
        angle_deg = round(np.rad2deg(angle_rad))
        base_name = source_path.stem
        # Use orientation index 'i' for the label
        new_filename = f"{base_name}_rot{i:02d}_{int(angle_deg)}deg.npy"
        target_path = target_folder / new_filename
        np.save(target_path, voxel_grid.astype(np.uint8))


def main():
    """Main function to drive the data preparation process."""
    # --- Parameters to configure ---
    parser = argparse.ArgumentParser(
        description="Preprocess ModelNet data by creating different orientations based on a predefined orientation map."
    )

    # --- Argument 1: The dataset name (Required) ---
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['modelnet10', 'modelnet40'], # Restricts input to these choices
        help="The name of the dataset to process (e.g., 'modelnet40')."
    )

    # --- Argument 2: Number of points (Optional, with a default value) ---
    parser.add_argument(
        '--grid_size',
        type=int,
        default=32, # If the user doesn't provide this, it will be 1024
        help="The number of points to sample from each mesh surface."
    )

    parser.add_argument(
        '--padding',
        type=int,
        default=1, # If the user doesn't provide this, it will be 1024
        help="The number of points to sample from each mesh surface."
    )

    args = parser.parse_args()
    DATASET_NAME = args.dataset  # Change to "modelnet40" if needed
    ROOT_DIR = Path(f"data/raw/{DATASET_NAME}")
    OUTPUT_DIR = Path(f"data/{DATASET_NAME}/VoxelizedOriented")
    GRID_SIZE = args.grid_size  # Final grid size (including padding)
    PADDING = args.padding  # Padding to apply on all sides
    # --------------------------------

    pose_plan = POSE_CONFIG[DATASET_NAME]
    print(f"Starting data preparation for {DATASET_NAME} with grid size {GRID_SIZE} and padding {PADDING} on each side")

    for class_name, num_orientations in pose_plan.items():
        print(f"\nProcessing class: {class_name} ({num_orientations} orientations)")

        for split in ['train', 'test']:
            source_folder = ROOT_DIR / class_name / split
            target_folder = OUTPUT_DIR / class_name / split
            target_folder.mkdir(parents=True, exist_ok=True)

            if not source_folder.exists():
                print(f"  - Warning: Directory not found, skipping: {source_folder}")
                continue

            files = list(source_folder.glob('*.off'))
            if not files:
                print(f"  - No .off files found in '{split}' split, skipping.")
                continue

            pbar = tqdm(files, desc=f"  - Processing '{split}' split")
            for source_path in pbar:
                process_mesh_file(
                    source_path,
                    target_folder,
                    num_orientations,
                    GRID_SIZE,
                    PADDING
                )

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
