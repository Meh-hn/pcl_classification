import os
import numpy as np
import trimesh
from tqdm import tqdm
from pathlib import Path
import argparse

def normalize_point_cloud(pc):
    """Center and scale point cloud to fit inside unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= max_dist
    return pc

def process_and_save_modelnet40_offs(
    root_dir,
    output_dir,
    num_points=2048,
    seed=42
):
    """
    Reads all .off files in ModelNet40, samples points from surface,
    normalizes them, and saves as .npy files.

    Args:
        root_dir (str): Path to ModelNet40 dataset root.
        output_dir (str): Path to save preprocessed point clouds.
        num_points (int): Number of points to sample from surface.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for cls in all_classes:
        print(f"Processing class: {cls}")
        class_dir = os.path.join(root_dir, cls)
        out_class_dir = os.path.join(output_dir, cls)
        os.makedirs(out_class_dir, exist_ok=True)

        for split in ['train', 'test']:
            split_dir = os.path.join(class_dir, split)
            out_split_dir = os.path.join(out_class_dir, split)
            os.makedirs(out_split_dir, exist_ok=True)

            for fname in tqdm(os.listdir(split_dir), desc=f"{cls}/{split}"):
                if fname.endswith('.off'):
                    in_path = os.path.join(split_dir, fname)
                    out_path = os.path.join(out_split_dir, fname.replace('.off', '.npy'))

                    mesh = trimesh.load(in_path)
                    if not isinstance(mesh, trimesh.Trimesh):
                        continue

                    points, _ = trimesh.sample.sample_surface(mesh, num_points)
                    points = normalize_point_cloud(points)
                    np.save(out_path, points)

    print("✅ Preprocessing complete. All point clouds saved.")

def main():
    """Main function to parse arguments and run the data preparation."""

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
        '--num_points',
        type=int,
        default=1024, # If the user doesn't provide this, it will be 1024
        help="The number of points to sample from each mesh surface."
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42, # If the user doesn't provide this, it will be 42
        help="The number of points to sample from each mesh surface."
    )

    args = parser.parse_args()
    DATASET_NAME = args.dataset  # Change to "modelnet40" if needed
    NUM_POINTS = args.num_points  # Number of points to sample from each mesh
    SEED = args.seed  # Random seed for reproducibility

    # Define paths based on the project structure
    ROOT_DIR = Path(f"data/raw/{DATASET_NAME}")
    OUTPUT_DIR = Path(f"data/{DATASET_NAME}/PointClouds")

    process_and_save_modelnet40_offs(
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR,
        num_points=NUM_POINTS,
        seed=SEED
    )

if __name__ == "__main__":
    main()
