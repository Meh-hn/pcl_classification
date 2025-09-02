# In file: pointcloud_classification_models/data_utils/modelnet_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
import os
import re

# --- 2. The Specialized Point Cloud Dataset ---


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset class for loading pre-processed ModelNet40
    point clouds saved as .npy files.
    """
    def __init__(self, root, split='train', n_points=1024, validation_split=0.2, transform=None, random_seed=42):
        self.root = Path(root)
        self.split = split
        self.n_points = n_points
        self.transform = transform

        # --- File Discovery and Train/Val Split ---
        # This logic correctly finds all categories and .npy files
        self.folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = {folder.name: i for i, folder in enumerate(self.folders)}
        self.shape_names = {i: name for name, i in self.classes.items()}
        self.files = []

        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")

        if self.split == 'test':
            source_dir_name = 'test'
            for category in self.classes.keys():
                source_dir = self.root / category / source_dir_name
                if source_dir.is_dir():
                    # Search for .npy files now
                    for file in source_dir.glob('*.npy'):
                        self.files.append((file, category))
        else:
            all_train_files = []
            source_dir_name = 'train'
            for category in self.classes.keys():
                source_dir = self.root / category / source_dir_name
                if source_dir.is_dir():
                    # Search for .npy files now
                    for file in source_dir.glob('*.npy'):
                        all_train_files.append((file, category))

            # This ensures your train/val split is always the same
            rng = random.Random(random_seed)
            rng.shuffle(all_train_files)

            split_idx = int(len(all_train_files) * (1 - validation_split))
            if self.split == 'train':
                self.files = all_train_files[:split_idx]
            else: # 'val'
                self.files = all_train_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, category = self.files[idx]
        label = self.classes[category]
        points = np.load(path) # Loads the (2048, 3) array

        # --- STEP 2: NORMALIZE THE POINTS ---
        # This is still essential, as the saved points are not normalized.
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 1e-6:
            points /= max_dist

        if self.split == 'train' and self.transform:
            points = self.transform(points)

        if len(points) >= self.n_points:
            indices = np.random.choice(len(points), self.n_points, replace=False)
        else: # Should not happen if all files have 2048 points, but good to have
            indices = np.random.choice(len(points), self.n_points, replace=True)
        points = points[indices, :]

        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).float()

        label = torch.tensor(label).long()

        return points, label


# --- 3. The Specialized Voxel Dataset ---
class VoxelizedOrientedDataset(Dataset):
    """
    Loads the ModelNet10 dataset from .npy files including different orientations.

    This class splits the original 'train' folder into new 'train' and 'val' sets,
    and uses the original 'test' folder for the 'test' set.
    """
    def __init__(self, root, split='train', validation_split=0.2, random_seed=42):
        """
        Args:
            root_dir (str): Path to the ModelNet10 directory.
            split (str): One of 'train', 'val', or 'test'.
            validation_split (float): The fraction of training data to use for validation.
            random_seed (int): Seed for shuffling to ensure consistent splits.
        """
        self.root_dir = root
        self.split = split
        self.file_list = []
        self.orientation_regex = re.compile(r'_rot(\d+)_')

        # Validate split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")
        # Class mapping for ModelNet
        self.folders = sorted([d for d in os.listdir(self.root_dir)])
        self.classes = {folder: i for i, folder in enumerate(self.folders)}

        # If the split is 'test', we just load from the 'test' folder
        if self.split == 'test':
            source_folder_name = 'test'
            for class_name, class_idx in self.classes.items():
                class_folder = os.path.join(self.root_dir, class_name, source_folder_name)
                if not os.path.isdir(class_folder): continue
                # The rest of the logic is the same for populating the file list
                self._populate_file_list(class_folder, class_idx)
        else:
            # For 'train' and 'val', we load everything from the 'train' folder first
            all_train_files = []
            source_folder_name = 'train'
            for class_name, class_idx in self.classes.items():
                class_folder = os.path.join(self.root_dir, class_name, source_folder_name)
                if not os.path.isdir(class_folder): continue
                # Populate a temporary list with all training files
                self._populate_file_list(class_folder, class_idx, target_list=all_train_files)

            # Shuffle the list for a random split
            random.seed(random_seed)
            random.shuffle(all_train_files)

            # Split the data
            split_idx = int(len(all_train_files) * (1 - validation_split))
            if self.split == 'train':
                self.file_list = all_train_files[:split_idx]
            else: # self.split == 'val'
                self.file_list = all_train_files[split_idx:]

    def _populate_file_list(self, folder_path, class_idx, target_list=None):
        """Helper function to find and add file info to a list."""
        if target_list is None:
            target_list = self.file_list

        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                match = self.orientation_regex.search(filename)
                if match:
                    orientation_idx = int(match.group(1))
                    target_list.append({
                        "path": os.path.join(folder_path, filename),
                        "class_idx": class_idx,
                        "orientation_idx": orientation_idx
                    })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        voxel_matrix = np.load(file_info["path"])
        voxel_tensor = torch.from_numpy(voxel_matrix).float().unsqueeze(0)
        class_label = torch.tensor(file_info["class_idx"], dtype=torch.long)
        orientation_label = torch.tensor(file_info["orientation_idx"], dtype=torch.long)

        return voxel_tensor, class_label, orientation_label