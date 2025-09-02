# In classification/utils/pcl_transforms.py

import torch
import numpy as np
import random

class PointCloudToTensor:
    """Converts a numpy.ndarray to a torch.Tensor."""
    def __call__(self, points):
        """
        Args:
            points (np.ndarray): The point cloud to be converted to a tensor.
        Returns:
            torch.Tensor: The converted point cloud.
        """
        return torch.from_numpy(points).float()

class RandomRotation:
    """
    Applies a random rotation to a point cloud.
    """
    def __init__(self, axis='y', angle_range=(-180, 180)):
        """
        Args:
            axis (str): The axis to rotate around ('x', 'y', or 'z').
            angle_range (tuple): The range of angles in degrees for the random rotation.
        """
        self.axis = axis.lower()
        self.angle_range = angle_range
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def __call__(self, points):
        """
        Args:
            points (np.ndarray): The point cloud to rotate, with shape (N, 3).
        Returns:
            np.ndarray: The rotated point cloud.
        """
        angle_deg = random.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = np.deg2rad(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        if self.axis == 'x':
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, c, -s],
                                        [0, s, c]])
        elif self.axis == 'y':
            rotation_matrix = np.array([[c, 0, s],
                                        [0, 1, 0],
                                        [-s, 0, c]])
        else: # axis == 'z'
            rotation_matrix = np.array([[c, -s, 0],
                                        [s, c, 0],
                                        [0, 0, 1]])

        # Apply rotation: (N, 3) @ (3, 3) -> (N, 3)
        return points @ rotation_matrix.T


class RandomScaling:
    """
    Applies a random isotropic scaling to a point cloud.
    """
    def __init__(self, scale_low=0.8, scale_high=1.25):
        """
        Args:
            scale_low (float): The minimum scaling factor.
            scale_high (float): The maximum scaling factor.
        """
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, points):
        """
        Args:
            points (np.ndarray): The point cloud to scale, with shape (N, 3).
        Returns:
            np.ndarray: The scaled point cloud.
        """
        scale = random.uniform(self.scale_low, self.scale_high)
        return points * scale


class RandomJitter:
    """
    Applies random jitter (small translation) to each point in a point cloud.
    """
    def __init__(self, sigma=0.01, clip=0.05):
        """
        Args:
            sigma (float): The standard deviation of the Gaussian noise.
            clip (float): The maximum value to clip the noise to.
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points):
        """
        Args:
            points (np.ndarray): The point cloud to jitter, with shape (N, 3).
        Returns:
            np.ndarray: The jittered point cloud.
        """
        N, C = points.shape
        noise = np.random.normal(0, self.sigma, (N, C))
        noise = np.clip(noise, -self.clip, self.clip)
        return points + noise


class RandomPointDropout:
    """
    Randomly drops a certain percentage of points from a point cloud.
    """
    def __init__(self, p=0.2):
        """
        Args:
            p (float): The probability of dropping a point (dropout ratio).
                       Value should be between 0 and 1.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p

    def __call__(self, points):
        """
        Args:
            points (np.ndarray): The point cloud, shape (N, 3).
        Returns:
            np.ndarray: The point cloud after dropout. Shape will be (N*(1-p), 3).
        """
        if self.p == 0.0:
            return points

        num_points = points.shape[0]
        # Create a random mask of points to keep
        mask = np.random.choice([True, False], size=num_points, p=[1 - self.p, self.p])
        return points[mask, :]

      # Add this new class to your pointcloud_transforms.py file
