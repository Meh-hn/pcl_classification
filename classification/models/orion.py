# In file: pointcloud_classification_models/models/orion.py

"""
Implementation of the ORION model, a 3D CNN for joint object
classification and orientation estimation from voxelized point clouds.
Sedaghat, N., Zolfaghari, M., Amiri, E., & Brox, T. (2016).  
Orientation-boosted Voxel Nets for 3D Object Recognition. (https://arxiv.org/abs/1604.03351)
"""
import torch
import torch.nn as nn

class ORION2Layer(nn.Module):
    """
    A 3D Convolutional Neural Network for multi-task learning.

    This model takes a 3D voxel grid as input and outputs two predictions:
    1. The object class.
    2. The object's orientation.
    """
    def __init__(self, num_classes: int, num_orientations: int, input_size: int = 32):
        """
        Initializes the ORION model layers.

        Args:
            num_classes (int): The number of object classes to predict.
            num_orientations (int): The number of discrete orientation bins to predict.
            input_size (int, optional): The size of one dimension of the input
                                        voxel grid. Defaults to 32.
        """
        super(ORION2Layer, self).__init__()

        # Feature extractor using 3D convolutions
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # A trick to dynamically calculate the flattened size after the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size, input_size, input_size)
            self._flattened_size = self.features(dummy_input).view(-1).shape[0]

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self._flattened_size, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        # Output heads
        self.class_head = nn.Linear(128, num_classes)
        self.orientation_head = nn.Linear(128, num_orientations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input voxel grid tensor.
                              Shape: (B, 1, D, H, W), e.g., (32, 1, 32, 32, 32)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - class_logits (torch.Tensor): Raw scores for each class.
                  Shape: (B, num_classes)
                - orientation_logits (torch.Tensor): Raw scores for each orientation.
                  Shape: (B, num_orientations)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.shared_fc(x)

        class_logits = self.class_head(x)
        orientation_logits = self.orientation_head(x)

        return class_logits, orientation_logits
    


    
class ORION_4Layer(nn.Module):
    """
    Implements the 4-layer "Extended Architecture" from Table 3 of the paper,
    but without Batch Normalization or Dropout.
    """
    def __init__(self, num_classes, num_orientations, input_size=32):
        super(ORION_4Layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self._flattened_size = 256 * 8 * 8 * 8

        # Fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self._flattened_size, 128),
            nn.LeakyReLU(True),
        )

        # Output heads
        self.class_head = nn.Linear(128, num_classes)
        self.orientation_head = nn.Linear(128, num_orientations)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        class_output = self.class_head(x)
        orientation_output = self.orientation_head(x)
        return class_output, orientation_output