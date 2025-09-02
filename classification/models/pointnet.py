"""
Implementation of the PointNet model for 3D point cloud classification.

This file contains the main PointNet class and the T-Net helper module, which
is used for spatial and feature transformations.

Based on the paper: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
by Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    A Transformation Network (T-Net) for spatial or feature alignment.

    This network learns a transformation matrix to align the input point cloud
    (k=3 for spatial alignment) or its features (k=64 for feature alignment).
    """
    def __init__(self, k: int = 3):
        """
        Initializes the T-Net layers.

        Args:
            k (int): The dimension of the input features. 3 for input points,
                     64 for intermediate features.
        """
        super(TNet, self).__init__()
        self.k = k

        # Shared MLP layers (implemented as 1D convolutions)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the T-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (B, k, N), where B is the
                              batch size, k is the feature dimension, and N is
                              the number of points.

        Returns:
            torch.Tensor: The learned transformation matrix of shape (B, k, k).
        """
        batch_size = x.size(0)

        # Pass through shared MLP layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling to get global features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Pass through fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as an identity matrix and add the learned transformation
        # This helps stabilize training at the beginning.
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k)
        iden = iden.repeat(batch_size, 1)
        x = x + iden

        # Reshape to a k x k matrix
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    """
    The PointNet model for 3D classification.
    """
    def __init__(self, num_classes: int = 10, feature_transform: bool = False):
        """
        Initializes the PointNet model layers.

        Args:
            num_classes (int): The number of classes to predict.
            feature_transform (bool): If True, use a T-Net to align features.
        """
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform

        # T-Net for input alignment (3x3 transformation)
        self.tnet = TNet(k=3)

        # Shared MLP layers for feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Optional T-Net for feature alignment (64x64 transformation)
        if self.feature_transform:
            self.ftnet = TNet(k=64)

        # Classification head (fully connected layers)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Defines the forward pass of the PointNet model.

        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, 3, N),
                              where B is batch size, 3 is for (x,y,z) coords,
                              and N is the number of points.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - log_softmax (torch.Tensor): The log-softmax probabilities for
                  each class. Shape: (B, num_classes).
                - trans_feat (torch.Tensor | None): The feature transformation
                  matrix from the feature T-Net. Used for a regularization loss.
                  Returns None if feature_transform is False.
        """
        # 1. Input Transformation
        trans = self.tnet(x)
        # Apply the learned transformation matrix to the input points
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans) # Batch-wise matrix multiplication
        x = x.transpose(2, 1)

        # 2. First block of shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))

        # 3. Feature Transformation (Optional)
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.ftnet(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # 4. Second block of shared MLPs
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # Note: No ReLU here before max pooling

        # 5. Symmetric Function (Max Pooling) to get global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # 6. Classification Head
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # 7. Output
        # The paper uses log_softmax for the NLLLoss function
        return F.log_softmax(x, dim=1), trans_feat