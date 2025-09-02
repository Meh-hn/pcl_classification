# In file: pointcloud_classification_models/models/voxnet.py
"""
Implementation of the VoxNet architecture used as a baseline in the paper.
This is identical to the shallow ORION model but with only the classification output.
"""
import torch
import torch.nn as nn

class VoxNet(nn.Module):
    """
    A 3D Convolutional Neural Network for voxelized objects.
    """

    def __init__(self, num_classes, input_size=32):
        super(VoxNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size, input_size, input_size)
            dummy_output = self.features(dummy_input)
            self._flattened_size = dummy_output.view(-1).shape[0]
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 128),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_output = self.classifier(x)

        return class_output