import torch.nn as nn
from torchvision import models


class ImageRater(nn.Module):
    """Initializes the ImageRater model using a pre-trained ResNet base and a regression head."""

    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet (remove the classification layer)
        self.base = models.resnet18(weights="DEFAULT")
        self.base.fc = nn.Identity()  # Remove the final layer
        # Add regression head
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    def forward(self, x):
        """Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor containing images.

        Returns:
            Tensor: Output tensor containing the predicted scores.
        """
        features = self.base(x)
        return self.regressor(features)
