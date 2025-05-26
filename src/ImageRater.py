import torch.nn as nn
from torchvision import models

class ImageRater(nn.Module):
    """Initializes the ImageRater model using a pre-trained ResNet base and a regression head."""

    def __init__(self, dropout_rate: float =0.2,model_type: str = 'resnet',input_size : int =512):
        """
        Initializes the ImageRater model using a pre-trained backbone (either ResNet or EfficientNet)
        and a regression head.

        Args:
            dropout_rate (float): The dropout rate applied in the regression head, used to prevent overfitting.
            model_type (str): The type of backbone model to use ('resnet' or 'efficientnet').
            input_size (int): The size of the input features to the first Linear layer of the regression head.
                              This determines the dimension of outputs from the base model which will be fed into the regressor.

        Raises:
            ValueError: If an unsupported model_type is provided.
        """
        super().__init__()
        # Load pre-trained model (remove the classification layer)
        if model_type == 'resnet':
            self.base = models.resnet152(weights="DEFAULT")
            self.output_size = 2048
        elif model_type == 'efficientnet':
            self.base = models.efficientnet_b5(weights="DEFAULT")
            self.output_size = 1000


        self.base.fc = nn.Identity()  # Remove the final layer
        # Add regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.output_size, input_size), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(input_size, 1)
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

    def update_dropout(self, new_dropout_rate: float):
        """Updates the dropout rate of the regression head.
        (For future updates)

        Args:
            new_dropout_rate (float): New dropout rate to set.
        """
        self.regressor[2] = nn.Dropout(new_dropout_rate)