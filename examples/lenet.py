import torch
from torch import nn


class LeNet(nn.Module):
    """Standard LeNet-5 for MNIST. Returns flat features before classifier."""

    def __init__(self) -> None:
        """Initialize the LeNet architecture."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(800, 500), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LeNet architecture.

        Arguments:
            x: Input tensor of shape (batch_size, 1, 28, 28) representing grayscale images.

        Returns:
            Tensor of shape (batch_size, 500) representing the features before the classifier.

        """
        return self.fc(self.features(x).flatten(1))
