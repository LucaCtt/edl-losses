import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from edl_losses import gen_loss
from examples.edl import get_uncertainty_edl
from examples.lenet import LeNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5


class GENModel(nn.Module):
    """Same classifier as EDL. OOD samples are generated externally during training."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the GEN model architecture."""
        super().__init__()
        self.backbone = LeNet()
        self.classifier = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GEN model.

        Arguments:
            x: Input tensor of shape (batch_size, 1, 28, 28) representing grayscale images.

        Returns:
            Tensor of shape (batch_size, num_classes) representing raw logits for classification.

        """
        return self.classifier(self.backbone(x))


def _make_ood_samples(x: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise to push samples off-distribution.

    In production this would be your VAE+GAN generator from the paper.

    Arguments:
        x: Input tensor of shape (batch_size, 1, 28, 28) representing grayscale images.

    Returns:
        Tensor of the same shape as x, but with added noise to simulate OOD samples.

    """
    return torch.clamp(x + 0.3 * torch.randn_like(x), 0, 1)


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = GENModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x_ood = _make_ood_samples(x.to(DEVICE))
        optimizer.zero_grad()
        loss = gen_loss(model(x.to(DEVICE)), model(x_ood), y.to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[GEN] Epoch {epoch}/{NUM_EPOCHS}  loss={total_loss / len(train_loader):.4f}")

model.eval()
correct = total = 0
mean_uncertainty_correct = mean_uncertainty_wrong = 0
n_correct = n_wrong = 0

with torch.no_grad():
    for x, y in test_loader:
        pred, uncertainty, _ = get_uncertainty_edl(model(x.to(DEVICE)))

        mask_correct = pred == y.to(DEVICE)
        mask_wrong = ~mask_correct

        correct += mask_correct.sum().item()
        total += y.size(0)
        mean_uncertainty_correct += uncertainty[mask_correct].sum().item()
        mean_uncertainty_wrong += uncertainty[mask_wrong].sum().item()
        n_correct += mask_correct.sum().item()
        n_wrong += mask_wrong.sum().item()

print(f"\nTest accuracy: {correct / total:.4f}")
print(f"Mean uncertainty — correct: {mean_uncertainty_correct / max(n_correct, 1):.4f}")
print(f"Mean uncertainty — wrong:   {mean_uncertainty_wrong / max(n_wrong, 1):.4f}")
