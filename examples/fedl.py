import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from edl_losses import fedl_loss
from examples.lenet import LeNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5


def get_uncertainty_fedl(
    alpha: torch.Tensor,
    p: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """F-EDL inference: returns predicted class and TU/EU/AU decomposition.

    Arguments:
        alpha: Tensor of shape (batch_size, num_classes) representing evidence for each class.
        p: Tensor of shape (batch_size, num_classes) representing allocation probabilities.
        tau: Tensor of shape (batch_size,) representing dispersion.

    Returns:
        predicted class indices (Tensor of shape (batch_size,))
        Total Uncertainty (TU) values (Tensor of shape (batch_size,))
        Epistemic Uncertainty (EU) values (Tensor of shape (batch_size,))
        Aleatoric Uncertainty (AU) values (Tensor of shape (batch_size,))

    """
    tau = tau.view(-1, 1)
    alpha0 = alpha.sum(dim=-1, keepdim=True)
    denom = alpha0 + tau
    denom1 = denom + 1

    expected_pi = (alpha + tau * p) / denom

    term1 = (alpha + tau * p) * (alpha0 - alpha + tau * (1.0 - p)) / (denom**2 * denom1)
    term2 = tau**2 * p * (1.0 - p) / (denom * denom1)
    variance_pi = term1 + term2

    total_uncertainty = 1.0 - (expected_pi**2).sum(dim=-1)
    epistemic_uncertainty = variance_pi.sum(dim=-1)
    aleatoric_uncertainty = total_uncertainty - epistemic_uncertainty

    return expected_pi.argmax(dim=-1), total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


class FEDLModel(nn.Module):
    """F-EDL classifier with LeNet backbone. Outputs separate heads for alpha, p, and tau."""

    def __init__(self, num_classes: int = 10, mlp_hidden: int = 64) -> None:
        """Initialize the F-EDL model architecture."""
        super().__init__()
        self.backbone = LeNet()

        # Three separate heads
        self.head_alpha = nn.Linear(500, num_classes)  # evidence
        self.head_p = nn.Linear(500, num_classes)  # allocation probs
        self.head_tau = nn.Sequential(  # dispersion
            nn.Linear(500, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the F-EDL model.

        Arguments:
            x: Input tensor of shape (batch_size, 1, 28, 28) representing grayscale images.

        Returns:
            alpha: Tensor of shape (batch_size, num_classes) representing evidence for each class.
            p: Tensor of shape (batch_size, num_classes) representing allocation probabilities.
            tau: Tensor of shape (batch_size,) representing dispersion.

        """
        z = self.backbone(x)
        alpha = torch.exp(self.head_alpha(z))
        p = torch.softmax(self.head_p(z), dim=-1)
        tau = func.softplus(self.head_tau(z)).squeeze(-1)
        return alpha, p, tau


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = FEDLModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = fedl_loss(*model(x.to(DEVICE)), y.to(DEVICE))  # pyright: ignore[reportCallIssue]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[F-EDL] Epoch {epoch}/{NUM_EPOCHS}  loss={total_loss / len(train_loader):.4f}")

model.eval()
correct = total = 0
sum_tu_correct = sum_eu_correct = sum_au_correct = n_correct = 0
sum_tu_wrong = sum_eu_wrong = sum_au_wrong = n_wrong = 0

with torch.no_grad():
    for x, y in test_loader:
        pred, tu, eu, au = get_uncertainty_fedl(*model(x.to(DEVICE)))

        mask_correct = pred == y.to(DEVICE)
        mask_wrong = ~mask_correct

        correct += mask_correct.sum().item()
        total += y.to(DEVICE).size(0)

        sum_tu_correct += tu[mask_correct].sum().item()
        sum_eu_correct += eu[mask_correct].sum().item()
        sum_au_correct += au[mask_correct].sum().item()
        n_correct += mask_correct.sum().item()

        sum_tu_wrong += tu[mask_wrong].sum().item()
        sum_eu_wrong += eu[mask_wrong].sum().item()
        sum_au_wrong += au[mask_wrong].sum().item()
        n_wrong += mask_wrong.sum().item()

nc, nw = max(n_correct, 1), max(n_wrong, 1)

print(f"\n[F-EDL] Test accuracy: {correct / total:.4f}")
print(f"Correct preds - TU: {sum_tu_correct / nc:.4f} EU: {sum_eu_correct / nc:.4f} AU: {sum_au_correct / nc:.4f}")
print(f"Wrong preds - TU: {sum_tu_wrong / nw:.4f} EU: {sum_eu_wrong / nw:.4f} AU: {sum_au_wrong / nw:.4f}")
