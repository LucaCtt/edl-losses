import torch
import torch.nn.functional as func


class FEDLLoss(torch.nn.Module):
    """F-EDL loss module, implementing the loss from Yoon and Kim 2026 (http://arxiv.org/abs/2510.18322).

    This loss should be called with the raw network outputs (logits) before any activation,
    since the F-EDL formulation uses separate activations for alpha (evidence), p (allocation), and tau (dispersion).
    Use exp for alpha, softmax for p, and softplus for tau.

    You may want to clamp alpha and tau (e.g. between 1e-6 and 1e4) if you observe numerical issues.

    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the F-EDL loss module.

        Arguments:
            eps (float): small constant for numerical stability.

        """
        super().__init__()

        self._eps = eps

    def forward(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        tau: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the F-EDL loss.

        Arguments:
            alpha (Tensor): concentration parameters, shape (B, K), must be > 0 (use exp activation).
            p (Tensor): allocation probabilities, shape (B, K), must sum to 1 (use softmax).
            tau (Tensor): dispersion parameter, shape (B,) or (B, 1), must be > 0 (use softplus).
            labels (Tensor): ground truth class indices, shape (B,).
            eps (float): small constant for numerical stability.

        Returns:
            Tensor: Scalar loss.

        """
        num_classes = alpha.shape[-1]
        y = func.one_hot(labels, num_classes=num_classes).float()

        tau = tau.view(-1, 1)
        alpha0 = alpha.sum(dim=-1, keepdim=True)

        # Expected class probabilities under FD
        denom = alpha0 + tau + self._eps
        expected_pi = (alpha + tau * p) / denom

        # Variance of FD
        denom1 = denom + 1
        term1 = (expected_pi * (1.0 - expected_pi)) / denom1
        var_p = p * (1.0 - p)
        term2 = (tau**2 * var_p) / (denom * denom1 + self._eps)
        variance_pi = term1 + term2

        # Expected MSE over FD
        l_mse = ((y - expected_pi) ** 2 + variance_pi).sum(dim=-1)

        # Regularization term (Brier score)
        l_reg = ((y - p) ** 2).sum(dim=-1)

        return (l_mse + l_reg).mean()


def fedl_inference(
    alpha: torch.Tensor,
    p: torch.Tensor,
    tau: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """F-EDL inference: returns predicted class and TU/EU/AU decomposition.

    Arguments:
        alpha (Tensor): evidence for each class, shape (batch_size, num_classes).
        p (Tensor): allocation probabilities, shape (batch_size, num_classes).
        tau (Tensor): dispersion parameter, shape (batch_size,) or (batch_size, 1).
        eps (float): small constant for numerical stability.

    Returns:
        tuple: A tuple containing:
            - class_indices (Tensor): predicted class indices, shape (batch_size,).
            - uncertainty (Tensor): Total Uncertainty (TU) values, shape (batch_size,).
            - class_probabilities (Tensor): class probabilities, shape (batch_size, num_classes).

    """
    tau = tau.view(-1, 1)
    alpha0 = alpha.sum(dim=-1, keepdim=True)
    expected_pi = (alpha + tau * p) / (alpha0 + tau + eps)

    total_uncertainty = 1.0 - (expected_pi**2).sum(dim=-1)

    return expected_pi.argmax(dim=-1), total_uncertainty, expected_pi
