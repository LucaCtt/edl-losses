import torch
import torch.nn.functional as func


def fedl_loss(
    alpha: torch.Tensor,
    p: torch.Tensor,
    tau: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """F-EDL loss from Yoon and Kim 2026 (http://arxiv.org/abs/2510.18322).

    Arguments:
        alpha: concentration parameters, shape (B, K), must be > 0 (use exp activation)
        p:     allocation probabilities, shape (B, K), must sum to 1 (use softmax)
        tau:   dispersion parameter, shape (B,) or (B, 1), must be > 0 (use softplus)
        labels: ground truth class indices, shape (B,)

    Returns:
        Scalar loss.

    """
    num_classes = alpha.shape[-1]  # K
    y = func.one_hot(labels, num_classes=num_classes).float()  # (B, K)

    tau = tau.view(-1, 1)  # (B, 1) for broadcasting
    alpha0 = alpha.sum(dim=-1, keepdim=True)  # (B, 1) — Dirichlet strength

    # Expected class probabilities under FD: E[pi_k] = (alpha_k + tau * p_k) / (alpha0 + tau)
    denom = alpha0 + tau  # (B, 1)
    expected_pi = (alpha + tau * p) / denom  # (B, K)

    # Variance of FD: Var[pi_k] = term1 + term2
    denom1 = denom + 1  # (B, 1)
    term1 = (alpha + tau * p) * (alpha0 - alpha + tau * (1.0 - p)) / (denom**2 * denom1)  # (B, K)
    term2 = tau**2 * p * (1.0 - p) / (denom * denom1)  # (B, K)
    variance_pi = term1 + term2  # (B, K)

    l_mse = ((y - expected_pi) ** 2 + variance_pi).sum(dim=-1)  # (B,)

    # L_reg = Brier score on p: sum_k (y_k - p_k)^2
    l_reg = ((y - p) ** 2).sum(dim=-1)  # (B,)

    return (l_mse + l_reg).mean()
