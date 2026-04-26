from typing import Literal

import torch
import torch.nn.functional as func
from torch import nn

from edl_losses.util import _kl_div_dirichlet


class GENLoss(nn.Module):
    """Full GEN loss implementing Eq. 4-6 from Sensoy et al. 2020 (http://arxiv.org/abs/2006.04183).

    You might want to clamp input logits to a reasonable range (e.g. [-10, 10])
    if you observe numerical instability during training.

    """

    def __init__(
        self,
        beta: float | Literal["auto", "anneal"] = "auto",
        anneal_epochs: int = 10,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the GEN loss module.

        Arguments:
            beta (float | Literal["auto", "anneal"]): regularization weight for L2 term.
                If "auto", uses the expected misclassification probability as in Eq. 6.
                If "anneal", linearly anneals from 0 to 1 over the first `anneal_epochs` epochs.
                Otherwise, uses the specified constant value. Set to 0 to disable the L2 term.
            anneal_epochs (int): number of epochs over which to anneal the beta parameter if `beta` is set to "anneal".
            eps (float): small constant for numerical stability.

        """
        super().__init__()

        self.__beta: float | Literal["auto", "anneal"] = beta
        self.__anneal_epochs = anneal_epochs
        self.__eps = eps

    def forward(self, logits_in: torch.Tensor, logits_out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the GEN loss.

        Arguments:
            logits_in (Tensor): raw network outputs for in-distribution batch, shape (B, K)
            logits_out (Tensor): raw network outputs for OOD batch, shape (B, K)
            labels (Tensor): ground truth class indices, shape (B,)

        Returns:
            Tensor: Scalar loss.

        """
        num_classes = logits_in.shape[-1]
        y = func.one_hot(labels, num_classes=num_classes).float()

        # L1 — Bernoulli NCE loss (Eq. 4)
        pos_per_class = (func.logsigmoid(logits_in) * y).sum(dim=0) / (y.sum(dim=0) + self.__eps)
        neg_per_class = func.logsigmoid(-logits_out).mean(dim=0)
        l1 = -(pos_per_class + neg_per_class).mean()

        # L2 — KL regularizer on misclassification Dirichlet (Eq. 5)
        evidence_in = torch.exp(logits_in)
        alpha_in = evidence_in + 1.0
        p_hat_k = (alpha_in * y).sum(dim=-1) / alpha_in.sum(dim=-1)
        alpha_minus_k = (1.0 - y) * alpha_in + y * 1.0  # (B, K)

        if self.__beta == "auto":
            weight = (1.0 - p_hat_k).detach()
        elif self.__beta == "anneal":
            weight = min(1.0, self.__anneal_epochs / self.__anneal_epochs)
        else:
            weight = self.__beta

        l2 = (weight * _kl_div_dirichlet(alpha_minus_k)).mean()

        # Overall loss (Eq. 6)
        return l1 + l2


def gen_inference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GEN inference outputs: predicted class, uncertainty, and class probabilities.

    Arguments:
        logits (Tensor): raw network outputs of shape (batch_size, num_classes).

    Returns:
        tuple[Tensor, Tensor, Tensor]: predicted class indices, uncertainty scores, and class probabilities.

    """
    evidence = torch.exp(logits)
    alpha = evidence + 1.0
    dirichlet_strength = alpha.sum(dim=-1)
    p_hat = alpha / dirichlet_strength.unsqueeze(-1)
    num_classes = logits.shape[-1]
    uncertainty = num_classes / dirichlet_strength

    return p_hat.argmax(dim=-1), uncertainty, p_hat
