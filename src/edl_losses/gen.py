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

    def __init__(self, beta: float | Literal["auto"] = "auto", eps: float = 1e-8) -> None:
        """Initialize the GEN loss module.

        Arguments:
            beta (float | Literal["auto"]): regularization weight for L2 term.
                If "auto", uses the expected misclassification probability as in Eq. 6.
            eps (float): small constant for numerical stability.

        """
        super().__init__()

        self.__beta: float | Literal["auto"] = beta
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
        weight = (1.0 - p_hat_k).detach() if self.__beta == "auto" else self.__beta
        alpha_wrong = alpha_in[~y.bool()].view(logits_in.shape[0], num_classes - 1)
        l2 = (weight * _kl_div_dirichlet(alpha_wrong)).mean()

        # Overall loss (Eq. 6)
        return l1 + l2
