from typing import Literal

import torch
import torch.nn.functional as func

from edl_losses.util import _kl_div_dirichlet


class EDLLoss(torch.nn.Module):
    """Evidential Deep Learning loss, implementing Eqs. 3-5 from Sensoy et al. 2018 (http://arxiv.org/abs/1806.01768).

    This loss should be called with the raw network outputs (logits) before any activation,
    since the EDL formulation replaces the usual softmax with a ReLU to produce "evidence" for each class.
    ReLU is already applied inside this loss.

    Arguments:
        loss_type (Literal["sse", "ce", "mse"]): which base loss to use
            - "sse": Sum of squares Bayes risk (Eq. 5) — recommended by paper.
            - "ce":  Cross-entropy Bayes risk (Eq. 4).
            - "mse": Type II Maximum Likelihood (Eq. 3).
        kl_reg (bool): whether to add KL regularization term.
        annealing_epochs (int): number of epochs over which to anneal the KL term.

    """

    def __init__(
        self,
        loss_type: Literal["sse", "ce", "mse"] = "sse",
        kl_reg: bool = True,
        annealing_epochs: int = 10,
    ) -> None:
        """Initialize the EDL loss module."""
        super().__init__()

        self.__loss_type = loss_type
        self.__kl_reg = kl_reg
        self.__annealing_epochs = annealing_epochs
        self.__current_epoch = 0

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        """Compute the EDL loss.

        Arguments:
            logits (Tensor): raw model outputs of shape (batch_size, num_classes).
            labels (Tensor): true class labels of shape (batch_size,).
            epoch (int, optional): current training epoch for KL annealing.
                If None, it will automatically increment the internal epoch counter on each call.

        Returns:
            Tensor: scalar loss value.

        """
        num_classes = logits.shape[-1]
        evidence = func.relu(logits)
        alpha = evidence + 1.0
        dirichlet_strength = alpha.sum(dim=-1, keepdim=True)
        p_hat = alpha / dirichlet_strength

        self.__current_epoch = epoch if epoch is not None else self.__current_epoch + 1

        # One-hot encode labels
        y = func.one_hot(labels, num_classes=num_classes).float()

        if self.__loss_type == "mse":
            # Eq. 3 - Type II Maximum Likelihood
            loss = (y * (torch.log(dirichlet_strength) - torch.log(alpha))).sum(dim=-1)
        elif self.__loss_type == "ce":
            # Eq. 4 - Cross-entropy Bayes risk (uses digamma)
            loss = (y * (torch.digamma(dirichlet_strength) - torch.digamma(alpha))).sum(dim=-1)
        else:
            # Eq. 5 - Sum of squares Bayes risk (recommended)
            err = (y - p_hat) ** 2  # (B, K)
            var = p_hat * (1 - p_hat) / (dirichlet_strength + 1)
            loss = (err + var).sum(dim=-1)

        # KL regularization (Section 4)
        if self.__kl_reg:
            # Remove non-misleading evidence: alpha_tilde = y + (1 - y) * alpha
            # This zeroes out evidence for the correct class before penalizing
            alpha_tilde = y + (1.0 - y) * alpha
            kl = _kl_div_dirichlet(alpha_tilde)

            # Annealing coefficient: ramps from 0 to 1 over the first `annealing_epochs` epochs
            lambda_t = min(1.0, self.__current_epoch / self.__annealing_epochs)

            loss = loss + lambda_t * kl

        return loss.mean()


def edl_inference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EDL inference: returns predicted class, uncertainty, and class probabilities.

    Arguments:
        logits (Tensor): Tensor of shape (batch_size, num_classes) representing raw logits for evidence.

    Returns:
        tuple: A tuple containing:
            - class_indices (Tensor): predicted class indices, shape (batch_size,).
            - uncertainty (Tensor): uncertainty values, shape (batch_size,).
            - class_probabilities (Tensor): class probabilities, shape (batch_size, num_classes).

    """
    evidence = func.relu(logits)
    alpha = evidence + 1.0
    dirichlet_strength = alpha.sum(dim=-1)
    p_hat = alpha / dirichlet_strength.unsqueeze(-1)
    num_classes = logits.shape[-1]
    uncertainty = num_classes / dirichlet_strength

    return p_hat.argmax(dim=-1), uncertainty, p_hat
