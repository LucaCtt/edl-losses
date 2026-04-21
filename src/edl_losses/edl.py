import torch
import torch.nn.functional as func

from edl_losses.util import kl_div_dirichlet


def edl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    loss_type: str = "sse",  # "sse" | "ce" | "mse"
    kl_reg: bool = True,
) -> torch.Tensor:
    """Evidential Deep Learning loss, implementing Eqs. 3-5 from Sensoy et al. 2018 (http://arxiv.org/abs/1806.01768).

    This loss should be called with the raw network outputs (logits) BEFORE any activation,
    since the EDL formulation replaces the usual softmax with a ReLU to produce "evidence" for each class.

    Arguments:
        logits (Tensor): raw network output BEFORE any activation, shape (B, K).
        labels (Tensor): ground truth class indices, shape (B,)
        epoch (int): current training epoch (used for KL annealing)
        loss_type (str): which base loss to use
            - "mse": Type II Maximum Likelihood (Eq. 3).
            - "ce":  Cross-entropy Bayes risk   (Eq. 4).
            - "sse": Sum of squares Bayes risk  (Eq. 5) — recommended by paper.
        kl_reg (bool): whether to add KL regularization term.

    Returns:
        loss (Tensor): Scalar loss.

    """
    num_classes = logits.shape[-1]
    evidence = func.relu(logits)
    alpha = evidence + 1.0
    dirichlet_strength = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / dirichlet_strength

    # One-hot encode labels
    y = func.one_hot(labels, num_classes=num_classes).float()

    if loss_type == "mse":
        # Eq. 3 - Type II Maximum Likelihood
        loss = (y * (torch.log(dirichlet_strength) - torch.log(alpha))).sum(dim=-1)

    elif loss_type == "ce":
        # Eq. 4 - Cross-entropy Bayes risk (uses digamma)
        loss = (y * (torch.digamma(dirichlet_strength) - torch.digamma(alpha))).sum(dim=-1)

    elif loss_type == "sse":
        # Eq. 5 - Sum of squares Bayes risk (recommended)
        err = (y - p_hat) ** 2  # (B, K)
        var = p_hat * (1 - p_hat) / (dirichlet_strength + 1)
        loss = (err + var).sum(dim=-1)

    else:
        msg = f"Unknown loss_type: {loss_type}"
        raise ValueError(msg)

    # KL regularization (Section 4)
    if kl_reg:
        # Remove non-misleading evidence: alpha_tilde = y + (1 - y) * alpha
        # This zeroes out evidence for the correct class before penalizing
        alpha_tilde = y + (1.0 - y) * alpha
        kl = kl_div_dirichlet(alpha_tilde)

        # Annealing coefficient: ramps from 0 to 1 over first 10 epochs
        lambda_t = min(1.0, epoch / 10)

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
