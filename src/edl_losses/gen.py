import torch
import torch.nn.functional as func

from edl_losses.util import kl_div_dirichlet


def gen_loss(
    logits_in: torch.Tensor,
    logits_out: torch.Tensor,
    labels: torch.Tensor,
    beta: float | str = "auto",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full GEN loss implementing Eq. 4-6 from Sensoy et al. 2020 (http://arxiv.org/abs/2006.04183).

    You might want to clamp input logits to a reasonable range (e.g. [-10, 10])
    if you observe numerical instability during training.

    Arguments:
        logits_in (Tensor): raw network outputs for in-distribution batch, shape (B, K)
        logits_out (Tensor): raw network outputs for OOD batch, shape (B, K)
        labels (Tensor): ground truth class indices, shape (B,)
        beta (float | str): regularization weight for L2 term.
            If "auto", uses the expected misclassification probability as in Eq. 6.
        eps (float): small constant for numerical stability.

    Returns:
        loss (Tensor): Scalar loss.

    """
    if isinstance(beta, str) and beta != "auto":
        msg = f"Invalid beta value: {beta}. Must be a float or 'auto'."
        raise ValueError(msg)

    num_classes = logits_in.shape[-1]
    y = func.one_hot(labels, num_classes=num_classes).float()

    # L1 — Bernoulli NCE loss (Eq. 4)
    pos_per_class = (func.logsigmoid(logits_in) * y).sum(dim=0) / (y.sum(dim=0) + eps)
    neg_per_class = torch.log(1 - torch.sigmoid(logits_out) + eps).mean(dim=0)
    l1 = -(pos_per_class + neg_per_class).mean()

    # L2 — KL regularizer on misclassification Dirichlet (Eq. 5)
    evidence_in = torch.exp(logits_in)
    alpha_in = evidence_in + 1.0
    p_hat_k = (alpha_in * y).sum(dim=-1) / alpha_in.sum(dim=-1)
    alpha_minus_k = (1.0 - y) * alpha_in + y * 1.0
    kl = kl_div_dirichlet(alpha_minus_k)
    weight = (1.0 - p_hat_k).detach() if beta == "auto" else beta
    l2 = (weight * kl).mean()

    # Overall loss (Eq. 6)
    return l1 + l2
