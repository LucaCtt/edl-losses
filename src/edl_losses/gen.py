import torch
import torch.nn.functional as func

from edl_losses.util import kl_div_dirichlet


def gen_classifier_loss(
    logits_in: torch.Tensor,
    logits_out: torch.Tensor,
    labels: torch.Tensor,
    beta: float | str = "auto",
) -> torch.Tensor:
    """Full GEN loss implementing Eq. 4-6 from Sensoy et al. 2020 (http://arxiv.org/abs/2006.04183).

    Arguments:
        logits_in: raw network outputs for in-distribution batch, shape (B, K)
        logits_out: raw network outputs for OOD batch, shape (B, K)
        labels:ground truth class indices, shape (B,)
        beta: regularization weight for L2 term. If "auto", uses the expected misclassification probability as in Eq. 6.

    Returns:
        Scalar loss.

    """
    if isinstance(beta, str) and beta != "auto":
        msg = f"Invalid beta value: {beta}. Must be a float or 'auto'."
        raise ValueError(msg)

    num_classes = logits_in.shape[-1]  # K
    y = func.one_hot(labels, num_classes=num_classes).float()  # (B, K)

    # L1 — Bernoulli NCE loss (Eq. 4)
    pos = func.logsigmoid(logits_in) * y
    neg = torch.log(1 - torch.sigmoid(logits_out) + 1e-8)
    l1 = -(pos.sum(dim=0) / (y.sum(dim=0) + 1e-8) + neg.mean(dim=0)).sum()

    # L2 — KL regularizer on misclassification Dirichlet (Eq. 5)
    evidence_in = torch.exp(logits_in)
    alpha_in = evidence_in + 1.0  # (B, K)
    p_hat_k = (alpha_in * y).sum(dim=-1)  # (B,)

    # Zero out true-class evidence, keep the rest — same spirit as alpha_tilde in EDL
    alpha_minus_k = (1.0 - y) * alpha_in + y * 1.0  # (B, K)

    kl = kl_div_dirichlet(alpha_minus_k)  # (B,)

    weight = (1.0 - p_hat_k).detach() if beta == "auto" else beta
    l2 = (weight * kl).mean()

    return l1 + l2
