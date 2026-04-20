import torch


def kl_div_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL divergence between Dirichlet(alpha_tilde) and Dirichlet(1,...,1).

    alpha_tilde is alpha with non-misleading evidence removed (eq. in Section 4).

    Arguments:
        alpha: parameters of the Dirichlet distribution, shape (B, K)

    Returns:
        KL divergence for each batch element, shape (B,)

    """
    num_classes = alpha.shape[-1]

    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    )
    second_term = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)

    return (first_term + second_term).squeeze(-1)
