import torch

from edl_losses.fedl import FEDLLoss, fedl_inference


def test_fedl_loss_supports_tau_as_vector_or_column() -> None:
    alpha = torch.tensor([[1.2, 2.3, 3.4], [2.0, 2.0, 2.0]], dtype=torch.float32)
    p = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5]], dtype=torch.float32)
    tau_vec = torch.tensor([0.8, 1.1], dtype=torch.float32)
    tau_col = tau_vec.view(-1, 1)
    labels = torch.tensor([1, 2], dtype=torch.int64)

    loss_fn = FEDLLoss()
    loss_vec = loss_fn(alpha, p, tau_vec, labels)
    loss_col = loss_fn(alpha, p, tau_col, labels)

    assert torch.allclose(loss_vec, loss_col, atol=1e-6)


def test_fedl_loss_backward_computes_gradients() -> None:
    alpha = torch.tensor([[2.0, 1.0, 3.0], [1.3, 2.2, 1.7]], dtype=torch.float32, requires_grad=True)
    p = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]], dtype=torch.float32, requires_grad=True)
    tau = torch.tensor([0.4, 1.2], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([1, 2], dtype=torch.int64)

    loss = FEDLLoss()(alpha, p, tau, labels)
    loss.backward()

    assert loss.ndim == 0
    assert alpha.grad is not None
    assert p.grad is not None
    assert tau.grad is not None
    assert alpha.grad.shape == alpha.shape
    assert p.grad.shape == p.shape
    assert tau.grad.shape == tau.shape
    assert torch.isfinite(alpha.grad).all()
    assert torch.isfinite(p.grad).all()
    assert torch.isfinite(tau.grad).all()


def test_fedl_loss_is_finite_for_small_positive_parameters() -> None:
    alpha = torch.full((2, 3), 1e-6, dtype=torch.float32)
    p = torch.tensor([[0.33, 0.33, 0.34], [0.2, 0.3, 0.5]], dtype=torch.float32)
    tau = torch.full((2,), 1e-6, dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.int64)

    loss = FEDLLoss(eps=1e-8)(alpha, p, tau, labels)
    assert torch.isfinite(loss)


def test_fedl_inference_known_values() -> None:
    alpha = torch.tensor([[2.0, 2.0], [3.0, 1.0]], dtype=torch.float32)
    p = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
    tau = torch.tensor([2.0, 2.0], dtype=torch.float32)

    pred, total_uncertainty, probs = fedl_inference(alpha, p, tau)

    expected_probs = torch.tensor([[0.5, 0.5], [0.5833333, 0.4166667]], dtype=torch.float32)
    expected_tu = torch.tensor([0.5, 1.0 - (0.5833333**2 + 0.4166667**2)], dtype=torch.float32)

    assert torch.equal(pred, torch.tensor([0, 0]))
    assert torch.allclose(probs, expected_probs, atol=1e-6)
    assert torch.allclose(total_uncertainty, expected_tu, atol=1e-6)


def test_fedl_inference_probabilities_sum_to_one_and_uncertainty_range() -> None:
    alpha = torch.tensor(
        [
            [1.2, 2.0, 3.1, 0.8],
            [4.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    p = torch.tensor(
        [
            [0.1, 0.2, 0.6, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    tau = torch.tensor([0.6, 1.5, 0.2], dtype=torch.float32)

    _, total_uncertainty, probs = fedl_inference(alpha, p, tau)

    assert probs.shape == alpha.shape
    assert total_uncertainty.shape == alpha.shape[:1]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(alpha.shape[0]), atol=1e-6)
    assert (total_uncertainty >= 0).all()
    assert (total_uncertainty <= 1).all()


def test_fedl_inference_higher_concentration_reduces_uncertainty() -> None:
    p = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    tau = torch.tensor([1.0], dtype=torch.float32)

    low_alpha = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    high_alpha = torch.tensor([[50.0, 30.0, 20.0]], dtype=torch.float32)

    _, u_low, _ = fedl_inference(low_alpha, p, tau)
    _, u_high, _ = fedl_inference(high_alpha, p, tau)

    assert u_high.item() < u_low.item()
