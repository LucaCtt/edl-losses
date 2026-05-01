import pytest
import torch

import edl_losses.edl as edl_module
from edl_losses.edl import EDLLoss, edl_inference


def test_edl_loss_anneal_requires_epoch() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    loss_fn = EDLLoss(beta="anneal", anneal_epochs=10)

    with pytest.raises(ValueError, match="Epoch must be provided"):
        loss_fn(logits, labels, epoch=None)


def test_edl_loss_anneal_weight_applied_and_saturates(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[1.0, 0.0, 0.5], [0.2, 0.3, 0.1]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.int64)

    def _constant_kl(alpha: torch.Tensor) -> torch.Tensor:
        return torch.full((alpha.shape[0],), 2.0, dtype=alpha.dtype, device=alpha.device)

    monkeypatch.setattr(edl_module, "kl_div_dirichlet", _constant_kl)

    loss_fn = EDLLoss(loss_type="sse", beta="anneal", anneal_epochs=10)

    base = EDLLoss(loss_type="sse", beta=0.0)(logits, labels, epoch=1)
    early = loss_fn(logits, labels, epoch=2)
    late = loss_fn(logits, labels, epoch=100)

    assert torch.allclose(early, base + 0.2 * 2.0, atol=1e-6)
    assert torch.allclose(late, base + 1.0 * 2.0, atol=1e-6)


def test_edl_loss_negative_beta_skips_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[0.4, -0.2, 0.6]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.int64)

    def _fail_if_called(_: torch.Tensor) -> torch.Tensor:
        msg = "KL should not be called when lambda_t <= 0"
        raise AssertionError(msg)

    monkeypatch.setattr(edl_module, "kl_div_dirichlet", _fail_if_called)

    no_kl = EDLLoss(loss_type="ce", beta=0.0)(logits, labels, epoch=1)
    negative_beta = EDLLoss(loss_type="ce", beta=-0.5)(logits, labels, epoch=1)

    assert torch.allclose(no_kl, negative_beta, atol=1e-6)


def test_edl_loss_unknown_loss_type_falls_back_to_sse() -> None:
    logits = torch.tensor([[0.2, 1.4, -0.8], [0.5, 0.7, 0.1]], dtype=torch.float32)
    labels = torch.tensor([1, 0], dtype=torch.int64)

    reference = EDLLoss(loss_type="sse", beta=0.0)(logits, labels, epoch=1)
    fallback = EDLLoss(loss_type="not-a-real-loss", beta=0.0)(logits, labels, epoch=1) # pyright: ignore[reportArgumentType]

    assert torch.allclose(reference, fallback, atol=1e-6)


def test_edl_loss_backward_computes_gradients() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.3], [1.5, 0.4, -2.0]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([2, 0], dtype=torch.int64)

    loss = EDLLoss(loss_type="mse", beta=1.0)(logits, labels, epoch=1)
    loss.backward()

    assert loss.ndim == 0
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()


def test_edl_inference_known_values() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    pred, uncertainty, probs = edl_inference(logits)

    expected_probs = torch.tensor(
        [
            [0.5, 0.25, 0.25],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=torch.float32,
    )
    expected_uncertainty = torch.tensor([0.75, 1.0], dtype=torch.float32)

    assert torch.equal(pred, torch.tensor([0, 0]))
    assert torch.allclose(probs, expected_probs, atol=1e-6)
    assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6)


def test_edl_inference_probabilities_sum_to_one_and_uncertainty_range() -> None:
    logits = torch.tensor(
        [
            [2.5, -3.0, 0.1, 1.2],
            [-1.0, -0.5, -2.0, -3.0],
            [10.0, 8.0, 6.0, 4.0],
        ],
        dtype=torch.float32,
    )
    _, uncertainty, probs = edl_inference(logits)

    assert probs.shape == logits.shape
    assert uncertainty.shape == logits.shape[:1]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(logits.shape[0]), atol=1e-6)
    assert (uncertainty > 0).all()
    assert (uncertainty <= 1).all()


def test_edl_inference_higher_evidence_reduces_uncertainty() -> None:
    low_evidence_logits = torch.zeros((1, 3), dtype=torch.float32)
    high_evidence_logits = torch.full((1, 3), 10.0, dtype=torch.float32)

    _, low_u, _ = edl_inference(low_evidence_logits)
    _, high_u, _ = edl_inference(high_evidence_logits)

    assert high_u.item() < low_u.item()
