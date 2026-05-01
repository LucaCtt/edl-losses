import pytest
import torch
import torch.nn.functional as func

import edl_losses.gen as gen_module
from edl_losses.gen import GENLoss, gen_inference
from edl_losses.util import kl_div_dirichlet


def _manual_l1(
    logits_in: torch.Tensor,
    logits_out: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    num_classes = logits_in.shape[-1]
    y = func.one_hot(labels, num_classes=num_classes).float()
    pos_per_class = (func.logsigmoid(logits_in) * y).sum(dim=0) / (y.sum(dim=0) + eps)
    neg_per_class = func.logsigmoid(-logits_out).mean(dim=0)
    return -(pos_per_class + neg_per_class).mean()


@pytest.mark.parametrize("beta", [0.0, -0.5])
def test_gen_loss_beta_leq_zero_returns_l1_and_skips_kl(beta: float, monkeypatch: pytest.MonkeyPatch) -> None:
    logits_in = torch.tensor([[0.3, -0.2, 1.0], [0.1, 0.4, -0.7]], dtype=torch.float32)
    logits_out = torch.tensor([[-0.4, 0.2, 0.9], [0.3, -0.8, 0.1]], dtype=torch.float32)
    labels = torch.tensor([2, 1], dtype=torch.int64)

    def _fail_if_called(_: torch.Tensor) -> torch.Tensor:
        msg = "KL should not be called when beta <= 0"
        raise AssertionError(msg)

    monkeypatch.setattr(gen_module, "kl_div_dirichlet", _fail_if_called)

    loss = GENLoss(beta=beta)(logits_in, logits_out, labels, epoch=5)
    expected_l1 = _manual_l1(logits_in, logits_out, labels)

    assert torch.allclose(loss, expected_l1, atol=1e-6)


def test_gen_loss_constant_beta_applies_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    logits_in = torch.tensor([[0.2, 0.1, -0.3], [0.8, -0.6, 0.4]], dtype=torch.float32)
    logits_out = torch.tensor([[0.1, -0.2, 0.7], [-0.5, 0.3, -0.1]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.int64)

    def _constant_kl(alpha: torch.Tensor) -> torch.Tensor:
        return torch.full((alpha.shape[0],), 2.0, dtype=alpha.dtype, device=alpha.device)

    monkeypatch.setattr(gen_module, "kl_div_dirichlet", _constant_kl)

    base = _manual_l1(logits_in, logits_out, labels)
    loss = GENLoss(beta=0.4)(logits_in, logits_out, labels)

    assert torch.allclose(loss, base + 0.8, atol=1e-6)


def test_gen_loss_anneal_requires_epoch() -> None:
    logits_in = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    logits_out = torch.tensor([[0.0, -0.2, 0.4]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(ValueError, match="Epoch must be provided"):
        GENLoss(beta="anneal", anneal_epochs=10)(logits_in, logits_out, labels, epoch=None)


def test_gen_loss_anneal_weight_applied_and_saturates(monkeypatch: pytest.MonkeyPatch) -> None:
    logits_in = torch.tensor([[0.5, -0.4, 0.1], [0.3, 0.2, -0.5]], dtype=torch.float32)
    logits_out = torch.tensor([[0.1, -0.1, 0.2], [-0.3, 0.4, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    def _constant_kl(alpha: torch.Tensor) -> torch.Tensor:
        return torch.full((alpha.shape[0],), 3.0, dtype=alpha.dtype, device=alpha.device)

    monkeypatch.setattr(gen_module, "kl_div_dirichlet", _constant_kl)

    loss_fn = GENLoss(beta="anneal", anneal_epochs=10)
    base = _manual_l1(logits_in, logits_out, labels)

    early = loss_fn(logits_in, logits_out, labels, epoch=2)
    late = loss_fn(logits_in, logits_out, labels, epoch=50)

    assert torch.allclose(early, base + 0.2 * 3.0, atol=1e-6)
    assert torch.allclose(late, base + 1.0 * 3.0, atol=1e-6)


def test_gen_loss_auto_matches_manual_formula() -> None:
    logits_in = torch.tensor([[0.2, -0.1, 0.7], [1.0, -0.4, 0.3]], dtype=torch.float32)
    logits_out = torch.tensor([[-0.6, 0.5, 0.1], [0.2, -0.7, 0.9]], dtype=torch.float32)
    labels = torch.tensor([2, 0], dtype=torch.int64)

    num_classes = logits_in.shape[-1]
    y = func.one_hot(labels, num_classes=num_classes).float()

    l1 = _manual_l1(logits_in, logits_out, labels)

    evidence_in = torch.exp(logits_in)
    alpha_in = evidence_in + 1.0
    p_hat_k = (alpha_in * y).sum(dim=-1) / alpha_in.sum(dim=-1)
    alpha_minus_k = (1.0 - y) * alpha_in + y * 1.0

    expected = l1 + ((1.0 - p_hat_k) * kl_div_dirichlet(alpha_minus_k)).mean()

    actual = GENLoss(beta="auto")(logits_in, logits_out, labels)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_gen_loss_handles_missing_classes_in_batch() -> None:
    logits_in = torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.1, 0.2]], dtype=torch.float32)
    logits_out = torch.tensor([[-0.2, 0.5, -0.3], [0.0, -0.6, 0.4]], dtype=torch.float32)
    labels = torch.tensor([0, 0], dtype=torch.int64)

    loss = GENLoss(beta=0.0)(logits_in, logits_out, labels)
    assert torch.isfinite(loss)


def test_gen_loss_backward_computes_gradients() -> None:
    logits_in = torch.tensor([[0.2, -0.3, 0.7], [0.5, 0.1, -0.4]], dtype=torch.float32, requires_grad=True)
    logits_out = torch.tensor([[-0.1, 0.4, -0.6], [0.3, -0.2, 0.8]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([2, 1], dtype=torch.int64)

    loss = GENLoss(beta=0.3)(logits_in, logits_out, labels)
    loss.backward()

    assert loss.ndim == 0
    assert logits_in.grad is not None
    assert logits_out.grad is not None
    assert logits_in.grad.shape == logits_in.shape
    assert logits_out.grad.shape == logits_out.shape
    assert torch.isfinite(logits_in.grad).all()
    assert torch.isfinite(logits_out.grad).all()


def test_gen_inference_known_values() -> None:
    logits = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    pred, uncertainty, probs = gen_inference(logits)

    e = torch.exp(torch.tensor(1.0))
    expected_probs = torch.tensor(
        [
            [0.5, 0.5],
            [(e + 1.0) / (e + 3.0), 2.0 / (e + 3.0)],
        ],
        dtype=torch.float32,
    )
    expected_uncertainty = torch.tensor([2.0 / 4.0, 2.0 / (e + 3.0)], dtype=torch.float32)

    assert torch.equal(pred, torch.tensor([0, 0]))
    assert torch.allclose(probs, expected_probs, atol=1e-6)
    assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6)


def test_gen_inference_probabilities_sum_to_one_and_uncertainty_range() -> None:
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, 1.2],
            [-2.0, -1.5, -3.0, -4.0],
            [8.0, 7.0, 6.0, 5.0],
        ],
        dtype=torch.float32,
    )

    _, uncertainty, probs = gen_inference(logits)

    assert probs.shape == logits.shape
    assert uncertainty.shape == logits.shape[:1]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(logits.shape[0]), atol=1e-6)
    assert (uncertainty > 0).all()
    assert (uncertainty <= 1).all()


def test_gen_inference_higher_evidence_reduces_uncertainty() -> None:
    low_evidence_logits = torch.full((1, 3), -5.0, dtype=torch.float32)
    high_evidence_logits = torch.full((1, 3), 5.0, dtype=torch.float32)

    _, low_u, _ = gen_inference(low_evidence_logits)
    _, high_u, _ = gen_inference(high_evidence_logits)

    assert high_u.item() < low_u.item()
