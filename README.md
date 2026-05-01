# edl-losses

A research library implementing *Evidential Deep Learning (EDL)* loss functions for uncertainty-aware classification in PyTorch.

EDL methods replace the standard softmax output with a distribution over class probabilities, typically a Dirichlet, allowing the model to express not just *which* class is most likely, but *how confident* it is in that prediction. This enables out-of-distribution detection, misclassification detection, and calibrated uncertainty estimates, all in a single forward pass with no sampling required.

The losses from the following papers are currently implemented:
- EDL: [Sensoy et al. (2018) — Evidential Deep Learning to Quantify Classification Uncertainty](http://arxiv.org/abs/1806.01768)
- GEN: [Sensoy et al. (2020) — Uncertainty-Aware Deep Classifiers using Generative Models](http://arxiv.org/abs/2006.04183)
- F-EDL: [Yoon and Kim (2026) — Uncertainty Estimation by Flexible Evidential Deep Learning](http://arxiv.org/abs/2510.18322)

See the [examples.ipynb](examples.ipynb) notebook, which compares the EDL losses on the task of classifying rotated MNIST digit "1" images.

## Installation

Requires Python 3.13+, Numpy 2.4+, and PyTorch 2.11+.

Using `uv`:
```bash
uv install edl-losses
```

Using `pip`:
```bash
pip install edl-losses
```

Or from source:
```bash
git clone https://github.com/LucaCtt/edl-losses
cd edl_losses
pip install -e .
```


## API

```python
from edl_losses.edl import EDLLoss, edl_inference
from edl_losses.gen import GENLoss, gen_inference
from edl_losses.fedl import FEDLLoss, fedl_inference
```


### `EDLLoss`

```python
edl_loss = EDLLoss(
    loss_type: Literal["sse", "ce", "mse"] = "sse",
    beta: float | Literal["anneal"] = "anneal",
    anneal_epochs: int = 10
)
edl_loss(
    logits: Tensor, # (B, K) raw network output, before any activation
    labels: Tensor, # (B,) ground truth class indices
    epoch: int | None, # current training epoch, required for KL annealing.
) -> Tensor # scalar
```

Implements Equations 3–5 of Sensoy et al. 2018.

Three base losses are available:
- `"sse"`: Sum of squares Bayes risk (recommended by the paper, most stable)
- `"ce"`: Cross-entropy Bayes risk
- `"mse"`: Type II Maximum Likelihood

When `beta` is `"anneal"`, the KL regularization term is weighted by `min(1, epoch / anneal_epochs)`, which gradually increases the influence of the KL term over the first `anneal_epochs` epochs. This helps prevent early underfitting when evidence is still low. When `beta` is set to a fixed value such as `1.0`, it applies a constant weight to the KL term throughout training. Setting `beta=0` disables the KL term entirely, which may lead to faster convergence but worse uncertainty estimates.


### `edl_inference`

```python
edl_inference(
    logits: Tensor, # (B, K) raw network output
) -> tuple[Tensor, Tensor, Tensor] # (predicted_classes (B,), uncertainty (B,), class_probs (B, K))
```

Uncertainty is `K / S` where `S = Σαₖ`. Values close to 1 indicate maximum uncertainty ("I don't know"); values close to 0 indicate high confidence.


### `GENLoss`

```python
gen_loss = GENLoss(
    beta: float | Literal["auto", "anneal"] = "auto", # KL weight; "auto" uses expected misclassification probability, "anneal" uses linear annealing
    anneal_epochs: int = 10, # number of epochs for KL annealing if `beta` is "anneal"
    eps:  float = 1e-8,
)
gen_loss(
    logits_in: Tensor, # (B, K) network output on in-distribution samples
    logits_out: Tensor, # (B, K) network output on OOD samples
    labels: Tensor, # (B,) ground truth class indices
    epoch: int | None = None, # current training epoch for KL annealing if `beta` is set to "anneal"
) -> Tensor # scalar
```

Implements Equations 4–6 of Sensoy et al. 2020. Requires OOD samples at training time. The loss has two components:

- **L1** — Bernoulli NCE loss: trains each output `fₖ` as a binary classifier distinguishing class-k samples from OOD samples.
- **L2** — KL regularizer: pushes the conditional Dirichlet over non-true classes toward uniform, weighted by `beta`.

When `beta="auto"`, the weight is set to `(1 - p̂ₖ)` per sample, i.e. the expected misclassification probability, which implements learned loss attenuation. When `beta="anneal"`, the KL term is weighted by `min(1, epoch / anneal_epochs)`, which gradually increases the influence of the KL term over the first `anneal_epochs` epochs. Setting `beta` to a fixed float applies a constant weight to the KL term throughout training. Setting `beta=0` disables the KL term, which may lead to faster convergence but worse uncertainty estimates.

> **Note:** You may want to clamp logits to a reasonable range (e.g. `[-10, 10]`) before passing to this loss to avoid numerical instability from the internal `exp()`.


### `gen_inference`

```python
gen_inference(
    logits: Tensor, # (B, K) raw network output
) -> tuple[Tensor, Tensor, Tensor] # (predicted_classes (B,), uncertainty (B,), class_probs (B, K))
```

This is the same as `edl_inference`, but `exp()` is used instead of `relu()` to compute evidence.


### `FEDLLoss`

```python
fedl_loss = FEDLLoss(
    eps: float = 1e-8,
)
fedl_loss(
    alpha: Tensor, # (B, K) concentration parameters, from exp() head
    p: Tensor, # (B, K) allocation probabilities, from softmax() head
    tau: Tensor, # (B,) or (B, 1) dispersion, from softplus() head
    labels: Tensor, # (B,)
) -> Tensor # scalar
```

Implements the objective from Section 3.2 and Appendix A.1 of Yoon & Kim 2026. The loss has two components:

- **L_MSE** — Expected MSE over the FD distribution, computed in closed form via FD moments.
- **L_reg** — Brier score on `p`, promoting well-calibrated allocation probabilities.

No KL term or annealing schedule is required. The model must expose three separate output heads:

```python
class MyFEDLModel(nn.Module):
    def forward(self, x):
        z = self.backbone(x)
        alpha = torch.exp(self.head_alpha(z))  # evidence, > 0
        p = torch.softmax(self.head_p(z), dim=-1)  # allocation probs, sums to 1
        tau = func.softplus(self.head_tau(z)).squeeze(-1)  # dispersion, > 0
        return alpha, p, tau
```


### `fedl_inference`

```python
fedl_inference(
    alpha: Tensor, # (B, K)
    p: Tensor, # (B, K)
    tau: Tensor, # (B,) or (B, 1)
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor] # (predicted_classes (B,), total_uncertainty (B,), class_probs (B, K))
```

Returns predicted classes, total uncertainty (TU), and expected class probabilities `E[π]`. TU is defined as `1 - Σ E[πₖ]²` and lies in `(0, 1]`.


## License

MIT. See [LICENSE](LICENSE).


## Author

Luca Cotti (<luca.cotti@unibs.it>)