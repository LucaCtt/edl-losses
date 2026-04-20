# EDL Losses for CSI-VAE

A research library implementing some **Evidential Deep Learning (EDL)** loss functions available in the literature.

Currently the losses from the following papers are implemented:
- [Sensoy et al. (2018)](http://arxiv.org/abs/1806.01768)
- [Sensoy et al. (2020)](http://arxiv.org/abs/2006.04183)
- [Yoon and Kim (2026)](http://arxiv.org/abs/2510.18322)

## Getting Started

### Prerequisites

- Python 3.13+

### Installation

Clone the parent repository and navigate to this module:

```bash
git clone <repo-url>
cd csi_vae/edl_losses
pip install -r requirements.txt
```

### Usage

```python
from losses import edl_loss

# Example usage
loss = edl_loss(predictions, targets, epoch)
loss.backward()
```

## License

MIT. See [LICENSE](/LICENSE).

## Author

Luca Cotti (<luca.cotti@unibs.it>)