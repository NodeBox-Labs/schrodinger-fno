"""Loss functions for complex wavefunction regression."""
import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """pred, target: (B, 2, N) real channels [Re, Im]. Returns mean relative L2."""
    diff = (pred - target).pow(2).sum(dim=1).sum(dim=-1).sqrt()  # (B,)
    norm = target.pow(2).sum(dim=1).sum(dim=-1).sqrt() + eps
    return (diff / norm).mean()


def norm_penalty(pred: torch.Tensor) -> torch.Tensor:
    """(‖ψ‖² - 1)² per sample averaged over batch. Discrete grid assumed unit dx."""
    norm2 = pred.pow(2).sum(dim=1).sum(dim=-1)  # (B,)
    return ((norm2 - 1.0) ** 2).mean()