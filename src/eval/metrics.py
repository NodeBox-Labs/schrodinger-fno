"""Evaluation metrics for quantum wavefunctions."""
import torch


def _normsq(psi: torch.Tensor) -> torch.Tensor:
    """psi: (B, 2, N) real -> (B,) L2² (discrete)."""
    return psi.pow(2).sum(dim=1).sum(dim=-1)


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = (pred - target).pow(2).sum(dim=1).sum(dim=-1).sqrt()
    denom = target.pow(2).sum(dim=1).sum(dim=-1).sqrt() + eps
    return (diff / denom).mean()


def norm_drift(pred: torch.Tensor) -> torch.Tensor:
    """|‖ψ‖² - 1| averaged over batch."""
    return (_normsq(pred) - 1.0).abs().mean()


def energy_error(pred: torch.Tensor, target: torch.Tensor,
                 V: torch.Tensor, dx: float, eps: float = 1e-8) -> torch.Tensor:
    """Relative energy conservation: |<E_pred> - <E_target>| / |<E_target>|."""
    def _energy(psi_channels, Vx):
        psi = torch.complex(psi_channels[:, 0], psi_channels[:, 1])  # (B, N)
        N = psi.shape[-1]
        k = 2 * torch.pi * torch.fft.fftfreq(N, d=dx).to(psi.device).double()
        psi_hat = torch.fft.fft(psi.to(torch.complex128), dim=-1)
        kinetic = 0.5 * (k ** 2) * psi_hat
        kin_psi = torch.fft.ifft(kinetic, dim=-1)
        # <E> = ∫ (½|∂ψ|² + V|ψ|²) dx  = ∫ ψ̄ (-½ ∂² + V) ψ dx
        E = (kin_psi.conj() * psi).real.sum(dim=-1) * dx \
            + (Vx.double() * psi.abs() ** 2).sum(dim=-1) * dx
        return E
    E_p = _energy(pred, V)
    E_t = _energy(target, V)
    return ((E_p - E_t).abs() / (E_t.abs() + eps)).mean()