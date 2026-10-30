"""
Split-step Fourier solver for the 1D Schrödinger equation (ħ = m = 1):

    i ∂ψ/∂t = [-½ ∂²/∂x² + V(x)] ψ

Strang splitting: half kinetic (Fourier) → full potential (real space) → half kinetic.
Complex128 for ground truth accuracy.
"""
import torch


def _kinetic_half(dt: float, k: torch.Tensor) -> torch.Tensor:
    """Half-step kinetic propagator exp(-i * k²/2 * dt/2).

    The Schrödinger equation dψ/dt = i/2 d²ψ/dx² - iVψ gives, in k-space,
    dψ̂/dt = -i (k²/2) ψ̂, hence propagator exp(-i k²/2 Δt).
    """
    return torch.exp(-0.5j * (k ** 2 / 2.0) * dt)


def _potential_full(dt: float, V: torch.Tensor) -> torch.Tensor:
    """Full potential propagator exp(-i V dt)."""
    return torch.exp(-1j * V * dt)


def split_step_solver(psi0: torch.Tensor, V: torch.Tensor, dx: float,
                      dt: float, n_steps: int) -> torch.Tensor:
    """Propagate psi0 under V for n_steps * dt using Strang splitting.

    Args:
        psi0: (B, N) complex128 initial wavefunction on uniform grid dx.
        V:    (B, N) real64 potential evaluated on the same grid.
        dx:   grid spacing (float).
        dt:   time step (float).
        n_steps: number of Strang steps.

    Returns:
        (B, N) complex128 final wavefunction.
    """
    assert psi0.dtype == torch.complex128, f"expected complex128, got {psi0.dtype}"
    assert V.dtype == torch.float64, f"expected float64, got {V.dtype}"

    N = psi0.shape[-1]
    k = 2.0 * torch.pi * torch.fft.fftfreq(N, d=dx).to(psi0.device).double()
    kh = _kinetic_half(dt, k)          # (N,)
    pf = _potential_full(dt, V)        # (B, N)

    psi = psi0.clone()
    for _ in range(n_steps):
        psi = torch.fft.ifft(torch.fft.fft(psi, dim=-1) * kh, dim=-1)
        psi = psi * pf
        psi = torch.fft.ifft(torch.fft.fft(psi, dim=-1) * kh, dim=-1)
    return psi


def free_gaussian_analytic(x: torch.Tensor, t: float,
                           x0: float, sigma: float, k0: float) -> torch.Tensor:
    """Analytic free Gaussian wave packet (V=0).

    ψ(x,t) = (2πσ²)^{-1/4} (1 + i t/σ²)^{-1/2}
             exp[-(x-x0)²/(4σ²(1+it/σ²)) + i k0 x - i k0² t/2]

    with normalization ∫|ψ|² dx = 1.
    """
    x = x.double()
    norm = (2 * torch.pi * sigma ** 2) ** (-0.25)
    denom = 1.0 + 1j * t / (sigma ** 2)
    gauss = torch.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2 * denom))
    phase = torch.exp(1j * k0 * x - 0.5j * k0 ** 2 * t)
    return (norm / torch.sqrt(denom)) * gauss * phase