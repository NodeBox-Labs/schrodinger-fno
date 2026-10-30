"""Random Gaussian wave packets."""
import numpy as np


def make_gaussian_packet(x: np.ndarray, rng: np.random.Generator,
                         sigma_min: float = 0.5, sigma_max: float = 2.0,
                         k0_max: float = 2.0, x0_margin: float = 4.0) -> np.ndarray:
    """Return normalized ψ0(x) = (2πσ²)^{-1/4} exp[-(x-x0)²/(4σ²) + i k0 x]."""
    L = x[-1] - x[0]
    sigma = rng.uniform(sigma_min, sigma_max)
    x0 = rng.uniform(-L / 2 + x0_margin, L / 2 - x0_margin)
    k0 = rng.uniform(-k0_max, k0_max)
    norm = (2 * np.pi * sigma ** 2) ** (-0.25)
    psi0 = norm * np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
    # discrete renormalization to enforce unit L2 on grid
    dx = x[1] - x[0]
    nrm = np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)
    return (psi0 / nrm).astype(np.complex128)