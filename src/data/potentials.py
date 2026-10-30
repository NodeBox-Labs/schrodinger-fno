"""Random potential generators (in-distribution and OOD)."""
import numpy as np


def _grid(L: float, Nx: int) -> np.ndarray:
    return np.linspace(-L / 2, L / 2, Nx, endpoint=False)


def grf_smooth(x: np.ndarray, rng: np.random.Generator,
               n_modes: int = 8, length_scale: float = 3.0,
               amplitude: float = 1.0) -> np.ndarray:
    """Gaussian random field: sum of low-frequency cosine modes."""
    L = x[-1] - x[0] + (x[1] - x[0])
    V = np.zeros_like(x)
    for n in range(1, n_modes + 1):
        k = 2 * np.pi * n / L
        weight = np.exp(-0.5 * (k * length_scale) ** 2)
        phase = rng.uniform(0, 2 * np.pi)
        V += amplitude * weight * np.cos(k * x + phase)
    return V


def gaussian_wells(x: np.ndarray, rng: np.random.Generator,
                   n_wells_min: int = 3, n_wells_max: int = 5,
                   width_min: float = 0.5, width_max: float = 1.5,
                   depth_min: float = -2.0, depth_max: float = 0.0) -> np.ndarray:
    n = rng.integers(n_wells_min, n_wells_max + 1)
    L = x[-1] - x[0]
    V = np.zeros_like(x)
    for _ in range(n):
        xc = rng.uniform(-L / 3, L / 3)
        w = rng.uniform(width_min, width_max)
        d = rng.uniform(depth_min, depth_max)
        V += d * np.exp(-0.5 * ((x - xc) / w) ** 2)
    return V


def harmonic_perturb(x: np.ndarray, rng: np.random.Generator,
                     omega: float = 0.5, perturb_amp: float = 0.2,
                     perturb_modes: int = 4) -> np.ndarray:
    L = x[-1] - x[0]
    V = 0.5 * omega ** 2 * x ** 2
    for n in range(1, perturb_modes + 1):
        k = 2 * np.pi * n / L
        V += perturb_amp * np.cos(k * x + rng.uniform(0, 2 * np.pi)) / n
    return V


def grf_rough(x: np.ndarray, rng: np.random.Generator,
              n_modes: int = 32, length_scale: float = 1.0,
              amplitude: float = 2.0) -> np.ndarray:
    return grf_smooth(x, rng, n_modes=n_modes,
                      length_scale=length_scale, amplitude=amplitude)


def double_well_large(x: np.ndarray, rng: np.random.Generator,
                      barrier: float = 4.0, scale: float = 2.0) -> np.ndarray:
    # V(x) = barrier * ((x/scale)² - 1)² with random anisotropy
    a = rng.uniform(0.8 * scale, 1.2 * scale)
    return barrier * ((x / a) ** 2 - 1.0) ** 2


def periodic_lattice(x: np.ndarray, rng: np.random.Generator,
                     n_sites: int = 8, depth: float = 3.0,
                     width: float = 0.4) -> np.ndarray:
    L = x[-1] - x[0]
    spacing = L / n_sites
    V = np.zeros_like(x)
    for i in range(n_sites):
        xc = -L / 2 + (i + 0.5) * spacing
        V += -depth * np.exp(-0.5 * ((x - xc) / width) ** 2)
    return V


_FAMILY_FUNCS = {
    "grf_smooth": grf_smooth,
    "gaussian_wells": gaussian_wells,
    "harmonic_perturb": harmonic_perturb,
    "grf_rough": grf_rough,
    "double_well_large": double_well_large,
    "periodic_lattice": periodic_lattice,
}


def make_potential(family: str, x: np.ndarray, rng: np.random.Generator,
                   params: dict) -> np.ndarray:
    if family not in _FAMILY_FUNCS:
        raise ValueError(f"unknown potential family: {family}")
    return _FAMILY_FUNCS[family](x, rng, **params)