"""Unit test: split-step solver must match the analytic free Gaussian."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.solver import split_step_solver, free_gaussian_analytic


def _grid(Nx: int, L: float):
    dx = L / Nx
    x = torch.arange(Nx, dtype=torch.float64) * dx - L / 2
    return x, dx


def test_free_gaussian_matches_analytic():
    Nx = 512
    L = 40.0
    t_final = 1.0
    dt = 1e-3
    n_steps = int(t_final / dt)

    x, dx = _grid(Nx, L)

    x0, sigma, k0 = 0.0, 1.0, 0.5
    psi0_np = free_gaussian_analytic(x, 0.0, x0, sigma, k0).numpy()
    psi0 = torch.from_numpy(psi0_np).to(torch.complex128).unsqueeze(0)
    V = torch.zeros(1, Nx, dtype=torch.float64)

    psiT = split_step_solver(psi0, V, dx=dx, dt=dt, n_steps=n_steps).squeeze(0)
    analytic = free_gaussian_analytic(x, t_final, x0, sigma, k0)

    err = torch.linalg.norm(psiT - analytic) / torch.linalg.norm(analytic)
    assert err.item() < 1e-6, f"relative error {err.item():.3e} too large"


def test_norm_conservation():
    Nx = 256
    L = 20.0
    x, dx = _grid(Nx, L)
    psi0 = torch.from_numpy(
        free_gaussian_analytic(x, 0.0, 0.0, 1.0, 0.0).numpy()
    ).to(torch.complex128).unsqueeze(0)
    V = (0.5 * x ** 2).unsqueeze(0)
    psiT = split_step_solver(psi0, V, dx=dx, dt=1e-3, n_steps=200).squeeze(0)
    norm2 = (psiT.abs() ** 2).sum() * dx
    assert abs(norm2.item() - 1.0) < 1e-6, f"norm drift {abs(norm2.item()-1):.3e}"
    