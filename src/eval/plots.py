"""Plotting helpers for evaluation results."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_norm_drift(drifts_dict: dict, out_path: str):
    plt.figure(figsize=(6, 4))
    for label, arr in drifts_dict.items():
        plt.plot(np.arange(1, len(arr) + 1), arr, marker="o", label=label)
    plt.xlabel("rollout step")
    plt.ylabel(r"$|\,\|\psi\|^2 - 1\,|$")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_scaling(xs: list, ys: list, xlabel: str, ylabel: str, out_path: str,
                 logx: bool = True):
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_wavefunction(psi_pred: np.ndarray, psi_true: np.ndarray, x: np.ndarray,
                      out_path: str, title: str = ""):
    """psi_*: (N,) complex arrays."""
    plt.figure(figsize=(8, 3.5))
    plt.plot(x, np.abs(psi_true) ** 2, label=r"$|\psi_\mathrm{true}|^2$", lw=2)
    plt.plot(x, np.abs(psi_pred) ** 2, "--", label=r"$|\psi_\mathrm{pred}|^2$", lw=2)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()