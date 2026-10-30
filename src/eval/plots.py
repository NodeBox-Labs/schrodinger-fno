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

def plot_worst_ood_failures(model, dataset, x, device, out_path: str, n=3):
    """Finds and plots the top N worst OOD failures."""
    errs = []
    preds = []
    trues = []
    potentials = []
    
    # Get all errors first
    import torch
    from .metrics import rel_l2
    with torch.no_grad():
        for i in range(len(dataset)):
            inp, tgt = dataset[i]
            pred = model(inp.unsqueeze(0).to(device)).cpu().squeeze(0)
            err = rel_l2(pred.unsqueeze(0), tgt.unsqueeze(0)).item()
            errs.append(err)
            
            if len(preds) < 1000: # Store all up to 1000 to avoid memory issues
                preds.append(pred.cpu().numpy())
                trues.append(tgt.cpu().numpy())
                potentials.append(inp[0].cpu().numpy())

    worst_idx = np.argsort(errs)[-n:][::-1]
    
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1: axes = [axes]
    
    for ax, idx in zip(axes, worst_idx):
        psi_pred = preds[idx][0] + 1j * preds[idx][1]
        psi_true = trues[idx][0] + 1j * trues[idx][1]
        
        ax.plot(x, np.abs(psi_true)**2, 'k-', lw=2, label=r'True \(|\psi|^2\)')
        ax.plot(x, np.abs(psi_pred)**2, 'r--', lw=2, label=r'Pred \(|\psi|^2\)')
        
        ax2 = ax.twinx()
        ax2.plot(x, potentials[idx], 'g:', alpha=0.5, label='V(x)')
        ax2.set_ylabel('Potential V(x)', color='g')
        
        ax.set_title(f"Worst OOD Failure #{idx} (Rel L2 = {errs[idx]:.4e})")
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
        ax2.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.xlabel('x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()