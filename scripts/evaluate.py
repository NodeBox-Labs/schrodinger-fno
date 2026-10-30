"""Evaluate a trained FNO: metrics table and plots."""
import os
import argparse
import time
import csv
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SchrodingerDataset
from src.models.fno import FNO1d
from src.eval.metrics import rel_l2, norm_drift, energy_error
from src.eval.rollout import rollout_norm_drift
from src.eval.plots import plot_norm_drift, plot_wavefunction


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    m = FNO1d(modes=cfg["model"]["modes"], width=cfg["model"]["width"],
              n_layers=cfg["model"]["n_layers"]).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, cfg


@torch.no_grad()
def evaluate_split(model, cfg, data_path, device, renorm=False):
    ds = SchrodingerDataset(data_path)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    all_rel, all_nd, all_ee = [], [], []
    dx = cfg["data"]["L"] / cfg["data"]["Nx"] if "L" in cfg["data"] else 20.0 / 256

    for inp, tgt in dl:
        inp, tgt = inp.to(device), tgt.to(device)
        pred = model(inp)
        if renorm:
            n = pred.pow(2).sum(dim=1).sum(dim=-1).sqrt().clamp_min(1e-8)
            pred = pred / n[:, None, None]
        all_rel.append(rel_l2(pred, tgt).item())
        all_nd.append(norm_drift(pred).item())
        V = inp[:, 0]
        all_ee.append(energy_error(pred, tgt, V, dx).item())
    return (float(np.mean(all_rel)), float(np.mean(all_nd)),
            float(np.mean(all_ee)))


def _time_forward(model, inp, device, n_warmup=5, n_iters=20):
    inp = inp.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inp)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(inp)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iters
    return dt  # seconds per batch


def _time_solver(split_cfg, n_samples=16, device="cpu"):
    from src.data.solver import split_step_solver
    from src.data.initial_states import make_gaussian_packet
    Nx = split_cfg["Nx"]; L = split_cfg["L"]
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    rng = np.random.default_rng(0)
    psi = torch.stack([
        torch.from_numpy(make_gaussian_packet(x, rng, **split_cfg["psi0"])).double()
        for _ in range(n_samples)
    ])
    V = torch.zeros(n_samples, Nx, dtype=torch.float64)
    dx = x[1] - x[0]
    n_steps = int(round(split_cfg["t_final"] / split_cfg["dt"]))
    t0 = time.perf_counter()
    _ = split_step_solver(psi, V, dx, split_cfg["dt"], n_steps)
    return (time.perf_counter() - t0) / n_samples  # per-sample


def main(ckpt, data_dir, split, out_dir, renorm):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(ckpt, device)

    row = {}
    for name, fname in [("ID", "test_id.pt"), ("OOD", "test_ood.pt")]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        r, nd, ee = evaluate_split(model, cfg, path, device, renorm=renorm)
        row[name] = (r, nd, ee)

    # timing
    ds = SchrodingerDataset(os.path.join(data_dir, "test_id.pt"))
    inp_batch = torch.stack([ds[i][0] for i in range(64)])
    t_fno = _time_forward(model, inp_batch, device) / 64.0  # per sample

    # solver timing
    t_solver = _time_solver(cfg["data"], n_samples=16)

    # norm drift over rollout
    T = 10
    drifts = rollout_norm_drift(model, inp_batch[:32], T=T, device=device)
    plot_norm_drift({"raw": drifts},
                    os.path.join(out_dir, f"norm_drift_{split}.png"))

    # sample wavefunction plot
    with torch.no_grad():
        inp, tgt = ds[0]
        pred = model(inp.unsqueeze(0).to(device)).cpu().squeeze(0)
    x = np.linspace(-cfg["data"]["Nx"] / 2, cfg["data"]["Nx"] / 2,
                    cfg["data"]["Nx"])  # placeholder
    x = np.linspace(-10, 10, cfg["data"]["Nx"])
    psi_p = (pred[0] + 1j * pred[1]).numpy()
    psi_t = (tgt[0] + 1j * tgt[1]).numpy()
    plot_wavefunction(psi_p, psi_t, x,
                      os.path.join(out_dir, f"sample_{split}.png"))

    # write CSV
    csv_path = os.path.join(out_dir, f"metrics_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "relL2", "norm_drift", "energy_err",
                    "t_fno_per_sample_s", "t_solver_per_sample_s"])
        for name, (r, nd, ee) in row.items():
            w.writerow([name, f"{r:.4e}", f"{nd:.4e}", f"{ee:.4e}",
                        f"{t_fno:.4e}", f"{t_solver:.4e}"])
    print(f"Wrote {csv_path}")
    for name, (r, nd, ee) in row.items():
        print(f"{name}:  relL2={r:.4e}  norm_drift={nd:.4e}  energy_err={ee:.4e}")
    print(f"FNO per-sample: {t_fno*1e3:.2f} ms | Solver per-sample: {t_solver*1e3:.2f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--out", default="results")
    ap.add_argument("--renorm", action="store_true")
    args = ap.parse_args()
    main(args.ckpt, args.data, args.split, args.out, args.renorm)