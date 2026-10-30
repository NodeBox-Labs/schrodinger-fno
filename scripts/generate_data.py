"""Generate train/val/test datasets with split-step solver."""
import os
import argparse
import yaml
import numpy as np
import torch

from src.data.potentials import make_potential
from src.data.initial_states import make_gaussian_packet
from src.data.solver import split_step_solver


def _make_split(cfg, families, n_samples, seed):
    rng = np.random.default_rng(seed)
    Nx = cfg["Nx"]
    L = cfg["L"]
    x = np.linspace(-L / 2, L / 2, Nx, endpoint=False)
    dx = x[1] - x[0]

    V_list, psi0_list, psiF_list = [], [], []

    for i in range(n_samples):
        fam = families[i % len(families)]
        params = cfg.get(fam, {})
        V_np = make_potential(fam, x, rng, params)
        psi0_np = make_gaussian_packet(x, rng, **cfg["psi0"])

        V_t = torch.from_numpy(V_np).double().unsqueeze(0)
        psi0_t = torch.from_numpy(psi0_np).double().unsqueeze(0)
        n_steps = int(round(cfg["t_final"] / cfg["dt"]))
        psiF_t = split_step_solver(psi0_t, V_t, dx=dx, dt=cfg["dt"],
                                   n_steps=n_steps)

        V_list.append(torch.from_numpy(V_np).float())
        psi0_list.append(torch.from_numpy(psi0_np))
        psiF_list.append(psiF_t.squeeze(0))

    return {
        "V": torch.stack(V_list),
        "psi0": torch.stack(psi0_list),
        "psi_final": torch.stack(psiF_list),
    }


def main(config_path: str, out_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg["seed"]

    # In-distribution    id_fams = cfg["id_families"]
    train = _make_split(cfg, id_fams, cfg["n_train"], seed)
    val = _make_split(cfg, id_fams, cfg["n_val"], seed + 1)
    test_id = _make_split(cfg, id_fams, cfg["n_test_id"], seed + 2)

    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    torch.save(test_id, os.path.join(out_dir, "test_id.pt"))

    # OOD
    ood_fams = cfg["ood_families"]
    test_ood = _make_split(cfg, ood_fams, cfg["n_test_ood"], seed + 3)
    torch.save(test_ood, os.path.join(out_dir, "test_ood.pt"))

    print(f"Saved {len(train['V'])} train, {len(val['V'])} val, "
          f"{len(test_id['V'])} test-ID, {len(test_ood['V'])} test-OOD.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.config, args.out)