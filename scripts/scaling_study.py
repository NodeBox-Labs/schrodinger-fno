"""Error vs training set size and vs number of Fourier modes."""
import os
import argparse
import yaml
import torch
import numpy as np

from src.train.loop import train_model
from src.data.dataset import SchrodingerDataset
from src.models.fno import FNO1d
from src.eval.metrics import rel_l2
from src.eval.plots import plot_scaling


def _subset_pt(src_path, dst_path, n):
    d = torch.load(src_path, map_location="cpu")
    for k in d:
        d[k] = d[k][:n]
    torch.save(d, dst_path)


def main(config_path: str, data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    # Split sizes
    sizes = [100, 500, 1000, 2000]
    errs = []
    for n in sizes:
        tmp_dir = os.path.join(out_dir, f"sub_{n}")
        os.makedirs(tmp_dir, exist_ok=True)
        _subset_pt(os.path.join(data_dir, "train.pt"),
                   os.path.join(tmp_dir, "train.pt"), n)
        # copy val
        import shutil
        shutil.copy(os.path.join(data_dir, "val.pt"),
                    os.path.join(tmp_dir, "val.pt"))
        cfg = dict(base_cfg)
        cfg["out_dir"] = os.path.join(tmp_dir, "results")
        ckpt = train_model(cfg, data_dir=tmp_dir, mode="single", seed=42)
        # evaluate on test
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        m = FNO1d(**{k: ck["cfg"]["model"][k]
                     for k in ("modes", "width", "n_layers")}).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        ds = SchrodingerDataset(os.path.join(data_dir, "test_id.pt"))
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        acc = 0.0
        with torch.no_grad():
            for inp, tgt in dl:
                inp, tgt = inp.to(dev), tgt.to(dev)
                acc += rel_l2(m(inp), tgt).item()
        errs.append(acc / len(dl))
        print(f"N_train={n}  relL2={errs[-1]:.4e}")

    plot_scaling(sizes, errs, "N_train", "relL2", 
                 os.path.join(out_dir, "scaling_ntrain.png"))

    # Modes sweep
    modes_list = [4, 8, 16, 32]
    errs_m = []
    for m_modes in modes_list:
        cfg = dict(base_cfg)
        cfg["model"] = dict(base_cfg["model"])
        cfg["model"]["modes"] = m_modes
        cfg["out_dir"] = os.path.join(out_dir, f"modes_{m_modes}")
        ckpt = train_model(cfg, data_dir=data_dir, mode="single", seed=42)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        m = FNO1d(**{k: ck["cfg"]["model"][k]
                     for k in ("modes", "width", "n_layers")}).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        ds = SchrodingerDataset(os.path.join(data_dir, "test_id.pt"))
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        acc = 0.0
        with torch.no_grad():
            for inp, tgt in dl:
                inp, tgt = inp.to(dev), tgt.to(dev)
                acc += rel_l2(m(inp), tgt).item()
        errs_m.append(acc / len(dl))
        print(f"modes={m_modes}  relL2={errs_m[-1]:.4e}")

    plot_scaling(modes_list, errs_m, "modes", "relL2",
                 os.path.join(out_dir, "scaling_modes.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    main(args.config, args.data, args.out)