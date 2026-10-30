"""Training loop for FNO on Schrödinger data (single-shot or autoregressive)."""
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import SchrodingerDataset
from ..models.fno import FNO1d
from .losses import relative_l2, norm_penalty


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)


def _resolve_device(cfg):
    d = cfg.get("device", "cuda")
    if d == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return d


def train_model(cfg: dict, data_dir: str, mode: str = "single", seed: int = 42):
    """Train a single-shot or autoregressive FNO.

    Autoregressive: dt_train = t_final / T is computed from the data, but
    since our dataset only stores ψ_final, we approximate AR by chunking
    T=ar_chunk times: we treat ψ(t_final · k/T) as unknown and supervise
    only the final output. AR variant here means: predict ψ_final as the
    sum of ar_chunk consecutive (Δt) predictions, each conditioned on the
    current state. This trains a step-wise model.
    """
    _set_seed(seed)
    device = _resolve_device(cfg)
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    train_ds = SchrodingerDataset(os.path.join(data_dir, "train.pt"))
    val_ds = SchrodingerDataset(os.path.join(data_dir, "val.pt"))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"],
                            shuffle=False, num_workers=0)

    model = FNO1d(modes=mcfg["modes"], width=mcfg["width"],
                  n_layers=mcfg["n_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                           weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tcfg["epochs"], eta_min=1e-5)

    # For autoregressive: recompute
    T = tcfg.get("ar_chunk", 4)
    def _rollout(inp, pred_mode):
        if pred_mode == "single":
            return model(inp)
        # autoregressive: T sequential calls, feeding prediction back
        # For our data we do not have intermediate labels; we treat this as
        # an unrolled prediction from ψ0 with T applications of the network.
        x = inp.clone()
        for _ in range(T):
            y = model(x)
            x = torch.cat([inp[:, :1], y], dim=1)  # keep V, update ψ
        return y

    best_val = float("inf")
    ckpt_path = os.path.join(cfg["out_dir"], f"fno_{mode}_seed{seed}.pt")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    lambda_norm = tcfg.get("norm_penalty", 0.0)

    for epoch in range(tcfg["epochs"]):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = _rollout(inp, mode)
            l = relative_l2(pred, tgt)
            if lambda_norm > 0:
                l = l + lambda_norm * norm_penalty(pred)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            train_loss += l.item()
            n_batches += 1
        sched.step()

        if (epoch + 1) % tcfg["eval_every"] == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    pred = _rollout(inp, mode)
                    val_loss += relative_l2(pred, tgt).item()
            val_loss /= len(val_loader)
            print(f"[{mode}] epoch {epoch+1:4d}  train {train_loss/n_batches:.4e}"
                  f"  val {val_loss:.4e}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    print(f"Saved {ckpt_path} (best val relL2 = {best_val:.4e})")
    return ckpt_path