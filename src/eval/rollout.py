"""Autoregressive rollout for norm-drift analysis."""
import torch


@torch.no_grad()
def rollout_norm_drift(model, inp: torch.Tensor, T: int, device="cpu"):
    """Return array of norm drift over T autoregressive steps."""
    model.eval().to(device)
    x = inp.to(device).clone()
    drifts = []
    for _ in range(T):
        y = model(x)
        norm2 = y.pow(2).sum(dim=1).sum(dim=-1)
        drifts.append((norm2 - 1.0).abs().mean().item())
        x = torch.cat([inp[:, :1].to(device), y], dim=1)
    return drifts