"""PyTorch Dataset wrappers for (V, ψ0) → ψ_final samples."""
import torch
from torch.utils.data import Dataset


class SchrodingerDataset(Dataset):
    """Loads a saved .pt dict with keys V, psi0, psi_final (all torch tensors)."""

    def __init__(self, path: str, dtype: torch.dtype = torch.float32):
        data = torch.load(path, map_location="cpu")
        self.V = data["V"].to(dtype)
        self.psi0 = data["psi0"]
        self.psi_final = data["psi_final"]
        # stack psi as 2 real channels: (N, 2, Nx)
        self.psi0_c = torch.stack([self.psi0.real, self.psi0.imag], dim=1).to(dtype)
        self.psi_f_c = torch.stack([self.psi_final.real, self.psi_final.imag], dim=1).to(dtype)

    def __len__(self):
        return self.V.shape[0]

    def __getitem__(self, i):
        # model input: (3, Nx) = [V, Re ψ0, Im ψ0]
        V = self.V[i].unsqueeze(0)
        inp = torch.cat([V, self.psi0_c[i]], dim=0)
        return inp, self.psi_f_c[i]