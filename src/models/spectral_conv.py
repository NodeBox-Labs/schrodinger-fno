"""1D spectral convolution with complex weights (native PyTorch complex)."""
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """Applies a truncated Fourier multiplier: out = ifft(W · fft(x)) for low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N) complex
        B, C, N = x.shape
        x_ft = torch.fft.fft(x, dim=-1)  # (B, C, N)
        # only low modes are kept; cast weight to input dtype/device
        w = self.weight.to(x.dtype)  # (Cin, Cout, m)
        m = self.modes
        out_ft = torch.zeros(B, self.out_channels, N, dtype=x.dtype, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], w)
        # negative freqs for Hermitian symmetry (uses modes-1 entries)
        out_ft[:, :, -m + 1:] = torch.einsum(
            "bix,iox->box", x_ft[:, :, -m + 1:], w[:, :, 1:]
        )
        return torch.fft.ifft(out_ft, dim=-1)