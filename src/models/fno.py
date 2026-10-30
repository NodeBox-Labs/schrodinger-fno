
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv import SpectralConv1d


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int, activation=nn.GELU()):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, 1)  # complex -> use two real convs
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x complex (B, W, N)
        y = self.spec(x)
        # pointwise: apply conv to real and imag separately
        yr = self.pointwise(y.real)
        yi = self.pointwise(y.imag)
        y = torch.complex(yr, yi)
        return self.act(y.real) + 1j * self.act(y.imag) if False else torch.complex(
            F.gelu(y.real), F.gelu(y.imag)
        )


class FNO1d(nn.Module):
    """Simple FNO with complex internal representation.

    Params:
        modes: number of Fourier modes retained
        width: number of complex channels
        n_layers: number of FNO blocks
    """

    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        self.modes = modes
        self.width = width

        # Lifting: map (V, Re ψ0, Im ψ0) -> complex feature of width `width`.
        # We use two real conv1d's to produce a complex feature.
        self.lift_real = nn.Conv1d(3, width, 1)
        self.lift_imag = nn.Conv1d(3, width, 1)

        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes) for _ in range(n_layers)]
        )

        # Projection: complex (B, width, N) -> (B, 2, N) via linear combinations.
        self.proj_real = nn.Conv1d(width, 1, 1)
        self.proj_imag = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N) real
        xr = self.lift_real(x)
        xi = self.lift_imag(x)
        h = torch.complex(xr, xi)  # (B, width, N)

        for block in self.blocks:
            h = block(h) + h  # residual

        yr = self.proj_real(h.real)  # (B, 1, N)
        yi = self.proj_imag(h.imag)  # (B, 1, N)
        return torch.cat([yr, yi], dim=1)  # (B, 2, N)