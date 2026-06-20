"""BiMamba v2: Bidirectional Mamba block compatible with Vim/SSAMBA checkpoints.

Implements the BiMamba v2 architecture from Vision Mamba (Vim) which processes
sequences bidirectionally within a single Mamba block by maintaining separate
SSM parameters for forward and backward directions.

This is a standalone re-implementation that uses ``selective_scan_fn`` from the
standard ``mamba-ssm`` package.  It does NOT require the Vim fork's custom CUDA
kernels (``bimamba_inner_fn``, ``mamba_inner_fn_no_out_proj``).

Reference:
    - Vim: https://github.com/hustvl/Vim
    - SSAMBA: https://github.com/SiavashShams/ssamba
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn

    _CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    causal_conv1d_fn = None
    _CAUSAL_CONV1D_AVAILABLE = False

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _SELECTIVE_SCAN_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    _SELECTIVE_SCAN_AVAILABLE = False


class BiMambaV2(nn.Module):
    """Bidirectional Mamba v2 block with exact Vim/SSAMBA parameter layout.

    Each block maintains separate SSM parameters for forward and backward
    scans (``A_b_log``, ``D_b``, ``conv1d_b``, ``x_proj_b``, ``dt_proj_b``).
    In the forward pass, the input is processed in both directions and the
    results are combined (averaged when ``if_divide_out=True``).

    The parameter names and shapes match the Vim repository exactly, enabling
    direct checkpoint loading from SSAMBA pretrained weights.

    Args:
        d_model: Model / embedding dimension.
        d_state: SSM state expansion factor.
        d_conv: Causal convolution kernel size.
        expand: Inner dimension expansion factor (``d_inner = expand * d_model``).
        dt_rank: Rank for delta projection (``"auto"`` = ``ceil(d_model/16)``).
        if_divide_out: Average forward and backward outputs (True in SSAMBA).
        layer_idx: Layer index for inference cache.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        if_divide_out: bool = True,
        layer_idx: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if not _SELECTIVE_SCAN_AVAILABLE:
            msg = (
                "BiMambaV2 requires 'mamba-ssm'. Install with: pip install mamba-ssm causal-conv1d"
            )
            raise ImportError(msg)

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_divide_out = if_divide_out
        self.layer_idx = layer_idx

        # Input projection: d_model -> 2 * d_inner  (x and gate z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # --- Forward direction ---
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # --- Backward direction ---
        self.conv1d_b = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # SSM parameters (forward)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # SSM parameters (backward)
        A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A_b = A_b.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.A_b_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True  # type: ignore[attr-defined]

        self.activation = "silu"
        self.act = nn.SiLU()

        # Initialize dt projections
        self._init_dt_proj(
            self.dt_proj, dt_init, dt_scale, dt_min, dt_max, dt_init_floor, factory_kwargs
        )
        self._init_dt_proj(
            self.dt_proj_b, dt_init, dt_scale, dt_min, dt_max, dt_init_floor, factory_kwargs
        )

    @staticmethod
    def _init_dt_proj(
        dt_proj: nn.Linear,
        dt_init: str,
        dt_scale: float,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        factory_kwargs: dict,
    ) -> None:
        dt_init_std = dt_proj.in_features**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(dt_proj.out_features, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

    def _scan(
        self,
        xz: torch.Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_log: nn.Parameter,
        D: nn.Parameter,
        seqlen: int,
    ) -> torch.Tensor:
        """Run SSM scan in one direction.

        Args:
            xz: ``(B, 2*d_inner, L)`` — concatenated input and gate.

        Returns:
            ``(B, d_inner, L)`` — gated SSM output.
        """
        x, z = xz.chunk(2, dim=1)

        # Causal convolution
        if _CAUSAL_CONV1D_AVAILABLE:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                bias=conv1d.bias,
                activation=self.activation,
            )
        else:
            x = self.act(conv1d(x)[..., :seqlen])

        # Project to dt, B, C
        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        A = -torch.exp(A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )
        return y

    def forward(self, hidden_states: torch.Tensor, inference_params: object = None) -> torch.Tensor:
        """Bidirectional SSM forward.

        Args:
            hidden_states: ``(B, L, D)``.

        Returns:
            ``(B, L, D)``.
        """
        batch, seqlen, dim = hidden_states.shape

        # Input projection -> (B, 2*d_inner, L)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Forward scan
        y_f = self._scan(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D, seqlen)

        # Backward scan (flip the entire sequence)
        y_b = self._scan(
            xz.flip([-1]),
            self.conv1d_b,
            self.x_proj_b,
            self.dt_proj_b,
            self.A_b_log,
            self.D_b,
            seqlen,
        )

        # Combine forward + flipped backward
        combined = y_f + y_b.flip([-1])
        if self.if_divide_out:
            combined = combined / 2

        return self.out_proj(rearrange(combined, "b d l -> b l d"))
