"""Audio Mamba (AuM): Bidirectional State Space Model for audio classification.

Implements the AuM architecture from Erol et al. (2024), combining AST-style
2D spectrogram patch embedding with Vision Mamba (ViM) bidirectional SSM
blocks.

The model processes log-mel spectrograms through:
  1. 2D patch embedding (Conv2d) with configurable strides
  2. Learnable positional embeddings
  3. Optional CLS token for classification
  4. Stacked Mamba blocks with prenorm residual connections
  5. Bidirectional processing via paired forward/backward layers

Two bidirectional modes are supported:

- **External bidirectional** (``bidirectional=True, use_bimamba=False``):
  layers are grouped in forward/backward pairs (original AuM behaviour).
- **Internal bidirectional** (``use_bimamba=True``): each block uses a
  BiMamba v2 mixer that scans forward *and* backward within a single
  layer.  This is the architecture used by Vim/SSAMBA and enables loading
  AudioSet-pretrained SSAMBA checkpoints.

Reference:
    - AuM: https://github.com/kaistmm/Audio-Mamba-AuM
    - Vim: https://github.com/hustvl/Vim
    - SSAMBA: https://github.com/SiavashShams/ssamba

Requires the ``mamba-ssm`` and ``causal-conv1d`` packages (CUDA only,
install via the ``[cuda]`` extras).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba

    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_patch_shape(
    fstride: int,
    tstride: int,
    input_fdim: int,
    input_tdim: int,
    patch_size: tuple[int, int] = (16, 16),
) -> tuple[int, int]:
    """Return ``(f_dim, t_dim)`` — the patch grid for the given input."""
    f_dim = (input_fdim - patch_size[0]) // fstride + 1
    t_dim = (input_tdim - patch_size[1]) // tstride + 1
    return f_dim, t_dim


class _DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------


class PatchEmbed2D(nn.Module):
    """2D patch embedding for spectrograms (as in AST / AuM).

    Args:
        patch_size: ``(freq, time)`` patch dimensions.
        fstride: Stride along frequency axis.
        tstride: Stride along time axis.
        in_channels: Number of input channels (1 for mono spectrogram).
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        fstride: int = 16,
        tstride: int = 16,
        in_channels: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=(fstride, tstride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and flatten 2D patches.

        Args:
            x: ``(B, 1, F, T)``.

        Returns:
            ``(B, num_patches, embed_dim)``.
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class MambaBlock(nn.Module):
    """Single Mamba block with prenorm residual connection.

    Follows the ViM / AuM pattern: ``Add → LN → Mamba``, passing both
    ``hidden_states`` and ``residual`` between blocks.

    Args:
        dim: Model dimension.
        d_state: SSM state expansion factor.
        drop_path: Stochastic depth rate.
        use_bimamba: Use BiMamba v2 mixer instead of vanilla Mamba.
        if_divide_out: Average forward/backward outputs in BiMamba v2.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        drop_path: float = 0.0,
        use_bimamba: bool = False,
        if_divide_out: bool = True,
    ) -> None:
        super().__init__()
        if use_bimamba:
            from bioaed.models.bimamba_v2 import BiMambaV2

            self.mixer = BiMambaV2(
                d_model=dim,
                d_state=d_state,
                if_divide_out=if_divide_out,
            )
        else:
            if not _MAMBA_AVAILABLE:
                msg = (
                    "AudioMamba requires 'mamba-ssm'. "
                    "Install with: pip install mamba-ssm causal-conv1d, "
                    "or use the [cuda] extras."
                )
                raise ImportError(msg)
            self.mixer = Mamba(d_model=dim, d_state=d_state)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prenorm residual forward.

        Returns:
            ``(hidden_states, residual)`` to be fed into the next block.
        """
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------


class AudioMamba(nn.Module):
    """Audio Mamba (AuM) for multi-label audio classification.

    Processes log-mel spectrograms through 2D patch embedding, positional
    embeddings, stacked Mamba blocks, and a classification head.

    Two bidirectional strategies are supported:

    - **External** (``bidirectional=True, use_bimamba=False``): layers are
      grouped in forward/backward pairs (requires ``depth`` to be even).
    - **Internal** (``use_bimamba=True``): each block uses a BiMamba v2
      mixer that processes both directions (compatible with SSAMBA
      pretrained checkpoints).

    Args:
        num_classes: Number of output classes.
        input_fdim: Number of frequency bins (mel bands).
        input_tdim: Number of time frames.
        patch_size: 2D patch size ``(freq, time)`` or single int.
        fstride: Patch stride on frequency axis.
        tstride: Patch stride on time axis.
        embed_dim: Embedding / model dimension.
        depth: Number of Mamba layers (must be even when
            ``bidirectional=True`` and ``use_bimamba=False``).
        d_state: SSM state expansion factor.
        bidirectional: Use external paired-layer bidirectional processing
            (ignored when ``use_bimamba=True``).
        use_cls_token: Use a learnable CLS token for classification;
            if ``False``, global average pooling is used instead.
        use_bimamba: Use BiMamba v2 mixer (Vim/SSAMBA architecture).
        use_middle_cls_token: Insert CLS at sequence midpoint instead of
            prepending (Vim/SSAMBA convention).  Only used when
            ``use_bimamba=True``.
        if_divide_out: Average forward/backward scan outputs in
            BiMamba v2 blocks.
        pretrained_path: Path to SSAMBA checkpoint (``.pth``) for
            pretrained weight loading.
        pool_type: Classification pooling strategy:
            ``"cls"`` — use CLS token output (default),
            ``"mean"`` — global average pooling over all tokens,
            ``"mean_no_cls"`` — mean of all tokens except CLS
            (SSAMBA default for fine-tuning).
        drop_rate: Dropout rate after positional embedding.
        drop_path_rate: Maximum stochastic-depth rate (linearly ramped).
    """

    def __init__(
        self,
        num_classes: int = 7,
        input_fdim: int = 64,
        input_tdim: int = 101,
        patch_size: tuple[int, int] | int = 16,
        fstride: int = 16,
        tstride: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        d_state: int = 16,
        bidirectional: bool = True,
        use_cls_token: bool = True,
        use_bimamba: bool = False,
        use_middle_cls_token: bool = False,
        if_divide_out: bool = True,
        pretrained_path: str | None = None,
        pool_type: str = "cls",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # use_bimamba implies internal bidirectional — disable external
        if use_bimamba:
            bidirectional = False

        if bidirectional and depth % 2 != 0:
            raise ValueError(f"Bidirectional mode requires even depth, got {depth}.")

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.bidirectional = bidirectional
        self.use_cls_token = use_cls_token
        self.use_bimamba = use_bimamba
        self.use_middle_cls_token = use_middle_cls_token and use_bimamba
        self.pool_type = pool_type
        self.embed_dim = embed_dim

        # 2D Patch Embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            fstride=fstride,
            tstride=tstride,
            in_channels=1,
            embed_dim=embed_dim,
        )

        # Patch grid
        f_dim, t_dim = _get_patch_shape(fstride, tstride, input_fdim, input_tdim, patch_size)
        self.num_patches = f_dim * t_dim
        num_tokens = 1 if use_cls_token else 0

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Mamba blocks with linearly ramped stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    drop_path=dpr[i],
                    use_bimamba=use_bimamba,
                    if_divide_out=if_divide_out,
                )
                for i in range(depth)
            ]
        )

        # Output norm + classification head
        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight init
        self.apply(self._init_weights)

        # Load pretrained weights (after init so pos_embed dimensions are set)
        if pretrained_path:
            self._load_pretrained(
                pretrained_path,
                input_fdim,
                input_tdim,
                fstride,
                tstride,
                patch_size,
            )

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def _interpolate_pos_embed(
        self,
        pretrained_pe: torch.Tensor,
        orig_f: int,
        orig_t: int,
        target_f: int,
        target_t: int,
    ) -> torch.Tensor:
        """Cut-or-interpolate pretrained positional embeddings to a new grid.

        Follows the SSAMBA convention: if the target dimension is smaller
        than the pretrained dimension, center-crop; otherwise bilinear
        interpolate.  CLS token pos-embed is always at position 0 in
        SSAMBA checkpoints.
        """
        # In SSAMBA checkpoints CLS is always at position 0
        if self.use_cls_token:
            cls_pe = pretrained_pe[:, :1, :]
            patch_pe = pretrained_pe[:, 1:, :]
        else:
            cls_pe = None
            patch_pe = pretrained_pe

        D = patch_pe.shape[2]
        # (1, orig_f, orig_t, D) → (1, D, orig_f, orig_t)
        patch_pe = patch_pe.float().reshape(1, orig_f, orig_t, D).permute(0, 3, 1, 2)

        # Cut-or-interpolate each axis independently (SSAMBA convention)
        if target_t < orig_t:
            # Center-crop time
            start = (orig_t - target_t) // 2
            patch_pe = patch_pe[:, :, :, start : start + target_t]
        elif target_t > orig_t:
            patch_pe = F.interpolate(
                patch_pe,
                size=(orig_f, target_t),
                mode="bilinear",
                align_corners=False,
            )

        if target_f < orig_f:
            # Center-crop frequency
            start = (orig_f - target_f) // 2
            patch_pe = patch_pe[:, :, start : start + target_f, :]
        elif target_f > orig_f:
            patch_pe = F.interpolate(
                patch_pe,
                size=(target_f, patch_pe.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, target_f * target_t, D)

        if cls_pe is not None:
            if self.use_middle_cls_token:
                new_mid = (target_f * target_t) // 2
                return torch.cat(
                    [patch_pe[:, :new_mid, :], cls_pe, patch_pe[:, new_mid:, :]],
                    dim=1,
                )
            return torch.cat([cls_pe, patch_pe], dim=1)
        return patch_pe

    def _load_pretrained(
        self,
        path: str,
        input_fdim: int,
        input_tdim: int,
        fstride: int,
        tstride: int,
        patch_size: tuple[int, int],
    ) -> None:
        """Load SSAMBA / Vim pretrained weights with pos-embed interpolation."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        # Strip DataParallel wrapper prefix (module.v.* for backbone,
        # module.* for pretraining heads)
        raw: dict[str, torch.Tensor] = {}
        for k, v in ckpt.items():
            if k.startswith("module.v."):
                raw[k[len("module.v.") :]] = v
            elif k.startswith("module."):
                raw[k[len("module.") :]] = v
            else:
                raw[k] = v

        # Read pretraining input dimensions for pos-embed interpolation
        p_fdim = int(raw.pop("p_input_fdim", torch.tensor(128)).item())
        p_tdim = int(raw.pop("p_input_tdim", torch.tensor(1024)).item())

        # Drop pretraining-only keys
        drop_prefixes = ("head.", "cpredlayer.", "gpredlayer.")
        drop_keys = {"mask_embed"}
        state_dict = {
            k: v
            for k, v in raw.items()
            if k not in drop_keys and not any(k.startswith(dp) for dp in drop_prefixes)
        }

        # Interpolate positional embeddings if grid sizes differ
        if "pos_embed" in state_dict:
            orig_f = p_fdim // patch_size[0]
            orig_t = p_tdim // patch_size[1]
            target_f, target_t = _get_patch_shape(
                fstride,
                tstride,
                input_fdim,
                input_tdim,
                patch_size,
            )
            if orig_f != target_f or orig_t != target_t:
                logger.info(
                    "Interpolating pos_embed: (%d×%d → %d×%d)",
                    orig_f,
                    orig_t,
                    target_f,
                    target_t,
                )
                state_dict["pos_embed"] = self._interpolate_pos_embed(
                    state_dict["pos_embed"],
                    orig_f,
                    orig_t,
                    target_f,
                    target_t,
                )

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights from %s", path)
        if msg.missing_keys:
            logger.info("Missing keys (expected — new head): %s", msg.missing_keys)
        if msg.unexpected_keys:
            logger.warning("Unexpected keys: %s", msg.unexpected_keys)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram ``(B, n_mels, time_frames)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        # (B, F, T) → (B, 1, F, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.shape[0]

        # 2D patch embedding → (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Insert CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            if self.use_middle_cls_token:
                mid = self.num_patches // 2
                x = torch.cat((x[:, :mid, :], cls_tokens, x[:, mid:, :]), dim=1)
            else:
                x = torch.cat((cls_tokens, x), dim=1)

        # Positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Mamba blocks (prenorm residual)
        residual: torch.Tensor | None = None
        hidden_states = x

        if self.bidirectional and not self.use_bimamba:
            # External paired-layer bidirectional (original AuM)
            for i in range(len(self.layers) // 2):
                hidden_f, residual_f = self.layers[i * 2](
                    hidden_states,
                    residual,
                )
                hidden_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                )
                hidden_states = hidden_f + hidden_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        else:
            # Sequential: each BiMamba block handles bidirectionality
            # internally, or unidirectional vanilla Mamba
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)

        # Final prenorm: fold last hidden_states into residual, then norm
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        x = self.norm_f(residual)

        # Classification output
        if self.pool_type == "mean_no_cls" and self.use_cls_token:
            # SSAMBA default: average all tokens except CLS
            if self.use_middle_cls_token:
                mid = self.num_patches // 2
                x = torch.cat([x[:, :mid, :], x[:, mid + 1 :, :]], dim=1).mean(dim=1)
            else:
                x = x[:, 1:, :].mean(dim=1)  # skip CLS at position 0
        elif self.pool_type == "mean" or not self.use_cls_token:
            x = x.mean(dim=1)
        else:
            # pool_type == "cls" (default)
            if self.use_middle_cls_token:
                x = x[:, self.num_patches // 2]
            else:
                x = x[:, 0]

        return self.head(x)
