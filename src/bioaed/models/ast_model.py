"""Audio Spectrogram Transformer (AST) per Gong et al. (Interspeech 2021).

Faithful re-implementation using a DeiT (distilled ViT) backbone with
overlapping patch embeddings and positional-embedding interpolation for
variable-length spectrograms.  Supports both ImageNet and AudioSet pretrained
weights.

Reference: https://github.com/YuanGongND/ast

Requires the ``timm`` package (``pip install timm``).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

logger = logging.getLogger(__name__)

# DeiT model identifiers (timm >= 0.9)
_DEIT_MODELS: dict[str, str] = {
    "tiny224": "deit_tiny_distilled_patch16_224",
    "small224": "deit_small_distilled_patch16_224",
    "base224": "deit_base_distilled_patch16_224",
    "base384": "deit_base_distilled_patch16_384",
}

# AudioSet pretrained checkpoint (0.459 mAP, fstride=tstride=10, 128 mel bins)
_AUDIOSET_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
_AUDIOSET_CKPT = "audioset_10_10_0.4593.pth"

# Patch grid of the AudioSet pretrained model (128 fdim, 1024 tdim, stride 10)
_AS_F, _AS_T, _AS_DIM = 12, 101, 768


def _get_patch_shape(
    fstride: int,
    tstride: int,
    input_fdim: int,
    input_tdim: int,
) -> tuple[int, int]:
    """Return ``(f_dim, t_dim)`` — the patch grid size for the given input."""
    f_dim = (input_fdim - 16) // fstride + 1
    t_dim = (input_tdim - 16) // tstride + 1
    return f_dim, t_dim


def _adapt_pos_embed(
    model: nn.Module,
    embed_dim: int,
    orig_f: int,
    orig_t: int,
    target_f: int,
    target_t: int,
    num_patches: int,
) -> None:
    """Crop or interpolate positional embeddings from *orig* to *target* grid."""
    orig_n = orig_f * orig_t
    pos = (
        model.pos_embed[:, 2:, :]
        .detach()
        .reshape(1, orig_n, embed_dim)
        .transpose(1, 2)
        .reshape(1, embed_dim, orig_f, orig_t)
    )

    # Time dimension
    if target_t <= orig_t:
        c = orig_t // 2
        pos = pos[:, :, :, c - target_t // 2 : c - target_t // 2 + target_t]
    else:
        pos = nn.functional.interpolate(pos, size=(orig_f, target_t), mode="bilinear")

    # Frequency dimension
    if target_f <= orig_f:
        c = orig_f // 2
        pos = pos[:, :, c - target_f // 2 : c - target_f // 2 + target_f, :]
    else:
        pos = nn.functional.interpolate(pos, size=(target_f, target_t), mode="bilinear")

    pos = pos.reshape(1, embed_dim, num_patches).transpose(1, 2)
    model.pos_embed = nn.Parameter(torch.cat([model.pos_embed[:, :2, :].detach(), pos], dim=1))


class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer for multi-label audio classification.

    Wraps a DeiT backbone with single-channel patch embedding, overlapping
    patches (stride < patch size), and positional-embedding interpolation for
    arbitrary spectrogram dimensions.

    Args:
        num_classes: Number of output classes.
        fstride: Patch stride on the frequency axis (default 10 → 6-pixel overlap).
        tstride: Patch stride on the time axis (default 10 → 6-pixel overlap).
        input_fdim: Number of frequency bins (mel bands) in the spectrogram.
        input_tdim: Number of time frames in the spectrogram.
        imagenet_pretrain: Load ImageNet-pretrained DeiT weights.
        audioset_pretrain: Load AudioSet + ImageNet pretrained weights.
            Requires ``imagenet_pretrain=True`` and ``model_size="base384"``.
        model_size: DeiT variant — ``tiny224``, ``small224``, ``base224``,
            or ``base384`` (recommended).
    """

    def __init__(
        self,
        num_classes: int = 527,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = True,
        model_size: str = "base384",
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "AudioSpectrogramTransformer requires 'timm'. Install with: pip install timm"
            )

        if audioset_pretrain and not imagenet_pretrain:
            raise ValueError("audioset_pretrain requires imagenet_pretrain=True.")
        if audioset_pretrain and model_size != "base384":
            raise ValueError("AudioSet pretrained weights only available for model_size='base384'.")

        timm_name = _DEIT_MODELS.get(model_size)
        if timm_name is None:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Choose from: {list(_DEIT_MODELS)}"
            )

        if not audioset_pretrain:
            self._init_from_imagenet(
                timm_name,
                imagenet_pretrain,
                fstride,
                tstride,
                input_fdim,
                input_tdim,
                num_classes,
            )
        else:
            self._init_from_audioset(
                fstride,
                tstride,
                input_fdim,
                input_tdim,
                num_classes,
            )

    # -------------------------------------------------------------- #
    # ImageNet (or scratch) init
    # -------------------------------------------------------------- #
    def _init_from_imagenet(
        self,
        timm_name: str,
        pretrained: bool,
        fstride: int,
        tstride: int,
        input_fdim: int,
        input_tdim: int,
        num_classes: int,
    ) -> None:
        self.v = timm.create_model(timm_name, pretrained=pretrained)
        orig_n = self.v.patch_embed.num_patches
        orig_hw = int(orig_n**0.5)
        self.embed_dim: int = self.v.pos_embed.shape[2]

        # Downstream classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes),
        )

        f_dim, t_dim = _get_patch_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        # 3-channel ImageNet → 1-channel spectrogram patch projection
        new_proj = nn.Conv2d(
            1,
            self.embed_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        if pretrained:
            new_proj.weight = nn.Parameter(self.v.patch_embed.proj.weight.sum(dim=1, keepdim=True))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # Adapt positional embeddings
        if pretrained:
            _adapt_pos_embed(
                self.v,
                self.embed_dim,
                orig_hw,
                orig_hw,
                f_dim,
                t_dim,
                num_patches,
            )
        else:
            new_pos = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
            nn.init.trunc_normal_(new_pos, std=0.02)
            self.v.pos_embed = new_pos

    # -------------------------------------------------------------- #
    # AudioSet + ImageNet init
    # -------------------------------------------------------------- #
    def _init_from_audioset(
        self,
        fstride: int,
        tstride: int,
        input_fdim: int,
        input_tdim: int,
        num_classes: int,
    ) -> None:
        # Download (or use cached) AudioSet checkpoint
        sd = torch.hub.load_state_dict_from_url(
            _AUDIOSET_URL,
            map_location="cpu",
            file_name=_AUDIOSET_CKPT,
        )
        # Strip DataParallel 'module.' key prefix
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        # Build a model matching the AudioSet checkpoint layout
        base = AudioSpectrogramTransformer(
            num_classes=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="base384",
        )
        base.load_state_dict(sd, strict=False)

        # Transfer backbone
        self.v = base.v
        self.embed_dim: int = self.v.pos_embed.shape[2]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes),
        )

        # Adapt positional embeddings from AudioSet dims to target dims
        f_dim, t_dim = _get_patch_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        _adapt_pos_embed(
            self.v,
            _AS_DIM,
            _AS_F,
            _AS_T,
            f_dim,
            t_dim,
            num_patches,
        )

    # -------------------------------------------------------------- #
    # Forward
    # -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram ``(B, n_mels, time_frames)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, F, T) → (B, 1, F, T)

        B = x.shape[0]

        # Patch embed — bypass PatchEmbed.forward to skip strict size checks
        x = self.v.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = self.v.patch_embed.norm(x)

        # Prepend [CLS] and [DIST] tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)

        x = self.v.norm(x)

        # Average CLS and distillation token outputs (DeiT convention)
        x = (x[:, 0] + x[:, 1]) / 2

        return self.mlp_head(x)
