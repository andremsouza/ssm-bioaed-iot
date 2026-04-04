"""Audio Mamba: Bidirectional State Space Model for audio classification.

Requires the ``mamba-ssm`` and ``causal-conv1d`` packages, available
only with CUDA (install via the ``[cuda]`` extras).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba

    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class PatchEmbedding1D(nn.Module):
    """1D patch embedding for flattening spectrograms into token sequences.

    Args:
        in_channels: Number of input channels (mel bins).
        d_model: Embedding dimension.
        patch_size: Size of each temporal patch.
    """

    def __init__(self, in_channels: int = 64, d_model: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed spectrogram patches.

        Args:
            x: Input of shape ``(batch, in_channels, time_frames)``.

        Returns:
            Token sequence of shape ``(batch, num_patches, d_model)``.
        """
        # Conv1d: (B, C, T) → (B, d_model, num_patches)
        x = self.proj(x)
        # Transpose to (B, num_patches, d_model) for Mamba
        x = x.transpose(1, 2)
        return self.norm(x)


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba block processing sequences forward and backward.

    Args:
        d_model: Model dimension.
        d_state: SSM state expansion factor.
    """

    def __init__(self, d_model: int = 192, d_state: int = 16) -> None:
        super().__init__()
        if not _MAMBA_AVAILABLE:
            msg = (
                "AudioMamba requires 'mamba-ssm'. "
                "Install with: pip install mamba-ssm causal-conv1d, "
                "or use the [cuda] extras."
            )
            raise ImportError(msg)

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional SSM processing.

        Args:
            x: Token sequence of shape ``(batch, seq_len, d_model)``.

        Returns:
            Processed sequence of shape ``(batch, seq_len, d_model)``.
        """
        residual = x
        x = self.norm(x)

        # Forward pass
        out_fwd = self.mamba_fwd(x)

        # Backward pass (reverse, process, reverse back)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])

        # Concatenate and project
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.proj(out)

        return out + residual


class AudioMamba(nn.Module):
    """Audio Mamba: Bidirectional SSM architecture for audio classification.

    Processes log-mel spectrograms through 1D patch embedding, stacked
    bidirectional Mamba blocks, and a multi-label classification head.
    Achieves linear-time complexity O(n) vs Transformer's O(n²).

    Args:
        num_classes: Number of output classes.
        d_model: Model/embedding dimension.
        n_layers: Number of stacked bidirectional Mamba blocks.
        d_state: SSM state expansion factor.
        bidirectional: Whether to use bidirectional processing.
        patch_size: Temporal patch size for the embedding layer.
        in_channels: Number of input mel frequency bins.
    """

    def __init__(
        self,
        num_classes: int = 7,
        d_model: int = 192,
        n_layers: int = 4,
        d_state: int = 16,
        bidirectional: bool = True,
        patch_size: int = 16,
        in_channels: int = 64,
    ) -> None:
        super().__init__()

        if not _MAMBA_AVAILABLE:
            msg = (
                "AudioMamba requires 'mamba-ssm'. "
                "Install with: pip install mamba-ssm causal-conv1d, "
                "or use the [cuda] extras."
            )
            raise ImportError(msg)

        self.patch_embed = PatchEmbedding1D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
        )

        # Stack Mamba blocks
        if bidirectional:
            self.blocks = nn.ModuleList(
                [BidirectionalMambaBlock(d_model, d_state) for _ in range(n_layers)]
            )
        else:
            # Unidirectional fallback
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(nn.LayerNorm(d_model), Mamba(d_model=d_model, d_state=d_state))
                    for _ in range(n_layers)
                ]
            )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram of shape ``(batch, n_mels, time_frames)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Process through Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling over sequence dimension
        x = self.norm(x)
        x = x.mean(dim=1)

        return self.classifier(x)
