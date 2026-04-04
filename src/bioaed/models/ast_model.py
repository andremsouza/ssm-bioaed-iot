"""Audio Spectrogram Transformer (AST) wrapper using timm ViT backbone.

Requires the ``timm`` package (install via ``pip install timm`` or
include it in the ``[cuda]`` extras).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer for multi-label audio classification.

    Wraps a ``timm`` Vision Transformer (ViT) backbone, treating the log-mel
    spectrogram as a single-channel image and adapting the patch embedding
    and classification head for audio tasks.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet-pretrained weights.
        model_name: timm model identifier (e.g., ``"vit_base_patch16_224"``).
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            msg = (
                "AudioSpectrogramTransformer requires 'timm'. "
                "Install it with: pip install timm, or use the [cuda] extras."
            )
            raise ImportError(msg)

        # Create ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # Single-channel spectrogram
            num_classes=0,  # Remove default head
        )

        # Get embedding dimension from the backbone
        embed_dim = self.backbone.num_features

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram of shape ``(batch, n_mels, time_frames)``.
                Will be reshaped to ``(batch, 1, n_mels, time_frames)``
                for the ViT patch embedding.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        # Add channel dimension: (B, n_mels, T) → (B, 1, n_mels, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Interpolate to ViT's expected input size (224x224)
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract features
        features = self.backbone(x)

        # Classify
        return self.classifier(features)
