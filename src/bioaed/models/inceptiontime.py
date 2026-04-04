"""InceptionTime: 1D-CNN for multivariate time-series classification.

Treats the log-mel spectrogram as a multivariate time series where
mel bins are independent channels processed by 1D convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """Single Inception module with multi-scale 1D convolutions.

    Args:
        in_channels: Number of input channels (mel bins).
        n_filters: Number of filters per convolution branch.
        kernel_sizes: List of kernel sizes for the parallel convolution branches.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 64,
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]

        # Use explicit symmetric padding (ks // 2) instead of padding='same'.
        # padding='same' on even kernel sizes with integer dilation forces
        # asymmetric padding → PyTorch must allocate a zero-padded copy of
        # the input.  Odd kernel sizes (e.g. 9, 19, 39) have integer symmetric
        # padding and avoid this entirely.
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    n_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=False,
                )
                for ks in kernel_sizes
            ]
        )

        # Max-pool branch
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )

        total_filters = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Inception module.

        Args:
            x: Input tensor of shape ``(batch, channels, time)``.

        Returns:
            Output tensor of shape ``(batch, total_filters, time)``.
        """
        branch_outputs = [branch(x) for branch in self.branches]
        mp_out = self.maxpool(x)
        branch_outputs.append(mp_out)
        # Align temporal dimension: trim all branches to the shortest length.
        # With odd kernel sizes and matching padding the lengths are always equal,
        # but guard against edge cases (e.g. very short inputs).
        min_len = min(b.shape[-1] for b in branch_outputs)
        branch_outputs = [b[:, :, :min_len] for b in branch_outputs]
        out = torch.cat(branch_outputs, dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime architecture for audio event detection.

    Stacks multiple Inception modules with residual connections,
    followed by global average pooling and a multi-label classification head.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (mel frequency bins).
        depth: Number of stacked Inception modules.
        n_filters: Number of filters per convolution branch.
        kernel_sizes: Kernel sizes for the multi-scale branches.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 64,
        depth: int = 2,
        n_filters: int = 64,
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        num_branches = len(kernel_sizes) + 1  # +1 for maxpool branch
        block_out_channels = n_filters * num_branches

        # Build Inception blocks
        modules: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(depth):
            modules.append(InceptionModule(current_channels, n_filters, kernel_sizes))
            current_channels = block_out_channels

        self.inception_blocks = nn.Sequential(*modules)

        # Residual shortcut
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, block_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(block_out_channels),
        )

        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(block_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram of shape ``(batch, n_mels, time_frames)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        residual = self.residual(x)
        out = self.inception_blocks(x)

        # Add residual (shape alignment via the 1x1 conv above)
        out = out + residual
        out = torch.relu(out)

        # Global average pooling → classifier
        out = self.gap(out).squeeze(-1)
        return self.classifier(out)
