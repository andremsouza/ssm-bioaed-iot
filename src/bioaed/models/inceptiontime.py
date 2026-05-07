"""InceptionTime: 1D-CNN for multivariate time-series classification.

Faithful re-implementation of Fawaz et al. (DMKD 2020).  Treats
the log-mel spectrogram as a multivariate time series where mel bins
are independent channels processed by 1D convolutions.

Key architectural elements from the reference:
  1. Bottleneck (1×1 Conv) before multi-scale branches
  2. Multi-scale branches with kernel sizes [k, k//2, k//4]
  3. Max-pool branch (pool_size=3 → 1×1 Conv)
  4. Residual shortcut every 3 blocks
  5. GAP → Linear classifier

Reference: https://github.com/hfawaz/InceptionTime
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """Single Inception module with bottleneck and multi-scale 1D convolutions.

    Following the original Keras implementation, the module consists of:
      - An optional 1×1 bottleneck convolution to reduce channel count
      - Three parallel Conv1d branches at different kernel sizes
      - A MaxPool → 1×1 Conv branch on the *original* input (pre-bottleneck)
      - Concatenation → BatchNorm → ReLU

    Args:
        in_channels: Number of input channels.
        n_filters: Number of filters per convolution branch.
        kernel_sizes: List of kernel sizes for the parallel branches.
            If ``None``, derived from ``kernel_size`` as ``[k, k//2, k//4]``.
        kernel_size: Single kernel size used to derive the three branch
            kernels when ``kernel_sizes`` is not provided.
        use_bottleneck: Whether to use a 1×1 bottleneck before branches.
        bottleneck_size: Number of channels in the bottleneck layer.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: list[int] | None = None,
        kernel_size: int = 41,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
    ) -> None:
        super().__init__()

        # Derive kernel sizes per the reference: [k, k//2, k//4]
        if kernel_sizes is None:
            ks = kernel_size - 1  # reference uses kernel_size-1
            kernel_sizes = [ks // (2**i) for i in range(3)]

        self._kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.use_bottleneck = use_bottleneck

        # Bottleneck (1×1 conv to reduce channels before multi-scale branches)
        self.bottleneck: nn.Conv1d | None = None
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels,
                bottleneck_size,
                kernel_size=1,
                bias=False,
            )
            branch_in = bottleneck_size
        else:
            branch_in = in_channels

        # Multi-scale convolution branches (on bottleneck output)
        # Use padding='same' to match Keras reference (handles even kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_in,
                    n_filters,
                    kernel_size=ks,
                    padding="same",
                    bias=False,
                )
                for ks in kernel_sizes
            ]
        )

        # Max-pool branch (on *original* input, bypassing bottleneck)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(
            in_channels,
            n_filters,
            kernel_size=1,
            bias=False,
        )

        self.out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Inception module.

        Args:
            x: Input tensor of shape ``(batch, channels, time)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, time)``.
        """
        # Bottleneck
        if self.use_bottleneck and self.bottleneck is not None:
            x_bn = self.bottleneck(x)
        else:
            x_bn = x

        # Multi-scale convolution branches
        outs = [branch(x_bn) for branch in self.branches]

        # Max-pool branch on original input (pre-bottleneck)
        outs.append(self.conv_maxpool(self.maxpool(x)))

        out = torch.cat(outs, dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime for audio event detection.

    Stacks ``depth`` Inception modules with a residual shortcut
    (1×1 Conv + BN) applied every 3 blocks, followed by global
    average pooling and a linear classification head.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (mel frequency bins).
        depth: Number of stacked Inception modules (default 6).
        n_filters: Number of filters per convolution branch (default 32).
        kernel_sizes: Explicit list of kernel sizes for the multi-scale
            branches.  If ``None``, derived from ``kernel_size``.
        kernel_size: Single kernel size producing ``[k-1, (k-1)//2,
            (k-1)//4]`` when ``kernel_sizes`` is not provided.
        use_bottleneck: Use 1×1 bottleneck before branches.
        bottleneck_size: Number of bottleneck channels.
        use_residual: Apply residual shortcut every 3 blocks.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 64,
        depth: int = 6,
        n_filters: int = 32,
        kernel_sizes: list[int] | None = None,
        kernel_size: int = 41,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.depth = depth

        # Build Inception blocks and residual shortcuts
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        current_channels = in_channels
        residual_channels = in_channels

        for d in range(depth):
            block = InceptionModule(
                in_channels=current_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                kernel_size=kernel_size,
                use_bottleneck=use_bottleneck,
                bottleneck_size=bottleneck_size,
            )
            self.blocks.append(block)

            # Residual shortcut every 3 blocks (d % 3 == 2)
            if use_residual and d % 3 == 2:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(residual_channels, block.out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(block.out_channels),
                    )
                )
                residual_channels = block.out_channels
            else:
                self.shortcuts.append(None)  # type: ignore[arg-type]

            current_channels = block.out_channels

        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram of shape ``(batch, n_mels, time_frames)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        residual_input = x

        for d, (block, shortcut) in enumerate(zip(self.blocks, self.shortcuts)):
            x = block(x)
            if self.use_residual and d % 3 == 2 and shortcut is not None:
                x = torch.relu(x + shortcut(residual_input))
                residual_input = x

        # Global average pooling → classifier
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)
