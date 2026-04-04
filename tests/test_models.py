"""Smoke tests for model architectures."""

from __future__ import annotations

import pytest
import torch

from bioaed.models import MODEL_REGISTRY, build_model
from bioaed.models.inceptiontime import InceptionTime


def _check_timm_available() -> bool:
    """Check if timm is importable."""
    try:
        import timm  # noqa: F401

        return True
    except ImportError:
        return False


def _check_mamba_available() -> bool:
    """Check if mamba_ssm is importable and CUDA is available."""
    try:
        import mamba_ssm  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


class TestInceptionTime:
    """Forward-pass smoke tests for InceptionTime."""

    def test_forward_shape(self, dummy_spectrogram: torch.Tensor) -> None:
        """Output shape should be (batch, num_classes)."""
        model = InceptionTime(num_classes=7, in_channels=64, depth=1, n_filters=16)
        logits = model(dummy_spectrogram)
        assert logits.shape == (4, 7)

    def test_forward_different_classes(self) -> None:
        """Model should work with different number of classes."""
        model = InceptionTime(num_classes=42, in_channels=64, depth=2, n_filters=32)
        x = torch.randn(2, 64, 200)
        logits = model(x)
        assert logits.shape == (2, 42)

    def test_gradients_flow(self, dummy_spectrogram: torch.Tensor) -> None:
        """Gradients should flow through the model."""
        model = InceptionTime(num_classes=7, in_channels=64, depth=1, n_filters=16)
        logits = model(dummy_spectrogram)
        loss = logits.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAudioSpectrogramTransformer:
    """Forward-pass smoke tests for AST (requires timm)."""

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_forward_shape(self) -> None:
        """Output shape should be (batch, num_classes)."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(num_classes=7, pretrained=False)
        x = torch.randn(2, 64, 100)
        logits = model(x)
        assert logits.shape == (2, 7)


class TestAudioMamba:
    """Forward-pass smoke tests for Audio Mamba (requires mamba-ssm + CUDA)."""

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_forward_shape(self) -> None:
        """Output shape should be (batch, num_classes)."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(num_classes=7, d_model=64, n_layers=1, d_state=8).cuda()
        x = torch.randn(2, 64, 100).cuda()
        logits = model(x)
        assert logits.shape == (2, 7)


class TestBuildModel:
    """Tests for the model registry and build_model factory."""

    def test_registry_contains_inceptiontime(self) -> None:
        assert "inceptiontime" in MODEL_REGISTRY

    def test_build_inceptiontime(self) -> None:
        model = build_model("inceptiontime", num_classes=7, in_channels=64)
        assert isinstance(model, InceptionTime)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model")

    def test_build_model_forward(self) -> None:
        model = build_model("inceptiontime", num_classes=5, in_channels=32, depth=1, n_filters=8)
        x = torch.randn(2, 32, 50)
        out = model(x)
        assert out.shape == (2, 5)
