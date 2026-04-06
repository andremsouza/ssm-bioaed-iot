"""Smoke tests for model architectures."""

from __future__ import annotations

import pytest
import torch

from bioaed.models import MODEL_REGISTRY, _CLASS_NAME_TO_KEY, build_model
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


class TestPatchEmbedding1D:
    """Tests for PatchEmbedding1D — no mamba-ssm or CUDA required."""

    def test_forward_shape(self) -> None:
        """Output should be (batch, num_patches, d_model)."""
        from bioaed.models.audio_mamba import PatchEmbedding1D

        pe = PatchEmbedding1D(in_channels=64, d_model=128, patch_size=16)
        x = torch.randn(2, 64, 96)  # 96/16 = 6 patches
        out = pe(x)
        assert out.shape == (2, 6, 128)

    def test_init_creates_layers(self) -> None:
        """Init should create proj (Conv1d) and norm (LayerNorm)."""
        import torch.nn as nn

        from bioaed.models.audio_mamba import PatchEmbedding1D

        pe = PatchEmbedding1D(in_channels=32, d_model=64, patch_size=8)
        assert isinstance(pe.proj, nn.Conv1d)
        assert isinstance(pe.norm, nn.LayerNorm)


class TestMambaImportError:
    """Test ImportError paths when mamba-ssm is not installed (CPU-only env)."""

    def test_bidir_mamba_block_raises_import_error(self) -> None:
        """BidirectionalMambaBlock must raise ImportError when mamba-ssm is absent."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm is installed; cannot test ImportError path")

        from bioaed.models.audio_mamba import BidirectionalMambaBlock

        with pytest.raises(ImportError, match="mamba-ssm"):
            BidirectionalMambaBlock(d_model=64, d_state=8)

    def test_audio_mamba_raises_import_error(self) -> None:
        """AudioMamba must raise ImportError when mamba-ssm is absent."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm is installed; cannot test ImportError path")

        from bioaed.models.audio_mamba import AudioMamba

        with pytest.raises(ImportError, match="mamba-ssm"):
            AudioMamba(num_classes=7, d_model=64)


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


class TestClassNameToKey:
    """Tests for the _CLASS_NAME_TO_KEY reverse-lookup mapping."""

    def test_inceptiontime_class_resolves(self) -> None:
        # Class name lowercased must map to the registry key.
        assert _CLASS_NAME_TO_KEY["inceptiontime"] == "inceptiontime"

    def test_ast_class_resolves(self) -> None:
        if "ast" not in MODEL_REGISTRY:
            pytest.skip("timm not installed; AST not registered")
        assert _CLASS_NAME_TO_KEY["audiospectrogramtransformer"] == "ast"

    def test_audio_mamba_class_resolves(self) -> None:
        if "audio_mamba" not in MODEL_REGISTRY:
            pytest.skip("mamba-ssm not installed; AudioMamba not registered")
        assert _CLASS_NAME_TO_KEY["audiomamba"] == "audio_mamba"

    def test_all_registry_keys_covered(self) -> None:
        # Every key in MODEL_REGISTRY must be reachable via _CLASS_NAME_TO_KEY.
        reachable_keys = set(_CLASS_NAME_TO_KEY.values())
        assert set(MODEL_REGISTRY.keys()) == reachable_keys

    def test_target_string_resolution_ast(self) -> None:
        # Simulate what ablation/train/evaluate do with a Hydra _target_ string.
        if "ast" not in MODEL_REGISTRY:
            pytest.skip("timm not installed")
        target = "bioaed.models.ast_model.AudioSpectrogramTransformer"
        class_name = target.rsplit(".", 1)[-1].lower()
        model_key = _CLASS_NAME_TO_KEY.get(class_name, class_name)
        assert model_key == "ast"

    def test_target_string_resolution_audio_mamba(self) -> None:
        if "audio_mamba" not in MODEL_REGISTRY:
            pytest.skip("mamba-ssm not installed")
        target = "bioaed.models.audio_mamba.AudioMamba"
        class_name = target.rsplit(".", 1)[-1].lower()
        model_key = _CLASS_NAME_TO_KEY.get(class_name, class_name)
        assert model_key == "audio_mamba"

    def test_target_string_resolution_inceptiontime(self) -> None:
        target = "bioaed.models.inceptiontime.InceptionTime"
        class_name = target.rsplit(".", 1)[-1].lower()
        model_key = _CLASS_NAME_TO_KEY.get(class_name, class_name)
        assert model_key == "inceptiontime"

    def test_unknown_class_falls_through(self) -> None:
        # An unknown class name should fall through unchanged (not raise here).
        model_key = _CLASS_NAME_TO_KEY.get("completelymadeupclassname", "completelymadeupclassname")
        assert model_key == "completelymadeupclassname"
