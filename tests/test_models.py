"""Smoke tests for model architectures."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from bioaed.models import _CLASS_NAME_TO_KEY, MODEL_REGISTRY, build_model
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
        model = InceptionTime(num_classes=7, in_channels=64, depth=3, n_filters=16)
        logits = model(dummy_spectrogram)
        assert logits.shape == (4, 7)

    def test_forward_different_classes(self) -> None:
        """Model should work with different number of classes."""
        model = InceptionTime(num_classes=42, in_channels=64, depth=6, n_filters=32)
        x = torch.randn(2, 64, 200)
        logits = model(x)
        assert logits.shape == (2, 42)

    def test_gradients_flow(self, dummy_spectrogram: torch.Tensor) -> None:
        """Gradients should flow through the model."""
        model = InceptionTime(num_classes=7, in_channels=64, depth=3, n_filters=16)
        logits = model(dummy_spectrogram)
        loss = logits.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_bottleneck_reduces_channels(self) -> None:
        """Bottleneck should create a 1x1 conv reducing channel count."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32, use_bottleneck=True, bottleneck_size=16)
        assert mod.bottleneck is not None
        assert mod.bottleneck.out_channels == 16

    def test_no_bottleneck_when_disabled(self) -> None:
        """Bottleneck should not exist when disabled."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32, use_bottleneck=False)
        assert mod.bottleneck is None

    def test_residual_every_3_blocks(self) -> None:
        """Shortcuts should only be present at blocks 2, 5, 8, etc."""
        model = InceptionTime(num_classes=7, in_channels=64, depth=6, n_filters=32)
        # Blocks 0..5: shortcuts at indices 2 and 5
        for d in range(6):
            if d % 3 == 2:
                assert model.shortcuts[d] is not None, f"Block {d} should have a shortcut"
            else:
                assert model.shortcuts[d] is None, f"Block {d} should NOT have a shortcut"

    def test_no_residual_when_disabled(self) -> None:
        """All shortcuts should be None when use_residual=False."""
        model = InceptionTime(
            num_classes=7, in_channels=64, depth=6, n_filters=32, use_residual=False
        )
        for d in range(6):
            assert model.shortcuts[d] is None

    def test_kernel_size_derivation(self) -> None:
        """kernel_size=41 should derive branches [40, 20, 10]."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32, kernel_size=41)
        actual_ks = [b.kernel_size[0] for b in mod.branches]
        assert actual_ks == [40, 20, 10]

    def test_explicit_kernel_sizes_override(self) -> None:
        """Explicit kernel_sizes should override kernel_size derivation."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32, kernel_sizes=[9, 19, 39])
        actual_ks = [b.kernel_size[0] for b in mod.branches]
        assert actual_ks == [9, 19, 39]

    def test_inception_module_output_channels(self) -> None:
        """Output channels should be n_filters * (3 branches + 1 maxpool)."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32)
        assert mod.out_channels == 32 * 4  # 3 conv branches + 1 maxpool

    def test_inception_module_forward_shape(self) -> None:
        """Module output should preserve time dimension."""
        from bioaed.models.inceptiontime import InceptionModule

        mod = InceptionModule(in_channels=64, n_filters=32)
        x = torch.randn(2, 64, 100)
        out = mod(x)
        assert out.shape == (2, 128, 100)


class TestAudioSpectrogramTransformer:
    """Tests for AST — DeiT backbone with overlapping patches and AudioSet pretraining."""

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_forward_shape_no_pretrain(self) -> None:
        """Forward with random init, small model for speed."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(
            num_classes=7,
            fstride=10,
            tstride=10,
            input_fdim=64,
            input_tdim=101,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        x = torch.randn(2, 64, 101)
        logits = model(x)
        assert logits.shape == (2, 7)

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_forward_different_input_dims(self) -> None:
        """AST should handle various spectrogram dimensions."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(
            num_classes=42,
            fstride=10,
            tstride=10,
            input_fdim=64,
            input_tdim=301,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        x = torch.randn(2, 64, 301)
        logits = model(x)
        assert logits.shape == (2, 42)

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_uses_deit_backbone(self) -> None:
        """Model should use DeiT with cls_token and dist_token."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(
            num_classes=7,
            input_fdim=64,
            input_tdim=101,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        assert hasattr(model.v, "cls_token")
        assert hasattr(model.v, "dist_token")

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_single_channel_patch_embed(self) -> None:
        """Patch projection should accept 1-channel input."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(
            num_classes=7,
            fstride=10,
            tstride=10,
            input_fdim=64,
            input_tdim=101,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        proj = model.v.patch_embed.proj
        assert proj.in_channels == 1
        assert proj.stride == (10, 10)

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_pos_embed_shape(self) -> None:
        """Positional embedding should have num_patches + 2 (cls+dist) tokens."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer, _get_patch_shape

        model = AudioSpectrogramTransformer(
            num_classes=7,
            fstride=10,
            tstride=10,
            input_fdim=64,
            input_tdim=101,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        f_dim, t_dim = _get_patch_shape(10, 10, 64, 101)
        expected_patches = f_dim * t_dim
        assert model.v.pos_embed.shape[1] == expected_patches + 2

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_validation_audioset_requires_imagenet(self) -> None:
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        with pytest.raises(ValueError, match="imagenet_pretrain"):
            AudioSpectrogramTransformer(
                imagenet_pretrain=False,
                audioset_pretrain=True,
            )

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_validation_audioset_requires_base384(self) -> None:
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        with pytest.raises(ValueError, match="base384"):
            AudioSpectrogramTransformer(
                audioset_pretrain=True,
                model_size="tiny224",
            )

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_validation_unknown_model_size(self) -> None:
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        with pytest.raises(ValueError, match="Unknown model_size"):
            AudioSpectrogramTransformer(
                model_size="nonexistent",
                audioset_pretrain=False,
            )

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_audioset_pretrain_with_mock(self) -> None:
        """AudioSet pretrained init with a mock checkpoint."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        # Create a fake AudioSet checkpoint (matching expected layout)
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
        fake_sd = {f"module.{k}": v for k, v in base.state_dict().items()}
        del base  # free memory

        with patch(
            "bioaed.models.ast_model.torch.hub.load_state_dict_from_url", return_value=fake_sd
        ):
            model = AudioSpectrogramTransformer(
                num_classes=7,
                fstride=10,
                tstride=10,
                input_fdim=64,
                input_tdim=101,
                imagenet_pretrain=True,
                audioset_pretrain=True,
                model_size="base384",
            )

        x = torch.randn(2, 64, 101)
        logits = model(x)
        assert logits.shape == (2, 7)
        # Verify embed_dim matches base384
        assert model.embed_dim == 768

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_gradients_flow(self) -> None:
        """Gradients should flow through backbone and classification head."""
        from bioaed.models.ast_model import AudioSpectrogramTransformer

        model = AudioSpectrogramTransformer(
            num_classes=7,
            input_fdim=64,
            input_tdim=101,
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size="tiny224",
        )
        x = torch.randn(2, 64, 101)
        logits = model(x)
        logits.sum().backward()
        # Verify gradients flow through the classification head
        for p in model.mlp_head.parameters():
            assert p.grad is not None
        # Verify gradients flow through the backbone transformer blocks
        for p in model.v.blocks.parameters():
            assert p.grad is not None


class TestASTHelpers:
    """Tests for AST module-level helper functions."""

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_get_patch_shape_aswine(self) -> None:
        from bioaed.models.ast_model import _get_patch_shape

        f, t = _get_patch_shape(10, 10, 64, 101)
        assert f == 5
        assert t == 9

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_get_patch_shape_anuraset(self) -> None:
        from bioaed.models.ast_model import _get_patch_shape

        f, t = _get_patch_shape(10, 10, 64, 301)
        assert f == 5
        assert t == 29

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_get_patch_shape_audioset(self) -> None:
        from bioaed.models.ast_model import _get_patch_shape

        f, t = _get_patch_shape(10, 10, 128, 1024)
        assert f == 12
        assert t == 101

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_adapt_pos_embed_crops(self) -> None:
        """Positional embeddings should be cropped for smaller input."""
        from bioaed.models.ast_model import _adapt_pos_embed

        embed_dim = 32
        # Create a fake model with pos_embed for a 6x6 grid + 2 tokens
        fake = torch.nn.Module()
        fake.pos_embed = torch.nn.Parameter(torch.randn(1, 38, embed_dim))
        _adapt_pos_embed(fake, embed_dim, 6, 6, 3, 4, 12)
        assert fake.pos_embed.shape == (1, 14, embed_dim)  # 12 patches + 2 tokens

    @pytest.mark.skipif(not _check_timm_available(), reason="timm not installed")
    def test_adapt_pos_embed_interpolates(self) -> None:
        """Positional embeddings should be interpolated for larger input."""
        from bioaed.models.ast_model import _adapt_pos_embed

        embed_dim = 32
        fake = torch.nn.Module()
        fake.pos_embed = torch.nn.Parameter(
            torch.randn(1, 6, embed_dim)
        )  # 4 patches (2x2) + 2 tokens
        _adapt_pos_embed(fake, embed_dim, 2, 2, 5, 5, 25)
        assert fake.pos_embed.shape == (1, 27, embed_dim)  # 25 patches + 2 tokens


class TestAudioMamba:
    """Forward-pass and architecture tests for Audio Mamba (AuM)."""

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_forward_shape_with_cls_token(self) -> None:
        """Output shape should be (batch, num_classes) with CLS token."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            use_cls_token=True,
        ).cuda()
        x = torch.randn(2, 64, 96).cuda()
        logits = model(x)
        assert logits.shape == (2, 7)

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_forward_shape_without_cls_token(self) -> None:
        """Output shape should be (batch, num_classes) with mean pooling."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            use_cls_token=False,
        ).cuda()
        x = torch.randn(2, 64, 96).cuda()
        logits = model(x)
        assert logits.shape == (2, 7)

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_forward_unidirectional(self) -> None:
        """Unidirectional mode should also produce valid output."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=5,
            embed_dim=64,
            depth=3,
            d_state=8,
            input_fdim=32,
            input_tdim=48,
            bidirectional=False,
        ).cuda()
        x = torch.randn(2, 32, 48).cuda()
        logits = model(x)
        assert logits.shape == (2, 5)

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_gradient_flow(self) -> None:
        """Gradients should flow to all parameters."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=3,
            embed_dim=32,
            depth=2,
            d_state=8,
            input_fdim=32,
            input_tdim=32,
        ).cuda()
        x = torch.randn(1, 32, 32).cuda()
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_bidirectional_requires_even_depth(self) -> None:
        """Bidirectional mode must reject odd depth."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        with pytest.raises(ValueError, match="even depth"):
            AudioMamba(num_classes=7, embed_dim=64, depth=3, bidirectional=True)

    def test_cls_token_parameter_exists(self) -> None:
        """CLS token should be a learnable parameter when enabled."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            use_cls_token=True,
        )
        assert hasattr(model, "cls_token")
        assert model.cls_token.shape == (1, 1, 64)

    def test_no_cls_token_when_disabled(self) -> None:
        """CLS token should not exist when disabled."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            use_cls_token=False,
        )
        assert not hasattr(model, "cls_token")

    def test_positional_embedding_shape(self) -> None:
        """Positional embedding should match num_patches + num_tokens."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        # 64 fdim, 96 tdim, patch=16, stride=16 → 4×6 = 24 patches + 1 CLS
        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            patch_size=16,
        )
        assert model.pos_embed.shape == (1, 25, 64)

    def test_layers_count_matches_depth(self) -> None:
        """Number of Mamba layers should match depth parameter."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(num_classes=7, embed_dim=64, depth=6, d_state=8)
        assert len(model.layers) == 6

    def test_pool_type_default_is_cls(self) -> None:
        """Default pool_type should be 'cls'."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(num_classes=7, embed_dim=64, depth=2, d_state=8)
        assert model.pool_type == "cls"

    def test_pool_type_mean_no_cls_stored(self) -> None:
        """pool_type='mean_no_cls' should be stored in model."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            pool_type="mean_no_cls",
        )
        assert model.pool_type == "mean_no_cls"

    @pytest.mark.skipif(not _check_mamba_available(), reason="mamba-ssm not installed or no CUDA")
    def test_pool_type_mean_no_cls_forward(self) -> None:
        """pool_type='mean_no_cls' should produce valid output."""
        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=64,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            use_cls_token=True,
            pool_type="mean_no_cls",
        ).cuda()
        x = torch.randn(2, 64, 96).cuda()
        logits = model(x)
        assert logits.shape == (2, 7)

    def test_interpolate_pos_embed_center_crops(self) -> None:
        """Pos-embed should center-crop when target < pretrained grid."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=32,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            patch_size=16,
            use_cls_token=True,
        )
        # Simulate pretrained PE: 8f × 64t patches + 1 CLS = 513 entries
        pretrained_pe = torch.randn(1, 513, 32)
        # Target grid: 4f × 6t = 24 patches (both dims smaller)
        result = model._interpolate_pos_embed(pretrained_pe, 8, 64, 4, 6)
        # Should be 24 patches + 1 CLS = 25
        assert result.shape == (1, 25, 32)

    def test_interpolate_pos_embed_interpolates_larger(self) -> None:
        """Pos-embed should interpolate when target > pretrained grid."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if not _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm not installed")

        from bioaed.models.audio_mamba import AudioMamba

        model = AudioMamba(
            num_classes=7,
            embed_dim=32,
            depth=2,
            d_state=8,
            input_fdim=64,
            input_tdim=96,
            patch_size=16,
            use_cls_token=True,
        )
        # Simulate small pretrained PE: 2f × 3t = 6 patches + 1 CLS
        pretrained_pe = torch.randn(1, 7, 32)
        # Target: 4f × 6t = 24 patches (both dims larger)
        result = model._interpolate_pos_embed(pretrained_pe, 2, 3, 4, 6)
        assert result.shape == (1, 25, 32)


class TestPatchEmbed2D:
    """Tests for PatchEmbed2D — no mamba-ssm or CUDA required."""

    def test_forward_shape(self) -> None:
        """Output should be (batch, num_patches, embed_dim)."""
        from bioaed.models.audio_mamba import PatchEmbed2D

        pe = PatchEmbed2D(patch_size=(16, 16), fstride=16, tstride=16, embed_dim=128)
        x = torch.randn(2, 1, 64, 96)  # (B, 1, F=64, T=96) → 4×6 = 24 patches
        out = pe(x)
        assert out.shape == (2, 24, 128)

    def test_overlapping_strides(self) -> None:
        """Overlapping strides should produce more patches."""
        from bioaed.models.audio_mamba import PatchEmbed2D

        pe = PatchEmbed2D(patch_size=(16, 16), fstride=10, tstride=10, embed_dim=64)
        x = torch.randn(2, 1, 64, 96)
        out = pe(x)
        # f_patches = (64-16)/10+1 = 5.8 → 5, t_patches = (96-16)/10+1 = 9
        assert out.shape == (2, 5 * 9, 64)

    def test_init_creates_conv2d(self) -> None:
        """Init should create a Conv2d projection layer."""
        import torch.nn as nn

        from bioaed.models.audio_mamba import PatchEmbed2D

        pe = PatchEmbed2D(patch_size=(16, 16), fstride=16, tstride=16, embed_dim=64)
        assert isinstance(pe.proj, nn.Conv2d)


class TestGetPatchShape:
    """Tests for the _get_patch_shape helper."""

    def test_no_overlap(self) -> None:
        from bioaed.models.audio_mamba import _get_patch_shape

        f, t = _get_patch_shape(16, 16, 64, 96)
        assert (f, t) == (4, 6)

    def test_with_overlap(self) -> None:
        from bioaed.models.audio_mamba import _get_patch_shape

        f, t = _get_patch_shape(10, 10, 64, 96, patch_size=(16, 16))
        assert (f, t) == (5, 9)


class TestDropPath:
    """Tests for the _DropPath module."""

    def test_no_drop_in_eval(self) -> None:
        from bioaed.models.audio_mamba import _DropPath

        dp = _DropPath(0.5)
        dp.eval()
        x = torch.randn(2, 4, 8)
        out = dp(x)
        assert torch.equal(x, out)

    def test_zero_drop_rate(self) -> None:
        from bioaed.models.audio_mamba import _DropPath

        dp = _DropPath(0.0)
        dp.train()
        x = torch.randn(2, 4, 8)
        out = dp(x)
        assert torch.equal(x, out)


class TestMambaImportError:
    """Test ImportError paths when mamba-ssm is not installed (CPU-only env)."""

    def test_mamba_block_raises_import_error(self) -> None:
        """MambaBlock must raise ImportError when mamba-ssm is absent."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm is installed; cannot test ImportError path")

        from bioaed.models.audio_mamba import MambaBlock

        with pytest.raises(ImportError, match="mamba-ssm"):
            MambaBlock(dim=64, d_state=8)

    def test_audio_mamba_raises_import_error(self) -> None:
        """AudioMamba must raise ImportError when mamba-ssm is absent."""
        from bioaed.models.audio_mamba import _MAMBA_AVAILABLE

        if _MAMBA_AVAILABLE:
            pytest.skip("mamba-ssm is installed; cannot test ImportError path")

        from bioaed.models.audio_mamba import AudioMamba

        with pytest.raises(ImportError, match="mamba-ssm"):
            AudioMamba(num_classes=7, embed_dim=64)


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
