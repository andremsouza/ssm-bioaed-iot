"""Tests for ablation study helper functions (no Hydra required)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from bioaed.ablation import _build_output_dir, _resolve_model_key


class TestResolveModelKey:
    def test_inceptiontime(self) -> None:
        cfg = OmegaConf.create({"model": {"_target_": "bioaed.models.inceptiontime.InceptionTime"}})
        assert _resolve_model_key(cfg) == "inceptiontime"

    def test_ast(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"_target_": "bioaed.models.ast_model.AudioSpectrogramTransformer"}}
        )
        assert _resolve_model_key(cfg) == "audiospectrogramtransformer"

    def test_missing_target(self) -> None:
        cfg = OmegaConf.create({"model": {"num_classes": 7}})
        assert _resolve_model_key(cfg) == "inceptiontime"


class TestBuildOutputDir:
    def test_dcqg_on_path(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "aswine"},
                "quality_gate": {"enabled": True},
                "seed": 42,
            }
        )
        out = _build_output_dir(cfg, "inceptiontime")
        assert "dcqg_on" in str(out)
        assert "seed_42" in str(out)

    def test_dcqg_off_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "aswine"},
                "quality_gate": {"enabled": False},
                "seed": 0,
            }
        )
        out = _build_output_dir(cfg, "inceptiontime")
        assert "dcqg_off" in str(out)

    def test_fold_included_in_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "anuraset"},
                "quality_gate": {"enabled": True},
                "seed": 1,
            }
        )
        out = _build_output_dir(cfg, "ast", fold=2)
        assert "fold_2" in str(out)
