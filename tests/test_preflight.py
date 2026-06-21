"""Tests for preflight validation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from bioaed.preflight import (
    _parse_sweep_combos,
    _resolve_dataset_config,
    _resolve_model_kwargs,
    _resolve_training_config,
    check_dataset_exists,
    check_hardware,
    check_model_smoke,
)


class TestParseSweepCombos:
    """Verify sweep parameter extraction from ablation profiles."""

    def test_quick_pilot_single_combo(self) -> None:
        models, datasets = _parse_sweep_combos("quick_pilot")
        assert models == ["inceptiontime"]
        assert datasets == ["aswine"]

    def test_extended_pilot_all_combos(self) -> None:
        models, datasets = _parse_sweep_combos("extended_pilot")
        assert set(models) == {"inceptiontime", "ast", "audio_mamba"}
        assert set(datasets) == {"aswine", "anuraset"}

    def test_full_matrix_all_combos(self) -> None:
        models, datasets = _parse_sweep_combos("full_matrix")
        assert set(models) == {"inceptiontime", "ast", "audio_mamba"}
        assert set(datasets) == {"aswine", "anuraset"}


class TestResolveConfigs:
    """Config resolution and interpolation."""

    def test_dataset_config_has_required_fields(self) -> None:
        cfg = _resolve_dataset_config("aswine")
        for field in ("n_mels", "time_frames", "num_classes", "sample_rate"):
            assert field in cfg, f"Missing field: {field}"

    def test_model_kwargs_resolve_interpolations(self) -> None:
        ds_cfg = _resolve_dataset_config("aswine")
        kwargs = _resolve_model_kwargs("inceptiontime", ds_cfg)
        assert kwargs["num_classes"] == ds_cfg["num_classes"]
        assert kwargs["in_channels"] == ds_cfg["n_mels"]
        assert "_target_" not in kwargs

    def test_ast_kwargs_resolve_input_dims(self) -> None:
        ds_cfg = _resolve_dataset_config("anuraset")
        kwargs = _resolve_model_kwargs("ast", ds_cfg)
        assert kwargs["input_fdim"] == ds_cfg["n_mels"]
        assert kwargs["input_tdim"] == ds_cfg["time_frames"]

    def test_training_config_profile_overrides(self) -> None:
        base = _resolve_training_config("quick_pilot")
        assert base["max_epochs"] == 30  # overridden from 100
        assert base["patience"] == 7  # overridden from 10

    def test_training_config_full_uses_defaults(self) -> None:
        base = _resolve_training_config("full_matrix")
        assert base["max_epochs"] == 100  # no override


class TestCheckDatasetExists:
    """Dataset directory validation."""

    @pytest.mark.skipif(
        not Path("data/aswine").exists(), reason="aSwine data not available"
    )
    def test_existing_dataset_no_errors(self) -> None:
        cfg = _resolve_dataset_config("aswine")
        errors = check_dataset_exists("aswine", cfg)
        assert errors == []

    def test_missing_root_dir_reports_error(self) -> None:
        cfg = {"root_dir": "/nonexistent/path"}
        errors = check_dataset_exists("fake", cfg)
        assert len(errors) == 1
        assert "not found" in errors[0]


class TestCheckModelSmoke:
    """Model instantiation and forward/backward smoke tests."""

    def test_inceptiontime_passes(self) -> None:
        ds_cfg = _resolve_dataset_config("aswine")
        kwargs = _resolve_model_kwargs("inceptiontime", ds_cfg)
        errors = check_model_smoke("inceptiontime", kwargs, ds_cfg)
        assert errors == []

    def test_bad_model_key_fails(self) -> None:
        ds_cfg = _resolve_dataset_config("aswine")
        errors = check_model_smoke("nonexistent_model", {}, ds_cfg)
        assert len(errors) == 1
        assert "instantiation failed" in errors[0].lower() or "unknown model" in errors[0].lower()

    def test_wrong_num_classes_detected(self) -> None:
        ds_cfg = _resolve_dataset_config("aswine")
        kwargs = _resolve_model_kwargs("inceptiontime", ds_cfg)
        # Mismatch: model produces 7 classes but we claim 999
        ds_cfg_modified = {**ds_cfg, "num_classes": 999}
        errors = check_model_smoke("inceptiontime", kwargs, ds_cfg_modified)
        assert any("shape" in e.lower() for e in errors)


class TestCheckHardware:
    """Hardware checks."""

    def test_standard_precision_ok(self) -> None:
        errors = check_hardware({"precision": "32-true"})
        assert errors == []

    def test_bf16_on_cpu_ok(self) -> None:
        with patch("bioaed.preflight.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            errors = check_hardware({"precision": "bf16-mixed"})
            assert errors == []
