"""Model registry for bioacoustic audio event detection architectures."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from bioaed.models.inceptiontime import InceptionTime

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "inceptiontime": InceptionTime,
}

# Conditionally register models with optional dependencies
try:
    from bioaed.models.ast_model import AudioSpectrogramTransformer

    MODEL_REGISTRY["ast"] = AudioSpectrogramTransformer
except ImportError:
    pass

try:
    from bioaed.models.audio_mamba import AudioMamba

    MODEL_REGISTRY["audio_mamba"] = AudioMamba
except ImportError:
    pass


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a model from the registry.

    Args:
        model_name: Key in the model registry (e.g., ``"inceptiontime"``).
        **kwargs: Model constructor keyword arguments.

    Returns:
        Instantiated ``nn.Module``.

    Raises:
        ValueError: If ``model_name`` is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}"
        raise ValueError(msg)
    return MODEL_REGISTRY[model_name](**kwargs)
