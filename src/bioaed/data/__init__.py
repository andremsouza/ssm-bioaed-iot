"""Data module: dataset adapters, data loaders, and data module.

Adapters:
    - :mod:`~bioaed.data.anuraset` — AnuraSet adapter (42-class anuran PAM, 3 s clips, 22 050 Hz)
    - :mod:`~bioaed.data.aswine` — aSwine adapter (7-class swine barn audio, 1 s clips, 16 000 Hz)

Factory:
    - :class:`~bioaed.data.datamodule.BioacousticDataModule` — Lightning-compatible data module
      that selects the correct adapter, splits, and wraps datasets in configured DataLoaders.
"""
