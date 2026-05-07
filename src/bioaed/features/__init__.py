"""Feature engineering: audio transforms.

Transforms:
    - :class:`~bioaed.features.transforms.ZNormalization` — zero-mean, unit-variance
      normalisation applied per spectrogram (instance-wise).
    - :class:`~bioaed.features.transforms.SpecAugment` — frequency masking and time masking
      following the SpecAugment policy; mask widths are configurable via Hydra config.

Pipeline:
    Transforms are composed into a configurable preprocessing pipeline controlled by the
    ``augmentation`` config group. During inference, only ZNormalization is applied.
"""
