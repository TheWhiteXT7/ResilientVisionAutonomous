"""Core pattern-overlay operation extracted from object_detection/rsa/utils.py."""

import numpy as np


def apply_pattern(base: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Overlay an additive rolling-shutter pattern onto an image.

    This preserves the original implementation's validation and pixel math:
    convert to float, add pixelwise, clip to [0, 255], then cast to uint8.
    """
    assert base.dtype == np.uint8
    assert pattern.dtype == np.uint8
    if base.ndim == 4:
        assert base.shape[1:] == pattern.shape[1:] or base.shape[1:] == pattern.shape
    elif base.ndim == 3:
        assert base.shape == pattern.shape
    else:
        raise Exception(
            "Incorrect number of dimensions for 'base' in 'apply_pattern', 3 or 4 expected"
        )
    b = base.astype(float)
    p = pattern.astype(float)
    return np.clip(b + p, 0, 255).astype(np.uint8)
