# attack_engine

`attack_engine` is a standalone extraction of the repository's rolling-shutter
laser-pattern overlay. It has no dependency on TensorFlow, detectors, BDD100K,
Docker, or evaluation scripts.

## Public API

```python
from attack_engine import apply_attack

attacked_image = apply_attack(image, pattern)
```

Both arguments must be RGB `numpy.ndarray` values with `dtype=np.uint8` and
shape `(H, W, 3)`. They must have identical shapes. The return value is a new
RGB `uint8` array with the same shape.

The pixel operation is intentionally unchanged from the original repository:

```text
attacked_image = uint8(clip(float(image) + float(pattern), 0, 255))
```

## Files

- `__init__.py` exposes the sole public API, `apply_attack`.
- `attack_generator.py` validates the public RGB-image contract and delegates
  to the extracted attack operation.
- `apply_pattern.py` contains the original additive, saturating attack math and
  its original dimensional/dtype checks.
- `pattern_loader.py` optionally loads an RGB PNG/JPEG pattern into a `uint8`
  NumPy array. It is separate because the public API accepts arrays directly.

## Dependencies

- Required: `numpy`.
- Optional, only for `pattern_loader.load_rgb_pattern`: `Pillow`.

## Usage

With NumPy arrays:

```python
import numpy as np
from attack_engine import apply_attack

image = np.zeros((360, 640, 3), dtype=np.uint8)
pattern = np.full((360, 640, 3), 40, dtype=np.uint8)
attacked_image = apply_attack(image, pattern)
```

With a pattern image file:

```python
from attack_engine import apply_attack
from attack_engine.pattern_loader import load_rgb_pattern

pattern = load_rgb_pattern("pattern.png")
attacked_image = apply_attack(image, pattern)
```

Patterns must already match the target image dimensions; resizing is not part
of the original attack operation and is intentionally not performed here.
