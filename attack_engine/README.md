# Attack Engine Layer

The `attack_engine` package provides a dataset-agnostic, modular, and non-destructive rolling-shutter laser pattern attack implementation for the `ResilientVisionAutonomous` framework. It operates on standard `PIL.Image` objects using pure Pillow and Python standard library operations without dependencies on OpenCV or external ML frameworks.

---

## Architecture

The attack engine separates pattern generation (geometry/data) from projection (rendering/blending):

```
+------------------+
|    PIL.Image     |
+--------+---------+
         |
         v
+------------------+     +------------------+
| AttackConfig     | --> | PatternGenerator |
+------------------+     +--------+---------+
                                  |
                                  v
                         +------------------+
                         |   LaserPattern   |
                         +--------+---------+
                                  |
                                  v
+------------------+     +------------------+
|    PIL.Image     | --> | ProjectionEngine |
+------------------+     +--------+---------+
                                  |
                                  v
                         +------------------+
                         |  Attacked Image  |
                         +------------------+
```

### Modular Responsibilities

1. **`attack_config.py` (`AttackConfig`)**:
   - Immutable dataclass storing configuration parameters: laser color (RGB), spot intensity, opacity alpha, Gaussian blur radius, spot radius, max spots, random seed, pattern type, and output dtype.
   - Enforces parameter validation in `__post_init__`.

2. **`laser_pattern.py` (`LaserSpot`, `LaserPattern`)**:
   - `LaserSpot`: Immutable dataclass representing a single spot's position `(x, y)`, radius, intensity, and RGB color.
   - `LaserPattern`: Container data structure representing collections of spots. Implements list operations (`add_spot`, `remove_spot`, `clear`, `__len__`, `__iter__`, `__getitem__`).
   - Represents pure data; contains zero drawing or image manipulation logic.

3. **`pattern_generator.py` (`PatternGenerator`)**:
   - Dataset-agnostic pattern generator. Computes geometric spot layouts based on canvas dimensions and `AttackConfig`.
   - Supports pattern strategies: `single_spot`, `random_spots`, `horizontal_line`, `vertical_line`, `grid`, and `custom`.
   - Has zero dependency on Pillow and uses isolated random number generators for thread safety and zero global state.

4. **`projection_engine.py` (`ProjectionEngine`)**:
   - Renders a `LaserPattern` onto a `PIL.Image` using Pillow operations.
   - Performs alpha blending, spot scaling, and optional Gaussian blurring.
   - Guaranteed non-destructive execution: never modifies the original source image and always returns a new `PIL.Image`.

5. **`attack_pipeline.py` (`AttackPipeline`, `apply_attack`)**:
   - High-level orchestrator connecting `PatternGenerator` and `ProjectionEngine`.
   - Exposes top-level `apply_attack(image, pattern_type="random", **kwargs)` API returning `(attacked_image, laser_pattern)`.

6. **`utils.py`**:
   - Helpers for array/PIL conversions and array pixel operations.

---

## Usage Example

```python
from PIL import Image
from attack_engine import AttackConfig, AttackPipeline, apply_attack

# Load any PIL Image
image = Image.open("sample.png")

# Quick functional attack
attacked_img, pattern = apply_attack(
    image,
    pattern_type="random",
    laser_color=(255, 0, 0),
    intensity=0.9,
    alpha=0.8,
    blur_radius=4.0,
    spot_radius=12.0,
    max_spots=6,
    random_seed=42,
)

# Save or inspect the attacked image
attacked_img.save("attacked_sample.png")
print(f"Generated {len(pattern)} laser spots.")
```

### Advanced Pipeline Usage

```python
from attack_engine import AttackConfig, AttackPipeline, PatternGenerator, ProjectionEngine

config = AttackConfig(
    laser_color=(0, 255, 0),
    spot_radius=20.0,
    blur_radius=8.0,
    random_seed=123,
)

pipeline = AttackPipeline(config=config)

# Apply grid attack
grid_attacked, grid_pattern = pipeline.execute(image, pattern_type="grid", rows=4, cols=4)

# Apply horizontal line attack
line_attacked, line_pattern = pipeline.execute(image, pattern_type="horizontal_line")
```

---

## Extension Points

1. **Custom Pattern Layouts**:
   - Implement new methods in `PatternGenerator` (e.g. diagonal lines, spiral patterns, bounding-box targeting).
   - Pass custom `LaserSpot` collections via `PatternGenerator.custom(spots)`.

2. **Custom Projection & Blending**:
   - Extend or subclass `ProjectionEngine` to implement non-linear blending, physical refraction models, or optical distortion filters.

3. **Data Logging and Provenance**:
   - Serialize `LaserPattern` to JSON/CSV for exact reproducibility and dataset provenance tracking.
