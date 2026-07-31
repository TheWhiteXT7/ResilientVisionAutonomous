"""Projection engine for rendering laser patterns onto PIL images."""

from PIL import Image, ImageDraw, ImageFilter

from .attack_config import AttackConfig
from .laser_pattern import LaserPattern


class ProjectionEngine:
    """Renders a LaserPattern onto a PIL image using Pillow operations.

    This engine operates strictly on PIL Image objects, performing alpha blending,
    Gaussian blur, and spot intensity scaling. The source image is never modified.
    """

    def render(
        self,
        image: Image.Image,
        pattern: LaserPattern,
        config: AttackConfig,
    ) -> Image.Image:
        """Render pattern onto image and return a new image instance.

        Args:
            image: Source PIL Image.
            pattern: LaserPattern containing laser spots to render.
            config: AttackConfig containing render properties (blur, intensity, alpha).

        Returns:
            A new PIL Image instance with the rendered attack pattern.

        Raises:
            TypeError: If arguments are not of the expected types.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image instance, got {type(image).__name__}.")
        if not isinstance(pattern, LaserPattern):
            raise TypeError(f"pattern must be a LaserPattern instance, got {type(pattern).__name__}.")
        if not isinstance(config, AttackConfig):
            raise TypeError(f"config must be an AttackConfig instance, got {type(config).__name__}.")

        width, height = image.size

        # Create RGBA copy of base image to avoid mutating original
        base_rgba = image.convert("RGBA")

        # Create transparent overlay layer for laser pattern
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for spot in pattern:
            # Effective alpha = spot.intensity * config.intensity * config.alpha
            effective_alpha = float(spot.intensity) * float(config.intensity) * float(config.alpha)
            alpha_byte = int(round(max(0.0, min(1.0, effective_alpha)) * 255.0))

            r, g, b = spot.color
            fill_color = (r, g, b, alpha_byte)

            rad = float(spot.radius)
            bbox = [
                float(spot.x) - rad,
                float(spot.y) - rad,
                float(spot.x) + rad,
                float(spot.y) + rad,
            ]
            draw.ellipse(bbox, fill=fill_color)

        # Apply Gaussian blur if blur_radius > 0
        if config.blur_radius > 0:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=float(config.blur_radius)))

        # Alpha composite overlay onto base image
        attacked_rgba = Image.alpha_composite(base_rgba, overlay)

        # Convert back to original mode if L or RGB
        if image.mode in ("RGB", "L"):
            return attacked_rgba.convert(image.mode)
        return attacked_rgba

    def __call__(
        self,
        image: Image.Image,
        pattern: LaserPattern,
        config: AttackConfig,
    ) -> Image.Image:
        """Shortcut for calling render directly on the instance."""
        return self.render(image, pattern, config)
