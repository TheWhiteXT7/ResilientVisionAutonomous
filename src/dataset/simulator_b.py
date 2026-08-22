"""
Simulator B - an INDEPENDENT laser-attack renderer for cross-simulator testing.
================================================================================
Deliberately different implementation and physics assumptions from
rolling_shutter_simulator_v7, so that a detector trained on v7 output can be
evaluated zero-shot against this renderer without sharing its artifacts:

    axis                  v7                          simulator B
    ------------------------------------------------------------------
    modulation        temporal PWM duty-cycle     pure sinusoidal bands
                      integrated per-row over     directly in image space
                      exposure time (rolling      (global shutter assumed)
                      shutter row timing)
    parameterization  frequency (Hz) + exposure   spatial period (cycles
                      time + phase                across frame) + phase
    beam coverage     hard coverage fraction +    smooth Gaussian envelope,
                      elliptical mask             optionally anisotropic
    post-processing   AE feedback iterations,     fixed exposure gain, bloom/
                      gamma range, JPEG p=0.4,    halation blur, additive
                      color temp shift            read noise, gamma 2.2
                                                  encode, NO JPEG, NO AE loop

Only generic infrastructure is shared with the v7 pipeline (background image
discovery and the source-level split policy). All rendering code here is new.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# wavelength (nm) -> per-channel gain multipliers (B, G, R order for cv2)
WAVELENGTH_GAINS = {
    405: (1.00, 0.10, 0.00),
    450: (0.95, 0.20, 0.00),
    488: (0.70, 0.55, 0.00),
    532: (0.15, 0.95, 0.05),
    650: (0.00, 0.15, 0.95),
    808: (0.00, 0.02, 0.12),
    980: (0.00, 0.00, 0.04),
    1064: (0.00, 0.00, 0.02),
}


@dataclass
class SimBParams:
    """Per-image rendering parameters, sampled by simb_builder."""
    band_cycles: float          # sinusoid periods spanning frame width (px scale)
    angle_deg: float            # stripe orientation in image plane
    amplitude: float            # peak modulation depth (fraction of bg level)
    wavelength_nm: int
    envelope_cx: float          # beam centre, fractions of width/height
    envelope_cy: float
    envelope_sigma_x: float     # Gaussian envelope sigmas as fraction of frame
    envelope_sigma_y: float
    phase: float                # stripe phase offset [0, 1) cycles
    bloom_sigma: float          # halation blur radius (px)
    noise_std: float            # additive read noise (8-bit levels)
    exposure_gain: float        # global multiplicative gain


def render_frame(bg: np.ndarray, p: SimBParams, rng: np.random.Generator):
    """Render one attacked frame. Returns (uint8 BGR image, meta dict).

    Physics story: continuous-wave laser creates smooth interference bands on
    the sensor (cover-glass etalon), attenuated by a Gaussian beam footprint;
    the camera integrates globally (no rolling-shutter row timing), so band
    contrast depends on spatial period, not exposure-vs-PWM phasing.
    """
    h, w = bg.shape[:2]
    bg_f = bg.astype(np.float32)

    # rotated distance coordinate: projection of pixel position onto the
    # direction perpendicular to the stripes
    theta = math.radians(p.angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(theta) + yy * math.sin(theta)

    # smooth sinusoidal bands (contrast ±amplitude around background level)
    period_px = max(w / max(p.band_cycles, 1e-6), 2.0)
    bands = np.sin(2.0 * math.pi * (u / period_px + p.phase))

    # anisotropic Gaussian beam envelope centred at (cx, cy)
    env = np.exp(-(((xx / w - p.envelope_cx) ** 2) / (2 * p.envelope_sigma_x ** 2)
                   + ((yy / h - p.envelope_cy) ** 2) / (2 * p.envelope_sigma_y ** 2)))

    # per-channel pattern with slight chromatic dispersion (phase lag per ch.)
    gB, gG, gR = WAVELENGTH_GAINS.get(p.wavelength_nm, (0.5, 0.5, 0.5))
    disp = 0.06 * period_px
    pat_b = bands
    pat_g = np.roll(bands, int(disp), axis=1)
    pat_r = np.roll(bands, int(2 * disp), axis=1)

    mod = np.stack([pat_b * gB, pat_g * gG, pat_r * gR], axis=-1)
    gain_map = 1.0 + p.amplitude * env[..., None] * mod

    # bloom / halation: blurred copy of the brightened signal added back
    out = bg_f * gain_map
    if p.bloom_sigma >= 0.5:
        k = int(max(3, 2 * round(2 * p.bloom_sigma) + 1))
        glow = cv2.GaussianBlur(out * env[..., None], (k, k), p.bloom_sigma)
        out = out + 0.18 * glow

    out *= p.exposure_gain

    # sensor chain: read noise -> gamma 2.2 encode -> quantise
    out += rng.normal(0.0, p.noise_std, out.shape).astype(np.float32)
    out = np.clip(out, 0, None)
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), 1.0 / 2.2)
    out_u8 = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    area_frac = float((env > 0.25).mean())
    meta = {
        "frequency": round(p.band_cycles, 3),
        "wavelength": int(p.wavelength_nm),
        "power_mw": round(p.amplitude * 100.0, 2),
        "duty_cycle": 0.5,
        "modulation": "sinusoidal",
        "coverage": area_frac,
        "angle_deg": round(p.angle_deg, 2),
        "distance_m": 0.0,
        "ellipticity": round(p.envelope_sigma_y / max(p.envelope_sigma_x, 1e-6), 3),
        "exposure_time": round(p.exposure_gain, 4),
        "ae_gain": 1.0,
        "peak_saturation": round(float((out_u8 >= 250).any(axis=2).mean()), 5),
        "attack_area_fraction": area_frac,
    }
    return out_u8, meta


def render_clean(bg: np.ndarray, rng: np.random.Generator):
    """Clean path through the SAME sensor chain (noise + gamma), no pattern."""
    out = bg.astype(np.float32) + rng.normal(0.0, 2.0, bg.shape).astype(np.float32)
    out = np.clip(out, 0, None)
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), 1.0 / 2.2)
    meta = {"frequency": 0, "wavelength": 0, "power_mw": 0, "duty_cycle": 0,
            "modulation": "none", "coverage": 0.0, "angle_deg": 0, "distance_m": 0,
            "ellipticity": 0, "exposure_time": 1.0, "ae_gain": 1.0,
            "peak_saturation": 0.0, "attack_area_fraction": 0.0}
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), meta
