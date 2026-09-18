"""Stage 3 - turn the alpha matte into deliverables.

The matte is the canonical artefact; everything here is derived from it and is safe to
regenerate. Two decisions worth knowing about:

  * Glass. A true matte gives car windows partial alpha, so a composite shows the new
    background straight through the side glass - technically correct, visually an X-ray.
    `solidify_glass` fills enclosed interior regions back to opaque while leaving the
    outer boundary soft, so the real reflections survive and the silhouette stays matted.
  * Shadow. The captured floor shadow and reflection are deliberately excluded from the
    alpha; we synthesise a contact shadow instead so it is consistent across a whole
    batch rather than inheriting whatever the parking garage was doing that day.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

GRADIENT_TOP = (232, 234, 237)
GRADIENT_BOTTOM = (255, 255, 255)


def alpha_to_u8(alpha: np.ndarray) -> np.ndarray:
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def solidify_glass(alpha_u8: np.ndarray, threshold: int = 16) -> np.ndarray:
    """Force enclosed interior regions (windows) opaque, keep the outline soft."""
    binary = alpha_u8 >= threshold
    filled = ndimage.binary_fill_holes(binary)
    interior = filled & ~binary
    out = alpha_u8.copy()
    out[interior] = 255
    return out


def make_cutout(rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """RGBA, premultiplication left alone so the file stays editable."""
    return np.dstack([rgb, alpha_u8])


def gradient_background(h: int, w: int) -> np.ndarray:
    top = np.array(GRADIENT_TOP, dtype=np.float32)
    bottom = np.array(GRADIENT_BOTTOM, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    return (top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp).repeat(w, axis=1)


def flat_background(h: int, w: int, colour=(255, 255, 255)) -> np.ndarray:
    return np.full((h, w, 3), colour, dtype=np.float32)


def contact_shadow(
    alpha_u8: np.ndarray,
    *,
    opacity: float = 0.35,
    height_ratio: float = 0.06,
    width_ratio: float = 1.02,
) -> np.ndarray:
    """Soft ellipse under the car's footprint, derived from the bottom edge of the matte.

    Returns a float mask in [0, 1] to multiply the background down by.
    """
    h, w = alpha_u8.shape
    shadow = np.zeros((h, w), dtype=np.float32)

    binary = alpha_u8 > 128
    if not binary.any():
        return shadow

    cols = np.nonzero(binary.any(axis=0))[0]
    x0, x1 = int(cols[0]), int(cols[-1])
    car_w = max(1, x1 - x0)

    # Baseline = where the wheels actually sit. The 95th percentile of the per-column
    # lowest opaque pixel ignores a stray aerial or wing mirror hanging low in one column.
    bottoms = [int(np.nonzero(binary[:, x])[0][-1]) for x in cols]
    baseline = int(np.percentile(bottoms, 95))

    axes = (int(car_w * width_ratio / 2), max(2, int(car_w * height_ratio)))
    cv2.ellipse(
        shadow,
        center=((x0 + x1) // 2, min(h - 1, baseline)),
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=1.0,
        thickness=-1,
    )

    blur = int(car_w * 0.08) | 1  # odd kernel
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    return np.clip(shadow * opacity, 0.0, 1.0)


def composite(
    rgb: np.ndarray,
    alpha_u8: np.ndarray,
    *,
    background: str = "gradient",
    shadow: bool = True,
) -> np.ndarray:
    h, w = alpha_u8.shape
    if background == "gradient":
        bg = gradient_background(h, w)
    elif background == "white":
        bg = flat_background(h, w)
    else:
        raise ValueError(f"unknown background {background!r}")

    if shadow:
        bg = bg * (1.0 - contact_shadow(alpha_u8)[..., None])

    a = (alpha_u8.astype(np.float32) / 255.0)[..., None]
    out = rgb.astype(np.float32) * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def debug_overlay(rgb: np.ndarray, selection) -> np.ndarray:
    """All vehicle candidates boxed, hero in green with its score - for eyeballing batches."""
    canvas = rgb.copy()
    for cand in selection.candidates:
        is_hero = cand is selection.hero
        colour = (0, 200, 0) if is_hero else (180, 180, 180)
        x0, y0, x1, y1 = [int(v) for v in cand.box]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3 if is_hero else 1)
        cv2.putText(
            canvas,
            f"{cand.label} {cand.score:.4f} c={cand.completeness:.2f}",
            (x0, max(14, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2 if is_hero else 1,
            cv2.LINE_AA,
        )
    for i, flag in enumerate(selection.flags):
        cv2.putText(
            canvas, f"! {flag}", (10, 24 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 40, 40), 2, cv2.LINE_AA,
        )
    return canvas
