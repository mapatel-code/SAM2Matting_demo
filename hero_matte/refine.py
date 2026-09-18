"""Post-processing on the alpha matte.

Two operations, both cheap and both optional:

  * `guided_refine` - a colour guided filter (He et al.) using the full-resolution photo
    as guide. The model's alpha arrives on a 512x512 grid and is upsampled, so its edges
    sit a pixel or two off the real ones; the guided filter snaps them back to image
    gradients. It sharpens what is there - it cannot invent detail the model never saw,
    which is why tiling matters more than this does.
  * `cleanup` - drop alpha blobs disconnected from the subject, e.g. a patch of floor the
    model decided was car.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _box(img: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.boxFilter(img, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_refine(rgb: np.ndarray, alpha: np.ndarray, radius: int = 4, eps: float = 1e-4) -> np.ndarray:
    """Colour guided filter: alpha in [0,1] snapped to the edges of the full-res image.

    Colour rather than greyscale because a silver car against a grey floor is nearly
    isoluminant - the edge only exists in chroma.
    """
    I = rgb.astype(np.float32) / 255.0
    p = alpha.astype(np.float32)

    mean_I = _box(I, radius)                       # H,W,3
    mean_p = _box(p, radius)                       # H,W
    mean_Ip = _box(I * p[..., None], radius)       # H,W,3
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]

    # Symmetric 3x3 covariance of the guide, per pixel.
    var = _box(np.stack(
        [I[..., 0] * I[..., 0], I[..., 0] * I[..., 1], I[..., 0] * I[..., 2],
         I[..., 1] * I[..., 1], I[..., 1] * I[..., 2], I[..., 2] * I[..., 2]], axis=-1),
        radius,
    )
    rr = var[..., 0] - mean_I[..., 0] * mean_I[..., 0] + eps
    rg = var[..., 1] - mean_I[..., 0] * mean_I[..., 1]
    rb = var[..., 2] - mean_I[..., 0] * mean_I[..., 2]
    gg = var[..., 3] - mean_I[..., 1] * mean_I[..., 1] + eps
    gb = var[..., 4] - mean_I[..., 1] * mean_I[..., 2]
    bb = var[..., 5] - mean_I[..., 2] * mean_I[..., 2] + eps

    # Analytic inverse of the symmetric 3x3.
    inv_rr = gg * bb - gb * gb
    inv_rg = gb * rb - rg * bb
    inv_rb = rg * gb - gg * rb
    inv_gg = rr * bb - rb * rb
    inv_gb = rb * rg - rr * gb
    inv_bb = rr * gg - rg * rg
    det = rr * inv_rr + rg * inv_rg + rb * inv_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    a = np.stack(
        [
            (inv_rr * cov_Ip[..., 0] + inv_rg * cov_Ip[..., 1] + inv_rb * cov_Ip[..., 2]) / det,
            (inv_rg * cov_Ip[..., 0] + inv_gg * cov_Ip[..., 1] + inv_gb * cov_Ip[..., 2]) / det,
            (inv_rb * cov_Ip[..., 0] + inv_gb * cov_Ip[..., 1] + inv_bb * cov_Ip[..., 2]) / det,
        ],
        axis=-1,
    )
    b = mean_p - np.sum(a * mean_I, axis=-1)

    q = np.sum(_box(a, radius) * I, axis=-1) + _box(b, radius)
    return np.clip(q, 0.0, 1.0)


def cleanup(alpha: np.ndarray, min_blob_frac: float = 0.01, threshold: float = 0.5) -> np.ndarray:
    """Keep the subject's connected component; erase detached alpha islands."""
    solid = alpha >= threshold
    if not solid.any():
        return alpha

    labelled, count = ndimage.label(solid)
    if count <= 1:
        return alpha

    sizes = ndimage.sum(solid, labelled, range(1, count + 1))
    keep = int(np.argmax(sizes)) + 1
    cutoff = sizes[keep - 1] * min_blob_frac

    out = alpha.copy()
    for idx in range(1, count + 1):
        if idx != keep and sizes[idx - 1] < cutoff:
            out[labelled == idx] = 0.0
    return out
