"""Stage 2 - SAM2Matting turns the coarse hero mask into an alpha matte.

Resolution is the thing to understand here. The model resizes any input to a 1024x1024
square (aspect ratio destroyed) and its progressive alpha chain ends at 512x512:

    features_16 (64px) -> alpha1 @128 -> alpha2 @256 -> alpha3 @512

`predict()` then bilinearly upsamples that 512x512 grid to the original resolution. So the
matte's real detail budget is 512x512 *for whatever the model was shown*. Feed it a whole
1152x864 photo and a car occupying a quarter of the frame gets ~387x238 alpha pixels -
below the size of a wheel-arch gap, so the matte collapses into a silhouette.

Hence the crop modes. `tiles` walks square tiles across the hero's silhouette at native
resolution, so each tile spends its full 512x512 budget on ~640x640 real pixels. Interior
and exterior tiles are filled analytically and never hit the GPU, so only boundary tiles
cost a forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VARIANTS = {
    "tiny": {
        "checkpoint": "SAM2Matting-SAM2.1Tiny.pt",
        "config": "configs/sam2matting-sam2.1tiny.yaml",
    },
    "base+": {
        "checkpoint": "SAM2Matting-SAM2.1Base+.pt",
        "config": "configs/sam2matting-sam2.1base+.yaml",
    },
}

# The model's internal alpha resolution. Tiles larger than this are upsampled guesses.
ALPHA_GRID = 512


def _tile_positions(start: int, end: int, tile: int, step: int, limit: int) -> List[int]:
    """Tile origins covering [start, end), clamped so every tile fits inside [0, limit)."""
    span = end - start
    if span <= tile:
        origin = start - (tile - span) // 2
        return [int(np.clip(origin, 0, max(0, limit - tile)))]

    positions = list(range(start, end - tile + 1, step))
    if not positions:
        positions = [start]
    if positions[-1] + tile < end:
        positions.append(end - tile)
    return [int(np.clip(p, 0, max(0, limit - tile))) for p in positions]


def _feather(size: int, ramp: int) -> np.ndarray:
    """1-D linear ramp window, flat in the middle, so overlapping tiles blend seamlessly."""
    w = np.ones(size, dtype=np.float32)
    ramp = max(1, min(ramp, size // 2))
    edge = np.linspace(0.0, 1.0, ramp + 2, dtype=np.float32)[1:-1]
    w[:ramp] = edge
    w[-ramp:] = edge[::-1]
    return w


class MattingRefiner:
    def __init__(self, repo_dir: Path, checkpoints_dir: Path, variant: str, runtime):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")

        repo_dir = Path(repo_dir).resolve()
        if not (repo_dir / "sam2").is_dir():
            raise FileNotFoundError(
                f"SAM2Matting repo not found at {repo_dir} (expected a 'sam2' package inside). "
                "Run setup.ps1 or pass --repo."
            )
        # Importing `sam2` registers its hydra config module, which is how the
        # 'configs/...' paths below resolve. No chdir required.
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        from sam2.build_sam import build_sam2matting
        from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

        spec = VARIANTS[variant]
        ckpt = Path(checkpoints_dir).resolve() / spec["checkpoint"]
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"checkpoint missing: {ckpt}\n"
                f"Download it from https://huggingface.co/FudanCVL/SAM2Matting/"
                f"resolve/main/checkpoints/{spec['checkpoint']}"
            )

        model = build_sam2matting(spec["config"], str(ckpt), device=runtime.device)
        self.predictor = SAM2MattingImagePredictor(model)
        self.runtime = runtime
        self.variant = variant
        self.last_forward_passes = 0

    # ---------------------------------------------------------------- one forward pass

    @torch.inference_mode()
    def _matte_patch(self, patch: Image.Image, patch_mask: np.ndarray) -> np.ndarray:
        """Run the model on one image patch. Prompt prep mirrors the upstream demo."""
        mask_u8 = patch_mask.astype(np.uint8) * 255

        with self.runtime.autocast():
            img = self.predictor.set_image(patch)

            raw_mask = (torch.from_numpy(mask_u8) / 255) > 0
            mask_input = (torch.from_numpy(mask_u8) > 0).float() * 20 - 10
            mask_input = mask_input.unsqueeze(0).unsqueeze(0)
            mask_input = F.interpolate(
                mask_input, size=(256, 256), mode="bilinear", align_corners=False
            )

            _, alpha, _ = self.predictor.predict(
                img=img,
                raw_mask=raw_mask,
                mask_input=mask_input,
                multimask_output=False,
            )

        self.last_forward_passes += 1
        return np.clip(np.asarray(alpha).squeeze(), 0.0, 1.0)

    # ---------------------------------------------------------------- public entry

    def refine(
        self,
        image: Image.Image,
        binary_mask: np.ndarray,
        *,
        mode: str = "tiles",
        tile: int = 640,
        overlap: int = 160,
        pad_frac: float = 0.06,
    ) -> np.ndarray:
        """Coarse boolean mask -> float alpha in [0, 1] at the image's native resolution."""
        self.last_forward_passes = 0

        if mode == "full":
            return self._matte_patch(image, binary_mask)
        if mode == "hero":
            return self._refine_hero(image, binary_mask, pad_frac)
        if mode == "tiles":
            return self._refine_tiled(image, binary_mask, tile, overlap, pad_frac)
        raise ValueError(f"unknown crop mode {mode!r}; use full|hero|tiles")

    def _hero_region(self, mask: np.ndarray, pad_frac: float) -> Tuple[int, int, int, int]:
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        px = int((x1 - x0) * pad_frac)
        py = int((y1 - y0) * pad_frac)
        return (
            max(0, x0 - px),
            max(0, y0 - py),
            min(w, x1 + px),
            min(h, y1 + py),
        )

    def _refine_hero(self, image, mask, pad_frac):
        """One forward pass over the padded hero bbox instead of the whole frame."""
        x0, y0, x1, y1 = self._hero_region(mask, pad_frac)
        patch = image.crop((x0, y0, x1, y1))
        alpha_patch = self._matte_patch(patch, mask[y0:y1, x0:x1])

        alpha = np.zeros(mask.shape, dtype=np.float32)
        alpha[y0:y1, x0:x1] = alpha_patch
        return alpha

    def _refine_tiled(self, image, mask, tile, overlap, pad_frac):
        """Square tiles across the hero region at native resolution, feather-blended.

        A tile whose prompt mask is entirely inside (or entirely outside) the subject has
        no boundary to matte, so it is filled analytically - typically only a third of the
        tiles touch the silhouette and need the model.
        """
        h, w = mask.shape
        tile = int(min(tile, h, w))
        step = max(1, tile - overlap)
        x0, y0, x1, y1 = self._hero_region(mask, pad_frac)

        accum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        window = np.outer(_feather(tile, overlap // 2), _feather(tile, overlap // 2))

        for ty in _tile_positions(y0, y1, tile, step, h):
            for tx in _tile_positions(x0, x1, tile, step, w):
                sub_mask = mask[ty : ty + tile, tx : tx + tile]
                if sub_mask.shape != (tile, tile):
                    continue

                if not sub_mask.any():
                    alpha_tile = np.zeros((tile, tile), dtype=np.float32)
                elif sub_mask.all():
                    alpha_tile = np.ones((tile, tile), dtype=np.float32)
                else:
                    patch = image.crop((tx, ty, tx + tile, ty + tile))
                    alpha_tile = self._matte_patch(patch, sub_mask)

                accum[ty : ty + tile, tx : tx + tile] += alpha_tile * window
                weight[ty : ty + tile, tx : tx + tile] += window

        covered = weight > 1e-6
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[covered] = accum[covered] / weight[covered]
        # Anything outside the tiled region is background by construction.
        return np.clip(alpha, 0.0, 1.0)
