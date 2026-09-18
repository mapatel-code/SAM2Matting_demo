"""Stage 2 - SAM2Matting turns the coarse hero mask into an alpha matte.

Mirrors the prompt preparation in the repo's `inference_image_sam2.py` exactly (the same
logit scaling and the same 256x256 resize), because that is the known-good path. What we
change: no hardcoded CUDA autocast, no hardcoded paths, and the mask comes from stage 1
rather than a PNG on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

    @torch.inference_mode()
    def refine(self, image: Image.Image, binary_mask: np.ndarray) -> np.ndarray:
        """Coarse boolean mask -> float alpha in [0, 1] at the image's native resolution."""
        mask_u8 = (binary_mask.astype(np.uint8) * 255)

        with self.runtime.autocast():
            img = self.predictor.set_image(image)

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

        return np.clip(np.asarray(alpha).squeeze(), 0.0, 1.0)
