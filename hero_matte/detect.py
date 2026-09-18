"""Stage 1 - find the hero car.

SAM2Matting's image path does NOT find anything: predict() dereferences `raw_mask`
unconditionally and goes straight to the alpha heads. So selecting *which* car is the
subject is entirely our problem. We use torchvision's Mask R-CNN (ships with torch, BSD
licence) for instance masks, then score:

    score = area_frac^A * completeness^C * centrality^N * det_score^D

`completeness` is the one that encodes "FULL car": any mask pixels sitting on the image
border mean the vehicle is cropped, and the penalty decays exponentially in how much of
its bounding-box perimeter touches an edge. A cropped foreground car can easily beat the
staged subject on raw area alone, which is exactly the failure this term prevents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import torch
from scipy import ndimage

# COCO class ids. SUVs and pickups are routinely labelled `truck`, and the odd van comes
# back as `bus`, so all three count as candidate vehicles.
VEHICLE_CLASSES = {3: "car", 6: "bus", 8: "truck"}

# Scoring exponents. completeness dominates, because "full car" was the explicit rule.
W_AREA = 1.0
W_COMPLETENESS = 1.5
W_CENTRALITY = 0.5
W_DETECTION = 0.5

# Fraction of bbox perimeter that may touch the image edge before completeness decays to 1/e.
EDGE_TOLERANCE = 0.05


@dataclass
class Candidate:
    label: str
    det_score: float
    box: List[float]
    area_frac: float
    completeness: float
    centrality: float
    score: float
    mask: np.ndarray = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask", None)
        d["box"] = [round(v, 1) for v in self.box]
        for k in ("det_score", "area_frac", "completeness", "centrality", "score"):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class Selection:
    hero: Candidate | None
    candidates: List[Candidate]
    flags: List[str]

    @property
    def ok(self) -> bool:
        return self.hero is not None


class VehicleDetector:
    """Lazy-loaded Mask R-CNN. Weights (~170 MB) are cached by torch.hub on first run."""

    def __init__(self, runtime, score_threshold: float = 0.5):
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(runtime.device)
        self.preprocess = weights.transforms()
        self.runtime = runtime
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray) -> List[dict]:
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        batch = [self.preprocess(tensor).to(self.runtime.device)]
        out = self.model(batch)[0]

        results = []
        for box, label, score, mask in zip(
            out["boxes"], out["labels"], out["scores"], out["masks"]
        ):
            label_id = int(label)
            if label_id not in VEHICLE_CLASSES or float(score) < self.score_threshold:
                continue
            results.append(
                {
                    "label": VEHICLE_CLASSES[label_id],
                    "score": float(score),
                    "box": [float(v) for v in box.cpu()],
                    # masks come back as (1, H, W) soft probabilities
                    "mask": mask[0].float().cpu().numpy(),
                }
            )
        return results


def _completeness(mask: np.ndarray, box: List[float]) -> float:
    """1.0 for a vehicle fully inside frame, decaying as its mask runs off the edge."""
    edge_px = (
        int(mask[0, :].sum())
        + int(mask[-1, :].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
    )
    if edge_px == 0:
        return 1.0
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    perimeter = 2.0 * (bw + bh)
    return float(math.exp(-edge_px / max(1.0, EDGE_TOLERANCE * perimeter)))


def _centrality(mask: np.ndarray) -> float:
    """1.0 at frame centre, 0.0 at a corner. Staged hero cars sit near the middle."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    dist = math.hypot(cx - w / 2.0, cy - h / 2.0)
    return float(max(0.0, 1.0 - dist / math.hypot(w / 2.0, h / 2.0)))


def clean_mask(soft_mask: np.ndarray, threshold: float = 0.5, dilate_px: int = 0) -> np.ndarray:
    """Soft Mask R-CNN output -> one solid binary blob suitable as a matting prompt.

    Largest connected component only (drops detached speckle), holes filled (windows must
    be inside the prompt or the matting head never considers them), optional dilation to
    reclaim thin structures Mask R-CNN habitually clips: aerials, wing mirrors, spoilers.
    """
    binary = soft_mask >= threshold
    if not binary.any():
        return binary

    labelled, count = ndimage.label(binary)
    if count > 1:
        sizes = ndimage.sum(binary, labelled, range(1, count + 1))
        binary = labelled == (int(np.argmax(sizes)) + 1)

    binary = ndimage.binary_fill_holes(binary)
    if dilate_px > 0:
        binary = ndimage.binary_dilation(binary, iterations=int(dilate_px))
    return binary


def select_hero(
    detections: List[dict],
    image_shape: tuple,
    *,
    min_area_frac: float = 0.05,
    min_completeness: float = 0.35,
    ambiguity_ratio: float = 0.85,
    mask_threshold: float = 0.5,
    dilate_px: int = 0,
) -> Selection:
    """Score every vehicle, return the winner plus any quality flags.

    Flags never change the outcome; they mark images a human should eyeball. Silent wrong
    -car selection is the failure mode that poisons a batch, so we would rather hand back
    a flagged result than a confident wrong one.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)
    flags: List[str] = []
    candidates: List[Candidate] = []

    for det in detections:
        mask = clean_mask(det["mask"], mask_threshold, dilate_px)
        if not mask.any():
            continue
        area_frac = float(mask.sum()) / image_area
        completeness = _completeness(mask, det["box"])
        centrality = _centrality(mask)
        score = (
            (area_frac ** W_AREA)
            * (completeness ** W_COMPLETENESS)
            * (centrality ** W_CENTRALITY)
            * (det["score"] ** W_DETECTION)
        )
        candidates.append(
            Candidate(
                label=det["label"],
                det_score=det["score"],
                box=det["box"],
                area_frac=area_frac,
                completeness=completeness,
                centrality=centrality,
                score=score,
                mask=mask,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        return Selection(hero=None, candidates=[], flags=["no_vehicle_detected"])

    hero = candidates[0]
    if hero.area_frac < min_area_frac:
        flags.append("hero_too_small")
    if hero.completeness < min_completeness:
        flags.append("hero_may_be_cropped")
    if len(candidates) > 1 and candidates[1].score >= ambiguity_ratio * hero.score:
        flags.append("ambiguous_hero")

    return Selection(hero=hero, candidates=candidates, flags=flags)
