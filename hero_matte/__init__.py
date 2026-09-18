"""Hero-car matting pipeline built on FudanCVL/SAM2Matting.

Stage 1 (detect):  Mask R-CNN finds every vehicle, we score them and pick the hero.
Stage 2 (matting): SAM2Matting refines the hero's coarse mask into an alpha matte.
Stage 3 (compose): alpha -> RGBA cutout, studio composite, synthetic contact shadow.
"""

__version__ = "0.2.0"
