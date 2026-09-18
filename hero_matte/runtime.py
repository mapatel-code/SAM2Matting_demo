"""Device / dtype selection.

The repo's demo scripts hardcode `torch.autocast("cuda", bfloat16)`, which dies on a
machine without CUDA. This module picks the right context for whatever box we're on:

  * CUDA + bf16 support (Ampere and newer, e.g. RTX 3060) -> autocast bf16
  * CUDA without bf16 (Pascal/older)                      -> autocast fp16
  * CPU                                                   -> plain fp32, no autocast
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Runtime:
    device: str
    dtype: torch.dtype | None
    name: str

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def autocast(self):
        """Context manager for the forward pass."""
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.split(":")[0], dtype=self.dtype)

    def describe(self) -> str:
        precision = str(self.dtype).replace("torch.", "") if self.dtype else "float32"
        return f"{self.name} [{self.device}, {precision}]"


def resolve(requested: str = "auto") -> Runtime:
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"unknown device {requested!r}; use auto|cuda|cpu")

    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return Runtime(device="cpu", dtype=None, name="CPU")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return Runtime(device="cuda", dtype=dtype, name=name)


def vram_report() -> str:
    if not torch.cuda.is_available():
        return "n/a (cpu)"
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{peak:.2f} GB peak / {total:.1f} GB total"
