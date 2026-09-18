# Hero-car matting with SAM2Matting

Extracts the **hero car** (biggest, fully-visible, centred vehicle) from a photograph and
produces an alpha matte, an RGBA cutout, and a studio composite with a synthetic contact
shadow.

> SAM2Matting is **CC-BY-NC-SA-4.0 — non-commercial research use only.** This project is
> a personal evaluation of the model. Commercial use requires licensing from the authors.

## Why there are two models

SAM2Matting's image path does not find anything. Its `predict()` dereferences `raw_mask`
unconditionally and goes straight to the alpha heads — it is a **refiner**, and the demo
feeds it a mask PNG from disk. So the pipeline is:

```
photo ──► Mask R-CNN ──► score every vehicle ──► hero's coarse mask ──► SAM2Matting ──► alpha
          (finds cars)   area × completeness      (binary)              (refines)
                         × centrality × conf
```

Hero selection lives entirely in our code, which is where you want it: it is the part
that encodes a domain rule, not a model capability.

## Hardware

| Device | Works | Notes |
|---|---|---|
| **RTX 3060 6 GB** | yes | ~2.0–2.5 GB for Tiny, ~3.0–3.5 GB for Base+. Ampere → native bf16. |
| Any CUDA GPU ≥ 4 GB | yes | fp16 autocast on pre-Ampere. |
| CPU only | yes, slowly | fp32, no autocast. Expect minutes per image, not seconds. |
| SAM3 variant | not supported here | 3.5 GB of weights; tight on 6 GB and not needed for stills. |

`hero_matte/runtime.py` picks device and precision automatically — the repo's own scripts
hardcode `torch.autocast("cuda", bfloat16)` and crash on a CPU-only box.

## Setup

```powershell
.\setup.ps1                 # auto-detects NVIDIA, installs CUDA or CPU wheels
.\setup.ps1 -Variant both   # also grab the Base+ checkpoint for comparison
```

That creates `.venv`, installs torch 2.8.0 / torchvision 0.23.0 from the right index,
clones the upstream repo into `third_party/SAM2Matting`, and downloads checkpoints.

Two upstream landmines it works around: `requirements.txt` pins `torch-tensorrt`
(NVIDIA-only, fails to install otherwise) and the repo asks for Python 3.10 — 3.12 is
fine, 3.14 is not (no torch 2.8 wheels).

Verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run

```powershell
# your three sample images
.\.venv\Scripts\python.exe run_hero_matte.py "$env:USERPROFILE\Downloads\Original" -o output

# compare Tiny against Base+ on the same inputs
.\.venv\Scripts\python.exe run_hero_matte.py "$env:USERPROFILE\Downloads\Original" -o output --variant base+
```

Outputs land in `output/<image-stem>__<variant>/`:

| File | What it is |
|---|---|
| `pha.png` | the alpha matte — canonical, everything else derives from it |
| `cutout.png` | RGBA, glass forced opaque |
| `cutout_glassmatte.png` | RGBA, model's raw alpha (see-through windows) |
| `composite.png` | studio gradient + synthetic contact shadow |
| `composite_glassmatte.png` | same, transparent glass |
| `debug.png` | every vehicle boxed, hero in green with its score |
| `meta.json` | chosen hero, all rejected candidates, flags, timings, peak VRAM |

Plus `output/manifest.csv` for the batch, listing anything flagged for review.

**Look at `debug.png` first.** It shows you whether hero selection agreed with you before
you start judging matte quality.

## Why matte resolution needs crop modes

The model force-resizes any input to 1024×1024 (aspect ratio destroyed) and its
progressive alpha chain ends at **512×512**:

```
features_16 @64 ──► alpha1 @128 ──► alpha2 @256 ──► alpha3 @512 ──► bilinear upsample to native
```

That 512×512 budget is spent on *whatever the model was shown*. Show it a whole 1152×864
photo where the car is a quarter of the frame and the entire car gets ~387×238 alpha
pixels — finer than that (wheel spokes, the wheel-arch gap, the few-pixel gap between a
front bumper and the floor) cannot be represented, and the matte collapses to a
silhouette with the floor smeared into it.

`--crop-mode` decides how that budget is spent:

| mode | forward passes | what the 512² grid covers |
|---|---|---|
| `full` | 1 | the whole frame — fastest, blobbiest |
| `hero` | 1 | the car's bbox — ~2× more detail |
| `tiles` *(default)* | ~8–20 | one 640×640 patch each, i.e. near 1:1 with source pixels |

Tiles whose prompt mask is entirely inside or outside the subject are filled
analytically and never hit the GPU, so only boundary tiles cost anything. On a 3060 a
forward pass is ~0.1 s, so `tiles` costs a couple of seconds per image.

`--mask-dilate` defaults to **0**. Growing the prompt pushes it into exactly the tight
gaps (under the bumper, inside wheel arches) you want preserved.

## Useful flags

```
--variant tiny|base+        which checkpoint
--crop-mode tiles|hero|full where the model spends its 512x512 alpha budget
--tile N / --overlap N      tile geometry for --crop-mode tiles
--refine guided|none        colour guided filter, snaps alpha edges to the photo
--device auto|cuda|cpu      override device detection
--glass solid|matte|both    window treatment in derived images
--background gradient|white|none
--no-shadow                 skip the synthetic contact shadow
--mask-dilate N             grow the coarse mask N px before matting (default 0)
--min-completeness F        below this the hero is flagged as possibly cropped
--overwrite                 redo images that already have a meta.json
--limit N                   process only the first N images
```

## Hero selection, precisely

```
score = area_frac^1.0 · completeness^1.5 · centrality^0.5 · det_confidence^0.5
```

`completeness` decays exponentially in how much of the vehicle's bounding-box perimeter
touches the image border, so a cropped foreground car loses to a smaller fully-visible
one. It is weighted hardest because "full car" was the explicit rule.

Nothing is ever silently guessed. These raise flags in `meta.json` and `manifest.csv`:

- `no_vehicle_detected` — nothing produced; image skipped, never a fallback guess
- `ambiguous_hero` — runner-up scored within 15% of the winner
- `hero_may_be_cropped` — winner's completeness below threshold
- `hero_too_small` — winner covers under 5% of the frame

## Layout

```
hero_matte/
  runtime.py   device + precision selection
  detect.py    Mask R-CNN, scoring, hero selection, mask cleanup
  matting.py   SAM2Matting wrapper (mirrors the upstream demo's prompt prep)
  compose.py   glass treatment, backgrounds, contact shadow, debug overlay
  cli.py       batch driver, sidecars, manifest, resume
third_party/SAM2Matting/   upstream clone, unmodified
checkpoints/               downloaded weights
```

See `CONTEXT.md` for the domain vocabulary these modules assume.
