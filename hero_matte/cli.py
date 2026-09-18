"""Folder in, folder out. One JSON sidecar per image, one manifest for the batch.

The sidecar is what makes this scale to batch work later without a rewrite: chosen box,
every rejected candidate and why, timings, model version. Re-running skips images that
already have a sidecar unless --overwrite is passed, so an interrupted batch resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from . import __version__, compose, runtime as rt
from .detect import VehicleDetector, select_hero
from .matting import MattingRefiner, VARIANTS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hero-matte",
        description="Extract the hero car from photos using SAM2Matting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="image file or directory of images")
    p.add_argument("-o", "--output", type=Path, default=Path("output"))

    g = p.add_argument_group("model")
    g.add_argument("--variant", choices=sorted(VARIANTS), default="tiny")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    g.add_argument("--repo", type=Path, default=Path("third_party/SAM2Matting"))
    g.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))

    g = p.add_argument_group("hero selection")
    g.add_argument("--det-threshold", type=float, default=0.5,
                   help="minimum Mask R-CNN confidence for a vehicle to be considered")
    g.add_argument("--min-area", type=float, default=0.05,
                   help="flag if the hero covers less than this fraction of the frame")
    g.add_argument("--min-completeness", type=float, default=0.35,
                   help="flag if the hero looks cropped by the frame edge")
    g.add_argument("--mask-dilate", type=int, default=3,
                   help="grow the coarse mask by N px to reclaim aerials/mirrors before matting")

    g = p.add_argument_group("output")
    g.add_argument("--glass", choices=["solid", "matte", "both"], default="both",
                   help="solid = windows forced opaque; matte = raw model alpha")
    g.add_argument("--background", choices=["gradient", "white", "none"], default="gradient")
    g.add_argument("--no-shadow", action="store_true", help="skip the synthetic contact shadow")
    g.add_argument("--no-debug", action="store_true", help="skip the detection overlay PNG")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    return p.parse_args(argv)


def collect_images(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise SystemExit(f"input not found: {target}")
    return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def process_one(path: Path, out_dir: Path, detector, refiner, args, runtime) -> dict:
    started = time.perf_counter()
    image = Image.open(path).convert("RGB")
    rgb = np.array(image)

    t0 = time.perf_counter()
    detections = detector.detect(rgb)
    selection = select_hero(
        detections,
        rgb.shape,
        min_area_frac=args.min_area,
        min_completeness=args.min_completeness,
        dilate_px=args.mask_dilate,
    )
    detect_s = time.perf_counter() - t0

    record = {
        "source": str(path),
        "pipeline_version": __version__,
        "variant": args.variant,
        "runtime": runtime.describe(),
        "image_size": [rgb.shape[1], rgb.shape[0]],
        "flags": selection.flags,
        "candidates": [c.to_dict() for c in selection.candidates],
        "hero": selection.hero.to_dict() if selection.hero else None,
        "timings_s": {"detect": round(detect_s, 2)},
        "status": "ok" if selection.ok else "skipped",
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    if not selection.ok:
        record["timings_s"]["total"] = round(time.perf_counter() - started, 2)
        (out_dir / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    t0 = time.perf_counter()
    alpha = refiner.refine(image, selection.hero.mask)
    record["timings_s"]["matting"] = round(time.perf_counter() - t0, 2)

    alpha_u8 = compose.alpha_to_u8(alpha)

    # Canonical artefact: the raw matte, untouched by any presentation choice.
    Image.fromarray(alpha_u8, mode="L").save(out_dir / "pha.png")

    wanted = ["solid", "matte"] if args.glass == "both" else [args.glass]
    for kind in wanted:
        a = compose.solidify_glass(alpha_u8) if kind == "solid" else alpha_u8
        suffix = "" if kind == "solid" else "_glassmatte"
        Image.fromarray(compose.make_cutout(rgb, a), mode="RGBA").save(
            out_dir / f"cutout{suffix}.png"
        )
        if args.background != "none":
            Image.fromarray(
                compose.composite(
                    rgb, a, background=args.background, shadow=not args.no_shadow
                )
            ).save(out_dir / f"composite{suffix}.png")

    if not args.no_debug:
        Image.fromarray(compose.debug_overlay(rgb, selection)).save(out_dir / "debug.png")

    record["timings_s"]["total"] = round(time.perf_counter() - started, 2)
    record["peak_vram"] = rt.vram_report()
    (out_dir / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    runtime = rt.resolve(args.device)
    images = collect_images(args.input)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images found in {args.input}")

    print(f"hero-matte {__version__} | {runtime.describe()} | variant={args.variant}")
    print(f"{len(images)} image(s) -> {args.output.resolve()}")

    detector = VehicleDetector(runtime, score_threshold=args.det_threshold)
    refiner = MattingRefiner(args.repo, args.checkpoints, args.variant, runtime)

    rows = []
    for i, path in enumerate(images, 1):
        out_dir = args.output / f"{path.stem}__{args.variant}"
        if (out_dir / "meta.json").exists() and not args.overwrite:
            print(f"[{i}/{len(images)}] {path.name}: skipped (already done)")
            rows.append(json.loads((out_dir / "meta.json").read_text(encoding="utf-8")))
            continue

        try:
            record = process_one(path, out_dir, detector, refiner, args, runtime)
        except Exception as exc:  # one bad image must not kill a batch
            record = {
                "source": str(path),
                "status": "error",
                "error": repr(exc),
                "flags": ["error"],
            }
            print(f"[{i}/{len(images)}] {path.name}: ERROR {exc}", file=sys.stderr)
        else:
            hero = record.get("hero")
            if hero:
                desc = (
                    f"{hero['label']} score={hero['score']:.4f} "
                    f"complete={hero['completeness']:.2f}"
                )
            else:
                desc = "NO HERO"
            flags = (" flags=" + ",".join(record["flags"])) if record["flags"] else ""
            total = record["timings_s"]["total"]
            print(f"[{i}/{len(images)}] {path.name}: {desc}{flags} ({total}s)")
        rows.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["source", "status", "hero_label", "score", "completeness", "flags", "seconds"]
        )
        for r in rows:
            hero = r.get("hero") or {}
            timings = r.get("timings_s") or {}
            writer.writerow([
                r.get("source", ""),
                r.get("status", ""),
                hero.get("label", ""),
                hero.get("score", ""),
                hero.get("completeness", ""),
                "|".join(r.get("flags", [])),
                timings.get("total", ""),
            ])

    flagged = [r for r in rows if r.get("flags")]
    print(f"\ndone. {len(rows)} processed, {len(flagged)} flagged for review.")
    for r in flagged:
        print(f"  ! {Path(r['source']).name}: {','.join(r['flags'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
