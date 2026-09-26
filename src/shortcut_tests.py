"""Intervention tests: change one property of the input, watch the output.

Grad-CAM shows where evidence is located, but at 7x7 it cannot say whether
a region matters because of the leaf or because of what surrounds it.
These tests answer the shortcut question causally: if the model relied on
a property, removing that property changes its predictions.

  background  White background -> the gray paper tone of the other sources
              (all white-background images). Tests "white -> Healthy".
  resolution  Studio 3000px -> 1024px + JPEG quality 75, like the resized_
              sources. Tests "sharp, large image -> Healthy".
  grayscale   Remove color (all images). Tests how much the decision
              depends on color (yellowing, browning).
  unseen      Other rose diseases the model never saw (optional). Tests
              whether it learned "black spot" or just "unhealthy".

Usage:
    python -m src.shortcut_tests --checkpoint <run>/best.pt \
        --data-dir "~/Desktop/final dataset/Dataset/Rose" --unseen
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.checkpoint import load_checkpoint
from src.dataset import (CLASS_NAMES, DEFAULT_SPLIT_FILE, POSITIVE_CLASS,
                         Sample, build_transforms, read_split)
from src.engine import get_device
from src.gradcam import WHITE_IMAGE, WHITE_LEVEL, load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

Edit = Callable[[Image.Image], Image.Image]


def paper_tone(samples: list[Sample], size: int) -> np.ndarray:
    """Median RGB of the top/bottom borders of gray-paper photos.

    Only non-studio photos whose border is both unsaturated (gray) and
    darker than the white threshold count as gray paper.
    """
    tones = []
    for sample in samples:
        if sample.source == "studio":
            continue
        rgb = np.asarray(load_image(sample.path, size)).astype(np.float32)
        band = size // 12
        border = np.concatenate([rgb[:band].reshape(-1, 3),
                                 rgb[-band:].reshape(-1, 3)])
        saturation = border.max(axis=1) - border.min(axis=1)
        if (np.median(saturation) < 15
                and np.median(border.mean(axis=1)) < WHITE_LEVEL):
            tones.append(np.median(border, axis=0))
    return np.median(np.array(tones), axis=0).round().astype(np.uint8)


def replace_background(tone: np.ndarray) -> Edit:
    """Paint every near-white pixel with `tone`."""
    def edit(image: Image.Image) -> Image.Image:
        rgb = np.asarray(image).copy()
        rgb[np.asarray(image.convert("L")) >= WHITE_LEVEL] = tone
        return Image.fromarray(rgb)
    return edit


def like_resized_source(image: Image.Image) -> Image.Image:
    """Downscale to 1024px and re-encode as JPEG (quality 75)."""
    small = image.resize((1024, 1024), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=75)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def to_grayscale(image: Image.Image) -> Image.Image:
    """Keep brightness, drop color (still 3 channels for the model)."""
    return image.convert("L").convert("RGB")


@torch.no_grad()
def p_black_spot(model: nn.Module, transform: transforms.Compose,
                 image: Image.Image, device: torch.device) -> float:
    """Softmax probability of Black_Spot for one image."""
    logits = model(transform(image)[None].to(device))
    return float(logits.softmax(dim=1)[0, POSITIVE_CLASS])


def run_test(
    name: str,
    samples: list[Sample],
    edit: Edit,
    load: Callable[[Sample], Image.Image],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> dict[str, dict]:
    """Predict each sample before and after `edit`, grouped by source."""
    pairs: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for sample in samples:
        image = load(sample)
        before = p_black_spot(model, transform, image, device)
        after = p_black_spot(model, transform, edit(image), device)
        key = f"{sample.source} (true {CLASS_NAMES[sample.label][5:]})"
        pairs[key].append((before, after, sample.label))

    results = {}
    print(f"\n{name}")
    for key, values in sorted(pairs.items()):
        before, after, labels = (np.array(v) for v in zip(*values))
        truth = labels == POSITIVE_CLASS
        results[key] = {
            "images": len(values),
            "accuracy_before": float(((before >= 0.5) == truth).mean()),
            "accuracy_after": float(((after >= 0.5) == truth).mean()),
            "flipped": int(((before >= 0.5) != (after >= 0.5)).sum()),
            "mean_p_black_spot_before": float(before.mean()),
            "mean_p_black_spot_after": float(after.mean()),
        }
        r = results[key]
        print(f"  {key:<28} n={r['images']:>3}  accuracy "
              f"{r['accuracy_before']:.3f} -> {r['accuracy_after']:.3f}  "
              f"flipped {r['flipped']:>3}  mean p(BS) "
              f"{r['mean_p_black_spot_before']:.3f} -> "
              f"{r['mean_p_black_spot_after']:.3f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut tests.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--unseen", action="store_true",
                        help="Also predict every other class folder in "
                             "--data-dir (diseases the model never saw).")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/shortcut_tests.json"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    _, transform = build_transforms(args.image_size)
    splits = read_split(args.split_file, data_dir)
    samples = [s for split in args.splits for s in splits[split]]

    def at_model_size(sample: Sample) -> Image.Image:
        return load_image(sample.path, args.image_size)

    def full_size(sample: Sample) -> Image.Image:
        with Image.open(sample.path) as img:
            return img.convert("RGB")

    white = [s for s in samples if (np.asarray(
        at_model_size(s).convert("L")) >= WHITE_LEVEL).mean() >= WHITE_IMAGE]
    tone = paper_tone(samples, args.image_size)
    print(f"checkpoint epoch {checkpoint['epoch']}, {len(samples)} images, "
          f"{len(white)} with a white background; paper tone {tone.tolist()}")

    report = {"checkpoint": str(args.checkpoint), "splits": args.splits,
              "paper_tone": tone.tolist()}
    report["background"] = run_test(
        "background: white -> paper gray", white, replace_background(tone),
        at_model_size, model, transform, device)
    report["resolution"] = run_test(
        "resolution: studio 3000px -> 1024px JPEG",
        [s for s in samples if s.source == "studio"], like_resized_source,
        full_size, model, transform, device)
    report["grayscale"] = run_test(
        "grayscale: color removed", samples, to_grayscale, at_model_size,
        model, transform, device)

    if args.unseen:
        report["unseen"] = {}
        print("\nunseen rose diseases: share predicted Black_Spot")
        for folder in sorted(p for p in data_dir.iterdir()
                             if p.is_dir() and p.name not in CLASS_NAMES):
            files = [f for f in sorted(folder.iterdir())
                     if f.suffix.lower() in IMAGE_EXTENSIONS]
            probs = np.array([p_black_spot(
                model, transform, load_image(f, args.image_size), device)
                for f in files])
            share = float((probs >= 0.5).mean())
            report["unseen"][folder.name] = {"images": len(files),
                                             "predicted_black_spot": share}
            print(f"  {folder.name:<28} n={len(files):>4}  {share:6.1%}")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
