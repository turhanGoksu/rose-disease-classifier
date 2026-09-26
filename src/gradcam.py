"""Grad-CAM, written by hand with a forward hook, plus a shortcut check.

Grad-CAM asks: for one class score, which feature maps of the last conv
layer matter, and where in the image are they active?
  1. Forward pass, keeping the target layer's output A: [512, 7, 7].
  2. Backward pass from the class logit, keeping dScore/dA (same shape).
  3. Weight of map k = mean of its gradient over the 7x7 grid.
  4. CAM = ReLU(sum_k weight_k * A_k): only regions that raise the score.
  5. Upsample 7x7 -> image size and overlay.

Usage:
    python -m src.gradcam --checkpoint <run>/best.pt \
        --data-dir "~/Desktop/final dataset/Dataset/Rose"
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image
from torch import nn
from torchvision import transforms

from src.checkpoint import load_checkpoint
from src.dataset import (CLASS_NAMES, DEFAULT_SPLIT_FILE, POSITIVE_CLASS,
                         Sample, build_transforms, read_split)
from src.engine import get_device

WHITE_LEVEL = 235  # Gray value above which a pixel counts as background.
WHITE_IMAGE = 0.5  # Share of such pixels that makes an image "white".
OVERLAY_COLOR = np.array([235, 104, 52], dtype=np.float32)  # Orange #eb6834.
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"


class GradCAM:
    """Grad-CAM for one conv layer of a classifier."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # Called on every forward pass through target_layer.
        self._handle = target_layer.register_forward_hook(self._keep)

    def _keep(self, module: nn.Module, inputs: tuple, output: torch.Tensor
              ) -> None:
        self.activations = output
        # Tensor hook: autograd calls it with dScore/dOutput during backward.
        output.register_hook(self._keep_gradient)

    def _keep_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __call__(
        self, image: torch.Tensor, class_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Explain one class score for one image.

        Args:
            image: [1, 3, H, W] normalized input on the model's device.
            class_index: The class whose logit is explained.

        Returns:
            (cam, probabilities): cam is the raw [h, w] map at the target
            layer's resolution (7x7 for ResNet18 at 224px), >= 0 and not yet
            normalized; probabilities is the softmax output.
        """
        self.model.eval()  # BatchNorm running stats, no dropout.
        self.model.zero_grad(set_to_none=True)
        # Grad-CAM needs a graph. requires_grad on the input guarantees one
        # even when every backbone layer is frozen.
        image = image.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(image)
            # The logit (before softmax), as in the Grad-CAM paper.
            logits[0, class_index].backward()

        # Step 3: one weight per feature map = its average gradient.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1,512,1,1]
        # Step 4: weighted sum over maps; keep only positive evidence.
        cam = F.relu((weights * self.activations).sum(dim=1))  # [1, 7, 7]
        probabilities = logits.softmax(dim=1)[0]
        return (cam[0].detach().cpu().numpy(),
                probabilities.detach().cpu().numpy())

    def remove(self) -> None:
        """Detach the hook so the model behaves normally again."""
        self._handle.remove()


def upsample(cam: np.ndarray, size: int,
             peak: float | None = None) -> np.ndarray:
    """Bilinear upsampling to size x size, divided by `peak`.

    peak=None scales the map by its own maximum (max = 1), which shows
    where the evidence is but hides how strong it is: a class with almost
    no evidence still gets a bright spot. Pass a shared peak to compare
    two maps of the same image.
    """
    tensor = torch.from_numpy(cam)[None, None]
    big = F.interpolate(tensor, size=(size, size), mode="bilinear",
                        align_corners=False)[0, 0].numpy()
    peak = big.max() if peak is None else peak
    return np.clip(big / peak, 0, 1) if peak > 0 else big


def overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Tint the photo orange where the CAM is high (opacity = CAM value)."""
    alpha = 0.75 * cam[..., None]
    mixed = (1 - alpha) * rgb.astype(np.float32) + alpha * OVERLAY_COLOR
    return mixed.clip(0, 255).astype(np.uint8)


def background_share(gray: np.ndarray, cam: np.ndarray) -> float:
    """Share of the CAM's total mass that lies on near-white pixels."""
    total = cam.sum()
    background = gray >= WHITE_LEVEL
    return float(cam[background].sum() / total) if total > 0 else 0.0


def leaf_only_cam(gray: np.ndarray, grid: int) -> np.ndarray:
    """The CAM of a model that looked ONLY at the leaf, at CAM resolution.

    A 7x7 map cannot follow a leaf's outline, so even this ideal map
    spills onto the background once upsampled. Its background share is the
    floor a real CAM should be compared with (a uniform CAM would instead
    match the background's share of the image area).
    """
    leaf = torch.from_numpy((gray < WHITE_LEVEL).astype(np.float32))
    return F.adaptive_avg_pool2d(leaf[None, None], grid)[0, 0].numpy()


def load_image(path: Path, size: int) -> Image.Image:
    """Decode an image as RGB, resized exactly like the eval transform."""
    with Image.open(path) as img:
        img.draft("RGB", (2 * size, 2 * size))
        return img.convert("RGB").resize((size, size),
                                         Image.Resampling.BILINEAR)


def group_of(sample: Sample, correct: bool, white: bool) -> str:
    """Put one explained image into a reporting group."""
    if not correct:
        return "errors"
    if sample.source == "studio":
        return "studio_healthy"
    if sample.source == "black_spot":
        return "black_spot_white_bg" if white else "black_spot"
    return "fresh_leaf_healthy"


def save_grid(rows: list[dict], image_size: int, title: str,
              output_path: Path) -> None:
    """One row per image: photo | Healthy CAM | Black_Spot CAM."""
    columns = ["photo"] + [f"Grad-CAM: {name.replace('Rose_', '')}"
                           for name in CLASS_NAMES]
    fig = Figure(figsize=(7.5, 2.7 * len(rows) + 0.6), facecolor=SURFACE)
    axes = np.atleast_2d(fig.subplots(len(rows), 3))
    for ax_row, row in zip(axes, rows):
        # One scale for both classes of this image, so the class with
        # little evidence looks faint instead of equally bright.
        peak = max(float(cam.max()) for cam in row["cams"])
        panels = [row["rgb"]] + [overlay(row["rgb"],
                                         upsample(cam, image_size, peak))
                                 for cam in row["cams"]]
        for col, (ax, panel) in enumerate(zip(ax_row, panels)):
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ax_row is axes[0]:
                ax.set_title(columns[col], color=INK, fontsize=9)
        sample = row["sample"]
        ax_row[0].set_ylabel(
            f"{sample.path.stem[-16:]}\ntrue {CLASS_NAMES[sample.label][5:]}"
            f"\npred {CLASS_NAMES[row['pred']][5:]}  "
            f"p(BS)={row['p_black_spot']:.3f}",
            color=INK, fontsize=8)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "Orange = Grad-CAM evidence for that class. Both "
             "maps of a row share one scale (the stronger map's peak).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(output_path, dpi=120, facecolor=SURFACE)


def validate_against_library(
    model: nn.Module,
    cam_model: GradCAM,
    records: list[dict],
    eval_transform: transforms.Compose,
    image_size: int,
    device: torch.device,
    count: int = 12,
) -> None:
    """Check this implementation against pytorch-grad-cam on a few images.

    The library min-max scales its maps and we only divide by the maximum,
    so maps can differ slightly in level; correlation should be ~1.
    """
    from pytorch_grad_cam import GradCAM as LibraryGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    errors = [r for r in records if r["group"] == "errors"]
    chosen = (errors + [r for r in records if r["group"] != "errors"])[:count]
    worst = 1.0
    for r in chosen:
        x = eval_transform(load_image(r["sample"].path, image_size))[None]
        x = x.to(device)
        for class_index in range(len(CLASS_NAMES)):
            ours = upsample(cam_model(x, class_index)[0], image_size)
            with LibraryGradCAM(model=model,
                                target_layers=[model.layer4]) as library:
                theirs = library(input_tensor=x, targets=[
                    ClassifierOutputTarget(class_index)])[0]
            if ours.max() == 0 and theirs.max() == 0:
                continue  # Both empty: agree, correlation undefined.
            worst = min(worst, float(np.corrcoef(ours.ravel(),
                                                 theirs.ravel())[0, 1]))
    print(f"validation vs pytorch-grad-cam on {len(chosen)} images x "
          f"{len(CLASS_NAMES)} classes: lowest correlation {worst:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM report.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/gradcam"))
    parser.add_argument("--per-group", type=int, default=5,
                        help="Images drawn per group (all errors are drawn).")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-with-library", action="store_true",
                        help="Compare with pytorch-grad-cam (pip install "
                             "grad-cam; not in requirements.txt).")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    cam_model = GradCAM(model, model.layer4)  # Last conv stage: [512, 7, 7].
    _, eval_transform = build_transforms(args.image_size)
    splits = read_split(args.split_file, args.data_dir.expanduser())

    records: list[dict] = []
    for split in args.splits:
        for sample in splits[split]:
            photo = load_image(sample.path, args.image_size)
            x = eval_transform(photo)[None].to(device)
            cams = []
            for class_index in range(len(CLASS_NAMES)):
                cam, probabilities = cam_model(x, class_index)
                cams.append(cam)
            pred = int(probabilities.argmax())
            gray = np.asarray(photo.convert("L"))
            white = float((gray >= WHITE_LEVEL).mean()) >= WHITE_IMAGE
            records.append({
                "split": split, "sample": sample, "pred": pred,
                "p_black_spot": float(probabilities[POSITIVE_CLASS]),
                "cams": cams, "white": white, "gray": gray,
                "group": group_of(sample, pred == sample.label, white),
            })
    if args.validate_with_library:
        validate_against_library(model, cam_model, records, eval_transform,
                                 args.image_size, device)
    cam_model.remove()

    # --- Where is the evidence on white-background images? ---------------
    # For the PREDICTED class, share of CAM mass on background pixels,
    # between two references: a leaf-only 7x7 CAM (floor) and a uniform CAM
    # (= background area share).
    shares: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in records:
        if r["white"]:
            raw = r["cams"][r["pred"]]
            cam = upsample(raw, args.image_size)
            floor = upsample(leaf_only_cam(r["gray"], raw.shape[0]),
                             args.image_size)
            true_name = CLASS_NAMES[r["sample"].label][5:]
            key = f"{r['sample'].source} (true {true_name})"
            shares[key].append((background_share(r["gray"], cam),
                                background_share(r["gray"], floor),
                                float((r["gray"] >= WHITE_LEVEL).mean())))
    summary = {"checkpoint": str(args.checkpoint),
               "epoch": checkpoint["epoch"], "splits": args.splits,
               "background_share": {}}
    print("Grad-CAM mass on white background (predicted class):")
    for key, values in sorted(shares.items()):
        cam_share, floor_share, area_share = np.mean(values, axis=0)
        summary["background_share"][key] = {
            "images": len(values), "cam": float(cam_share),
            "leaf_only_floor": float(floor_share),
            "uniform_cam": float(area_share)}
        print(f"  {key:<28} n={len(values):>3}  CAM {cam_share:6.1%}  "
              f"(leaf-only floor {floor_share:5.1%}, "
              f"uniform {area_share:5.1%})")

    # --- Example grids ----------------------------------------------------
    rng = random.Random(args.seed)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)
    for group, rows in sorted(by_group.items()):
        chosen = rows if group == "errors" else rng.sample(
            rows, min(args.per_group, len(rows)))
        for r in chosen:
            r["rgb"] = np.asarray(load_image(r["sample"].path,
                                             args.image_size))
        save_grid(chosen, args.image_size,
                  f"{group} ({len(chosen)} of {len(rows)})",
                  output_dir / f"{group}.png")
        summary.setdefault("groups", {})[group] = {
            "images": len(rows),
            "shown": [r["sample"].path.name for r in chosen]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote grids and summary.json to {output_dir}")


if __name__ == "__main__":
    main()
