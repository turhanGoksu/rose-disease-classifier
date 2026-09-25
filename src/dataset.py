"""Dataset loading: split file, transforms, DataLoaders."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

# Statistics of the ImageNet training set. The pretrained backbone learned its
# weights on inputs normalized with these values, so we must use the same ones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fixed on purpose, NOT alphabetical: Black_Spot is the positive class (1).
# sklearn's binary metrics use pos_label=1 by default, so "recall" then means
# "share of diseased leaves we caught", not the recall of healthy leaves.
CLASS_NAMES = ("Rose_Healthy", "Rose_Black_Spot")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
POSITIVE_CLASS = CLASS_TO_IDX["Rose_Black_Spot"]

SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_FILE = Path("splits/rose_split.csv")


@dataclass(frozen=True)
class Sample:
    """One row of the split file, with the path made absolute."""

    path: Path
    label: int
    source: str  # studio / fresh_leaf / black_spot (see docs/data_audit.md)
    group_id: int


def read_split(split_file: Path, data_dir: Path) -> dict[str, list[Sample]]:
    """Read the split file and return the samples of each split.

    Paths in the file are relative to the Rose folder, so the same file
    works locally and on Kaggle; `data_dir` supplies the machine's prefix.
    """
    splits: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    with split_file.open() as f:
        for row in csv.DictReader(f):
            splits[row["split"]].append(Sample(
                path=data_dir / row["path"],
                label=CLASS_TO_IDX[row["label"]],
                source=row["source"],
                group_id=int(row["group_id"]),
            ))

    # Fail now with a clear message, not in a DataLoader worker mid-epoch.
    missing = [s.path for samples in splits.values() for s in samples
               if not s.path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files from {split_file} are "
                                f"missing under {data_dir}, e.g. {missing[0]}")
    return splits


class RandomRightAngleRotation:
    """Rotate by 0, 90, 180 or 270 degrees. A leaf has no 'up', so every
    right-angle rotation is an image that could really occur."""

    def __call__(self, image: Image.Image) -> Image.Image:
        # torch RNG, not random: DataLoader seeds it separately per worker.
        k = int(torch.randint(4, ()).item())
        return image if k == 0 else image.rotate(90 * k, expand=True)


def build_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        # Random crop covering 70-100% of the image, resized to image_size.
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRightAngleRotation(),
        # No hue jitter: yellowing around the spots is a real symptom.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


class RoseDataset(Dataset):
    """Loads one image per index and applies a transform."""

    def __init__(
        self,
        samples: list[Sample],
        transform: transforms.Compose,
        decode_size: int = 448,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.decode_size = decode_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            # JPEG-only: decode at a reduced scale (still >= decode_size)
            # instead of the full 3000x3000, which is ~10x slower and gets
            # resized to image_size anyway.
            img.draft("RGB", (self.decode_size, self.decode_size))
            image = img.convert("RGB")  # Guarantees 3 channels.
        return self.transform(image), sample.label


def build_dataloaders(
    data_dir: Path,
    split_file: Path = DEFAULT_SPLIT_FILE,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    train_sampler: Sampler | None = None,
) -> dict[str, DataLoader]:
    """Build the train/val/test DataLoaders from the split file.

    Args:
        train_sampler: Decides which training images each batch draws
            (step 7). None means plain shuffling, as in Project 2A.
    """
    splits = read_split(split_file, data_dir)
    train_transform, eval_transform = build_transforms(image_size)
    decode_size = 2 * image_size  # Headroom for RandomResizedCrop.

    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,  # Faster host-to-GPU copies; no-op on CPU.
        "persistent_workers": num_workers > 0,  # Keep workers across epochs.
    }
    loaders = {"train": DataLoader(
        RoseDataset(splits["train"], train_transform, decode_size),
        # DataLoader accepts a sampler OR shuffle=True, never both.
        shuffle=train_sampler is None, sampler=train_sampler,
        **loader_kwargs,
    )}
    for name in ("val", "test"):
        # No shuffling: prediction i belongs to dataset.samples[i], which
        # lets us report metrics per source later.
        loaders[name] = DataLoader(
            RoseDataset(splits[name], eval_transform, decode_size),
            shuffle=False, **loader_kwargs,
        )
    return loaders


def main() -> None:
    """Smoke test: build the loaders and inspect one training batch."""
    parser = argparse.ArgumentParser(description="Inspect the DataLoaders.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    loaders = build_dataloaders(args.data_dir.expanduser(), args.split_file,
                                num_workers=args.num_workers)
    for name, loader in loaders.items():
        samples = loader.dataset.samples
        labels = Counter(CLASS_NAMES[s.label] for s in samples)
        sources = Counter(s.source for s in samples)
        print(f"{name:<5} {len(samples):>5} images, {len(loader):>3} batches "
              f"| {dict(labels)} | {dict(sources)}")

    images, labels = next(iter(loaders["train"]))
    print(f"images: shape={tuple(images.shape)} dtype={images.dtype} "
          f"min={images.min():.2f} max={images.max():.2f}")
    print(f"labels in first train batch: "
          f"{dict(Counter(CLASS_NAMES[i] for i in labels.tolist()))}")


if __name__ == "__main__":  # Required on macOS when num_workers > 0.
    main()
