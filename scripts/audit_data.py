"""Audit the Rose subset before choosing a train/val split.

Answers two questions:
  A) Shortcut risk: does any property unrelated to the disease (resolution,
     file size, background, filename pattern, camera) line up with the label?
  B) Leakage risk: are some images copies of each other (exact copies, or
     mirrored, cropped, recolored or re-shot versions of the same leaf)?

Near-duplicates are found in two stages, because every whole-image
similarity measure we tried (dHash, thumbnail correlation) also scores
*different* single leaves on a plain background as near-identical:
  1) Candidates: each image's K most similar images by thumbnail correlation.
  2) Verification: ORB keypoint matching + RANSAC. A pair counts as the same
     photo/leaf only if many keypoints match under one consistent geometric
     transform (rotation, scale, shift), which unrelated leaves do not have.

Usage:
    python scripts/audit_data.py \
        --data-dir "~/Desktop/final dataset/Dataset/Rose"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASSES = ("Rose_Healthy", "Rose_Black_Spot")
WORK_SIZE = 512  # Longest side used for all pixel-based checks.
WHITE_LEVEL = 235  # Gray value above which a pixel counts as background.

# The 8 orientations of a square (identity, 3 rotations, 4 mirrorings),
# so mirrored or rotated copies can still be found in stage 1.
ORIENTATIONS: tuple[Image.Transpose | None, ...] = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)

# Filename families seen in the folders; anything else is reported as "other".
FILENAME_PATTERNS = {
    "IMG (N)": re.compile(r"^IMG \(\d+\)$"),
    "resized_Fresh Leaf_N": re.compile(r"^resized_Fresh Leaf_\d+$"),
    "resized_Black Spot_N": re.compile(r"^resized_Black Spot_\d+$"),
}

# (keypoint xy, descriptors). A runtime alias, so Optional instead of
# "| None", which Python 3.9 cannot evaluate outside annotations.
Features = tuple[np.ndarray, Optional[np.ndarray]]


@dataclass
class ImageRecord:
    """Everything the audit measures about one image file."""

    label: str
    filename: str
    width: int
    height: int
    resolution: str  # "WxH", kept as one value for the class cross-table.
    file_kb: float
    bits_per_pixel: float  # File bits / pixels: lower = stronger compression.
    white_fraction: float  # Share of near-white pixels (plain background).
    extension: str
    filename_pattern: str
    camera: str
    md5: str


def filename_pattern(stem: str) -> str:
    """Return the name of the first filename family that matches `stem`."""
    for name, pattern in FILENAME_PATTERNS.items():
        if pattern.match(stem):
            return name
    return "other"


def read_camera(img: Image.Image) -> str:
    """Return 'Make Model' from EXIF, or 'no EXIF' if it is missing."""
    exif = img.getexif()
    make = str(exif.get(ExifTags.Base.Make, "")).strip("\x00 ")
    model = str(exif.get(ExifTags.Base.Model, "")).strip("\x00 ")
    return f"{make} {model}".strip() or "no EXIF"


def thumbnail_vectors(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Leaf-cropped, normalized thumbnails in all 8 orientations.

    Cropping to the non-white bounding box stops a plain background from
    dominating the comparison. Each vector has zero mean and unit length, so
    a dot product is the Pearson correlation and ignores brightness/contrast
    changes (a recolored copy still correlates highly).

    Returns:
        [8, size * size] float32 array, one row per orientation.
    """
    ys, xs = np.nonzero(gray < WHITE_LEVEL)
    image = Image.fromarray(gray)
    if len(ys):
        image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    vectors = []
    for op in ORIENTATIONS:
        oriented = image if op is None else image.transpose(op)
        small = oriented.resize((size, size), Image.Resampling.LANCZOS)
        v = np.asarray(small, dtype=np.float32).ravel()
        v = (v - v.mean()) / (v.std() + 1e-6)
        vectors.append(v / np.sqrt(v.size))
    return np.stack(vectors)


def orb_features(gray: np.ndarray, orb: cv2.ORB) -> tuple[Features, Features]:
    """ORB keypoints of the image and of its mirror image.

    ORB already handles rotation and scale, but not mirroring, so a mirrored
    copy is matched against the mirrored features.
    """
    out = []
    for array in (gray, np.ascontiguousarray(gray[:, ::-1])):
        keypoints, descriptors = orb.detectAndCompute(array, None)
        points = np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)
        out.append((points, descriptors))
    return out[0], out[1]


def audit_image(
    path: Path, label: str, orb: cv2.ORB
) -> tuple[ImageRecord, np.ndarray, tuple[Features, Features]]:
    """Measure one image.

    Returns:
        (record, thumbnail vectors [8, 4096], ORB features (upright, mirror)).
    """
    raw = path.read_bytes()
    with Image.open(path) as img:
        # Read size and EXIF before draft(), which changes the decoded size.
        width, height = img.size
        camera = read_camera(img)
        # JPEG-only speedup: decode at 1/2..1/8 scale straight from the DCT
        # coefficients; the checks below never need full resolution.
        img.draft("L", (WORK_SIZE, WORK_SIZE))
        small = img.convert("L")
        small.thumbnail((WORK_SIZE, WORK_SIZE))
    gray = np.asarray(small)

    record = ImageRecord(
        label=label,
        filename=path.name,
        width=width,
        height=height,
        resolution=f"{width}x{height}",
        file_kb=round(len(raw) / 1024, 1),
        bits_per_pixel=round(len(raw) * 8 / (width * height), 3),
        white_fraction=round(float((gray >= WHITE_LEVEL).mean()), 3),
        extension=path.suffix,  # Keep case: ".jpg" vs ".JPG" is a signal too.
        filename_pattern=filename_pattern(path.stem),
        camera=camera,
        md5=hashlib.md5(raw).hexdigest(),
    )
    return record, thumbnail_vectors(gray), orb_features(gray, orb)


def correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    """Highest correlation between every pair over all 8 orientations.

    Args:
        vectors: [N, 8, D] normalized thumbnails from thumbnail_vectors().

    Returns:
        [N, N] matrix; the diagonal is -1 so an image is not its own match.
    """
    upright = vectors[:, 0, :]
    best = np.full((len(vectors),) * 2, -1.0, dtype=np.float32)
    for k in range(vectors.shape[1]):
        best = np.maximum(best, upright @ vectors[:, k, :].T)
    best = np.maximum(best, best.T)  # Symmetric regardless of direction.
    np.fill_diagonal(best, -1.0)
    return best


def candidate_pairs(corr: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Stage 1: each image paired with its k most correlated images."""
    pairs = set()
    for i, row in enumerate(corr):
        for j in np.argsort(-row)[:k]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def ransac_inliers(
    a: Features, b: Features, matcher: cv2.BFMatcher
) -> int:
    """Stage 2: number of keypoint matches that agree on one geometric
    transform (rotation + uniform scale + shift) between the two images."""
    (points_a, desc_a), (points_b, desc_b) = a, b
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return 0
    # Lowe's ratio test: keep a match only if it is clearly better than the
    # second-best candidate, which drops ambiguous (repetitive) texture.
    good = [m[0] for m in matcher.knnMatch(desc_a, desc_b, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 3:  # Need at least 3 points to fit the transform.
        return 0
    src = points_a[[m.queryIdx for m in good]]
    dst = points_b[[m.trainIdx for m in good]]
    _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
    return 0 if mask is None else int(mask.sum())


def connected_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find: merge linked indices into groups (only groups of 2+)."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path halving.
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [members for members in groups.values() if len(members) > 1]


def print_crosstab(
    title: str, records: list[ImageRecord], key: str, classes: list[str]
) -> None:
    """Print how often each value of `key` occurs in each class."""
    counts = Counter((getattr(r, key), r.label) for r in records)
    values = sorted({getattr(r, key) for r in records}, key=str)
    print(f"\n{title}")
    print(f"  {'value':<32}" + "".join(f"{c:>18}" for c in classes))
    for value in values:
        row = "".join(f"{counts[(value, c)]:>18}" for c in classes)
        print(f"  {str(value):<32}{row}")


def print_numeric_summary(
    title: str, records: list[ImageRecord], key: str
) -> None:
    """Print median/min/max of a numeric field per (class, resolution)."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        buckets[(r.label, r.resolution)].append(getattr(r, key))
    print(f"\n{title}")
    for (label, size), values in sorted(buckets.items()):
        arr = np.array(values)
        print(f"  {label:<18} {size:>10}  n={len(arr):>4}  "
              f"median={np.median(arr):8.2f}  "
              f"min={arr.min():8.2f}  max={arr.max():8.2f}")


def save_pair_grid(
    pairs: list[tuple[int, int, int]],
    paths: list[Path],
    records: list[ImageRecord],
    out_path: Path,
    thumb: int = 120,
    columns: int = 4,
) -> None:
    """Save side-by-side thumbnails of image pairs, labeled with the score."""
    cell_w, cell_h = thumb * 2 + 12, thumb + 28
    n_rows = max(1, (len(pairs) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_w, n_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (i, j, score) in enumerate(pairs):
        x0, y0 = (k % columns) * cell_w, (k // columns) * cell_h
        names = [f"{records[m].label.replace('Rose_', '')}/"
                 f"{Path(records[m].filename).stem[-12:]}" for m in (i, j)]
        draw.text((x0, y0), f"{score}: {names[0]}", fill="black")
        draw.text((x0, y0 + 12), f"   {names[1]}", fill="black")
        for col, idx in enumerate((i, j)):
            with Image.open(paths[idx]) as img:
                img.draft("RGB", (thumb, thumb))
                img = img.convert("RGB")
                img.thumbnail((thumb, thumb))
                sheet.paste(img, (x0 + col * thumb, y0 + 26))
    sheet.save(out_path)


def save_source_samples(
    paths: list[Path],
    records: list[ImageRecord],
    out_path: Path,
    per_source: int = 6,
    thumb: int = 150,
    seed: int = 0,
) -> None:
    """Save one row of random thumbnails per filename family (source)."""
    rng = random.Random(seed)
    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        by_source[r.filename_pattern].append(idx)
    label_h = 22
    sheet = Image.new("RGB", (per_source * thumb,
                              len(by_source) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (source, members) in enumerate(sorted(by_source.items())):
        y = row * (thumb + label_h)
        first = records[members[0]]
        draw.text((4, y + 5), f"{source}: {first.label}, {len(members)} "
                  f"images, {first.resolution}", fill="black")
        for col, idx in enumerate(rng.sample(members,
                                             min(per_source, len(members)))):
            with Image.open(paths[idx]) as img:
                img.draft("RGB", (thumb, thumb))
                img = img.convert("RGB")
                img.thumbnail((thumb, thumb))
                sheet.paste(img, (col * thumb, y + label_h))
    sheet.save(out_path)


def scan(data_dir: Path, classes: list[str]) -> list[tuple[Path, str]]:
    """List (path, class name) for every image in the chosen class folders."""
    items: list[tuple[Path, str]] = []
    for label in classes:
        folder = data_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The Rose folder with one sub-folder per class.")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/audit"))
    parser.add_argument("--candidates", type=int, default=10,
                        help="Stage 1: most similar images checked per image.")
    parser.add_argument("--min-inliers", type=int, default=20,
                        help="Stage 2: RANSAC inliers needed to call a pair "
                             "a near-duplicate.")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = scan(args.data_dir.expanduser(), args.classes)
    paths = [path for path, _ in items]
    orb = cv2.ORB_create(nfeatures=1000)
    records: list[ImageRecord] = []
    vectors: list[np.ndarray] = []
    features: list[tuple[Features, Features]] = []
    for k, (path, label) in enumerate(items, start=1):
        record, vecs, feats = audit_image(path, label, orb)
        records.append(record)
        vectors.append(vecs)
        features.append(feats)
        if k % 500 == 0:
            print(f"  audited {k}/{len(items)} images")

    # --- A) Shortcut risk: properties vs. class --------------------------
    print_crosstab("Images per class", records, "label", args.classes)
    print_crosstab("Resolution x class", records, "resolution", args.classes)
    print_crosstab("Extension x class", records, "extension", args.classes)
    print_crosstab("Filename pattern x class", records, "filename_pattern",
                   args.classes)
    print_crosstab("Camera (EXIF) x class", records, "camera", args.classes)
    print_numeric_summary("File size in KB", records, "file_kb")
    print_numeric_summary("Bits per pixel (compression)", records,
                          "bits_per_pixel")
    print_numeric_summary("Near-white pixel fraction (background)", records,
                          "white_fraction")

    # --- B) Leakage risk: exact and near-duplicates ----------------------
    by_md5: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        by_md5[r.md5].append(idx)
    exact = [group for group in by_md5.values() if len(group) > 1]
    print(f"\nExact duplicates (same MD5): {len(exact)} groups")

    corr = correlation_matrix(np.stack(vectors))
    candidates = candidate_pairs(corr, args.candidates)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    scored: list[tuple[int, int, int]] = []
    for i, j in candidates:
        upright_i = features[i][0]
        # Compare i with j as-is and with j mirrored; keep the better match.
        inliers = max(ransac_inliers(upright_i, features[j][0], matcher),
                      ransac_inliers(upright_i, features[j][1], matcher))
        scored.append((i, j, inliers))
    scored.sort(key=lambda p: -p[2])
    print(f"Stage 1 candidate pairs: {len(candidates)}")
    print("Stage 2 RANSAC inliers per candidate pair:")
    bands = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 10**6)]
    for lo, hi in bands:
        band = [p for p in scored if lo <= p[2] < hi]
        same = sum(records[i].label == records[j].label for i, j, _ in band)
        print(f"  {lo:>3}-{'' if hi == 10**6 else hi - 1:<4} {len(band):>6} "
              f"pairs ({same} same-class, {len(band) - same} cross-class)")

    duplicates = [p for p in scored if p[2] >= args.min_inliers]
    groups = connected_groups(len(records), [(i, j) for i, j, _ in duplicates])
    cross = [g for g in groups if len({records[i].label for i in g}) > 1]
    print(f"\nNear-duplicates at >= {args.min_inliers} inliers: "
          f"{len(duplicates)} pairs -> {len(groups)} groups "
          f"(sizes: {dict(sorted(Counter(map(len, groups)).items()))})")
    in_groups = Counter(records[i].label for g in groups for i in g)
    for label in args.classes:
        total = sum(r.label == label for r in records)
        print(f"  {label:<18} {in_groups[label]:>4} / {total} images "
              f"are in a group")
    print(f"  groups spanning both classes: {len(cross)}")

    # --- Files for a closer look ----------------------------------------
    with (output_dir / "images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)
    with (output_dir / "near_duplicate_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["inliers", "correlation", "image_a", "image_b"])
        for i, j, inliers in scored:
            writer.writerow([inliers, round(float(corr[i, j]), 4),
                             f"{records[i].label}/{records[i].filename}",
                             f"{records[j].label}/{records[j].filename}"])
    save_source_samples(paths, records, output_dir / "sources.png")
    # One grid per band, to check by eye where real copies stop.
    for lo, hi in bands[1:]:
        band = [p for p in scored if lo <= p[2] < hi][:24]
        if band:
            save_pair_grid(band, paths, records,
                           output_dir / f"pairs_inliers_{lo}.png")
    print(f"\nWrote images.csv, near_duplicate_pairs.csv, sources.png and "
          f"pair grids to {output_dir}")


if __name__ == "__main__":
    main()
