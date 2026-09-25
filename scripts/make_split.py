"""Build the leakage-safe train/val/test split from the audit results.

Near-duplicate pairs found by audit_data.py are merged into groups, and whole
groups (never single images) are assigned to a split. Each source keeps
roughly the same share in train, val and test.

Usage (after running scripts/audit_data.py):
    python scripts/make_split.py \
        --data-dir "~/Desktop/final dataset/Dataset/Rose"
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_data import (DEFAULT_CLASSES, connected_groups, filename_pattern,
                        scan)

SPLITS = ("train", "val", "test")

# Filename family -> short source name used in the split file.
SOURCES = {
    "IMG (N)": "studio",                    # Healthy, 3000px, white cut-out
    "resized_Fresh Leaf_N": "fresh_leaf",   # Healthy, 1024px
    "resized_Black Spot_N": "black_spot",   # Black_Spot, 1024px
}


def source_of(filename: str) -> str:
    """Map a filename to its source; fail loudly on an unknown family."""
    pattern = filename_pattern(Path(filename).stem)
    if pattern not in SOURCES:
        raise ValueError(f"Unknown filename family: {filename}")
    return SOURCES[pattern]


def load_duplicate_pairs(
    pairs_csv: Path, index: dict[str, int], min_inliers: int
) -> list[tuple[int, int]]:
    """Read audit pairs with at least `min_inliers` RANSAC inliers."""
    pairs: list[tuple[int, int]] = []
    with pairs_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["inliers"]) < min_inliers:
                continue
            a, b = row["image_a"], row["image_b"]
            if a not in index or b not in index:
                raise KeyError(f"Pair not in the dataset: {a} | {b}")
            pairs.append((index[a], index[b]))
    return pairs


def assign_groups(
    group_members: list[list[int]],
    sources: list[str],
    fractions: dict[str, float],
    seed: int,
) -> list[str]:
    """Assign every group to a split, balancing each source separately.

    Within a source, groups are handled largest first (random order among
    equal sizes), and each one goes to the split that is furthest below its
    target count. Large groups placed early leave small ones to even out the
    totals.

    Returns:
        The split name of every group, indexed like `group_members`.
    """
    rng = random.Random(seed)  # Local RNG: reproducible, no global effects.
    by_source: dict[str, list[int]] = defaultdict(list)
    for gid, members in enumerate(group_members):
        # A false match can link two sources; the majority decides.
        majority = Counter(sources[i] for i in members).most_common(1)[0][0]
        by_source[majority].append(gid)

    group_split = [""] * len(group_members)
    for source in sorted(by_source):
        gids = by_source[source]
        rng.shuffle(gids)
        gids.sort(key=lambda g: -len(group_members[g]))  # Stable sort.
        total = sum(len(group_members[g]) for g in gids)
        filled = dict.fromkeys(SPLITS, 0)
        for gid in gids:
            split = max(SPLITS,
                        key=lambda s: fractions[s] * total - filled[s])
            group_split[gid] = split
            filled[split] += len(group_members[gid])
    return group_split


def print_table(rows: list[dict[str, str]], key: str) -> None:
    """Print image counts (and share of that row) per split."""
    counts = Counter((r[key], r["split"]) for r in rows)
    print(f"\n  {key:<18}" + "".join(f"{s:>14}" for s in SPLITS))
    for value in sorted({r[key] for r in rows}):
        total = sum(counts[(value, s)] for s in SPLITS)
        cells = "".join(
            f"{counts[(value, s)]:>6} ({counts[(value, s)] / total:4.0%})"
            for s in SPLITS)
        print(f"  {value:<18}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The Rose folder with one sub-folder per class.")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--pairs", type=Path,
                        default=Path("outputs/audit/near_duplicate_pairs.csv"))
    parser.add_argument("--min-inliers", type=int, default=20,
                        help="Same threshold as the audit (see its report).")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("splits/rose_split.csv"))
    args = parser.parse_args()

    items = scan(args.data_dir.expanduser(), args.classes)
    # Paths relative to the Rose folder, so the file works on any machine.
    paths = [f"{label}/{path.name}" for path, label in items]
    labels = [label for _, label in items]
    sources = [source_of(path.name) for path, _ in items]
    index = {path: i for i, path in enumerate(paths)}

    pairs = load_duplicate_pairs(args.pairs, index, args.min_inliers)
    linked = connected_groups(len(paths), pairs)
    in_linked = {i for members in linked for i in members}
    # Every image without a duplicate becomes a group of one.
    group_members = linked + [[i] for i in range(len(paths))
                              if i not in in_linked]
    group_members.sort(key=min)  # Group ids follow file order: stable ids.

    fractions = {"val": args.val_fraction, "test": args.test_fraction}
    fractions["train"] = 1.0 - fractions["val"] - fractions["test"]
    group_split = assign_groups(group_members, sources, fractions, args.seed)

    rows: list[dict[str, str]] = []
    for gid, members in enumerate(group_members):
        for i in members:
            rows.append({"path": paths[i], "label": labels[i],
                         "source": sources[i], "group_id": str(gid),
                         "split": group_split[gid]})
    rows.sort(key=lambda r: r["path"])

    # The leakage guarantee, checked on the output itself.
    splits_per_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        splits_per_group[r["group_id"]].add(r["split"])
    leaked = [g for g, s in splits_per_group.items() if len(s) > 1]
    if leaked:
        raise RuntimeError(f"Groups found in more than one split: {leaked}")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} images, {len(group_members)} groups "
          f"({len(linked)} with 2+ images, from {len(pairs)} pairs)")
    print("check passed: no group appears in more than one split")
    print_table(rows, "source")
    print_table(rows, "label")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
