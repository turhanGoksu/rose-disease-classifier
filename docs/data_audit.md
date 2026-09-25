# Data Audit: Rose_Healthy vs Rose_Black_Spot

Run before any split or training decision:

```bash
python scripts/audit_data.py --data-dir "<path>/Dataset/Rose"
```

## Summary

1. **The two classes come from different sources, and the sources are visible
   in the pixels.** Two thirds of the Healthy images (990 of 1488) are
   3000x3000 studio shots of a single leaf cut out on a pure white background.
   No Black_Spot image comes from this source. This is a strong
   shortcut-learning risk: "white background → Healthy" is right for 990
   training images without looking at the leaf at all.
2. **There are no exact duplicates, but there are near-duplicates.** 125
   pairs form 78 groups of 2–7 images: mirrored, cropped/zoomed and recolored
   copies of the same photo, plus the same leaf photographed more than once.
   A random split would put members of the same group into both train and
   val.
3. **The obvious duplicate check (dHash) failed on this data.** It reported
   single leaves on a plain background as copies of each other, and at the
   usual threshold it chained 1,105 unrelated images into one "group". A
   two-stage check with keypoint matching replaced it (see Method).

## A) Shortcut risk: source properties vs. class

| Source (filename family) | Class | Images | Resolution | Background |
|---|---|---:|---|---|
| `IMG (N)` | Healthy | 990 | 3000x3000 | pure white cut-out (median 75% near-white pixels) |
| `resized_Fresh Leaf_N` | Healthy | 498 | 1024x1024 | mixed: gray paper and field photos (34 mostly white) |
| `resized_Black Spot_N` | Black_Spot | 535 | 1024x1024 | mixed: gray paper and field photos (29 mostly white) |

Properties that line up with the class:

| Property | Healthy | Black_Spot | Risk |
|---|---|---|---|
| 3000x3000 resolution | 990 | 0 | High. Survives resizing to 224 as sharper texture. |
| Mostly white background (≥50% near-white) | 1024 | 29 | High. 97% of white-background images are Healthy. |
| `.JPG` upper-case extension | 714 | 0 | None for the model (not in the pixels), but confirms separate sources. |
| EXIF camera | none | none | None: EXIF was stripped from every file. |

Properties that do **not** separate the classes on their own:

- Compression: within the shared 1024x1024 source, median bits per pixel is
  0.43 (Healthy) vs 0.34 (Black_Spot), with heavily overlapping ranges
  (0.18–2.29 vs 0.16–2.28).

What this means for the project:

- The `resized_` sources (498 Healthy + 535 Black_Spot, same resolution, same
  photo styles) are the only part of the data where the classes can be
  compared on equal terms.
- Grad-CAM (step 8) must check whether the model looks at the leaf or at the
  background, and the evaluation should report metrics on the `resized_`
  subset separately, not only on the full val set.
- Observed while inspecting samples, not measured: many Black_Spot leaves are
  also yellowed or brown, while Healthy leaves are green. Leaf color is a
  second possible shortcut (a yellow healthy leaf is still healthy).

## B) Leakage risk: duplicates

| Check | Result |
|---|---|
| Exact duplicates (same MD5) | 0 |
| Near-duplicate pairs (≥ 20 RANSAC inliers) | 125 pairs → 78 groups (60 of size 2, 12 of 3, 4 of 4, 1 of 5, 1 of 7) |
| Images in a group | Healthy 93 / 1488 (6%), Black_Spot 91 / 535 (17%) |
| Groups spanning both classes | 6, all false matches on inspection (different leaves) |

Kinds of near-duplicates found by eye in the pair grids:

- the same field photo mirrored left-right (common in `resized_Fresh Leaf`),
- the same photo cropped or zoomed,
- the same photo with shifted colors (augmentation traces),
- the same diseased leaf photographed several times on paper, slightly
  moved or rotated (common in `resized_Black Spot`).

The filename numbers (`_2108`, `_3737`, ...) are not contiguous, which fits
a subset taken from a larger, augmented original collection.

## Method

**Why dHash failed.** dHash shrinks an image to 9x8 gray pixels and keeps 64
bits of "brighter than the right neighbour?". On a single dark leaf in the
middle of a white frame, those 64 bits mostly describe the background and
the rough leaf outline, which almost every studio image shares. With
orientation matching, the median nearest-neighbour distance among the 3000px
images was 4 of 64 bits, so unrelated leaves looked like copies. Correlating
64x64 leaf-cropped thumbnails had the same problem (different leaves scored
0.93–0.97 against true copies at 0.94–1.00): there was no threshold that
separated them.

**Two-stage check used instead.**

1. *Candidates:* for every image, its 10 most similar images by thumbnail
   correlation (leaf-cropped, all 8 orientations). Cheap, catches copies,
   and has many false candidates.
2. *Verification:* ORB keypoints + RANSAC on each candidate pair (the image
   and its mirror). A pair counts only if at least 20 keypoint matches agree
   on a single rotation/scale/shift. Copies of the same photo or leaf share
   exact local details (vein junctions, spot edges); different leaves do not.

Inlier counts on the 16,184 candidate pairs:

| Inliers | Pairs | Cross-class | By eye |
|---|---:|---:|---|
| 0–9 | 15,719 | 3,343 | unrelated |
| 10–19 | 340 | 35 | mostly unrelated |
| 20–29 | 24 | 6 | mixed: re-shot leaves and mirrors, plus false matches |
| 30–49 | 18 | 2 | mostly true copies |
| 50–99 | 21 | 0 | true copies |
| 100+ | 62 | 0 | true copies |

The threshold of 20 errs on the side of grouping. A false match only merges
two unrelated images into one group, which costs a little split flexibility.
A missed copy is leakage.

**Limitations.** Only the 10 nearest candidates per image are verified, so a
copy that is not among them is missed. Mirrored copies are matched; copies
that are both mirrored and heavily cropped may not be. Same-leaf re-shots
with large pose changes, or the two sides of a leaf, are not detected.
