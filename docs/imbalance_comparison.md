# Class Imbalance: No Balancing vs Sampler vs Loss Weights

The training set has 1040 Healthy and 376 Black_Spot images (2.8 : 1). Three
runs were trained on Kaggle (Tesla T4) from the same commit, with the same
split, seed (42), epochs (15), learning rates and augmentation. Only
`--balance` differs:

| Run | What changes | Details |
|---|---|---|
| `none` | nothing | Plain shuffling: a batch of 32 holds about 24 Healthy and 8 Black_Spot. |
| `sampler` | what the model **sees** | `WeightedRandomSampler`, weight 1/class size, with replacement. One epoch drew 710 Black_Spot samples covering only 322 of the 376 images, and 706 Healthy samples covering 508 of the 1040. |
| `loss` | how much each error **counts** | `CrossEntropyLoss(weight=[0.68, 1.88])`. Every image is seen once per epoch. Val/test loss stays unweighted. |

On average both balancing methods optimize the same objective: a Black_Spot
image is drawn 1.88x per epoch by the sampler, and a Black_Spot error is
weighted 1.88x by the loss. They differ in side effects. The sampler repeats
the same Black_Spot images and skips about half of the Healthy images in
each epoch. Loss weighting keeps all the data but makes gradients from
batches with few Black_Spot images larger and noisier. Neither creates new
information: the sampler still shows the same 376 diseased leaves.

## Results

Full tables: [`results/comparison.md`](../results/comparison.md). Per-run
metrics and training history: [`results/`](../results/).

| Run | Best epoch (by val) | Val macro-F1 | Test macro-F1 | Test same-source macro-F1 | Test Black_Spot recall | Test missed Black_Spot | Test false alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 15 | 0.996 | 0.991 | 0.987 | 0.987 | 1 | 1 |
| `sampler` | 8 | **1.000** | 0.983 | 0.974 | 0.949 | 4 | 0 |
| `loss` | 10 | 0.996 | 0.991 | 0.987 | 0.975 | 2 | 0 |

## The differences are noise

- **They are 1 to 4 images.** The sampler's val lead comes from one image
  (`resized_Black Spot_2893.jpg`), and its test deficit from three.
- **Rerunning the same configuration moves the numbers as much.** An
  earlier `none` run, with identical code path, seed and data, scored 0.987
  test macro-F1 (1 missed, 2 false alarms) instead of 0.991. The only
  difference is non-deterministic GPU arithmetic.
- **McNemar's exact test**, which compares two models only on the images
  where they disagree, finds no significant difference on test:

  | Pair | Images only the first gets right | Images only the second gets right | p |
  |---|---:|---:|---:|
  | none vs sampler | 3 | 1 | 0.625 |
  | none vs loss | 1 | 1 | 1.000 |
  | sampler vs loss | 0 | 2 | 0.500 |

The textbook effect of balancing (Black_Spot recall up, precision down) did
not appear either.

**Conclusion:** at 2.8 : 1, with an ImageNet-pretrained ResNet18 and this
dataset, class imbalance was not a problem that needed fixing. Neither the
sampler nor loss weighting made a measurable difference. A reliable ranking
would need several seeds per method, not one run each.

## Which model is "the" model

The rule was fixed before training: choose on val, report test once. By
that rule the selected model is **`sampler`** (val macro-F1 1.000), and its
test score is **0.983**.

It is not swapped for `none` or `loss` after seeing that they score higher
on test. Picking the luckiest of three equivalent runs *on the test set*
would turn the test set into a second validation set, and the reported
number would be inflated by that choice.

## The one error that is not noise

All three models miss the same test image, `resized_Black Spot_2266.jpg`,
and all three call it Healthy with at least 98.6% confidence. It is one of
the 29 Black_Spot images on a white background (see
[data_audit.md](data_audit.md)). Grad-CAM (step 8) examines it first.
