# Rose Disease Classifier: Imbalance, Leakage, Metrics, and Grad-CAM

Binary image classifier (`Rose_Healthy` vs `Rose_Black_Spot`) on a pretrained
ResNet18, trained with a plain PyTorch loop. The focus is not the classifier
itself but four judgment calls: handling class imbalance, building a
leakage-safe train/val split, choosing the right metric, and using Grad-CAM to
check whether the model looks at the disease or at a shortcut.

Work in progress.
