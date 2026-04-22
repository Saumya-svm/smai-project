"""
T3.3 — Analysis: correct/incorrect predictions, confusion matrix, hardest classes.
Run: /home/ajoy/miniconda3/envs/mvg/bin/python analyse.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CKPT   = "checkpoints/20260415_063836/ckpt_epoch020_best.pth"
DATA   = "../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5"
OUT    = Path("analysis")
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 32

# ── Model ─────────────────────────────────────────────────────────────────────
class TamilCNN(nn.Module):
    def __init__(self, num_classes, img_size=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_size = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
with h5py.File(DATA, "r") as f:
    x_test  = f["Test Data/x_test"][:]
    y_test  = f["Test Data/y_test"][:]
    x_train = f["Train Data/x_train"][:]

NUM_CLASSES = len(np.unique(y_test))
mean = x_train.mean() / 255.0
std  = x_train.std()  / 255.0

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model ...")
ckpt  = torch.load(CKPT, map_location=DEVICE)
model = TamilCNN(NUM_CLASSES, IMG_SIZE).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# ── Run inference on full test set ────────────────────────────────────────────
print("Running inference ...")
all_preds, all_probs, all_labels = [], [], []

BATCH = 256
for i in range(0, len(x_test), BATCH):
    imgs   = x_test[i:i+BATCH].astype(np.float32) / 255.0
    tensor = torch.from_numpy(imgs).unsqueeze(1)           # (B,1,64,64)
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    tensor = (tensor - mean) / std
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
    all_preds.extend(preds)
    all_probs.extend(probs)
    all_labels.extend(y_test[i:i+BATCH])

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

correct_mask   = all_preds == all_labels
test_acc       = correct_mask.mean()
print(f"Test accuracy: {test_acc*100:.2f}%")

# ── 1. Correct predictions grid ───────────────────────────────────────────────
correct_idxs = np.where(correct_mask)[0]
# Pick 20 high-confidence correct predictions
conf_correct  = all_probs[correct_idxs, all_preds[correct_idxs]]
top_correct   = correct_idxs[np.argsort(conf_correct)[::-1][:20]]

fig, axes = plt.subplots(4, 5, figsize=(13, 11))
fig.suptitle(f"Correct Predictions (high confidence) — Test acc {test_acc*100:.2f}%",
             fontsize=14, fontweight="bold")
for ax, idx in zip(axes.flat, top_correct):
    ax.imshow(x_test[idx], cmap="gray")
    conf = all_probs[idx, all_preds[idx]] * 100
    ax.set_title(f"Label {all_labels[idx]}\nPred {all_preds[idx]}  ({conf:.1f}%)",
                 fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "correct_predictions.png", dpi=120)
plt.close()
print("Saved correct_predictions.png")

# ── 2. Wrong predictions grid ─────────────────────────────────────────────────
wrong_idxs = np.where(~correct_mask)[0]
# Pick 20 high-confidence wrong predictions (model was very sure but wrong)
conf_wrong = all_probs[wrong_idxs, all_preds[wrong_idxs]]
top_wrong  = wrong_idxs[np.argsort(conf_wrong)[::-1][:20]]

fig, axes = plt.subplots(4, 5, figsize=(13, 11))
fig.suptitle("Wrong Predictions (model was most confident — hardest mistakes)",
             fontsize=14, fontweight="bold")
for ax, idx in zip(axes.flat, top_wrong):
    ax.imshow(x_test[idx], cmap="gray")
    conf = all_probs[idx, all_preds[idx]] * 100
    ax.set_title(f"True {all_labels[idx]}  → Pred {all_preds[idx]}\n({conf:.1f}% conf)",
                 fontsize=8, color="red")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "wrong_predictions.png", dpi=120)
plt.close()
print("Saved wrong_predictions.png")

# ── 3. Confusion matrix (full 156×156) ───────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))

fig, ax = plt.subplots(figsize=(20, 18))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted class", fontsize=12)
ax.set_ylabel("True class", fontsize=12)
ax.set_title("Confusion Matrix — Tamil Character Recognition (156 classes)", fontsize=14)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_xticks(np.arange(0, NUM_CLASSES, 10))
ax.set_yticks(np.arange(0, NUM_CLASSES, 10))
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion_matrix.png")

# ── 4. Most confused class pairs ─────────────────────────────────────────────
cm_nodiag = cm.copy()
np.fill_diagonal(cm_nodiag, 0)

# Top 20 off-diagonal entries
flat_idx   = np.argsort(cm_nodiag.ravel())[::-1][:20]
true_cls   = flat_idx // NUM_CLASSES
pred_cls   = flat_idx %  NUM_CLASSES
conf_count = cm_nodiag.ravel()[flat_idx]

print("\nTop 20 most confused class pairs:")
print(f"{'Rank':>4}  {'True':>5}  {'Pred':>5}  {'#Errors':>8}  {'Per-class err%':>14}")
print("-" * 50)
for rank, (t, p, c) in enumerate(zip(true_cls, pred_cls, conf_count), 1):
    n_true = cm[t].sum()
    pct    = c / n_true * 100 if n_true > 0 else 0
    print(f"{rank:4d}  {t:5d}  {p:5d}  {c:8d}  {pct:13.1f}%")

# ── 5. Zoomed confusion matrix: top confused classes ─────────────────────────
# Find the 20 classes with the most errors (lowest per-class accuracy)
per_class_acc  = cm.diagonal() / cm.sum(axis=1).clip(min=1)
worst_classes  = np.argsort(per_class_acc)[:20]
cm_sub         = cm[np.ix_(worst_classes, worst_classes)]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_sub, cmap="Reds")
ax.set_xticks(np.arange(len(worst_classes)))
ax.set_yticks(np.arange(len(worst_classes)))
ax.set_xticklabels(worst_classes, rotation=90, fontsize=9)
ax.set_yticklabels(worst_classes, fontsize=9)
ax.set_xlabel("Predicted class", fontsize=12)
ax.set_ylabel("True class", fontsize=12)
ax.set_title("Confusion Matrix — 20 Hardest Classes", fontsize=14)
for i in range(len(worst_classes)):
    for j in range(len(worst_classes)):
        if cm_sub[i, j] > 0:
            ax.text(j, i, str(cm_sub[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm_sub[i, j] > cm_sub.max()*0.5 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_hardest.png", dpi=130)
plt.close()
print("Saved confusion_matrix_hardest.png")

# ── 6. Per-class accuracy bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 5))
colors = ["#d62728" if a < 0.90 else "#2ca02c" for a in per_class_acc]
ax.bar(np.arange(NUM_CLASSES), per_class_acc * 100, color=colors, width=0.8)
ax.axhline(90, color="orange", linestyle="--", linewidth=1, label="90% threshold")
ax.axhline(test_acc * 100, color="blue", linestyle="--", linewidth=1,
           label=f"Overall {test_acc*100:.1f}%")
ax.set_xlabel("Class index")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-class Test Accuracy (red = below 90%)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "per_class_accuracy.png", dpi=120)
plt.close()
print("Saved per_class_accuracy.png")

# ── 7. Summary report ─────────────────────────────────────────────────────────
n_below90 = (per_class_acc < 0.90).sum()
print(f"\n{'='*55}")
print(f"Overall test accuracy : {test_acc*100:.2f}%")
print(f"Classes below 90%     : {n_below90} / {NUM_CLASSES}")
print(f"\n20 worst classes (by accuracy):")
for cls in worst_classes:
    acc_pct = per_class_acc[cls] * 100
    n_err   = cm[cls].sum() - cm[cls, cls]
    # What does it get confused with most?
    confused_with = np.argsort(cm_nodiag[cls])[::-1][0]
    print(f"  Class {cls:3d}  acc={acc_pct:5.1f}%  errors={n_err:3d}  most confused with class {confused_with}")
print(f"\nAll plots saved to: {OUT}/")
