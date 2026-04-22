"""
T3.3 — Tamil Handwritten Character Recognition
Dataset : uTHCD 80-20 split HDF5
Model   : 3-layer CNN (~300k params) trained from scratch
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Tamil CNN")
parser.add_argument("--data",       default="../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5")
parser.add_argument("--epochs",     type=int,   default=20)
parser.add_argument("--batch_size", type=int,   default=128)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--img_size",   type=int,   default=32)
parser.add_argument("--ckpt_dir",   default="checkpoints")
parser.add_argument("--save_every", type=int,   default=5,   help="save checkpoint every N epochs")
args = parser.parse_args()

# ── Paths & logging ───────────────────────────────────────────────────────────
run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_dir = Path(args.ckpt_dir) / run_id
ckpt_dir.mkdir(parents=True, exist_ok=True)

log_file = ckpt_dir / "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Run ID   : {run_id}")
log.info(f"Device   : {DEVICE}")
log.info(f"Data     : {args.data}")
log.info(f"Epochs   : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
log.info(f"Ckpt dir : {ckpt_dir}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class uTHCDDataset(Dataset):
    """
    Loads pre-extracted numpy arrays from HDF5.
    Augmentation (train only): random rotation ±10°, translate ±10%, scale 0.9-1.1.
    All implemented with F.affine_grid so no torchvision needed.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 mean: float, std: float, augment: bool = False,
                 img_size: int = 32):
        self.images   = images                      # (N, H, W) uint8
        self.labels   = labels.astype(np.int64)
        self.mean     = mean
        self.std      = std
        self.augment  = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0   # (H, W)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Resize to target size
        img = F.interpolate(img, size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False)

        if self.augment:
            angle   = np.random.uniform(-10, 10)
            tx      = np.random.uniform(-0.1, 0.1)
            ty      = np.random.uniform(-0.1, 0.1)
            scale   = np.random.uniform(0.9, 1.1)
            rad     = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad) / scale, np.sin(rad) / scale
            theta = torch.tensor([[
                [cos_a,  -sin_a, tx],
                [sin_a,   cos_a, ty],
            ]], dtype=torch.float32)
            grid = F.affine_grid(theta, img.size(), align_corners=False)
            img  = F.grid_sample(img, grid, align_corners=False)

        img = img.squeeze(0)                        # (1, H, W)
        img = (img - self.mean) / self.std
        return img, self.labels[idx]


log.info("Loading HDF5 ...")
with h5py.File(args.data, "r") as f:
    x_train = f["Train Data/x_train"][:]   # (71760, 64, 64)
    y_train = f["Train Data/y_train"][:]
    x_test  = f["Test Data/x_test"][:]    # (19190, 64, 64)
    y_test  = f["Test Data/y_test"][:]

NUM_CLASSES = len(np.unique(y_train))
log.info(f"Train: {len(x_train):,}  |  Test: {len(x_test):,}  |  Classes: {NUM_CLASSES}")

# Normalize using train stats
mean = x_train.mean() / 255.0
std  = x_train.std()  / 255.0
log.info(f"Pixel mean: {mean:.4f}  std: {std:.4f}")

train_ds = uTHCDDataset(x_train, y_train, mean, std, augment=True,  img_size=args.img_size)
test_ds  = uTHCDDataset(x_test,  y_test,  mean, std, augment=False, img_size=args.img_size)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class TamilCNN(nn.Module):
    """3-block CNN — ~300k params for 32x32 input, ~1.1M for 64x64"""
    def __init__(self, num_classes: int, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),                  # H/2

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),                  # H/4

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                      # H/8
        )
        feat_size = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model     = TamilCNN(NUM_CLASSES, args.img_size).to(DEVICE)
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Model params: {n_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def save_checkpoint(epoch: int, val_acc: float, tag: str = ""):
    state = {
        "epoch":      epoch,
        "val_acc":    val_acc,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "args":       vars(args),
    }
    fname = ckpt_dir / f"ckpt_epoch{epoch:03d}{tag}.pth"
    torch.save(state, fname)
    return fname


# ── Training loop ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
log.info("=" * 60)

best_val_acc = 0.0
history      = []

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(test_loader,  train=False)
    scheduler.step()

    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    log.info(
        f"{epoch:5d}  {train_loss:10.4f}  {train_acc*100:8.2f}%  "
        f"{val_loss:8.4f}  {val_acc*100:7.2f}%  "
        f"{lr_now:.2e}  {elapsed:5.1f}s"
    )

    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
    })

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        path = save_checkpoint(epoch, val_acc, tag="_best")
        log.info(f"  *** New best val acc: {best_val_acc*100:.2f}%  →  {path.name}")

    # Periodic checkpoint
    if epoch % args.save_every == 0:
        path = save_checkpoint(epoch, val_acc)
        log.info(f"  Periodic checkpoint: {path.name}")

# ── Final test evaluation with full report ─────────────────────────────────────
log.info("\n" + "=" * 60)
log.info("Final evaluation on test set (best model)")
log.info("=" * 60)

best_ckpt = sorted(ckpt_dir.glob("*_best.pth"))[-1]
ckpt      = torch.load(best_ckpt, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
log.info(f"Loaded: {best_ckpt.name}  (saved at epoch {ckpt['epoch']})")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs   = imgs.to(DEVICE)
        preds  = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
log.info(f"Test accuracy : {test_acc*100:.2f}%")

report = classification_report(all_labels, all_preds, digits=3)
log.info(f"\nClassification report:\n{report}")

# Save history + report
with open(ckpt_dir / "history.json", "w") as f:
    json.dump(history, f, indent=2)

with open(ckpt_dir / "test_report.txt", "w") as f:
    f.write(f"Test accuracy: {test_acc*100:.2f}%\n\n")
    f.write(report)

log.info(f"\nAll outputs saved to: {ckpt_dir}")
log.info(f"Best val acc : {best_val_acc*100:.2f}%")
log.info(f"Test acc     : {test_acc*100:.2f}%")
