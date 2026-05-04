"""
T3.3 - Tamil Handwritten Character Recognition
Dataset : uTHCD 80-20 split HDF5
Model   : 3-block CNN trained from scratch on 32x32 grayscale inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tamil CNN")
    parser.add_argument("--data", default="../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5")
    parser.add_argument("--label_map", default="idx_to_class.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--save_every", type=int, default=5, help="save checkpoint every N epochs")
    parser.add_argument("--val_split", type=float, default=0.1, help="fraction of Train Data reserved for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


args = parse_args()

if not 0.0 < args.val_split < 0.5:
    raise ValueError("--val_split must be between 0 and 0.5")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(args.seed)


# -- Paths and logging ---------------------------------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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
PIN_MEMORY = DEVICE == "cuda"

log.info(f"Run ID    : {run_id}")
log.info(f"Device    : {DEVICE}")
log.info(f"Data      : {args.data}")
log.info(f"Label map : {args.label_map}")
log.info(f"Epochs    : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
log.info(f"Val split : {args.val_split:.2f}  |  Seed: {args.seed}  |  Workers: {args.num_workers}")
log.info(f"Ckpt dir  : {ckpt_dir}")


def resolve_existing_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, BASE_DIR / path, Path.cwd() / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_label_map(path_str: str, num_classes: int) -> dict:
    path = resolve_existing_path(path_str)
    if path is None:
        log.warning(f"Label map not found at {path_str}; checkpoints will not embed class labels.")
        return {}

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    label_map = {str(k): str(v) for k, v in raw.items()}
    if len(label_map) != num_classes:
        log.warning(
            f"Label map has {len(label_map)} entries but the dataset has {num_classes} classes."
        )
    return label_map


# -- Dataset ------------------------------------------------------------------
class UTHCDDataset(Dataset):
    """
    Loads numpy arrays from HDF5.
    Augmentation (train only): random rotation +-10 degrees, translate +-10%,
    scale 0.9-1.1. Implemented with affine_grid/grid_sample so torchvision is
    not required.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mean: float,
        std: float,
        augment: bool = False,
        img_size: int = 32,
        indices: np.ndarray | None = None,
    ):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        img = self.images[real_idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = F.interpolate(
            img,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.augment:
            angle = np.random.uniform(-10, 10)
            tx = np.random.uniform(-0.1, 0.1)
            ty = np.random.uniform(-0.1, 0.1)
            scale = np.random.uniform(0.9, 1.1)
            rad = np.deg2rad(angle)
            cos_a = np.cos(rad) / scale
            sin_a = np.sin(rad) / scale
            theta = torch.tensor(
                [[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]],
                dtype=torch.float32,
            )
            grid = F.affine_grid(theta, img.size(), align_corners=False)
            img = F.grid_sample(img, grid, align_corners=False)

        img = img.squeeze(0)
        img = (img - self.mean) / self.std
        return img, int(self.labels[real_idx])


log.info("Loading HDF5 ...")
with h5py.File(args.data, "r") as f:
    x_train_full = f["Train Data/x_train"][:]
    y_train_full = f["Train Data/y_train"][:]
    x_test = f["Test Data/x_test"][:]
    y_test = f["Test Data/y_test"][:]

NUM_CLASSES = len(np.unique(y_train_full))
label_map = load_label_map(args.label_map, NUM_CLASSES)

train_idx, val_idx = train_test_split(
    np.arange(len(y_train_full)),
    test_size=args.val_split,
    random_state=args.seed,
    shuffle=True,
    stratify=y_train_full,
)

mean = x_train_full[train_idx].mean() / 255.0
std = x_train_full[train_idx].std() / 255.0

log.info(
    "Split sizes: train=%s  |  val=%s  |  test=%s  |  classes=%s",
    f"{len(train_idx):,}",
    f"{len(val_idx):,}",
    f"{len(x_test):,}",
    NUM_CLASSES,
)
log.info(f"Pixel mean: {mean:.4f}  std: {std:.4f}")

train_ds = UTHCDDataset(
    x_train_full,
    y_train_full,
    mean,
    std,
    augment=True,
    img_size=args.img_size,
    indices=train_idx,
)
val_ds = UTHCDDataset(
    x_train_full,
    y_train_full,
    mean,
    std,
    augment=False,
    img_size=args.img_size,
    indices=val_idx,
)
test_ds = UTHCDDataset(
    x_test,
    y_test,
    mean,
    std,
    augment=False,
    img_size=args.img_size,
)

loader_kwargs = {
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "pin_memory": PIN_MEMORY,
    "persistent_workers": args.num_workers > 0,
}
train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)


# -- Model --------------------------------------------------------------------
class TamilCNN(nn.Module):
    """3-block CNN with ~1.27M trainable params for 32x32 inputs."""

    def __init__(self, num_classes: int, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_size = (img_size // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model = TamilCNN(NUM_CLASSES, args.img_size).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Model params: {n_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# -- Helpers ------------------------------------------------------------------
def run_train_epoch(loader: DataLoader) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def save_checkpoint(epoch: int, val_acc: float, tag: str = "") -> Path:
    state = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model": model.state_dict(),
        "num_classes": NUM_CLASSES,
        "data_mean": mean,
        "data_std": std,
        "raw_img_size": int(x_train_full.shape[-1]),
        "label_map": label_map,
        "split": {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(y_test),
            "val_split": args.val_split,
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    fname = ckpt_dir / f"ckpt_epoch{epoch:03d}{tag}.pth"
    torch.save(state, fname)
    return fname


# -- Training loop ------------------------------------------------------------
log.info("=" * 68)
log.info(
    f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
    f"{'Val Loss':>8}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}"
)
log.info("=" * 68)

best_val_acc = 0.0
best_epoch = 0
history: list[dict] = []

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    train_loss, train_acc = run_train_epoch(train_loader)
    val_loss, val_acc, _, _ = evaluate(val_loader)
    scheduler.step()

    elapsed = time.time() - t0
    lr_now = scheduler.get_last_lr()[0]

    log.info(
        f"{epoch:5d}  {train_loss:10.4f}  {train_acc*100:8.2f}%  "
        f"{val_loss:8.4f}  {val_acc*100:7.2f}%  {lr_now:.2e}  {elapsed:5.1f}s"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_now,
        }
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        path = save_checkpoint(epoch, val_acc, tag="_best")
        log.info(f"  *** New best val acc: {best_val_acc*100:.2f}%  ->  {path.name}")

    if epoch % args.save_every == 0:
        path = save_checkpoint(epoch, val_acc)
        log.info(f"  Periodic checkpoint: {path.name}")


# -- Final evaluation ---------------------------------------------------------
log.info("\n" + "=" * 68)
log.info("Final evaluation")
log.info("=" * 68)

best_ckpt = sorted(ckpt_dir.glob("*_best.pth"))[-1]
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
log.info(f"Loaded best checkpoint: {best_ckpt.name}  (epoch {ckpt['epoch']})")

val_loss, val_acc, val_preds, val_labels = evaluate(val_loader)
test_loss, test_acc, test_preds, test_labels = evaluate(test_loader)

log.info(f"Validation loss : {val_loss:.4f}")
log.info(f"Validation acc  : {val_acc*100:.2f}%")
log.info(f"Test loss       : {test_loss:.4f}")
log.info(f"Test accuracy   : {test_acc*100:.2f}%")

val_report = classification_report(val_labels, val_preds, digits=3, zero_division=0)
test_report = classification_report(test_labels, test_preds, digits=3, zero_division=0)

log.info(f"\nValidation classification report:\n{val_report}")
log.info(f"\nTest classification report:\n{test_report}")

summary = {
    "run_id": run_id,
    "best_epoch": best_epoch,
    "best_val_acc": best_val_acc,
    "final_val_loss": val_loss,
    "final_val_acc": val_acc,
    "final_test_loss": test_loss,
    "final_test_acc": test_acc,
    "num_classes": NUM_CLASSES,
    "train_size": len(train_idx),
    "val_size": len(val_idx),
    "test_size": len(y_test),
    "mean": mean,
    "std": std,
}

with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(ckpt_dir / "val_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Validation loss: {val_loss:.4f}\n")
    f.write(f"Validation accuracy: {val_acc*100:.2f}%\n\n")
    f.write(val_report)

with open(ckpt_dir / "test_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_acc*100:.2f}%\n\n")
    f.write(test_report)

log.info(f"\nAll outputs saved to: {ckpt_dir}")
log.info(f"Best epoch    : {best_epoch}")
log.info(f"Best val acc  : {best_val_acc*100:.2f}%")
log.info(f"Final test acc: {test_acc*100:.2f}%")
