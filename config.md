# Model and Training Configuration

This document describes the configuration for the current trained model:

```text
checkpoints/20260415_063836/ckpt_epoch020_best.pth
```

The checkpoint was produced by `train.py` during run `20260415_063836`.

Equivalent training command:

```bash
python train.py \
  --data ../uTHCD_b\(80-20-split\)/80-20-split/uTHCD_8020_compressed.h5 \
  --epochs 20 \
  --batch_size 128 \
  --lr 0.001 \
  --img_size 32 \
  --ckpt_dir checkpoints \
  --save_every 5
```

## Dataset

- Dataset: uTHCD Tamil handwritten character dataset, 80-20 split.
- Data file:
  `../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5`
- Input arrays:
  - Train images: `Train Data/x_train`
  - Train labels: `Train Data/y_train`
  - Test images: `Test Data/x_test`
  - Test labels: `Test Data/y_test`
- Training samples: `71,760`
- Test samples: `19,190`
- Classes: `156`
- Original image size: `64 x 64`
- Model input size: `32 x 32`
- Channels: `1` grayscale channel
- Label format: integer class IDs from the HDF5 label arrays.
- Normalization:
  - Mean: `0.8013`
  - Standard deviation: `0.3991`
  - Statistics computed from the training images.

Evaluation protocol note: the script uses the HDF5 test split as the validation
loader during training and then evaluates the best checkpoint on that same test
split at the end. There is no separate validation split or unseen final holdout
in the current training script.

## Data Preprocessing

For each image:

1. Convert image values to `float32`.
2. Scale pixel values to `[0, 1]` by dividing by `255.0`.
3. Add channel dimension, producing shape `(1, H, W)`.
4. Resize to `32 x 32` using bilinear interpolation.
5. Normalize with the training-set mean and standard deviation.

## Data Augmentation

Augmentation is applied only to the training dataset.

- Random rotation: `-10` to `+10` degrees.
- Random horizontal translation: `-10%` to `+10%`.
- Random vertical translation: `-10%` to `+10%`.
- Random scale: `0.9` to `1.1`.
- Implementation: `torch.nn.functional.affine_grid` and `grid_sample`.

No augmentation is applied to the test dataset.

## Model Architecture

Model class: `TamilCNN`

Input shape:

```text
batch_size x 1 x 32 x 32
```

Feature extractor:

| Block | Layers | Output channels | Downsampling |
| --- | --- | ---: | --- |
| Block 1 | Conv2d, BatchNorm2d, ReLU, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d | 32 | `32 x 32` to `16 x 16` |
| Block 2 | Conv2d, BatchNorm2d, ReLU, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d | 64 | `16 x 16` to `8 x 8` |
| Block 3 | Conv2d, BatchNorm2d, ReLU, MaxPool2d | 128 | `8 x 8` to `4 x 4` |

Classifier:

| Layer | Configuration |
| --- | --- |
| Flatten | `128 * 4 * 4 = 2048` features |
| Dropout | `p = 0.4` |
| Linear | `2048 -> 512` |
| ReLU | In-place |
| Dropout | `p = 0.3` |
| Linear | `512 -> 156` |

Trainable parameters: `1,268,604`

Layer-by-layer definition:

| Module | Layer | Configuration |
| --- | --- | --- |
| `features.0` | Conv2d | `in_channels=1`, `out_channels=32`, `kernel_size=3`, `stride=1`, `padding=1` |
| `features.1` | BatchNorm2d | `num_features=32` |
| `features.2` | ReLU | `inplace=True` |
| `features.3` | Conv2d | `in_channels=32`, `out_channels=32`, `kernel_size=3`, `stride=1`, `padding=1` |
| `features.4` | BatchNorm2d | `num_features=32` |
| `features.5` | ReLU | `inplace=True` |
| `features.6` | MaxPool2d | `kernel_size=2` |
| `features.7` | Dropout2d | `p=0.1` |
| `features.8` | Conv2d | `in_channels=32`, `out_channels=64`, `kernel_size=3`, `stride=1`, `padding=1` |
| `features.9` | BatchNorm2d | `num_features=64` |
| `features.10` | ReLU | `inplace=True` |
| `features.11` | Conv2d | `in_channels=64`, `out_channels=64`, `kernel_size=3`, `stride=1`, `padding=1` |
| `features.12` | BatchNorm2d | `num_features=64` |
| `features.13` | ReLU | `inplace=True` |
| `features.14` | MaxPool2d | `kernel_size=2` |
| `features.15` | Dropout2d | `p=0.1` |
| `features.16` | Conv2d | `in_channels=64`, `out_channels=128`, `kernel_size=3`, `stride=1`, `padding=1` |
| `features.17` | BatchNorm2d | `num_features=128` |
| `features.18` | ReLU | `inplace=True` |
| `features.19` | MaxPool2d | `kernel_size=2` |
| `classifier.0` | Dropout | `p=0.4` |
| `classifier.1` | Linear | `in_features=2048`, `out_features=512` |
| `classifier.2` | ReLU | `inplace=True` |
| `classifier.3` | Dropout | `p=0.3` |
| `classifier.4` | Linear | `in_features=512`, `out_features=156` |

## Training Configuration

- Device used in the completed run: `cuda`
- Epochs: `20`
- Batch size: `128`
- Initial learning rate: `0.001`
- Checkpoint root: `checkpoints`
- Run checkpoint directory: `checkpoints/20260415_063836`
- Periodic checkpoint interval: every `5` epochs
- Best-checkpoint policy: save whenever validation accuracy improves.
- Random seed: not fixed in the current script.
- Determinism: not enforced; augmentation sampling, data-loader workers, and
  CUDA kernels may make reruns differ slightly.
- Gradient clipping: not used.
- Early stopping: not used.

Loss:

```text
CrossEntropyLoss(label_smoothing=0.1)
```

Optimizer:

```text
AdamW(lr=0.001, weight_decay=0.0001)
```

Learning-rate scheduler:

```text
CosineAnnealingLR(T_max=20)
```

Scheduler details from the checkpoint:

- Base learning rate: `0.001`
- Minimum learning rate: PyTorch default `eta_min=0`
- Last epoch in scheduler state: `20`
- Final recorded learning rate: `0.0`

DataLoader settings:

- Training loader:
  - `shuffle=True`
  - `num_workers=4`
  - `pin_memory=True`
- Test loader:
  - `shuffle=False`
  - `num_workers=4`
  - `pin_memory=True`

## Training Result

Best checkpoint:

```text
checkpoints/20260415_063836/ckpt_epoch020_best.pth
```

Best epoch: `20`

Final epoch metrics:

| Metric | Value |
| --- | ---: |
| Train loss | `1.0548` |
| Train accuracy | `97.70%` |
| Validation loss | `1.0132` |
| Validation accuracy | `96.39%` |
| Learning rate | `0.00e+00` |

Final evaluation:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | `96.39%` |
| Test accuracy | `96.39%` |
| Macro precision | `0.964` |
| Macro recall | `0.964` |
| Macro F1-score | `0.964` |
| Weighted precision | `0.964` |
| Weighted recall | `0.964` |
| Weighted F1-score | `0.964` |

Selected training-history checkpoints:

| Epoch | Train loss | Train acc | Val loss | Val acc | LR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `2.1949` | `65.16%` | `1.4300` | `86.33%` | `9.94e-04` |
| 10 | `1.1193` | `96.45%` | `1.0539` | `95.68%` | `5.00e-04` |
| 20 | `1.0548` | `97.70%` | `1.0132` | `96.39%` | `0.00e+00` |

## Checkpoint Format

The tracked checkpoint is a PyTorch dictionary with these top-level keys:

| Key | Contents |
| --- | --- |
| `epoch` | Best epoch number, `20` |
| `val_acc` | Best validation accuracy, `0.9639395518499219` |
| `model` | Model `state_dict` with `39` entries |
| `optimizer` | AdamW optimizer state |
| `scheduler` | CosineAnnealingLR scheduler state |
| `args` | Training arguments saved from `argparse` |

Saved checkpoint arguments:

```json
{
  "data": "../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5",
  "epochs": 20,
  "batch_size": 128,
  "lr": 0.001,
  "img_size": 32,
  "ckpt_dir": "checkpoints",
  "save_every": 5
}
```

## Saved Artifacts

Tracked in Git:

- `checkpoints/20260415_063836/ckpt_epoch020_best.pth`

Generated locally but ignored by Git:

- Intermediate checkpoints from the same run.
- `checkpoints/20260415_063836/history.json`
- `checkpoints/20260415_063836/test_report.txt`
- `checkpoints/20260415_063836/train.log`
- `analysis/`
- `run.log`

## Current App Compatibility Note

The current Streamlit app in `app.py` does not yet load this checkpoint directly.
The trained checkpoint stores a dictionary with the model state under the
`model` key, and the app's local `TamilCNN` definition does not match the
architecture used in `train.py`. The app should be updated to use the same
`TamilCNN` architecture and to load:

```python
ckpt = torch.load("checkpoints/20260415_063836/ckpt_epoch020_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
```
