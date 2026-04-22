# Model and Training Configuration

This document describes the configuration for the current trained model:

```text
checkpoints/20260415_063836/ckpt_epoch020_best.pth
```

The checkpoint was produced by `train.py` during run `20260415_063836`.

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
- Normalization:
  - Mean: `0.8013`
  - Standard deviation: `0.3991`
  - Statistics computed from the training images.

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
