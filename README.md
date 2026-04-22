# SMAI Project

Repository name: `smai-project`

SMAI Assignment 3, Task T3.3. This project trains and evaluates a CNN for
Tamil handwritten character recognition using the uTHCD dataset, with a
Streamlit interface for drawing, uploading, and practicing characters.

## Current Status

As of 2026-04-22, the project has source code, a completed local training run,
and generated analysis artifacts. The Streamlit app still needs model-loading
wiring before it is ready to run end to end.

Completed:

- Local Git repository initialized on branch `main`.
- Dataset found locally at:
  `../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5`
- Training completed for run `20260415_063836`.
- Best checkpoint produced:
  `checkpoints/20260415_063836/ckpt_epoch020_best.pth`
- Best validation accuracy: `96.39%`
- Test accuracy: `96.39%` on `19,190` test samples across `156` classes.
- Analysis outputs generated in `analysis/`, including confusion matrices,
  correct/wrong prediction grids, and per-class accuracy plots.

Known issues:

- `app.py` currently expects `tamil_cnn.pth` and `label_map.json` in the project
  root, but those files are not present.
- The app's `TamilCNN` class does not match the architecture saved in the
  trained checkpoint, so the checkpoint cannot be loaded by the app without
  updating the app model definition or exporting a compatible app checkpoint.
- `README` instructions were updated to describe the current state, but the app
  still needs a small follow-up fix before `streamlit run app.py` will work.

## Project Files

- `train.py` - trains the CNN and writes timestamped checkpoints and reports.
- `analyse.py` - evaluates a trained checkpoint and saves diagnostic plots.
- `app.py` - Streamlit UI for drawing, uploading, and practicing Tamil
  character recognition.
- `download_data.py` - Kaggle dataset download helper.
- `requirements.txt` - Python dependencies.

Generated files such as checkpoints, logs, analysis plots, local datasets, and
virtual environments are intentionally ignored by Git.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If the dataset is not already available, place `kaggle.json` in `~/.kaggle/`
and run:

```bash
python download_data.py
```

The training script currently defaults to:

```text
../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5
```

Use `--data` if your dataset path differs.

## Train

```bash
python train.py
```

Useful options:

```bash
python train.py --data /path/to/uTHCD_8020_compressed.h5 --epochs 20 --batch_size 128
```

Training writes outputs under:

```text
checkpoints/<run_id>/
```

Each run includes checkpoint files, `history.json`, `train.log`, and
`test_report.txt`.

## Analyze

The analysis script is currently pinned to the completed local run:

```bash
python analyse.py
```

It writes plots under:

```text
analysis/
```

## Run the App

The intended command is:

```bash
streamlit run app.py
```

Current caveat: the app needs to be updated to load the trained checkpoint format
from `train.py`, or a compatible `tamil_cnn.pth` plus `label_map.json` must be
exported first.
