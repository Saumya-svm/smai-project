# SMAI Project

Repository name: `smai-project`

SMAI Assignment 3, Task T3.3. This project trains and evaluates a CNN for
Tamil handwritten character recognition using the uTHCD dataset, with a
Streamlit interface for drawing, uploading, and practicing characters.

## Current Status

As of 2026-05-03, the project has source code, a completed local training run,
generated analysis artifacts, and a working Streamlit app.

Completed:

- Local Git repository initialized on branch `main`.
- Dataset found locally at:
  `../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5`
- Training completed for run `20260415_063836`.
- Best checkpoint produced:
  `checkpoints/20260415_063836/ckpt_epoch020_best.pth`
- The best checkpoint is tracked in Git; intermediate checkpoints remain
  ignored locally.
- Historical accuracy of the tracked checkpoint: `96.39%` on `19,190` test
  samples across `156` classes.
- Analysis outputs generated in `analysis/`, including confusion matrices,
  correct/wrong prediction grids, and per-class accuracy plots.

Known issues:

- The tracked checkpoint was produced before the training script was updated to
  use a separate validation split, so its reported `96.39%` comes from the old
  training pipeline. Retrain with the current `train.py` for clean validation
  and test metrics.
- `label_map.json` is a legacy mapping that does not match the tracked
  checkpoint. Use `idx_to_class.json`.

## Project Files

- `train.py` - trains the CNN and writes timestamped checkpoints and reports.
- `analyse.py` - evaluates a trained checkpoint and saves diagnostic plots.
- `app.py` - Streamlit UI for drawing, uploading, and practicing Tamil
  character recognition.
- `download_data.py` - Kaggle dataset download helper.
- `requirements.txt` - Python dependencies.

Generated files such as logs, analysis plots, local datasets, intermediate
checkpoints, and virtual environments are intentionally ignored by Git. The
best checkpoint from the completed training run is tracked so the trained model
artifact is available with the repository.

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

Each run includes checkpoint files, `history.json`, `metrics.json`, `train.log`,
`val_report.txt`, and `test_report.txt`.

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

The app now loads the tracked checkpoint directly. The sidebar defaults to the
correct label map (`idx_to_class.json`), and the app includes a `Sample Demo`
tab that runs the model on real test-set examples.

If you retrain a new model, either:

- paste the new checkpoint path into the sidebar, or
- update `MODEL_PATH` in `app.py`.
