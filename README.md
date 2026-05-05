# Tamil Handwritten Character Recognition

Repository name: `smai-project`

SMAI Assignment 3, Task T3.3. This project trains and evaluates a CNN for
Tamil handwritten character recognition using the uTHCD dataset, with a
Streamlit interface for drawing, uploading, practicing, and sampling test-set
examples.

## Current Status

As of 2026-05-05, the project includes source code, a completed retraining run
with a clean train/validation/test protocol, generated analysis artifacts, and
a working Streamlit app.

Current local run used by the report and app:

- Dataset:
  `../uTHCD_b(80-20-split)/80-20-split/uTHCD_8020_compressed.h5`
- Retraining run ID: `20260503_214617`
- Best checkpoint:
  `checkpoints_retrain/20260503_214617/ckpt_epoch020_best.pth`
- Validation accuracy: `98.29%`
- Test accuracy: `96.01%`
- Split sizes: `64,584` train / `7,176` validation / `19,190` test
- Normalization stats: mean `0.8013`, std `0.3990`
- Analysis outputs:
  `analysis_retrain/20260503_214617/`

Historical artifacts retained in the folder:

- `checkpoints/20260415_063836/ckpt_epoch020_best.pth`
- `analysis/`

These older artifacts come from the pre-retrain pipeline and are useful for
comparison, but they are not the final results cited in the current report.

## Project Files

- `train.py` - trains the CNN and writes timestamped checkpoints and reports.
- `analyse.py` - evaluates a trained checkpoint and saves diagnostic plots.
- `app.py` - Streamlit UI for drawing, uploading, and practicing Tamil
  character recognition.
- `download_data.py` - Kaggle dataset download helper.
- `requirements.txt` - Python dependencies.

The report-aligned outputs live under `checkpoints_retrain/` and
`analysis_retrain/`. Older artifacts under `checkpoints/` and `analysis/`
predate the corrected validation split.

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

To reproduce the final clean-split run used in the current report:

```bash
python train.py \
  --data ../uTHCD_b\(80-20-split\)/80-20-split/uTHCD_8020_compressed.h5 \
  --ckpt_dir checkpoints_retrain \
  --epochs 20 \
  --batch_size 128 \
  --val_split 0.1 \
  --seed 42 \
  --num_workers 0
```

Training writes outputs under:

```text
checkpoints/<run_id>/
```

If `--ckpt_dir checkpoints_retrain` is used, outputs are written under:

```text
checkpoints_retrain/<run_id>/
```

Each run includes checkpoint files, `history.json`, `metrics.json`,
`train.log`, `val_report.txt`, and `test_report.txt`.

## Analyze

The analysis script defaults to the retrained run used in the current report:

```bash
python analyse.py
```

It writes plots under:

```text
analysis_retrain/20260503_214617/
```

The default checkpoint and output directory can be overridden with
`--ckpt` and `--out`.

## Run the App

The intended command is:

```bash
streamlit run app.py
```

The app currently defaults to:

- checkpoint:
  `checkpoints_retrain/20260503_214617/ckpt_epoch020_best.pth`
- label map:
  `idx_to_class.json`

The interface supports:

- draw and predict
- practice mode with dataset-backed reference samples when the dataset is
  available
- upload-image inference
- a `Sample Demo` tab that runs the model on real held-out test examples

Predictions in the drawing and practice tabs are button-triggered via
`Predict` and `Check`; they are not streamed continuously while drawing.

If you retrain a new model, either:

- paste the new checkpoint path into the sidebar, or
- update `MODEL_PATH` in `app.py`.
