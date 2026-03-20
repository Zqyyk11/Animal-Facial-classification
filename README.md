# Pet facial expression classification

Pipeline: **one script** (`train_and_predict.py`) builds a 3-class classifier for **`Angry`**, **`Happy`**, **`Sad`** pet face images and writes **`submission.csv`**.

## What’s inside

| Piece | Role |
|--------|------|
| **EfficientNetB2** (ImageNet) | Frozen backbone; only the head trains |
| **Train-time augmentation** | Random flip, rotation, zoom, translation, contrast, brightness (inside the model) |
| **Label smoothing** | Custom loss for sparse integer labels (older Keras-friendly) |
| **Stratified K-fold** | Default `N_FOLDS = 5`; probabilities averaged across folds |
| **TTA** | Original + h-flip + brighter + darker views; average softmax |
| **Callbacks** | `ReduceLROnPlateau` on `val_loss`, `EarlyStopping` with best weights restored |

Optional **mixed precision** (`USE_MIXED_PRECISION`) speeds training on supported GPUs.

## Repository layout

```text
train_and_predict.py   # train + predict + write submission
requirements.txt
sample_submission.csv  # defines test id order + column format

train/train/
  Angry/
  Happy/
  Sad/

test/test/
  <image_id>.jpg       # filenames match sample_submission id column
```

## Setup

Python 3.10+ recommended. Create a venv, then:

```bash
pip install -r requirements.txt
```

## Run

From the **project root** (so paths `train/train`, `test/test` resolve):

```bash
python3 train_and_predict.py
```

Output: **`submission.csv`** with columns `id`, `label` (quoted CSV to match typical Kaggle samples).

## Tuning (see constants at top of `train_and_predict.py`)

| Constant | Typical effect |
|----------|----------------|
| `IMG_SIZE` | Larger → better detail, more RAM/time |
| `BATCH_SIZE` | Larger → faster epochs if memory allows |
| `EPOCHS` | Max epochs per fold; **EarlyStopping** often finishes earlier |
| `N_FOLDS` | More folds → stronger ensemble, longer total runtime |
| `SEED` | Reproducibility + fold splits |
| `LABEL_SMOOTHING` | Lower or `0` if labels are very clean |
| `USE_MIXED_PRECISION` | On for GPU; harmless fallback if unavailable |

**Early stopping patience** is passed in `train_one_fold()` (`make_callbacks(patience_es=...)`). Increase patience if validation improves late; decrease for faster experiments.

If the process is **killed** or RAM spikes, lower `BATCH_SIZE` and/or `IMG_SIZE` first.

## Notes

- Labels can be noisy or subjective; augmentation + smoothing + ensembling aim to reduce overfitting to spurious cues.
- The leaderboard metric is usually **accuracy** on held-out test labels; this repo does not ship private labels.

## References (tools / building blocks)

- [TensorFlow / Keras](https://www.tensorflow.org/)
- [EfficientNet (Keras Applications)](https://keras.io/api/applications/)
- [scikit-learn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
