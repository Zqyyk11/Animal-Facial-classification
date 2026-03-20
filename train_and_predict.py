"""
Pet facial-expression classifier using a 5-fold stratified ensemble.

Predicts three labels from face images: `Angry`, `Happy`, `Sad`.

Pipeline:
    1. Load and resize all train/test images once (`preload_images`).
    2. For each fold: train EfficientNetB2 with a frozen backbone and a small
       MLP head; stop early when `val_loss` plateaus (`EarlyStopping`).
    3. For each fold: run test-time augmentation (TTA) on the test set and
       average softmax probabilities.
    4. Average probabilities across folds, argmax to labels, write
       `submission.csv` (columns `id`, `label`).

Techniques: ImageNet-pretrained EfficientNetB2, in-model train augmentation,
AdamW + label smoothing (sparse labels), `ReduceLROnPlateau`,
optional mixed precision (`USE_MIXED_PRECISION`).

Run from project root: `python train_and_predict.py` or `python3 ...`.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
layers = keras.layers

# --- Paths (run from project root) ---
TRAIN_DIR = "train/train"
TEST_DIR = "test/test"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

# Tunable training / IO settings (reduce IMG_SIZE or BATCH_SIZE if OOM or OS kill).
IMG_SIZE = (260, 260)
BATCH_SIZE = 16
SEED = 44
N_FOLDS = 5

# Upper bound on epochs per fold; EarlyStopping usually ends sooner.
EPOCHS = 40

# Slight smoothing helps with subjective/ambiguous labels, but not too much.
LABEL_SMOOTHING = 0.03

USE_MIXED_PRECISION = True


def configure_mixed_precision():
    """
    Enable TensorFlow mixed precision.

    Mixed precision can speed up training and reduce memory usage on supported
    GPUs. If mixed precision cannot be enabled, the script continues normally
    in float32 mode.
    """
    if not USE_MIXED_PRECISION:
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: enabled (mixed_float16)")
    except Exception:
        print("Mixed precision: could not enable; continuing normally.")


def build_augmentation():
    """
    Build the train-time augmentation pipeline.

    The augmentation layers live inside the model so they are automatically
    disabled during inference when the model is called with `training=False`.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name="augmentation",
    )


def build_model(num_classes: int):
    """
    Build the EfficientNetB2-based classifier model.

    Model structure:
    - EfficientNetB2 backbone (frozen by default).
    - Global average pooling.
    - A small MLP head with BatchNorm + Dropout regularization.

    Notes:
    - `efficientnet.preprocess_input` is applied after augmentation.
    - The final softmax is forced to float32 for numeric stability under mixed
      precision.
    """
    base = keras.applications.EfficientNetB2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = build_augmentation()(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    # Ensure numeric dtype stays float32 for stability/softmax under mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs)


def make_sparse_label_smoothing_loss(num_classes: int, smoothing: float):
    """
    Create label smoothing loss for sparse integer labels.

    This script implements smoothing manually to stay compatible with older
    Keras/TensorFlow versions that may not support `label_smoothing` directly
    for sparse label losses.
    """

    def loss(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y, depth=num_classes, dtype=tf.float32)
        k = tf.cast(num_classes, tf.float32)
        s = tf.cast(smoothing, tf.float32)
        targets = one_hot * (1.0 - s) + (1.0 - one_hot) * (s / tf.maximum(k - 1.0, 1.0))
        return tf.reduce_mean(keras.losses.categorical_crossentropy(targets, y_pred))

    return loss


def make_callbacks(patience_es: int):
    """
    Create callbacks used during training.

    Includes:
    - ReduceLROnPlateau: lowers the learning rate when `val_loss` stops improving.
    - EarlyStopping: stops training early and restores the best weights.
    """
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.45,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def collect_paths_and_labels():
    """
    Collect training image paths and integer labels.

    Class index ordering is alphabetical by folder name inside `TRAIN_DIR`,
    ensuring consistent mapping between class names and model output indices.
    """
    class_names = sorted(
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )
    paths, labels = [], []
    for ci, name in enumerate(class_names):
        folder = os.path.join(TRAIN_DIR, name)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                paths.append(os.path.join(folder, f))
                labels.append(ci)
    return class_names, np.array(paths), np.array(labels, dtype=np.int64)


def preload_images(paths):
    """
    Load and resize all images to `IMG_SIZE` once.

    Preloading avoids decoding from disk on every cross-validation fold.

    Args:
        paths: Iterable of filesystem paths to image files.

    Returns:
        `uint8` array of shape `(N, H, W, 3)` with pixels in ~[0, 255].
    """
    imgs = []
    for p in paths:
        img = keras.utils.load_img(p, target_size=IMG_SIZE)
        arr = keras.utils.img_to_array(img).astype(np.uint8)  # 0..255
        imgs.append(arr)
    return np.stack(imgs)


def make_tf_dataset_from_arrays(images_arr_uint8, labels_arr, shuffle, seed):
    """
    Create a `tf.data.Dataset` from preloaded numpy arrays.

    Args:
        images_arr_uint8: uint8 images in range ~[0, 255].
        labels_arr: integer labels.
        shuffle: whether to shuffle the dataset.
        seed: random seed for shuffling.

    Returns:
        A dataset that yields `(float32_images, int32_labels)` batched by `BATCH_SIZE`.
    """
    images_arr_uint8 = tf.convert_to_tensor(images_arr_uint8)
    labels_arr = tf.convert_to_tensor(labels_arr)
    ds = tf.data.Dataset.from_tensor_slices((images_arr_uint8, labels_arr))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images_arr_uint8), seed=seed)

    def cast_fn(x, y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.int32)

    ds = ds.map(cast_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def tta_predict_probs(model, images_uint8):
    """
    Predict class probabilities using test-time augmentation (TTA).

    For each image, creates multiple fixed variants and averages the softmax
    probabilities to reduce sensitivity to:
    - horizontal mirroring
    - simple brightness changes

    Args:
        model: trained Keras model.
        images_uint8: input images as uint8 arrays.

    Returns:
        NumPy array of shape `(N, num_classes)` with averaged softmax probs.
    """
    # Important: convert to float32 BEFORE brightness scaling, otherwise
    # expressions like `images * 1.07` crash because `images` is uint8.
    images = tf.cast(tf.convert_to_tensor(images_uint8), tf.float32)
    variants = []
    variants.append(images)  # original
    variants.append(tf.reverse(images, axis=[2]))  # horizontal flip (width axis=2)
    variants.append(tf.clip_by_value(images * 1.07, 0.0, 255.0))  # brighter
    variants.append(tf.clip_by_value(images * 0.93, 0.0, 255.0))  # darker

    accum = None
    for v in variants:
        p = model(v, training=False)
        if hasattr(p, "numpy"):
            p = p.numpy()
        accum = p if accum is None else accum + p
    return accum / float(len(variants))


def train_one_fold(fold_idx, x_all_uint8, y_all, train_idx, val_idx, num_classes):
    """
    Train one fold: frozen EfficientNetB2 + trainable classification head.

    Uses `AdamW`, label smoothing, `ReduceLROnPlateau`, and `EarlyStopping`
    (`restore_best_weights=True`). Fold RNG is offset from `SEED` for diversity.

    Args:
        fold_idx: Zero-based fold index (used for RNG offsets).
        x_all_uint8: Full training images `(N, H, W, 3)` uint8.
        y_all: Integer labels `(N,)`.
        train_idx, val_idx: Indices for this split.
        num_classes: Number of classes (3).

    Returns:
        Fitted `keras.Model` for this fold.
    """
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + fold_idx * 17)

    x_train = x_all_uint8[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all_uint8[val_idx]
    y_val = y_all[val_idx]

    train_ds = make_tf_dataset_from_arrays(
        x_train, y_train, shuffle=True, seed=SEED + fold_idx * 3
    )
    val_ds = make_tf_dataset_from_arrays(
        x_val, y_val, shuffle=False, seed=SEED + fold_idx * 3
    )

    train_loss = make_sparse_label_smoothing_loss(num_classes, LABEL_SMOOTHING)
    model = build_model(num_classes)

    base_lr = 1e-3
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=base_lr, weight_decay=1e-4, clipnorm=1.0
        ),
        loss=train_loss,
        metrics=["accuracy"],
    )

    print(
        f"\n--- Fold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)} val={len(val_idx)} ---"
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(patience_es=15),
        verbose=1,
    )

    return model


def main():
    """
    Run `N_FOLDS` stratified CV, ensemble TTA test probabilities, save submission.

    Reads `sample_submission.csv` for test `id` order and paths under
    `TEST_DIR`. Writes `SUBMISSION_PATH` (default `submission.csv`).
    """
    configure_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)

    class_names, paths, labels = collect_paths_and_labels()
    num_classes = len(class_names)
    n_samples = len(paths)
    print(f"Classes: {class_names} | Total images: {n_samples}")

    # Preload train images once for speed.
    print("Preloading train images (once)...")
    x_all_uint8 = preload_images(paths)

    # Load test image ids in submission order.
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    ids = submission["id"].tolist()

    print("Preloading test images (once)...")
    test_imgs_uint8 = []
    for img_id in ids:
        path = os.path.join(TEST_DIR, img_id)
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        test_imgs_uint8.append(keras.utils.img_to_array(img).astype(np.uint8))
    x_test_uint8 = np.stack(test_imgs_uint8)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ensemble_probs = np.zeros((len(ids), num_classes), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(n_samples), labels)
    ):
        model = train_one_fold(
            fold_idx,
            x_all_uint8,
            labels,
            train_idx=train_idx,
            val_idx=val_idx,
            num_classes=num_classes,
        )
        fold_probs = tta_predict_probs(model, x_test_uint8)
        ensemble_probs += fold_probs.astype(np.float32)
        keras.backend.clear_session()

    ensemble_probs /= float(N_FOLDS)
    pred_indices = np.argmax(ensemble_probs, axis=1)
    pred_labels = [class_names[i] for i in pred_indices]

    out = pd.DataFrame({"id": ids, "label": pred_labels})
    out.to_csv(SUBMISSION_PATH, index=False, quoting=1)
    print(f"\nSaved 5-fold ensemble predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
