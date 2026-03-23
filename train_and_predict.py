"""
Pet face emotions: predict Angry, Happy, or Sad from a photo.

What this program does (big picture):
  1. Load all training and test pictures into memory once.
  2. Split training data into 5 parts (folds). Each part takes a turn being the
     "quiz set" while the other 4 parts are used to learn. Class counts stay
     balanced in each fold (stratified).
  3. For each fold, train a neural network: a frozen EfficientNet (pre-trained
     on ImageNet) + a small "head" that learns our 3 classes.
  4. For each test image, average predictions from 4 slightly different views
     (original, mirror, brighter, darker), then average across the 5 folds.
  5. Save labels to submission.csv for Kaggle.

Run from the project folder:  python3 train_and_predict.py
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

# Keras layers (short name so the model code reads like a recipe)
layers = keras.layers

# ---------------------------------------------------------------------------
# Folders and files (paths are relative to where you run the script)
# ---------------------------------------------------------------------------
TRAIN_DIR = "train/train"
TEST_DIR = "test/test"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

# ---------------------------------------------------------------------------
# Settings you can change (if the computer runs out of memory, lower BATCH_SIZE
# or the image height/width in IMG_SIZE)
# ---------------------------------------------------------------------------
IMG_SIZE = (260, 260)  # every image is resized to this width x height
BATCH_SIZE = 16  # how many images the model sees at once during training
SEED = 44  # random seed so splits and training are repeatable
N_FOLDS = 5  # how many models we train, then we average their answers

EPOCHS = 40  # max training rounds per fold (often stops earlier automatically)

# Label smoothing: don't force the model to be 100% sure on every example;
# helps when some labels might be a bit wrong or fuzzy.
LABEL_SMOOTHING = 0.03

# Faster math on many GPUs (safe to leave True; if it fails, we fall back)
USE_MIXED_PRECISION = True

# Test-time augmentation: multiply pixel brightness (must use float images, not uint8)
TTA_BRIGHTER = 1.07
TTA_DARKER = 0.93


def configure_mixed_precision():
    """Turn on float16 training where supported (faster, less VRAM)."""
    if not USE_MIXED_PRECISION:
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: enabled (mixed_float16)")
    except Exception:
        print("Mixed precision: could not enable; continuing normally.")


def build_augmentation():
    """
    Random image tweaks used ONLY while training (flip, rotate, zoom, etc.).
    They sit inside the model so when we call the model for testing, we set
    training=False and these random layers do nothing.
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
    Build the network:
      photo -> augment (train only) -> EfficientNet sizing -> EfficientNetB2 (frozen)
      -> pool -> two small dense layers -> 3-class softmax.
    """
    backbone = keras.applications.EfficientNetB2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = False  # we only train the "head" below

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = build_augmentation()(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    # float32 here keeps softmax stable when the rest uses float16
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs)


def make_sparse_label_smoothing_loss(num_classes: int, smoothing: float):
    """
    Loss function with "label smoothing": instead of target = [1,0,0] for class 0,
    we use slightly softer targets so the model is not pushed to overconfidence.
    (Written by hand so older Keras versions still work.)
    """

    def loss(y_true, y_pred):
        # True labels come in as integers 0, 1, 2; turn them into one-hot vectors
        y = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.one_hot(y, depth=num_classes, dtype=tf.float32)
        k = tf.cast(num_classes, tf.float32)
        s = tf.cast(smoothing, tf.float32)
        # Spread a little probability mass onto the wrong classes
        smoothed = one_hot * (1.0 - s) + (1.0 - one_hot) * (
            s / tf.maximum(k - 1.0, 1.0)
        )
        return tf.reduce_mean(
            keras.losses.categorical_crossentropy(smoothed, y_pred)
        )

    return loss


def make_callbacks(patience_es: int):
    """
    Two helpers during training:
      - If validation loss stops improving, shrink the learning rate.
      - If it still doesn't improve for many epochs, stop and reload the best weights.
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
    Walk train/train/<ClassName>/ and collect every image path + its class index.
    Folder names are sorted alphabetically, so indices are always:
      Angry=0, Happy=1, Sad=2 (for default folder names).
    """
    class_names = sorted(
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )
    paths, labels = [], []
    for class_index, name in enumerate(class_names):
        folder = os.path.join(TRAIN_DIR, name)
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                paths.append(os.path.join(folder, filename))
                labels.append(class_index)
    return class_names, np.array(paths), np.array(labels, dtype=np.int64)


def preload_images(paths):
    """Load every image from disk once and resize to IMG_SIZE. Returns uint8 array."""
    stacked = []
    for path in paths:
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = keras.utils.img_to_array(img).astype(np.uint8)
        stacked.append(arr)
    return np.stack(stacked)


def make_tf_dataset_from_arrays(images_arr_uint8, labels_arr, shuffle, seed):
    """Wrap numpy arrays in a TensorFlow dataset: shuffle, cast, batch, prefetch."""
    images_tensor = tf.convert_to_tensor(images_arr_uint8)
    labels_tensor = tf.convert_to_tensor(labels_arr)
    ds = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images_tensor), seed=seed)

    def to_float_images_and_int_labels(image, label):
        return tf.cast(image, tf.float32), tf.cast(label, tf.int32)

    ds = ds.map(to_float_images_and_int_labels, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def tta_predict_probs(model, images_uint8):
    """
    For each test image, run the model on 4 versions and average the probability
    vectors. Then we average those across folds in main().
    """
    # uint8 * 1.07 would error in TensorFlow; use float32 first
    images = tf.cast(tf.convert_to_tensor(images_uint8), tf.float32)

    version_0_original = images
    version_1_mirrored = tf.reverse(images, axis=[2])  # flip left-right
    version_2_brighter = tf.clip_by_value(
        images * TTA_BRIGHTER, 0.0, 255.0
    )
    version_3_darker = tf.clip_by_value(images * TTA_DARKER, 0.0, 255.0)

    versions = [
        version_0_original,
        version_1_mirrored,
        version_2_brighter,
        version_3_darker,
    ]

    total_probs = None
    for version in versions:
        probs = model(version, training=False)
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        total_probs = probs if total_probs is None else total_probs + probs

    return total_probs / float(len(versions))


def train_one_fold(fold_idx, x_all_uint8, y_all, train_idx, val_idx, num_classes):
    """Train on train_idx, validate on val_idx; return the model for this fold."""
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

    loss_fn = make_sparse_label_smoothing_loss(num_classes, LABEL_SMOOTHING)
    model = build_model(num_classes)

    learning_rate = 1e-3
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0
        ),
        loss=loss_fn,
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
    configure_mixed_precision()
    tf.keras.utils.set_random_seed(SEED)

    class_names, paths, labels = collect_paths_and_labels()
    num_classes = len(class_names)
    n_samples = len(paths)
    print(f"Classes: {class_names} | Total images: {n_samples}")

    print("Preloading train images (once)...")
    x_all_uint8 = preload_images(paths)

    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_ids = submission["id"].tolist()

    print("Preloading test images (once)...")
    test_images_list = []
    for image_id in test_ids:
        full_path = os.path.join(TEST_DIR, image_id)
        img = keras.utils.load_img(full_path, target_size=IMG_SIZE)
        test_images_list.append(keras.utils.img_to_array(img).astype(np.uint8))
    x_test_uint8 = np.stack(test_images_list)

    kfold_splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Sum of probabilities from each fold; divide by N_FOLDS at the end
    sum_of_probs = np.zeros((len(test_ids), num_classes), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold_splitter.split(np.arange(n_samples), labels)
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
        sum_of_probs += fold_probs.astype(np.float32)
        keras.backend.clear_session()

    average_probs = sum_of_probs / float(N_FOLDS)
    predicted_class_index = np.argmax(average_probs, axis=1)
    predicted_labels = [class_names[i] for i in predicted_class_index]

    result_table = pd.DataFrame({"id": test_ids, "label": predicted_labels})
    result_table.to_csv(SUBMISSION_PATH, index=False, quoting=1)
    print(f"\nSaved 5-fold ensemble predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
