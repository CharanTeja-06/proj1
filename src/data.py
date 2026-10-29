import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "mpl")

import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT / "Dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Single authoritative class list — identical between training, evaluation, and inference
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASSES)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image_file(path: Path) -> bool:
    """
    Ultra-fast magic byte header validation.
    Detects real image headers (JPEG, PNG, BMP, WEBP) and filters out corrupted or
    accidental non-image files (such as renamed python scripts) in microseconds.
    """
    try:
        if path.stat().st_size < 32:
            return False
        with open(path, "rb") as f:
            header = f.read(16)
        if header.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        if header.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        if header.startswith(b"BM"):  # BMP
            return True
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":  # WEBP
            return True
        return False
    except Exception:
        return False


def preprocess_input(image):
    """
    MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1].
    Identical function used across training, evaluation, and webcam inference
    to guarantee zero train/inference distribution mismatch.
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def get_valid_images(split):
    """
    Scans image paths and numeric labels for the given split (Training, Validation, Testing).
    Uses ultra-fast directory traversal and binary header verification.
    """
    records = []
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Folder not found: {split_dir}")

    for label, letter in enumerate(CLASSES):
        class_dir = split_dir / letter
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in class_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if not is_valid_image_file(path):
                print(f"Skipping non-image file: {path.name} in {class_dir.name}")
                continue
            records.append((str(path), label))

    if not records:
        raise RuntimeError(f"No valid images found in {split_dir}")

    print(f"{split}: {len(records)} valid images across {NUM_CLASSES} classes")
    return records


def load_raw_image(path, label):
    """
    Reads, decodes, and resizes raw image to float32 [0, 255].
    """
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_augmentation():
    """
    Targeted augmentation for ISL gestures (specifically improving S, Z, etc.):
    - Small rotation (±20 deg) to handle slight tilt without altering gesture meaning
    - Scale/zoom variation (±10%) to simulate varying hand distance to camera
    - Small translation (±5%) to simulate hand placement variations
    - Brightness and contrast shifts (±15%) on [0, 255] raw pixel values
    * Horizontal flip is deliberately EXCLUDED because ISL signs are chirality-sensitive.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom((-0.1, 0.1)),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.15),
    ], name="isl_augmentation")


def make_dataset(split, training=False, batch_size=BATCH_SIZE):
    """
    Builds an optimized tf.data.Dataset pipeline:
    1. Reads raw image in [0, 255]
    2. Applies geometric & photometric augmentation (training only)
    3. Preprocesses into MobileNetV2 range [-1, 1]
    4. Batches and prefetches with AUTOTUNE
    """
    records = get_valid_images(split)
    paths, labels = zip(*records)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        ds = ds.shuffle(min(len(records), 5000), reshuffle_each_iteration=True)

    ds = ds.map(load_raw_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmentation = make_augmentation()
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Normalize to [-1, 1] AFTER augmentation
    ds = ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    for split in ["Training", "Validation", "Testing"]:
        records = get_valid_images(split)
        print(f"Validated {split}: {len(records)} images.")
