import argparse
import json
import re
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    DEFAULT_MAX_PER_CLASS,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    REPORT_PATH,
    TEST_SIZE,
    TRAIN_DIR,
)


def label_from_filename(path: Path):
    match = re.match(r"^(cat|dog)\.", path.name.lower())
    if not match:
        return None
    return 0 if match.group(1) == "cat" else 1


def extract_features(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
    )

    return features.astype(np.float32)


def collect_files(max_per_class: int):
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"{TRAIN_DIR} does not exist. Run 'python download_data.py' first."
        )

    cat_files = sorted(TRAIN_DIR.glob("cat.*.jpg"))[:max_per_class]
    dog_files = sorted(TRAIN_DIR.glob("dog.*.jpg"))[:max_per_class]

    if not cat_files or not dog_files:
        raise RuntimeError("Could not find both cat and dog images.")

    files = cat_files + dog_files
    print(f"Using {len(cat_files)} cat images and {len(dog_files)} dog images.")
    return files


def build_dataset(files):
    X, y = [], []
    skipped = 0

    for index, path in enumerate(files, start=1):
        feature = extract_features(path)
        label = label_from_filename(path)

        if feature is None or label is None:
            skipped += 1
            continue

        X.append(feature)
        y.append(label)

        if index % 500 == 0:
            print(f"Processed {index}/{len(files)} images...")

    if not X:
        raise RuntimeError("No valid images were processed.")

    return np.asarray(X), np.asarray(y), skipped


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
    )
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train an SVM for Cats vs Dogs.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help="Maximum number of images used from each class.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.max_per_class)

    print("Extracting HOG features...")
    X, y, skipped = build_dataset(files)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Skipped images: {skipped}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training RBF SVM...")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["cat", "dog"],
        output_dict=True,
        zero_division=0,
    )

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["cat", "dog"],
        zero_division=0,
    ))

    joblib.dump(pipeline, MODEL_PATH)
    save_confusion_matrix(y_test, y_pred)

    report_data = {
        "model": "SVC",
        "kernel": "rbf",
        "feature_extractor": "HOG",
        "image_size": list(IMAGE_SIZE),
        "images_per_class": args.max_per_class,
        "total_images_processed": int(len(X)),
        "skipped_images": int(skipped),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy),
        "classification_report": report,
        "classes": ["cat", "dog"],
    }

    REPORT_PATH.write_text(
        json.dumps(report_data, indent=4),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
