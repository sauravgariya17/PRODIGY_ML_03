import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from config import (
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
    MODEL_PATH,
)


def extract_features(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
    )

    return features.reshape(1, -1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a cat/dog image.")
    args = parser.parse_args()

    image_path = Path(args.image)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    features = extract_features(image_path)

    prediction = int(model.predict(features)[0])
    label = "cat" if prediction == 0 else "dog"

    print(f"Prediction: {label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        print(f"Confidence: {confidence:.2%}")
    else:
        print("Confidence: not available")


if __name__ == "__main__":
    main()
