from pathlib import Path
import uuid

import joblib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from config import MODEL_PATH, UPLOAD_DIR
from predict import extract_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    uploaded_name = None

    if request.method == "POST":
        if not MODEL_PATH.exists():
            error = "Model not found. Train the model first with: python train.py"
            return render_template("index.html", error=error)

        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image."
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Allowed formats: JPG, JPEG, PNG, WEBP."
            return render_template("index.html", error=error)

        safe_name = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{safe_name}"
        destination = UPLOAD_DIR / filename
        file.save(destination)
        uploaded_name = filename

        try:
            model = joblib.load(MODEL_PATH)
            features = extract_features(Path(destination))
            predicted = int(model.predict(features)[0])
            prediction = "Cat" if predicted == 0 else "Dog"

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(features)[0][predicted])
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_name=uploaded_name,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
