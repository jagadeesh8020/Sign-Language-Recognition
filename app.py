import base64
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from slr.predictor import SignLanguagePredictor


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "8")) * 1024 * 1024

predictor = SignLanguagePredictor()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/assets/<path:filename>")
def project_asset(filename):
    if filename.startswith("docs/"):
        return send_from_directory(BASE_DIR / "docs", filename.removeprefix("docs/"))
    if filename.startswith("resources/"):
        return send_from_directory(BASE_DIR / "resources", filename.removeprefix("resources/"))

    return jsonify({"error": "Asset not found"}), 404


@app.post("/predict")
def predict():
    image_bytes = _read_image_from_request()
    if not image_bytes:
        return jsonify({"error": "Send an image file or base64 image data."}), 400

    image = _decode_image(image_bytes)
    if image is None:
        return jsonify({"error": "The uploaded image could not be decoded."}), 400

    result = predictor.predict_image(image)
    return jsonify(result)


def _read_image_from_request():
    upload = request.files.get("image")
    if upload and upload.filename:
        return upload.read()

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if isinstance(image_data, str):
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            try:
                return base64.b64decode(image_data)
            except (ValueError, TypeError):
                return None

    return None


def _decode_image(image_bytes):
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv.imdecode(buffer, cv.IMREAD_COLOR)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
