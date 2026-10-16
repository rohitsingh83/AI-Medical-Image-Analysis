"""
app.py - Flask Web Application for AI-Powered Medical Image Analysis
====================================================================
Simulates how a hospital / radiology lab would use the system:

    Doctor opens the web page -> uploads a chest X-ray
    -> AI returns "NORMAL" or "PNEUMONIA" with confidence + Grad-CAM heatmap

Run
---
    python app.py
    then open http://127.0.0.1:5000

Endpoints
---------
GET  /            -> upload form (web UI)
POST /upload      -> form submission -> result page
POST /api/predict -> JSON API (for Postman / integration tests)
GET  /health      -> service health check (industry practice)
"""

import os
from pathlib import Path

import cv2
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)
from werkzeug.utils import secure_filename

from src import config
from src.predict import predict_image

app = Flask(__name__)
app.secret_key = "medical-ai-demo-secret"          # demo only - use env var in production

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "dcm"}
config.ensure_dirs()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Landing page with the upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle the form upload, run the AI, render the result page."""
    if "image" not in request.files:
        flash("No file selected. Please upload a chest X-ray image.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected. Please upload a chest X-ray image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Use PNG / JPG / JPEG / BMP.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = config.UPLOAD_DIR / filename
    file.save(save_path)

    result = predict_image(save_path)
    gradcam_name = Path(result.get("gradcam_path", "")).name if result.get("gradcam_path") else None
    return render_template("result.html", result=result,
                           uploaded_name=filename, gradcam_name=gradcam_name)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve the user's uploaded image back to the result page."""
    return send_from_directory(config.UPLOAD_DIR, filename)


@app.route("/gradcam/<filename>")
def gradcam_file(filename):
    """Serve the Grad-CAM overlay image."""
    return send_from_directory(config.PREDICTIONS_DIR, filename)


# ---------------------------------------------------------------------------
# JSON API (industry-style programmatic access)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API:
        curl -F "image=@xray.png" http://127.0.0.1:5000/api/predict
    """
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    filename = secure_filename(file.filename)
    save_path = config.UPLOAD_DIR / filename
    file.save(save_path)
    result = predict_image(save_path)
    return jsonify(result)


@app.route("/health")
def health():
    """Health-check endpoint - standard in production services."""
    model_path = config.MODELS_DIR / config.MODEL_NAME
    return jsonify({
        "status": "healthy" if model_path.exists() else "model_missing",
        "model": config.MODEL_NAME,
        "image_size": config.IMG_SIZE,
        "classes": config.CLASS_NAMES,
    })


if __name__ == "__main__":
    # host 0.0.0.0 so the app is reachable in cloud notebooks / containers
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[app] AI Medical Image Analysis running at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
