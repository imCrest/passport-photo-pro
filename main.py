from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

app = Flask(__name__)

PASSPORT_W = 413
PASSPORT_H = 531

def process_passport(image_bytes):
    no_bg = remove(image_bytes)
    img = Image.open(io.BytesIO(no_bg)).convert("RGB")

    open_cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    margin = int(0.5 * h)

    y1 = max(0, y - margin)
    y2 = min(open_cv_img.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(open_cv_img.shape[1], x + w + margin)

    crop = open_cv_img[y1:y2, x1:x2]

    final = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    final = final.resize((PASSPORT_W, PASSPORT_H))

    output = io.BytesIO()
    final.save(output, format="JPEG", quality=95)
    output.seek(0)

    return output

@app.route("/")
def home():
    return "Passport API Running"

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    result = process_passport(image_bytes)

    if result is None:
        return jsonify({"error": "Face not detected"}), 400

    return send_file(result, mimetype="image/jpeg")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
