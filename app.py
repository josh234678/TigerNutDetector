import os
import joblib
from flask import Flask, request, render_template, jsonify
from utils import extract_features_from_bytes, CLASSES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max upload

MODEL_PATH = "tiger_nut_detector.pkl"

clf = None


def load_model():
    global clf
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Run train.py first.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded. Please run train.py first."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    allowed = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": "Only JPG, PNG, and WEBP images are supported."}), 400

    img_bytes = file.read()
    features = extract_features_from_bytes(img_bytes)

    if features is None:
        return jsonify({"error": "Could not process the image. Please try another."}), 400

    pred = clf.predict([features])[0]
    proba = clf.predict_proba([features])[0]

    label = CLASSES[pred]
    display_label = "Tiger Nut" if label == "tiger_nut" else "Stone"

    return jsonify({
        "label": display_label,
        "raw_label": label,
        "confidence": round(float(proba[pred]) * 100, 1),
        "tiger_nut_pct": round(float(proba[0]) * 100, 1),
        "stone_pct": round(float(proba[1]) * 100, 1),
    })


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
