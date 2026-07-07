import os
import joblib
from flask import Flask, request, render_template, jsonify
from utils import extract_features_from_bytes, CLASSES
from sorter import SortingController

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH   = "tiger_nut_detector.pkl"
SERIAL_PORT  = os.environ.get("SERIAL_PORT", None)   # set in Railway env vars when deploying with hardware
CONF_THRESH  = float(os.environ.get("CONF_THRESHOLD", "0.65"))

clf        = None
controller = None


def load_model():
    global clf, controller
    if os.path.exists(MODEL_PATH):
        clf        = joblib.load(MODEL_PATH)
        controller = SortingController(
            confidence_threshold=CONF_THRESH,
            serial_port=SERIAL_PORT,
        )
        print("Model and sorting controller ready.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Run train.py first.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return jsonify({"error": "Only JPG, PNG, WEBP supported."}), 400

    features = extract_features_from_bytes(file.read())
    if features is None:
        return jsonify({"error": "Could not process image."}), 400

    pred       = clf.predict([features])[0]
    proba      = clf.predict_proba([features])[0]
    raw_label  = CLASSES[pred]
    confidence = round(float(proba[pred]) * 100, 1)

    # Run through sorting controller — applies threshold, sends serial, tracks stats
    action = controller.execute(raw_label, confidence)

    return jsonify({
        "label":        action["label"],
        "raw_label":    raw_label,
        "direction":    action["direction"],
        "command":      action["command"],
        "action":       action["direction"].lower(),
        "confidence":   confidence,
        "good_nut_pct": round(float(proba[0]) * 100, 1),
        "bad_nut_pct":  round(float(proba[1]) * 100, 1),
        "stone_pct":    round(float(proba[2]) * 100, 1),
    })


@app.route("/stats")
def stats():
    if controller is None:
        return jsonify({"error": "Controller not ready."}), 500
    return jsonify(controller.get_stats())


@app.route("/reset", methods=["POST"])
def reset():
    if controller:
        controller.reset_stats()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
