import cv2
import numpy as np
import joblib
import sys
import os
from train import extract_features, CLASSES, IMG_SIZE

MODEL_PATH = "tiger_nut_detector.pkl"


def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found. Run train.py first.")
        return

    clf = joblib.load(MODEL_PATH)

    feats = extract_features(image_path)
    if feats is None:
        print(f"[!] Could not read image: {image_path}")
        return

    pred = clf.predict([feats])[0]
    proba = clf.predict_proba([feats])[0]
    confidence = proba[pred] * 100
    label = CLASSES[pred].replace("_", " ").title()

    print(f"\nImage : {os.path.basename(image_path)}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  Tiger Nut: {proba[0]*100:.1f}%  |  Stone: {proba[1]*100:.1f}%")

    # Show result on image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))
    color = (0, 180, 0) if pred == 0 else (0, 0, 220)
    text = f"{label} ({confidence:.0f}%)"
    cv2.rectangle(img, (0, 0), (400, 55), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Tiger Nut Detector", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_folder(folder_path):
    results = {"tiger_nut": [], "stone": []}
    exts = (".jpg", ".jpeg", ".png")

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(exts):
            fpath = os.path.join(folder_path, fname)
            clf = joblib.load(MODEL_PATH)
            feats = extract_features(fpath)
            if feats is not None:
                pred = clf.predict([feats])[0]
                conf = clf.predict_proba([feats])[0][pred] * 100
                results[CLASSES[pred]].append((fname, conf))

    print(f"\n=== Batch Results ===")
    print(f"Tiger Nuts detected: {len(results['tiger_nut'])}")
    for name, conf in results["tiger_nut"]:
        print(f"  {name} ({conf:.0f}%)")
    print(f"\nStones detected: {len(results['stone'])}")
    for name, conf in results["stone"]:
        print(f"  {name} ({conf:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image : python predict.py path/to/image.jpg")
        print("  Folder batch : python predict.py path/to/folder/")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        predict_folder(target)
    else:
        predict_image(target)
