import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
MODEL_PATH = "tiger_nut_detector.pkl"
IMG_SIZE = (128, 128)
CLASSES = ["tiger_nut", "stone"]


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)

    # Color features — HSV histogram (captures warm-brown vs cool-grey difference)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = []
    bins = [32, 32, 32]
    for ch, b in zip(range(3), bins):
        hist = cv2.calcHist([hsv], [ch], None, [b], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        color_feats.extend(hist)

    # Texture features — LBP (captures wrinkled tiger nut vs smooth stone)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([color_feats, lbp_hist])


def load_dataset():
    X, y = [], []
    counts = {cls: 0 for cls in CLASSES}

    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"[!] Missing folder: {cls_dir}")
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            feats = extract_features(os.path.join(cls_dir, fname))
            if feats is not None:
                X.append(feats)
                y.append(label)
                counts[cls] += 1

    print(f"\nDataset loaded: {counts['tiger_nut']} tiger nuts | {counts['stone']} stones")
    return np.array(X), np.array(y)


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    print("=== Tiger Nut vs Stone Detector — Training ===\n")

    X, y = load_dataset()
    if len(X) == 0:
        print("No images found. Add images to dataset/tiger_nut/ and dataset/stone/")
        exit(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples\n")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"Validation Accuracy: {acc:.1f}%\n")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    plot_confusion(y_val, y_pred)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    if acc < 80:
        print("\n[!] Accuracy below 80% -- consider collecting more varied images or upgrading to CNN.")
    elif acc >= 95:
        print("\n[OK] Excellent accuracy! Ready for real-world testing.")
    else:
        print("\n[OK] Good accuracy. Test on real packaging photos next.")
