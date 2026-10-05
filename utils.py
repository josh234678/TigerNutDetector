import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMG_SIZE = (128, 128)
CLASSES = ["tiger_nut", "stone"]


def _compute_features(img):
    img = cv2.resize(img, IMG_SIZE)

    # HSV color histogram — captures warm-brown vs grey vs discolored
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = []
    for ch, b in zip(range(3), [32, 32, 32]):
        hist = cv2.calcHist([hsv], [ch], None, [b], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        color_feats.extend(hist)

    # LBP texture — captures wrinkles vs smooth vs damaged surface
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    # Multi-radius LBP — catches both fine and coarse texture damage
    lbp2 = local_binary_pattern(gray, P=16, R=2, method="uniform")
    lbp2_hist, _ = np.histogram(lbp2.ravel(), bins=18, range=(0, 18), density=True)

    # Dark spot ratio — bad nuts have dark mold/rot patches
    dark_mask = (hsv[:, :, 2] < 60).astype(np.float32)
    dark_ratio = dark_mask.mean()

    # Color saturation stats — bad nuts often lose saturation or go greenish
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    sat_mean = sat.mean()
    sat_std  = sat.std()

    # Hue stats — bad nuts shift away from warm orange-brown hue (10-25 in HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    hue_mean = hue.mean() / 180.0
    hue_std  = hue.std()  / 180.0

    # Brightness variance — damage creates local brightness inconsistency
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    val_std = val.std()

    extra = np.array([dark_ratio, sat_mean, sat_std, hue_mean, hue_std, val_std])

    return np.concatenate([color_feats, lbp_hist, lbp2_hist, extra])


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return _compute_features(img)


# Expected length of the feature vector _compute_features() returns, so
# callers (train.py, app.py) can fail loudly on a mismatch instead of
# scikit-learn raising an opaque "X has N features" error at predict time.
FEATURE_DIM = 32 * 3 + 10 + 18 + 6


def extract_features_from_bytes(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return _compute_features(img)
