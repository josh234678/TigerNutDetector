import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMG_SIZE = (128, 128)
CLASSES = ["tiger_nut", "stone"]


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = []
    for ch, b in zip(range(3), [32, 32, 32]):
        hist = cv2.calcHist([hsv], [ch], None, [b], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        color_feats.extend(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([color_feats, lbp_hist])


def extract_features_from_bytes(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_feats = []
    for ch, b in zip(range(3), [32, 32, 32]):
        hist = cv2.calcHist([hsv], [ch], None, [b], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        color_feats.extend(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([color_feats, lbp_hist])
