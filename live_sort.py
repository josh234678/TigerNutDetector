"""
Live camera sorting mode.
Captures from webcam, classifies each frame, sends sorting command.

Usage:
  python live_sort.py                        # webcam only
  python live_sort.py --port COM3            # webcam + serial to Arduino
  python live_sort.py --port COM3 --cam 1   # second camera
"""

import cv2
import joblib
import argparse
import numpy as np
from utils import extract_features_from_bytes, CLASSES
from sorter import SortingController, SORT_RULES

MODEL_PATH = "tiger_nut_detector.pkl"

COLOR_MAP = {
    "green":  (50,  205, 50),
    "orange": (0,   165, 255),
    "red":    (0,   0,   220),
    "grey":   (150, 150, 150),
}

INSTRUCTION = "[SPACE] = capture & sort   [R] = reset stats   [Q] = quit"


def draw_overlay(frame, action, confidence, stats):
    h, w = frame.shape[:2]
    color = COLOR_MAP[action["color"]]

    # Direction banner
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, action["label"], (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{action['direction']}  ({confidence:.0f}%)", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Direction arrow
    arrow_map = {"STRAIGHT": "v", "LEFT": "<", "RIGHT": ">", "STRAIGHT": "v"}
    arrow = {"LEFT": "<--", "RIGHT": "-->", "STRAIGHT": " | "}.get(action["direction"], "?")
    cv2.putText(frame, arrow, (w - 90, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Stats panel at bottom
    cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Total: {stats['total']}   Good: {stats['good_nut']}   "
                       f"Bad: {stats['bad_nut']}   Stone: {stats['stone']}",
                (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, INSTRUCTION, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


def run(camera_index=0, serial_port=None):
    print(f"\nLoading model from {MODEL_PATH}...")
    clf = joblib.load(MODEL_PATH)
    print("Model loaded.")

    controller = SortingController(
        confidence_threshold=0.60,
        serial_port=serial_port,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Could not open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nCamera ready. {INSTRUCTION}\n")

    last_action = SORT_RULES["uncertain"]
    last_conf   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        display = draw_overlay(display, last_action, last_conf, controller.get_stats())
        cv2.imshow("Tiger Nut Sorter - Live", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            controller.reset_stats()
            last_action = SORT_RULES["uncertain"]
            last_conf   = 0.0
            print("Stats reset.")

        elif key == ord(' '):
            # Capture and classify current frame
            _, buf = cv2.imencode(".jpg", frame)
            feats = extract_features_from_bytes(buf.tobytes())

            if feats is not None:
                pred    = clf.predict([feats])[0]
                proba   = clf.predict_proba([feats])[0]
                conf    = round(float(proba[pred]) * 100, 1)
                label   = CLASSES[pred]

                last_action = controller.execute(label, conf)
                last_conf   = conf

                stats = controller.get_stats()
                print(f"[{stats['total']:04d}] {last_action['label']:20s} | "
                      f"{last_action['direction']:8s} | Conf: {conf}%")

    cap.release()
    cv2.destroyAllWindows()
    controller.close()
    print("\nFinal stats:", controller.get_stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port for machine control (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--cam",  type=int, default=0,
                        help="Camera index (default 0)")
    args = parser.parse_args()
    run(camera_index=args.cam, serial_port=args.port)
