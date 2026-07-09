"""
Raspberry Pi live sorting — uses Pi Camera + GPIO deflectors.

Usage:
  python live_sort_pi.py                  # Pi Camera, GPIO output
  python live_sort_pi.py --sim            # simulation mode (no GPIO / no camera needed)
  python live_sort_pi.py --cam usb        # USB webcam instead of Pi Camera
  python live_sort_pi.py --conf 0.70      # stricter confidence threshold

Controls (keyboard when preview window is open):
  SPACE  =  capture current frame and sort
  A      =  auto mode (captures every N seconds automatically)
  R      =  reset session stats
  Q      =  quit
"""

import cv2
import time
import joblib
import argparse
import numpy as np
from utils import extract_features_from_bytes, CLASSES
from sorter import SortingController, SORT_RULES

MODEL_PATH     = "tiger_nut_detector.pkl"
AUTO_INTERVAL  = 2.0   # seconds between captures in auto mode

COLOR_BGR = {
    "green":  (50,  205,  50),
    "orange": (0,   165, 255),
    "red":    (0,     0, 220),
    "grey":   (150, 150, 150),
}


def open_camera(use_usb=False):
    """Try Pi Camera first, fall back to USB webcam."""
    if not use_usb:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            ))
            cam.start()
            time.sleep(1)
            print("Pi Camera ready.")
            return ("picamera", cam)
        except Exception as e:
            print(f"Pi Camera unavailable ({e}), trying USB webcam...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if cap.isOpened():
        print("USB webcam ready.")
        return ("usb", cap)

    raise RuntimeError("No camera found. Check connections.")


def read_frame(cam_info):
    kind, cam = cam_info
    if kind == "picamera":
        frame = cam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        return frame if ret else None


def release_camera(cam_info):
    kind, cam = cam_info
    if kind == "picamera":
        cam.stop()
    else:
        cam.release()


def draw_hud(frame, last_action, last_conf, stats, auto_mode):
    h, w = frame.shape[:2]
    color = COLOR_BGR[last_action["color"]]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.putText(frame, last_action["label"],
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    dir_txt = f"{last_action['direction']}  |  conf: {last_conf:.0f}%  |  cmd: {last_action['command']}"
    cv2.putText(frame, dir_txt,
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

    # Direction arrow (top right)
    arrow = {"LEFT": "<--", "RIGHT": "-->", "STRAIGHT": " || "}.get(last_action["direction"], "?")
    cv2.putText(frame, arrow, (w - 100, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # Bottom stats bar
    cv2.rectangle(frame, (0, h - 82), (w, h), (15, 15, 15), -1)
    stat_txt = (f"Total: {stats['total']}   "
                f"Good: {stats['good_nut']}   "
                f"Bad: {stats['bad_nut']}   "
                f"Stone: {stats['stone']}")
    cv2.putText(frame, stat_txt, (10, h - 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    mode_lbl = "AUTO MODE" if auto_mode else "MANUAL"
    mode_clr = (0, 220, 100) if auto_mode else (180, 180, 180)
    cv2.putText(frame, f"[{mode_lbl}]  SPACE=capture  A=auto  R=reset  Q=quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_clr, 1)

    return frame


def classify_frame(frame, clf, controller):
    _, buf = cv2.imencode(".jpg", frame)
    feats  = extract_features_from_bytes(buf.tobytes())
    if feats is None:
        return SORT_RULES["uncertain"], 0.0

    pred   = clf.predict([feats])[0]
    proba  = clf.predict_proba([feats])[0]
    label  = CLASSES[pred]
    conf   = round(float(proba[pred]) * 100, 1)
    action = controller.execute(label, conf)
    return action, conf


def run(use_usb=False, conf_threshold=0.65, sim=False):
    print(f"\nLoading model: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print("Model loaded.\n")

    controller = SortingController(
        confidence_threshold=conf_threshold,
        use_gpio=(not sim),
    )

    if sim:
        print("Running in SIMULATION mode — no GPIO output.\n")

    cam_info   = open_camera(use_usb=use_usb)
    last_action = SORT_RULES["uncertain"]
    last_conf   = 0.0
    auto_mode   = False
    last_auto_t = 0.0

    print("Ready. Press SPACE to sort an item, A to toggle auto mode, Q to quit.\n")

    while True:
        frame = read_frame(cam_info)
        if frame is None:
            break

        now = time.time()
        if auto_mode and (now - last_auto_t) >= AUTO_INTERVAL:
            last_action, last_conf = classify_frame(frame, clf, controller)
            last_auto_t = now
            s = controller.get_stats()
            print(f"[{s['total']:04d}] {last_action['label']:22s} | "
                  f"{last_action['direction']:8s} | {last_conf:.0f}%")

        display = draw_hud(frame.copy(), last_action, last_conf,
                           controller.get_stats(), auto_mode)
        cv2.imshow("Tiger Nut Sorter", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            last_action, last_conf = classify_frame(frame, clf, controller)
            s = controller.get_stats()
            print(f"[{s['total']:04d}] {last_action['label']:22s} | "
                  f"{last_action['direction']:8s} | {last_conf:.0f}%")
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"Auto mode: {'ON' if auto_mode else 'OFF'}")
        elif key == ord('r'):
            controller.reset_stats()
            last_action = SORT_RULES["uncertain"]
            last_conf   = 0.0
            print("Stats reset.")

    cv2.destroyAllWindows()
    release_camera(cam_info)
    controller.close()

    s = controller.get_stats()
    print(f"\n=== Session Summary ===")
    print(f"Total items  : {s['total']}")
    print(f"Good nuts    : {s['good_nut']}")
    print(f"Bad nuts     : {s['bad_nut']}")
    print(f"Stones       : {s['stone']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",  type=str, default="pi",
                        help="Camera: 'pi' for Pi Camera, 'usb' for USB webcam")
    parser.add_argument("--conf", type=float, default=0.65,
                        help="Confidence threshold 0.0-1.0 (default 0.65)")
    parser.add_argument("--sim",  action="store_true",
                        help="Simulation mode — no GPIO, no Pi Camera required")
    args = parser.parse_args()
    run(use_usb=(args.cam == "usb"), conf_threshold=args.conf, sim=args.sim)
