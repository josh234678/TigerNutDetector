import time
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SORTER] %(message)s")
log = logging.getLogger(__name__)

# Sorting rules: classification -> physical command
SORT_RULES = {
    "tiger_nut": {"command": "S", "direction": "STRAIGHT", "color": "green", "label": "Good Tiger Nut"},
    "stone":     {"command": "R", "direction": "RIGHT",    "color": "red",   "label": "Stone"},
    "uncertain": {"command": "S", "direction": "STRAIGHT", "color": "grey",  "label": "Uncertain - Manual Check"},
}

# S = Straight (good nut flows freely)
# R = Right deflector activated (stone)


class SortingController:
    def __init__(self, confidence_threshold=0.65,
                 use_gpio=False,
                 serial_port=None, baud_rate=9600):
        self.threshold  = confidence_threshold
        self.stats      = defaultdict(int)
        self.history    = []
        self._serial    = None
        self._use_gpio  = use_gpio

        if use_gpio:
            import gpio_controller  # initializes GPIO on import
            self._gpio = gpio_controller
            log.info("GPIO sorting controller active.")
        elif serial_port:
            self._connect_serial(serial_port, baud_rate)

    def _connect_serial(self, port, baud_rate):
        try:
            import serial
            self._serial = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)
            log.info(f"Serial connected on {port} at {baud_rate} baud")
        except ImportError:
            log.warning("pyserial not installed. Run: pip install pyserial")
        except Exception as e:
            log.warning(f"Serial connection failed: {e}. Running without hardware output.")

    def decide(self, raw_label, confidence):
        if confidence < self.threshold * 100:
            return SORT_RULES["uncertain"]
        return SORT_RULES.get(raw_label, SORT_RULES["uncertain"])

    def execute(self, raw_label, confidence):
        action = self.decide(raw_label, confidence)

        # GPIO output (Raspberry Pi)
        if self._use_gpio:
            cmd = action["command"]
            if cmd == "R":
                self._gpio.sort_right()
            else:
                self._gpio.sort_straight()

        # Serial output (Arduino / USB)
        elif self._serial and self._serial.is_open:
            try:
                self._serial.write((action["command"] + "\n").encode())
                log.info(f"Serial -> {action['command']} ({action['direction']})")
            except Exception as e:
                log.warning(f"Serial write failed: {e}")

        # Track stats
        self.stats[raw_label] += 1
        self.stats["total"] += 1

        entry = {
            "label":      action["label"],
            "direction":  action["direction"],
            "command":    action["command"],
            "confidence": confidence,
            "timestamp":  time.strftime("%H:%M:%S"),
        }
        self.history.append(entry)
        if len(self.history) > 100:
            self.history.pop(0)

        log.info(f"Item: {action['label']} | Direction: {action['direction']} | Conf: {confidence}%")
        return action

    def get_stats(self):
        return {
            "total":        self.stats["total"],
            "good_nut":     self.stats["tiger_nut"],
            "stone":        self.stats["stone"],
            "uncertain":    self.stats["uncertain"],
            "history":      self.history[-20:],
        }

    def reset_stats(self):
        self.stats.clear()
        self.history.clear()
        log.info("Stats reset.")

    def close(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            log.info("Serial port closed.")
        if self._use_gpio:
            self._gpio.cleanup()
