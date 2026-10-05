"""
GPIO controller for the Tiger Nut sorting machine.

Physical wiring (BCM pin numbering):
  GPIO 27  -->  RIGHT relay (stone deflector)
  GPIO 22  -->  Green LED   (good nut indicator)
  GPIO 24  -->  Red LED     (stone indicator)
  GPIO 25  -->  Buzzer      (alert on stone)

Relay logic:
  - Relay module is ACTIVE HIGH (most common)
  - ON  = GPIO HIGH = deflector extends
  - OFF = GPIO LOW  = deflector retracted (default / good nut path)

Belt timing:
  - BELT_DELAY_SEC: time (seconds) between camera capture and deflector activation
    Calculate as: distance_camera_to_deflector(cm) / belt_speed(cm/s)
  - DEFLECTOR_DURATION_SEC: how long the deflector stays active per item
"""

import time
import threading
import logging

log = logging.getLogger(__name__)

# BCM GPIO pin assignments
PIN_RIGHT_RELAY = 27   # stone    → right
PIN_LED_GREEN   = 22   # good nut LED
PIN_LED_RED     = 24   # stone LED
PIN_BUZZER      = 25   # alert buzzer

# Timing (seconds) — adjust to match your conveyor
BELT_DELAY_SEC        = 0.3   # delay after capture before deflector fires
DEFLECTOR_DURATION_SEC = 0.4  # how long deflector stays active
LED_DURATION_SEC       = 1.0  # how long LED stays on

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    OUTPUT_PINS = [
        PIN_RIGHT_RELAY,
        PIN_LED_GREEN, PIN_LED_RED,
        PIN_BUZZER,
    ]
    GPIO.setup(OUTPUT_PINS, GPIO.OUT, initial=GPIO.LOW)
    GPIO_AVAILABLE = True
    log.info("GPIO initialized (BCM mode)")
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    log.warning("RPi.GPIO not available — running in simulation mode (no physical output)")


def _pulse(pin, duration):
    if not GPIO_AVAILABLE:
        log.info(f"[SIM] GPIO {pin} HIGH for {duration}s")
        return
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)


def _fire(relay_pin, led_pin, buzz=False):
    """Non-blocking: fires deflector + LED + optional buzzer in background thread."""
    def _run():
        time.sleep(BELT_DELAY_SEC)           # wait for item to travel to deflector
        _pulse(relay_pin, DEFLECTOR_DURATION_SEC)
        if led_pin:
            _pulse(led_pin, LED_DURATION_SEC)
        if buzz and GPIO_AVAILABLE:
            _pulse(PIN_BUZZER, 0.15)

    threading.Thread(target=_run, daemon=True).start()


def sort_right():
    """Activate right deflector — stone."""
    log.info("ACTION: RIGHT deflector fired (stone)")
    _fire(PIN_RIGHT_RELAY, PIN_LED_RED, buzz=True)


def sort_straight():
    """No deflector — good tiger nut flows freely."""
    log.info("ACTION: STRAIGHT (good nut — no deflector)")
    if GPIO_AVAILABLE:
        threading.Thread(
            target=lambda: _pulse(PIN_LED_GREEN, LED_DURATION_SEC),
            daemon=True
        ).start()
    else:
        log.info("[SIM] GPIO 22 (green LED) HIGH for 1.0s")


def cleanup():
    if GPIO_AVAILABLE:
        GPIO.cleanup()
        log.info("GPIO cleaned up.")
