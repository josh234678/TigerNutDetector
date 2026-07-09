#!/bin/bash
# Run on Raspberry Pi: bash install_pi.sh

set -e
echo "=== Tiger Nut Sorter — Raspberry Pi Setup ==="

# System packages
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv python3-opencv libatlas-base-dev \
  libhdf5-dev libhdf5-serial-dev python3-picamera2 git

# Clone / pull project
cd /home/pi
if [ -d "TigerNutDetector" ]; then
  echo "Updating existing repo..."
  cd TigerNutDetector && git pull
else
  echo "Cloning repo..."
  git clone https://github.com/josh234678/TigerNutDetector.git
  cd TigerNutDetector
fi

# Python virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements_pi.txt -q

echo ""
echo "=== Testing GPIO (simulation) ==="
python3 -c "from gpio_controller import sort_left, sort_right, sort_straight; sort_left(); sort_right(); sort_straight(); print('GPIO test OK')"

echo ""
echo "=== Installing systemd service ==="
sudo cp tigernut.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tigernut.service
sudo systemctl start  tigernut.service

echo ""
echo "=== Done! ==="
echo "Service status: sudo systemctl status tigernut"
echo "View logs:      sudo journalctl -u tigernut -f"
echo "Stop service:   sudo systemctl stop tigernut"
echo "Manual run:     source venv/bin/activate && python live_sort_pi.py"
