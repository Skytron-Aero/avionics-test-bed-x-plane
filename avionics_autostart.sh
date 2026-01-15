#!/bin/bash
# Avionics Autostart - Launches SVS and MFD on separate displays
# Run this script on system startup

SCRIPT_DIR="/home/skytron-pc/Documents/avionics-test-bed-x-plane"
cd "$SCRIPT_DIR"

# Wait for display to be ready
sleep 3

# Start the server if not running
if ! pgrep -f "uvicorn app.main" > /dev/null; then
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    sleep 4
fi

# Create separate Firefox profiles for kiosk isolation (if they don't exist)
firefox --headless --CreateProfile "avionics-svs" 2>/dev/null || true
firefox --headless --CreateProfile "avionics-mfd" 2>/dev/null || true

# Launch SVS (Synthetic Vision) on primary display (left, position 0,0)
firefox --kiosk -P "avionics-svs" --class="AvionicsSVS" http://localhost:8000/synthetic-vision &

sleep 2

# Launch MFD (Live Map) on secondary display (right, position 1920,0)
firefox --kiosk -P "avionics-mfd" --class="AvionicsMFD" http://localhost:8000/live-map &

echo "Avionics displays started"
echo "  SVS: http://localhost:8000/synthetic-vision"
echo "  MFD: http://localhost:8000/live-map"
echo ""
echo "Use Alt+F7 to move windows between displays if needed"
