#!/bin/bash
# MFD Viewer - Opens live-map in Firefox kiosk mode
# Automatically opens on the secondary display

# Start the server if not running
if ! pgrep -f "uvicorn app.main" > /dev/null; then
    cd /home/skytron-pc/Documents/avionics-test-bed-x-plane
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    sleep 3
fi

# Open Firefox in kiosk mode on live-map page
firefox --kiosk http://localhost:8000/live-map
