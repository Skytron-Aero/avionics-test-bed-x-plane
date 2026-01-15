#!/usr/bin/env python3
"""
Automated Test Flight System
Runs continuous test flights from LAX to TOA to train and evaluate landing performance.
"""

import asyncio
import aiohttp
import json
import time
import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/xplane"
DEPARTURE = "KLAX"
DESTINATION = "KTOA"
TEST_DURATION_HOURS = 12
LOG_DIR = Path(__file__).parent.parent / "logs" / "test_flights"

# Flight state tracking
class FlightTestRunner:
    def __init__(self):
        self.running = True
        self.current_flight = None
        self.flights_completed = 0
        self.flights_failed = 0
        self.total_landing_score = 0
        self.start_time = None
        self.ws = None
        self.session = None
        self.log_file = None

        # Flight state
        self.flight_active = False
        self.flight_phase = "IDLE"
        self.last_position = None
        self.landing_data = {}

    async def setup(self):
        """Initialize connections and logging."""
        # Create log directory
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"test_run_{timestamp}.log"

        self.log(f"=== Automated Test Flight System ===")
        self.log(f"Route: {DEPARTURE} → {DESTINATION}")
        self.log(f"Duration: {TEST_DURATION_HOURS} hours")
        self.log(f"Log file: {self.log_file}")
        self.log("")

        # Create HTTP session
        self.session = aiohttp.ClientSession()

        # Check server health
        try:
            async with self.session.get(f"{SERVER_URL}/health") as resp:
                if resp.status != 200:
                    self.log("ERROR: Server not responding")
                    return False
                self.log("Server connection: OK")
        except Exception as e:
            self.log(f"ERROR: Cannot connect to server: {e}")
            return False

        # Check X-Plane connection
        try:
            async with self.session.get(f"{SERVER_URL}/api/xplane/status") as resp:
                data = await resp.json()
                if not data.get("connected"):
                    self.log("WARNING: X-Plane not connected - waiting...")
                    # Wait for X-Plane to connect
                    for i in range(30):
                        await asyncio.sleep(2)
                        async with self.session.get(f"{SERVER_URL}/api/xplane/status") as resp2:
                            data2 = await resp2.json()
                            if data2.get("connected"):
                                self.log("X-Plane connected!")
                                break
                    else:
                        self.log("ERROR: X-Plane did not connect within 60 seconds")
                        return False
                else:
                    self.log(f"X-Plane connection: OK (packets: {data.get('packets_received', 0)})")
        except Exception as e:
            self.log(f"ERROR: Cannot check X-Plane status: {e}")
            return False

        return True

    def log(self, message):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")

    async def connect_websocket(self):
        """Connect to X-Plane WebSocket for flight monitoring."""
        try:
            self.ws = await self.session.ws_connect(WS_URL)
            self.log("WebSocket connected")
            return True
        except Exception as e:
            self.log(f"WebSocket connection failed: {e}")
            return False

    async def start_flight(self):
        """Start a new test flight."""
        self.log(f"\n--- Starting Flight #{self.flights_completed + 1} ---")
        self.log(f"Route: {DEPARTURE} → {DESTINATION}")

        # Upload flight plan via WebSocket
        flight_plan = {
            "type": "upload_flight_plan",
            "waypoints": [
                {"name": DEPARTURE, "lat": 33.9425, "lon": -118.4081, "altitude_ft": 0, "type": "departure"},
                {"name": DESTINATION, "lat": 33.8034, "lon": -118.1518, "altitude_ft": 0, "type": "destination"}
            ],
            "cruise_altitude": 5000
        }

        await self.ws.send_json(flight_plan)
        self.log("Flight plan uploaded")

        # Wait a moment for flight plan to be processed
        await asyncio.sleep(1)

        # Start flight
        start_cmd = {"type": "start_flight"}
        await self.ws.send_json(start_cmd)
        self.log("Flight started")

        self.flight_active = True
        self.flight_phase = "TAKEOFF"
        self.current_flight = {
            "start_time": datetime.now(),
            "departure": DEPARTURE,
            "destination": DESTINATION,
            "phases": [],
            "landing": None
        }

    async def monitor_flight(self):
        """Monitor flight progress and detect completion."""
        while self.flight_active and self.running:
            try:
                msg = await asyncio.wait_for(self.ws.receive(), timeout=5.0)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Handle ping
                    if data.get("type") == "ping":
                        await self.ws.send_json({"type": "pong"})
                        continue

                    # Handle automation status
                    if data.get("type") == "automation_status":
                        new_phase = data.get("phase", "UNKNOWN")
                        if new_phase != self.flight_phase:
                            self.log(f"Phase: {self.flight_phase} → {new_phase}")
                            self.current_flight["phases"].append({
                                "phase": new_phase,
                                "time": datetime.now().isoformat()
                            })
                            self.flight_phase = new_phase

                        # Check for flight completion
                        if new_phase == "COMPLETED" or data.get("is_active") == False:
                            self.log("Flight completed!")
                            await self.record_landing(data)
                            self.flight_active = False
                            return True

                        # Check for autoland
                        if data.get("autoland_active"):
                            self.log(f"Autoland active - Phase: {data.get('autoland_phase', 'UNKNOWN')}")

                    # Handle flight data updates
                    if "lat" in data and "lon" in data:
                        self.last_position = {
                            "lat": data["lat"],
                            "lon": data["lon"],
                            "alt": data.get("alt_msl", 0),
                            "alt_agl": data.get("alt_agl", 0),
                            "gs": data.get("ground_speed", 0),
                            "vs": data.get("vertical_speed", 0)
                        }

                        # Detect landing by low AGL
                        if self.flight_phase == "LANDING" and data.get("alt_agl", 1000) < 5:
                            self.log("Touchdown detected!")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.log(f"WebSocket error: {msg.data}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.log("WebSocket closed")
                    break

            except asyncio.TimeoutError:
                # Check if still flying
                if self.last_position and self.last_position.get("alt_agl", 1000) < 10:
                    # On ground for a while - might be done
                    pass
            except Exception as e:
                self.log(f"Monitor error: {e}")

        return False

    async def record_landing(self, data):
        """Record landing performance metrics."""
        landing_data = {
            "time": datetime.now().isoformat(),
            "vertical_speed_fpm": data.get("vertical_speed", 0),
            "ground_speed_kts": data.get("ground_speed", 0),
            "position": self.last_position,
            "touchdown_vs": abs(data.get("vertical_speed", 0)),
        }

        # Score the landing (lower VS = better)
        vs = abs(landing_data["touchdown_vs"])
        if vs < 100:
            score = 100
            rating = "EXCELLENT"
        elif vs < 200:
            score = 80
            rating = "GOOD"
        elif vs < 400:
            score = 60
            rating = "ACCEPTABLE"
        elif vs < 600:
            score = 40
            rating = "HARD"
        else:
            score = 20
            rating = "VERY HARD"

        landing_data["score"] = score
        landing_data["rating"] = rating

        self.current_flight["landing"] = landing_data
        self.total_landing_score += score

        self.log(f"Landing Score: {score}/100 ({rating})")
        self.log(f"Touchdown VS: {vs:.0f} fpm")

    async def wait_for_reset(self):
        """Wait for aircraft to be repositioned for next flight."""
        self.log("Waiting for aircraft reset...")

        # In X-Plane, you'd need to reposition the aircraft manually
        # or use a plugin/command to reset to departure airport

        # For now, wait for the aircraft to be on ground at departure
        timeout = 300  # 5 minutes max wait
        start = time.time()

        while time.time() - start < timeout and self.running:
            try:
                async with self.session.get(f"{SERVER_URL}/api/xplane/status") as resp:
                    data = await resp.json()

                    # Check if on ground
                    if data.get("flight_data", {}).get("alt_agl", 1000) < 50:
                        # Check if near departure airport (rough check)
                        lat = data.get("flight_data", {}).get("lat", 0)
                        lon = data.get("flight_data", {}).get("lon", 0)

                        # KLAX is around 33.94, -118.41
                        if abs(lat - 33.94) < 0.1 and abs(lon + 118.41) < 0.1:
                            self.log("Aircraft at departure airport - ready for next flight")
                            return True

            except Exception as e:
                pass

            await asyncio.sleep(5)

        self.log("Reset timeout - please reposition aircraft manually")
        return False

    async def run(self):
        """Main test loop."""
        if not await self.setup():
            return

        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(hours=TEST_DURATION_HOURS)

        self.log(f"\nTest run started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Will run until {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        # Connect WebSocket
        if not await self.connect_websocket():
            return

        try:
            while datetime.now() < end_time and self.running:
                # Start a new flight
                await self.start_flight()

                # Monitor until completion
                success = await self.monitor_flight()

                if success:
                    self.flights_completed += 1
                else:
                    self.flights_failed += 1

                # Print summary
                elapsed = datetime.now() - self.start_time
                self.log(f"\n=== Progress Update ===")
                self.log(f"Elapsed: {elapsed}")
                self.log(f"Flights completed: {self.flights_completed}")
                self.log(f"Flights failed: {self.flights_failed}")
                if self.flights_completed > 0:
                    avg_score = self.total_landing_score / self.flights_completed
                    self.log(f"Average landing score: {avg_score:.1f}/100")
                self.log("")

                # Wait for reset
                if self.running and datetime.now() < end_time:
                    self.log("Please reset aircraft to KLAX for next test flight...")
                    self.log("(Press Ctrl+C to stop testing)")
                    await self.wait_for_reset()

        except Exception as e:
            self.log(f"Error during test run: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        self.log("\n=== Final Summary ===")
        if self.start_time:
            total_time = datetime.now() - self.start_time
            self.log(f"Total time: {total_time}")
        self.log(f"Flights completed: {self.flights_completed}")
        self.log(f"Flights failed: {self.flights_failed}")
        if self.flights_completed > 0:
            avg_score = self.total_landing_score / self.flights_completed
            self.log(f"Average landing score: {avg_score:.1f}/100")

        # Close connections
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()

        self.log("\nTest run ended.")

    def stop(self):
        """Stop the test run."""
        self.log("\nStopping test run...")
        self.running = False


async def main():
    runner = FlightTestRunner()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        runner.stop()

    signal.signal(signal.SIGINT, signal_handler)

    await runner.run()


if __name__ == "__main__":
    print("=" * 50)
    print("Automated Test Flight System")
    print("=" * 50)
    print()
    print("This script will run continuous test flights")
    print("from LAX to TOA to train landing performance.")
    print()
    print("Prerequisites:")
    print("  1. Server must be running (localhost:8000)")
    print("  2. X-Plane must be running and connected")
    print("  3. Aircraft should be at KLAX ready for takeoff")
    print()
    print("Press Ctrl+C to stop at any time.")
    print()

    asyncio.run(main())
