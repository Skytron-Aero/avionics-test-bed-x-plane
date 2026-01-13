"""
X-Plane 12 UDP Bridge

Receives flight data from X-Plane 12 via UDP and exposes it for WebSocket streaming.
X-Plane sends binary packets containing flight parameters (position, attitude, speeds, etc.).

X-Plane UDP Packet Format:
- Header: "DATA" (4 bytes) + null byte (1 byte)
- Data blocks: Each block is 36 bytes
  - Index (4 bytes, int32): identifies the data group
  - 8 float values (32 bytes): the actual data

Data Group Indices Used:
- 3: Speeds (IAS, TAS, ground speed, etc.)
- 4: Mach, VVI, G-loads
- 17: Pitch, roll, headings
- 18: Alpha, beta, flight path angles
- 19: Compass heading, magnetic variation
- 20: Lat, lon, altitude MSL/AGL

Weather Data Indices (requires X-Plane Data Output config):
- 13: Atmosphere (SLP, temperature, density)
- 45: Wind at altitude (speed, direction, turbulence)
- 48: Weather/visibility
- 63-65: Cloud layers (type, coverage, base, top)
"""

import asyncio
import struct
import logging
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Optional, Callable, Set

logger = logging.getLogger("xplane_bridge")


@dataclass
class XPlaneFlightData:
    """Flight data received from X-Plane 12"""

    # Timestamp
    timestamp: str = ""

    # Position
    lat: float = 0.0
    lon: float = 0.0
    alt_msl: float = 0.0      # feet MSL
    alt_agl: float = 0.0      # feet AGL

    # Attitude
    pitch: float = 0.0        # degrees (nose up positive)
    roll: float = 0.0         # degrees (right wing down positive)
    heading_true: float = 0.0  # degrees true
    heading_mag: float = 0.0   # degrees magnetic
    mag_var: float = 0.0       # magnetic variation (degrees)

    # Speeds
    airspeed_ind: float = 0.0    # knots indicated
    airspeed_true: float = 0.0   # knots true
    groundspeed: float = 0.0     # knots

    # Vertical
    vertical_speed: float = 0.0  # feet per minute

    # Additional flight parameters
    alpha: float = 0.0           # angle of attack (degrees)
    beta: float = 0.0            # sideslip angle (degrees)
    flight_path_angle: float = 0.0  # degrees

    # G-forces
    g_normal: float = 1.0        # normal G (vertical)
    g_axial: float = 0.0         # axial G (forward/back)
    g_side: float = 0.0          # side G (lateral)

    # Mach
    mach: float = 0.0

    # Weather / Atmosphere (Index 13)
    sea_level_pressure: float = 29.92  # inHg
    sea_level_temp: float = 15.0       # Celsius
    local_temp: float = 15.0           # Celsius at aircraft
    local_density: float = 1.225       # kg/m³
    speed_of_sound: float = 661.0      # knots
    visibility_sm: float = 10.0        # statute miles

    # Cloud Layers (Indices 63, 64, 65)
    # Each layer: {type, coverage (0-1), base_ft, top_ft}
    cloud_layer_0: dict = field(default_factory=lambda: {"type": 0, "coverage": 0.0, "base_ft": 0, "top_ft": 0})
    cloud_layer_1: dict = field(default_factory=lambda: {"type": 0, "coverage": 0.0, "base_ft": 0, "top_ft": 0})
    cloud_layer_2: dict = field(default_factory=lambda: {"type": 0, "coverage": 0.0, "base_ft": 0, "top_ft": 0})

    # Wind (Index 7 or derived)
    wind_speed_kts: float = 0.0
    wind_direction_true: float = 0.0
    turbulence_intensity: float = 0.0  # 0-1 scale

    # Precipitation
    precipitation_rate: float = 0.0    # 0-1 scale
    precipitation_type: str = "none"   # none, rain, snow, mixed

    # Connection status
    connected: bool = False
    last_packet_time: float = 0.0
    packets_received: int = 0
    source_ip: str = ""
    source_port: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class XPlaneProtocol(asyncio.DatagramProtocol):
    """Asyncio protocol for receiving X-Plane UDP packets"""

    def __init__(self, callback: Callable[[bytes, tuple], None]):
        self.callback = callback
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple):
        self.callback(data, addr)

    def error_received(self, exc):
        logger.error(f"UDP error received: {exc}")

    def connection_lost(self, exc):
        if exc:
            logger.error(f"UDP connection lost: {exc}")


class XPlaneUDPListener:
    """
    Async UDP listener for X-Plane 12 flight data.
    Parses binary UDP packets and updates XPlaneFlightData.
    """

    # X-Plane data group indices - Flight Data
    INDEX_SPEEDS = 3          # IAS, TAS, true airspeed, ground speed, etc.
    INDEX_MACH_VVI_G = 4      # Mach, VVI, G-loads
    INDEX_PITCH_ROLL_HDG = 17  # Pitch, roll, true heading, magnetic heading
    INDEX_AOA_SIDESLIP = 18   # Alpha, beta, hpath, vpath
    INDEX_COMPASS = 19        # Compass heading, magnetic variation
    INDEX_LAT_LON_ALT = 20    # Lat, lon, alt MSL, alt AGL

    # X-Plane data group indices - Weather/Atmosphere
    INDEX_ATMOSPHERE = 13     # Sea level pressure, temps, density
    INDEX_WEATHER = 48        # Visibility, precipitation, runway friction
    INDEX_CLOUD_LAYER_0 = 63  # Cloud layer 0: type, coverage, base, top
    INDEX_CLOUD_LAYER_1 = 64  # Cloud layer 1
    INDEX_CLOUD_LAYER_2 = 65  # Cloud layer 2
    INDEX_WIND_ALOFT = 45     # Wind at altitude

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 49000,
        on_data_callback: Optional[Callable[[XPlaneFlightData], None]] = None,
        timeout_seconds: float = 5.0
    ):
        self.host = host
        self.port = port
        self.on_data_callback = on_data_callback
        self.timeout_seconds = timeout_seconds
        self.flight_data = XPlaneFlightData()
        self.running = False
        self._transport = None
        self._protocol = None
        self._timeout_task = None

    async def start(self):
        """Start the UDP listener"""
        loop = asyncio.get_event_loop()

        # Create UDP endpoint
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: XPlaneProtocol(self._handle_packet),
            local_addr=(self.host, self.port)
        )

        self.running = True

        # Start timeout monitoring task
        self._timeout_task = asyncio.create_task(self._monitor_timeout())

        logger.info(f"X-Plane UDP listener started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the UDP listener"""
        self.running = False

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        if self._transport:
            self._transport.close()

        logger.info("X-Plane UDP listener stopped")

    async def _monitor_timeout(self):
        """Monitor for X-Plane connection timeout"""
        while self.running:
            await asyncio.sleep(1.0)

            if self.flight_data.connected:
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - self.flight_data.last_packet_time

                if time_since_last > self.timeout_seconds:
                    self.flight_data.connected = False
                    logger.warning(
                        f"X-Plane connection timeout "
                        f"(no data for {time_since_last:.1f}s)"
                    )

                    # Notify via callback
                    if self.on_data_callback:
                        try:
                            self.on_data_callback(self.flight_data)
                        except Exception as e:
                            logger.error(f"Error in data callback: {e}")

    def _estimate_magnetic_declination(self, lat: float, lon: float) -> float:
        """
        Estimate magnetic declination based on position.
        Simple model for continental US - accurate to within ~2 degrees.
        Positive = East declination, Negative = West declination.

        For more accuracy, use the World Magnetic Model (WMM).
        """
        if lat == 0 and lon == 0:
            return 0.0  # No position data yet

        # Simple linear model for continental US (2020-2025 epoch)
        # Declination varies roughly linearly with longitude in the US
        # West coast (~-120°): ~+11 to +14° East
        # East coast (~-70°): ~-10 to -15° West
        # The variation also changes slightly with latitude

        # Base declination estimate from longitude
        # At lon=-120: decl ≈ +12°, at lon=-70: decl ≈ -13°
        decl = -0.5 * (lon + 95)  # Roughly linear approximation

        # Latitude adjustment (declination is slightly more East in southern US)
        lat_adjustment = (lat - 40) * 0.1
        decl += lat_adjustment

        # Clamp to reasonable range
        decl = max(-20, min(20, decl))

        return decl

    def _handle_packet(self, data: bytes, addr: tuple):
        """Parse X-Plane UDP packet"""
        try:
            # Validate minimum packet size and header
            if len(data) < 5:
                return

            # Check for DATA header
            if data[:4] != b'DATA':
                # Could be other X-Plane packet types (BECN, etc.) - ignore
                return

            # Parse data blocks (each block: 4-byte index + 8 floats)
            offset = 5  # Skip "DATA\0"
            block_size = 36  # 4 + 8*4 bytes

            while offset + block_size <= len(data):
                index = struct.unpack('<i', data[offset:offset + 4])[0]
                values = struct.unpack('<8f', data[offset + 4:offset + 36])

                self._parse_data_group(index, values)
                offset += block_size

            # Update metadata
            self.flight_data.timestamp = datetime.utcnow().isoformat() + 'Z'
            self.flight_data.connected = True
            self.flight_data.last_packet_time = asyncio.get_event_loop().time()
            self.flight_data.packets_received += 1
            self.flight_data.source_ip = addr[0]
            self.flight_data.source_port = addr[1]

            # Callback for WebSocket broadcast
            if self.on_data_callback:
                try:
                    self.on_data_callback(self.flight_data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")

        except struct.error as e:
            logger.error(f"Error unpacking X-Plane packet: {e}")
        except Exception as e:
            logger.error(f"Error parsing X-Plane packet: {e}")

    def _parse_data_group(self, index: int, values: tuple):
        """Parse a single data group from X-Plane"""

        if index == self.INDEX_SPEEDS:
            # Index 3: Speeds
            # [0] Vind kias, [1] Vind keas, [2] Vtrue ktas,
            # [3] Vtrue ktgs, [4] Vind mph, [5] Vtrue mphas,
            # [6] Vtrue mpgs, [7] (unused)
            self.flight_data.airspeed_ind = values[0]   # KIAS
            self.flight_data.airspeed_true = values[2]  # KTAS
            self.flight_data.groundspeed = values[3]    # Ground speed knots

        elif index == self.INDEX_MACH_VVI_G:
            # Index 4: Mach, VVI, G-loads
            # [0] Mach, [1] (unused), [2] (unused), [3] VVI fpm,
            # [4] Gnorm, [5] Gaxil, [6] Gside, [7] (unused)
            self.flight_data.mach = values[0]
            self.flight_data.vertical_speed = values[3]  # VVI in ft/min
            self.flight_data.g_normal = values[4]
            self.flight_data.g_axial = values[5]
            self.flight_data.g_side = values[6]

        elif index == self.INDEX_PITCH_ROLL_HDG:
            # Index 17: Pitch, Roll, Headings
            # [0] pitch deg, [1] roll deg, [2] hding true, [3] hding mag,
            # [4-7] (unused)
            self.flight_data.pitch = values[0]
            self.flight_data.roll = values[1]
            self.flight_data.heading_true = values[2]
            # Only use mag heading if valid (X-Plane sends -999 for no data)
            if values[3] > -900:
                self.flight_data.heading_mag = values[3]
            else:
                # Compute magnetic heading from true heading using estimated declination
                # Based on aircraft position (simple model for continental US)
                declination = self._estimate_magnetic_declination(
                    self.flight_data.lat, self.flight_data.lon
                )
                self.flight_data.mag_var = declination
                mag_hdg = (values[2] - declination) % 360
                self.flight_data.heading_mag = mag_hdg

        elif index == self.INDEX_AOA_SIDESLIP:
            # Index 18: Alpha, Beta, Paths
            # [0] alpha deg, [1] beta deg, [2] hpath deg, [3] vpath deg,
            # [4-7] (unused)
            self.flight_data.alpha = values[0]
            self.flight_data.beta = values[1]
            self.flight_data.flight_path_angle = values[3]

        elif index == self.INDEX_COMPASS:
            # Index 19: Compass
            # [0] mag hdg deg, [1] mag var deg (positive = East)
            # [2-7] (unused)
            # NOTE: Index 19 data appears unreliable in X-Plane 12 - using Index 17 instead
            compass_mag = values[0]
            mag_var = values[1]
            # Only use valid mag_var (reasonable range -30 to +30 degrees)
            if -30 < mag_var < 30:
                self.flight_data.mag_var = mag_var
            # Don't overwrite heading_mag from Index 19 - data is often garbage

        elif index == self.INDEX_LAT_LON_ALT:
            # Index 20: Lat, Lon, Altitude
            # [0] lat deg, [1] lon deg, [2] alt ftmsl, [3] alt ftagl,
            # [4-7] (unused)
            self.flight_data.lat = values[0]
            self.flight_data.lon = values[1]
            self.flight_data.alt_msl = values[2]
            self.flight_data.alt_agl = values[3]

        # ==================== WEATHER DATA ====================

        elif index == self.INDEX_ATMOSPHERE:
            # Index 13: Atmosphere
            # [0] SLprs (inHg), [1] SLtmp (°C), [2] LEtmp (°C at aircraft),
            # [3] LErho (kg/m³), [4] Qstatic (??), [5] Aspeed (ktas for sound),
            # [6] (unused), [7] (unused)
            self.flight_data.sea_level_pressure = values[0]
            self.flight_data.sea_level_temp = values[1]
            self.flight_data.local_temp = values[2]
            self.flight_data.local_density = values[3]
            self.flight_data.speed_of_sound = values[5]

        elif index == self.INDEX_WEATHER:
            # Index 48: Weather/visibility
            # [0] visibility (sm), [1] precip (0-1), [2] runway friction,
            # [3-7] (varies by X-Plane version)
            vis = values[0]
            if vis > 0 and vis < 1000:  # Sanity check
                self.flight_data.visibility_sm = vis
            precip = values[1]
            if 0 <= precip <= 1:
                self.flight_data.precipitation_rate = precip
                if precip > 0.5:
                    self.flight_data.precipitation_type = "rain"
                elif precip > 0:
                    self.flight_data.precipitation_type = "light"
                else:
                    self.flight_data.precipitation_type = "none"

        elif index == self.INDEX_CLOUD_LAYER_0:
            # Index 63: Cloud layer 0
            # [0] type, [1] tops (ft MSL), [2] bases (ft MSL), [3] coverage (0-1)
            # [4-7] (unused)
            self.flight_data.cloud_layer_0 = {
                "type": int(values[0]) if values[0] >= 0 else 0,
                "top_ft": values[1] if values[1] > 0 else 0,
                "base_ft": values[2] if values[2] > 0 else 0,
                "coverage": max(0, min(1, values[3]))
            }

        elif index == self.INDEX_CLOUD_LAYER_1:
            # Index 64: Cloud layer 1
            self.flight_data.cloud_layer_1 = {
                "type": int(values[0]) if values[0] >= 0 else 0,
                "top_ft": values[1] if values[1] > 0 else 0,
                "base_ft": values[2] if values[2] > 0 else 0,
                "coverage": max(0, min(1, values[3]))
            }

        elif index == self.INDEX_CLOUD_LAYER_2:
            # Index 65: Cloud layer 2
            self.flight_data.cloud_layer_2 = {
                "type": int(values[0]) if values[0] >= 0 else 0,
                "top_ft": values[1] if values[1] > 0 else 0,
                "base_ft": values[2] if values[2] > 0 else 0,
                "coverage": max(0, min(1, values[3]))
            }

        # NOTE: Index 45 is NOT wind data in X-Plane 12 - it appears to be fuel weight or similar
        # Wind data should be fetched from weather API based on aircraft position instead
        # elif index == self.INDEX_WIND_ALOFT:
        #     pass

    def get_flight_data(self) -> XPlaneFlightData:
        """Get current flight data"""
        return self.flight_data

    def is_connected(self) -> bool:
        """Check if receiving data from X-Plane"""
        return self.flight_data.connected


class XPlaneCommandSender:
    """
    Sends commands and dataref values to X-Plane via UDP.

    X-Plane UDP Command Format:
    - DREF packet: Set a dataref value
      Format: "DREF" + null + float32 value + dataref name (500 bytes total)
    - CMND packet: Trigger a command
      Format: "CMND" + null + command name (null terminated)

    Common Autopilot Datarefs:
    - sim/cockpit/autopilot/heading_mag: Heading bug value (degrees)
    - sim/cockpit/autopilot/altitude: Altitude bug value (feet)
    - sim/cockpit/autopilot/vertical_velocity: VS bug value (fpm)
    - sim/cockpit/autopilot/airspeed: Speed bug value (knots)

    Common Autopilot Commands:
    - sim/autopilot/heading: Toggle heading hold
    - sim/autopilot/NAV: Toggle nav mode
    - sim/autopilot/altitude_hold: Toggle altitude hold
    - sim/autopilot/servos_on: Engage autopilot
    - sim/autopilot/servos_off: Disengage autopilot
    """

    def __init__(self, xplane_ip: str = "127.0.0.1", xplane_port: int = 49000):
        self.xplane_ip = xplane_ip
        self.xplane_port = xplane_port
        self._socket = None

    def _ensure_socket(self):
        """Create UDP socket if not exists"""
        if self._socket is None:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_dref(self, dataref: str, value: float) -> bool:
        """
        Send a DREF packet to set a dataref value in X-Plane.

        Args:
            dataref: The dataref path (e.g., "sim/cockpit/autopilot/heading_mag")
            value: The float value to set

        Returns:
            True if packet was sent successfully
        """
        try:
            self._ensure_socket()

            # Build DREF packet: "DREF" + null + float32 + dataref (padded to 500 bytes)
            packet = b'DREF\x00'
            packet += struct.pack('<f', float(value))

            # Dataref name, null-terminated and padded to 500 bytes
            dataref_bytes = dataref.encode('utf-8') + b'\x00'
            dataref_bytes = dataref_bytes.ljust(500, b'\x00')
            packet += dataref_bytes

            self._socket.sendto(packet, (self.xplane_ip, self.xplane_port))
            logger.info(f"Sent DREF: {dataref} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error sending DREF: {e}")
            return False

    def send_command(self, command: str) -> bool:
        """
        Send a CMND packet to trigger a command in X-Plane.

        Args:
            command: The command path (e.g., "sim/autopilot/heading")

        Returns:
            True if packet was sent successfully
        """
        try:
            self._ensure_socket()

            # Build CMND packet: "CMND" + null + command name (null terminated)
            packet = b'CMND\x00'
            packet += command.encode('utf-8') + b'\x00'

            self._socket.sendto(packet, (self.xplane_ip, self.xplane_port))
            logger.info(f"Sent CMND: {command}")
            return True

        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

    def set_heading_bug(self, heading: float) -> bool:
        """Set the autopilot heading bug value (0-360 degrees)"""
        heading = heading % 360
        return self.send_dref("sim/cockpit/autopilot/heading_mag", heading)

    def set_altitude_bug(self, altitude: float) -> bool:
        """Set the autopilot altitude bug value (feet)"""
        return self.send_dref("sim/cockpit/autopilot/altitude", altitude)

    def set_vs_bug(self, vs: float) -> bool:
        """Set the autopilot vertical speed bug value (fpm)"""
        return self.send_dref("sim/cockpit/autopilot/vertical_velocity", vs)

    def set_speed_bug(self, speed: float) -> bool:
        """Set the autopilot speed bug value (knots)"""
        return self.send_dref("sim/cockpit/autopilot/airspeed", speed)

    def engage_heading_mode(self) -> bool:
        """Engage autopilot heading hold mode"""
        # First ensure flight director is on
        self.send_command("sim/autopilot/fdir_on")
        # Then engage heading select mode
        return self.send_command("sim/autopilot/heading")

    def engage_nav_mode(self) -> bool:
        """Engage autopilot nav mode (follow flight plan)"""
        return self.send_command("sim/autopilot/NAV")

    def engage_altitude_hold(self) -> bool:
        """Engage autopilot altitude hold mode"""
        return self.send_command("sim/autopilot/altitude_hold")

    def engage_vs_mode(self) -> bool:
        """Engage autopilot vertical speed mode"""
        return self.send_command("sim/autopilot/vertical_speed")

    def engage_autopilot(self) -> bool:
        """Engage autopilot servos (master on)"""
        # Turn on flight director first
        self.send_command("sim/autopilot/fdir_on")
        # Then engage servos
        return self.send_command("sim/autopilot/servos_on")

    def disengage_autopilot(self) -> bool:
        """Disengage autopilot servos (master off)"""
        return self.send_command("sim/autopilot/servos_off")

    def disengage_heading_mode(self) -> bool:
        """Disengage heading mode (back to wing level)"""
        return self.send_command("sim/autopilot/wing_leveler")

    def disengage_nav_mode(self) -> bool:
        """Disengage NAV mode"""
        return self.send_command("sim/autopilot/NAV")  # Toggle off

    def disengage_altitude_hold(self) -> bool:
        """Disengage altitude hold"""
        return self.send_command("sim/autopilot/altitude_hold")  # Toggle off

    def disengage_vs_mode(self) -> bool:
        """Disengage vertical speed mode"""
        return self.send_command("sim/autopilot/vertical_speed")  # Toggle off

    # ==================== POSITION CONTROL ====================

    def set_position(self, lat: float, lon: float, elevation_m: float, heading: float) -> bool:
        """
        Set aircraft position and orientation.
        Used to reposition aircraft to departure airport.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            elevation_m: Elevation in meters MSL
            heading: Heading in degrees true
        """
        try:
            # First, pause to prevent physics issues during teleport
            self.send_command("sim/operation/pause_toggle")

            # Set position
            self.send_dref("sim/flightmodel/position/latitude", lat)
            self.send_dref("sim/flightmodel/position/longitude", lon)
            self.send_dref("sim/flightmodel/position/elevation", elevation_m)
            self.send_dref("sim/flightmodel/position/psi", heading)  # True heading

            # Reset velocities to prevent crash
            self.send_dref("sim/flightmodel/position/local_vx", 0.0)
            self.send_dref("sim/flightmodel/position/local_vy", 0.0)
            self.send_dref("sim/flightmodel/position/local_vz", 0.0)

            # Reset angular velocities
            self.send_dref("sim/flightmodel/position/P", 0.0)  # Roll rate
            self.send_dref("sim/flightmodel/position/Q", 0.0)  # Pitch rate
            self.send_dref("sim/flightmodel/position/R", 0.0)  # Yaw rate

            # Set on ground
            self.send_dref("sim/flightmodel/position/y_agl", 0.0)

            # Unpause
            self.send_command("sim/operation/pause_toggle")

            logger.info(f"Position set: lat={lat}, lon={lon}, elev={elevation_m}m, hdg={heading}")
            return True

        except Exception as e:
            logger.error(f"Error setting position: {e}")
            return False

    # ==================== ENGINE CONTROL ====================

    async def start_engines_async(self) -> bool:
        """Start all engines with full startup sequence - async version that holds starter"""
        import asyncio
        try:
            # ===== STEP 1: POWER ON =====
            print("[ENGINE START] Step 1: Power on")
            self.send_dref("sim/cockpit/electrical/battery_on", 1)
            self.send_dref("sim/cockpit/electrical/battery_array_on[0]", 1)
            self.send_dref("sim/cockpit/electrical/generator_on[0]", 1)
            self.send_dref("sim/cockpit2/switches/avionics_power_on", 1)
            await asyncio.sleep(1.0)

            # ===== STEP 2: FUEL TANK TO RIGHT =====
            print("[ENGINE START] Step 2: Selecting RIGHT fuel tank")
            # Try EVERY possible fuel tank dataref
            self.send_dref("sim/cockpit2/fuel/fuel_tank_selector", 2)
            self.send_dref("sim/cockpit/fuel/fuel_tank_selector", 2)
            self.send_dref("sim/cockpit2/fuel/fuel_tank_selector[0]", 2)
            self.send_dref("sim/cockpit/fuel/fuel_tank_selector[0]", 2)
            # For aircraft with individual tank selectors
            self.send_dref("sim/cockpit2/fuel/fuel_tank_selector_left", 0)
            self.send_dref("sim/cockpit2/fuel/fuel_tank_selector_right", 1)
            # Commands
            self.send_command("sim/fuel/fuel_tank_selector_rgt")
            self.send_command("sim/fuel/fuel_selector_rgt")
            # Cirrus-specific - try clicking the fuel selector
            self.send_command("sim/fuel/fuel_tank_pump_1_on")
            await asyncio.sleep(1.0)

            # ===== STEP 3: FUEL & MIXTURE =====
            print("[ENGINE START] Step 3: Fuel pump ON, mixture RICH")
            self.send_dref("sim/cockpit/engine/fuel_pump_on", 1)
            self.send_dref("sim/cockpit2/fuel/fuel_tank_pump_on[0]", 1)
            self.send_dref("sim/cockpit2/engine/actuators/mixture_ratio_all", 1.0)
            self.send_dref("sim/cockpit2/engine/actuators/mixture_ratio[0]", 1.0)
            await asyncio.sleep(0.5)

            # ===== STEP 4: MAGNETOS/IGNITION =====
            print("[ENGINE START] Step 4: Ignition ON")
            self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 3)  # BOTH
            self.send_dref("sim/cockpit/engine/ignition_on", 1)
            await asyncio.sleep(0.5)

            # ===== STEP 5: CRANK ENGINE FOR 8 SECONDS =====
            print("[ENGINE START] Step 5: CRANKING ENGINE (holding for 8 seconds)...")

            # Set ignition key to START position and HOLD IT
            self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 4)  # START position

            # Also set starter running datarefs
            self.send_dref("sim/cockpit/engine/starter_on[0]", 1)
            self.send_dref("sim/cockpit2/engine/actuators/starter_hit[0]", 1)

            # Now wait while holding - just refresh the values periodically
            for second in range(8):
                # Keep the starter engaged by re-sending every second
                self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 4)
                self.send_dref("sim/cockpit/engine/starter_on[0]", 1)
                self.send_command("sim/starters/engage_starter_1")
                self.send_command("sim/ignition/engage_starter_1")

                print(f"  Cranking... {second + 1}s")
                await asyncio.sleep(1.0)

            # ===== STEP 6: RELEASE STARTER =====
            print("[ENGINE START] Step 6: Releasing starter")
            self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 3)  # Back to BOTH
            self.send_dref("sim/cockpit/engine/starter_on[0]", 0)
            self.send_dref("sim/cockpit2/engine/actuators/throttle_ratio_all", 0.1)
            await asyncio.sleep(1.0)

            print("[ENGINE START] Complete!")
            return True
        except Exception as e:
            print(f"[ENGINE START] ERROR: {e}")
            return False

    def start_engines(self) -> bool:
        """Start all engines - sync wrapper (use start_engines_async when possible)"""
        try:
            print("[ENGINE START] Step 1: Power on - battery and alternator")
            self.send_dref("sim/cockpit/electrical/battery_on", 1)
            self.send_dref("sim/cockpit/electrical/generator_on[0]", 1)
            self.send_dref("sim/cockpit2/switches/avionics_power_on", 1)

            print("[ENGINE START] Step 2: Fuel system - selecting RIGHT tank")
            self.send_dref("sim/cockpit2/fuel/fuel_tank_selector", 2)  # RIGHT tank
            self.send_dref("sim/cockpit/engine/fuel_pump_on", 1)
            self.send_dref("sim/cockpit2/engine/actuators/mixture_ratio_all", 1.0)  # Full rich

            print("[ENGINE START] Step 3: Magnetos to BOTH then START")
            self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 3)  # BOTH

            print("[ENGINE START] Step 4: Engage starter")
            self.send_command("sim/starters/engage_starter_1")
            self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", 4)  # START
            self.send_dref("sim/cockpit/engine/ignition_on", 1)

            print("[ENGINE START] Starter engaged (sync mode)")
            return True
        except Exception as e:
            print(f"[ENGINE START] ERROR: {e}")
            return False

    def set_mixture(self, ratio: float) -> bool:
        """Set mixture for all engines (0.0 = cutoff, 1.0 = full rich)"""
        ratio = max(0.0, min(1.0, ratio))
        return self.send_dref("sim/cockpit2/engine/actuators/mixture_ratio_all", ratio)

    def set_magnetos(self, position: int) -> bool:
        """Set magnetos (0=off, 1=left, 2=right, 3=both, 4=start)"""
        return self.send_dref("sim/cockpit2/engine/actuators/ignition_key[0]", position)

    # ==================== FLIGHT CONTROLS ====================

    def set_throttle(self, ratio: float) -> bool:
        """
        Set throttle position for all engines.

        Args:
            ratio: Throttle ratio 0.0 (idle) to 1.0 (full)
        """
        ratio = max(0.0, min(1.0, ratio))
        return self.send_dref("sim/cockpit2/engine/actuators/throttle_ratio_all", ratio)

    def set_throttle_smoothly(self, target: float, current: float, step: float = 0.05) -> float:
        """
        Increment throttle toward target value.
        Returns new throttle value.

        Args:
            target: Target throttle ratio (0.0-1.0)
            current: Current throttle ratio
            step: Increment per call
        """
        if current < target:
            new_value = min(current + step, target)
        else:
            new_value = max(current - step, target)
        self.set_throttle(new_value)
        return new_value

    def set_flaps(self, ratio: float) -> bool:
        """
        Set flap position.

        Args:
            ratio: Flap ratio 0.0 (up) to 1.0 (full)
        """
        ratio = max(0.0, min(1.0, ratio))
        return self.send_dref("sim/flightmodel/controls/flaprqst", ratio)

    def set_parking_brake(self, engaged: bool) -> bool:
        """Set parking brake on/off"""
        print(f"[PARKING BRAKE] {'ENGAGED' if engaged else 'Releasing...'}")
        if engaged:
            self.send_dref("sim/flightmodel/controls/parkbrake", 1.0)
            self.send_dref("sim/cockpit2/controls/parking_brake_ratio", 1.0)
        else:
            # Use BOTH commands and datarefs for maximum compatibility
            # Commands
            self.send_command("sim/flight_controls/brakes_off")
            # Datarefs - set everything to 0
            self.send_dref("sim/flightmodel/controls/parkbrake", 0.0)
            self.send_dref("sim/cockpit2/controls/parking_brake_ratio", 0.0)
            self.send_dref("sim/cockpit2/controls/left_brake_ratio", 0.0)
            self.send_dref("sim/cockpit2/controls/right_brake_ratio", 0.0)
            self.send_dref("sim/flightmodel/controls/l_brake_add", 0.0)
            self.send_dref("sim/flightmodel/controls/r_brake_add", 0.0)
            # Also try the toe brakes
            self.send_dref("sim/cockpit2/controls/toe_brakes_left", 0.0)
            self.send_dref("sim/cockpit2/controls/toe_brakes_right", 0.0)
            print("[PARKING BRAKE] RELEASED (command + datarefs)")
        return True

    def release_parking_brake(self) -> bool:
        """Release parking brake and all wheel brakes"""
        print("[PARKING BRAKE] Releasing all brakes...")
        # Use command first
        self.send_command("sim/flight_controls/brakes_off")
        # Then datarefs
        self.send_dref("sim/flightmodel/controls/parkbrake", 0.0)
        self.send_dref("sim/cockpit2/controls/parking_brake_ratio", 0.0)
        self.send_dref("sim/cockpit2/controls/left_brake_ratio", 0.0)
        self.send_dref("sim/cockpit2/controls/right_brake_ratio", 0.0)
        self.send_dref("sim/flightmodel/controls/l_brake_add", 0.0)
        self.send_dref("sim/flightmodel/controls/r_brake_add", 0.0)
        self.send_dref("sim/cockpit2/controls/toe_brakes_left", 0.0)
        self.send_dref("sim/cockpit2/controls/toe_brakes_right", 0.0)
        print("[PARKING BRAKE] ALL BRAKES RELEASED")
        return True

    def set_gear(self, down: bool) -> bool:
        """
        Set landing gear position.

        Args:
            down: True for gear down, False for gear up
        """
        return self.send_dref("sim/cockpit/switches/gear_handle_status", 1.0 if down else 0.0)

    def gear_up(self) -> bool:
        """Retract landing gear"""
        return self.set_gear(False)

    def gear_down(self) -> bool:
        """Extend landing gear"""
        return self.set_gear(True)

    def set_pitch_trim(self, trim: float) -> bool:
        """
        Set pitch trim.

        Args:
            trim: Trim value -1.0 (nose down) to 1.0 (nose up)
        """
        trim = max(-1.0, min(1.0, trim))
        return self.send_dref("sim/cockpit2/controls/elevator_trim", trim)

    def set_rudder(self, value: float) -> bool:
        """
        Set rudder position for yaw control.

        Args:
            value: Rudder value -1.0 (full left) to 1.0 (full right)
        """
        value = max(-1.0, min(1.0, value))
        # Override joystick to take control
        self.send_dref("sim/operation/override/override_joystick", 1)
        # Use yoke_heading_ratio only (simpler, avoids conflicts)
        self.send_dref("sim/cockpit2/controls/yoke_heading_ratio", value)
        return True

    def set_nosewheel_steering(self, value: float) -> bool:
        """
        Set nosewheel steering for ground control.

        Args:
            value: Steering value -1.0 (full left) to 1.0 (full right)
        """
        value = max(-1.0, min(1.0, value))
        return self.send_dref("sim/flightmodel/controls/nwheel_steer", value)

    def set_elevator(self, value: float) -> bool:
        """
        Set elevator position for pitch control.

        Args:
            value: Elevator value -1.0 (full down/nose down) to 1.0 (full up/nose up)
        """
        value = max(-1.0, min(1.0, value))
        # Override joystick to take control
        self.send_dref("sim/operation/override/override_joystick", 1)
        # Only use yoke_pitch_ratio - simplest and most reliable
        # X-Plane: positive = pull back = nose UP
        self.send_dref("sim/cockpit2/controls/yoke_pitch_ratio", value)
        return True

    def set_aileron(self, value: float) -> bool:
        """
        Set aileron position for roll/bank control.

        Args:
            value: Aileron value -1.0 (full left/bank left) to 1.0 (full right/bank right)
        """
        value = max(-1.0, min(1.0, value))
        # Override joystick to take control
        self.send_dref("sim/operation/override/override_joystick", 1)
        # Use yoke_roll_ratio only (simpler, avoids conflicts with flightmodel controls)
        # Positive value = roll right, negative = roll left
        self.send_dref("sim/cockpit2/controls/yoke_roll_ratio", value)
        return True

    def engage_wing_leveler(self) -> bool:
        """Engage wing leveler mode to keep wings level"""
        print("[AUTOPILOT] Engaging wing leveler (LVL)")
        # First ensure autopilot is on
        self.send_command("sim/autopilot/fdir_on")
        self.send_command("sim/autopilot/servos_on")
        # Then engage wing leveler
        return self.send_command("sim/autopilot/wing_leveler")

    def release_joystick_override(self) -> bool:
        """
        Release joystick override - gives control back to the user's hardware.
        Call this when you want the pilot to take over.
        """
        print("[CONTROL] Releasing joystick override - PILOT HAS CONTROL")
        # Release override
        self.send_dref("sim/operation/override/override_joystick", 0)
        # Also reset all control surfaces to neutral to avoid sudden movements
        self.send_dref("sim/cockpit2/controls/yoke_pitch_ratio", 0)
        self.send_dref("sim/cockpit2/controls/yoke_roll_ratio", 0)
        self.send_dref("sim/cockpit2/controls/yoke_heading_ratio", 0)
        return True

    def engage_autopilot_for_handoff(self) -> bool:
        """
        Engage X-Plane's autopilot before releasing manual control.
        This prevents the plane from crashing when automation stops.
        """
        print("[CONTROL] Engaging X-Plane autopilot for handoff...")
        # Engage flight director
        self.send_command("sim/autopilot/fdir_on")
        # Engage autopilot servos
        self.send_command("sim/autopilot/servos_on")
        # Engage altitude hold (hold current altitude)
        self.send_command("sim/autopilot/altitude_arm")
        self.send_command("sim/autopilot/altitude_hold")
        # Engage heading hold
        self.send_command("sim/autopilot/heading")
        return True

    def close(self):
        """Close the UDP socket"""
        if self._socket:
            self._socket.close()
            self._socket = None
