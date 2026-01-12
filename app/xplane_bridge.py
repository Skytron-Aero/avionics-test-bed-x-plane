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
                # Use true heading as magnetic (close enough for display purposes)
                # The UI can apply mag_var if needed
                self.flight_data.heading_mag = values[2]

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

    def close(self):
        """Close the UDP socket"""
        if self._socket:
            self._socket.close()
            self._socket = None
