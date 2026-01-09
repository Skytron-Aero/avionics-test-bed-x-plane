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
- 20: Lat, lon, altitude MSL/AGL
"""

import asyncio
import struct
import logging
from dataclasses import dataclass, field, asdict
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

    # X-Plane data group indices
    INDEX_SPEEDS = 3          # IAS, TAS, true airspeed, ground speed, etc.
    INDEX_MACH_VVI_G = 4      # Mach, VVI, G-loads
    INDEX_PITCH_ROLL_HDG = 17  # Pitch, roll, true heading, magnetic heading
    INDEX_AOA_SIDESLIP = 18   # Alpha, beta, hpath, vpath
    INDEX_COMPASS = 19        # Compass heading, magnetic variation
    INDEX_LAT_LON_ALT = 20    # Lat, lon, alt MSL, alt AGL

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
            self.flight_data.heading_mag = values[3]

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
            compass_mag = values[0]
            mag_var = values[1]
            self.flight_data.mag_var = mag_var
            # Use compass magnetic heading if valid (not -999)
            if compass_mag > -900:
                self.flight_data.heading_mag = compass_mag

        elif index == self.INDEX_LAT_LON_ALT:
            # Index 20: Lat, Lon, Altitude
            # [0] lat deg, [1] lon deg, [2] alt ftmsl, [3] alt ftagl,
            # [4-7] (unused)
            self.flight_data.lat = values[0]
            self.flight_data.lon = values[1]
            self.flight_data.alt_msl = values[2]
            self.flight_data.alt_agl = values[3]

    def get_flight_data(self) -> XPlaneFlightData:
        """Get current flight data"""
        return self.flight_data

    def is_connected(self) -> bool:
        """Check if receiving data from X-Plane"""
        return self.flight_data.connected
