"""
X-Plane Bridge Configuration

Configuration settings for X-Plane 12 UDP integration.
Settings can be overridden via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional, Dict


class AircraftVSpeeds(BaseModel):
    """V-speeds for a specific aircraft type"""
    name: str
    vr: int  # Rotation speed (kts)
    v2: int  # Takeoff safety speed (kts)
    vx: int  # Best angle of climb (kts)
    vy: int  # Best rate of climb (kts)
    vref: int  # Reference landing speed (kts)
    vne: int  # Never exceed speed (kts)
    flaps_takeoff: float  # Flap setting for takeoff (0.0-1.0)
    flaps_landing: float  # Flap setting for landing (0.0-1.0)
    rotation_pitch: float  # Target pitch for rotation (degrees)
    climb_pitch: float  # Target pitch for climb (degrees)
    gear_retract_agl: int  # AGL to retract gear (ft)
    flaps_retract_agl: int  # AGL to retract flaps (ft)


# Pre-configured aircraft V-speeds
AIRCRAFT_PROFILES: Dict[str, AircraftVSpeeds] = {
    "cessna_172": AircraftVSpeeds(
        name="Cessna 172 Skyhawk",
        vr=55,
        v2=65,
        vx=62,
        vy=74,
        vref=65,
        vne=163,
        flaps_takeoff=0.0,
        flaps_landing=1.0,
        rotation_pitch=10,
        climb_pitch=8,
        gear_retract_agl=0,  # Fixed gear
        flaps_retract_agl=500,
    ),
    "cirrus_sr22": AircraftVSpeeds(
        name="Cirrus SR22",
        vr=70,
        v2=80,
        vx=78,
        vy=100,
        vref=80,
        vne=201,
        flaps_takeoff=0.0,  # Clean takeoff - reduces pitch sensitivity
        flaps_landing=1.0,
        rotation_pitch=8,   # Reduced for smoother rotation
        climb_pitch=6,
        gear_retract_agl=0,  # Fixed gear
        flaps_retract_agl=400,
    ),
    "baron_58": AircraftVSpeeds(
        name="Beechcraft Baron 58",
        vr=84,
        v2=96,
        vx=96,
        vy=110,
        vref=95,
        vne=223,
        flaps_takeoff=0.0,
        flaps_landing=1.0,
        rotation_pitch=10,
        climb_pitch=8,
        gear_retract_agl=200,
        flaps_retract_agl=500,
    ),
    "king_air_350": AircraftVSpeeds(
        name="Beechcraft King Air 350",
        vr=105,
        v2=115,
        vx=115,
        vy=135,
        vref=105,
        vne=263,
        flaps_takeoff=0.0,
        flaps_landing=1.0,
        rotation_pitch=12,
        climb_pitch=10,
        gear_retract_agl=200,
        flaps_retract_agl=1000,
    ),
    "lancair_evolution": AircraftVSpeeds(
        name="Lancair Evolution",
        vr=85,
        v2=95,
        vx=110,
        vy=140,
        vref=85,
        vne=270,
        flaps_takeoff=0.0,
        flaps_landing=1.0,
        rotation_pitch=8,
        climb_pitch=7,
        gear_retract_agl=200,
        flaps_retract_agl=500,
    ),
    "default": AircraftVSpeeds(
        name="Generic Light Aircraft",
        vr=60,
        v2=70,
        vx=70,
        vy=85,
        vref=70,
        vne=180,
        flaps_takeoff=0.0,
        flaps_landing=1.0,
        rotation_pitch=10,
        climb_pitch=8,
        gear_retract_agl=200,
        flaps_retract_agl=500,
    ),
}


def get_aircraft_profile(aircraft_type: str = "default") -> AircraftVSpeeds:
    """Get V-speeds for an aircraft type, falling back to default if not found"""
    return AIRCRAFT_PROFILES.get(aircraft_type, AIRCRAFT_PROFILES["default"])


class XPlaneSettings(BaseSettings):
    """X-Plane connection settings"""

    # UDP listener settings - receives data FROM X-Plane
    # NOTE: Use a different port than 49000 to avoid conflict with X-Plane's command port
    XPLANE_UDP_HOST: str = "0.0.0.0"  # Listen on all interfaces
    XPLANE_UDP_PORT: int = 49001       # Port to receive X-Plane data (configure X-Plane to send here)

    # X-Plane PC settings - for sending commands TO X-Plane
    # X-Plane always listens for commands on port 49000
    XPLANE_PC_IP: Optional[str] = None  # IP of PC running X-Plane (None = 127.0.0.1)
    XPLANE_PC_PORT: int = 49000         # X-Plane command input port (do not change)

    # Feature flags
    XPLANE_BRIDGE_ENABLED: bool = True  # Enable/disable X-Plane integration

    # Connection monitoring
    XPLANE_TIMEOUT_SECONDS: float = 5.0  # Consider disconnected after no data
    XPLANE_UPDATE_RATE_HZ: int = 30      # Target WebSocket broadcast rate to clients

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
xplane_settings = XPlaneSettings()
