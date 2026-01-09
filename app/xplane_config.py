"""
X-Plane Bridge Configuration

Configuration settings for X-Plane 12 UDP integration.
Settings can be overridden via environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class XPlaneSettings(BaseSettings):
    """X-Plane connection settings"""

    # UDP listener settings - receives data FROM X-Plane
    XPLANE_UDP_HOST: str = "0.0.0.0"  # Listen on all interfaces
    XPLANE_UDP_PORT: int = 49000       # X-Plane default data output port

    # X-Plane PC settings (for potential future bidirectional control)
    XPLANE_PC_IP: Optional[str] = None  # IP of PC running X-Plane
    XPLANE_PC_PORT: int = 49000         # X-Plane command input port

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
