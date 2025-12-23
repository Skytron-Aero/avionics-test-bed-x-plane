"""
Pytest configuration and fixtures for Aviation Weather API tests.
DO-178C DAL D Compliance: Test infrastructure for verification process.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_icao():
    """Sample ICAO airport code for testing."""
    return "KJFK"


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing (JFK Airport)."""
    return {"lat": 40.6413, "lon": -73.7781}
