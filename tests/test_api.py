"""
API Endpoint Tests for Aviation Weather API.
DO-178C DAL D Compliance: Software Verification Process - Test Coverage Analysis.

These tests verify the core functionality of all API endpoints to ensure
compliance with ASTM F3269 (Data Quality) and ASTM F3153 (Weather Information).
"""

import pytest


class TestHealthEndpoints:
    """Tests for health monitoring endpoints (ASTM F3269 compliance)."""

    def test_root_endpoint(self, client):
        """Test that root endpoint serves the frontend."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_health_dashboard(self, client):
        """Test health dashboard endpoint."""
        response = client.get("/health-dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_data_health_endpoint(self, client):
        """Test /api/data-health returns valid health data."""
        response = client.get("/api/data-health")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "uptime_seconds" in data
        assert "endpoints" in data
        assert "external_services" in data

    def test_external_services_connectivity(self, client):
        """Test that external services connectivity is checked."""
        response = client.get("/api/data-health")
        assert response.status_code == 200
        data = response.json()

        # Should have at least 2 external services
        assert len(data["external_services"]) >= 2


class TestComplianceEndpoints:
    """Tests for regulatory compliance endpoints (DO-178C & ASTM)."""

    def test_compliance_dashboard(self, client):
        """Test compliance dashboard endpoint."""
        response = client.get("/compliance-dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_compliance_api(self, client):
        """Test /api/compliance returns valid compliance data."""
        response = client.get("/api/compliance")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "overall_score" in data
        assert "static_compliance" in data
        assert "dynamic_compliance" in data
        assert "summary" in data

    def test_compliance_score_range(self, client):
        """Test that compliance score is within valid range."""
        response = client.get("/api/compliance")
        data = response.json()

        assert 0 <= data["overall_score"] <= 100

    def test_do178c_dal_d_requirements(self, client):
        """Test that DO-178C DAL D requirements are present."""
        response = client.get("/api/compliance")
        data = response.json()

        assert "do178c_dal_d" in data["static_compliance"]
        dal_d = data["static_compliance"]["do178c_dal_d"]

        # Should have 9 DAL D requirements
        assert len(dal_d) == 9

        # Verify requirement structure
        for req in dal_d:
            assert "id" in req
            assert "requirement" in req
            assert "status" in req
            assert req["status"] in ["compliant", "partial", "not_applicable", "reviewed"]

    def test_astm_f3269_requirements(self, client):
        """Test that ASTM F3269 Data Quality requirements are present."""
        response = client.get("/api/compliance")
        data = response.json()

        assert "astm_f3269_data_quality" in data["static_compliance"]
        f3269 = data["static_compliance"]["astm_f3269_data_quality"]

        # Should have 8 requirements
        assert len(f3269) == 8

    def test_astm_f3153_requirements(self, client):
        """Test that ASTM F3153 Weather requirements are present."""
        response = client.get("/api/compliance")
        data = response.json()

        assert "astm_f3153_weather" in data["static_compliance"]
        f3153 = data["static_compliance"]["astm_f3153_weather"]

        # Should have 9 requirements
        assert len(f3153) == 9

    def test_dynamic_compliance_checks(self, client):
        """Test that dynamic compliance checks are present."""
        response = client.get("/api/compliance")
        data = response.json()

        assert len(data["dynamic_compliance"]) == 4

        check_names = [c["check_name"] for c in data["dynamic_compliance"]]
        assert "API Success Rate" in check_names
        assert "Average Response Time" in check_names
        assert "External Service Availability" in check_names
        assert "METAR Data Freshness" in check_names


class TestWeatherEndpoints:
    """Tests for weather data endpoints (ASTM F3153 compliance)."""

    def test_metar_endpoint_valid_icao(self, client, sample_icao):
        """Test METAR endpoint with valid ICAO code."""
        response = client.get(f"/api/metar/{sample_icao}")
        # May be 200 or 503/404 depending on external service availability
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "icao" in data
            assert "raw_text" in data  # ASTM F3153-3: Raw text preservation

    def test_metar_endpoint_invalid_icao(self, client):
        """Test METAR endpoint with invalid ICAO code."""
        response = client.get("/api/metar/INVALID")
        assert response.status_code in [404, 503]

    def test_taf_endpoint_valid_icao(self, client, sample_icao):
        """Test TAF endpoint with valid ICAO code."""
        response = client.get(f"/api/taf/{sample_icao}")
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "icao" in data
            assert "raw_text" in data  # ASTM F3153-3: Raw text preservation

    def test_forecast_endpoint(self, client, sample_coordinates):
        """Test forecast endpoint with coordinates."""
        response = client.get(
            "/api/weather/forecast",
            params={"latitude": sample_coordinates["lat"], "longitude": sample_coordinates["lon"]}
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Response is a list of WeatherForecast objects
            assert isinstance(data, list)


class TestFAARegulations:
    """Tests for FAA regulation checking endpoints."""

    def test_vfr_regulations_check(self, client, sample_icao):
        """Test VFR regulations check endpoint."""
        response = client.get(
            "/api/regulations/check/vfr",
            params={"station": sample_icao, "altitude_agl": 3000, "airspace_class": "E"}
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Response is a list of FAARegulationCheck objects
            assert isinstance(data, list)
            if len(data) > 0:
                assert "regulation" in data[0]

    def test_fuel_regulations_check(self, client):
        """Test fuel regulations check endpoint."""
        response = client.get(
            "/api/regulations/check/fuel",
            params={"flight_type": "vfr", "flight_time_minutes": 120}
        )
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            data = response.json()
            assert "regulation" in data


class TestDataValidation:
    """Tests for data validation (ASTM F3269 compliance)."""

    def test_metar_data_validation(self, client, sample_icao):
        """Test that METAR data is properly validated (F3269-3)."""
        response = client.get(f"/api/metar/{sample_icao}")

        if response.status_code == 200:
            data = response.json()
            # Pydantic validation ensures type safety
            assert isinstance(data.get("icao", ""), str)
            if "temperature_c" in data and data["temperature_c"] is not None:
                assert isinstance(data["temperature_c"], (int, float))

    def test_error_response_format(self, client):
        """Test that errors return proper HTTP status codes (F3269-4, F3269-5)."""
        response = client.get("/api/metar/XXXX")  # Non-existent airport

        # Should return appropriate error code, not 500
        assert response.status_code in [404, 503]
        assert response.status_code != 500


class TestAPIDocumentation:
    """Tests for API documentation (DO-D-2: Software Requirements Standards)."""

    def test_openapi_spec_available(self, client):
        """Test that OpenAPI specification is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_swagger_ui_available(self, client):
        """Test that Swagger UI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
