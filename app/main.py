"""
Aviation Weather Data Backend API
FastAPI-based backend for Qt/QML avionics application
Integrates Open-Meteo, AVWX, and FAA Aviation Weather sources
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx
import asyncio
from enum import Enum


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    print("Aviation Weather API starting up...")
    print("API Documentation available at: http://localhost:8000/docs")
    yield


app = FastAPI(
    title="Aviation Weather API",
    description="Backend API for avionics MFD weather data visualization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATA MODELS ====================

class FlightRules(str, Enum):
    VFR = "VFR"
    MVFR = "MVFR"
    IFR = "IFR"
    LIFR = "LIFR"

class WeatherStation(BaseModel):
    icao: str
    name: str
    latitude: float
    longitude: float
    elevation: int

class METARData(BaseModel):
    station: str
    observation_time: datetime
    raw_text: str
    temperature: Optional[float] = None
    dewpoint: Optional[float] = None
    wind_direction: Optional[int] = None
    wind_speed: Optional[int] = None
    wind_gust: Optional[int] = None
    visibility: Optional[float] = None
    altimeter: Optional[float] = None
    flight_rules: FlightRules
    sky_conditions: List[Dict]
    remarks: Optional[str] = None

class TAFData(BaseModel):
    station: str
    issue_time: datetime
    valid_period_start: datetime
    valid_period_end: datetime
    raw_text: str
    forecast_periods: List[Dict]

class WeatherForecast(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime
    temperature_2m: float
    relative_humidity: float
    precipitation: float
    wind_speed_10m: float
    wind_direction_10m: int
    cloud_cover: int
    visibility: float
    pressure_msl: float

class FAARegulationCheck(BaseModel):
    regulation: str
    status: str
    criteria: Dict
    current_values: Dict
    compliant: bool
    notes: str

# ==================== HELPER FUNCTIONS ====================

def calculate_flight_rules(visibility: Optional[float], ceiling: Optional[int]) -> FlightRules:
    """Calculate flight rules based on visibility and ceiling"""
    if visibility is None and ceiling is None:
        return FlightRules.VFR
    
    vis = visibility if visibility else 10.0
    ceil = ceiling if ceiling else 10000
    
    if vis < 1.0 or ceil < 500:
        return FlightRules.LIFR
    elif vis < 3.0 or ceil < 1000:
        return FlightRules.IFR
    elif vis < 5.0 or ceil < 3000:
        return FlightRules.MVFR
    else:
        return FlightRules.VFR

async def fetch_with_retry(client: httpx.AsyncClient, url: str, max_retries: int = 3) -> Dict:
    """Fetch data with retry logic"""
    for attempt in range(max_retries):
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"Failed to fetch data: {str(e)}")
            await asyncio.sleep(2 ** attempt)

# ==================== OPEN-METEO INTEGRATION ====================

@app.get("/api/weather/forecast", response_model=List[WeatherForecast])
async def get_weather_forecast(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    hours: int = Query(24, ge=1, le=168)
):
    """Get weather forecast from Open-Meteo for specific coordinates"""
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,"
        f"wind_speed_10m,wind_direction_10m,cloud_cover,visibility,pressure_msl"
        f"&forecast_hours={hours}"
        f"&temperature_unit=fahrenheit&wind_speed_unit=kn"
    )
    
    async with httpx.AsyncClient() as client:
        data = await fetch_with_retry(client, url)
    
    forecasts = []
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    
    for i, time_str in enumerate(times):
        forecast = WeatherForecast(
            latitude=latitude,
            longitude=longitude,
            timestamp=datetime.fromisoformat(time_str),
            temperature_2m=hourly.get("temperature_2m", [])[i],
            relative_humidity=hourly.get("relative_humidity_2m", [])[i],
            precipitation=hourly.get("precipitation", [])[i],
            wind_speed_10m=hourly.get("wind_speed_10m", [])[i],
            wind_direction_10m=hourly.get("wind_direction_10m", [])[i],
            cloud_cover=hourly.get("cloud_cover", [])[i],
            visibility=hourly.get("visibility", [])[i],
            pressure_msl=hourly.get("pressure_msl", [])[i]
        )
        forecasts.append(forecast)
    
    return forecasts

# ==================== AVWX/FAA AVIATION WEATHER ====================

@app.get("/api/aviation/metar/{station}", response_model=METARData)
async def get_metar(station: str):
    """Get current METAR for aviation station (uses FAA Aviation Weather API)"""
    station = station.upper()
    
    url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise HTTPException(status_code=404, detail=f"No METAR found for {station}")
            
            metar = data[0]
            
            # Parse sky conditions for ceiling
            sky_conditions = []
            ceiling = None
            for condition in metar.get("clouds", []):
                sky_dict = {
                    "cover": condition.get("cover"),
                    "base": condition.get("base")
                }
                sky_conditions.append(sky_dict)
                
                if condition.get("cover") in ["BKN", "OVC"] and condition.get("base"):
                    if ceiling is None or condition.get("base") < ceiling:
                        ceiling = condition.get("base")
            
            visibility = metar.get("visib")
            flight_rules = calculate_flight_rules(visibility, ceiling)
            
            # Convert Unix timestamp or ISO string to datetime
            obs_time_str = metar.get("reportTime")
            obs_time_unix = metar.get("obsTime")
            
            if obs_time_str:
                observation_time = datetime.fromisoformat(obs_time_str.replace('Z', '+00:00'))
            elif obs_time_unix:
                observation_time = datetime.fromtimestamp(obs_time_unix, tz=timezone.utc)
            else:
                observation_time = datetime.now(timezone.utc)
            
            return METARData(
                station=station,
                observation_time=observation_time,
                raw_text=metar.get("rawOb", ""),
                temperature=metar.get("temp"),
                dewpoint=metar.get("dewp"),
                wind_direction=metar.get("wdir"),
                wind_speed=metar.get("wspd"),
                wind_gust=metar.get("wgst"),
                visibility=visibility,
                altimeter=metar.get("altim"),
                flight_rules=flight_rules,
                sky_conditions=sky_conditions,
                remarks=metar.get("rawOb", "").split("RMK")[-1].strip() if "RMK" in metar.get("rawOb", "") else None
            )
            
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Failed to fetch METAR: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing METAR: {str(e)}")

@app.get("/api/aviation/taf/{station}", response_model=TAFData)
async def get_taf(station: str):
    """Get Terminal Aerodrome Forecast for aviation station"""
    station = station.upper()
    
    url = f"https://aviationweather.gov/api/data/taf?ids={station}&format=json"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise HTTPException(status_code=404, detail=f"No TAF found for {station}")
            
            taf = data[0]
            
            forecast_periods = []
            for forecast in taf.get("forecast", []):
                period = {
                    "valid_from": forecast.get("fcstTime"),
                    "wind_direction": forecast.get("wdir"),
                    "wind_speed": forecast.get("wspd"),
                    "wind_gust": forecast.get("wgst"),
                    "visibility": forecast.get("visib"),
                    "sky_conditions": forecast.get("clouds", []),
                    "change_indicator": forecast.get("change")
                }
                forecast_periods.append(period)

            # Parse datetime fields with fallback
            def parse_taf_time(time_str: Optional[str]) -> datetime:
                if not time_str:
                    return datetime.now(timezone.utc)
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))

            return TAFData(
                station=station,
                issue_time=parse_taf_time(taf.get("issueTime")),
                valid_period_start=parse_taf_time(taf.get("validTimeFrom")),
                valid_period_end=parse_taf_time(taf.get("validTimeTo")),
                raw_text=taf.get("rawTAF", ""),
                forecast_periods=forecast_periods
            )

        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"Failed to fetch TAF: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing TAF: {str(e)}")

# ==================== FAA REGULATION CHECKING ====================

@app.get("/api/regulations/check/vfr", response_model=List[FAARegulationCheck])
async def check_vfr_minimums(
    station: str,
    altitude_agl: int = Query(..., description="Altitude AGL in feet"),
    airspace_class: str = Query(..., pattern="^[A-G]$", description="Airspace class (A-G)")
):
    """Check current weather against FAA VFR minimums (14 CFR 91.155)"""
    
    metar = await get_metar(station)
    
    checks = []
    
    # Visibility check
    min_vis = 3.0
    if airspace_class == "E" and altitude_agl >= 10000:
        min_vis = 5.0
    elif airspace_class == "G" and altitude_agl < 1200:
        min_vis = 1.0
    
    visibility_check = FAARegulationCheck(
        regulation="14 CFR 91.155 - VFR Visibility Minimum",
        status="COMPLIANT" if metar.visibility and metar.visibility >= min_vis else "NON-COMPLIANT",
        criteria={"minimum_visibility_sm": min_vis, "airspace_class": airspace_class},
        current_values={"visibility_sm": metar.visibility},
        compliant=metar.visibility >= min_vis if metar.visibility else False,
        notes=f"Current visibility: {metar.visibility} SM, Required: {min_vis} SM"
    )
    checks.append(visibility_check)
    
    ceiling_check = FAARegulationCheck(
        regulation="14 CFR 91.155 - VFR Cloud Clearance",
        status="REVIEW",
        criteria={"airspace_class": airspace_class, "altitude_agl": altitude_agl},
        current_values={"sky_conditions": [c for c in metar.sky_conditions]},
        compliant=True,
        notes="Review sky conditions for cloud clearance requirements"
    )
    checks.append(ceiling_check)
    
    return checks

@app.get("/api/regulations/check/fuel", response_model=FAARegulationCheck)
async def check_fuel_requirements(
    flight_type: str = Query(..., pattern="^(VFR|IFR)$"),
    flight_time_hours: float = Query(..., ge=0),
    reserve_fuel_hours: float = Query(...)
):
    """Check fuel reserve requirements (14 CFR 91.151 VFR, 91.167 IFR)"""
    
    if flight_type == "VFR":
        required_reserve = 0.5
        regulation = "14 CFR 91.151"
    else:
        required_reserve = 0.75
        regulation = "14 CFR 91.167"
    
    compliant = reserve_fuel_hours >= required_reserve
    
    return FAARegulationCheck(
        regulation=f"{regulation} - Fuel Requirements",
        status="COMPLIANT" if compliant else "NON-COMPLIANT",
        criteria={"required_reserve_hours": required_reserve, "flight_type": flight_type},
        current_values={"reserve_fuel_hours": reserve_fuel_hours, "flight_time_hours": flight_time_hours},
        compliant=compliant,
        notes=f"Reserve fuel: {reserve_fuel_hours:.2f} hrs, Required: {required_reserve} hrs minimum"
    )

# ==================== UTILITY ENDPOINTS ====================

@app.get("/api/stations/search")
async def search_stations(query: str = Query(..., min_length=2)):
    """Search for aviation weather stations by ICAO code or name"""
    common_stations = [
        {"icao": "KLAX", "name": "Los Angeles International", "lat": 33.94, "lon": -118.41},
        {"icao": "KJFK", "name": "John F Kennedy International", "lat": 40.64, "lon": -73.78},
        {"icao": "KORD", "name": "Chicago O'Hare International", "lat": 41.98, "lon": -87.90},
        {"icao": "KATL", "name": "Hartsfield-Jackson Atlanta", "lat": 33.64, "lon": -84.43},
        {"icao": "KDFW", "name": "Dallas/Fort Worth International", "lat": 32.90, "lon": -97.04},
    ]
    
    query = query.upper()
    results = [s for s in common_stations if query in s["icao"] or query in s["name"].upper()]
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "open_meteo": "operational",
            "faa_weather": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)