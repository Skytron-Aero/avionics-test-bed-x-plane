"""
Database models for caching weather data locally
SQLAlchemy models for SQLite/PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# ==================== WEATHER STATION MODEL ====================

class WeatherStation(Base):
    __tablename__ = "weather_stations"
    
    id = Column(Integer, primary_key=True)
    icao = Column(String(4), unique=True, index=True, nullable=False)
    name = Column(String(200))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Integer)
    country = Column(String(2))
    state = Column(String(2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ==================== METAR CACHE MODEL ====================

class METARCache(Base):
    __tablename__ = "metar_cache"
    
    id = Column(Integer, primary_key=True)
    station_icao = Column(String(4), index=True, nullable=False)
    observation_time = Column(DateTime, nullable=False, index=True)
    raw_text = Column(Text, nullable=False)
    
    # Parsed data
    temperature = Column(Float)
    dewpoint = Column(Float)
    wind_direction = Column(Integer)
    wind_speed = Column(Integer)
    wind_gust = Column(Integer)
    visibility = Column(Float)
    altimeter = Column(Float)
    flight_rules = Column(String(10))
    
    # JSON fields for complex data
    sky_conditions = Column(JSON)
    weather_phenomena = Column(JSON)
    remarks = Column(Text)
    
    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50), default="FAA")
    
    def __repr__(self):
        return f"<METAR {self.station_icao} {self.observation_time}>"

# ==================== TAF CACHE MODEL ====================

class TAFCache(Base):
    __tablename__ = "taf_cache"
    
    id = Column(Integer, primary_key=True)
    station_icao = Column(String(4), index=True, nullable=False)
    issue_time = Column(DateTime, nullable=False, index=True)
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=False)
    raw_text = Column(Text, nullable=False)
    
    # Forecast periods stored as JSON
    forecast_periods = Column(JSON)
    
    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(50), default="FAA")
    
    def __repr__(self):
        return f"<TAF {self.station_icao} {self.issue_time}>"

# ==================== WEATHER FORECAST CACHE ====================

class WeatherForecastCache(Base):
    __tablename__ = "weather_forecast_cache"
    
    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    forecast_time = Column(DateTime, nullable=False, index=True)
    
    # Weather parameters
    temperature_2m = Column(Float)
    relative_humidity = Column(Float)
    precipitation = Column(Float)
    wind_speed_10m = Column(Float)
    wind_direction_10m = Column(Integer)
    cloud_cover = Column(Integer)
    visibility = Column(Float)
    pressure_msl = Column(Float)
    
    # Additional parameters
    dewpoint = Column(Float)
    surface_pressure = Column(Float)
    precipitation_probability = Column(Integer)
    
    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    model_source = Column(String(50), default="OpenMeteo")
    
    def __repr__(self):
        return f"<Forecast {self.latitude},{self.longitude} {self.forecast_time}>"

# ==================== AIRMET/SIGMET CACHE ====================

class AirmetSigmetCache(Base):
    __tablename__ = "airmet_sigmet_cache"
    
    id = Column(Integer, primary_key=True)
    hazard_type = Column(String(20), nullable=False)  # AIRMET, SIGMET, CONVECTIVE_SIGMET
    issue_time = Column(DateTime, nullable=False, index=True)
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=False)
    
    # Geographic data
    region = Column(String(100))
    affected_states = Column(JSON)
    coordinates = Column(JSON)  # Polygon or area definition
    
    # Hazard information
    hazard = Column(String(50))  # Icing, Turbulence, IFR, Mountain Obscuration, etc.
    severity = Column(String(20))
    flight_levels = Column(String(50))
    
    raw_text = Column(Text)
    
    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<{self.hazard_type} {self.hazard} {self.issue_time}>"

# ==================== REGULATION CHECK LOG ====================

class RegulationCheckLog(Base):
    __tablename__ = "regulation_check_log"
    
    id = Column(Integer, primary_key=True)
    check_time = Column(DateTime, default=datetime.utcnow, index=True)
    regulation = Column(String(100), nullable=False)
    
    # Check parameters
    station_icao = Column(String(4))
    latitude = Column(Float)
    longitude = Column(Float)
    airspace_class = Column(String(1))
    altitude_agl = Column(Integer)
    
    # Check results
    status = Column(String(20))  # COMPLIANT, NON_COMPLIANT, WARNING
    compliant = Column(Boolean)
    criteria = Column(JSON)
    current_values = Column(JSON)
    notes = Column(Text)
    
    # User context (optional)
    aircraft_id = Column(String(50))
    flight_id = Column(String(50))
    
    def __repr__(self):
        return f"<RegCheck {self.regulation} {self.status}>"

# ==================== DATABASE INITIALIZATION ====================

class DatabaseManager:
    """Database manager for handling connections and sessions"""
    
    def __init__(self, database_url: str = "sqlite:///./aviation_weather.db"):
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManager("sqlite:///./aviation_weather.db")
    db_manager.create_tables()
    
    print("Database tables created successfully!")
    print("\nTables created:")
    print("- weather_stations")
    print("- metar_cache")
    print("- taf_cache")
    print("- weather_forecast_cache")
    print("- airmet_sigmet_cache")
    print("- regulation_check_log")