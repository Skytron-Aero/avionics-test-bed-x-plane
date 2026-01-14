"""
Flight Manager Module

Handles automated flight operations including:
- Flight plan management
- Waypoint-based navigation
- Auto-takeoff sequence
- Flight phase management
"""

import asyncio
import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Tuple
from datetime import datetime

from .xplane_config import get_aircraft_profile, AircraftVSpeeds

logger = logging.getLogger(__name__)


class FlightPhase(Enum):
    """Flight phases for automation"""
    IDLE = "idle"
    PREFLIGHT = "preflight"
    TAXI = "taxi"
    TAKEOFF_ROLL = "takeoff_roll"
    ROTATION = "rotation"
    INITIAL_CLIMB = "initial_climb"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    COMPLETE = "complete"
    ABORTED = "aborted"


class TakeoffState(Enum):
    """Auto-takeoff state machine states"""
    IDLE = "idle"
    PREFLIGHT_CHECK = "preflight_check"
    BRAKE_SET = "brake_set"
    THROTTLE_UP = "throttle_up"
    BRAKE_RELEASE = "brake_release"
    TAKEOFF_ROLL = "takeoff_roll"
    ROTATION = "rotation"
    LIFTOFF = "liftoff"
    INITIAL_CLIMB = "initial_climb"
    GEAR_UP = "gear_up"
    FLAPS_UP = "flaps_up"
    CLIMB_OUT = "climb_out"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class Waypoint:
    """Represents a navigation waypoint"""
    name: str
    lat: float
    lon: float
    altitude_ft: int = 0
    waypoint_type: str = "enroute"  # departure, enroute, destination
    speed_kts: Optional[int] = None

    def __str__(self):
        return f"{self.name} ({self.lat:.4f}, {self.lon:.4f}) @ {self.altitude_ft}ft"


@dataclass
class FlightPlan:
    """Stores flight plan data"""
    waypoints: List[Waypoint] = field(default_factory=list)
    departure_icao: str = ""
    destination_icao: str = ""
    cruise_altitude_ft: int = 8000
    departure_runway: Optional[str] = None
    departure_heading: float = 0.0

    current_waypoint_index: int = 0
    phase: FlightPhase = FlightPhase.IDLE

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint"""
        if 0 <= self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None

    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get the next waypoint after current"""
        next_idx = self.current_waypoint_index + 1
        if next_idx < len(self.waypoints):
            return self.waypoints[next_idx]
        return None

    def get_previous_waypoint(self) -> Optional[Waypoint]:
        """Get the previous waypoint (start of current leg)"""
        prev_idx = self.current_waypoint_index - 1
        if prev_idx >= 0:
            return self.waypoints[prev_idx]
        return None

    def advance_waypoint(self) -> bool:
        """Advance to next waypoint. Returns False if at end."""
        if self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
            logger.info(f"Advanced to waypoint {self.current_waypoint_index}: {self.get_current_waypoint()}")
            return True
        return False

    def reset(self):
        """Reset flight plan to beginning"""
        self.current_waypoint_index = 0
        self.phase = FlightPhase.IDLE


@dataclass
class TakeoffConfig:
    """Aircraft-specific takeoff parameters"""
    flap_setting: float = 0.0        # 0-1 (0 = clean takeoff)
    v1_speed_kts: int = 55           # Decision speed
    vr_speed_kts: int = 60           # Rotation speed
    v2_speed_kts: int = 70           # Takeoff safety speed
    initial_climb_pitch: float = 10  # degrees nose up
    initial_climb_vs_fpm: int = 1000 # feet per minute
    target_climb_speed_kts: int = 90 # knots IAS for climb
    flap_retract_alt_agl: int = 400  # feet AGL
    gear_retract_alt_agl: int = 200  # feet AGL
    acceleration_alt_agl: int = 1000 # feet AGL before acceleration

    @classmethod
    def from_aircraft_profile(cls, profile: AircraftVSpeeds) -> "TakeoffConfig":
        """Create TakeoffConfig from an AircraftVSpeeds profile"""
        return cls(
            flap_setting=profile.flaps_takeoff,
            v1_speed_kts=profile.vr - 5,  # V1 typically ~5kts below Vr
            vr_speed_kts=profile.vr,
            v2_speed_kts=profile.v2,
            initial_climb_pitch=profile.rotation_pitch,
            initial_climb_vs_fpm=1000,
            target_climb_speed_kts=profile.vy,
            flap_retract_alt_agl=profile.flaps_retract_agl,
            gear_retract_alt_agl=profile.gear_retract_agl,
            acceleration_alt_agl=1000
        )


@dataclass
class NavigationConfig:
    """Navigation parameters"""
    waypoint_capture_radius_nm: float = 2.0  # Capture waypoint within this radius
    heading_update_threshold_deg: float = 2.0  # Only update if > this many degrees off
    altitude_capture_threshold_ft: int = 200  # Consider altitude captured within this
    descent_rate_fpm: int = 500  # Standard descent rate
    climb_rate_fpm: int = 800    # Standard climb rate
    # Track-following / LNAV parameters
    max_intercept_angle_deg: float = 45.0  # Max angle to intercept track
    xte_gain: float = 30.0  # Degrees of intercept per nm of XTE (capped by max_intercept_angle)
    track_capture_nm: float = 0.1  # Consider on-track when XTE < this value


class NavigationCalculator:
    """Calculates navigation parameters between points"""

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate initial bearing from point 1 to point 2.
        Returns bearing in degrees (0-360).
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon_diff = math.radians(lon2 - lon1)

        x = math.sin(lon_diff) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon_diff)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def calculate_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.
        Returns distance in nautical miles.
        """
        R = 3440.065  # Earth radius in nautical miles

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat/2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    @staticmethod
    def normalize_heading(heading: float) -> float:
        """Normalize heading to 0-360 range"""
        return (heading + 360) % 360

    @staticmethod
    def heading_difference(hdg1: float, hdg2: float) -> float:
        """
        Calculate the shortest angular difference between two headings.
        Returns signed value (-180 to +180).
        """
        diff = hdg2 - hdg1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff


class FlightManager:
    """
    Main flight automation manager.
    Coordinates navigation, takeoff, and flight phases.
    """

    def __init__(self, xplane_commander, get_flight_data_func: Callable, aircraft_type: str = "default"):
        """
        Initialize flight manager.

        Args:
            xplane_commander: XPlaneCommandSender instance for sending commands
            get_flight_data_func: Function that returns current XPlaneFlightData
            aircraft_type: Aircraft type for V-speeds (e.g., "cessna_172", "default")
        """
        self.commander = xplane_commander
        self.get_flight_data = get_flight_data_func
        self.aircraft_type = aircraft_type

        self.flight_plan: Optional[FlightPlan] = None

        # Load aircraft profile and create takeoff config
        self.aircraft_profile = get_aircraft_profile(aircraft_type)
        self.takeoff_config = TakeoffConfig.from_aircraft_profile(self.aircraft_profile)
        self.nav_config = NavigationConfig()

        logger.info(f"Flight manager initialized with aircraft: {self.aircraft_profile.name} (Vr={self.takeoff_config.vr_speed_kts}kts)")

        self.nav_calculator = NavigationCalculator()

        self.active = False
        self.takeoff_state = TakeoffState.IDLE
        self._nav_task: Optional[asyncio.Task] = None
        self._takeoff_task: Optional[asyncio.Task] = None

        # Status callback for UI updates
        self.status_callback: Optional[Callable[[dict], Any]] = None

        # Throttle tracking for smooth advancement
        self._current_throttle = 0.0

        # Manual heading override state (from SVS compass click)
        self.manual_heading_override = False
        self.override_target_heading: Optional[float] = None
        self.heading_capture_tolerance = 5.0  # degrees

    def set_status_callback(self, callback: Callable[[dict], Any]):
        """Set callback function for status updates"""
        self.status_callback = callback

    def set_aircraft_type(self, aircraft_type: str):
        """Change the aircraft type and reload V-speeds"""
        self.aircraft_type = aircraft_type
        self.aircraft_profile = get_aircraft_profile(aircraft_type)
        self.takeoff_config = TakeoffConfig.from_aircraft_profile(self.aircraft_profile)
        logger.info(f"Aircraft type changed to: {self.aircraft_profile.name} (Vr={self.takeoff_config.vr_speed_kts}kts)")

    async def broadcast_status(self, status: dict):
        """Send status update via callback"""
        if self.status_callback:
            try:
                await self.status_callback(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def set_manual_heading_override(self, heading: float, override: bool):
        """
        Set manual heading override from SVS compass click.
        When active, flight manager will maintain this heading instead of updating
        to waypoint bearing. Once heading is captured, override releases and
        normal navigation resumes.
        """
        self.manual_heading_override = override
        self.override_target_heading = heading if override else None

        if override:
            logger.info(f"Manual heading override engaged: {heading}°")
            # Set the heading bug in X-Plane
            if self.commander:
                self.commander.set_heading_bug(heading)
        else:
            logger.info("Manual heading override released, resuming navigation")

    def check_heading_captured(self, current_heading: float) -> bool:
        """
        Check if the manual override heading has been captured.
        Returns True when aircraft is within tolerance of target heading.
        """
        if not self.manual_heading_override or self.override_target_heading is None:
            return False

        # Calculate heading difference (handle 360/0 wrap)
        diff = abs(current_heading - self.override_target_heading)
        if diff > 180:
            diff = 360 - diff

        return diff <= self.heading_capture_tolerance

    def _calculate_cross_track_error(
        self,
        start_lat: float, start_lon: float, track_heading: float,
        current_lat: float, current_lon: float
    ) -> float:
        """
        Calculate cross-track error (perpendicular distance from track line).

        Args:
            start_lat, start_lon: Starting point of the track (runway threshold)
            track_heading: Direction of the track in degrees
            current_lat, current_lon: Current aircraft position

        Returns:
            Cross-track error in nautical miles (positive = right of track, negative = left)
        """
        # Calculate bearing from start to current position
        bearing_to_aircraft = self.nav_calculator.calculate_bearing(
            start_lat, start_lon, current_lat, current_lon
        )

        # Calculate distance from start to current position
        distance_to_aircraft = self.nav_calculator.calculate_distance_nm(
            start_lat, start_lon, current_lat, current_lon
        )

        # Calculate the angular difference between track and bearing to aircraft
        angle_diff = math.radians(bearing_to_aircraft - track_heading)

        # Cross-track error = distance * sin(angle_diff)
        # Positive = right of track, Negative = left of track
        cross_track_nm = distance_to_aircraft * math.sin(angle_diff)

        return cross_track_nm

    def calculate_track_intercept_heading(
        self,
        prev_wp: Waypoint,
        current_wp: Waypoint,
        current_lat: float,
        current_lon: float
    ) -> Tuple[float, float, float]:
        """
        Calculate the heading to intercept and follow the track between waypoints.

        Uses cross-track error to compute an intercept angle that will guide
        the aircraft back to the desired track line.

        Args:
            prev_wp: Previous waypoint (start of current leg)
            current_wp: Current target waypoint (end of current leg)
            current_lat, current_lon: Current aircraft position

        Returns:
            Tuple of (intercept_heading, track_bearing, cross_track_error_nm)
            - intercept_heading: Heading to fly to intercept/follow track (0-360)
            - track_bearing: Direct track bearing from prev to current waypoint
            - cross_track_error_nm: Distance off track (positive = right, negative = left)
        """
        # Calculate the desired track (bearing from prev waypoint to current waypoint)
        track_bearing = self.nav_calculator.calculate_bearing(
            prev_wp.lat, prev_wp.lon,
            current_wp.lat, current_wp.lon
        )

        # Calculate cross-track error from the track line
        xte = self._calculate_cross_track_error(
            prev_wp.lat, prev_wp.lon, track_bearing,
            current_lat, current_lon
        )

        # Calculate intercept angle based on XTE
        # xte > 0 means we're RIGHT of track, need to turn LEFT (subtract from track bearing)
        # xte < 0 means we're LEFT of track, need to turn RIGHT (add to track bearing)
        intercept_angle = -xte * self.nav_config.xte_gain

        # Clamp intercept angle to max
        intercept_angle = max(-self.nav_config.max_intercept_angle_deg,
                              min(self.nav_config.max_intercept_angle_deg, intercept_angle))

        # Calculate final intercept heading
        intercept_heading = self.nav_calculator.normalize_heading(track_bearing + intercept_angle)

        return intercept_heading, track_bearing, xte

    def set_flight_plan(
        self,
        waypoints: List[dict],
        cruise_altitude: int,
        departure: str,
        destination: str,
        departure_runway: Optional[str] = None,
        departure_heading: float = 0.0
    ):
        """
        Load a new flight plan.

        Args:
            waypoints: List of waypoint dicts with name, lat, lon, altitude
            cruise_altitude: Cruise altitude in feet
            departure: Departure airport ICAO
            destination: Destination airport ICAO
            departure_runway: Departure runway identifier (e.g., "25L")
            departure_heading: Runway heading in degrees
        """
        wp_list = []
        for wp in waypoints:
            wp_list.append(Waypoint(
                name=wp.get('name', 'WPT'),
                lat=wp.get('lat', 0),
                lon=wp.get('lon', 0),
                altitude_ft=wp.get('altitude', cruise_altitude),
                waypoint_type=wp.get('type', 'enroute')
            ))

        self.flight_plan = FlightPlan(
            waypoints=wp_list,
            departure_icao=departure,
            destination_icao=destination,
            cruise_altitude_ft=cruise_altitude,
            departure_runway=departure_runway,
            departure_heading=departure_heading
        )

        logger.info(f"Flight plan loaded: {departure} -> {destination}, {len(wp_list)} waypoints, FL{cruise_altitude//100}")

    async def position_to_departure(self, lat: float, lon: float, elevation_m: float, heading: float):
        """
        Reposition aircraft to departure airport.

        Args:
            lat: Airport/runway latitude
            lon: Airport/runway longitude
            elevation_m: Airport elevation in meters
            heading: Runway heading in degrees
        """
        if self.commander:
            logger.info(f"Positioning to departure: {lat}, {lon}, {elevation_m}m, {heading}deg")
            self.commander.set_position(lat, lon, elevation_m, heading)

            # Give X-Plane time to settle
            await asyncio.sleep(2)

            # Set initial state
            self.commander.set_parking_brake(True)
            self.commander.set_throttle(0.0)
            self.commander.set_flaps(self.takeoff_config.flap_setting)
            self.commander.gear_down()

            if self.flight_plan:
                self.flight_plan.phase = FlightPhase.PREFLIGHT
                self.flight_plan.departure_heading = heading

            await self.broadcast_status({
                "type": "position_set",
                "phase": "PREFLIGHT",
                "lat": lat,
                "lon": lon,
                "heading": heading
            })

    def reset_takeoff(self):
        """Reset a stuck takeoff - allows starting fresh"""
        print("[TAKEOFF] Resetting takeoff state...")
        self.active = False
        self.takeoff_state = TakeoffState.IDLE
        self._current_throttle = 0.0
        if self._takeoff_task and not self._takeoff_task.done():
            self._takeoff_task.cancel()
            self._takeoff_task = None
        print("[TAKEOFF] Reset complete")

    async def start_auto_takeoff(self, runway_heading: float):
        """
        Start the auto-takeoff sequence.

        Args:
            runway_heading: Runway heading in degrees
        """
        print(f"[TAKEOFF] start_auto_takeoff called with heading {runway_heading}")

        if self._takeoff_task and not self._takeoff_task.done():
            print("[TAKEOFF] Takeoff already in progress - resetting first...")
            self.reset_takeoff()

        print("[TAKEOFF] Starting new takeoff sequence...")
        self.active = True
        self.takeoff_state = TakeoffState.PREFLIGHT_CHECK
        self._current_throttle = 0.0

        self._takeoff_task = asyncio.create_task(
            self._takeoff_sequence(runway_heading)
        )

    async def _takeoff_sequence(self, runway_heading: float):
        """Execute the auto-takeoff state machine - throttle, steering, and climb only"""
        try:
            print(f"=== TAKEOFF SEQUENCE STARTED ===")

            # Get the ACTUAL aircraft heading and position - this defines the runway centerline
            flight_data = self.get_flight_data()
            if flight_data and flight_data.heading_mag > 0:
                actual_heading = flight_data.heading_mag
                print(f"Using ACTUAL aircraft heading as runway heading: {actual_heading:.0f}°")
                runway_heading = actual_heading
            else:
                print(f"Using provided runway heading: {runway_heading}°")

            # Capture initial position as runway centerline reference
            initial_lat = flight_data.lat if flight_data else 0
            initial_lon = flight_data.lon if flight_data else 0
            print(f"Runway centerline start: {initial_lat:.6f}, {initial_lon:.6f}")
            print(f"Runway heading: {runway_heading:.0f}°")
            print(f"Vr={self.takeoff_config.vr_speed_kts}kts")
            logger.info(f"Starting auto-takeoff, runway heading {runway_heading}")

            # ========== STEP 1: RELEASE PARKING BRAKE ==========
            print("Step 1: Releasing parking brake (3x to ensure)")
            for i in range(3):
                self.commander.set_parking_brake(False)
                await asyncio.sleep(0.2)

            # ========== STEP 2: THROTTLE UP ==========
            self.takeoff_state = TakeoffState.THROTTLE_UP
            print("Step 2: Advancing throttle to full")
            await self.broadcast_status({"type": "takeoff_status", "state": "THROTTLE_UP"})

            # Smoothly advance throttle to full, releasing brakes each iteration
            while self._current_throttle < 1.0:
                self.commander.set_parking_brake(False)  # Keep releasing brakes
                self._current_throttle = self.commander.set_throttle_smoothly(
                    1.0, self._current_throttle, step=0.15
                )
                await asyncio.sleep(0.1)
            print("  Throttle full")

            # One more brake release for good measure
            self.commander.set_parking_brake(False)

            # ========== STEP 3: TAKEOFF ROLL - CENTERLINE TRACKING ==========
            self.takeoff_state = TakeoffState.TAKEOFF_ROLL
            print("Step 3: Takeoff roll - PID centerline tracking")
            await self.broadcast_status({"type": "takeoff_status", "state": "TAKEOFF_ROLL"})

            if self.flight_plan:
                self.flight_plan.phase = FlightPhase.TAKEOFF_ROLL

            last_speed_print = -1  # Start at -1 so first print happens immediately
            elevator = 0.0
            iteration = 0

            # PID controller state for heading
            heading_integral = 0.0
            last_heading_error = 0.0
            last_time = asyncio.get_event_loop().time()

            # P-factor compensation: Single-engine props with clockwise rotation (from pilot view)
            # create strong left-turning tendency. Apply RIGHT rudder bias to counteract.
            # SR22 has IO-550 with clockwise prop - needs significant right rudder on takeoff
            P_FACTOR_BIAS = 0.25  # Base right rudder to counteract P-factor/torque

            while self.active:
                flight_data = self.get_flight_data()
                if flight_data:
                    ias = flight_data.airspeed_ind
                    iteration += 1
                    current_time = asyncio.get_event_loop().time()
                    dt = current_time - last_time
                    last_time = current_time

                    # Print first iteration to confirm data
                    if iteration == 1:
                        print(f"  [START] IAS:{ias:.0f} PITCH:{flight_data.pitch:.1f}° AGL:{flight_data.alt_agl:.0f}")
                        print(f"  [PID] Using P-factor bias: +{P_FACTOR_BIAS:.2f} right rudder")

                    # Release brakes once at the start of roll (iteration 1), not repeatedly
                    if iteration == 1:
                        self.commander.set_parking_brake(False)
                    current_heading = flight_data.heading_mag
                    current_roll = flight_data.roll
                    current_lat = flight_data.lat
                    current_lon = flight_data.lon

                    # === CROSS-TRACK ERROR (lateral deviation from runway centerline) ===
                    cross_track_error = self._calculate_cross_track_error(
                        initial_lat, initial_lon, runway_heading,
                        current_lat, current_lon
                    )

                    # === HEADING ERROR with wraparound handling ===
                    heading_error = runway_heading - current_heading
                    if heading_error > 180:
                        heading_error -= 360
                    elif heading_error < -180:
                        heading_error += 360

                    # === PID STEERING CONTROL ===
                    # heading_error > 0 means we're LEFT of target, need to turn RIGHT (positive rudder)
                    # heading_error < 0 means we're RIGHT of target, need to turn LEFT (negative rudder)

                    # PID gains - AGGRESSIVE for ground roll
                    Kp = 0.08   # Proportional: strong response to error
                    Ki = 0.02   # Integral: fight persistent drift
                    Kd = 0.01   # Derivative: dampen oscillations

                    # Update integral (with anti-windup)
                    heading_integral += heading_error * dt
                    heading_integral = max(-50.0, min(50.0, heading_integral))  # Anti-windup

                    # Calculate derivative
                    heading_derivative = (heading_error - last_heading_error) / max(dt, 0.001)
                    last_heading_error = heading_error

                    # PID output
                    pid_output = (Kp * heading_error) + (Ki * heading_integral) + (Kd * heading_derivative)

                    # Add P-factor bias (always push right to counteract left-turning tendency)
                    # Scale bias with speed - more effect as prop wash increases
                    speed_factor = min(1.0, ias / 60.0)  # Full bias by 60 kts
                    p_factor_compensation = P_FACTOR_BIAS * speed_factor

                    # Total rudder command
                    rudder = pid_output + p_factor_compensation
                    rudder = max(-1.0, min(1.0, rudder))  # FULL rudder authority

                    # Nosewheel steering - stronger at low speed, same direction as rudder
                    if ias < 40:
                        nosewheel = rudder * 2.0  # Strong nosewheel at low speed
                    elif ias < 60:
                        nosewheel = rudder * 1.0  # Moderate
                    else:
                        nosewheel = rudder * 0.3  # Light as rudder takes over
                    nosewheel = max(-1.0, min(1.0, nosewheel))

                    self.commander.set_rudder(rudder)
                    self.commander.set_nosewheel_steering(nosewheel)

                    # === ROLL CONTROL (keep wings level) ===
                    aileron = -current_roll * 0.03  # Wing leveling
                    aileron = max(-0.3, min(0.3, aileron))
                    self.commander.set_aileron(aileron)

                    # === THROTTLE - MAINTAIN FULL POWER ===
                    self.commander.set_throttle(1.0)  # Keep throttle at full throughout takeoff roll

                    # === GRADUAL PITCH UP FOR ROTATION ===
                    if ias < 55:
                        elevator = 0.0  # No pitch input below 55kts
                    elif ias < self.takeoff_config.vr_speed_kts:
                        progress = (ias - 55) / max(1, self.takeoff_config.vr_speed_kts - 55)
                        elevator = progress * 0.2  # Up to 20% elevator at Vr
                    else:
                        elevator = 0.25  # Gentle pitch up for rotation
                    self.commander.set_elevator(elevator)

                    # Print speed updates with PID debug info
                    if iteration <= 5 or ias >= last_speed_print + 5:
                        print(f"  IAS:{ias:.0f} HDG:{current_heading:.0f}°->TGT:{runway_heading:.0f}° ERR:{heading_error:.1f}° | PID:{pid_output:.2f} BIAS:{p_factor_compensation:.2f} RUD:{rudder:.2f} NW:{nosewheel:.2f}")
                        if ias >= last_speed_print + 5:
                            last_speed_print = int(ias)

                    # Check for liftoff
                    if flight_data.alt_agl > 20:
                        print(f"  LIFTOFF! AGL={flight_data.alt_agl:.0f} ft, IAS={ias:.0f} kts")
                        break

                    # Safety abort: if heading error exceeds 45°, we've lost control
                    if abs(heading_error) > 45 and ias > 30:
                        print(f"  [ABORT] Heading deviation too large: {heading_error:.1f}° - cutting throttle")
                        self.commander.set_throttle(0.0)
                        self.commander.set_rudder(0.0)
                        self.commander.set_nosewheel_steering(0.0)
                        await asyncio.sleep(0.5)
                        break

                await asyncio.sleep(0.05)  # 20Hz control loop

            if not self.active:
                print("Takeoff aborted")
                return

            # ========== STEP 4: CLIMB OUT ==========
            self.takeoff_state = TakeoffState.INITIAL_CLIMB
            print("Step 4: Climb out - manual flight control")
            await self.broadcast_status({"type": "takeoff_status", "state": "INITIAL_CLIMB"})

            if self.flight_plan:
                self.flight_plan.phase = FlightPhase.INITIAL_CLIMB

            # CUSTOM AUTOPILOT: Keep manual control and fly the plane ourselves
            # Use conservative climb rate to avoid stall
            target_vs = 500  # Conservative 500 fpm climb
            target_heading = runway_heading
            target_altitude = self.flight_plan.cruise_altitude_ft if self.flight_plan else 8000

            # Stall protection parameters - use actual stall speed, not V2
            MIN_SAFE_SPEED = max(60, self.takeoff_config.v2_speed_kts - 15)  # ~65kts for SR22
            PITCH_LIMIT = 12.0  # Max pitch angle in degrees

            print(f"  Manual flight control active")
            print(f"  Target: HDG {target_heading:.0f}°, VS {target_vs} fpm, ALT {target_altitude} ft")
            print(f"  Stall protection: Min speed {MIN_SAFE_SPEED}kts (actual stall), Max pitch {PITCH_LIMIT}°")

            # Fly until we reach cruise altitude
            climb_start = asyncio.get_event_loop().time()
            max_climb_time = 180  # 3 minutes max
            last_log_time = 0

            while self.active:
                flight_data = self.get_flight_data()
                if not flight_data:
                    await asyncio.sleep(0.1)
                    continue

                current_alt = flight_data.alt_msl
                current_vs = flight_data.vertical_speed
                current_hdg = flight_data.heading_mag
                current_roll = flight_data.roll
                current_pitch = flight_data.pitch
                current_ias = flight_data.airspeed_ind

                # Check if we've reached cruise altitude
                if current_alt >= target_altitude - 100:
                    print(f"  Reached cruise altitude: {current_alt:.0f} ft")
                    break

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - climb_start
                if elapsed > max_climb_time:
                    print(f"  Climb timeout after {elapsed:.0f}s at {current_alt:.0f} ft")
                    break

                # === STALL PROTECTION (only when airborne and moving) ===
                stall_correction = 0
                if flight_data.alt_agl > 50 and current_ias > 30:
                    stall_margin = current_ias - MIN_SAFE_SPEED
                    if stall_margin < 5:
                        # Getting close to stall - reduce pitch aggressively
                        stall_correction = (5 - stall_margin) * -0.03
                        print(f"  !! STALL PROTECTION !! IAS:{current_ias:.0f} - pushing nose down")

                # === PITCH CONTROL: Maintain climb with VS feedback ===
                vs_error = target_vs - current_vs
                pitch_correction = vs_error * 0.0001  # Gentle VS correction
                pitch_correction = max(-0.1, min(0.1, pitch_correction))

                # Base elevator - matches rotation for smooth transition
                elevator = 0.15 + pitch_correction + stall_correction

                # Hard pitch limit - don't exceed max pitch angle
                if current_pitch > PITCH_LIMIT:
                    elevator -= (current_pitch - PITCH_LIMIT) * 0.02
                elif current_pitch < -5:
                    elevator += 0.1  # Prevent dive

                # Clamp elevator to safe range
                elevator = max(-0.1, min(0.3, elevator))
                self.commander.set_elevator(elevator)

                # === ROLL/BANK CONTROL: Keep wings level + heading correction ===
                roll_correction = -current_roll * 0.03

                hdg_error = target_heading - current_hdg
                if hdg_error > 180:
                    hdg_error -= 360
                elif hdg_error < -180:
                    hdg_error += 360
                hdg_correction = hdg_error * 0.01

                aileron = roll_correction + hdg_correction
                aileron -= 0.01  # Slight left bias for P-factor
                aileron = max(-0.3, min(0.3, aileron))
                self.commander.set_aileron(aileron)

                # Rudder for coordination
                rudder = hdg_error * 0.005
                rudder = max(-0.15, min(0.15, rudder))
                self.commander.set_rudder(rudder)

                # Log every 3 seconds
                if elapsed - last_log_time >= 3:
                    print(f"  IAS:{current_ias:.0f} PITCH:{current_pitch:.1f}° ALT:{current_alt:.0f} VS:{current_vs:.0f} HDG:{current_hdg:.0f} elev:{elevator:.2f}")
                    last_log_time = elapsed

                await asyncio.sleep(0.05)  # 20Hz control loop

            print("  Climb complete, entering envelope protection...")

            # ========== STEP 7: GEAR & FLAPS (if retractable) ==========
            # Wait for gear retract altitude (skip for fixed gear like SR22)
            if self.takeoff_config.gear_retract_alt_agl > 0:
                print(f"  Waiting for gear retract altitude ({self.takeoff_config.gear_retract_alt_agl} ft AGL)...")
                while self.active:
                    flight_data = self.get_flight_data()
                    if flight_data and flight_data.alt_agl > self.takeoff_config.gear_retract_alt_agl:
                        print(f"  Gear up at {flight_data.alt_agl:.0f} ft AGL")
                        self.commander.gear_up()
                        break
                    await asyncio.sleep(0.2)

            # Wait for flap retract altitude
            print(f"  Waiting for flap retract altitude ({self.takeoff_config.flap_retract_alt_agl} ft AGL)...")
            while self.active:
                flight_data = self.get_flight_data()
                if flight_data and flight_data.alt_agl > self.takeoff_config.flap_retract_alt_agl:
                    print(f"  Flaps up at {flight_data.alt_agl:.0f} ft AGL")
                    self.commander.set_flaps(0.0)
                    break
                await asyncio.sleep(0.2)

            if not self.active:
                return

            # ========== STEP 5: ENVELOPE PROTECTION MODE ==========
            self.takeoff_state = TakeoffState.CLIMB_OUT
            print("Step 5: Engaging ENVELOPE PROTECTION (continuous wing leveling)")
            await self.broadcast_status({"type": "takeoff_status", "state": "ENVELOPE_PROTECTION"})

            if self.flight_plan:
                self.flight_plan.phase = FlightPhase.CRUISE

            # Set cruise altitude bug for reference
            cruise_alt = self.flight_plan.cruise_altitude_ft if self.flight_plan else 8000
            self.commander.set_altitude_bug(cruise_alt)

            # Capture current heading as target
            flight_data = self.get_flight_data()
            target_heading = flight_data.heading_mag if flight_data else runway_heading
            target_altitude = cruise_alt
            self.commander.set_heading_bug(target_heading)

            print(f"  ENVELOPE PROTECTION ACTIVE")
            print(f"  Target: HDG {target_heading:.0f}°, ALT {target_altitude} ft")
            print(f"  Bank limit: ±30°, Pitch limit: ±20°")
            print(f"  Pilot can override - system will auto-level when released")

            # Start continuous envelope protection loop (runs until stopped)
            await self._envelope_protection_loop(target_heading, target_altitude)

            # Envelope protection ended (user stopped or released control)
            self.takeoff_state = TakeoffState.COMPLETE
            await self.broadcast_status({"type": "takeoff_status", "state": "COMPLETE"})
            print("=== ENVELOPE PROTECTION ENDED ===")

        except Exception as e:
            logger.error(f"Takeoff error: {e}")
            self.takeoff_state = TakeoffState.ABORTED
            await self.broadcast_status({"type": "takeoff_status", "state": "ABORTED", "error": str(e)})

    async def _envelope_protection_loop(self, initial_heading: float, initial_altitude: float):
        """
        Continuous envelope protection with integrated waypoint navigation.
        This simulates a fly-by-wire system that:
        - Keeps the plane level and within safe parameters
        - Navigates to waypoints in the flight plan
        - Maintains altitude and heading

        Features:
        - Bank angle limiting (±30°)
        - Pitch angle limiting (±20°)
        - Automatic wing leveling
        - Waypoint navigation (updates target heading based on flight plan)
        - Altitude management per waypoint
        - Runs perpetually until stopped
        """
        BANK_LIMIT = 25.0  # Maximum bank angle (degrees)
        PITCH_LIMIT = 12.0  # Maximum pitch angle (degrees) - conservative to avoid stall
        WAYPOINT_CAPTURE_NM = 2.0  # Capture waypoint within 2nm
        # Stall protection: use actual stall speed (V2 - 15), not V2
        # V2 is takeoff safety speed, actual stall is much lower
        MIN_SAFE_SPEED = max(60, self.takeoff_config.v2_speed_kts - 15)  # ~65kts for SR22

        # Initialize targets
        target_heading = initial_heading
        target_altitude = initial_altitude

        last_log_time = 0
        last_nav_update = 0
        loop_start = asyncio.get_event_loop().time()

        # Navigation state
        destination_reached = False

        print("  === ENVELOPE PROTECTION + NAVIGATION STARTED ===")
        print(f"  Stall protection: Min {MIN_SAFE_SPEED}kts, Bank limit ±{BANK_LIMIT}°, Pitch limit ±{PITCH_LIMIT}°")
        if self.flight_plan and len(self.flight_plan.waypoints) > 0:
            print(f"  Flight plan loaded: {len(self.flight_plan.waypoints)} waypoints")
            print(f"  Route: {self.flight_plan.departure_icao} -> {self.flight_plan.destination_icao}")
        else:
            print(f"  No flight plan - holding HDG {target_heading:.0f}° ALT {target_altitude}ft")

        current_waypoint_name = "DIRECT"
        first_nav_update = True

        while self.active:
            try:
                flight_data = self.get_flight_data()
                if not flight_data:
                    await asyncio.sleep(0.05)
                    continue

                current_roll = flight_data.roll
                current_pitch = flight_data.pitch
                current_hdg = flight_data.heading_mag
                current_alt = flight_data.alt_msl
                current_vs = flight_data.vertical_speed
                current_lat = flight_data.lat
                current_lon = flight_data.lon

                elapsed = asyncio.get_event_loop().time() - loop_start

                # === NAVIGATION UPDATE (1Hz) ===
                if elapsed - last_nav_update >= 1.0 and self.flight_plan and not destination_reached:
                    last_nav_update = elapsed

                    current_wp = self.flight_plan.get_current_waypoint()
                    if current_wp:
                        # Get previous waypoint for track-following
                        prev_wp = self.flight_plan.get_previous_waypoint()

                        # Calculate distance to current waypoint
                        distance = self.nav_calculator.calculate_distance_nm(
                            current_lat, current_lon,
                            current_wp.lat, current_wp.lon
                        )

                        # Calculate target heading using track-following (LNAV)
                        if prev_wp:
                            # We have a previous waypoint - use track-following with XTE correction
                            desired_heading, track_bearing, xte = self.calculate_track_intercept_heading(
                                prev_wp, current_wp, current_lat, current_lon
                            )
                            bearing = track_bearing  # For logging
                        else:
                            # First leg - no previous waypoint, use direct-to
                            bearing = self.nav_calculator.calculate_bearing(
                                current_lat, current_lon,
                                current_wp.lat, current_wp.lon
                            )
                            desired_heading = bearing
                            xte = 0.0

                        # Rate-limit heading changes (max 5° per second for smooth turns)
                        MAX_HDG_CHANGE_PER_SEC = 5.0
                        hdg_diff = desired_heading - target_heading
                        if hdg_diff > 180:
                            hdg_diff -= 360
                        elif hdg_diff < -180:
                            hdg_diff += 360

                        if abs(hdg_diff) > MAX_HDG_CHANGE_PER_SEC:
                            # Limit the heading change
                            if hdg_diff > 0:
                                target_heading = (target_heading + MAX_HDG_CHANGE_PER_SEC) % 360
                            else:
                                target_heading = (target_heading - MAX_HDG_CHANGE_PER_SEC) % 360
                            print(f"  [NAV] Heading rate-limited: {hdg_diff:.0f}° change, now targeting {target_heading:.0f}° (track {bearing:.0f}°, XTE {xte:.2f}nm)")
                        else:
                            target_heading = desired_heading

                        current_waypoint_name = current_wp.name

                        # Log first navigation update
                        if first_nav_update:
                            print(f"  [NAV] First waypoint: {current_wp.name} at {distance:.1f}nm, track {bearing:.0f}°, XTE {xte:.2f}nm (current HDG {current_hdg:.0f}°)")
                            first_nav_update = False
                        elif abs(xte) > self.nav_config.track_capture_nm:
                            print(f"  [LNAV] XTE: {xte:.2f}nm, Track: {bearing:.0f}°, Intercept HDG: {target_heading:.0f}°")

                        # Update target altitude from waypoint (or field elevation for destination)
                        if current_wp.altitude_ft > 0:
                            target_altitude = current_wp.altitude_ft

                        # Check for waypoint capture
                        if distance < WAYPOINT_CAPTURE_NM:
                            print(f"  [NAV] CAPTURED: {current_wp.name} (dist: {distance:.1f}nm)")
                            await self.broadcast_status({
                                "type": "waypoint_captured",
                                "waypoint": current_wp.name,
                                "index": self.flight_plan.current_waypoint_index
                            })

                            if not self.flight_plan.advance_waypoint():
                                print(f"  [NAV] *** DESTINATION REACHED - INITIATING AUTOLAND ***")
                                destination_reached = True
                                current_waypoint_name = "LAND"

                                # Get destination info for landing
                                dest_wp = self.flight_plan.waypoints[-1] if self.flight_plan.waypoints else None
                                # Use current heading as runway heading (simplified - real impl would look up runway)
                                runway_hdg = current_hdg
                                field_elev = dest_wp.altitude_ft if dest_wp and dest_wp.altitude_ft > 0 else 0

                                # Start autoland sequence (this will take over control)
                                await self.execute_autoland(runway_hdg, field_elev)
                                break  # Exit envelope protection loop after landing
                            else:
                                next_wp = self.flight_plan.get_current_waypoint()
                                if next_wp:
                                    print(f"  [NAV] Next: {next_wp.name} ({distance:.1f}nm away)")

                # === ROLL/BANK CONTROL ===
                # Calculate desired bank based on heading error
                hdg_error = target_heading - current_hdg
                if hdg_error > 180:
                    hdg_error -= 360
                elif hdg_error < -180:
                    hdg_error += 360

                # Determine desired bank angle based on heading error
                # Use gentler bank angles for smoother turns
                if abs(hdg_error) > 3:  # Reduced deadband from 5 to 3
                    # Need to turn - calculate desired bank (max 15 degrees for gentle turns)
                    desired_bank = hdg_error * 0.25  # Slightly gentler response
                    desired_bank = max(-15, min(15, desired_bank))
                else:
                    # On heading - wings level
                    desired_bank = 0

                # Bank limiting - if exceeding limit, actively correct toward wings level
                if abs(current_roll) > BANK_LIMIT:
                    # Exceeding bank limit - force wings level
                    desired_bank = 0  # Command wings level when exceeding limits
                    print(f"  !! BANK LIMIT EXCEEDED: {current_roll:.1f}° - commanding wings level")

                # Calculate aileron to achieve desired bank
                bank_error = desired_bank - current_roll
                aileron = bank_error * 0.08  # Strong gain to overcome existing bank

                aileron = max(-0.6, min(0.6, aileron))  # Full aileron authority when needed
                self.commander.set_aileron(aileron)

                # Log aileron commands in first 30 seconds for debugging
                if elapsed < 30 and int(elapsed * 2) % 2 == 0:  # Every 0.5 seconds
                    print(f"    [ROLL] hdg_err:{hdg_error:.0f} desired_bank:{desired_bank:.1f} roll:{current_roll:.1f} aileron:{aileron:.3f}")

                # === STALL PROTECTION (only when airborne and moving) ===
                current_ias = flight_data.airspeed_ind
                stall_correction = 0
                if flight_data.alt_agl > 50 and current_ias > 30:
                    stall_margin = current_ias - MIN_SAFE_SPEED
                    if stall_margin < 5:
                        # Getting close to stall - push nose down!
                        stall_correction = (5 - stall_margin) * -0.03
                        print(f"  !! STALL PROTECTION !! IAS:{current_ias:.0f}kts - pushing nose down")

                # === PITCH/ALTITUDE PROTECTION ===
                alt_error = target_altitude - current_alt

                # Desired VS based on altitude error
                if abs(alt_error) < 100:
                    target_vs_cmd = 0  # Level flight
                elif alt_error > 0:
                    target_vs_cmd = min(800, alt_error * 2)  # Max 800 fpm climb
                else:
                    target_vs_cmd = max(-1200, alt_error * 2)  # Max 1200 fpm descent (faster)

                vs_error = target_vs_cmd - current_vs
                pitch_correction = vs_error * 0.0001  # Lower gain

                # Base elevator for level flight + stall protection
                elevator = 0.03 + pitch_correction + stall_correction

                # Hard pitch limiting
                if current_pitch > PITCH_LIMIT:
                    elevator -= (current_pitch - PITCH_LIMIT) * 0.05
                elif current_pitch < -PITCH_LIMIT:
                    elevator += (-PITCH_LIMIT - current_pitch) * 0.05

                # More conservative limits
                elevator = max(-0.2, min(0.2, elevator))
                self.commander.set_elevator(elevator)

                # === RUDDER (coordinated flight) ===
                rudder = hdg_error * 0.008  # Help with turns
                # Counter any slip/skid
                if hasattr(flight_data, 'beta') and flight_data.beta != 0:
                    rudder += -flight_data.beta * 0.02
                rudder = max(-0.3, min(0.3, rudder))
                self.commander.set_rudder(rudder)

                # Log every 5 seconds
                if elapsed - last_log_time >= 5:
                    if destination_reached:
                        status = "HOLD"
                    elif current_ias < MIN_SAFE_SPEED + 5:
                        status = "SLOW!"
                    elif abs(current_roll) > BANK_LIMIT:
                        status = "BANK LIM"
                    elif abs(current_pitch) > PITCH_LIMIT:
                        status = "PITCH LIM"
                    elif self.flight_plan:
                        status = "NAV"
                    else:
                        status = "FLY"
                    print(f"  [{status}] IAS:{current_ias:.0f} ROLL:{current_roll:.1f}° HDG:{current_hdg:.0f}°->{target_heading:.0f}° (err:{hdg_error:.0f}) WPT:{current_waypoint_name} ALT:{current_alt:.0f}")
                    last_log_time = elapsed

                # Broadcast status periodically
                if int(elapsed) % 2 == 0:
                    await self.broadcast_status({
                        "type": "envelope_protection_status",
                        "active": True,
                        "roll": round(current_roll, 1),
                        "pitch": round(current_pitch, 1),
                        "heading": round(current_hdg),
                        "target_heading": round(target_heading),
                        "altitude": round(current_alt),
                        "target_altitude": round(target_altitude),
                        "waypoint": current_waypoint_name,
                        "bank_limit": BANK_LIMIT,
                        "pitch_limit": PITCH_LIMIT
                    })

                await asyncio.sleep(0.05)  # 20Hz control loop

            except Exception as e:
                logger.error(f"Envelope protection error: {e}")
                await asyncio.sleep(0.1)

        print("  === ENVELOPE PROTECTION + NAVIGATION ENDED ===")

    async def start_navigation(self):
        """Start the navigation control loop"""
        if self._nav_task and not self._nav_task.done():
            logger.warning("Navigation already running")
            return

        self.active = True
        self._nav_task = asyncio.create_task(self._navigation_loop())

    async def _navigation_loop(self):
        """Main navigation loop - runs at ~1Hz"""
        logger.info("Navigation loop started")

        while self.active and self.flight_plan:
            try:
                flight_data = self.get_flight_data()
                if not flight_data or not flight_data.connected:
                    await asyncio.sleep(1)
                    continue

                current_wp = self.flight_plan.get_current_waypoint()
                if current_wp is None:
                    # Flight complete
                    self.flight_plan.phase = FlightPhase.COMPLETE
                    await self.broadcast_status({
                        "type": "navigation_status",
                        "phase": "COMPLETE",
                        "message": "Flight plan complete"
                    })
                    logger.info("Flight plan complete")
                    break

                # Current position
                pos_lat = flight_data.lat
                pos_lon = flight_data.lon
                pos_alt = flight_data.alt_msl
                pos_hdg = flight_data.heading_mag

                # Get previous waypoint for track-following
                prev_wp = self.flight_plan.get_previous_waypoint()

                # Calculate distance to current waypoint
                distance = self.nav_calculator.calculate_distance_nm(
                    pos_lat, pos_lon,
                    current_wp.lat, current_wp.lon
                )

                # Calculate target heading using track-following (LNAV)
                if prev_wp:
                    # We have a previous waypoint - use track-following with XTE correction
                    target_heading, track_bearing, xte = self.calculate_track_intercept_heading(
                        prev_wp, current_wp, pos_lat, pos_lon
                    )
                    bearing = track_bearing  # For status display
                else:
                    # First leg - no previous waypoint, use direct-to
                    # Could use departure position if stored, but for now use direct-to
                    bearing = self.nav_calculator.calculate_bearing(
                        pos_lat, pos_lon,
                        current_wp.lat, current_wp.lon
                    )
                    target_heading = bearing
                    xte = 0.0

                # Check for waypoint capture
                if distance < self.nav_config.waypoint_capture_radius_nm:
                    logger.info(f"Waypoint captured: {current_wp.name}")
                    await self.broadcast_status({
                        "type": "waypoint_captured",
                        "waypoint": current_wp.name,
                        "index": self.flight_plan.current_waypoint_index
                    })

                    if not self.flight_plan.advance_waypoint():
                        # No more waypoints - destination reached!
                        logger.info("DESTINATION REACHED - holding current heading")
                        self.flight_plan.phase = FlightPhase.COMPLETE
                        await self.broadcast_status({
                            "type": "navigation_status",
                            "phase": "COMPLETE",
                            "message": "Destination reached - holding heading"
                        })
                        # Hold current heading - don't update heading bug anymore
                        break

                    # Recalculate for new waypoint with track-following
                    current_wp = self.flight_plan.get_current_waypoint()
                    prev_wp = self.flight_plan.get_previous_waypoint()
                    if current_wp:
                        distance = self.nav_calculator.calculate_distance_nm(
                            pos_lat, pos_lon,
                            current_wp.lat, current_wp.lon
                        )
                        if prev_wp:
                            target_heading, bearing, xte = self.calculate_track_intercept_heading(
                                prev_wp, current_wp, pos_lat, pos_lon
                            )
                        else:
                            bearing = self.nav_calculator.calculate_bearing(
                                pos_lat, pos_lon,
                                current_wp.lat, current_wp.lon
                            )
                            target_heading = bearing
                            xte = 0.0
                        logger.info(f"Next waypoint: {current_wp.name} at {distance:.1f}nm, track {bearing:.0f}°, XTE {xte:.2f}nm")

                # Check for manual heading override from SVS compass click
                if self.manual_heading_override:
                    # Check if the override heading has been captured
                    if self.check_heading_captured(pos_hdg):
                        # Heading captured - release override and resume navigation
                        self.manual_heading_override = False
                        self.override_target_heading = None
                        logger.info(f"Manual heading captured, resuming route navigation")
                        await self.broadcast_status({
                            "type": "heading_captured",
                            "message": "Manual heading captured, resuming route"
                        })
                        # Now update to track-following heading
                        self.commander.set_heading_bug(target_heading)
                    # else: keep flying the override heading, skip waypoint heading update
                else:
                    # Normal navigation - update heading if needed (using track-following heading)
                    heading_diff = abs(self.nav_calculator.heading_difference(pos_hdg, target_heading))
                    if heading_diff > self.nav_config.heading_update_threshold_deg:
                        self.commander.set_heading_bug(target_heading)
                        if abs(xte) > self.nav_config.track_capture_nm:
                            print(f"  [LNAV] XTE: {xte:.2f}nm, Track: {bearing:.0f}°, Intercept HDG: {target_heading:.0f}°")

                # Keep autopilot engaged (re-engage every loop iteration to be safe)
                self.commander.engage_autopilot()
                self.commander.engage_heading_mode()

                # Manage altitude based on flight phase and waypoint
                await self._manage_altitude(pos_alt, current_wp, distance)

                # Check vertical speed and apply pitch correction if descending when shouldn't be
                vs = flight_data.vertical_speed if hasattr(flight_data, 'vertical_speed') else 0
                target_alt = current_wp.altitude_ft if current_wp else (self.flight_plan.cruise_altitude_ft if self.flight_plan else 8000)
                alt_error = target_alt - pos_alt

                # If losing altitude significantly when we should be level or climbing
                if alt_error > 100 and vs < -200:
                    # We're below target and descending - apply pitch up correction
                    print(f"  ALT CORRECTION: {pos_alt:.0f} -> {target_alt}, VS={vs:.0f}, applying climb")
                    self.commander.set_vs_bug(500)  # Climb at 500 fpm
                    self.commander.engage_vs_mode()

                # Broadcast status
                await self.broadcast_status({
                    "type": "navigation_status",
                    "phase": self.flight_plan.phase.value,
                    "current_waypoint": current_wp.name,
                    "waypoint_index": self.flight_plan.current_waypoint_index,
                    "total_waypoints": len(self.flight_plan.waypoints),
                    "distance_nm": round(distance, 1),
                    "bearing": round(bearing),
                    "target_heading": round(target_heading),
                    "xte_nm": round(xte, 2),
                    "current_heading": round(pos_hdg),
                    "altitude_ft": round(pos_alt),
                    "target_altitude": current_wp.altitude_ft
                })

            except Exception as e:
                logger.error(f"Navigation loop error: {e}")

            await asyncio.sleep(1)  # 1Hz update rate

        logger.info("Navigation loop ended")

    async def _manage_altitude(self, current_alt: float, waypoint: Waypoint, distance_nm: float):
        """Manage altitude changes based on waypoint and distance"""
        target_alt = waypoint.altitude_ft
        alt_diff = target_alt - current_alt

        # Determine if we need to climb or descend
        if abs(alt_diff) < self.nav_config.altitude_capture_threshold_ft:
            # At target altitude - ensure altitude hold is engaged
            self.commander.set_altitude_bug(target_alt)
            self.commander.engage_altitude_hold()
            if self.flight_plan.phase in [FlightPhase.CLIMB, FlightPhase.DESCENT]:
                self.flight_plan.phase = FlightPhase.CRUISE
                print(f"  Altitude captured: {current_alt:.0f}ft, holding {target_alt}ft")
            return

        # Calculate when to start descent (3:1 rule - 3nm per 1000ft)
        descent_distance_nm = abs(alt_diff) / 1000 * 3

        if alt_diff < -self.nav_config.altitude_capture_threshold_ft:
            # Need to descend
            if distance_nm <= descent_distance_nm:
                if self.flight_plan.phase != FlightPhase.DESCENT:
                    self.flight_plan.phase = FlightPhase.DESCENT
                    self.commander.set_altitude_bug(target_alt)
                    self.commander.set_vs_bug(-self.nav_config.descent_rate_fpm)
                    self.commander.engage_vs_mode()
                    logger.info(f"Starting descent to {target_alt}ft")

        elif alt_diff > self.nav_config.altitude_capture_threshold_ft:
            # Need to climb
            if self.flight_plan.phase != FlightPhase.CLIMB:
                self.flight_plan.phase = FlightPhase.CLIMB
                self.commander.set_altitude_bug(target_alt)
                self.commander.set_vs_bug(self.nav_config.climb_rate_fpm)
                self.commander.engage_vs_mode()
                logger.info(f"Climbing to {target_alt}ft")

    def release_to_pilot(self):
        """
        Gracefully hand control back to the pilot.
        Engages X-Plane autopilot first, then releases joystick override.
        """
        logger.info("RELEASING CONTROL TO PILOT")
        print("=== RELEASING CONTROL TO PILOT ===")

        self.active = False
        self.takeoff_state = TakeoffState.COMPLETE

        # Cancel running tasks
        if self._nav_task and not self._nav_task.done():
            self._nav_task.cancel()
        if self._takeoff_task and not self._takeoff_task.done():
            self._takeoff_task.cancel()

        if self.commander:
            # First, engage X-Plane's autopilot so plane doesn't crash
            self.commander.engage_autopilot_for_handoff()
            # Then release joystick override - PILOT HAS CONTROL
            self.commander.release_joystick_override()

        print("=== PILOT NOW HAS FULL CONTROL ===")

    def emergency_stop(self):
        """Emergency stop - disengage all automation and release to pilot"""
        logger.warning("EMERGENCY STOP triggered")
        print("EMERGENCY STOP triggered")

        self.active = False
        self.takeoff_state = TakeoffState.ABORTED

        if self.flight_plan:
            self.flight_plan.phase = FlightPhase.ABORTED

        # Cancel running tasks
        if self._nav_task and not self._nav_task.done():
            self._nav_task.cancel()
        if self._takeoff_task and not self._takeoff_task.done():
            self._takeoff_task.cancel()

        # Release joystick override FIRST - give pilot immediate control
        if self.commander:
            self.commander.release_joystick_override()
            self.commander.disengage_autopilot()
            self.commander.set_throttle(0.5)  # Set to idle-ish

    def stop(self):
        """Gracefully stop automation"""
        logger.info("Stopping flight automation")
        self.active = False

        if self._nav_task and not self._nav_task.done():
            self._nav_task.cancel()
        if self._takeoff_task and not self._takeoff_task.done():
            self._takeoff_task.cancel()

    def get_status(self) -> dict:
        """Get current automation status"""
        status = {
            "active": self.active,
            "takeoff_state": self.takeoff_state.value,
            "flight_phase": self.flight_plan.phase.value if self.flight_plan else "none",
        }

        if self.flight_plan:
            wp = self.flight_plan.get_current_waypoint()
            status.update({
                "current_waypoint": wp.name if wp else None,
                "waypoint_index": self.flight_plan.current_waypoint_index,
                "total_waypoints": len(self.flight_plan.waypoints),
                "departure": self.flight_plan.departure_icao,
                "destination": self.flight_plan.destination_icao,
                "cruise_altitude": self.flight_plan.cruise_altitude_ft
            })

        return status

    async def execute_autoland(self, runway_heading: float, field_elevation: float = 0):
        """
        Execute automated landing sequence.

        Phases:
        1. APPROACH - Descend to pattern altitude, align with runway
        2. FINAL - Descend on glideslope (3°)
        3. FLARE - Reduce descent rate near ground
        4. ROLLOUT - Touchdown, brakes, decelerate

        Args:
            runway_heading: Runway heading in degrees (magnetic)
            field_elevation: Airport elevation in feet MSL
        """
        print(f"\n=== AUTOLAND SEQUENCE STARTED ===")
        print(f"Runway heading: {runway_heading:.0f}°, Field elevation: {field_elevation:.0f}ft")

        if self.flight_plan:
            self.flight_plan.phase = FlightPhase.APPROACH

        await self.broadcast_status({
            "type": "autoland_status",
            "phase": "APPROACH",
            "runway_heading": runway_heading
        })

        # Landing configuration
        PATTERN_ALT_AGL = 1000  # Pattern altitude above field
        GLIDESLOPE_DEG = 3.0    # Standard ILS glideslope
        FLARE_HEIGHT_FT = 30    # Begin flare
        TOUCHDOWN_PITCH = 5     # Nose-up attitude for touchdown

        pattern_altitude = field_elevation + PATTERN_ALT_AGL

        # Phase tracking
        phase = "APPROACH"
        last_log_time = 0
        loop_start = asyncio.get_event_loop().time()

        # Stabilization
        aligned = False
        on_glideslope = False

        while self.active:
            try:
                flight_data = self.get_flight_data()
                if not flight_data:
                    await asyncio.sleep(0.1)
                    continue

                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - loop_start

                current_alt = flight_data.alt_msl
                current_agl = flight_data.alt_agl
                current_hdg = flight_data.heading_mag
                current_vs = flight_data.vertical_speed
                current_ias = flight_data.airspeed_ind
                current_pitch = flight_data.pitch
                current_roll = flight_data.roll

                # Calculate heading error to runway
                hdg_error = runway_heading - current_hdg
                if hdg_error > 180:
                    hdg_error -= 360
                elif hdg_error < -180:
                    hdg_error += 360

                # === PHASE: APPROACH ===
                if phase == "APPROACH":
                    # Descend to pattern altitude and align with runway
                    target_alt = pattern_altitude

                    # Check if aligned (within 10° of runway heading)
                    if abs(hdg_error) < 10:
                        aligned = True

                    # Check if at pattern altitude
                    if current_alt < pattern_altitude + 200 and aligned:
                        phase = "FINAL"
                        print(f"  [LAND] Transitioning to FINAL approach")
                        if self.flight_plan:
                            self.flight_plan.phase = FlightPhase.LANDING
                        await self.broadcast_status({"type": "autoland_status", "phase": "FINAL"})

                    # Altitude control - descend to pattern
                    alt_error = target_alt - current_alt
                    if alt_error < -100:
                        target_vs = max(-800, alt_error * 2)  # Descend
                    else:
                        target_vs = 0

                # === PHASE: FINAL APPROACH ===
                elif phase == "FINAL":
                    # Calculate glideslope descent rate
                    # 3° glideslope at ~120kts groundspeed ≈ 600 fpm descent
                    gs_kts = flight_data.groundspeed if hasattr(flight_data, 'groundspeed') else current_ias
                    target_vs = -gs_kts * math.tan(math.radians(GLIDESLOPE_DEG)) * 101.3  # Convert to fpm
                    target_vs = max(-800, min(-400, target_vs))  # Clamp between 400-800 fpm descent

                    # Reduce speed for landing
                    if current_ias > 90:
                        self.commander.set_throttle(0.3)  # Reduce power
                    elif current_ias > 75:
                        self.commander.set_throttle(0.4)
                    else:
                        self.commander.set_throttle(0.5)  # Maintain approach speed

                    # Transition to flare when close to ground
                    if current_agl < FLARE_HEIGHT_FT:
                        phase = "FLARE"
                        print(f"  [LAND] FLARE at {current_agl:.0f}ft AGL")
                        await self.broadcast_status({"type": "autoland_status", "phase": "FLARE"})

                # === PHASE: FLARE ===
                elif phase == "FLARE":
                    # Reduce descent rate, pitch up for touchdown
                    target_vs = -100  # Very gentle descent

                    # Reduce throttle to idle
                    self.commander.set_throttle(0.0)

                    # Pitch up for flare
                    target_pitch = TOUCHDOWN_PITCH
                    pitch_error = target_pitch - current_pitch
                    elevator = pitch_error * 0.05
                    elevator = max(-0.3, min(0.5, elevator))
                    self.commander.set_elevator(elevator)

                    # Check for touchdown (very low AGL and low vertical speed)
                    if current_agl < 5 or (current_agl < 15 and abs(current_vs) < 50):
                        phase = "ROLLOUT"
                        print(f"  [LAND] TOUCHDOWN! AGL:{current_agl:.0f}ft VS:{current_vs:.0f}fpm")
                        await self.broadcast_status({"type": "autoland_status", "phase": "ROLLOUT"})

                # === PHASE: ROLLOUT ===
                elif phase == "ROLLOUT":
                    # On the ground - apply brakes, maintain centerline
                    self.commander.set_throttle(0.0)  # Idle
                    self.commander.set_elevator(-0.3)  # Nose down to keep weight on wheels

                    # Apply brakes
                    self.commander.set_parking_brake(False)  # Make sure parking brake is off
                    # Use differential braking for steering if available, otherwise symmetric braking

                    # Rudder/nosewheel to maintain runway heading
                    rudder = hdg_error * 0.05
                    rudder = max(-0.5, min(0.5, rudder))
                    self.commander.set_rudder(rudder)
                    self.commander.set_nosewheel_steering(rudder * 2)

                    # Check if stopped
                    if current_ias < 10:
                        print(f"  [LAND] *** LANDING COMPLETE ***")
                        self.commander.set_parking_brake(True)
                        if self.flight_plan:
                            self.flight_plan.phase = FlightPhase.COMPLETE
                        await self.broadcast_status({
                            "type": "autoland_status",
                            "phase": "COMPLETE",
                            "message": "Landing complete"
                        })
                        break

                    # Skip normal control loop for rollout
                    await asyncio.sleep(0.05)
                    continue

                # === CONTROL INPUTS (for APPROACH and FINAL phases) ===
                if phase in ["APPROACH", "FINAL"]:
                    # Heading control - align with runway
                    desired_bank = hdg_error * 0.5
                    desired_bank = max(-20, min(20, desired_bank))

                    bank_error = desired_bank - current_roll
                    aileron = bank_error * 0.06
                    aileron = max(-0.5, min(0.5, aileron))
                    self.commander.set_aileron(aileron)

                    # Pitch/VS control
                    vs_error = target_vs - current_vs
                    pitch_correction = vs_error * 0.0002
                    elevator = 0.02 + pitch_correction  # Base nose-up trim
                    elevator = max(-0.3, min(0.3, elevator))
                    self.commander.set_elevator(elevator)

                    # Rudder for coordination
                    rudder = hdg_error * 0.01
                    rudder = max(-0.3, min(0.3, rudder))
                    self.commander.set_rudder(rudder)

                # Logging
                if elapsed - last_log_time >= 3:
                    print(f"  [{phase}] ALT:{current_alt:.0f}(AGL:{current_agl:.0f}) HDG:{current_hdg:.0f}°->{runway_heading:.0f}° VS:{current_vs:.0f} IAS:{current_ias:.0f}")
                    last_log_time = elapsed

                await asyncio.sleep(0.05)  # 20Hz

            except Exception as e:
                logger.error(f"Autoland error: {e}")
                await asyncio.sleep(0.1)

        print("=== AUTOLAND SEQUENCE ENDED ===")


# ============================================================================
# AUTO-TUNING SYSTEM
# Runs multiple flights autonomously to optimize controller parameters
# ============================================================================

@dataclass
class TuningParameters:
    """All tunable parameters for the flight controller"""
    # Takeoff PID gains (OPTIMIZED from 100-flight tuning session)
    takeoff_kp: float = 0.1208  # Was 0.08, increased 51% for better heading correction
    takeoff_ki: float = 0.02
    takeoff_kd: float = 0.0107  # Was 0.01, slight increase for damping

    # P-factor compensation - balanced for straight tracking
    p_factor_bias: float = 0.20  # Reduced - was overcorrecting to the right

    # Control limits
    rudder_limit: float = 1.0
    nosewheel_multiplier_low: float = 2.0
    nosewheel_multiplier_high: float = 0.3

    # Heading abort threshold
    heading_abort_deg: float = 45.0

    # In-flight gains
    flight_bank_gain: float = 0.25
    flight_aileron_gain: float = 0.08
    flight_rudder_gain: float = 0.008

    def to_dict(self) -> dict:
        return {
            'takeoff_kp': self.takeoff_kp,
            'takeoff_ki': self.takeoff_ki,
            'takeoff_kd': self.takeoff_kd,
            'p_factor_bias': self.p_factor_bias,
            'rudder_limit': self.rudder_limit,
            'nosewheel_multiplier_low': self.nosewheel_multiplier_low,
            'nosewheel_multiplier_high': self.nosewheel_multiplier_high,
            'heading_abort_deg': self.heading_abort_deg,
            'flight_bank_gain': self.flight_bank_gain,
            'flight_aileron_gain': self.flight_aileron_gain,
            'flight_rudder_gain': self.flight_rudder_gain,
        }


@dataclass
class FlightMetrics:
    """Metrics collected from a single flight"""
    flight_number: int = 0
    success: bool = False
    aborted: bool = False
    abort_reason: str = ""

    # Takeoff metrics
    max_heading_deviation: float = 0.0
    avg_heading_deviation: float = 0.0
    time_to_liftoff: float = 0.0
    liftoff_speed: float = 0.0
    heading_oscillations: int = 0

    # Flight metrics
    max_bank_angle: float = 0.0
    max_pitch_deviation: float = 0.0
    track_following_error_avg: float = 0.0

    # Landing metrics
    touchdown_vs: float = 0.0
    landing_success: bool = False

    # Overall score (higher is better)
    score: float = 0.0

    def calculate_score(self) -> float:
        """Calculate overall flight score (0-100)"""
        score = 100.0

        # Penalize for abort
        if self.aborted:
            return 0.0

        # Takeoff scoring (40 points max)
        # Heading deviation penalty
        score -= min(20, self.max_heading_deviation * 0.5)
        score -= min(10, self.avg_heading_deviation * 1.0)
        # Oscillation penalty
        score -= min(10, self.heading_oscillations * 2)

        # Flight scoring (30 points max)
        score -= min(15, self.max_bank_angle / 3)
        score -= min(15, self.track_following_error_avg * 10)

        # Landing scoring (30 points max)
        if self.landing_success:
            score += 10
            # Soft landing bonus
            if abs(self.touchdown_vs) < 200:
                score += 10
            elif abs(self.touchdown_vs) < 400:
                score += 5

        self.score = max(0, score)
        return self.score


class AutoTuner:
    """
    Autonomous flight tuning system.
    Runs multiple flights, collects metrics, and optimizes parameters.
    """

    def __init__(self, flight_manager: 'FlightManager'):
        self.fm = flight_manager
        self.params = TuningParameters()
        self.best_params = TuningParameters()
        self.best_score = 0.0
        self.flight_history: List[FlightMetrics] = []
        self.running = False

        # Runway configuration (will be set when starting)
        self.runway_lat = 33.9425  # KLAX default
        self.runway_lon = -118.4081
        self.runway_heading = 250.0
        self.runway_elevation = 125.0  # feet

        # Tuning configuration
        self.total_flights = 0
        self.successful_flights = 0
        self.parameter_history: List[dict] = []

    def set_runway(self, lat: float, lon: float, heading: float, elevation: float = 0):
        """Configure the runway for tuning flights"""
        self.runway_lat = lat
        self.runway_lon = lon
        self.runway_heading = heading
        self.runway_elevation = elevation
        print(f"[TUNER] Runway set: {lat:.6f}, {lon:.6f}, HDG {heading:.0f}°, ELEV {elevation:.0f}ft")

    def apply_parameters(self):
        """Apply current tuning parameters to the flight manager"""
        # These will be read by the takeoff sequence
        # Store in flight manager for access during flight
        self.fm._tuning_params = self.params

    def collect_takeoff_metrics(self, heading_errors: List[float], liftoff_time: float,
                                 liftoff_speed: float) -> FlightMetrics:
        """Analyze takeoff data and create metrics"""
        metrics = FlightMetrics(flight_number=self.total_flights)

        if heading_errors:
            metrics.max_heading_deviation = max(abs(e) for e in heading_errors)
            metrics.avg_heading_deviation = sum(abs(e) for e in heading_errors) / len(heading_errors)

            # Count oscillations (sign changes in error)
            oscillations = 0
            for i in range(1, len(heading_errors)):
                if heading_errors[i] * heading_errors[i-1] < 0:
                    oscillations += 1
            metrics.heading_oscillations = oscillations

        metrics.time_to_liftoff = liftoff_time
        metrics.liftoff_speed = liftoff_speed

        return metrics

    def adjust_parameters(self, metrics: FlightMetrics):
        """
        Adjust parameters based on flight results.
        Uses a simple hill-climbing approach with random perturbations.
        """
        import random

        # If this was a good flight, small adjustments
        # If this was a bad flight, larger adjustments
        if metrics.score > self.best_score:
            # New best! Save these parameters
            self.best_score = metrics.score
            self.best_params = TuningParameters(**self.params.to_dict())
            print(f"[TUNER] NEW BEST SCORE: {metrics.score:.1f}")
            adjustment_scale = 0.05  # Small tweaks
        elif metrics.aborted:
            # Flight failed - make bigger changes
            adjustment_scale = 0.2
            # Revert to best known parameters with perturbation
            self.params = TuningParameters(**self.best_params.to_dict())
        else:
            adjustment_scale = 0.1

        # Analyze specific issues and adjust accordingly
        if metrics.max_heading_deviation > 30:
            # Too much deviation - increase gains
            self.params.takeoff_kp *= (1 + random.uniform(0, adjustment_scale))
            self.params.p_factor_bias *= (1 + random.uniform(0, adjustment_scale))
            print(f"[TUNER] Increasing Kp to {self.params.takeoff_kp:.4f}, P-factor to {self.params.p_factor_bias:.3f}")

        elif metrics.heading_oscillations > 5:
            # Too much oscillation - decrease Kp, increase Kd
            self.params.takeoff_kp *= (1 - random.uniform(0, adjustment_scale))
            self.params.takeoff_kd *= (1 + random.uniform(0, adjustment_scale))
            print(f"[TUNER] Reducing oscillation: Kp={self.params.takeoff_kp:.4f}, Kd={self.params.takeoff_kd:.4f}")

        elif metrics.avg_heading_deviation > 10:
            # Persistent drift - increase Ki
            self.params.takeoff_ki *= (1 + random.uniform(0, adjustment_scale))
            print(f"[TUNER] Increasing Ki to {self.params.takeoff_ki:.4f}")

        # Random exploration with small probability
        if random.random() < 0.2:
            param_to_adjust = random.choice(['takeoff_kp', 'takeoff_ki', 'takeoff_kd', 'p_factor_bias'])
            current_val = getattr(self.params, param_to_adjust)
            new_val = current_val * (1 + random.uniform(-0.15, 0.15))
            setattr(self.params, param_to_adjust, new_val)
            print(f"[TUNER] Random exploration: {param_to_adjust} = {new_val:.4f}")

        # Clamp parameters to reasonable ranges
        self.params.takeoff_kp = max(0.01, min(0.5, self.params.takeoff_kp))
        self.params.takeoff_ki = max(0.001, min(0.2, self.params.takeoff_ki))
        self.params.takeoff_kd = max(0.001, min(0.1, self.params.takeoff_kd))
        self.params.p_factor_bias = max(0.0, min(0.8, self.params.p_factor_bias))

        # Save parameter history
        self.parameter_history.append({
            'flight': self.total_flights,
            'score': metrics.score,
            **self.params.to_dict()
        })

    async def run_single_flight(self, flight_type: str = "full") -> FlightMetrics:
        """
        Run a single tuning flight and collect metrics.

        Args:
            flight_type: "takeoff_only", "pattern", or "full"
        """
        self.total_flights += 1
        print(f"\n{'='*60}")
        print(f"[TUNER] FLIGHT #{self.total_flights} (Mode: {flight_type})")
        print(f"{'='*60}")
        print(f"Parameters: Kp={self.params.takeoff_kp:.4f} Ki={self.params.takeoff_ki:.4f} Kd={self.params.takeoff_kd:.4f} P-factor={self.params.p_factor_bias:.3f}")

        # Reset aircraft to runway
        self.fm.commander.reposition_aircraft(
            self.runway_lat, self.runway_lon,
            self.runway_elevation, self.runway_heading,
            on_ground=True
        )
        await asyncio.sleep(2.0)  # Wait for position to settle

        # Apply current parameters
        self.apply_parameters()

        # Collect metrics during flight
        heading_errors = []
        bank_angles = []
        track_errors = []
        start_time = asyncio.get_event_loop().time()
        liftoff_time = 0
        liftoff_speed = 0
        aborted = False
        abort_reason = ""

        # Flight phase configuration
        PATTERN_ALTITUDE = 1000  # feet AGL
        CRUISE_DURATION = 60.0   # seconds of cruise flight (1 minute)
        TOTAL_FLIGHT_TIME = 180.0  # max 3 minutes per flight

        # Run takeoff with metric collection
        try:
            self.fm.active = True
            self.fm._current_throttle = 0.0

            # Set initial state
            self.fm.commander.set_parking_brake(True)
            self.fm.commander.set_throttle(0.0)
            await asyncio.sleep(0.5)

            # Release brakes and throttle up
            self.fm.commander.set_parking_brake(False)
            for _ in range(10):
                self.fm._current_throttle = min(1.0, self.fm._current_throttle + 0.15)
                self.fm.commander.set_throttle(self.fm._current_throttle)
                await asyncio.sleep(0.1)

            # ============ PHASE 1: TAKEOFF ROLL ============
            # With centerline tracking and proper rotation timing
            print("[TUNER] Phase: TAKEOFF ROLL (Centerline Tracking)")
            iteration = 0
            heading_integral = 0.0
            last_heading_error = 0.0
            last_time = asyncio.get_event_loop().time()

            # Store initial position for centerline tracking
            initial_lat = None
            initial_lon = None

            # Rotation parameters - Cirrus SR22 Vr is ~65-70 KIAS
            ROTATION_START_SPEED = 50   # Start applying back pressure
            ROTATION_FULL_SPEED = 65    # Full rotation authority
            ELEVATOR_AUTHORITY = 0.5    # Increased from 0.25

            # Centerline tracking gain
            CROSSTRACK_GAIN = 0.15      # Heading correction per foot of deviation

            while self.fm.active and self.running:
                flight_data = self.fm.get_flight_data()
                if not flight_data:
                    await asyncio.sleep(0.05)
                    continue

                iteration += 1
                current_time = asyncio.get_event_loop().time()
                dt = current_time - last_time
                last_time = current_time
                elapsed = current_time - start_time

                ias = flight_data.airspeed_ind
                current_heading = flight_data.heading_mag
                current_lat = flight_data.lat
                current_lon = flight_data.lon

                # Initialize reference position on first iteration
                if initial_lat is None:
                    initial_lat = current_lat
                    initial_lon = current_lon

                # ===== CENTERLINE TRACKING =====
                # Calculate cross-track error (perpendicular distance from runway centerline)
                import math

                # Convert position difference to meters
                lat_diff_m = (current_lat - initial_lat) * 111320  # North is positive
                lon_diff_m = (current_lon - initial_lon) * 111320 * math.cos(math.radians(initial_lat))  # East is positive

                # Runway direction (unit vector pointing down runway)
                rwy_rad = math.radians(self.runway_heading)

                # Cross-track error using 2D cross product
                # XTE = position × runway_direction (perpendicular component)
                # Positive XTE = right of centerline, Negative XTE = left of centerline
                crosstrack_m = lat_diff_m * math.sin(rwy_rad) - lon_diff_m * math.cos(rwy_rad)
                crosstrack_ft = crosstrack_m * 3.28084

                # ===== HEADING CONTROL WITH CENTERLINE CORRECTION =====
                # Base heading error (positive = need to turn right)
                heading_error = self.runway_heading - current_heading
                if heading_error > 180:
                    heading_error -= 360
                elif heading_error < -180:
                    heading_error += 360

                # Centerline correction: steer toward centerline
                # If right of centerline (positive XTE), steer left (negative heading adjustment)
                # Use proportional + rate-based correction
                centerline_correction = -crosstrack_ft * CROSSTRACK_GAIN
                centerline_correction = max(-8, min(8, centerline_correction))  # Limit to 8°

                # Only apply centerline correction at higher speeds when rudder is effective
                if ias < 30:
                    centerline_correction *= 0.3  # Reduce at low speed

                # Combined heading error
                total_heading_error = heading_error + centerline_correction

                heading_errors.append(heading_error)

                # PID control with current parameters
                heading_integral += total_heading_error * dt
                heading_integral = max(-50.0, min(50.0, heading_integral))
                heading_derivative = (total_heading_error - last_heading_error) / max(dt, 0.001)
                last_heading_error = total_heading_error

                pid_output = (self.params.takeoff_kp * total_heading_error +
                             self.params.takeoff_ki * heading_integral +
                             self.params.takeoff_kd * heading_derivative)

                speed_factor = min(1.0, ias / 60.0)
                p_factor_compensation = self.params.p_factor_bias * speed_factor

                rudder = pid_output + p_factor_compensation
                rudder = max(-self.params.rudder_limit, min(self.params.rudder_limit, rudder))

                if ias < 40:
                    nosewheel = rudder * self.params.nosewheel_multiplier_low
                else:
                    nosewheel = rudder * self.params.nosewheel_multiplier_high
                nosewheel = max(-1.0, min(1.0, nosewheel))

                # Apply controls
                self.fm.commander.set_throttle(1.0)
                self.fm.commander.set_rudder(rudder)
                self.fm.commander.set_nosewheel_steering(nosewheel)
                self.fm.commander.set_aileron(-flight_data.roll * 0.03)

                # ===== IMPROVED ROTATION =====
                # Start rotation earlier with more authority
                if ias >= ROTATION_START_SPEED:
                    # Progressive rotation from start speed to full speed
                    progress = min(1.0, (ias - ROTATION_START_SPEED) / (ROTATION_FULL_SPEED - ROTATION_START_SPEED))
                    elevator = progress * ELEVATOR_AUTHORITY
                    self.fm.commander.set_elevator(elevator)

                # Check for liftoff
                if flight_data.alt_agl > 20:
                    liftoff_time = elapsed
                    liftoff_speed = ias
                    print(f"[TUNER] LIFTOFF at {liftoff_time:.1f}s, {liftoff_speed:.0f}kts, XTE:{crosstrack_ft:.1f}ft")
                    self.successful_flights += 1
                    break

                # Check for abort conditions
                if abs(heading_error) > self.params.heading_abort_deg and ias > 30:
                    aborted = True
                    abort_reason = f"Heading deviation {heading_error:.0f}°"
                    print(f"[TUNER] ABORT: {abort_reason}")
                    self.fm.commander.set_throttle(0.0)
                    break

                if elapsed > 60:  # Timeout
                    aborted = True
                    abort_reason = "Timeout - no liftoff in 60s"
                    print(f"[TUNER] ABORT: {abort_reason}")
                    self.fm.commander.set_throttle(0.0)
                    break

                # Log every 20 iterations (with cross-track error)
                if iteration % 20 == 0:
                    print(f"  IAS:{ias:.0f} HDG:{current_heading:.0f}° ERR:{heading_error:.1f}° XTE:{crosstrack_ft:+.0f}ft RUD:{rudder:.2f}")

                await asyncio.sleep(0.05)

            # ============ PHASE 2: CLIMB WITH SMOOTH ALTITUDE CAPTURE ============
            if not aborted and self.running and flight_type in ["pattern", "full"]:
                print(f"[TUNER] Phase: CLIMB to {PATTERN_ALTITUDE}ft AGL (Smooth Capture)")
                target_heading = self.runway_heading
                climb_start = asyncio.get_event_loop().time()

                # Smooth altitude capture parameters
                CAPTURE_START_ALT = PATTERN_ALTITUDE * 0.5  # Start smoothing at 50% of target (earlier!)
                INITIAL_CLIMB_PITCH = 8.0   # Gentler initial climb pitch
                last_pitch_target = INITIAL_CLIMB_PITCH
                pitch_rate_limit = 1.5  # Slower pitch changes for smoother transitions
                last_vs = 0

                while self.fm.active and self.running:
                    flight_data = self.fm.get_flight_data()
                    if not flight_data:
                        await asyncio.sleep(0.05)
                        continue

                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time
                    dt = max(0.01, current_time - last_time)

                    alt_agl = flight_data.alt_agl
                    ias = flight_data.airspeed_ind
                    current_heading = flight_data.heading_mag
                    roll = flight_data.roll
                    pitch = flight_data.pitch
                    vs = flight_data.vertical_speed

                    # Collect bank angle data
                    bank_angles.append(abs(roll))

                    # Heading control
                    heading_error = target_heading - current_heading
                    if heading_error > 180:
                        heading_error -= 360
                    elif heading_error < -180:
                        heading_error += 360
                    track_errors.append(abs(heading_error))

                    # Bank angle to correct heading - gentler during climb
                    target_bank = heading_error * self.params.flight_bank_gain * 0.5  # Reduced during climb
                    target_bank = max(-15, min(15, target_bank))  # Limit bank to 15° during climb

                    # Aileron to achieve target bank
                    bank_error = target_bank - roll
                    aileron = bank_error * self.params.flight_aileron_gain
                    aileron = max(-0.3, min(0.3, aileron))

                    # Rudder to coordinate turn
                    rudder = roll * self.params.flight_rudder_gain
                    rudder = max(-0.2, min(0.2, rudder))

                    # ===== SMOOTH ALTITUDE CAPTURE =====
                    # Gradually reduce pitch AND use VS to control level-off
                    remaining_alt = PATTERN_ALTITUDE - alt_agl

                    if alt_agl < CAPTURE_START_ALT:
                        # Full climb - constant pitch
                        raw_target_pitch = INITIAL_CLIMB_PITCH
                    else:
                        # Altitude capture - use both altitude AND vertical speed
                        # Calculate desired VS based on remaining altitude (decelerate climb rate)
                        # At 500ft remaining: VS ~1000, at 100ft remaining: VS ~200, at 0ft: VS ~0
                        desired_vs = remaining_alt * 2.0  # Simple linear relationship
                        desired_vs = max(0, min(1200, desired_vs))

                        # Calculate pitch needed to achieve desired VS
                        # If VS is too high, reduce pitch; if VS is too low, increase pitch
                        vs_error = desired_vs - vs
                        vs_correction = vs_error * 0.003  # Convert VS error to pitch adjustment

                        # Base pitch decreases as we approach target
                        capture_progress = 1.0 - (remaining_alt / (PATTERN_ALTITUDE - CAPTURE_START_ALT))
                        capture_progress = max(0, min(1, capture_progress))

                        # Smooth S-curve
                        smooth_factor = capture_progress * capture_progress * (3 - 2 * capture_progress)
                        base_pitch = INITIAL_CLIMB_PITCH * (1 - smooth_factor * 0.9)  # Don't go all the way to 0

                        # Combine base pitch with VS correction
                        raw_target_pitch = base_pitch + vs_correction
                        raw_target_pitch = max(-2, min(INITIAL_CLIMB_PITCH, raw_target_pitch))

                    # Rate-limit the pitch target for smooth transitions
                    pitch_change = raw_target_pitch - last_pitch_target
                    max_change = pitch_rate_limit * dt
                    if abs(pitch_change) > max_change:
                        pitch_change = max_change if pitch_change > 0 else -max_change
                    target_pitch = last_pitch_target + pitch_change
                    last_pitch_target = target_pitch

                    # Pitch control with integral for steady-state accuracy
                    pitch_error = target_pitch - pitch
                    elevator = pitch_error * 0.07
                    elevator = max(-0.4, min(0.4, elevator))

                    # Throttle management - gradual reduction during capture
                    if alt_agl > CAPTURE_START_ALT:
                        # Reduce throttle progressively as we level off
                        capture_pct = (alt_agl - CAPTURE_START_ALT) / (PATTERN_ALTITUDE - CAPTURE_START_ALT)
                        throttle = 1.0 - (capture_pct * 0.25)  # 1.0 -> 0.75 as we level
                    else:
                        throttle = 1.0  # Full power for climb

                    # Apply controls
                    self.fm.commander.set_throttle(throttle)
                    self.fm.commander.set_aileron(aileron)
                    self.fm.commander.set_rudder(rudder)
                    self.fm.commander.set_elevator(elevator)

                    # Check if we reached pattern altitude with low VS (stabilized)
                    # Need both altitude AND low vertical speed for smooth handoff
                    if alt_agl >= PATTERN_ALTITUDE * 0.95 and abs(vs) < 200:
                        print(f"[TUNER] Level at {alt_agl:.0f}ft AGL, VS:{vs:.0f}fpm (stabilized)")
                        break
                    # Alternative: if above altitude and descending slightly, we've captured
                    if alt_agl >= PATTERN_ALTITUDE and vs < 100:
                        print(f"[TUNER] Altitude captured at {alt_agl:.0f}ft AGL, VS:{vs:.0f}fpm")
                        break

                    # Safety checks
                    if alt_agl < 10 and (current_time - climb_start) > 10:
                        aborted = True
                        abort_reason = "Failed to climb - crashed"
                        print(f"[TUNER] ABORT: {abort_reason}")
                        break

                    if elapsed > TOTAL_FLIGHT_TIME:
                        print(f"[TUNER] Time limit reached during climb")
                        break

                    # Log every second
                    if int(current_time) != int(last_time):
                        print(f"  ALT:{alt_agl:.0f}ft IAS:{ias:.0f}kts PITCH:{pitch:.1f}° (tgt:{target_pitch:.1f}°) VS:{vs:+.0f}fpm")

                    last_time = current_time
                    await asyncio.sleep(0.05)

            # ============ PHASE 3: CRUISE/LEVEL FLIGHT ============
            if not aborted and self.running and flight_type in ["pattern", "full"]:
                print(f"[TUNER] Phase: CRUISE for {CRUISE_DURATION:.0f}s (Stabilized)")
                cruise_start = asyncio.get_event_loop().time()
                target_heading = self.runway_heading
                target_altitude = PATTERN_ALTITUDE

                # Cruise PID state
                alt_integral = 0.0
                hdg_integral = 0.0
                last_alt_error = 0.0
                last_hdg_error = 0.0

                while self.fm.active and self.running:
                    flight_data = self.fm.get_flight_data()
                    if not flight_data:
                        await asyncio.sleep(0.05)
                        continue

                    current_time = asyncio.get_event_loop().time()
                    cruise_elapsed = current_time - cruise_start
                    total_elapsed = current_time - start_time
                    dt = max(0.01, current_time - last_time)

                    alt_agl = flight_data.alt_agl
                    ias = flight_data.airspeed_ind
                    current_heading = flight_data.heading_mag
                    roll = flight_data.roll
                    pitch = flight_data.pitch
                    vs = flight_data.vertical_speed

                    # Collect metrics
                    bank_angles.append(abs(roll))

                    # ===== HEADING CONTROL WITH PID =====
                    heading_error = target_heading - current_heading
                    if heading_error > 180:
                        heading_error -= 360
                    elif heading_error < -180:
                        heading_error += 360
                    track_errors.append(abs(heading_error))

                    # Heading PID
                    hdg_integral += heading_error * dt
                    hdg_integral = max(-30, min(30, hdg_integral))
                    hdg_derivative = (heading_error - last_hdg_error) / dt
                    last_hdg_error = heading_error

                    # Target bank from heading error (gentler gains for stability)
                    target_bank = (heading_error * 0.8 +          # P
                                   hdg_integral * 0.05 +           # I
                                   hdg_derivative * 0.1)           # D
                    target_bank = max(-15, min(15, target_bank))   # Limit bank to 15°

                    # Aileron to achieve target bank
                    bank_error = target_bank - roll
                    aileron = bank_error * 0.06
                    aileron = max(-0.3, min(0.3, aileron))

                    # Rudder to coordinate (adverse yaw compensation)
                    rudder = aileron * 0.3 + roll * 0.005
                    rudder = max(-0.2, min(0.2, rudder))

                    # ===== ALTITUDE HOLD WITH PID =====
                    alt_error = target_altitude - alt_agl

                    # Altitude PID
                    alt_integral += alt_error * dt
                    alt_integral = max(-500, min(500, alt_integral))
                    alt_derivative = (alt_error - last_alt_error) / dt
                    last_alt_error = alt_error

                    # Also use vertical speed for damping
                    vs_error = 0 - vs  # Target VS is 0 for level flight

                    # Pitch from altitude error + VS damping
                    target_pitch = (alt_error * 0.015 +            # P - altitude error
                                    alt_integral * 0.001 +          # I - accumulated error
                                    vs_error * 0.003)               # D - VS damping
                    target_pitch = max(-5, min(8, target_pitch))

                    pitch_error = target_pitch - pitch
                    elevator = pitch_error * 0.05
                    elevator = max(-0.25, min(0.25, elevator))

                    # ===== THROTTLE FOR SPEED CONTROL =====
                    target_ias = 110  # knots (slightly higher for stability)
                    speed_error = target_ias - ias
                    throttle = 0.65 + (speed_error * 0.008)
                    throttle = max(0.4, min(0.9, throttle))

                    # Apply controls
                    self.fm.commander.set_throttle(throttle)
                    self.fm.commander.set_aileron(aileron)
                    self.fm.commander.set_rudder(rudder)
                    self.fm.commander.set_elevator(elevator)

                    # Check if cruise duration completed
                    if cruise_elapsed >= CRUISE_DURATION:
                        print(f"[TUNER] Cruise phase complete - Final ALT:{alt_agl:.0f}ft HDG:{current_heading:.0f}°")
                        break

                    # Safety checks
                    if alt_agl < 200:
                        aborted = True
                        abort_reason = f"Altitude too low during cruise ({alt_agl:.0f}ft)"
                        print(f"[TUNER] ABORT: {abort_reason}")
                        break

                    if total_elapsed > TOTAL_FLIGHT_TIME:
                        print(f"[TUNER] Total flight time limit reached")
                        break

                    # Log every 5 seconds
                    if int(cruise_elapsed) % 5 == 0 and int(cruise_elapsed) != int(cruise_elapsed - 0.05):
                        print(f"  ALT:{alt_agl:.0f}ft IAS:{ias:.0f}kts HDG:{current_heading:.0f}° BANK:{roll:.1f}° VS:{vs:+.0f} t={cruise_elapsed:.0f}s")

                    last_time = current_time
                    await asyncio.sleep(0.05)

        except Exception as e:
            aborted = True
            abort_reason = str(e)
            print(f"[TUNER] ERROR: {e}")

        finally:
            # Stop the aircraft
            self.fm.commander.set_throttle(0.0)
            self.fm.active = False

        # Create metrics with extended data
        metrics = self.collect_takeoff_metrics(heading_errors, liftoff_time, liftoff_speed)
        metrics.aborted = aborted
        metrics.abort_reason = abort_reason
        metrics.success = not aborted and liftoff_time > 0

        # Add flight metrics
        if bank_angles:
            metrics.max_bank_angle = max(bank_angles)
        if track_errors:
            metrics.track_following_error_avg = sum(track_errors) / len(track_errors)

        metrics.calculate_score()

        total_flight_time = asyncio.get_event_loop().time() - start_time
        print(f"[TUNER] Flight #{self.total_flights} Score: {metrics.score:.1f} Duration: {total_flight_time:.1f}s")
        self.flight_history.append(metrics)

        return metrics

    async def run_tuning_session(self, num_flights: int = 100, duration_hours: float = 5.0):
        """
        Run an extended tuning session.

        Args:
            num_flights: Maximum number of flights to run
            duration_hours: Maximum duration in hours
        """
        print(f"\n{'='*60}")
        print(f"[TUNER] STARTING {duration_hours}-HOUR TUNING SESSION")
        print(f"[TUNER] Max flights: {num_flights}")
        print(f"{'='*60}\n")

        self.running = True
        start_time = asyncio.get_event_loop().time()
        max_duration = duration_hours * 3600  # Convert to seconds

        try:
            flight_num = 0
            while self.running and flight_num < num_flights:
                elapsed = asyncio.get_event_loop().time() - start_time

                # Check time limit
                if elapsed > max_duration:
                    print(f"[TUNER] Time limit reached ({duration_hours} hours)")
                    break

                # Run a flight
                metrics = await self.run_single_flight()

                # Adjust parameters based on results
                self.adjust_parameters(metrics)

                flight_num += 1

                # Print progress every 10 flights
                if flight_num % 10 == 0:
                    hours_elapsed = elapsed / 3600
                    success_rate = (self.successful_flights / self.total_flights * 100) if self.total_flights > 0 else 0
                    print(f"\n[TUNER] === PROGRESS: {flight_num} flights, {hours_elapsed:.1f}h elapsed ===")
                    print(f"[TUNER] Success rate: {success_rate:.1f}%")
                    print(f"[TUNER] Best score: {self.best_score:.1f}")
                    print(f"[TUNER] Best params: Kp={self.best_params.takeoff_kp:.4f} Ki={self.best_params.takeoff_ki:.4f} Kd={self.best_params.takeoff_kd:.4f} P-factor={self.best_params.p_factor_bias:.3f}")
                    print()

                # Brief pause between flights
                await asyncio.sleep(3.0)

        except asyncio.CancelledError:
            print("[TUNER] Session cancelled")
        except Exception as e:
            print(f"[TUNER] Session error: {e}")
        finally:
            self.running = False

        # Final report
        self.print_final_report()

    def stop(self):
        """Stop the tuning session"""
        self.running = False
        print("[TUNER] Stopping tuning session...")

    def print_final_report(self):
        """Print comprehensive tuning results"""
        print(f"\n{'='*60}")
        print(f"[TUNER] FINAL TUNING REPORT")
        print(f"{'='*60}")
        print(f"Total flights: {self.total_flights}")
        print(f"Successful flights: {self.successful_flights}")
        success_rate = (self.successful_flights / self.total_flights * 100) if self.total_flights > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        print(f"\nBest score: {self.best_score:.1f}")
        print(f"\nOptimal Parameters:")
        print(f"  takeoff_kp = {self.best_params.takeoff_kp:.6f}")
        print(f"  takeoff_ki = {self.best_params.takeoff_ki:.6f}")
        print(f"  takeoff_kd = {self.best_params.takeoff_kd:.6f}")
        print(f"  p_factor_bias = {self.best_params.p_factor_bias:.6f}")
        print(f"  rudder_limit = {self.best_params.rudder_limit:.2f}")
        print(f"  nosewheel_multiplier_low = {self.best_params.nosewheel_multiplier_low:.2f}")
        print(f"  nosewheel_multiplier_high = {self.best_params.nosewheel_multiplier_high:.2f}")

        # Score distribution
        if self.flight_history:
            scores = [m.score for m in self.flight_history]
            print(f"\nScore distribution:")
            print(f"  Min: {min(scores):.1f}")
            print(f"  Max: {max(scores):.1f}")
            print(f"  Avg: {sum(scores)/len(scores):.1f}")

        print(f"{'='*60}\n")

    def get_status(self) -> dict:
        """Get current tuning status"""
        return {
            'running': self.running,
            'total_flights': self.total_flights,
            'successful_flights': self.successful_flights,
            'best_score': self.best_score,
            'current_params': self.params.to_dict(),
            'best_params': self.best_params.to_dict(),
        }
