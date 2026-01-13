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
from typing import List, Optional, Callable, Any
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
            print("Step 3: Takeoff roll - centerline tracking with rudder")
            await self.broadcast_status({"type": "takeoff_status", "state": "TAKEOFF_ROLL"})

            if self.flight_plan:
                self.flight_plan.phase = FlightPhase.TAKEOFF_ROLL

            last_speed_print = -1  # Start at -1 so first print happens immediately
            elevator = 0.0
            brake_release_counter = 0
            iteration = 0
            while self.active:
                flight_data = self.get_flight_data()
                if flight_data:
                    ias = flight_data.airspeed_ind
                    iteration += 1

                    # Print first iteration to confirm data
                    if iteration == 1:
                        print(f"  [START] IAS:{ias:.0f} PITCH:{flight_data.pitch:.1f}° AGL:{flight_data.alt_agl:.0f}")

                    # Keep brakes released during roll (every 10 iterations = 0.5s)
                    brake_release_counter += 1
                    if brake_release_counter >= 10:
                        self.commander.set_parking_brake(False)
                        brake_release_counter = 0
                    current_heading = flight_data.heading_mag
                    current_roll = flight_data.roll
                    current_lat = flight_data.lat
                    current_lon = flight_data.lon

                    # === CROSS-TRACK ERROR (lateral deviation from runway centerline) ===
                    # Calculate how far off the centerline we are (in meters, then convert to correction)
                    # Use the perpendicular distance from current position to the runway line
                    cross_track_error = self._calculate_cross_track_error(
                        initial_lat, initial_lon, runway_heading,
                        current_lat, current_lon
                    )
                    # cross_track_error: positive = right of centerline, negative = left

                    # === HEADING ERROR ===
                    heading_error = runway_heading - current_heading
                    if heading_error > 180:
                        heading_error -= 360
                    elif heading_error < -180:
                        heading_error += 360

                    # === STEERING CONTROL ===
                    # P-FACTOR COMPENSATION: Single-engine props yaw LEFT on takeoff
                    # Need RIGHT rudder (positive) to counteract
                    # Also apply heading correction to stay on runway heading

                    # Strong P-factor compensation - props really pull left hard
                    if ias < 30:
                        p_factor_rudder = 0.5  # Very strong right rudder at low speed
                    elif ias < 50:
                        p_factor_rudder = 0.4  # Strong
                    elif ias < 70:
                        p_factor_rudder = 0.25  # Moderate
                    else:
                        p_factor_rudder = 0.15  # Light at high speed

                    # Heading correction to maintain runway heading
                    # heading_error > 0 means we need to turn RIGHT
                    # heading_error < 0 means we need to turn LEFT
                    heading_correction = heading_error * 0.05  # Proportional correction
                    heading_correction = max(-0.3, min(0.3, heading_correction))

                    # Total rudder = P-factor compensation + heading correction
                    rudder = p_factor_rudder + heading_correction
                    rudder = max(-0.5, min(0.5, rudder))

                    # Nosewheel - NOTE: may need opposite sign in X-Plane
                    # Try NEGATIVE to see if nosewheel convention is inverted
                    nosewheel = -rudder * 2.0  # Inverted and stronger
                    nosewheel = max(-1.0, min(1.0, nosewheel))

                    self.commander.set_rudder(rudder)
                    self.commander.set_nosewheel_steering(nosewheel)

                    # === ROLL CONTROL (keep wings level) ===
                    aileron = -current_roll * 0.02  # Gentle wing leveling
                    aileron = max(-0.15, min(0.15, aileron))
                    self.commander.set_aileron(aileron)

                    # === GRADUAL PITCH UP FOR ROTATION ===
                    if ias < 55:
                        elevator = 0.0  # No pitch input below 55kts
                    elif ias < self.takeoff_config.vr_speed_kts:
                        # Build elevator gradually from 55kts to Vr (70kts)
                        progress = (ias - 55) / max(1, self.takeoff_config.vr_speed_kts - 55)
                        elevator = progress * 0.2  # Up to 20% elevator at Vr
                    else:
                        # At or above Vr - apply rotation pitch (gentle)
                        elevator = 0.25  # Gentle pitch up for rotation
                    self.commander.set_elevator(elevator)

                    # Print speed updates - every 5kts, plus first few iterations
                    xte_ft = cross_track_error * 6076  # nm to feet
                    raw_hdg_error = runway_heading - current_heading
                    if raw_hdg_error > 180:
                        raw_hdg_error -= 360
                    elif raw_hdg_error < -180:
                        raw_hdg_error += 360
                    if iteration <= 5 or ias >= last_speed_print + 5:
                        print(f"  IAS:{ias:.0f} HDG:{current_heading:.0f}°->TGT:{runway_heading:.0f}° ERR:{raw_hdg_error:.1f}° | Pfac:{p_factor_rudder:.2f} HdgCorr:{heading_correction:.2f} RUD:{rudder:.2f} NW:{nosewheel:.2f}")
                        if ias >= last_speed_print + 5:
                            last_speed_print = int(ias)

                    # Check for liftoff
                    if flight_data.alt_agl > 20:
                        print(f"  LIFTOFF! AGL={flight_data.alt_agl:.0f} ft, IAS={ias:.0f} kts")
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
                        # Calculate bearing and distance to waypoint
                        bearing = self.nav_calculator.calculate_bearing(
                            current_lat, current_lon,
                            current_wp.lat, current_wp.lon
                        )
                        distance = self.nav_calculator.calculate_distance_nm(
                            current_lat, current_lon,
                            current_wp.lat, current_wp.lon
                        )

                        # Rate-limit heading changes (max 5° per second for smooth turns)
                        MAX_HDG_CHANGE_PER_SEC = 5.0
                        hdg_diff = bearing - target_heading
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
                            print(f"  [NAV] Heading rate-limited: {hdg_diff:.0f}° change, now targeting {target_heading:.0f}° (WPT bearing {bearing:.0f}°)")
                        else:
                            target_heading = bearing

                        current_waypoint_name = current_wp.name

                        # Log first navigation update
                        if first_nav_update:
                            print(f"  [NAV] First waypoint: {current_wp.name} at {distance:.1f}nm, bearing {bearing:.0f}° (current HDG {current_hdg:.0f}°)")
                            first_nav_update = False

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
                                print(f"  [NAV] *** DESTINATION REACHED - HOLDING HEADING {current_hdg:.0f}° ***")
                                destination_reached = True
                                current_waypoint_name = "HOLD"
                                # Hold current heading, descend to field elevation
                                target_heading = current_hdg
                                target_altitude = 1000  # Descend to pattern altitude
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

                # Calculate bearing and distance to waypoint
                bearing = self.nav_calculator.calculate_bearing(
                    pos_lat, pos_lon,
                    current_wp.lat, current_wp.lon
                )
                distance = self.nav_calculator.calculate_distance_nm(
                    pos_lat, pos_lon,
                    current_wp.lat, current_wp.lon
                )

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

                    # Recalculate for new waypoint
                    current_wp = self.flight_plan.get_current_waypoint()
                    if current_wp:
                        bearing = self.nav_calculator.calculate_bearing(
                            pos_lat, pos_lon,
                            current_wp.lat, current_wp.lon
                        )
                        distance = self.nav_calculator.calculate_distance_nm(
                            pos_lat, pos_lon,
                            current_wp.lat, current_wp.lon
                        )
                        logger.info(f"Next waypoint: {current_wp.name} at {distance:.1f}nm, bearing {bearing:.0f}°")

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
                        # Now update to waypoint bearing
                        self.commander.set_heading_bug(bearing)
                    # else: keep flying the override heading, skip waypoint heading update
                else:
                    # Normal navigation - update heading if needed
                    heading_diff = abs(self.nav_calculator.heading_difference(pos_hdg, bearing))
                    if heading_diff > self.nav_config.heading_update_threshold_deg:
                        self.commander.set_heading_bug(bearing)

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
