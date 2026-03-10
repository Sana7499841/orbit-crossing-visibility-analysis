"""
Q1 — Space-based tracker crossing + visibility analysis for a TLE object
Chosen sensor for visibility: SPACE-BASED TRACKER
Objective:
In the next 24 hours after the tracker epoch, determine:
(1) The crossing events: the moments when the object moves through the camera’s FOV (geometry only). 
(2) The visible/detectable events: crossing AND the object is sunlit AND the distance is within 1000 km. 
Model: 
- The target object: propagated with SGP4 using the given TLE (output in TEME).
- The space tracker: propagated with a two-body Keplerian model
- The camera: 30° full cone angle (half-angle = 15°), boresight points along the direction of the tracker’s velocity. 
- Sunlit: simple cylindrical Earth shadow model (no penumbra). 
Method of running: 
python -m pip install numpy sgp4 
python q1_space_tracker_crossings.py 

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from sgp4.api import Satrec, jday


# -----------------------------
# Constants / configuration
# -----------------------------
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.1363

FOV_DEG = 30.0
HALF_ANGLE_RAD = np.deg2rad(FOV_DEG / 2.0)

MAX_RANGE_KM = 1000.0

DT_SECONDS = 10.0          # sampling step (sec)
DURATION_HOURS = 24.0      # analysis duration


# -----------------------------
# Time / math helpers
# -----------------------------
def datetime_to_jd(dt_utc: datetime) -> Tuple[float, float]:
    """Convert timezone-aware UTC datetime to (jd, fr) for SGP4."""
    if dt_utc.tzinfo is None or dt_utc.tzinfo.utcoffset(dt_utc) is None:
        raise ValueError("Input datetime must be timezone-aware UTC.")
    return jday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute,
        dt_utc.second + dt_utc.microsecond * 1e-6
    )


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v * 0.0
    return v / n


# -----------------------------
# Sun vector + eclipse (low precision)
# -----------------------------
def sun_vector_eci_km(dt_utc: datetime) -> np.ndarray:
    """
    Approximate Earth->Sun vector (km) in an ECI-like frame.
    Sufficient for eclipse gating in this assignment context.
    """
    jd, fr = datetime_to_jd(dt_utc)
    T = ((jd + fr) - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Mean longitude and anomaly (deg)
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T**2) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T**2) % 360.0
    M_rad = np.deg2rad(M)

    # Ecliptic longitude (deg)
    lam = (L0
           + 1.914602 * np.sin(M_rad)
           + 0.019993 * np.sin(2 * M_rad)
           + 0.000289 * np.sin(3 * M_rad)) % 360.0
    lam_rad = np.deg2rad(lam)

    # Obliquity (deg)
    eps = 23.439291 - 0.0130042 * T
    eps_rad = np.deg2rad(eps)

    # Sun distance (AU -> km)
    r_au = 1.00014 - 0.01671 * np.cos(M_rad) - 0.00014 * np.cos(2 * M_rad)
    AU_KM = 149597870.7
    r_km = r_au * AU_KM

    # Ecliptic -> equatorial
    x = r_km * np.cos(lam_rad)
    y = r_km * np.cos(eps_rad) * np.sin(lam_rad)
    z = r_km * np.sin(eps_rad) * np.sin(lam_rad)
    return np.array([x, y, z], dtype=float)


def is_sunlit_cylindrical(r_obj_km: np.ndarray, dt_utc: datetime) -> bool:
    """
    Cylindrical Earth shadow test:
    - Object is eclipsed if it is behind Earth relative to Sun AND
      within Earth's radius cylinder around the Sun-line.
    """
    r_sun = sun_vector_eci_km(dt_utc)
    s_hat = unit(r_sun)

    proj = float(np.dot(r_obj_km, s_hat))  # scalar projection onto Sun direction
    if proj > 0.0:
        return True  # object on the Sun-facing side of Earth

    r_perp = r_obj_km - proj * s_hat
    d_perp = float(np.linalg.norm(r_perp))

    # In shadow if within Earth's radius cylinder
    return d_perp > R_EARTH_KM


# -----------------------------
# Space-based tracker propagation (2-body Kepler)
# -----------------------------
@dataclass(frozen=True)
class KeplerElements:
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    M0_deg: float

def kepler_to_rv_eci_km(el: KeplerElements, epoch: datetime, t: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-body propagation from 'epoch' to time 't'.
    For e=0: mean anomaly = true anomaly.

    Returns:
      r_eci (km), v_eci (km/s)
    """
    a = el.a_km
    e = el.e

    i = np.deg2rad(el.i_deg)
    raan = np.deg2rad(el.raan_deg)
    argp = np.deg2rad(el.argp_deg)
    M0 = np.deg2rad(el.M0_deg)

    dt_sec = (t - epoch).total_seconds()
    n = np.sqrt(MU_EARTH_KM3_S2 / (a**3))  # rad/s
    M = (M0 + n * dt_sec) % (2 * np.pi)

    # Perifocal position/velocity
    if abs(e) < 1e-12:
        nu = M
        r_pf = np.array([a * np.cos(nu), a * np.sin(nu), 0.0], dtype=float)
        v_pf = np.array([-a * n * np.sin(nu), a * n * np.cos(nu), 0.0], dtype=float)
    else:
        # Not required for this case; included for completeness.
        E = M
        for _ in range(30):
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            E -= f / fp
        nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0),
                              np.sqrt(1 - e) * np.cos(E / 2.0))
        p = a * (1.0 - e**2)
        r = p / (1.0 + e * np.cos(nu))
        r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0], dtype=float)
        v_pf = np.sqrt(MU_EARTH_KM3_S2 / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0], dtype=float)

    # Rotation PQW -> ECI: R3(RAAN) * R1(i) * R3(argp)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R3_O = np.array([[cO, -sO, 0.0],
                     [sO,  cO, 0.0],
                     [0.0, 0.0, 1.0]])

    R1_i = np.array([[1.0, 0.0,  0.0],
                     [0.0,  ci, -si],
                     [0.0,  si,  ci]])

    R3_w = np.array([[cw, -sw, 0.0],
                     [sw,  cw, 0.0],
                     [0.0, 0.0, 1.0]])

    Q = R3_O @ R1_i @ R3_w

    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci


# -----------------------------
# Interval extraction
# -----------------------------
def boolean_intervals(times: List[datetime], mask: np.ndarray) -> List[Tuple[datetime, datetime]]:
    """
    Convert boolean mask into [start, end) intervals using discrete sampling.
    - start: first sample where False->True
    - end: first sample where True->False
    Note: time precision ~ ±DT_SECONDS
    """
    intervals: List[Tuple[datetime, datetime]] = []
    in_event = False
    start = None

    for k in range(len(mask)):
        if mask[k] and not in_event:
            in_event = True
            start = times[k]
        elif (not mask[k]) and in_event:
            intervals.append((start, times[k]))
            in_event = False
            start = None

    if in_event and start is not None:
        intervals.append((start, times[-1]))

    return intervals


def print_intervals(title: str, intervals: List[Tuple[datetime, datetime]]) -> None:
    print("\n" + title)
    if not intervals:
        print("  None")
        return
    for i, (ts, te) in enumerate(intervals, 1):
        print(f"  {i:02d}. Start: {ts.isoformat()}   End: {te.isoformat()}")


# -----------------------------
# Note:
# The target state from SGP4 is returned in TEME. The tracker is propagated # with a simplified two-body inertial model and treated as being in the 
# same inertial-like frame
# for this short-duration screening analysis. Small frame transformation 
# differences are neglected.
# Main
# -----------------------------
def main() -> None:
    # Inputs from prompt
    tle_l1 = "1 63223U 25052P 25244.59601767 .00010814 00000-0 51235-3 0 9991"
    tle_l2 = "2 63223 97.4217 137.0451 0006365 74.2830 285.9107 15.19475170 25990"

    tracker_epoch = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)

    tracker_el = KeplerElements(
        a_km=6878.0,
        e=0.0,
        i_deg=97.4,
        raan_deg=72.628,
        argp_deg=331.7425,
        M0_deg=0.0,
    )

    # Setup propagator for target object
    sat = Satrec.twoline2rv(tle_l1, tle_l2)

    # Time grid
    duration = timedelta(hours=DURATION_HOURS)
    n_steps = int(duration.total_seconds() / DT_SECONDS) + 1
    times = [tracker_epoch + timedelta(seconds=k * DT_SECONDS) for k in range(n_steps)]

    crossing = np.zeros(n_steps, dtype=bool)
    visible = np.zeros(n_steps, dtype=bool)

    for k, t in enumerate(times):
        jd, fr = datetime_to_jd(t)

        # Propagate object with SGP4 (TEME)
        err, r_obj_km, v_obj_km_s = sat.sgp4(jd, fr)
        if err != 0:
            continue
        r_obj = np.array(r_obj_km, dtype=float)

        # Propagate tracker with 2-body model
        r_trk, v_trk = kepler_to_rv_eci_km(tracker_el, tracker_epoch, t)

        # LOS from tracker to object
        rho = r_obj - r_trk
        rng = float(np.linalg.norm(rho))
        if rng < 1e-12:
            continue

        # Boresight = tracker velocity direction
        b_hat = unit(v_trk)

        # Angle between LOS and boresight
        cosang = float(np.dot(b_hat, rho / rng))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang = np.arccos(cosang)

        # Crossing condition
        in_fov = ang <= HALF_ANGLE_RAD
        crossing[k] = in_fov

        # Visibility condition: crossing + range gate + sunlit
        if in_fov and (rng <= MAX_RANGE_KM):
            sunlit = is_sunlit_cylindrical(r_obj, t)
            visible[k] = bool(sunlit)

    # Convert boolean time series to intervals
    crossing_intervals = boolean_intervals(times, crossing)
    visible_intervals = boolean_intervals(times, visible)

    # Print summary
    print("===== Q1: SPACE-BASED TRACKER ANALYSIS =====")
    print(f"Epoch (UTC): {tracker_epoch.isoformat()}")
    print(f"Window: 24 hours | Δt = {DT_SECONDS} s | Samples = {len(times)}")
    print(f"FOV: {FOV_DEG:.1f} deg (half-angle = {np.rad2deg(HALF_ANGLE_RAD):.1f} deg)")
    print(f"Visibility: crossing AND sunlit AND range < {MAX_RANGE_KM:.0f} km")

    print_intervals("Crossing events (geometry only):", crossing_intervals)
    print_intervals("Visible/Detectable events:", visible_intervals)

    print("\nDone.")


if __name__ == "__main__":
    main()
