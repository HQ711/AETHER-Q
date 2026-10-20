"""
Real satellites (Starlink or OneWeb) from CelesTrak TLEs, propagated with SGP4.

Frames: the "view" frame is TEME rotated by -GMST(t0), an inertial frame with x on the prime
meridian at the start. Earth-fixed = view rotated by -theta(t), theta(t) = GMST(t) - GMST(t0).
"""

from __future__ import annotations

import datetime as dt
import os
import time
import urllib.request

import numpy as np

import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
J2000_UTC = dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

try:
    from sgp4.api import Satrec, SatrecArray
    from sgp4.propagation import gstime
    HAVE_SGP4 = True
except ImportError:
    HAVE_SGP4 = False


# ---------------------------------------------------------------------- time
def parse_utc(value) -> dt.datetime:
    """ISO string (UTC if no zone), datetime, or None for now."""
    if value is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    t = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def julian_date(t: dt.datetime) -> float:
    return 2451545.0 + (t - J2000_UTC).total_seconds() / 86400.0


def gmst_rad(jd: float) -> float:
    if HAVE_SGP4:
        return gstime(jd)
    return np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def rotz(r: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vectors (..., 3) about z by +angle [rad]."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * r[..., 0] - s * r[..., 1], s * r[..., 0] + c * r[..., 1], r[..., 2]], -1)


# ---------------------------------------------------------------------- TLE files
def ensure_tle_file(key: str, allow_download: bool = True) -> str:
    """Path of the cached TLE file; downloaded when missing, or when stale and allow_download is True."""
    info = C.CONSTELLATIONS[key]
    url = C.TLE_URL_TEMPLATE.format(group=info["group"])
    path = os.path.join(HERE, C.TLE_DIR, f"{key}.tle")
    age_h = (time.time() - os.path.getmtime(path)) / 3600.0 if os.path.exists(path) else None
    if age_h is not None and (age_h < C.TLE_MAX_AGE_HOURS or not allow_download):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        print(f"Downloading {info['label']} TLEs: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "AETHER-Q"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode("ascii", errors="replace")
        if data.count("\n1 ") < 10:
            raise RuntimeError("response does not look like a TLE file: " + data[:120].replace("\n", " "))
        with open(path + ".part", "w", newline="\n") as f:
            f.write(data)
        os.replace(path + ".part", path)
    except Exception as e:
        if age_h is None:
            raise SystemExit(f"Could not download {info['label']} TLEs ({e}) and there is no cached file at {path}.")
        print(f"  download failed ({e}); using the cached file from {age_h:.1f} h ago")
    return path


def read_tles(path: str) -> list[tuple[str, str, str]]:
    lines = [ln.rstrip() for ln in open(path, encoding="ascii", errors="replace") if ln.strip()]
    out, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            out.append(("", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def tle_epoch_jd(line1: str) -> float:
    yy, doy = int(line1[18:20]), float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return julian_date(dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1.0))


def kepler_teme(tles, jd: float) -> np.ndarray:
    """Two-body positions (N, 3) km from the TLE mean elements (used only without sgp4)."""
    l2 = [t[2] for t in tles]
    inc = np.radians([float(s[8:16]) for s in l2])
    raan = np.radians([float(s[17:25]) for s in l2])
    ecc = np.array([float("0." + s[26:33].strip()) for s in l2])
    argp = np.radians([float(s[34:42]) for s in l2])
    m0 = np.radians([float(s[43:51]) for s in l2])
    n = np.array([float(s[52:63]) for s in l2]) * 2 * np.pi / 86400.0
    a = (C.MU_EARTH_KM3_S2 / n**2) ** (1.0 / 3.0)
    M = m0 + n * (jd - np.array([tle_epoch_jd(t[1]) for t in tles])) * 86400.0
    E = M.copy()
    for _ in range(8):
        E -= (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))
    xp, yp = a * (np.cos(E) - ecc), a * np.sqrt(1 - ecc**2) * np.sin(E)
    cO, sO, cw, sw, ci, si = np.cos(raan), np.sin(raan), np.cos(argp), np.sin(argp), np.cos(inc), np.sin(inc)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], -1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], -1)
    return xp[:, None] * P + yp[:, None] * Q


# ---------------------------------------------------------------------- satellites
class SatelliteMotion:
    """All satellites of one constellation, propagated from start_utc (simulation time t = 0)."""

    def __init__(self, key: str, start_utc: dt.datetime, step_s: float = C.PROPAGATION_STEP_S):
        self.key = key
        self.label = C.CONSTELLATIONS[key]["label"]
        near_now = abs((dt.datetime.now(dt.timezone.utc) - start_utc).total_seconds()) < 86400
        tles = read_tles(ensure_tle_file(key, allow_download=near_now))
        self.start_utc = start_utc
        self.jd0 = julian_date(start_utc)
        self.gmst0 = gmst_rad(self.jd0)
        self.step = step_s
        self.backend = "SGP4" if HAVE_SGP4 else "Kepler (pip install sgp4 for SGP4)"

        self._set(tles)
        r0, _ = self._teme(np.array([0.0]))
        alt0 = np.linalg.norm(r0[:, 0], axis=1) - C.EARTH_RADIUS_KM
        keep = np.isfinite(alt0) & (alt0 >= C.CONSTELLATIONS[key]["min_altitude_km"])
        self._set([t for t, k in zip(tles, keep) if k])

        self.names = [t[0] or t[1][2:7].strip() for t in self.tles]
        self.inclination_deg = np.array([float(t[2][8:16]) for t in self.tles])
        age = np.abs(self.jd0 - np.array([tle_epoch_jd(t[1]) for t in self.tles]))
        if np.median(age) > C.TLE_WARN_AGE_DAYS:
            print(f"WARNING: the start is {np.median(age):.1f} days from the TLE epochs; positions may be "
                  f"off by tens of km or more.")
        self._cache_ta = None

    @property
    def num_sats(self) -> int:
        return len(self.tles)

    def _set(self, tles):
        self.tles = tles
        if HAVE_SGP4:
            self._array = SatrecArray([Satrec.twoline2rv(l1, l2) for _, l1, l2 in tles])

    def _teme(self, t: np.ndarray):
        """TEME positions and velocities (N, T, 3) at simulation times t [s]; NaN where SGP4 fails."""
        t = np.asarray(t, dtype=float)
        if HAVE_SGP4:
            day = np.floor(self.jd0)
            err, r, v = self._array.sgp4(np.full(len(t), day), (self.jd0 - day) + t / 86400.0)
            r[err != 0] = np.nan
            v[err != 0] = np.nan
            return r, v
        r = np.stack([kepler_teme(self.tles, self.jd0 + ti / 86400.0) for ti in t], axis=1)
        r1 = np.stack([kepler_teme(self.tles, self.jd0 + (ti + 1.0) / 86400.0) for ti in t], axis=1)
        return r, r1 - r

    def earth_angle(self, t: float) -> float:
        """How far the Earth has turned in the view frame since the start [rad]."""
        return gmst_rad(self.jd0 + t / 86400.0) - self.gmst0

    def view_positions(self, t: float) -> np.ndarray:
        """View-frame positions (N, 3) km for the animation (cubic Hermite between SGP4 samples)."""
        if self._cache_ta is None or not (self._cache_ta <= t <= self._cache_ta + self.step):
            ta = np.floor(t / self.step) * self.step
            r, v = self._teme(np.array([ta, ta + self.step]))
            r, v = rotz(r, -self.gmst0), rotz(v, -self.gmst0)
            self._cache_ta, self._ra, self._rb, self._va, self._vb = ta, r[:, 0], r[:, 1], v[:, 0], v[:, 1]
        s, h = (t - self._cache_ta) / self.step, self.step
        return ((2 * s**3 - 3 * s**2 + 1) * self._ra + (s**3 - 2 * s**2 + s) * h * self._va
                + (-2 * s**3 + 3 * s**2) * self._rb + (s**3 - s**2) * h * self._vb)

    def ecef(self, t: float) -> np.ndarray:
        """Earth-fixed positions (N, 3) km at simulation time t, directly from SGP4 (used for routing)."""
        r, _ = self._teme(np.array([t]))
        return rotz(r[:, 0], -gmst_rad(self.jd0 + t / 86400.0))

    def utc(self, t: float) -> dt.datetime:
        return self.start_utc + dt.timedelta(seconds=float(t))
