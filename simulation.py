"""
Requests, light-path computation and logging, without graphics.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

import config as C
import physics as P
from constellation import HERE, SatelliteMotion
from network import build_graph, latlon_rad, latlon_to_ecef
from routing import select_light_path


# ---------------------------------------------------------------------- stations
@dataclass
class Station:
    name: str
    lat_deg: float
    lon_deg: float

    def ecef(self, altitude_km: float = 0.0) -> np.ndarray:
        return latlon_to_ecef(self.lat_deg, self.lon_deg, C.EARTH_RADIUS_KM + altitude_km)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_station(text: str) -> Station:
    """A GROUND_STATIONS entry (case, spaces and a unique prefix are enough: "washington") or "lat,lon"."""
    m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d*)?)\s*,\s*([-+]?\d+(?:\.\d*)?)\s*", text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return Station(f"({lat:g}, {lon:g})", lat, lon)
    names = list(C.GROUND_STATIONS)
    hits = [n for n in names if _key(n) == _key(text)] or [n for n in names if _key(n).startswith(_key(text))]
    if len(hits) != 1 or not _key(text):
        raise SystemExit(f"unknown station {text!r}; use \"lat,lon\" or one of: {', '.join(names)}")
    return Station(hits[0], *C.GROUND_STATIONS[hits[0]])


# ---------------------------------------------------------------------- requests
class RequestSchedule:
    """Decision k is taken at simulation time time_of(k), the k-th full period after the start."""

    def __init__(self, start_utc: dt.datetime, period_s=C.REQUEST_PERIOD_S,
                 probability=C.REQUEST_PROBABILITY, seed=C.RANDOM_SEED):
        epoch = start_utc.timestamp()
        self.t_first = np.floor(epoch / period_s) * period_s + period_s - epoch
        self.period = period_s
        self.probability = probability
        self._rng = np.random.default_rng(seed)
        self._decisions: list[bool] = []

    def time_of(self, k: int) -> float:
        return self.t_first + k * self.period

    def decision(self, k: int) -> bool:
        while len(self._decisions) <= k:
            self._decisions.append(bool(self._rng.random() < self.probability))
        return self._decisions[k]

    def next_request(self, k_from: int, limit: int = 100000) -> int:
        for k in range(k_from, k_from + limit):
            if self.decision(k):
                return k
        raise RuntimeError("no request found; is REQUEST_PROBABILITY 0?")


# ---------------------------------------------------------------------- light path
@dataclass
class RouteResult:
    t_sim: float
    time_utc: dt.datetime
    source: str
    destination: str
    status: str                      # served / infeasible / no path / F_min unreachable
    f_min: float
    length_limit_km: float
    gs_ecef: np.ndarray              # (2, 3) km: source, destination station
    hap_ecef: np.ndarray             # (2, 3) km: their HAPs
    sat_indices: list = field(default_factory=list)
    sat_names: list = field(default_factory=list)
    node_ecef: np.ndarray = None     # (K + 2, 3) km: source HAP, satellites, destination HAP
    length_km: float = np.nan
    eta_p: float = 0.0
    pi: float = 0.0                  # end-to-end success probability per round
    f_e2e: float = np.nan            # delivered fidelity
    edr: float = 0.0                 # [ebits/s]
    selected: str = ""
    n_candidates: int = 0
    note: str = ""
    refs: dict = field(default_factory=dict)   # name -> {node_ecef, length_km, eta_p, f_e2e, edr, status, n_sats}
    compute_s: float = 0.0

    @property
    def served(self) -> bool:
        return self.status == "served"


def compute_route(motion: SatelliteMotion, t: float, src: Station, dst: Station, f_min: float) -> RouteResult:
    """Light path between src and dst at simulation time t."""
    wall0 = time.perf_counter()
    gs = np.array([src.ecef(), dst.ecef()])
    haps = np.array([src.ecef(C.HAP_ALTITUDE_KM), dst.ecef(C.HAP_ALTITUDE_KM)])
    l_max = P.length_limit_km(f_min)
    res = RouteResult(t_sim=t, time_utc=motion.utc(t), source=src.name, destination=dst.name, status="",
                      f_min=f_min, length_limit_km=l_max, gs_ecef=gs, hap_ecef=haps)
    if l_max <= 0:
        res.status, res.note = "F_min unreachable", f"F_min {f_min:g} is above the best possible fidelity"
        res.compute_s = time.perf_counter() - wall0
        return res

    graph = build_graph(motion.ecef(t), haps[0], haps[1])
    route = select_light_path(graph, l_max)
    res.status, res.note = route.status, route.note
    if C.SHOW_REFERENCE_PATHS:
        for name, ref in route.refs.items():
            res.refs[name] = {"node_ecef": graph.node_pos[ref.nodes], "length_km": ref.length_km,
                              "eta_p": ref.eta, "f_e2e": float(P.e2e_fidelity(P.round_trip_s(ref.length_km))),
                              "edr": P.edr(ref.eta, ref.length_km), "status": ref.status, "n_sats": ref.n_sats}
    if res.served:
        res.sat_indices = [int(i) for i in route.nodes[1:-1]]
        res.sat_names = [motion.names[i] for i in res.sat_indices]
        res.node_ecef = graph.node_pos[route.nodes]
        res.length_km, res.eta_p = route.length_km, route.eta
        res.pi = P.success_probability(route.eta)
        res.f_e2e = float(P.e2e_fidelity(P.round_trip_s(route.length_km)))
        res.edr = P.edr(route.eta, route.length_km)
        res.selected, res.n_candidates = route.selected, route.n_candidates
    res.compute_s = time.perf_counter() - wall0
    return res


def _pair_gamma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, la = latlon_rad(a)
    pb, lb = latlon_rad(b)
    return P.central_angle(pa, la, pb, lb)


def path_validity(hap_s: np.ndarray, sats: np.ndarray, hap_d: np.ndarray, l_max_km: float) -> dict:
    """Is the fixed path hap_s -> sats -> hap_d (Earth-fixed, km) still usable at this instant?"""
    pts = np.vstack([hap_s, sats, hap_d])
    L = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    el_up = np.degrees(P.elevation(np.linalg.norm(hap_s), np.linalg.norm(sats[0]),
                                   _pair_gamma(hap_s[None], sats[:1])[0]))
    el_dn = np.degrees(P.elevation(np.linalg.norm(hap_d), np.linalg.norm(sats[-1]),
                                   _pair_gamma(hap_d[None], sats[-1:])[0]))
    isl_ok = True
    if len(sats) > 1:
        g = _pair_gamma(sats[:-1], sats[1:])
        isl_ok = bool(np.all(P.isl_visible(np.linalg.norm(sats[:-1], axis=1), np.linalg.norm(sats[1:], axis=1), g)))
    reasons = []
    if not el_up >= C.HAP_MIN_ELEVATION_DEG:
        reasons.append(f"first satellite at {el_up:.1f} deg, below {C.HAP_MIN_ELEVATION_DEG:.0f} deg")
    if not el_dn >= C.HAP_MIN_ELEVATION_DEG:
        reasons.append(f"last satellite at {el_dn:.1f} deg, below {C.HAP_MIN_ELEVATION_DEG:.0f} deg")
    if not isl_ok:
        reasons.append("inter-satellite line of sight lost")
    if not L.sum() <= l_max_km:
        reasons.append(f"path {L.sum():,.0f} km, longer than {l_max_km:,.0f} km")
    return {"valid": not reasons, "reasons": reasons, "eta_p": float(np.prod(P.edge_efficiency(L))),
            "length_km": float(L.sum()), "el_up_deg": float(el_up), "el_down_deg": float(el_dn)}


# ---------------------------------------------------------------------- log
class RequestLog:
    FIELDS = ["utc", "request", "status", "n_satellites", "length_km", "eta_p", "fidelity", "pi", "edr_ebits_s",
              "selected", "compute_s", "max_eta_length_km", "max_eta_status", "shortest_length_km",
              "shortest_status", "satellites"]

    def __init__(self, key: str, src: Station, dst: Station, start_utc: dt.datetime, tag: str = ""):
        folder = os.path.join(HERE, C.RESULTS_DIR)
        os.makedirs(folder, exist_ok=True)
        pair = "_".join(re.sub(r"[^A-Za-z0-9.,-]", "", s.name).replace(",", "_") for s in (src, dst))
        self.path = os.path.join(folder, f"requests{tag}_{key}_{pair}_{start_utc:%Y%m%dT%H%M%S}.csv")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)
        self.n_checks = self.n_requests = self.n_served = 0

    def no_request(self, when: dt.datetime):
        self.n_checks += 1
        self._write([f"{when:%Y-%m-%d %H:%M:%S}", 0] + [""] * (len(self.FIELDS) - 2))

    def route(self, r: RouteResult):
        self.n_checks += 1
        self.n_requests += 1
        self.n_served += int(r.served)
        ok = r.served
        refs = []
        for name in ("max_eta", "shortest"):
            ref = r.refs.get(name)
            refs += [f"{ref['length_km']:.1f}", ref["status"]] if ref else ["", ""]
        self._write([f"{r.time_utc:%Y-%m-%d %H:%M:%S}", 1, r.status, len(r.sat_indices),
                     f"{r.length_km:.1f}" if ok else "", f"{r.eta_p:.6g}" if ok else "",
                     f"{r.f_e2e:.6f}" if ok else "", f"{r.pi:.6g}", f"{r.edr:.6g}", r.selected,
                     f"{r.compute_s:.2f}"] + refs + [" ".join(r.sat_names)])

    def _write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def describe(r: RouteResult) -> str:
    head = f"{r.time_utc:%Y-%m-%d %H:%M:%S} UTC  {r.source} -> {r.destination}: "
    if not r.served:
        return head + f"{r.status} ({r.note})"
    return head + (f"{len(r.sat_indices)} satellites, {r.length_km:,.0f} km, eta {r.eta_p:.3f}, "
                   f"fidelity {r.f_e2e:.4f}, EDR {r.edr:.2f} ebits/s ({r.selected}, {r.compute_s:.1f} s)")


def run_headless(motion: SatelliteMotion, src: Station, dst: Station, f_min: float, hours: float):
    """Run the request schedule for `hours` without a window and log every decision."""
    sched = RequestSchedule(motion.start_utc)
    log = RequestLog(motion.key, src, dst, motion.start_utc, tag="_headless")
    k = 0
    while sched.time_of(k) <= hours * 3600.0:
        t = sched.time_of(k)
        if sched.decision(k):
            r = compute_route(motion, t, src, dst, f_min)
            log.route(r)
            print("  " + describe(r))
        else:
            log.no_request(motion.utc(t))
        k += 1
    print(f"{log.n_requests} requests, {log.n_served} served -> {log.path}")
    return log
