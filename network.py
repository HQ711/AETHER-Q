"""
Relay graph at one instant: all satellites plus the source and destination HAPs. No other HAP is in
the graph, so a path never passes through another station. Every edge has its length [km] and the
cost -ln(efficiency); both are stored once per pair (row < column) in sparse matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import config as C
import physics as P


def latlon_rad(r: np.ndarray):
    return np.arcsin(r[..., 2] / np.linalg.norm(r, axis=-1)), np.arctan2(r[..., 1], r[..., 0])


def latlon_to_ecef(lat_deg, lon_deg, radius_km):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return radius_km * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def central_angle_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Central angles (A, B) between the point sets a (A, 3) and b (B, 3)."""
    pa, la = latlon_rad(a)
    pb, lb = latlon_rad(b)
    return P.central_angle(pa[:, None], la[:, None], pb[None, :], lb[None, :])


def hap_satellite_edges(hap: np.ndarray, sats: np.ndarray):
    """Satellites seen from one HAP above HAP_MIN_ELEVATION_DEG: (indices, lengths km)."""
    r_low = np.linalg.norm(hap)
    r = np.linalg.norm(sats, axis=1)
    gamma = central_angle_matrix(hap[None, :], sats)[0]
    ok = np.degrees(P.elevation(r_low, r, gamma)) >= C.HAP_MIN_ELEVATION_DEG
    return np.nonzero(ok)[0], P.slant_range(r_low, r, gamma)[ok]


def isl_edges(sats: np.ndarray, chunk: int = 1024):
    """All satellite pairs (i < j) with a line of sight above H_ATM_KM: (i, j, lengths km)."""
    n = len(sats)
    r = np.linalg.norm(sats, axis=1)
    I, J, Ls = [], [], []
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        gamma = central_angle_matrix(sats[rows], sats)
        vis = P.isl_visible(r[rows][:, None], r[None, :], gamma)
        vis &= np.arange(n)[None, :] > rows[:, None]
        L = P.slant_range(r[rows][:, None], r[None, :], gamma)
        if C.ISL_MAX_RANGE_KM is not None:
            vis &= L <= C.ISL_MAX_RANGE_KM
        a, b = np.nonzero(vis)
        I.append(rows[a]); J.append(b); Ls.append(L[a, b])
    return np.concatenate(I), np.concatenate(J), np.concatenate(Ls)


@dataclass
class RelayGraph:
    n_sats: int
    src_node: int                 # = n_sats (source HAP)
    dst_node: int                 # = n_sats + 1 (destination HAP)
    node_pos: np.ndarray          # (n_sats + 2, 3) km, Earth-fixed
    length: sp.csr_matrix         # edge length [km], upper triangle
    cost: sp.csr_matrix           # -ln(edge efficiency), upper triangle


def build_graph(sats: np.ndarray, hap_s: np.ndarray, hap_d: np.ndarray) -> RelayGraph:
    """Graph of the satellites sats (N, 3) and the two HAPs, all Earth-fixed [km]."""
    n = len(sats)
    i, j, L = isl_edges(sats)
    rows, cols, lens = [i], [j], [L]
    for k, hap in enumerate((hap_s, hap_d)):
        s_idx, Lh = hap_satellite_edges(hap, sats)
        rows.append(s_idx)
        cols.append(np.full(len(s_idx), n + k))
        lens.append(Lh)
    rows, cols, lens = np.concatenate(rows), np.concatenate(cols), np.concatenate(lens)
    cost = -np.log(P.edge_efficiency(lens))
    shape = (n + 2, n + 2)
    length = sp.csr_matrix((np.maximum(lens, 1e-9), (rows, cols)), shape=shape)
    costm = sp.csr_matrix((np.maximum(cost, 1e-12), (rows, cols)), shape=shape)
    return RelayGraph(n_sats=n, src_node=n, dst_node=n + 1, node_pos=np.vstack([sats, hap_s, hap_d]),
                      length=length, cost=costm)
