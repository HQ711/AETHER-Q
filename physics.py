"""
Link, fidelity and rate model of AETHER-Q.
"""

from functools import lru_cache

import numpy as np

import config as C


# ---------------------------------------------------------------------- geometry
def central_angle(phi_a, lam_a, phi_b, lam_b):
    """Central angle [rad] between two points given by latitude / longitude [rad]."""
    cos_gamma = np.sin(phi_a) * np.sin(phi_b) + np.cos(phi_a) * np.cos(phi_b) * np.cos(lam_b - lam_a)
    return np.arccos(np.clip(cos_gamma, -1.0, 1.0))


def slant_range(r_low, r, gamma):
    """Distance [km] between points at radii r_low and r separated by the central angle gamma."""
    return np.sqrt(r_low**2 + r**2 - 2.0 * r_low * r * np.cos(gamma))


def elevation(r_low, r, gamma):
    """Elevation [rad] of the upper point above the local horizon of the lower point."""
    return np.arctan2(np.cos(gamma) - r_low / r, np.sin(gamma))


def isl_visible(r1, r2, gamma, h_atm=C.H_ATM_KM):
    """True where the line between two satellites stays above altitude h_atm."""
    ra = C.EARTH_RADIUS_KM + h_atm
    return gamma <= np.arccos(np.clip(ra / r1, -1, 1)) + np.arccos(np.clip(ra / r2, -1, 1))


# ---------------------------------------------------------------------- free-space hops
def beam_radius_m(z_m, w0=C.WAIST_M, wavelength=C.WAVELENGTH_M):
    """Gaussian beam radius [m] at distance z [m] from the waist."""
    z_r = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1.0 + (z_m / z_r) ** 2)


def collected_fraction(w_m, rho=C.RX_RADIUS_M):
    """Fraction of a Gaussian beam of radius w collected by a lens of radius rho."""
    return 1.0 - np.exp(-2.0 * rho**2 / np.asarray(w_m) ** 2)


def edge_efficiency(L_km, eta_hw=C.ETA_HW):
    """Efficiency of a free-space hop of length L [km] above the atmosphere."""
    return eta_hw * collected_fraction(beam_radius_m(np.asarray(L_km) * 1e3))


# ---------------------------------------------------------------------- ground-HAP links
def eta_atm(el_rad, t_zen=C.ZENITH_TRANSMITTANCE):
    """Atmospheric transmittance along a slant path at elevation el [rad]."""
    return t_zen ** (1.0 / np.sin(el_rad))


def cn2_hufnagel_valley(h_m, v=C.HV_WIND_SPEED_M_S, a=C.HV_GROUND_CN2):
    """Refractive-index structure constant Cn^2 [m^(-2/3)] at altitude h [m]."""
    h = np.asarray(h_m, dtype=float)
    return (0.00594 * (v / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
            + 2.7e-16 * np.exp(-h / 1500.0) + a * np.exp(-h / 100.0))


def fried_parameter_uplink_m(z_m, el_rad, wavelength=C.WAVELENGTH_M, n_steps=20001):
    """Uplink Fried parameter r0 [m] over a slant path of length z [m] at elevation el."""
    k = 2.0 * np.pi / wavelength
    zp = np.linspace(0.0, z_m, n_steps)
    f = cn2_hufnagel_valley(zp * np.sin(el_rad)) * ((z_m - zp) / z_m) ** (5.0 / 3.0)
    integral = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(zp)))
    return (0.42 * k**2 * integral) ** (-3.0 / 5.0)


def long_term_radius_uplink_m(z_m, el_rad, w0=C.WAIST_M, wavelength=C.WAVELENGTH_M):
    """Uplink long-term beam radius [m]: diffraction plus turbulence broadening."""
    k = 2.0 * np.pi / wavelength
    r0 = fried_parameter_uplink_m(z_m, el_rad, wavelength)
    return np.sqrt(beam_radius_m(z_m, w0, wavelength) ** 2 + 2.0 * (4.0 * z_m / (k * r0)) ** 2)


def ground_hap_geometry():
    """(length [km], elevation [rad]) of the link between a station and the HAP straight above it."""
    r = C.EARTH_RADIUS_KM + C.HAP_ALTITUDE_KM
    return float(slant_range(C.EARTH_RADIUS_KM, r, 0.0)), float(elevation(C.EARTH_RADIUS_KM, r, 0.0))


@lru_cache(maxsize=None)
def uplink_efficiency():
    """Efficiency of the uplink from the source station to its HAP."""
    L, el = ground_hap_geometry()
    return float(C.ETA_HW * eta_atm(el) * collected_fraction(long_term_radius_uplink_m(L * 1e3, el)))


@lru_cache(maxsize=None)
def downlink_efficiency():
    """Efficiency of the downlink from the destination HAP to its station (diffraction only)."""
    L, el = ground_hap_geometry()
    return float(C.ETA_HW * eta_atm(el) * collected_fraction(beam_radius_m(L * 1e3)))


# ---------------------------------------------------------------------- fidelity
def werner_decay(F0, tau_s):
    """Fidelity of a Werner pair whose qubit has been stored for tau [s]."""
    return (F0 - 0.25) * np.exp(-(np.asarray(tau_s) / C.T_COH_S) ** C.DECAY_EXPONENT_K) + 0.25


def swap_fidelity(F1, F2):
    """Fidelity after swapping two Werner pairs with imperfect gates and measurement."""
    s = C.O1 * C.O2 * (4 * C.ALPHA**2 - 1) / 3
    return 0.25 * (1 + 3 * s * (4 * F1 - 1) / 3 * (4 * F2 - 1) / 3)


def e2e_fidelity(tau_s):
    """End-to-end fidelity after the swaps at the two HAPs, with every qubit stored for tau [s]."""
    f_gh, f_hh = werner_decay(C.F0_GH, tau_s), werner_decay(C.F0_HH, tau_s)
    return swap_fidelity(f_gh, swap_fidelity(f_gh, f_hh))


def round_trip_s(L_km):
    """Heralding round trip 2 L / c [s] of a HAP-HAP segment of length L [km]."""
    return 2.0 * np.asarray(L_km) / C.SPEED_OF_LIGHT_KM_S


def max_wait_time(f_min, tau_hi=3600.0):
    """Longest storage time [s] with e2e_fidelity >= f_min (0 if f_min is unreachable)."""
    if e2e_fidelity(0.0) < f_min:
        return 0.0
    if e2e_fidelity(tau_hi) >= f_min:
        return tau_hi
    lo, hi = 0.0, tau_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if e2e_fidelity(mid) >= f_min else (lo, mid)
    return lo


def length_limit_km(f_min):
    """Longest HAP-HAP segment [km] that still delivers fidelity >= f_min."""
    return C.SPEED_OF_LIGHT_KM_S * max_wait_time(f_min) / 2.0


# ---------------------------------------------------------------------- success probability and rate
def link_success(eta):
    """Probability that a link with per-photon efficiency eta succeeds within N_ATTEMPTS attempts."""
    return -np.expm1(C.N_ATTEMPTS * np.log1p(-C.Q_GEN * np.asarray(eta)))


def success_probability(eta_p):
    """End-to-end success probability per round for a HAP-HAP segment of efficiency eta_p."""
    q_ls = link_success(uplink_efficiency())
    q_ld = link_success(downlink_efficiency())
    return float(C.Q_SWP**2 * q_ls * link_success(eta_p) * q_ld)


def edr(eta_p, L_km):
    """Entanglement distribution rate [ebits/s]: one round per heralding round trip."""
    return float(success_probability(eta_p) / round_trip_s(L_km))
