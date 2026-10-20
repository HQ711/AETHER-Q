"""
AETHER-Q settings. Units: km, s, degrees (beam optics in m).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Scenario (each can be overridden on the command line, see aetherq.py --help)
# ----------------------------------------------------------------------------
CONSTELLATION = "starlink"                # "starlink" or "oneweb"                    --const
SOURCE = "Washington DC"                  # a GROUND_STATIONS name or "lat,lon"         --src
DESTINATION = "London"                    #                                             --dst
START_UTC = None                          # e.g. "2026-10-19T00:00:00"; None = now      --start
F_MIN = 0.84                              # required end-to-end fidelity                --fmin

GROUND_STATIONS = {                       # name: (latitude deg, longitude deg, east positive)
    "Beijing": (39.9042, 116.4074),
    "Washington DC": (38.9072, -77.0369),
    "Hefei": (31.8206, 117.2272),
    "Shanghai": (31.2304, 121.4737),
    "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198),
    "Sydney": (-33.8688, 151.2093),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Vienna": (48.2082, 16.3738),
    "New York": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "Los Angeles": (34.0522, -118.2437),
    "Sao Paulo": (-23.5505, -46.6333),
    "Buenos Aires": (-34.6037, -58.3816),
    "Cape Town": (-33.9249, 18.4241),
}

# ----------------------------------------------------------------------------
# Physical constants (spherical Earth)
# ----------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
SPEED_OF_LIGHT_KM_S = 299792.458

# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
HAP_ALTITUDE_KM = 20.0                    # one HAP straight above each ground station
HAP_MIN_ELEVATION_DEG = 5.0               # HAP -> satellite links need at least this elevation
H_ATM_KM = 20.0                           # an inter-satellite link must stay above this altitude
ISL_MAX_RANGE_KM = None                   # optional extra inter-satellite range limit; None = geometry only

# ----------------------------------------------------------------------------
# Optics (passive lens relays) and atmosphere
# ----------------------------------------------------------------------------
WAVELENGTH_M = 810e-9
RX_RADIUS_M = 1.0                                   # receive lens radius rho
WAIST_M = 2.0 * RX_RADIUS_M / (1.22 * np.pi)        # transmit beam waist w0 (~0.522 m)
ETA_HW = 1.0 - 0.035                                # coupling and detection efficiency per lens set
ZENITH_TRANSMITTANCE = 0.5                          # atmospheric transmittance at zenith
HV_WIND_SPEED_M_S = 21.0                            # Hufnagel-Valley turbulence profile
HV_GROUND_CN2 = 1.7e-14

# ----------------------------------------------------------------------------
# Entanglement, memories and fidelity
# ----------------------------------------------------------------------------
F0_GH = 0.97                              # initial fidelity of ground-HAP pairs
F0_HH = 0.97                              # initial fidelity of HAP-HAP pairs
T_COH_S = 1.65                            # memory coherence time
DECAY_EXPONENT_K = 2.2                    # stretch exponent of the memory decay
O1 = 0.99                                 # one-qubit gate reliability
O2 = 0.99                                 # two-qubit gate reliability
ALPHA = 0.99                              # Bell-state measurement reliability
Q_GEN = 0.05                              # pair-generation probability per attempt
Q_SWP = 0.5                               # entanglement-swapping success probability
N_ATTEMPTS = 100                          # attempts per link

# ----------------------------------------------------------------------------
# Constellations: CelesTrak TLEs (cached in assets/), propagated with SGP4
# ----------------------------------------------------------------------------
CONSTELLATIONS = {
    "starlink": {"label": "Starlink", "group": "starlink", "min_altitude_km": 250.0},
    "oneweb": {"label": "OneWeb", "group": "oneweb", "min_altitude_km": 1000.0},
}
TLE_URL_TEMPLATE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
TLE_DIR = "assets"
TLE_MAX_AGE_HOURS = 12.0                  # re-download when older (only when the start is near now)
TLE_WARN_AGE_DAYS = 3.0                   # warn when the start is this far from the TLE epochs

# ----------------------------------------------------------------------------
# Requests: every REQUEST_PERIOD_S a seeded random draw decides whether SOURCE requests
# entanglement with DESTINATION; on a request the time freezes and the light path is shown
# ----------------------------------------------------------------------------
REQUESTS_ENABLED = False                  # R key in the 3D view (--headless always runs them)
REQUEST_PERIOD_S = 60.0
REQUEST_PROBABILITY = 0.5
RANDOM_SEED = 2026
RESULTS_DIR = "results"

# ----------------------------------------------------------------------------
# 3D view
# ----------------------------------------------------------------------------
TARGET_FPS = 120
TIME_SCALE = 60.0                                     # simulated seconds per real second    --speed
TIME_SCALE_STEPS = [1, 10, 30, 60, 120, 300, 600, 1200, 3000]
PROPAGATION_STEP_S = 20.0                             # SGP4 every 20 s, Hermite interpolation in between
PATH_FOLLOW = False                                   # P key: keep tracking the path after Enter

EARTH_TEXTURE_FILE = "assets/earth_bluemarble.jpg"
EARTH_TEXTURE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/world.200407.3x5400x2700.jpg"
WINDOW_SIZE = (1600, 1000)
BACKGROUND_COLOR = "#05070d"
EARTH_COLOR = (0.10, 0.35, 0.85, 1.0)                 # used when no texture is available
SHOW_GRID = False
GRID_COLOR = (0.75, 0.85, 1.0, 0.30)
SAT_SIZE_PX = {"starlink": 3, "oneweb": 5}
SAT_SHELLS = {                                        # satellite color by inclination (within 2 deg)
    "starlink": [
        (43.0, (1.00, 0.55, 0.20, 0.9)),
        (53.1, (0.35, 0.85, 1.00, 0.9)),
        (70.0, (0.60, 1.00, 0.45, 0.9)),
        (97.6, (1.00, 0.45, 0.85, 0.9)),
    ],
    "oneweb": [
        (87.9, (0.95, 0.85, 0.45, 0.9)),
    ],
}
SAT_OTHER_COLOR = (0.80, 0.80, 0.80, 0.9)
STATION_COLOR = (1.0, 0.85, 0.1, 1.0)
STATION_SIZE_PX = 18
HAP_COLOR = (0.2, 0.95, 1.0, 1.0)
HAP_SIZE_PX = 10

GS_MIN_ELEVATION_DEG = 20.0                           # GS -> satellite links (display only)
LINK_MODES = ["GS", "HAP", "GS+HAP"]                  # H key / on-screen button
LINK_MODE = "HAP"
SHOW_LINKS = True                                     # L key
SHOW_FOOTPRINTS = True                                # C key
SHOW_ISL = False                                      # I key
ISL_MAX_LINES_PER_REP = None                          # None = a line to every neighbor
LINK_COLOR = (1.0, 0.85, 0.1, 0.8)                    # GS -> satellite
HAP_LINK_COLOR = (0.2, 0.95, 1.0, 0.8)                # HAP -> satellite
ISL_COLOR = (1.0, 0.35, 1.0, 0.6)
ISL_REP_COLOR = (1.0, 1.0, 1.0, 1.0)
BUTTON_COLOR = (0.15, 0.2, 0.3, 0.9)
BUTTON_BORDER = (0.6, 0.75, 1.0, 1.0)

ROUTE_COLOR = (0.35, 1.0, 0.35, 0.95)                 # selected light path
ROUTE_WIDTH = 3.0
ROUTE_SAT_SIZE_PX = 10
SHOW_REFERENCE_PATHS = True                           # also draw the two Dijkstra paths the search starts from
MAX_ETA_PATH_COLOR = (1.0, 0.55, 0.10, 0.9)           # maximum-efficiency path
SHORTEST_PATH_COLOR = (0.45, 0.60, 1.00, 0.9)         # shortest path
REFERENCE_PATH_WIDTH = 2.0

HUD_FONT_SIZE = 11
HUD_FONT_FACES = ["Consolas", "Courier New", "DejaVu Sans Mono"]
HUD_LINE_SPACING = 1.8
HUD_STATUS_CHARS = 48
HUD_TEXT_COLOR = (1.0, 1.0, 1.0, 1.0)
HUD_DIM_COLOR = (0.65, 0.72, 0.85, 1.0)
HUD_ON_COLOR = (0.30, 1.00, 0.40, 1.0)
HUD_OFF_COLOR = (1.00, 0.35, 0.35, 1.0)
HUD_WARN_COLOR = (1.00, 0.75, 0.25, 1.0)
HUD_FRAME_COLOR = (0.45, 0.52, 0.65, 0.8)
HUD_LINK_BLOCK_LINES = 6
HUD_MAX_BLOCK_LINES = 40
HUD_SCROLL_LINES = 2
