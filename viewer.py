"""
Real-time 3D view of AETHER-Q (VisPy + OpenGL).

    Earth        NASA Blue Marble texture, optional grid
    Dots         real Starlink or OneWeb satellites, colored by orbital shell
    Stars        source and destination stations; diamonds: their HAPs (straight above)
    Green line   selected light path GS -> HAP -> satellites -> HAP -> GS
    Orange/blue  maximum-efficiency / shortest Dijkstra paths (only where they differ from green)
    Yellow/cyan  GS / HAP -> satellite links, reach circles
    Magenta      inter-satellite links of the highest satellite above each station

Every REQUEST_PERIOD_S a seeded random draw decides whether the source requests entanglement. On a
request the time freezes, the light path is computed and drawn, and Enter resumes.
The on-screen panel lists all keys.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
import urllib.request

import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import MatrixTransform

try:
    from vispy.visuals.filters import TextureFilter
except ImportError:
    TextureFilter = None

import config as C
import physics as P
import simulation as S
from constellation import HERE, rotz
from network import latlon_rad

R_E = C.EARTH_RADIUS_KM


# ---------------------------------------------------------------------- Earth helpers
def load_texture(max_width: int = 4096):
    path = os.path.join(HERE, C.EARTH_TEXTURE_FILE)
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Downloading Earth texture: {C.EARTH_TEXTURE_URL}")
            urllib.request.urlretrieve(C.EARTH_TEXTURE_URL, path + ".part")
            os.replace(path + ".part", path)
        except Exception as e:
            print(f"  texture download failed ({e}); using a solid Earth")
            return None
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; using a solid Earth")
        return None
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, max_width // 2), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def uv_sphere(rows=90, cols=180):
    lat, lon = np.linspace(-90, 90, rows + 1), np.linspace(-180, 180, cols + 1)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    la, lo = np.radians(LA), np.radians(LO)
    verts = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1).reshape(-1, 3)
    tex = np.stack([(LO + 180) / 360, (90 - LA) / 180], -1).reshape(-1, 2)
    idx = np.arange((rows + 1) * (cols + 1)).reshape(rows + 1, cols + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return verts.astype(np.float32), faces.astype(np.uint32), tex.astype(np.float32)


def graticule(radius=1.003, step=30, n=181):
    lines = []
    t = np.radians(np.linspace(-180, 180, n))
    for lat in range(-90 + step, 90, step):
        la = np.radians(lat)
        lines.append(np.stack([np.cos(la) * np.cos(t), np.cos(la) * np.sin(t), np.full_like(t, np.sin(la))], 1))
    p = np.radians(np.linspace(-90, 90, n))
    for lon in range(-180, 180, step):
        lo = np.radians(lon)
        lines.append(np.stack([np.cos(p) * np.cos(lo), np.cos(p) * np.sin(lo), np.sin(p)], 1))
    return radius * np.concatenate([np.repeat(l, 2, axis=0)[1:-1] for l in lines]).astype(np.float32)


def gamma_to(points_km: np.ndarray, sat_ll) -> np.ndarray:
    """Central angles (K, N) from K points to N satellites whose (lat, lon) [rad] are sat_ll."""
    pa, la = latlon_rad(points_km)
    return P.central_angle(pa[:, None], la[:, None], sat_ll[0][None, :], sat_ll[1][None, :])


def elevation_deg(lows_km: np.ndarray, sats_km: np.ndarray, sat_ll, sat_r) -> np.ndarray:
    """Elevation (G, N) [deg] of N satellites seen from G lower ends (GS or HAP), same frame."""
    gamma = gamma_to(lows_km, sat_ll)
    return np.degrees(P.elevation(np.linalg.norm(lows_km, axis=1)[:, None], sat_r[None, :], gamma))


def isl_rows(sats_km: np.ndarray, rows, sat_ll, sat_r) -> tuple[np.ndarray, np.ndarray]:
    """
    ISL visibility (K, N) from the satellites `rows` to all satellites (line of sight
    above R_E + h_atm, optional ISL_MAX_RANGE_KM), and the lengths L (K, N) [km]. Never to itself;
    satellites with NaN positions are never visible.
    """
    rows = np.asarray(rows, dtype=int)
    r1, r2 = sat_r[rows][:, None], sat_r[None, :]
    gamma = gamma_to(sats_km[rows], sat_ll)
    vis = P.isl_visible(r1, r2, gamma)
    L = P.slant_range(r1, r2, gamma)
    if C.ISL_MAX_RANGE_KM is not None:
        vis &= L <= C.ISL_MAX_RANGE_KM
    vis &= np.isfinite(L)
    vis[np.arange(len(rows)), rows] = False
    return vis, L


def reach_half_angle(r_low_km: float, r_sat_km: float, min_el_deg: float) -> float:
    """Earth-central half-angle [rad] within which a satellite at r_sat is seen at El >= min_el
    from a point at r_low."""
    e = np.radians(min_el_deg)
    return float(np.arccos(r_low_km * np.cos(e) / r_sat_km) - e)


def footprint_ring(center_ecef: np.ndarray, half_angle: float, radius: float = 1.004, n: int = 120) -> np.ndarray:
    """Circle of Earth-central half-angle `half_angle` around `center_ecef`, as line segments (Earth radii)."""
    up = center_ecef / np.linalg.norm(center_ecef)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    th = np.linspace(0, 2 * np.pi, n + 1)
    ring = radius * (np.cos(half_angle) * up
                     + np.sin(half_angle) * (np.cos(th)[:, None] * east + np.sin(th)[:, None] * north))
    return np.repeat(ring, 2, axis=0)[1:-1].astype(np.float32)


def hms(seconds: float) -> str:
    s = max(int(round(seconds)), 0)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------- viewer
def shell_colors(motion) -> np.ndarray:
    """Satellite colors by orbital shell (nearest inclination within 2 deg)."""
    shells = C.SAT_SHELLS.get(motion.key, [])
    colors = np.array([s[1] for s in shells] + [C.SAT_OTHER_COLOR], dtype=np.float32)
    if not shells:
        return np.repeat(colors, motion.num_sats, axis=0)
    inc = np.array([s[0] for s in shells])
    d = np.abs(motion.inclination_deg[:, None] - inc[None, :])
    return colors[np.where(d.min(axis=1) <= 2.0, d.argmin(axis=1), len(shells))]


class Viewer:
    def __init__(self, motion, schedule: S.RequestSchedule, log: S.RequestLog, src: S.Station, dst: S.Station,
                 f_min: float, time_scale: float = C.TIME_SCALE):
        self.motion, self.schedule, self.log = motion, schedule, log
        self.src, self.dst, self.f_min = src, dst, f_min
        self.time_scale = time_scale
        self.sim_t = 0.0
        self.k_next = 0                       # index of the next request decision
        self.requests_on = C.REQUESTS_ENABLED # R key
        self.mode = "RUNNING"                 # RUNNING / PAUSED / COMPUTING / SHOWING
        self.follow_enabled = C.PATH_FOLLOW
        self.follow = None                    # RouteResult being tracked after Enter (path follow)
        self.follow_info = ""
        self.result = None                    # last RouteResult
        self.message = ""                     # last "no request" note
        self._executor = cf.ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._compute_t0 = 0.0
        self._last_wall = time.perf_counter()
        self._fps_n, self._fps_t0, self.fps = 0, time.perf_counter(), 0.0
        self._text_colors = {}
        self.show_links = C.SHOW_LINKS
        self.link_mode = C.LINK_MODE
        self.show_footprints = C.SHOW_FOOTPRINTS
        self.show_isl = C.SHOW_ISL
        self.station_names = [src.name, dst.name]
        n_st = len(self.station_names)
        self.gs_counts = [0] * n_st             # satellites in view of each GS / of its HAP
        self.hap_counts = [0] * n_st
        self.isl_counts = [0] * n_st            # ISL neighbors of each representative satellite
        self._count_t = {}                      # sim time at which each kind of count was computed

        self.canvas = scene.SceneCanvas(keys="interactive", size=C.WINDOW_SIZE, bgcolor=C.BACKGROUND_COLOR,
                                        title=f"AETHER-Q - {src.name} -> {dst.name}", show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=40, distance=4.5, elevation=25, azimuth=40)

        # Earth (rotating node: texture, grid, stations, HAPs)
        self.earth_node = scene.Node(parent=self.view.scene)
        self.earth_tf = MatrixTransform()
        self.earth_node.transform = self.earth_tf
        self._make_earth()
        self.grid = scene.visuals.Line(pos=graticule(), connect="segments", color=C.GRID_COLOR, parent=self.earth_node)
        self.grid.visible = C.SHOW_GRID
        self.gs_ecef_km = np.array([src.ecef(), dst.ecef()])
        self.hap_ecef_km = self.gs_ecef_km * (R_E + C.HAP_ALTITUDE_KM) / R_E     # same as the routing
        gs = self.gs_ecef_km / R_E
        stars = scene.visuals.Markers(parent=self.earth_node)
        stars.set_data((gs * 1.012).astype(np.float32), symbol="star", size=C.STATION_SIZE_PX,
                       face_color=C.STATION_COLOR, edge_color=(0, 0, 0, 1), edge_width=1)
        haps = scene.visuals.Markers(parent=self.earth_node)
        haps.set_data((gs * 1.016).astype(np.float32), symbol="diamond", size=C.HAP_SIZE_PX,
                      face_color=C.HAP_COLOR, edge_color=(0, 0, 0, 1), edge_width=1)
        for name, p in zip(self.station_names, gs):
            lab = scene.visuals.Text(name, pos=p * 1.05, color=C.STATION_COLOR, font_size=9,
                                     anchor_x="left", anchor_y="center", parent=self.earth_node)
            lab.set_gl_state("translucent", depth_test=True)

        # Reach circles (Earth-fixed), for a satellite at the median constellation altitude
        r_sat = float(np.nanmedian(np.linalg.norm(motion.view_positions(0.0), axis=1)))
        self.median_alt_km = r_sat - R_E
        self.gs_half_angle = reach_half_angle(R_E, r_sat, C.GS_MIN_ELEVATION_DEG)
        self.hap_half_angle = reach_half_angle(R_E + C.HAP_ALTITUDE_KM, r_sat, C.HAP_MIN_ELEVATION_DEG)
        self.fp_gs = scene.visuals.Line(pos=np.concatenate([footprint_ring(p, self.gs_half_angle) for p in gs]),
                                        connect="segments", color=C.LINK_COLOR, width=2, parent=self.earth_node)
        self.fp_hap = scene.visuals.Line(pos=np.concatenate([footprint_ring(p, self.hap_half_angle) for p in gs]),
                                         connect="segments", color=C.HAP_LINK_COLOR, width=2, parent=self.earth_node)

        # Satellites, visibility overlays and the light path (drawn last, on top)
        self.sat_colors = shell_colors(motion)
        self.sat_size = C.SAT_SIZE_PX.get(motion.key, 3)
        self.sats = scene.visuals.Markers(parent=self.view.scene)
        self.gs_links = scene.visuals.Line(pos=np.zeros((2, 3), np.float32), connect="segments",
                                           color=C.LINK_COLOR, width=1.5, parent=self.view.scene)
        self.hap_links = scene.visuals.Line(pos=np.zeros((2, 3), np.float32), connect="segments",
                                            color=C.HAP_LINK_COLOR, width=1.5, parent=self.view.scene)
        self.isl_lines = scene.visuals.Line(pos=np.zeros((2, 3), np.float32), connect="segments",
                                            color=C.ISL_COLOR, width=1.5, parent=self.view.scene)
        self.isl_neighbors = scene.visuals.Markers(parent=self.view.scene)
        self.isl_rep_marks = scene.visuals.Markers(parent=self.view.scene)
        self.ref_visuals = {}                   # the two Dijkstra paths, drawn below the route
        for name, color in (("max_eta", C.MAX_ETA_PATH_COLOR), ("shortest", C.SHORTEST_PATH_COLOR)):
            self.ref_visuals[name] = (
                scene.visuals.Line(pos=np.zeros((2, 3), np.float32), color=color, width=C.REFERENCE_PATH_WIDTH,
                                   parent=self.view.scene),
                scene.visuals.Markers(parent=self.view.scene), color)
        self.route_line = scene.visuals.Line(pos=np.zeros((2, 3), np.float32), color=C.ROUTE_COLOR,
                                             width=C.ROUTE_WIDTH, parent=self.view.scene)
        self.route_sats = scene.visuals.Markers(parent=self.view.scene)
        self._hide_route()

        self._hud_metrics()
        self._build_controls()
        self._build_status()
        self.canvas.events.resize.connect(lambda ev: self._layout_status())
        self.canvas.events.key_press.connect(self.on_key)
        self.canvas.events.mouse_press.connect(self.on_mouse)
        self._shown_result = None
        for ev in (self.canvas.events.mouse_move, self.canvas.events.mouse_press):
            ev.connect(self.on_hover)                       # runs before the camera (VisPy connects first)
        self.canvas.events.mouse_wheel.connect(self.on_wheel)
        self._apply_link_mode()
        self._update_scene()
        self._update_hud()
        self.timer = app.Timer(interval=1.0 / C.TARGET_FPS, connect=self.on_timer, start=True)

    # ---------------------------------------------------------------- scene
    def _make_earth(self):
        img = load_texture() if TextureFilter is not None else None
        if img is not None:
            v, f, t = uv_sphere()
            mesh = scene.visuals.Mesh(vertices=v, faces=f, color=(1, 1, 1, 1), shading=None, parent=self.earth_node)
            mesh.attach(TextureFilter(img, t))
        else:
            scene.visuals.Sphere(radius=1.0, rows=64, cols=128, method="latitude",
                                 color=C.EARTH_COLOR, parent=self.earth_node)

    def _update_scene(self):
        pos = self.motion.view_positions(self.sim_t) / R_E
        self.sats.set_data(np.nan_to_num(pos).astype(np.float32), face_color=self.sat_colors,
                           edge_width=0, size=self.sat_size)
        theta = self.motion.earth_angle(self.sim_t)
        self.earth_tf.reset()
        self.earth_tf.rotate(np.degrees(theta) % 360.0, (0, 0, 1))
        self._update_overlays(pos * R_E, theta)
        if self.follow is not None:
            self._update_follow(pos * R_E)

    def _visibility(self, sats_km, theta, kinds):
        """
        Visibility in the view frame (km) for kinds among "gs", "hap", "isl"; also refreshes the
        corresponding counts of the SATELLITES IN VIEW table. Only what is asked for is computed.
        """
        out = {"st_km": rotz(self.gs_ecef_km, theta), "hap_km": rotz(self.hap_ecef_km, theta)}
        if not kinds:
            return out
        with np.errstate(invalid="ignore"):
            sat_ll, sat_r = latlon_rad(sats_km), np.linalg.norm(sats_km, axis=1)
            if "gs" in kinds or "isl" in kinds:
                el_gs = elevation_deg(out["st_km"], sats_km, sat_ll, sat_r)
                out["gs_vis"] = el_gs >= C.GS_MIN_ELEVATION_DEG
                self.gs_counts = out["gs_vis"].sum(axis=1).tolist()
                self._count_t["gs"] = self.sim_t
            if "hap" in kinds:
                out["hap_vis"] = elevation_deg(out["hap_km"], sats_km, sat_ll, sat_r) >= C.HAP_MIN_ELEVATION_DEG
                self.hap_counts = out["hap_vis"].sum(axis=1).tolist()
                self._count_t["hap"] = self.sim_t
            if "isl" in kinds:
                out["reps"] = np.nan_to_num(el_gs, nan=-1e9).argmax(axis=1)   # highest satellite above each GS
                out["isl_vis"], out["isl_L"] = isl_rows(sats_km, out["reps"], sat_ll, sat_r)
                self.isl_counts = out["isl_vis"].sum(axis=1).tolist()
                self._count_t["isl"] = self.sim_t
        return out

    def _refresh_counts(self):
        """Table counts that were not computed for the current frame (their overlay is hidden)."""
        missing = [k for k in ("gs", "hap", "isl") if self._count_t.get(k) != self.sim_t]
        if missing:
            self._visibility(self.motion.view_positions(self.sim_t), self.motion.earth_angle(self.sim_t), missing)

    def _update_overlays(self, sats_km, theta):
        """GS / HAP -> satellite links and representative ISLs: only the visible ones are computed."""
        parts = self.link_mode.split("+")
        show_gs = self.show_links and "GS" in parts
        show_hap = self.show_links and "HAP" in parts
        kinds = [k for k, on in (("gs", show_gs), ("hap", show_hap), ("isl", self.show_isl)) if on]
        g = self._visibility(sats_km, theta, kinds)
        self._set_links(self.gs_links, g["st_km"], g.get("gs_vis"), C.LINK_COLOR, show_gs, sats_km)
        self._set_links(self.hap_links, g["hap_km"], g.get("hap_vis"), C.HAP_LINK_COLOR, show_hap, sats_km)
        if not self.show_isl:
            for v in (self.isl_lines, self.isl_neighbors, self.isl_rep_marks):
                v.visible = False
            return
        reps, vis, L = g["reps"], g["isl_vis"], g["isl_L"]
        seg, nbr = [], []
        for k, rep in enumerate(reps):
            idx = np.nonzero(vis[k])[0]
            near = idx[np.argsort(L[k, idx])]                                  # nearest first
            if C.ISL_MAX_LINES_PER_REP is not None:
                near = near[:C.ISL_MAX_LINES_PER_REP]
            nbr.append(near)
            pairs = np.empty((2 * len(near), 3))
            pairs[0::2], pairs[1::2] = sats_km[rep], sats_km[near]
            seg.append(pairs)
        seg, nbr = np.concatenate(seg), np.unique(np.concatenate(nbr))
        if len(seg):
            self.isl_lines.set_data(pos=(seg / R_E).astype(np.float32), connect="segments", color=C.ISL_COLOR)
        self.isl_lines.visible = len(seg) > 0
        if len(nbr):
            self.isl_neighbors.set_data((sats_km[nbr] / R_E).astype(np.float32), face_color=C.ISL_COLOR,
                                        edge_width=0, size=self.sat_size + 3)
        self.isl_neighbors.visible = len(nbr) > 0
        self.isl_rep_marks.set_data((sats_km[reps] / R_E).astype(np.float32), face_color=(0, 0, 0, 0),
                                    edge_color=C.ISL_REP_COLOR, edge_width=2, size=self.sat_size + 12)
        self.isl_rep_marks.visible = True

    def _set_links(self, visual, origins_km, vis, color, enabled, sats_km):
        if not enabled or vis is None or not vis.any():
            visual.visible = False
            return
        o_idx, s_idx = np.nonzero(vis)
        seg = np.empty((2 * len(o_idx), 3), np.float32)
        seg[0::2] = origins_km[o_idx] / R_E
        seg[1::2] = sats_km[s_idx] / R_E
        visual.set_data(pos=seg, connect="segments", color=color)
        visual.visible = True

    def _apply_link_mode(self):
        """Reach circles and the button caption follow the link mode (the HAP markers stay: routing uses them)."""
        parts = self.link_mode.split("+")
        self.fp_gs.visible = self.show_footprints and "GS" in parts
        self.fp_hap.visible = self.show_footprints and "HAP" in parts
        self._set_text(self.btn_text, self._button_caption(self.link_mode), self._mode_color())

    def _cycle_link_mode(self):
        modes = C.LINK_MODES
        self.link_mode = modes[(modes.index(self.link_mode) + 1) % len(modes)]
        self._apply_link_mode()
        return f"link mode {self.link_mode}"

    def on_mouse(self, event):
        if event.pos is None or event.button != 1:
            return
        x, y = event.pos[:2]
        x0, y0, x1, y1 = self.btn_rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            print(f"[button] {self._cycle_link_mode()}")
            self._update_scene()
            self._update_hud()
            self.canvas.update()

    def _hide_route(self):
        self.route_line.visible = False
        self.route_sats.visible = False
        self._hide_refs()

    def _hide_refs(self):
        for line, marks, _ in self.ref_visuals.values():
            line.visible = marks.visible = False

    @staticmethod
    def _same_as_route(r, ref):
        return (r.served and r.node_ecef is not None and ref["node_ecef"].shape == r.node_ecef.shape
                and np.allclose(ref["node_ecef"], r.node_ecef))

    def _draw_refs(self, r, theta):
        """The two Dijkstra paths of a request (only where they differ from the selected path)."""
        self._hide_refs()
        for name, ref in r.refs.items():
            if name not in self.ref_visuals or self._same_as_route(r, ref):
                continue
            line, marks, color = self.ref_visuals[name]
            pts = np.vstack([r.gs_ecef[0], ref["node_ecef"], r.gs_ecef[1]])
            view = (rotz(pts, theta) / R_E).astype(np.float32)
            line.set_data(pos=view, color=color, width=C.REFERENCE_PATH_WIDTH)
            marks.set_data(view[2:-2], face_color=color, edge_width=0, size=C.ROUTE_SAT_SIZE_PX - 3)
            line.visible = marks.visible = True

    def _draw_route(self, gs_ecef, hap_s, sats_ecef, hap_d, theta):
        """Draw GS -> HAP -> satellites -> HAP -> GS; Earth-fixed inputs are turned by theta into the view."""
        pts = np.vstack([gs_ecef[0], hap_s, sats_ecef, hap_d, gs_ecef[1]])
        view = rotz(pts, theta) / R_E
        self.route_line.set_data(pos=view.astype(np.float32), color=C.ROUTE_COLOR, width=C.ROUTE_WIDTH)
        self.route_sats.set_data(view[2:-2].astype(np.float32), face_color=C.ROUTE_COLOR, edge_color=(0, 0, 0, 1),
                                 edge_width=1, size=C.ROUTE_SAT_SIZE_PX)
        self.route_line.visible = self.route_sats.visible = True

    def _update_follow(self, view_km):
        r = self.follow
        theta = self.motion.earth_angle(self.sim_t)
        sats = rotz(view_km[r.sat_indices], -theta)                     # Earth-fixed now
        v = S.path_validity(r.hap_ecef[0], sats, r.hap_ecef[1], r.length_limit_km)
        dt_s = self.sim_t - r.t_sim
        if v["valid"]:
            self._draw_route(r.gs_ecef, r.hap_ecef[0], sats, r.hap_ecef[1], theta)
            self.follow_info = (f"follow +{hms(dt_s)}: eta {v['eta_p']:.3f}, "
                                f"elevation {v['el_up_deg']:.0f} / {v['el_down_deg']:.0f} deg")
        else:
            self.follow = None
            self._hide_route()
            self.follow_info = f"path broke at +{hms(dt_s)}: {v['reasons'][0]}"
            print(f"  path of {r.time_utc:%H:%M:%S} UTC broke at +{dt_s:.0f} s: {'; '.join(v['reasons'])}")

    # ---------------------------------------------------------------- time and requests
    def on_timer(self, event):
        now = time.perf_counter()
        dt_wall, self._last_wall = now - self._last_wall, now
        if self.mode == "RUNNING":
            t_new = self.sim_t + dt_wall * self.time_scale
            while not self.requests_on and self.schedule.time_of(self.k_next) <= t_new:
                self.k_next += 1                                      # requests off: no decision
            while self.schedule.time_of(self.k_next) <= t_new:
                k = self.k_next
                tk = self.schedule.time_of(k)
                self.k_next += 1
                if self.schedule.decision(k):
                    self.sim_t = tk                                   # freeze exactly at the request
                    self._start_compute()
                    break
                self.log.no_request(self.motion.utc(tk))
                self.message = f"{self.motion.utc(tk):%H:%M} UTC: no request"
            else:
                self.sim_t = t_new
        elif self.mode == "COMPUTING" and self._future is not None and self._future.done():
            self._finish_compute()

        self._update_scene()
        self._fps_n += 1
        if now - self._fps_t0 >= 0.5:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_n, self._fps_t0 = 0, now
            self._update_hud()
        self.canvas.update()

    def _start_compute(self):
        self.mode = "COMPUTING"
        self.follow = None
        self.follow_info = ""
        self._hide_route()
        self._compute_t0 = time.perf_counter()
        # Fill the animation cache at the frozen instant first, so that while the model is frozen the
        # main thread never propagates at the same time as the routing thread
        self.motion.view_positions(self.sim_t)
        print(f"{self.motion.utc(self.sim_t):%Y-%m-%d %H:%M:%S} UTC: request {self.src.name} -> {self.dst.name}; "
              f"model frozen, computing the light path ...")
        self._future = self._executor.submit(S.compute_route, self.motion, self.sim_t,
                                             self.src, self.dst, self.f_min)
        self._update_hud()

    def _finish_compute(self):
        try:
            r = self._future.result()
        except Exception as e:                                        # keep the viewer alive
            print(f"  route computation failed: {e!r}")
            self.result, self.mode = None, "SHOWING"
            self.message = f"route computation failed: {type(e).__name__}"
            self._future = None
            self._update_hud()
            return
        self._future = None
        self.result = r
        self.log.route(r)
        print("  " + S.describe(r))
        self._draw_refs(r, self.motion.earth_angle(r.t_sim))
        if r.served:
            self._draw_route(r.gs_ecef, r.hap_ecef[0], r.node_ecef[1:-1], r.hap_ecef[1],
                             self.motion.earth_angle(r.t_sim))
        self.mode = "SHOWING"
        self._update_hud()

    def _resume(self):
        r = self.result
        self._hide_refs()                                             # references belong to the frozen instant
        if self.follow_enabled and r is not None and r.served:
            self.follow = r
        else:
            self._hide_route()
        self.mode = "RUNNING"
        self._last_wall = time.perf_counter()

    def _jump_to_next_request(self):
        k = self.schedule.next_request(self.k_next)
        self.follow = None
        self._hide_route()
        self.sim_t = max(self.sim_t, self.schedule.time_of(k) - 1e-6)   # skipped checks are logged on the way
        self.mode = "RUNNING"
        self._last_wall = time.perf_counter()

    # ---------------------------------------------------------------- keys
    def on_key(self, event):
        name = ((event.key.name if event.key is not None else "") or "").upper()
        text = (event.text or "").lower()
        msg = None
        if name in ("ENTER", "RETURN") or text in ("\r", "\n"):
            if self.mode == "SHOWING":
                self._resume()
                msg = "resumed"
        elif name == "R" or text == "r":
            self.requests_on = not self.requests_on
            if self.requests_on:
                t1 = self.schedule.time_of(self.k_next)
                msg = f"requests ON, first check {self.motion.utc(t1):%Y-%m-%d %H:%M} UTC"
                self.message = f"requests ON, first check {self.motion.utc(t1):%H:%M} UTC"
            else:
                busy = self.mode in ("COMPUTING", "SHOWING")
                msg = "requests OFF" + (" (the current request is still completed)" if busy else "")
                self.message = "requests OFF: satellites only"
        elif name == "N" or text == "n":
            if not self.requests_on:
                msg = "N ignored: requests are OFF (press R first)"
                self.message = "requests OFF: press R before N"
            elif self.mode in ("RUNNING", "PAUSED", "SHOWING"):
                self._jump_to_next_request()
                msg = f"jump to the request at {self.motion.utc(self.schedule.time_of(self.schedule.next_request(self.k_next))):%Y-%m-%d %H:%M} UTC"
        elif name == "SPACE":
            if self.mode in ("RUNNING", "PAUSED"):
                self.mode = "PAUSED" if self.mode == "RUNNING" else "RUNNING"
                self._last_wall = time.perf_counter()
                msg = self.mode.lower()
        elif name in ("+", "=", "PLUS", "EQUAL", "UP") or text in ("+", "="):
            bigger = [v for v in C.TIME_SCALE_STEPS if v > self.time_scale]
            self.time_scale = bigger[0] if bigger else self.time_scale
            msg = f"speed x{self.time_scale:g}"
        elif name in ("-", "_", "MINUS", "DOWN") or text in ("-", "_"):
            smaller = [v for v in C.TIME_SCALE_STEPS if v < self.time_scale]
            self.time_scale = smaller[-1] if smaller else self.time_scale
            msg = f"speed x{self.time_scale:g}"
        elif name == "P" or text == "p":
            self.follow_enabled = not self.follow_enabled
            if not self.follow_enabled and self.follow is not None:
                self.follow = None
                self._hide_route()
            msg = f"path follow {'ON' if self.follow_enabled else 'OFF'}"
        elif name == "G" or text == "g":
            self.grid.visible = not self.grid.visible
            msg = f"grid {'ON' if self.grid.visible else 'OFF'}"
        elif name == "L" or text == "l":
            self.show_links = not self.show_links
            msg = f"satellite links {'ON' if self.show_links else 'OFF'}"
        elif name == "H" or text == "h":
            msg = self._cycle_link_mode()
        elif name == "C" or text == "c":
            self.show_footprints = not self.show_footprints
            self._apply_link_mode()
            msg = f"reach circles {'ON' if self.show_footprints else 'OFF'}"
        elif name == "I" or text == "i":
            self.show_isl = not self.show_isl
            msg = f"inter-satellite links {'ON' if self.show_isl else 'OFF'}"
        if msg:
            print(f"[key] {msg}")
        self._update_hud()
        self.canvas.update()

    # ---------------------------------------------------------------- HUD
    def _hud_metrics(self):
        face, mono = "OpenSans", False
        try:
            from vispy.util.fonts import list_fonts
            available = {f.lower(): f for f in list_fonts()}
            for want in C.HUD_FONT_FACES:
                if want.lower() in available:
                    face, mono = available[want.lower()], True
                    break
        except Exception:
            pass
        dpi = float(getattr(self.canvas, "dpi", 0) or 96.0)
        self._em = C.HUD_FONT_SIZE * dpi / 72.0
        self._face = face
        self._cw = self._em * (0.60 if mono else 0.70)
        self._lh = self._em * C.HUD_LINE_SPACING
        self._margin = 1.2 * self._em

    def _text(self, pos, color=C.HUD_TEXT_COLOR, anchor_x="left", text=""):
        return scene.visuals.Text(text, parent=self.canvas.scene, color=color, font_size=C.HUD_FONT_SIZE,
                                  face=self._face, anchor_x=anchor_x, anchor_y="center", pos=pos)

    def _set_text(self, visual, text, color=None):
        if visual.text != text:
            visual.text = text
        if color is not None:
            key, col = id(visual), tuple(color)
            if self._text_colors.get(key) != col:
                visual.color = color
                self._text_colors[key] = col

    @staticmethod
    def _onoff(flag):
        return ("ON", C.HUD_ON_COLOR) if flag else ("OFF", C.HUD_OFF_COLOR)

    def _mode_color(self):
        return {"GS": C.LINK_COLOR, "HAP": C.HAP_LINK_COLOR}.get(self.link_mode, C.HUD_TEXT_COLOR)

    @staticmethod
    def _button_caption(mode):
        return f"Click: link mode {mode}"

    def _build_controls(self):
        """Top-left: key hints with the current state of every switch (green ON / red OFF)."""
        x0, y0 = self._margin, self._margin + self._lh / 2
        run = {"RUNNING": ("RUNNING", C.HUD_ON_COLOR), "PAUSED": ("PAUSED", C.HUD_OFF_COLOR)}
        rows = [                                                  # (label, color, state function or None)
            ("[R]     Requests", C.HUD_TEXT_COLOR, lambda: self._onoff(self.requests_on)),
            ("[Enter] Resume after a route", C.HUD_TEXT_COLOR, None),
            ("[N]     Jump to next request", C.HUD_TEXT_COLOR, None),
            ("[Space] Pause / run", C.HUD_TEXT_COLOR, lambda: run.get(self.mode, ("FROZEN", C.HUD_WARN_COLOR))),
            ("[+/-]   Speed", C.HUD_TEXT_COLOR, lambda: (f"x{self.time_scale:g}", C.HUD_TEXT_COLOR)),
            ("[P]     Path follow", C.HUD_TEXT_COLOR, lambda: self._onoff(self.follow_enabled)),
            ("[G]     Grid", C.HUD_TEXT_COLOR, lambda: self._onoff(self.grid.visible)),
            ("[L]     Satellite links", C.HUD_TEXT_COLOR, lambda: self._onoff(self.show_links)),
            ("[H]     Link mode", C.HUD_TEXT_COLOR, lambda: (self.link_mode, self._mode_color())),
            ("[C]     Reach circles", C.HUD_TEXT_COLOR, lambda: self._onoff(self.show_footprints)),
            ("[I]     Inter-sat links", C.HUD_TEXT_COLOR, lambda: self._onoff(self.show_isl)),
            ("Mouse: drag = rotate, wheel = zoom", C.HUD_DIM_COLOR, None),
            ("[F11] Full screen   [Esc] Quit", C.HUD_DIM_COLOR, None),
        ]
        x_state = x0 + (max(len(lbl) for lbl, _, fn in rows if fn is not None) + 3) * self._cw
        self._text((x0, y0), C.HUD_DIM_COLOR, text="CONTROLS")
        self.ctrl_rows = []
        for i, (label, color, fn) in enumerate(rows):
            y = y0 + (i + 1) * self._lh
            self._text((x0, y), color, text=label)
            if fn is not None:
                self.ctrl_rows.append((self._text((x_state, y)), fn))

        # Clickable link-mode button below the controls, sized for its longest caption
        longest = max(len(self._button_caption(m)) for m in C.LINK_MODES)
        bw, bh = (longest + 4) * self._cw, 1.8 * self._em
        by0 = y0 + (len(rows) + 0.8) * self._lh
        self.btn_rect = (x0, by0, x0 + bw, by0 + bh)                          # x0, y0, x1, y1
        scene.visuals.Rectangle(center=(x0 + bw / 2, by0 + bh / 2), width=bw, height=bh,
                                color=C.BUTTON_COLOR, border_color=C.BUTTON_BORDER, parent=self.canvas.scene)
        self.btn_text = self._text((x0 + bw / 2, by0 + bh / 2), anchor_x="center")

    def _build_status(self):
        """
        Right panel: fixed STATUS lines, then two framed blocks that scroll with the mouse wheel while
        the cursor is over them: SATELLITE LINKS (per-station table) and SOLUTION (last request).
        """
        self.st_top = [self._text((0, 0)) for _ in range(5)]
        self._name_chars = max(len("Station"), *(len(n) for n in self.station_names))
        self.blocks = {}
        for key, title, n_cols in (("links", "SATELLITE LINKS", 4), ("solution", "SOLUTION", 1)):
            anchors = ["left"] + ["right"] * (n_cols - 1)
            self.blocks[key] = {
                "title": self._text((0, 0), C.HUD_DIM_COLOR, text=title),
                "hint": self._text((0, 0), C.HUD_DIM_COLOR, anchor_x="right"),
                "frame": scene.visuals.Rectangle(center=(1, 1), width=1, height=1, color=(0, 0, 0, 0),
                                                 border_color=C.HUD_FRAME_COLOR, parent=self.canvas.scene),
                "thumb": scene.visuals.Rectangle(center=(1, 1), width=1, height=1, color=C.HUD_DIM_COLOR,
                                                 parent=self.canvas.scene),
                "slots": [[self._text((0, 0), anchor_x=anc) for anc in anchors] for _ in range(C.HUD_MAX_BLOCK_LINES)],
                "lines": [], "offset": 0, "n": 1, "rect": (0, 0, 0, 0)}
        self._layout_status()

    def _layout_status(self):
        """Place the right panel against the right edge (again whenever the window is resized)."""
        cw, lh, W, H = self._cw, self._lh, self.canvas.size[0], self.canvas.size[1]
        x0 = W - self._margin - C.HUD_STATUS_CHARS * cw
        x1 = W - self._margin
        pad = 0.6 * self._em
        self.panel_rect = (x0 - pad, 0, W, H)
        y = self._margin + lh / 2
        for t in self.st_top:
            t.pos = (x0, y)
            y += lh
        y += 0.5 * lh
        rows_links = C.HUD_LINK_BLOCK_LINES
        for key in ("links", "solution"):
            blk = self.blocks[key]
            blk["title"].pos = (x0, y)
            blk["hint"].pos = (x1, y)
            top = y + 0.6 * lh
            if key == "links":
                n = rows_links
            else:                                               # the solution block fills the rest of the height
                n = int((H - self._margin - top - 2 * pad) // lh)
            n = int(np.clip(n, 3, C.HUD_MAX_BLOCK_LINES))
            height = n * lh + 2 * pad
            blk["n"], blk["rect"] = n, (x0 - pad, top, x1 + pad, top + height)
            blk["frame"].center = (x0 - pad + (x1 - x0 + 2 * pad) / 2, top + height / 2)
            blk["frame"].width, blk["frame"].height = x1 - x0 + 2 * pad, height
            for i, slot in enumerate(blk["slots"]):
                yy = top + pad + (i + 0.5) * lh
                xs = [x0] if len(slot) == 1 else self._link_columns(x0)
                for t, x in zip(slot, xs):
                    t.pos = (x, yy)
                    t.visible = i < n
            y = top + height + 0.9 * lh
        self._render_blocks()

    def _link_columns(self, x0):
        x_gs = x0 + (self._name_chars + 6) * self._cw                     # right edges of the count columns
        return [x0, x_gs, x_gs + 7 * self._cw, x_gs + 15 * self._cw]

    def _render_blocks(self):
        """Show each block's lines from its scroll offset, and its scrollbar when the content overflows."""
        width = C.HUD_STATUS_CHARS - 2
        for key, blk in self.blocks.items():
            lines, n = blk["lines"], blk["n"]
            blk["offset"] = int(np.clip(blk["offset"], 0, max(0, len(lines) - n)))
            off = blk["offset"]
            for i, slot in enumerate(blk["slots"][:n]):
                item = lines[off + i] if off + i < len(lines) else ("", None)
                if len(slot) == 1:
                    self._set_text(slot[0], item[0][:width], item[1])
                elif isinstance(item[0], tuple):                              # table row: 4 cells
                    for t, (txt, col) in zip(slot, item):
                        self._set_text(t, txt, col)
                else:
                    self._set_text(slot[0], item[0][:width], item[1])
                    for t in slot[1:]:
                        self._set_text(t, "")
            over = len(lines) > n
            x_a, y_a, x_b, y_b = blk["rect"]
            if over:
                track = (y_b - y_a) - 4
                h = max(track * n / len(lines), 6)
                y_top = y_a + 2 + (track - h) * off / (len(lines) - n)
                blk["thumb"].center = (x_b - 4, y_top + h / 2)
                blk["thumb"].width, blk["thumb"].height = 3, h
            blk["thumb"].visible = over
            self._set_text(blk["hint"], f"{off + 1}-{min(off + n, len(lines))} of {len(lines)}, wheel" if over else "")

    def _block_at(self, pos):
        x, y = pos[:2]
        for key, (xa, ya, xb, yb) in ((k, b["rect"]) for k, b in self.blocks.items()):
            if xa <= x <= xb and ya <= y <= yb:
                return key
        return None

    def on_hover(self, event):
        """While the cursor is over the right panel, the camera ignores the mouse (the wheel scrolls)."""
        if event.pos is None:
            return
        xa, ya, xb, yb = self.panel_rect
        over = xa <= event.pos[0] <= xb and self._block_at(event.pos) is not None
        self.view.camera.interactive = not over

    def on_wheel(self, event):
        self.on_hover(event)
        key = self._block_at(event.pos) if event.pos is not None else None
        if key is None:
            return
        step = int(np.sign(event.delta[1])) * C.HUD_SCROLL_LINES
        self.blocks[key]["offset"] -= step
        self._render_blocks()
        self.canvas.update()

    def _update_hud(self):
        self._refresh_counts()
        for state, fn in self.ctrl_rows:
            self._set_text(state, *fn())
        self._set_text(self.btn_text, self._button_caption(self.link_mode), self._mode_color())

        m, s, lg = self.motion, self.schedule, self.log
        t_next = s.time_of(self.k_next)
        top = [
            ("STATUS", C.HUD_DIM_COLOR),
            (f"UTC {m.utc(self.sim_t):%Y-%m-%d %H:%M:%S}   x{self.time_scale:g}", C.HUD_TEXT_COLOR),
            (f"FPS {self.fps:5.1f}   {m.label} {m.num_sats} sats",
             C.HUD_ON_COLOR if self.fps >= 0.9 * C.TARGET_FPS else C.HUD_WARN_COLOR),
            (f"Next check {m.utc(t_next):%H:%M:%S} UTC, in {hms(t_next - self.sim_t)}", C.HUD_TEXT_COLOR)
            if self.requests_on else ("Requests OFF, press R to start", C.HUD_OFF_COLOR),
            (f"Checks {lg.n_checks}  requests {lg.n_requests}  served {lg.n_served}", C.HUD_TEXT_COLOR),
        ]
        for t, (txt, col) in zip(self.st_top, top):
            self._set_text(t, txt[:C.HUD_STATUS_CHARS], col)

        # SATELLITE LINKS: per-station table and notes
        links = [(("Station", C.HUD_DIM_COLOR), ("GS", C.LINK_COLOR), ("HAP", C.HAP_LINK_COLOR),
                  ("ISL", C.ISL_COLOR))]
        for name, g, h, i in zip(self.station_names, self.gs_counts, self.hap_counts, self.isl_counts):
            links.append(((name, C.HUD_TEXT_COLOR), (str(g), C.LINK_COLOR), (str(h), C.HAP_LINK_COLOR),
                          (str(i), C.ISL_COLOR)))
        links += [("", None),
                  (f"GS: satellites above {C.GS_MIN_ELEVATION_DEG:.0f} deg elevation", C.HUD_DIM_COLOR),
                  (f"HAP: satellites above {C.HAP_MIN_ELEVATION_DEG:.0f} deg elevation", C.HUD_DIM_COLOR),
                  ("ISL: neighbors of the ringed satellite,", C.HUD_DIM_COLOR),
                  (f"     line of sight above {C.H_ATM_KM:.0f} km altitude", C.HUD_DIM_COLOR),
                  (f"Circles: satellites at {self.median_alt_km:.0f} km altitude", C.HUD_DIM_COLOR)]
        self.blocks["links"]["lines"] = links

        # SOLUTION: state of the request cycle and the last result
        sol = []
        if self.mode == "COMPUTING":
            sol.append((f"FROZEN: computing route {time.perf_counter() - self._compute_t0:4.1f} s", C.HUD_WARN_COLOR))
        elif self.mode == "SHOWING":
            ok = self.result is not None and self.result.served
            sol.append(("FROZEN: route shown, press Enter" if ok else "FROZEN: no route, press Enter",
                        C.HUD_ON_COLOR if ok else C.HUD_OFF_COLOR))
        else:
            sol.append((self.message or ("waiting for the next request" if self.requests_on
                                         else "requests OFF: satellites only"), C.HUD_DIM_COLOR))
        if self.follow_info:
            sol.append((self.follow_info, C.HUD_ON_COLOR if self.follow is not None else C.HUD_WARN_COLOR))
        sol += [("", None), (f"{self.src.name} -> {self.dst.name}", C.HUD_DIM_COLOR)]
        r = self.result
        if r is None:
            sol += [("no request yet", C.HUD_DIM_COLOR)]
        else:
            sol += [(f"time        {r.time_utc:%Y-%m-%d %H:%M:%S} UTC", C.HUD_TEXT_COLOR),
                    (f"result      {r.status}", C.HUD_ON_COLOR if r.served else C.HUD_OFF_COLOR)]
            if r.served:
                sol += [(f"satellites  {len(r.sat_indices)}", C.HUD_TEXT_COLOR),
                        (f"length      {r.length_km:,.0f} km, limit {r.length_limit_km:,.0f} km", C.HUD_TEXT_COLOR),
                        (f"eta         {r.eta_p:.3f}", C.HUD_TEXT_COLOR),
                        (f"fidelity    {r.f_e2e:.4f}, required {r.f_min:g}", C.HUD_TEXT_COLOR),
                        (f"Pi          {r.pi:.3f}", C.HUD_TEXT_COLOR),
                        (f"EDR         {r.edr:.2f} ebits/s", C.HUD_TEXT_COLOR),
                        (f"selected    {r.selected}", C.HUD_TEXT_COLOR),
                        (f"            best EDR of {r.n_candidates} feasible paths", C.HUD_DIM_COLOR)]
            else:
                sol += [(r.note, C.HUD_TEXT_COLOR),
                        (f"limit       {r.length_limit_km:,.0f} km", C.HUD_TEXT_COLOR),
                        (f"fidelity    required {r.f_min:g}", C.HUD_TEXT_COLOR)]
            sol.append((f"computed in {r.compute_s:.1f} s", C.HUD_DIM_COLOR))
            for name, label, color in (("max_eta", "Dijkstra, max eta", C.MAX_ETA_PATH_COLOR),
                                       ("shortest", "Dijkstra, shortest", C.SHORTEST_PATH_COLOR)):
                ref = r.refs.get(name)
                if ref is not None:
                    tag = "same as result" if self._same_as_route(r, ref) else ref["status"]
                    sol += [("", None), (label, color),
                            (f"  {ref['n_sats']} sats, {ref['length_km']:,.0f} km, eta {ref['eta_p']:.3f}", color),
                            (f"  fidelity {ref['f_e2e']:.4f}, {tag}", color)]
                    if ref["status"] == "within limit":
                        sol.append((f"  EDR {ref['edr']:.2f} ebits/s", color))
            if r.served:
                sol += [("", None), ("relay satellites", C.HUD_DIM_COLOR)]
                sol += [(f"  {k:2d}  {name}", C.ROUTE_COLOR) for k, name in enumerate(r.sat_names, start=1)]
        if self.result is not self._shown_result:                          # new result: back to the top
            self.blocks["solution"]["offset"] = 0
            self._shown_result = self.result
        self.blocks["solution"]["lines"] = sol
        self._render_blocks()


def run_viewer(motion, src: S.Station, dst: S.Station, f_min: float, time_scale: float = C.TIME_SCALE):
    schedule = S.RequestSchedule(motion.start_utc)
    log = S.RequestLog(motion.key, src, dst, motion.start_utc)
    Viewer(motion, schedule, log, src, dst, f_min, time_scale)
    app.run()
