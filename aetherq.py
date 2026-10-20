"""
AETHER-Q: entanglement distribution ground station -> HAP -> passive LEO lens satellites -> HAP ->
ground station over the real Starlink or OneWeb constellation, in a real-time 3D view.

Keys in the 3D view: R requests on / off, Enter resume after a route, N next request, Space pause,
+/- speed, P path follow, G grid, L links, H link mode, C reach circles, I inter-satellite links,
mouse drag rotate, wheel zoom, F11 full screen, Esc quit.
"""

import argparse

import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", default=C.CONSTELLATION, type=str.lower, choices=sorted(C.CONSTELLATIONS),
                    help=f"constellation (default {C.CONSTELLATION})")
    ap.add_argument("--src", default=C.SOURCE, help=f'source station or "lat,lon" (default {C.SOURCE})')
    ap.add_argument("--dst", default=C.DESTINATION, help=f'destination station or "lat,lon" (default {C.DESTINATION})')
    ap.add_argument("--start", default=C.START_UTC, help="UTC start, e.g. 2026-10-19T00:00:00 (default now)")
    ap.add_argument("--fmin", type=float, default=C.F_MIN, help=f"required fidelity (default {C.F_MIN:g})")
    ap.add_argument("--speed", type=float, default=C.TIME_SCALE, help=f"time speed-up (default {C.TIME_SCALE:g})")
    ap.add_argument("--headless", action="store_true", help="no window: run the requests and log them")
    ap.add_argument("--hours", type=float, default=24.0, help="simulated hours with --headless (default 24)")
    args = ap.parse_args()

    from constellation import SatelliteMotion, parse_utc
    from simulation import find_station, run_headless

    src, dst = find_station(args.src), find_station(args.dst)
    if (src.lat_deg, src.lon_deg) == (dst.lat_deg, dst.lon_deg):
        raise SystemExit("source and destination are the same station")
    motion = SatelliteMotion(args.const, parse_utc(args.start))
    print(f"{motion.num_sats} {motion.label} satellites ({motion.backend}), start "
          f"{motion.start_utc:%Y-%m-%d %H:%M:%S} UTC, {src.name} -> {dst.name}, F_min {args.fmin:g}")
    if args.headless:
        run_headless(motion, src, dst, args.fmin, args.hours)
    else:
        import viewer                                   # only the 3D view needs vispy
        viewer.run_viewer(motion, src, dst, args.fmin, args.speed)


if __name__ == "__main__":
    main()
