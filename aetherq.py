import argparse, sys

import sim


def main():
    parser = argparse.ArgumentParser(description='AETHER-Q: GS -> HAP -> LEO lens satellites -> HAP -> GS, 3D view')
    parser.add_argument('--const',default=sim.CONSTELLATION,choices=list(sim.CONSTS),help='starlink or oneweb')
    parser.add_argument('--src',default=sim.SOURCE,help='station name or "lat,lon"')
    parser.add_argument('--dst',default=sim.DESTINATION,help='station name or "lat,lon"')
    parser.add_argument('--start',default=sim.START_UTC,help='UTC start, e.g. 2026-10-19T00:00:00 (default now)')
    parser.add_argument('--fmin',type=float,default=sim.F_MIN,help='required fidelity [default=%(default)s]')
    parser.add_argument('--speed',type=float,default=60.0,help='time speed up [default=%(default)s]')
    parser.add_argument('--headless',action='store_true',help='no window, just run the requests')
    parser.add_argument('--hours',type=float,default=24.0,help='hours for --headless [default=%(default)s]')
    options = parser.parse_args()

    src,dst = sim.find_station(options.src),sim.find_station(options.dst)
    if src is None or dst is None:
        sys.stderr.write('unknown station, use "lat,lon" or one of: %s\n' % ', '.join(sim.STATIONS))
        sys.exit(1)
    if src[1:] == dst[1:]:
        sys.stderr.write('source and destination are the same\n')
        sys.exit(1)

    sat = sim.satellites(options.const.lower(),sim.parse_utc(options.start))
    print('%d %s satellites (%s), start %s UTC, %s -> %s, F_min %g' % (sat.n,sat.label,sat.backend,
          sat.start.strftime('%Y-%m-%d %H:%M:%S'),src[0],dst[0],options.fmin))

    if not options.headless:
        import viewer
        viewer.run(sat,src,dst,options.fmin,options.speed)
        return

    req = sim.requests(sat.start)
    log = sim.request_log(sat.const,src,dst,sat.start,'_headless')
    k = 0
    while req.time_of(k) <= options.hours*3600.0:
        t = req.time_of(k)
        if req.decision(k):
            r = sim.compute_route(sat,t,src,dst,options.fmin)
            log.route(r)
            print('  '+sim.describe(r))
        else:
            log.no_request(sat.utc(t))
        k += 1
    print('%d requests, %d served -> %s' % (log.n_req,log.n_served,log.path))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
