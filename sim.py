import os, re, csv, time
import urllib.request
import datetime as dt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

try:
    from sgp4.api import Satrec, SatrecArray
    from sgp4.propagation import gstime
    have_sgp4 = True
except:
    have_sgp4 = False

here = os.path.dirname(os.path.abspath(__file__))

# default scenario (aetherq.py can change these from the command line)
CONSTELLATION = 'starlink'      # starlink or oneweb
SOURCE = 'Washington DC'
DESTINATION = 'London'
START_UTC = None                # e.g. '2026-10-19T00:00:00', None = now
F_MIN = 0.84

STATIONS = {
    'Beijing': (39.9042,116.4074),
    'Washington DC': (38.9072,-77.0369),
    'Hefei': (31.8206,117.2272),
    'Shanghai': (31.2304,121.4737),
    'Tokyo': (35.6762,139.6503),
    'Singapore': (1.3521,103.8198),
    'Sydney': (-33.8688,151.2093),
    'London': (51.5074,-0.1278),
    'Paris': (48.8566,2.3522),
    'Vienna': (48.2082,16.3738),
    'New York': (40.7128,-74.0060),
    'Chicago': (41.8781,-87.6298),
    'Los Angeles': (34.0522,-118.2437),
    'Sao Paulo': (-23.5505,-46.6333),
    'Buenos Aires': (-34.6037,-58.3816),
    'Cape Town': (-33.9249,18.4241),
}

R_E = 6371.0                    # km
MU = 398600.4418
C_KM_S = 299792.458

H_HAP = 20.0                    # hap straight above each station, km
HAP_MIN_EL = 5.0                # deg
H_ATM = 20.0                    # isl must stay above this altitude, km
ISL_MAX_KM = None

WAVELENGTH = 810e-9
RHO = 1.0                       # lens radius, m
W0 = 2.0*RHO/(1.22*np.pi)       # beam waist, ~0.522 m
ETA_HW = 1.0-0.035
T_ZEN = 0.5
HV_WIND = 21.0
HV_CN2 = 1.7e-14

F0_GH = 0.97
F0_HH = 0.97
T_COH = 1.65                    # s
K_DECAY = 2.2
O1 = 0.99
O2 = 0.99
ALPHA = 0.99
Q_GEN = 0.05
Q_SWP = 0.5
N_ATTEMPTS = 100

CONSTS = {
    'starlink': ('Starlink','starlink',250.0),     # label, celestrak group, min altitude km
    'oneweb': ('OneWeb','oneweb',1000.0),
}
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle'
TLE_MAX_AGE_H = 12.0
STEP_S = 20.0                   # sgp4 every 20 s, hermite in between

REQUEST_PERIOD = 60.0
REQUEST_PROB = 0.5
SEED = 2026
SHOW_REFS = True


# /////////////////////////////////////////////////////////////////////////////
#                                   physics
# /////////////////////////////////////////////////////////////////////////////

def central_angle(p1,l1,p2,l2):
    c = np.sin(p1)*np.sin(p2) + np.cos(p1)*np.cos(p2)*np.cos(l2-l1)
    return np.arccos(np.clip(c,-1.0,1.0))

def slant_range(r_low,r,g):
    return np.sqrt(r_low**2 + r**2 - 2.0*r_low*r*np.cos(g))

def elevation(r_low,r,g):
    return np.arctan2(np.cos(g)-r_low/r,np.sin(g))

def isl_visible(r1,r2,g):
    ra = R_E+H_ATM
    return g <= np.arccos(np.clip(ra/r1,-1,1)) + np.arccos(np.clip(ra/r2,-1,1))

def beam_radius(z):
    zr = np.pi*W0**2/WAVELENGTH
    return W0*np.sqrt(1.0+(z/zr)**2)

def collected(w):
    return 1.0-np.exp(-2.0*RHO**2/np.asarray(w)**2)

def hop_eta(L_km):
    return ETA_HW*collected(beam_radius(np.asarray(L_km)*1e3))

def cn2(h):
    h = np.asarray(h,dtype=float)
    return (0.00594*(HV_WIND/27.0)**2*(1e-5*h)**10*np.exp(-h/1000.0)
            + 2.7e-16*np.exp(-h/1500.0) + HV_CN2*np.exp(-h/100.0))

def fried_r0(z,el,n=20001):
    # uplink, turbulence near the transmitter counts most
    k = 2.0*np.pi/WAVELENGTH
    zp = np.linspace(0.0,z,n)
    f = cn2(zp*np.sin(el))*((z-zp)/z)**(5.0/3.0)
    integral = float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(zp)))
    return (0.42*k**2*integral)**(-3.0/5.0)

def gs_hap_link():
    r = R_E+H_HAP
    return float(slant_range(R_E,r,0.0)), float(elevation(R_E,r,0.0))

_cache = {}
def ground_etas():
    if 'up' not in _cache:
        L,el = gs_hap_link()
        z,k = L*1e3, 2.0*np.pi/WAVELENGTH
        w_lt = np.sqrt(beam_radius(z)**2 + 2.0*(4.0*z/(k*fried_r0(z,el)))**2)
        atm = T_ZEN**(1.0/np.sin(el))
        _cache['up'] = float(ETA_HW*atm*collected(w_lt))
        _cache['down'] = float(ETA_HW*atm*collected(beam_radius(z)))   # downlink: diffraction only
    return _cache['up'],_cache['down']

def werner(f0,tau):
    return (f0-0.25)*np.exp(-(np.asarray(tau)/T_COH)**K_DECAY) + 0.25

def swap(f1,f2):
    s = O1*O2*(4*ALPHA**2-1)/3
    return 0.25*(1 + 3*s*(4*f1-1)/3*(4*f2-1)/3)

def fidelity(tau):
    f_gh,f_hh = werner(F0_GH,tau),werner(F0_HH,tau)
    return swap(f_gh,swap(f_gh,f_hh))

def round_trip(L_km):
    return 2.0*np.asarray(L_km)/C_KM_S

def length_limit(f_min):
    # longest hap-hap segment whose fidelity is still >= f_min
    if fidelity(0.0) < f_min:
        return 0.0
    lo,hi = 0.0,3600.0
    if fidelity(hi) >= f_min:
        return C_KM_S*hi/2.0
    for i in range(200):
        mid = 0.5*(lo+hi)
        if fidelity(mid) >= f_min:
            lo = mid
        else:
            hi = mid
    return C_KM_S*lo/2.0

def q_link(eta):
    return -np.expm1(N_ATTEMPTS*np.log1p(-Q_GEN*np.asarray(eta)))

def success_prob(eta_p):
    up,down = ground_etas()
    return float(Q_SWP**2*q_link(up)*q_link(eta_p)*q_link(down))

def edr(eta_p,L_km):
    return float(success_prob(eta_p)/round_trip(L_km))


# /////////////////////////////////////////////////////////////////////////////
#                                   satellites
# /////////////////////////////////////////////////////////////////////////////

j2000 = dt.datetime(2000,1,1,12,0,0,tzinfo=dt.timezone.utc)

def parse_utc(s):
    if s is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    t = dt.datetime.fromisoformat(str(s))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)

def jday(t):
    return 2451545.0 + (t-j2000).total_seconds()/86400.0

def gmst(jd):
    if have_sgp4:
        return gstime(jd)
    return np.radians((280.46061837 + 360.98564736629*(jd-2451545.0)) % 360.0)

def rotz(r,a):
    c,s = np.cos(a),np.sin(a)
    return np.stack([c*r[...,0]-s*r[...,1],s*r[...,0]+c*r[...,1],r[...,2]],-1)

def latlon(r):
    return np.arcsin(r[...,2]/np.linalg.norm(r,axis=-1)), np.arctan2(r[...,1],r[...,0])

def to_ecef(lat,lon,radius):
    lat,lon = np.radians(lat),np.radians(lon)
    return radius*np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])

def get_tle(const,download=True):
    label,group,_ = CONSTS[const]
    path = os.path.join(here,'assets',const+'.tle')
    age = (time.time()-os.path.getmtime(path))/3600.0 if os.path.exists(path) else None
    if age is not None and (age < TLE_MAX_AGE_H or not download):
        return path
    os.makedirs(os.path.dirname(path),exist_ok=True)
    try:
        print('downloading %s TLEs ...' % label)
        req = urllib.request.Request(TLE_URL % group,headers={'User-Agent':'AETHER-Q'})
        data = urllib.request.urlopen(req,timeout=60).read().decode('ascii',errors='replace')
        if data.count('\n1 ') < 10:
            raise RuntimeError('not a TLE file: '+data[:100])
        open(path+'.part','w',newline='\n').write(data)
        os.replace(path+'.part',path)
    except Exception as e:
        if age is None:
            raise SystemExit('could not download %s TLEs (%s)' % (label,e))
        print('  download failed (%s), using the old file (%.1f h)' % (e,age))
    return path

def read_tle(path):
    lines = [l.rstrip() for l in open(path,encoding='ascii',errors='replace') if l.strip()]
    out,i = [],0
    while i < len(lines)-1:
        if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
            out.append(('',lines[i],lines[i+1]))
            i += 2
        elif i+2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            out.append((lines[i].strip(),lines[i+1],lines[i+2]))
            i += 3
        else:
            i += 1
    return out

def tle_epoch(l1):
    yy,doy = int(l1[18:20]),float(l1[20:32])
    year = 2000+yy if yy < 57 else 1900+yy
    return jday(dt.datetime(year,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=doy-1.0))

def kepler(tles,jd):
    # only used when sgp4 is not installed
    l2 = [t[2] for t in tles]
    inc = np.radians([float(s[8:16]) for s in l2])
    raan = np.radians([float(s[17:25]) for s in l2])
    ecc = np.array([float('0.'+s[26:33].strip()) for s in l2])
    argp = np.radians([float(s[34:42]) for s in l2])
    m0 = np.radians([float(s[43:51]) for s in l2])
    n = np.array([float(s[52:63]) for s in l2])*2*np.pi/86400.0
    a = (MU/n**2)**(1.0/3.0)
    M = m0 + n*(jd-np.array([tle_epoch(t[1]) for t in tles]))*86400.0
    E = M.copy()
    for i in range(8):
        E -= (E-ecc*np.sin(E)-M)/(1-ecc*np.cos(E))
    xp,yp = a*(np.cos(E)-ecc), a*np.sqrt(1-ecc**2)*np.sin(E)
    cO,sO,cw,sw,ci,si = np.cos(raan),np.sin(raan),np.cos(argp),np.sin(argp),np.cos(inc),np.sin(inc)
    P = np.stack([cO*cw-sO*sw*ci,sO*cw+cO*sw*ci,sw*si],-1)
    Q = np.stack([-cO*sw-sO*cw*ci,-sO*sw+cO*cw*ci,cw*si],-1)
    return xp[:,None]*P + yp[:,None]*Q


class satellites:
    # positions in the view frame (inertial, x on the prime meridian at the start) and earth fixed
    def __init__(self,const,start):
        self.const = const
        self.label = CONSTS[const][0]
        near_now = abs((dt.datetime.now(dt.timezone.utc)-start).total_seconds()) < 86400
        tles = read_tle(get_tle(const,near_now))
        self.start = start
        self.jd0 = jday(start)
        self.gmst0 = gmst(self.jd0)
        self.backend = 'SGP4' if have_sgp4 else 'Kepler (pip install sgp4)'
        self.load(tles)
        r0,_ = self.teme(np.array([0.0]))
        alt = np.linalg.norm(r0[:,0],axis=1)-R_E
        keep = np.isfinite(alt) & (alt >= CONSTS[const][2])
        self.load([t for t,k in zip(tles,keep) if k])
        self.names = [t[0] or t[1][2:7].strip() for t in self.tles]
        self.inc = np.array([float(t[2][8:16]) for t in self.tles])
        self.n = len(self.tles)
        age = np.abs(self.jd0-np.array([tle_epoch(t[1]) for t in self.tles]))
        if np.median(age) > 3.0:
            print('WARNING: start is %.1f days away from the TLEs, positions can be off' % np.median(age))
        self.ta = None

    def load(self,tles):
        self.tles = tles
        if have_sgp4:
            self.arr = SatrecArray([Satrec.twoline2rv(l1,l2) for _,l1,l2 in tles])

    def teme(self,t):
        t = np.asarray(t,dtype=float)
        if have_sgp4:
            day = np.floor(self.jd0)
            err,r,v = self.arr.sgp4(np.full(len(t),day),(self.jd0-day)+t/86400.0)
            r[err!=0] = np.nan
            v[err!=0] = np.nan
            return r,v
        r = np.stack([kepler(self.tles,self.jd0+x/86400.0) for x in t],axis=1)
        r1 = np.stack([kepler(self.tles,self.jd0+(x+1.0)/86400.0) for x in t],axis=1)
        return r,r1-r

    def earth_angle(self,t):
        return gmst(self.jd0+t/86400.0)-self.gmst0

    def view_pos(self,t):
        if self.ta is None or not (self.ta <= t <= self.ta+STEP_S):
            ta = np.floor(t/STEP_S)*STEP_S
            r,v = self.teme(np.array([ta,ta+STEP_S]))
            r,v = rotz(r,-self.gmst0),rotz(v,-self.gmst0)
            self.ta,self.ra,self.rb,self.va,self.vb = ta,r[:,0],r[:,1],v[:,0],v[:,1]
        s,h = (t-self.ta)/STEP_S,STEP_S
        return ((2*s**3-3*s**2+1)*self.ra + (s**3-2*s**2+s)*h*self.va
                + (-2*s**3+3*s**2)*self.rb + (s**3-s**2)*h*self.vb)

    def ecef(self,t):
        r,_ = self.teme(np.array([t]))
        return rotz(r[:,0],-gmst(self.jd0+t/86400.0))

    def utc(self,t):
        return self.start+dt.timedelta(seconds=float(t))


# /////////////////////////////////////////////////////////////////////////////
#                                   graph and routing
# /////////////////////////////////////////////////////////////////////////////

def angles(a,b):
    pa,la = latlon(a)
    pb,lb = latlon(b)
    return central_angle(pa[:,None],la[:,None],pb[None,:],lb[None,:])

def build_graph(sats,hap_s,hap_d):
    # nodes 0..n-1 satellites, n source hap, n+1 destination hap. no other hap in the graph
    n = len(sats)
    r = np.linalg.norm(sats,axis=1)
    rows,cols,lens = [],[],[]
    for st in range(0,n,1024):
        idx = np.arange(st,min(st+1024,n))
        g = angles(sats[idx],sats)
        vis = isl_visible(r[idx][:,None],r[None,:],g) & (np.arange(n)[None,:] > idx[:,None])
        L = slant_range(r[idx][:,None],r[None,:],g)
        if ISL_MAX_KM is not None:
            vis &= L <= ISL_MAX_KM
        a,b = np.nonzero(vis)
        rows.append(idx[a]); cols.append(b); lens.append(L[a,b])
    for k,hap in enumerate((hap_s,hap_d)):
        rh = np.linalg.norm(hap)
        g = angles(hap[None,:],sats)[0]
        ok = np.degrees(elevation(rh,r,g)) >= HAP_MIN_EL
        rows.append(np.nonzero(ok)[0]); cols.append(np.full(ok.sum(),n+k)); lens.append(slant_range(rh,r,g)[ok])
    rows,cols,lens = np.concatenate(rows),np.concatenate(cols),np.concatenate(lens)
    cost = -np.log(hop_eta(lens))
    graph = {'n':n,'src':n,'dst':n+1,'pos':np.vstack([sats,hap_s,hap_d])}
    graph['length'] = sp.csr_matrix((np.maximum(lens,1e-9),(rows,cols)),shape=(n+2,n+2))
    graph['cost'] = sp.csr_matrix((np.maximum(cost,1e-12),(rows,cols)),shape=(n+2,n+2))
    return graph

def shortest(w,src,dst):
    dist,pred = dijkstra(w,directed=False,indices=src,return_predecessors=True)
    if not np.isfinite(dist[dst]):
        return None
    path,v = [dst],dst
    while v != src:
        v = int(pred[v])
        path.append(v)
    return path[::-1]

def totals(graph,path):
    c = l = 0.0
    for u,v in zip(path[:-1],path[1:]):
        a,b = min(u,v),max(u,v)
        c += graph['cost'][a,b]
        l += graph['length'][a,b]
    return float(c),float(l)

def path_edr(c,l):
    return edr(np.exp(-c),l)

def select_path(graph,l_max,max_iter=50,score=path_edr):
    # p_c: max eta, p_d: shortest. if p_c is too long, larac walks between the two.
    # every path within l_max found on the way is a candidate, the largest edr wins
    src,dst = graph['src'],graph['dst']
    out = {'status':'','nodes':[],'refs':{},'selected':'','n_cand':0,'note':''}
    p_c = shortest(graph['cost'],src,dst)
    if p_c is None:
        out['status'],out['note'] = 'no path','the two haps are not connected'
        return out
    c_c,l_c = totals(graph,p_c)
    p_d = shortest(graph['length'],src,dst)
    c_d,l_d = totals(graph,p_d)
    for name,p,c,l in (('max_eta',p_c,c_c,l_c),('shortest',p_d,c_d,l_d)):
        out['refs'][name] = (p,c,l,'within limit' if l <= l_max else 'too long')
    if l_d > l_max:
        out['status'],out['note'] = 'infeasible','shortest path %.0f km > limit %.0f km' % (l_d,l_max)
        return out

    cand = [(p_d,c_d,l_d,'shortest path')]
    if l_c <= l_max:
        cand.insert(0,(p_c,c_c,l_c,'max-eta path'))
    else:
        for it in range(1,max_iter+1):
            lam = (c_c-c_d)/(l_d-l_c)
            w = graph['cost'].copy()
            w.data = graph['cost'].data + lam*graph['length'].data
            p = shortest(w,src,dst)
            c,l = totals(graph,p)
            if np.isclose(c+lam*l,c_c+lam*l_c,rtol=1e-12,atol=1e-12):
                break
            if l <= l_max:
                p_d,c_d,l_d = p,c,l
                cand.append((p,c,l,'LARAC iterate %d' % it))
            else:
                p_c,c_c,l_c = p,c,l

    best,seen = None,set()
    for p,c,l,label in cand:
        if tuple(p) in seen:
            continue
        seen.add(tuple(p))
        e = score(c,l)
        if best is None or e > best[0]+1e-12 or (abs(e-best[0]) <= 1e-12 and c < best[2]):
            best = (e,p,c,l,label)
    out['status'] = 'served'
    out['edr'],out['nodes'],out['cost'],out['length'],out['selected'] = best
    out['n_cand'] = len(seen)
    return out


# /////////////////////////////////////////////////////////////////////////////
#                                   requests
# /////////////////////////////////////////////////////////////////////////////

def find_station(s):
    m = re.fullmatch(r'\s*([-+]?\d+(?:\.\d*)?)\s*,\s*([-+]?\d+(?:\.\d*)?)\s*',s)
    if m:
        lat,lon = float(m.group(1)),float(m.group(2))
        return ('(%g, %g)' % (lat,lon),lat,lon)
    key = lambda x: re.sub(r'[^a-z0-9]','',x.lower())
    hits = [n for n in STATIONS if key(n) == key(s)] or [n for n in STATIONS if key(n).startswith(key(s))]
    if len(hits) != 1 or key(s) == '':
        return None
    return (hits[0],)+STATIONS[hits[0]]


class requests:
    # decision k at time_of(k), the k-th full period after the start. seeded, so runs repeat
    def __init__(self,start):
        e = start.timestamp()
        self.t0 = np.floor(e/REQUEST_PERIOD)*REQUEST_PERIOD + REQUEST_PERIOD - e
        self.rng = np.random.default_rng(SEED)
        self.d = []

    def time_of(self,k):
        return self.t0 + k*REQUEST_PERIOD

    def decision(self,k):
        while len(self.d) <= k:
            self.d.append(bool(self.rng.random() < REQUEST_PROB))
        return self.d[k]

    def next_request(self,k):
        while not self.decision(k):
            k += 1
        return k


class result:
    def __init__(self,t,utc,src,dst,f_min,l_max):
        self.t,self.utc,self.f_min,self.l_max = t,utc,f_min,l_max
        self.source,self.destination = src[0],dst[0]
        self.gs = np.array([to_ecef(src[1],src[2],R_E),to_ecef(dst[1],dst[2],R_E)])
        self.hap = self.gs*(R_E+H_HAP)/R_E
        self.status,self.note,self.selected = '','',''
        self.served = False
        self.sats,self.sat_names,self.nodes = [],[],None
        self.length,self.eta,self.pi,self.f_e2e,self.edr = np.nan,0.0,0.0,np.nan,0.0
        self.n_cand = 0
        self.refs = {}
        self.compute_s = 0.0


def compute_route(sat,t,src,dst,f_min):
    t_wall = time.perf_counter()
    l_max = length_limit(f_min)
    r = result(t,sat.utc(t),src,dst,f_min,l_max)
    if l_max <= 0:
        r.status,r.note = 'F_min unreachable','F_min %g is above the best possible fidelity' % f_min
        r.compute_s = time.perf_counter()-t_wall
        return r
    graph = build_graph(sat.ecef(t),r.hap[0],r.hap[1])
    out = select_path(graph,l_max)
    r.status,r.note = out['status'],out['note']
    r.served = r.status == 'served'
    if SHOW_REFS:
        for name,(p,c,l,st) in out['refs'].items():
            eta = float(np.exp(-c))
            r.refs[name] = {'pos':graph['pos'][p],'length':l,'eta':eta,'f_e2e':float(fidelity(round_trip(l))),
                            'edr':edr(eta,l),'status':st,'n_sats':len(p)-2}
    if r.served:
        r.sats = [int(i) for i in out['nodes'][1:-1]]
        r.sat_names = [sat.names[i] for i in r.sats]
        r.nodes = graph['pos'][out['nodes']]
        r.length,r.eta = out['length'],float(np.exp(-out['cost']))
        r.pi = success_prob(r.eta)
        r.f_e2e = float(fidelity(round_trip(r.length)))
        r.edr = edr(r.eta,r.length)
        r.selected,r.n_cand = out['selected'],out['n_cand']
    r.compute_s = time.perf_counter()-t_wall
    return r


def check_path(hap_s,sats,hap_d,l_max):
    # is a path found earlier still usable now? all positions earth fixed
    pts = np.vstack([hap_s,sats,hap_d])
    L = np.linalg.norm(pts[1:]-pts[:-1],axis=1)
    g = lambda a,b: central_angle(latlon(a)[0],latlon(a)[1],latlon(b)[0],latlon(b)[1])
    el_up = np.degrees(elevation(np.linalg.norm(hap_s),np.linalg.norm(sats[0]),g(hap_s[None],sats[:1])[0]))
    el_dn = np.degrees(elevation(np.linalg.norm(hap_d),np.linalg.norm(sats[-1]),g(hap_d[None],sats[-1:])[0]))
    isl_ok = True
    if len(sats) > 1:
        isl_ok = bool(np.all(isl_visible(np.linalg.norm(sats[:-1],axis=1),np.linalg.norm(sats[1:],axis=1),
                                         g(sats[:-1],sats[1:]))))
    why = []
    if not el_up >= HAP_MIN_EL:
        why.append('first satellite at %.1f deg' % el_up)
    if not el_dn >= HAP_MIN_EL:
        why.append('last satellite at %.1f deg' % el_dn)
    if not isl_ok:
        why.append('inter-satellite link lost')
    if not L.sum() <= l_max:
        why.append('path %.0f km > %.0f km' % (L.sum(),l_max))
    return why,float(np.prod(hop_eta(L))),float(el_up),float(el_dn)


class request_log:
    fields = ['utc','request','status','n_satellites','length_km','eta','fidelity','pi','edr','selected',
              'compute_s','max_eta_length_km','max_eta_status','shortest_length_km','shortest_status','satellites']

    def __init__(self,const,src,dst,start,tag=''):
        os.makedirs(os.path.join(here,'results'),exist_ok=True)
        pair = '_'.join(re.sub(r'[^A-Za-z0-9.,-]','',s[0]).replace(',','_') for s in (src,dst))
        self.path = os.path.join(here,'results','requests%s_%s_%s_%s.csv' % (tag,const,pair,start.strftime('%Y%m%dT%H%M%S')))
        self.write(self.fields,'w')
        self.n_checks,self.n_req,self.n_served = 0,0,0

    def write(self,row,mode='a'):
        with open(self.path,mode,newline='') as f:
            csv.writer(f).writerow(row)

    def no_request(self,utc):
        self.n_checks += 1
        self.write([utc.strftime('%Y-%m-%d %H:%M:%S'),0]+['']*(len(self.fields)-2))

    def route(self,r):
        self.n_checks += 1
        self.n_req += 1
        self.n_served += int(r.served)
        ok = r.served
        refs = []
        for name in ('max_eta','shortest'):
            x = r.refs.get(name)
            refs += ['%.1f' % x['length'],x['status']] if x else ['','']
        self.write([r.utc.strftime('%Y-%m-%d %H:%M:%S'),1,r.status,len(r.sats),'%.1f' % r.length if ok else '',
                    '%.6g' % r.eta if ok else '','%.6f' % r.f_e2e if ok else '','%.6g' % r.pi,'%.6g' % r.edr,
                    r.selected,'%.2f' % r.compute_s]+refs+[' '.join(r.sat_names)])


def describe(r):
    s = '%s UTC  %s -> %s: ' % (r.utc.strftime('%Y-%m-%d %H:%M:%S'),r.source,r.destination)
    if not r.served:
        return s+'%s (%s)' % (r.status,r.note)
    return s+'%d satellites, %.0f km, eta %.3f, fidelity %.4f, EDR %.2f ebits/s (%s, %.1f s)' % (
        len(r.sats),r.length,r.eta,r.f_e2e,r.edr,r.selected,r.compute_s)
