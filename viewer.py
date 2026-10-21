import os, time, urllib.request
import concurrent.futures as cf
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import MatrixTransform

try:
    from vispy.visuals.filters import TextureFilter
except:
    TextureFilter = None

import sim
from sim import R_E, rotz, latlon

TEXTURE = os.path.join(sim.here,'assets','earth_bluemarble.jpg')
TEXTURE_URL = 'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/world.200407.3x5400x2700.jpg'

FPS = 120
SPEEDS = [1,10,30,60,120,300,600,1200,3000]
WINDOW = (1600,1000)
BG = '#05070d'
EARTH_COLOR = (0.10,0.35,0.85,1.0)          # if no texture
GRID_COLOR = (0.75,0.85,1.0,0.30)
SAT_SIZE = {'starlink':3,'oneweb':5}
SHELLS = {                                  # color by inclination
    'starlink': [(43.0,(1.00,0.55,0.20,0.9)),(53.1,(0.35,0.85,1.00,0.9)),
                 (70.0,(0.60,1.00,0.45,0.9)),(97.6,(1.00,0.45,0.85,0.9))],
    'oneweb': [(87.9,(0.95,0.85,0.45,0.9))],
}
OTHER_COLOR = (0.80,0.80,0.80,0.9)
STATION_COLOR = (1.0,0.85,0.1,1.0)
HAP_COLOR = (0.2,0.95,1.0,1.0)
GS_MIN_EL = 20.0                            # gs -> satellite links, only drawn
LINK_MODES = ['GS','HAP','GS+HAP']
GS_LINK = (1.0,0.85,0.1,0.8)
HAP_LINK = (0.2,0.95,1.0,0.8)
ISL_COLOR = (1.0,0.35,1.0,0.6)
ROUTE_COLOR = (0.35,1.0,0.35,0.95)
MAX_ETA_COLOR = (1.0,0.55,0.10,0.9)
SHORTEST_COLOR = (0.45,0.60,1.00,0.9)

FONT = 11
FACES = ['Consolas','Courier New','DejaVu Sans Mono']
PANEL_CHARS = 48
WHITE = (1.0,1.0,1.0,1.0)
DIM = (0.65,0.72,0.85,1.0)
ON = (0.30,1.00,0.40,1.0)
OFF = (1.00,0.35,0.35,1.0)
WARN = (1.00,0.75,0.25,1.0)
FRAME = (0.45,0.52,0.65,0.8)
LINK_LINES = 6
MAX_LINES = 40


def load_texture():
    if not os.path.exists(TEXTURE):
        try:
            os.makedirs(os.path.dirname(TEXTURE),exist_ok=True)
            print('downloading earth texture ...')
            urllib.request.urlretrieve(TEXTURE_URL,TEXTURE+'.part')
            os.replace(TEXTURE+'.part',TEXTURE)
        except Exception as e:
            print('  texture download failed (%s), solid earth' % e)
            return None
    try:
        from PIL import Image
        img = Image.open(TEXTURE).convert('RGB')
        if img.width > 4096:
            img = img.resize((4096,2048),Image.LANCZOS)
        return np.asarray(img,dtype=np.uint8)
    except:
        return None

def uv_sphere(rows=90,cols=180):
    LA,LO = np.meshgrid(np.linspace(-90,90,rows+1),np.linspace(-180,180,cols+1),indexing='ij')
    la,lo = np.radians(LA),np.radians(LO)
    v = np.stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)],-1).reshape(-1,3)
    tex = np.stack([(LO+180)/360,(90-LA)/180],-1).reshape(-1,2)
    idx = np.arange((rows+1)*(cols+1)).reshape(rows+1,cols+1)
    a,b,c,d = idx[:-1,:-1],idx[:-1,1:],idx[1:,:-1],idx[1:,1:]
    f = np.concatenate([np.stack([a,b,d],-1).reshape(-1,3),np.stack([a,d,c],-1).reshape(-1,3)])
    return v.astype(np.float32),f.astype(np.uint32),tex.astype(np.float32)

def graticule(step=30,n=181):
    lines = []
    t = np.radians(np.linspace(-180,180,n))
    for lat in range(-90+step,90,step):
        la = np.radians(lat)
        lines.append(np.stack([np.cos(la)*np.cos(t),np.cos(la)*np.sin(t),np.full_like(t,np.sin(la))],1))
    p = np.radians(np.linspace(-90,90,n))
    for lon in range(-180,180,step):
        lo = np.radians(lon)
        lines.append(np.stack([np.cos(p)*np.cos(lo),np.cos(p)*np.sin(lo),np.sin(p)],1))
    return 1.003*np.concatenate([np.repeat(l,2,axis=0)[1:-1] for l in lines]).astype(np.float32)

def gamma_to(pts,sat_ll):
    pa,la = latlon(pts)
    return sim.central_angle(pa[:,None],la[:,None],sat_ll[0][None,:],sat_ll[1][None,:])

def elev_deg(lows,sat_ll,sat_r):
    return np.degrees(sim.elevation(np.linalg.norm(lows,axis=1)[:,None],sat_r[None,:],gamma_to(lows,sat_ll)))

def reach_angle(r_low,r_sat,min_el):
    e = np.radians(min_el)
    return float(np.arccos(r_low*np.cos(e)/r_sat)-e)

def ring(center,half,n=120):
    up = center/np.linalg.norm(center)
    east = np.cross([0.0,0.0,1.0],up)
    east /= np.linalg.norm(east)
    north = np.cross(up,east)
    th = np.linspace(0,2*np.pi,n+1)
    r = 1.004*(np.cos(half)*up + np.sin(half)*(np.cos(th)[:,None]*east+np.sin(th)[:,None]*north))
    return np.repeat(r,2,axis=0)[1:-1].astype(np.float32)

def hms(s):
    s = max(int(round(s)),0)
    return '%02d:%02d:%02d' % (s//3600,s%3600//60,s%60)

def onoff(flag):
    return ('ON',ON) if flag else ('OFF',OFF)

def empty_line(color,width,parent):
    return scene.visuals.Line(pos=np.zeros((2,3),np.float32),connect='segments',color=color,width=width,parent=parent)


class viewer:
    def __init__(self,sat,req,log,src,dst,f_min,speed):
        self.sat,self.req,self.log = sat,req,log
        self.src,self.dst,self.f_min = src,dst,f_min
        self.speed = speed
        self.t = 0.0
        self.k = 0                          # next request decision
        self.requests_on = False            # R key
        self.mode = 'RUNNING'               # RUNNING, PAUSED, COMPUTING, SHOWING
        self.follow_on = False
        self.follow = None
        self.follow_info = ''
        self.res = None
        self.shown = None
        self.msg = ''
        self.pool = cf.ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.job_t0 = 0.0
        self.last = time.perf_counter()
        self.fps,self.fps_n,self.fps_t0 = 0.0,0,time.perf_counter()
        self.colors = {}
        self.show_links,self.show_rings,self.show_isl = True,True,False
        self.link_mode = 'HAP'
        self.names = [src[0],dst[0]]
        self.counts = {'gs':[0,0],'hap':[0,0],'isl':[0,0]}
        self.count_t = {}

        self.canvas = scene.SceneCanvas(keys='interactive',size=WINDOW,bgcolor=BG,show=True,
                                        title='AETHER-Q - %s -> %s' % (src[0],dst[0]))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=40,distance=4.5,elevation=25,azimuth=40)

        # earth node turns with the earth: texture, grid, stations, haps
        self.earth = scene.Node(parent=self.view.scene)
        self.earth_tf = MatrixTransform()
        self.earth.transform = self.earth_tf
        img = load_texture() if TextureFilter is not None else None
        if img is not None:
            v,f,tex = uv_sphere()
            mesh = scene.visuals.Mesh(vertices=v,faces=f,color=(1,1,1,1),shading=None,parent=self.earth)
            mesh.attach(TextureFilter(img,tex))
        else:
            scene.visuals.Sphere(radius=1.0,rows=64,cols=128,method='latitude',color=EARTH_COLOR,parent=self.earth)
        self.grid = scene.visuals.Line(pos=graticule(),connect='segments',color=GRID_COLOR,parent=self.earth)
        self.grid.visible = False
        self.gs_km = np.array([sim.to_ecef(src[1],src[2],R_E),sim.to_ecef(dst[1],dst[2],R_E)])
        self.hap_km = self.gs_km*(R_E+sim.H_HAP)/R_E
        gs = self.gs_km/R_E
        m = scene.visuals.Markers(parent=self.earth)
        m.set_data((gs*1.012).astype(np.float32),symbol='star',size=18,face_color=STATION_COLOR,
                   edge_color=(0,0,0,1),edge_width=1)
        m = scene.visuals.Markers(parent=self.earth)
        m.set_data((gs*1.016).astype(np.float32),symbol='diamond',size=10,face_color=HAP_COLOR,
                   edge_color=(0,0,0,1),edge_width=1)
        for name,p in zip(self.names,gs):
            lab = scene.visuals.Text(name,pos=p*1.05,color=STATION_COLOR,font_size=9,anchor_x='left',
                                     anchor_y='center',parent=self.earth)
            lab.set_gl_state('translucent',depth_test=True)

        # reach circles for a satellite at the median altitude
        r_sat = float(np.nanmedian(np.linalg.norm(sat.view_pos(0.0),axis=1)))
        self.alt = r_sat-R_E
        a_gs = reach_angle(R_E,r_sat,GS_MIN_EL)
        a_hap = reach_angle(R_E+sim.H_HAP,r_sat,sim.HAP_MIN_EL)
        self.ring_gs = scene.visuals.Line(pos=np.concatenate([ring(p,a_gs) for p in gs]),connect='segments',
                                          color=GS_LINK,width=2,parent=self.earth)
        self.ring_hap = scene.visuals.Line(pos=np.concatenate([ring(p,a_hap) for p in gs]),connect='segments',
                                           color=HAP_LINK,width=2,parent=self.earth)

        # satellite colors by shell
        shells = SHELLS.get(sat.const,[])
        cols = np.array([s[1] for s in shells]+[OTHER_COLOR],dtype=np.float32)
        if shells:
            d = np.abs(sat.inc[:,None]-np.array([s[0] for s in shells])[None,:])
            self.sat_colors = cols[np.where(d.min(axis=1) <= 2.0,d.argmin(axis=1),len(shells))]
        else:
            self.sat_colors = np.repeat(cols,sat.n,axis=0)
        self.sat_size = SAT_SIZE.get(sat.const,3)
        sc = self.view.scene
        self.dots = scene.visuals.Markers(parent=sc)
        self.gs_links = empty_line(GS_LINK,1.5,sc)
        self.hap_links = empty_line(HAP_LINK,1.5,sc)
        self.isl_lines = empty_line(ISL_COLOR,1.5,sc)
        self.isl_nbr = scene.visuals.Markers(parent=sc)
        self.isl_rep = scene.visuals.Markers(parent=sc)
        self.refs = {}
        for name,color in (('max_eta',MAX_ETA_COLOR),('shortest',SHORTEST_COLOR)):
            self.refs[name] = (empty_line(color,2.0,sc),scene.visuals.Markers(parent=sc),color)
        self.route = empty_line(ROUTE_COLOR,3.0,sc)
        self.route_sats = scene.visuals.Markers(parent=sc)
        self.hide_route()

        self.setup_text()
        self.build_controls()
        self.build_panel()
        self.canvas.events.resize.connect(lambda ev: self.layout())
        self.canvas.events.key_press.connect(self.on_key)
        self.canvas.events.mouse_press.connect(self.on_mouse)
        for ev in (self.canvas.events.mouse_move,self.canvas.events.mouse_press):
            ev.connect(self.on_hover)           # before the camera gets it
        self.canvas.events.mouse_wheel.connect(self.on_wheel)
        self.apply_link_mode()
        self.update_scene()
        self.update_hud()
        self.timer = app.Timer(interval=1.0/FPS,connect=self.on_timer,start=True)

    # ------------------------------------------------------------------ scene
    def update_scene(self):
        pos = self.sat.view_pos(self.t)/R_E
        self.dots.set_data(np.nan_to_num(pos).astype(np.float32),face_color=self.sat_colors,edge_width=0,
                           size=self.sat_size)
        th = self.sat.earth_angle(self.t)
        self.earth_tf.reset()
        self.earth_tf.rotate(np.degrees(th) % 360.0,(0,0,1))
        self.update_links(pos*R_E,th)
        if self.follow is not None:
            self.update_follow(pos*R_E)

    def visibility(self,sats,th,kinds):
        out = {'gs':rotz(self.gs_km,th),'hap':rotz(self.hap_km,th)}
        if not kinds:
            return out
        with np.errstate(invalid='ignore'):
            ll,r = latlon(sats),np.linalg.norm(sats,axis=1)
            if 'gs' in kinds or 'isl' in kinds:
                el = elev_deg(out['gs'],ll,r)
                out['gs_vis'] = el >= GS_MIN_EL
                self.counts['gs'],self.count_t['gs'] = out['gs_vis'].sum(axis=1).tolist(),self.t
            if 'hap' in kinds:
                out['hap_vis'] = elev_deg(out['hap'],ll,r) >= sim.HAP_MIN_EL
                self.counts['hap'],self.count_t['hap'] = out['hap_vis'].sum(axis=1).tolist(),self.t
            if 'isl' in kinds:
                reps = np.nan_to_num(el,nan=-1e9).argmax(axis=1)       # highest satellite above each gs
                r1,r2 = r[reps][:,None],r[None,:]
                g = gamma_to(sats[reps],ll)
                vis = sim.isl_visible(r1,r2,g)
                L = sim.slant_range(r1,r2,g)
                if sim.ISL_MAX_KM is not None:
                    vis &= L <= sim.ISL_MAX_KM
                vis &= np.isfinite(L)
                vis[np.arange(len(reps)),reps] = False
                out['reps'],out['isl_vis'],out['isl_L'] = reps,vis,L
                self.counts['isl'],self.count_t['isl'] = vis.sum(axis=1).tolist(),self.t
        return out

    def update_links(self,sats,th):
        parts = self.link_mode.split('+')
        show_gs = self.show_links and 'GS' in parts
        show_hap = self.show_links and 'HAP' in parts
        kinds = [k for k,on in (('gs',show_gs),('hap',show_hap),('isl',self.show_isl)) if on]
        g = self.visibility(sats,th,kinds)
        self.set_links(self.gs_links,g['gs'],g.get('gs_vis'),GS_LINK,show_gs,sats)
        self.set_links(self.hap_links,g['hap'],g.get('hap_vis'),HAP_LINK,show_hap,sats)
        if not self.show_isl:
            self.isl_lines.visible = self.isl_nbr.visible = self.isl_rep.visible = False
            return
        seg,nbr = [],[]
        for k,rep in enumerate(g['reps']):
            idx = np.nonzero(g['isl_vis'][k])[0]
            near = idx[np.argsort(g['isl_L'][k,idx])]
            nbr.append(near)
            pairs = np.empty((2*len(near),3))
            pairs[0::2],pairs[1::2] = sats[rep],sats[near]
            seg.append(pairs)
        seg,nbr = np.concatenate(seg),np.unique(np.concatenate(nbr))
        if len(seg):
            self.isl_lines.set_data(pos=(seg/R_E).astype(np.float32),connect='segments',color=ISL_COLOR)
        self.isl_lines.visible = len(seg) > 0
        if len(nbr):
            self.isl_nbr.set_data((sats[nbr]/R_E).astype(np.float32),face_color=ISL_COLOR,edge_width=0,
                                  size=self.sat_size+3)
        self.isl_nbr.visible = len(nbr) > 0
        self.isl_rep.set_data((sats[g['reps']]/R_E).astype(np.float32),face_color=(0,0,0,0),edge_color=WHITE,
                              edge_width=2,size=self.sat_size+12)
        self.isl_rep.visible = True

    def set_links(self,visual,origins,vis,color,on,sats):
        if not on or vis is None or not vis.any():
            visual.visible = False
            return
        o,s = np.nonzero(vis)
        seg = np.empty((2*len(o),3),np.float32)
        seg[0::2],seg[1::2] = origins[o]/R_E,sats[s]/R_E
        visual.set_data(pos=seg,connect='segments',color=color)
        visual.visible = True

    def refresh_counts(self):
        missing = [k for k in ('gs','hap','isl') if self.count_t.get(k) != self.t]
        if missing:
            self.visibility(self.sat.view_pos(self.t),self.sat.earth_angle(self.t),missing)

    def mode_color(self):
        return {'GS':GS_LINK,'HAP':HAP_LINK}.get(self.link_mode,WHITE)

    def apply_link_mode(self):
        parts = self.link_mode.split('+')
        self.ring_gs.visible = self.show_rings and 'GS' in parts
        self.ring_hap.visible = self.show_rings and 'HAP' in parts
        self.set_text(self.btn_text,'Click: link mode '+self.link_mode,self.mode_color())

    def next_link_mode(self):
        self.link_mode = LINK_MODES[(LINK_MODES.index(self.link_mode)+1) % len(LINK_MODES)]
        self.apply_link_mode()
        return 'link mode '+self.link_mode

    def on_mouse(self,event):
        if event.pos is None or event.button != 1:
            return
        x,y = event.pos[:2]
        x0,y0,x1,y1 = self.btn
        if x0 <= x <= x1 and y0 <= y <= y1:
            print('[button]',self.next_link_mode())
            self.update_scene()
            self.update_hud()
            self.canvas.update()

    # ------------------------------------------------------------------ route
    def hide_route(self):
        self.route.visible = self.route_sats.visible = False
        self.hide_refs()

    def hide_refs(self):
        for line,marks,_ in self.refs.values():
            line.visible = marks.visible = False

    def same_as_route(self,r,ref):
        return r.served and r.nodes is not None and ref['pos'].shape == r.nodes.shape and np.allclose(ref['pos'],r.nodes)

    def draw_refs(self,r,th):
        self.hide_refs()
        for name,ref in r.refs.items():
            if name not in self.refs or self.same_as_route(r,ref):
                continue
            line,marks,color = self.refs[name]
            v = (rotz(np.vstack([r.gs[0],ref['pos'],r.gs[1]]),th)/R_E).astype(np.float32)
            line.set_data(pos=v,color=color,width=2.0,connect='strip')
            marks.set_data(v[2:-2],face_color=color,edge_width=0,size=7)
            line.visible = marks.visible = True

    def draw_route(self,gs,hap_s,sats,hap_d,th):
        v = (rotz(np.vstack([gs[0],hap_s,sats,hap_d,gs[1]]),th)/R_E).astype(np.float32)
        self.route.set_data(pos=v,color=ROUTE_COLOR,width=3.0,connect='strip')
        self.route_sats.set_data(v[2:-2],face_color=ROUTE_COLOR,edge_color=(0,0,0,1),edge_width=1,size=10)
        self.route.visible = self.route_sats.visible = True

    def update_follow(self,view_km):
        r = self.follow
        th = self.sat.earth_angle(self.t)
        sats = rotz(view_km[r.sats],-th)
        why,eta,el_up,el_dn = sim.check_path(r.hap[0],sats,r.hap[1],r.l_max)
        dt = self.t-r.t
        if not why:
            self.draw_route(r.gs,r.hap[0],sats,r.hap[1],th)
            self.follow_info = 'follow +%s: eta %.3f, elevation %.0f / %.0f deg' % (hms(dt),eta,el_up,el_dn)
        else:
            self.follow = None
            self.hide_route()
            self.follow_info = 'path broke at +%s: %s' % (hms(dt),why[0])
            print('  path of %s UTC broke at +%.0f s: %s' % (r.utc.strftime('%H:%M:%S'),dt,'; '.join(why)))

    # ------------------------------------------------------------------ time
    def on_timer(self,event):
        now = time.perf_counter()
        dt,self.last = now-self.last,now
        if self.mode == 'RUNNING':
            t_new = self.t + dt*self.speed
            while not self.requests_on and self.req.time_of(self.k) <= t_new:
                self.k += 1
            while self.req.time_of(self.k) <= t_new:
                k = self.k
                tk = self.req.time_of(k)
                self.k += 1
                if self.req.decision(k):
                    self.t = tk                 # freeze right at the request
                    self.start_compute()
                    break
                self.log.no_request(self.sat.utc(tk))
                self.msg = '%s UTC: no request' % self.sat.utc(tk).strftime('%H:%M')
            else:
                self.t = t_new
        elif self.mode == 'COMPUTING' and self.job is not None and self.job.done():
            self.finish_compute()
        self.update_scene()
        self.fps_n += 1
        if now-self.fps_t0 >= 0.5:
            self.fps = self.fps_n/(now-self.fps_t0)
            self.fps_n,self.fps_t0 = 0,now
            self.update_hud()
        self.canvas.update()

    def start_compute(self):
        self.mode = 'COMPUTING'
        self.follow,self.follow_info = None,''
        self.hide_route()
        self.job_t0 = time.perf_counter()
        self.sat.view_pos(self.t)               # fill the cache first so both threads do not propagate together
        print('%s UTC: request %s -> %s, computing the light path ...' % (
            self.sat.utc(self.t).strftime('%Y-%m-%d %H:%M:%S'),self.src[0],self.dst[0]))
        self.job = self.pool.submit(sim.compute_route,self.sat,self.t,self.src,self.dst,self.f_min)
        self.update_hud()

    def finish_compute(self):
        try:
            r = self.job.result()
        except Exception as e:
            print('  route computation failed:',repr(e))
            self.res,self.mode,self.job = None,'SHOWING',None
            self.msg = 'route computation failed'
            self.update_hud()
            return
        self.job = None
        self.res = r
        self.log.route(r)
        print('  '+sim.describe(r))
        th = self.sat.earth_angle(r.t)
        self.draw_refs(r,th)
        if r.served:
            self.draw_route(r.gs,r.hap[0],r.nodes[1:-1],r.hap[1],th)
        self.mode = 'SHOWING'
        self.update_hud()

    def resume(self):
        self.hide_refs()
        if self.follow_on and self.res is not None and self.res.served:
            self.follow = self.res
        else:
            self.hide_route()
        self.mode = 'RUNNING'
        self.last = time.perf_counter()

    # ------------------------------------------------------------------ keys
    def on_key(self,event):
        name = ((event.key.name if event.key is not None else '') or '').upper()
        text = (event.text or '').lower()
        msg = None
        if name in ('ENTER','RETURN') or text in ('\r','\n'):
            if self.mode == 'SHOWING':
                self.resume()
                msg = 'resumed'
        elif name == 'R' or text == 'r':
            self.requests_on = not self.requests_on
            if self.requests_on:
                t1 = self.sat.utc(self.req.time_of(self.k))
                msg = 'requests ON, first check %s UTC' % t1.strftime('%Y-%m-%d %H:%M')
                self.msg = 'requests ON, first check %s UTC' % t1.strftime('%H:%M')
            else:
                msg = 'requests OFF'
                self.msg = 'requests OFF: satellites only'
        elif name == 'N' or text == 'n':
            if not self.requests_on:
                msg = 'N ignored: requests are OFF (press R first)'
                self.msg = 'requests OFF: press R before N'
            elif self.mode in ('RUNNING','PAUSED','SHOWING'):
                k = self.req.next_request(self.k)
                self.follow = None
                self.hide_route()
                self.t = max(self.t,self.req.time_of(k)-1e-6)
                self.mode = 'RUNNING'
                self.last = time.perf_counter()
                msg = 'jump to the request at %s UTC' % self.sat.utc(self.req.time_of(k)).strftime('%Y-%m-%d %H:%M')
        elif name == 'SPACE':
            if self.mode in ('RUNNING','PAUSED'):
                self.mode = 'PAUSED' if self.mode == 'RUNNING' else 'RUNNING'
                self.last = time.perf_counter()
                msg = self.mode.lower()
        elif name in ('+','=','PLUS','EQUAL','UP') or text in ('+','='):
            bigger = [v for v in SPEEDS if v > self.speed]
            self.speed = bigger[0] if bigger else self.speed
            msg = 'speed x%g' % self.speed
        elif name in ('-','_','MINUS','DOWN') or text in ('-','_'):
            smaller = [v for v in SPEEDS if v < self.speed]
            self.speed = smaller[-1] if smaller else self.speed
            msg = 'speed x%g' % self.speed
        elif name == 'P' or text == 'p':
            self.follow_on = not self.follow_on
            if not self.follow_on and self.follow is not None:
                self.follow = None
                self.hide_route()
            msg = 'path follow '+onoff(self.follow_on)[0]
        elif name == 'G' or text == 'g':
            self.grid.visible = not self.grid.visible
            msg = 'grid '+onoff(self.grid.visible)[0]
        elif name == 'L' or text == 'l':
            self.show_links = not self.show_links
            msg = 'satellite links '+onoff(self.show_links)[0]
        elif name == 'H' or text == 'h':
            msg = self.next_link_mode()
        elif name == 'C' or text == 'c':
            self.show_rings = not self.show_rings
            self.apply_link_mode()
            msg = 'reach circles '+onoff(self.show_rings)[0]
        elif name == 'I' or text == 'i':
            self.show_isl = not self.show_isl
            msg = 'inter-satellite links '+onoff(self.show_isl)[0]
        if msg:
            print('[key]',msg)
        self.update_hud()
        self.canvas.update()

    # ------------------------------------------------------------------ text panels
    def setup_text(self):
        self.face,mono = 'OpenSans',False
        try:
            from vispy.util.fonts import list_fonts
            have = {f.lower():f for f in list_fonts()}
            for f in FACES:
                if f.lower() in have:
                    self.face,mono = have[f.lower()],True
                    break
        except:
            pass
        dpi = float(getattr(self.canvas,'dpi',0) or 96.0)
        self.em = FONT*dpi/72.0
        self.cw = self.em*(0.60 if mono else 0.70)
        self.lh = self.em*1.8
        self.margin = 1.2*self.em

    def text(self,pos,color=WHITE,anchor='left',s=''):
        return scene.visuals.Text(s,parent=self.canvas.scene,color=color,font_size=FONT,face=self.face,
                                  anchor_x=anchor,anchor_y='center',pos=pos)

    def set_text(self,t,s,color=None):
        if t.text != s:
            t.text = s
        if color is not None and self.colors.get(id(t)) != tuple(color):
            t.color = color
            self.colors[id(t)] = tuple(color)

    def build_controls(self):
        x0,y0 = self.margin,self.margin+self.lh/2
        run = {'RUNNING':('RUNNING',ON),'PAUSED':('PAUSED',OFF)}
        rows = [
            ('[R]     Requests',WHITE,lambda: onoff(self.requests_on)),
            ('[Enter] Resume after a route',WHITE,None),
            ('[N]     Jump to next request',WHITE,None),
            ('[Space] Pause / run',WHITE,lambda: run.get(self.mode,('FROZEN',WARN))),
            ('[+/-]   Speed',WHITE,lambda: ('x%g' % self.speed,WHITE)),
            ('[P]     Path follow',WHITE,lambda: onoff(self.follow_on)),
            ('[G]     Grid',WHITE,lambda: onoff(self.grid.visible)),
            ('[L]     Satellite links',WHITE,lambda: onoff(self.show_links)),
            ('[H]     Link mode',WHITE,lambda: (self.link_mode,self.mode_color())),
            ('[C]     Reach circles',WHITE,lambda: onoff(self.show_rings)),
            ('[I]     Inter-sat links',WHITE,lambda: onoff(self.show_isl)),
            ('Mouse: drag = rotate, wheel = zoom',DIM,None),
            ('[F11] Full screen   [Esc] Quit',DIM,None),
        ]
        xs = x0 + (max(len(l) for l,_,f in rows if f is not None)+3)*self.cw
        self.text((x0,y0),DIM,s='CONTROLS')
        self.ctrl = []
        for i,(label,color,f) in enumerate(rows):
            y = y0+(i+1)*self.lh
            self.text((x0,y),color,s=label)
            if f is not None:
                self.ctrl.append((self.text((xs,y)),f))
        bw,bh = (len('Click: link mode GS+HAP')+4)*self.cw,1.8*self.em
        by = y0+(len(rows)+0.8)*self.lh
        self.btn = (x0,by,x0+bw,by+bh)
        scene.visuals.Rectangle(center=(x0+bw/2,by+bh/2),width=bw,height=bh,color=(0.15,0.2,0.3,0.9),
                                border_color=(0.6,0.75,1.0,1.0),parent=self.canvas.scene)
        self.btn_text = self.text((x0+bw/2,by+bh/2),anchor='center')

    def build_panel(self):
        # right side: status lines, then two framed blocks the mouse wheel can scroll
        self.top = [self.text((0,0)) for i in range(5)]
        self.name_chars = max(len('Station'),*(len(n) for n in self.names))
        self.blocks = {}
        for key,title,ncol in (('links','SATELLITE LINKS',4),('solution','SOLUTION',1)):
            anchors = ['left']+['right']*(ncol-1)
            self.blocks[key] = {
                'title':self.text((0,0),DIM,s=title),
                'hint':self.text((0,0),DIM,anchor='right'),
                'frame':scene.visuals.Rectangle(center=(1,1),width=1,height=1,color=(0,0,0,0),border_color=FRAME,
                                                parent=self.canvas.scene),
                'thumb':scene.visuals.Rectangle(center=(1,1),width=1,height=1,color=DIM,parent=self.canvas.scene),
                'slots':[[self.text((0,0),anchor=a) for a in anchors] for i in range(MAX_LINES)],
                'lines':[],'off':0,'n':1,'rect':(0,0,0,0)}
        self.layout()

    def layout(self):
        cw,lh,W,H = self.cw,self.lh,self.canvas.size[0],self.canvas.size[1]
        x0,x1 = W-self.margin-PANEL_CHARS*cw,W-self.margin
        pad = 0.6*self.em
        self.panel = (x0-pad,0,W,H)
        y = self.margin+lh/2
        for t in self.top:
            t.pos = (x0,y)
            y += lh
        y += 0.5*lh
        for key in ('links','solution'):
            b = self.blocks[key]
            b['title'].pos,b['hint'].pos = (x0,y),(x1,y)
            top = y+0.6*lh
            n = LINK_LINES if key == 'links' else int((H-self.margin-top-2*pad)//lh)
            n = int(np.clip(n,3,MAX_LINES))
            h = n*lh+2*pad
            b['n'],b['rect'] = n,(x0-pad,top,x1+pad,top+h)
            b['frame'].center = (x0-pad+(x1-x0+2*pad)/2,top+h/2)
            b['frame'].width,b['frame'].height = x1-x0+2*pad,h
            xg = x0+(self.name_chars+6)*cw
            for i,slot in enumerate(b['slots']):
                yy = top+pad+(i+0.5)*lh
                xs = [x0] if len(slot) == 1 else [x0,xg,xg+7*cw,xg+15*cw]
                for t,x in zip(slot,xs):
                    t.pos = (x,yy)
                    t.visible = i < n
            y = top+h+0.9*lh
        self.render_blocks()

    def render_blocks(self):
        width = PANEL_CHARS-2
        for b in self.blocks.values():
            lines,n = b['lines'],b['n']
            b['off'] = int(np.clip(b['off'],0,max(0,len(lines)-n)))
            off = b['off']
            for i,slot in enumerate(b['slots'][:n]):
                item = lines[off+i] if off+i < len(lines) else ('',None)
                if len(slot) == 1:
                    self.set_text(slot[0],item[0][:width],item[1])
                elif isinstance(item[0],tuple):           # table row
                    for t,(s,c) in zip(slot,item):
                        self.set_text(t,s,c)
                else:
                    self.set_text(slot[0],item[0][:width],item[1])
                    for t in slot[1:]:
                        self.set_text(t,'')
            over = len(lines) > n
            xa,ya,xb,yb = b['rect']
            if over:
                track = (yb-ya)-4
                h = max(track*n/len(lines),6)
                b['thumb'].center = (xb-4,ya+2+(track-h)*off/(len(lines)-n)+h/2)
                b['thumb'].width,b['thumb'].height = 3,h
            b['thumb'].visible = over
            self.set_text(b['hint'],'%d-%d of %d, wheel' % (off+1,min(off+n,len(lines)),len(lines)) if over else '')

    def block_at(self,pos):
        x,y = pos[:2]
        for key,b in self.blocks.items():
            xa,ya,xb,yb = b['rect']
            if xa <= x <= xb and ya <= y <= yb:
                return key
        return None

    def on_hover(self,event):
        if event.pos is None:
            return
        over = self.panel[0] <= event.pos[0] <= self.panel[2] and self.block_at(event.pos) is not None
        self.view.camera.interactive = not over

    def on_wheel(self,event):
        self.on_hover(event)
        key = self.block_at(event.pos) if event.pos is not None else None
        if key is None:
            return
        self.blocks[key]['off'] -= int(np.sign(event.delta[1]))*2
        self.render_blocks()
        self.canvas.update()

    def update_hud(self):
        self.refresh_counts()
        for t,f in self.ctrl:
            self.set_text(t,*f())
        self.set_text(self.btn_text,'Click: link mode '+self.link_mode,self.mode_color())

        sat,log = self.sat,self.log
        t_next = self.req.time_of(self.k)
        top = [('STATUS',DIM),
               ('UTC %s   x%g' % (sat.utc(self.t).strftime('%Y-%m-%d %H:%M:%S'),self.speed),WHITE),
               ('FPS %5.1f   %s %d sats' % (self.fps,sat.label,sat.n),ON if self.fps >= 0.9*FPS else WARN),
               ('Next check %s UTC, in %s' % (sat.utc(t_next).strftime('%H:%M:%S'),hms(t_next-self.t)),WHITE)
               if self.requests_on else ('Requests OFF, press R to start',OFF),
               ('Checks %d  requests %d  served %d' % (log.n_checks,log.n_req,log.n_served),WHITE)]
        for t,(s,c) in zip(self.top,top):
            self.set_text(t,s[:PANEL_CHARS],c)

        links = [(('Station',DIM),('GS',GS_LINK),('HAP',HAP_LINK),('ISL',ISL_COLOR))]
        for i,name in enumerate(self.names):
            links.append(((name,WHITE),(str(self.counts['gs'][i]),GS_LINK),(str(self.counts['hap'][i]),HAP_LINK),
                          (str(self.counts['isl'][i]),ISL_COLOR)))
        links += [('',None),
                  ('GS: satellites above %.0f deg elevation' % GS_MIN_EL,DIM),
                  ('HAP: satellites above %.0f deg elevation' % sim.HAP_MIN_EL,DIM),
                  ('ISL: neighbors of the ringed satellite,',DIM),
                  ('     line of sight above %.0f km altitude' % sim.H_ATM,DIM),
                  ('Circles: satellites at %.0f km altitude' % self.alt,DIM)]
        self.blocks['links']['lines'] = links

        sol = []
        if self.mode == 'COMPUTING':
            sol.append(('FROZEN: computing route %4.1f s' % (time.perf_counter()-self.job_t0),WARN))
        elif self.mode == 'SHOWING':
            ok = self.res is not None and self.res.served
            sol.append(('FROZEN: route shown, press Enter' if ok else 'FROZEN: no route, press Enter',ON if ok else OFF))
        else:
            sol.append((self.msg or ('waiting for the next request' if self.requests_on
                                     else 'requests OFF: satellites only'),DIM))
        if self.follow_info:
            sol.append((self.follow_info,ON if self.follow is not None else WARN))
        sol += [('',None),('%s -> %s' % (self.src[0],self.dst[0]),DIM)]
        r = self.res
        if r is None:
            sol.append(('no request yet',DIM))
        else:
            sol += [('time        %s UTC' % r.utc.strftime('%Y-%m-%d %H:%M:%S'),WHITE),
                    ('result      '+r.status,ON if r.served else OFF)]
            if r.served:
                sol += [('satellites  %d' % len(r.sats),WHITE),
                        ('length      %.0f km, limit %.0f km' % (r.length,r.l_max),WHITE),
                        ('eta         %.3f' % r.eta,WHITE),
                        ('fidelity    %.4f, required %g' % (r.f_e2e,r.f_min),WHITE),
                        ('Pi          %.3f' % r.pi,WHITE),
                        ('EDR         %.2f ebits/s' % r.edr,WHITE),
                        ('selected    '+r.selected,WHITE),
                        ('            best EDR of %d feasible paths' % r.n_cand,DIM)]
            else:
                sol += [(r.note,WHITE),('limit       %.0f km' % r.l_max,WHITE),('fidelity    required %g' % r.f_min,WHITE)]
            sol.append(('computed in %.1f s' % r.compute_s,DIM))
            for name,label,color in (('max_eta','Dijkstra, max eta',MAX_ETA_COLOR),
                                     ('shortest','Dijkstra, shortest',SHORTEST_COLOR)):
                ref = r.refs.get(name)
                if ref is None:
                    continue
                tag = 'same as result' if self.same_as_route(r,ref) else ref['status']
                sol += [('',None),(label,color),
                        ('  %d sats, %.0f km, eta %.3f' % (ref['n_sats'],ref['length'],ref['eta']),color),
                        ('  fidelity %.4f, %s' % (ref['f_e2e'],tag),color)]
                if ref['status'] == 'within limit':
                    sol.append(('  EDR %.2f ebits/s' % ref['edr'],color))
            if r.served:
                sol += [('',None),('relay satellites',DIM)]
                sol += [('  %2d  %s' % (i+1,name),ROUTE_COLOR) for i,name in enumerate(r.sat_names)]
        if self.res is not self.shown:              # new result, back to the top
            self.blocks['solution']['off'] = 0
            self.shown = self.res
        self.blocks['solution']['lines'] = sol
        self.render_blocks()


def run(sat,src,dst,f_min,speed):
    req = sim.requests(sat.start)
    log = sim.request_log(sat.const,src,dst,sat.start)
    viewer(sat,req,log,src,dst,f_min,speed)
    app.run()
