import numpy as np
import re
import matplotlib as mpl
from collections import OrderedDict
from flame_utils import ModelFlame
from flame_utils import collect_data
from scipy.interpolate import interp1d
from scipy.special import expit

n_kick_def = 0
common_enge = np.array([0.296471, 4.533219, -2.270982, 1.068627, -0.036391, 0.022261])

class AdvQuad(object):
    def __init__(self, fm, qe, param, d1=None, d2=None, n_kick=None, default_enge=False):
        self.fm = fm
        self.qe = qe
        self.default_enge = default_enge
        if n_kick is None:
            n_kick = n_kick_def
        self.n_kick = n_kick
        c = fm.machine.conf
        qei = fm.find(qe)
        qc0 = c(qei[0])

        if n_kick > 0:
            self.ot = qe + '_KICK'
        else:
            self.ot = None

        if len(qei) == 1:
            for i in range(n_kick):
                fm.insert_element(qei[-1]+1, {'name':self.ot, 'type': 'orbtrim'})
                fm.insert_element(qei[-1]+2, c(qei[0]))
                qei = fm.find(qe)
            fm.reconfigure(self.qe, {'L': qc0['L']/(n_kick+1)})

        c = fm.machine.conf

        if d1 is not None:
            self.d1 = d1
        else:
            self.d1 = c(qei[0]-1)['name']

        if fm.machine.conf(fm.find(self.d1)[0])['type'] != 'drift':
            raise TypeError('upstream element {} must be a drift'.format(self.d1))

        if d2 is not None:
            self.d2 = d2
        else:
            self.d2 = c(qei[-1]+1)['name']

        if fm.machine.conf(fm.find(self.d2)[0])['type'] != 'drift':
            raise TypeError('downstream element {} must be a drift'.format(self.d2))

        self.qen = len(qei)
        self.update_length()
        self.load_param(param)

    def update_length(self):
        c = self.fm.machine.conf
        self.qel = c(self.fm.find(self.qe)[0])['L']
        self.d1l = c(self.fm.find(self.d1)[0])['L']
        self.d2l = c(self.fm.find(self.d2)[0])['L']

    def load_param(self, param):
        with open(param, 'r') as f:
            ls = f.readlines()
            hls = [l.split(',')[0].replace('\n', '') for l in ls]
            ri = hls.index('M5_PARAM_TABLE_V01')
            rr = eval(hls[ri+1])
            s = hls.index('START table')
            e = hls.index('END')
            r = np.array([ls[i].split(',') for i in range(s+1, e)], dtype=np.float32)
        self.rr = rr

        i2 = r[:, 0]
        g2 = r[:, 1]
        b2 = r[:, 2]
        l2 = r[:, 3]
        self.f_ig = interp1d(i2, g2, fill_value='extrapolate')
        self.f_ib = interp1d(i2, b2, fill_value='extrapolate')
        self.f_il = interp1d(i2, l2, fill_value='extrapolate')

        self.f_gi = interp1d(g2, i2, fill_value='extrapolate')
        self.f_gb = interp1d(g2, b2, fill_value='extrapolate')
        self.f_gl = interp1d(g2, l2, fill_value='extrapolate')

        self.f_bi = interp1d(b2, i2, fill_value='extrapolate')
        self.f_bg = interp1d(b2, g2, fill_value='extrapolate')
        self.f_bl = interp1d(b2, l2, fill_value='extrapolate')

        self.engec0 = []
        self.engec1 = []
        if isinstance(self.default_enge, bool):
            if self.default_enge:
                self.engec0 = [lambda x, y=v: y for v in common_enge]
                self.engec1 = [lambda x, y=v: y for v in common_enge]
            else:
                for i in range(4, r.shape[1], 2):
                    if all(r[:, i] == 0.0) and i != 4:
                        break
                    self.engec0.append(interp1d(i2, r[:, i  ], fill_value='extrapolate'))
                    self.engec1.append(interp1d(i2, r[:, i+1], fill_value='extrapolate'))
        elif isinstance(self.default_enge, (list, tuple, np.ndarray)):
            self.engec0 = [lambda x, y=v: y for v in self.default_enge]
            self.engec1 = [lambda x, y=v: y for v in self.default_enge]

    def convert(self, i2=None, g2=None, b2=None):
        if i2 is not None:
            l2 = self.f_il(np.abs(i2))
            g2 = self.f_ig(np.abs(i2))*np.sign(i2)
            b2 = self.f_ib(np.abs(i2))*np.sign(i2)
        elif g2 is not None:
            l2 = self.f_gl(np.abs(g2))
            i2 = self.f_gi(np.abs(g2))*np.sign(g2)
            b2 = self.f_gb(np.abs(g2))*np.sign(g2)
        elif b2 is not None:
            l2 = self.f_bl(np.abs(b2))
            i2 = self.f_bi(np.abs(b2))*np.sign(b2)
            g2 = self.f_bg(np.abs(b2))*np.sign(b2)
        else:
            return None

        return [i2, g2, b2, l2]

    def reconf(self, i2=None, g2=None, b2=None, scl=1.0):
        ret = self.convert(i2, g2, b2)

        if ret is None:
            return None
        else:
            i2, g2, b2, l2 = ret

        self.update_length()

        self.fm.reconfigure(self.qe, {'B2': float(g2)*scl,
                                       'G': float(g2),
                                       'B': float(b2),
                                       'I': float(i2),
                                       'L': float(l2)/self.qen,
                                       'L_total': float(l2),
                                       'aper': self.rr})

        ldif = 0.5*(l2 - self.qel*float(self.qen))
        self.fm.reconfigure(self.d1, {'L': float(self.d1l - ldif)})
        self.fm.reconfigure(self.d2, {'L': float(self.d2l - ldif)})

    def kick(self, tm_xkick=None, tm_ykick=None, theta_x=None, theta_y=None):
        d = {}
        if tm_xkick is not None:
            d['tm_xkick_total'] = tm_xkick
            d['tm_xkick'] = tm_xkick/self.n_kick
            d['realpara'] = 1
        if tm_ykick is not None:
            d['tm_ykick_total'] = tm_ykick
            d['tm_ykick'] = tm_ykick/self.n_kick
            d['realpara'] = 1
        if theta_x is not None:
            d['theta_x_total'] = theta_x
            d['theta_x'] = theta_x/self.n_kick
            d['realpara'] = 0
        if theta_y is not None:
            d['theta_y_total'] = theta_y
            d['theta_y'] = theta_y/self.n_kick
            d['realpara'] = 0

        self.fm.reconfigure(self.ot, d)

    def reconfigure(self, key, kicker=False):
        if not kicker:
            self.fm.reconfigure(self.qe, key)
        elif self.ot is not None:
            self.fm.reconfigure(self.ot, key)

    def conf(self, key=None, kicker=False):
        if not kicker:
            c = self.fm.machine.conf(self.fm.find(self.qe)[0])
        elif self.ot is not None:
            c = self.fm.machine.conf(self.fm.find(self.ot)[0])
        else:
            return None

        if key is None:
            return c
        else:
            return c[key]

class Combine(object):
    def __init__(self, qls, dl=0.005, enge_type=1):
        self.qls = qls
        self.dl = dl
        self.elems = []
        self.qname = []
        self.total = 0.0
        self.qcen = []
        self.fm = qls[0].fm
        self.enge_type = enge_type

        for q in qls:
            if q.fm != self.fm:
                raise ValueError('Quadrupoles must be refer to the same ModelFlame.')

            if not isinstance(q, AdvQuad):
                raise TypeError('{} is not AdvQuad. All list elements must be an AdvQuad.'.format(q.qe))

            if not q.fm.conf(q.d1)[0]['type'] == 'drift':
                raise TypeError('{} is already combined with other AdvQuad.')

            if not q.fm.conf(q.d2)[0]['type'] == 'drift':
                raise TypeError('{} is already combined with other AdvQuad.')

        for i, q in enumerate(qls):
            if i > 0 and (qls[i-1].d2 != q.d1):
                raise TypeError('Quadrupoles {} and {} must be connected by the drift {}'.format(
                    qls[i-1].qe, q.qe, q.d1
                ))

            if i == 0:
                md1 = q.fm.get_element(q.d1)
                if len(md1) != 1:
                    raise TypeError('{} is separated to multiple part.'.format(q.d1))
                else:
                    md1 = md1[0]
                    q.fm.pop_element(md1['index'])
                    md1['properties']['type'] = 'quadrupole'
                    md1['properties']['B2'] = 0
                    q.fm.insert_element(md1['index'], md1['properties'])
                    self.elems += q.fm.find(q.d1)
                    self.total += md1['properties']['L']

            self.qname.append(q.qe)
            self.elems += q.fm.find(q.qe)
            qlen = q.fm.conf(q.qe)[0]['L']*(q.n_kick+1)
            self.total += qlen
            self.qcen.append(self.total - qlen/2.0)

            md2 = q.fm.get_element(q.d2)
            if len(md2) != 1:
                raise TypeError('{} is separated to multiple part.'.format(q.d2))
            else:
                md2 = md2[0]
                q.fm.pop_element(md2['index'])
                md2['properties']['type'] = 'quadrupole'
                md2['properties']['B2'] = 0
                q.fm.insert_element(md2['index'], md2['properties'])
                self.elems += q.fm.find(q.d2)
                self.total += md2['properties']['L']


        self.default_g2 = [q.conf('B2') for q in self.qls]
        self.reconf_all(g2=self.default_g2)

    def reset_field(self):
        self.reconf_all(g2=self.default_g2)

    def reconf_all(self, i2=None, g2=None, b2=None):
        if i2 is not None:
            if len(i2) != len(self.qls):
                raise ValueError('Input array size does not match.')
            val = np.array([q.convert(i2=v) for q, v in zip(self.qls, i2)])
        elif g2 is not None:
            if len(g2) != len(self.qls):
                raise ValueError('Input array size does not match.')
            val = np.array([q.convert(g2=v) for q, v in zip(self.qls, g2)])
        elif b2 is not None:
            if len(b2) != len(self.qls):
                raise ValueError('Input array size does not match.')
            val = np.array([q.convert(b2=v) for q, v in zip(self.qls, b2)])
        else:
            return None

        i2s, g2s, b2s, l2s = val.transpose()

        length = []
        zpos = 0.0
        for q, v1, v2 in zip(self.qls, self.qcen, l2s):
            length.append(v1 - v2*0.5 - zpos)
            length += [v2/(q.n_kick+1)]*(q.n_kick+1)
            zpos = v1 + v2*0.5
        length.append(self.total-zpos)
        length = np.asarray(length)
        nstep = np.floor(length/self.dl).astype('int64') + 1

        zls = []
        zpos = 0.0
        for l, n in zip(length, nstep):
            zls.append(np.linspace(l/n*0.5, l - l/n*0.5, n)+zpos)
            zpos += l
        zls = np.concatenate(zls)

        field = [] #0.0
        for qc, q, l, i2t in zip(self.qcen, self.qls, l2s, i2s):
            coef0 = np.array([f(i2t) for f in q.engec0])
            coef1 = np.array([f(i2t) for f in q.engec1])
            d = q.rr*2
            field.append(enge(zls - qc, l, d, coef0, coef1, self.enge_type))

        idx = 0
        for i, j in enumerate(self.elems):
            prop = {
                'L': length[i],
                'ncurve': len(field)
            }
            for k, (g2t, f) in enumerate(zip(g2s, field)):
                prop['curve{}'.format(k)] = f[idx:idx+nstep[i]]
                prop['scl_fac{}'.format(k)] = g2t

            idx += nstep[i]
            self.fm.reconfigure(j, prop)

        for i, q in enumerate(self.qls):
            q.reconfigure({
                'B2': float(g2s[i]),
                'G': float(g2s[i]),
                'B': float(b2s[i]),
                'I': float(i2s[i]),
            })

        self.zls = zls
        self.field = np.array(field)*np.transpose([g2s])

    def reconf(self, i, i2=None, g2=None, b2=None):
        if i > len(self.qls):
            return None

        if i2 is not None:
            val = self.qls[i].convert(i2=i2)
        elif g2 is not None:
            val = self.qls[i].convert(g2=g2)
        elif b2 is not None:
            val = self.qls[i].convert(b2=b2)
        else:
            return None

        g2_new = [q.conf('B2') for q in self.qls]
        g2_new[i] = val[1]

        self.reconf_all(g2 = g2_new)

    def conf(self, key):
        return np.array([q.conf(key) for q in self.qls])

def enge0(z, d, coef):
    v = 0.0
    for i, c in enumerate(coef):
        v += c*(z/d)**i
    #return (1.0/(1.0 + np.exp(v)))
    return expit(-v)

def enge(zls, l, d, coef0, coef1, enge_type):
    zli =  zls - l*0.5
    zle = -zls - l*0.5
    if enge_type == 0:
        fz  = enge0(zli, d, coef0) + enge0(zle, d, coef1) - 1.0
    elif enge_type == 1:
        fz  = enge0(zli, d, coef0) * enge0(zle, d, coef1)
    return fz * (zle<2*l) * (zli<2*l)

def get_dsp(r):

    m = np.eye(7)
    d = []
    dp = []

    for i in range(len(r)):
        m = np.dot(r[i][1].transfer_matrix[:,:,0],m)
        #   dispersion in mm
        d.append(m[0,5]*(1 + 1/r[i][1].ref_gamma)*
                    r[i][1].ref_IonEk*1e-6)
        #   change in dispersion in rad
        dp.append(m[1,5]*(1 + 1/r[i][1].ref_gamma)*
                         r[i][1].ref_IonEk*1e-6)

    return np.array(d), np.array(dp)

def tws2cov(alpha, beta, eps):
    """Function to convert Twiss parameters to Sigma-matrix (covariance)
    """
    mat = np.zeros([2,2])
    mat[0,0] = beta*eps
    mat[0,1] = mat[1,0] = -alpha*eps
    mat[1,1] = (1.0 + alpha*alpha)/beta*eps
    return mat


def ellipse(cen, cov, facecolor='none', **kws):
    """Calculate eigenequation to transform the ellipse
    """
    v, w = np.linalg.eigh(cov)
    u = w[0]/np.linalg.norm(w[0])
    ang = np.arctan2(u[1], u[0])*180.0/np.pi
    v = 2.0*np.sqrt(v)
    ell = mpl.patches.Ellipse(cen, v[0], v[1], 180+ang, facecolor=facecolor, **kws)
    return ell

def cov2tws(cov):
    eps = np.sqrt(np.linalg.det(cov))
    beta = cov[0, 0]/eps
    alpha = -cov[1, 0]/eps
    return alpha, beta, eps

def load_cosy(fname, scale=True):
    with open(fname, 'r') as f:
        ls = f.readlines()
    ret = []
    ret2 = []
    flg = False
    for i, l in enumerate(ls):
        if 'RPR_' in l:
            brho = eval(ls[i+1]) if scale else 1.0
            flg = True
        if 'END' in l:
            break
        if flg and re.search('^[A-Z]', l) is not None\
               and not '#' in l\
               and not 'FSD' in l\
               and not 'THD1' in l:
            ret.append(eval(ls[i+1])/brho)
            try:
                ret2.append(eval(ls[i+2])/brho)
            except:
                pass
    return np.array(ret), np.array(ret2)

def load_saveset(fname, header='Q_D'):
    with open(fname, 'r') as f:
        ls = f.readlines()

    brho = OrderedDict()
    quads = []
    flg1 = False
    flg2 = False
    for lt in ls:
        l = lt.split()
        if flg1:
            if len(l) > 1:
                key = l[0][0:5]
                if re.match('BTS[0-9][0-9]', key) is not None:
                    brho[l[0][0:5]] = eval(l[1])
                else:
                    flg1 = False
            else:
                flg1 = False
        if len(l) > 1 and l[0] == 'Section':
            flg1 = True

        if flg2:
            if '--' in lt:
                name = lt.replace('F1S1', 'BTS01')
                for key in ['F1S2', 'F2S1', 'F2S2', 'F3S1', 'F3S2', ' ', '-', '\n']:
                    name = name.replace(key, '')
                if name in brho.keys():
                    brho0 = brho[name]
            else:
                quad = OrderedDict()
                l = lt.split()
                if header in l[0]:
                    quad['Sec'] = name
                    quad['Name'] = l[0]
                    quad['I'] = eval(l[6])
                    quad['B'] = eval(l[2])
                    quad['Ratio'] = eval(l[4])
                    quad['Brho'] = brho0
                    quads.append(quad)

        if len(l) > 1 and l[0] == 'Name':
            flg2 = True
    return quads

#interpolate envelope
def env_interp(d):
    cx = []
    cy = []
    zr = []
    for i in range(1, len(d['pos'])):
        #dz = z-d['pos'][i-1] != 0.0:
        z0 = d['pos'][i-1]
        z1 = d['pos'][i]
        if z1-z0 != 0.0:
            z = z1-z0
            z2 = z*z
            z3 = z2*z

            x0 = d['xtwsb'][i-1]
            a0 = d['xtwsa'][i-1]*-2
            y0 = d['ytwsb'][i-1]
            b0 = d['ytwsa'][i-1]*-2

            x1 = d['xtwsb'][i]
            a1 = d['xtwsa'][i]*-2
            y1 = d['ytwsb'][i]
            b1 = d['ytwsa'][i]*-2

            g = np.array([[z2, z3], [2.0*z, 3.0*z2]])
            tx = np.array([x1-x0-a0*z, a1-a0])
            mx = np.linalg.solve(g, tx)
            ty = np.array([y1-y0-b0*z, b1-b0])
            my = np.linalg.solve(g, ty)

            cx.append([x0, a0, mx[0], mx[1]])
            cy.append([y0, b0, my[0], my[1]])
            zr.append([z0, z1])
    return np.asarray(cx), np.asarray(cy), np.asarray(zr)

def env_curve(zi, cx, cy, zr):
    rx = []
    ry = []
    for i, z in enumerate(zi):
        j = len(zr)-1 if z >= zr[-1, 1] else len(zr[zr[:, 1] <= z])
        zt = z - zr[j, 0]
        zt2 = zt*zt
        zt3 = zt2*zt
        vz = np.array([1.0, zt, zt2, zt3])
        rx.append(np.sum(cx[j]*vz))
        ry.append(np.sum(cy[j]*vz))
    return np.asarray(rx), np.asarray(ry)

def env_smooth(zl, r):
    """Cubic interplation of envelope by using twiss beta + beta'
    """
    d = collect_data(r, 'pos', 'xtwsb', 'xtwsa', 'ytwsb', 'ytwsa', 'xeps', 'yeps')
    cx, cy, zr = env_interp(d)
    rx, ry = env_curve(zl, cx, cy, zr)
    rx = np.sqrt(rx*np.interp(zl, d['pos'], d['xeps']))
    ry = np.sqrt(ry*np.interp(zl, d['pos'], d['yeps']))
    return rx, ry


def calc_TPM(r):
    """Calculate transfer matrix with TRANSPORT unit: [cm, mrad, cm, mrad, cm, dp/p%]
    """

    tm = np.eye(6)
    for idx in range(1, len(r)):
        s = r[idx][1]
        L = r[idx][1].pos - r[idx-1][1].pos
        # conv to [cm, mrad, cm, mrad, cm, dp/p%]
        c0 = 299792458
        wavel = c0/s.ref_SampleFreq*100/np.pi/2.0
        cv4 = -1/wavel/s.ref_beta
        cv5 = (1 + 1/s.ref_gamma)*s.ref_IonEk*1e-6/100
        conv = np.array([10, 1e-3, 10, 1e-3, cv4, cv5])
        conv = conv*np.transpose([1/conv])
        mat = s.transfer_matrix[:6,:6,0]*conv
        mat[4, 5] = mat[4, 5] + (L/wavel/s.ref_IonEs/s.ref_bg**3)/cv4*cv5*1e8
        tm = np.dot(mat, tm)
    return tm

def uplim(val, lim):
    if val > lim:
        return val-lim
    else:
        return 0.0