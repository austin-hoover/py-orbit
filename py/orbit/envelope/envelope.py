"""
This module contains functions related to the envelope parameterization of the
Danilov distribution.
"""

import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange

# PyORBIT modules
from bunch import Bunch
from orbit.analysis import covmat2vec
from orbit.utils.helper_funcs import initialize_bunch


def rotation_matrix(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])
    
def phase_adv_matrix(phi1, phi2):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(phi1)
    R[2:, 2:] = rotation_matrix(phi2)
    return R
    
def norm_mat_2D(alpha, beta):
    return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)

def norm_mat(alpha_x, beta_x, alpha_y, beta_y):
    """4D normalization matrix (uncoupled)"""
    V = np.zeros((4, 4))
    V[:2, :2] = norm_mat_2D(alpha_x, beta_x)
    V[2:, 2:] = norm_mat_2D(alpha_y, beta_y)
    return V
    

class Envelope:

    def __init__(self, params=None, mass=1., energy=1.):
        if not params:
            self.params = np.array([1, 0, 0, 1, 0, -1, 1, 0])
        else:
            self.params = params
        self.mass = mass
        self.energy = energy
                
    def matrix(self):
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        return P.flatten()
        
    def cov(self):
        P = self.matrix()
        return 0.25 * np.matmul(P, P.T)
        
    def twiss(self):
        S = self.cov()
        ex = np.sqrt(la.det(S[:2, :2]))
        ey = np.sqrt(la.det(S[2:, 2:]))
        bx = S[0, 0] / ex
        by = S[2, 2] / ey
        ax = -S[0, 1] / ex
        ay = -S[2, 3] / ey
        return ax, ay, bx, by, ex, ey
        
    def tilt_angle(self, xvar='x', yvar='y'):
        """Return tilt angle of ellipse (ccw)."""
        a, b, ap, bp, e, f, ep, fp = self.params
        var_to_params = {
            'x': (a, b),
            'y': (e, f),
            'xp': (ap, bp),
            'yp': (ep, fp)
        }
        u1, u2 = var_to_params[xvar]
        v1, v2 = var_to_params[yvar]
        return 0.5 * np.arctan2(2*(u1*v1 + u2*v2), u1**2 + u2**2 - v1**2 - v2**2)

    def radii(self, xvar='x', yvar='y'):
        """Return radii of ellipse in real space."""
        a, b, ap, bp, e, f, ep, fp = self.params
        phi = self.tilt_angle(xvar, yvar)
        cos, sin = np.cos(phi), np.sin(phi)
        cos2, sin2 = cos**2, sin**2
        var_to_params = {
            'x': (a, b),
            'y': (e, f),
            'xp': (ap, bp),
            'yp': (ep, fp)
        }
        a, b = var_to_params[xvar]
        e, f = var_to_params[yvar]
        a2b2, e2f2 = a**2 + b**2, e**2 + f**2
        area = a*f - b*e
        cx = np.sqrt(area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
        cy = np.sqrt(area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
        return cx, cy
    
    def phase_diff(self):
        P = self.matrix()
        ax, ay, bx, by, _, _ = self.twiss()
        V = norm_mat(ax, bx, ay, by)
        a, b, ap, bp, e, f, ep, fp = np.matmul(la.inv(V), P).flatten()
        mux, muy = np.arctan2(ap, a), np.arctan2(ep, e)
        return muy - mux
        
    def fit_to_cov_mat(self, S, verbose=0):
        """Return the parameters which generate the covariance matrix S."""
        def mismatch(params):
            self.params = params
            return 1e12 * covmat2vec(S - self.cov())
        result = opt.least_squares(
            mismatch,
            self.params,
            xtol=1e-12, verbose=verbose)
        return result.x
        
    def from_bunch(self, bunch):
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params
        
    def from_twiss(self, ax, ay, bx, by, ex, ey, nu, mode):
        mux = np.pi/2 - nu
        if mode == 2:
            mux *= -1
        R = phase_adv_matrix(mux, 0.)
        # Check if emittance is negative (will happen if very close to 0)
        if ex <= 0:
            ex = 1e-12
        if ey <= 0:
            ey = 1e-12
        A = np.sqrt(4 * np.diag([ex, ex, ey, ey]))
        V = norm_mat(ax, bx, ay, by)
        # Get envelope parameters
        if mode == 1:
            P = np.array([[1, 0], [0, 1], [0, -1], [+1, 0]])
        elif mode == 2:
            P = np.array([[1, 0], [0, 1], [0, +1], [-1, 0]])
        self.params = la.multi_dot([V, A, R, P]).flatten()
        return self.params
        
    def to_bunch(self):
        bunch, params_dict = initialize_bunch(self.mass, self.energy)
        a, b, ap, bp, e, f, ep, fp = self.params
        bunch.addParticle(a, ap, e, ep, 0, 0)
        bunch.addParticle(b, bp, f, fp, 0, 0)
        return bunch, params_dict
        
    def match(self, lattice, mode=1, nturns=1, verbose=0):
    
        def cost(param_vec, lattice, mode, e_mode, nturns=1):
            # Compute initial moments
            ax, ay, bx, by, r, nu = param_vec
            ex, ey = r * e_mode, (1 - r) * e_mode
            self.from_twiss(ax, ay, bx, by, ex, ey, nu, mode)
            initial_cov_mat = self.cov()
            # Track and compute mismatch
            bunch, params_dict = self.to_bunch()
            for _ in range(nturns):
                lattice.trackBunch(bunch, params_dict)
            self.from_bunch(bunch)
            return 1e6 * covmat2vec(self.cov() - initial_cov_mat)
            
        # Set up parameter vector and bounds
        ax, ay, bx, by, ex, ey = self.twiss()
        nu = abs(self.phase_diff())
        e_mode = ex + ey
        param_vec = np.array([ax, ay, bx, by, ex/e_mode, nu])
        pad = 1e-5
        bounds = (
            (-np.inf, -np.inf, pad, pad, pad, pad),
            (np.inf, np.inf, np.inf, np.inf, 1-pad, np.pi-pad)
        )
        # Run optimizer
        result = opt.least_squares(
            cost,
            param_vec,
            args=(lattice, mode, e_mode, nturns),
            bounds=bounds,
            verbose=verbose,
            xtol=1e-12,
            ftol=1e-12,
        )
        # Extract envelope parameters
        ax, ay, bx, by, r, nu = result.x
        ex, ey = r * e_mode, (1 - r) * e_mode
        self.from_twiss(ax, ay, bx, by, ex, ey, nu, mode)
        return result

    def track(self, lattice, nturns=1, output_file=None, mm_mrad=True):
        bunch, params_dict = self.to_bunch()
        tracked_params = np.zeros((nturns + 1, 8))
        tracked_params[0] = self.params
        for i in trange(nturns):
            lattice.trackBunch(bunch, params_dict)
            tracked_params[i + 1] = self.from_bunch(bunch)
        self.params = tracked_params[-1]
        if mm_mrad:
            tracked_params *= 1000 # mm*mrad
        s = np.zeros((nturns + 1, 1))
        tracked_params = np.hstack([s, tracked_params])
        if output_file is not None:
            np.savetxt(output_file, tracked_params, fmt='%1.15f')
        return tracked_params
