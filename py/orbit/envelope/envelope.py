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
    
def SigmaBL(ax, ay, bx, by, u, nu, eps, mode=1):
    """Bogacz & Lebedev covariance matrix for either eps1 = 0 or eps2 = 0."""
    cos, sin = np.cos(nu), np.sin(nu)
    if mode == 1:
        s11, s33 = bx, by
        s12, s34 = -ax, -ay
        s22 = ((1-u)**2 + ax**2) / bx
        s44 = (u**2 + ay**2) / by
        s13 = np.sqrt(bx*by) * cos
        s14 = np.sqrt(bx/by) * (u*sin - ay*cos)
        s23 = -np.sqrt(by/bx) * ((1-u)*sin + ax*cos)
        s24 = ((ay*(1-u) - ax*u)*sin + (u*(1-u) + ax*ay)*cos) / np.sqrt(bx*by)
    elif mode == 2:
        s11, s33 = bx, by
        s12, s34 = -ax, -ay
        s22 = (u**2 + ax**2) / bx
        s44 = ((1-u)**2 + ay**2) / by
        s13 = np.sqrt(bx*by) * cos
        s14 = -np.sqrt(bx/by) * ((1-u)*sin + ay*cos)
        s23 = np.sqrt(by/bx) * (u*sin - ax*cos)
        s24 = ((ax*(1-u) - ay*u)*sin + (u*(1-u) + ax*ay)*cos) / np.sqrt(bx*by)
    return eps * np.array([[s11, s12, s13, s14],
                           [s12, s22, s23, s24],
                           [s13, s23, s33, s34],
                           [s14, s24, s34, s44]])
        

class Envelope:

    def __init__(self, params=None, mass=1., energy=1.):
        if params is None:
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
        
    def twissBL(self, mode):
        ax, ay, bx, by, ex, ey = self.twiss()
        eps = ex + ey
        if mode == 1:
            u = ey / eps
            bx *= (1 - u)
            ax *= (1 - u)
            by *= u
            ay *= u
        elif mode == 2:
            u = ex / eps
            bx *= u
            ax *= u
            by *= (1 - u)
            ay *= (1 - u)
        return ax, ay, bx, by, u
        
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
        Vinv = la.inv(V)
        a, b, ap, bp, e, f, ep, fp = self.to_vec(np.matmul(Vinv, P))
        # Positive phase is clockwise
        mux = -np.arctan2(ap, a)
        muy = -np.arctan2(ep, e)
        # Put phases in in range [0, 2pi]
        if mux < 0:
            mux += 2*np.pi
        if muy < 0:
            muy += 2*np.pi
        # Absolute difference modulo pi
        nu = abs(muy - mux)
        if nu > np.pi:
            nu = 2*np.pi - nu
        return nu
        
    def fit_to_cov_mat(self, S, verbose=0):
        """Return the parameters which generate the covariance matrix S."""
        def mismatch(params, S):
            self.params = params
            return 1e12 * covmat2vec(S - self.cov())
        result = opt.least_squares(
            mismatch,
            self.params,
            args=(S,),
            xtol=1e-12, verbose=verbose)
        return result.x
        
    def from_bunch(self, bunch):
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params
        
    def from_twiss(self, ax, ay, bx, by, ex, ey, nu, mode):
        mu = np.pi/2 - nu
        if mode == 2:
            mu *= -1
        R = phase_adv_matrix(0., mu)
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
        
    def get_part_coords(self, psi=0):
        a, b, ap, bp, e, f, ep, fp = self.params
        cos, sin = np.cos(psi), np.sin(psi)
        x = a*cos + b*sin
        y = e*cos + f*sin
        xp = ap*cos + bp*sin
        yp = ep*cos + fp*sin
        return np.array([x, xp, y, yp])
        
    def generate_dist(self, nparts):
        psis = np.linspace(0, 2*np.pi, nparts)
        X = np.array([self.get_part_coords(psi) for psi in psis])
        radii = np.sqrt(np.random.random(size=(nparts, 1)))
        return X * np.sqrt(np.random.random((nparts, 1)))
        
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
            return 1e12 * covmat2vec(self.cov() - initial_cov_mat)
            
        # Set up parameter vector and bounds
        ax, ay, bx, by, ex, ey = self.twiss()
        nu = self.phase_diff()
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
        
        
    def get_transfer_matrix(self, lattice):
        """Compute the linear transfer matrix with the inclusion of space charge.
        
        For this to have meaning, the envelope parameters should already be
        matched to the lattice.
        """
        bunch, params_dict = self.to_bunch()
        
        step_arr_init = np.full(6, 1e-6)
        step_arr = np.copy(step_arr_init)
        step_reduce = 20.
        bunch.addParticle(0., 0., 0., 0., 0., 0.);
        bunch.addParticle(step_arr[0]/step_reduce, 0., 0., 0., 0., 0.)
        bunch.addParticle(0., step_arr[1]/step_reduce, 0., 0., 0., 0.)
        bunch.addParticle(0., 0., step_arr[2]/step_reduce, 0., 0., 0.)
        bunch.addParticle(0., 0., 0., step_arr[3]/step_reduce, 0., 0.)
        bunch.addParticle(step_arr[0], 0., 0., 0., 0., 0.)
        bunch.addParticle(0., step_arr[1], 0., 0., 0., 0.)
        bunch.addParticle(0., 0., step_arr[2], 0., 0., 0.)
        bunch.addParticle(0., 0., 0., step_arr[3], 0., 0.)
        
        lattice.trackBunch(bunch, params_dict)
        X = np.array([[bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
                      for i in range(2, bunch.getSize())])
                      
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                x1 = step_arr[i] / step_reduce
                x2 = step_arr[i]
                y0 = X[0, j]
                y1 = X[i + 1, j]
                y2 = X[i + 1 + 4, j]
                M[j, i] = ((y1-y0)*x2*x2 - (y2-y0)*x1*x1) / (x1*x2*(x2-x1))
        return M
        
        
        
        
    def matchBL(self, lattice, mode=1, nturns=1, max_attempts=100, tol=1e-8, verbose=0):
        """Try to use the BL formulation to match."""
    
        def cost(param_vec, lattice, mode, eps, nturns=1):
            ax, ay, bx, by, u, nu = param_vec # BL Twiss
            Sigma0 = SigmaBL(ax, ay, bx, by, u, nu, eps, mode)
            self.fit_to_cov_mat(Sigma0)
            # Track and compute mismatch
            bunch, params_dict = self.to_bunch()
            for _ in range(nturns):
                lattice.trackBunch(bunch, params_dict)
            self.from_bunch(bunch)
            return 1e12 * covmat2vec(self.cov() - Sigma0)
        
        for i in range(max_attempts):
        
            # Set up parameter vector and bounds
            _, _, _, _, ex, ey = self.twiss()
            eps = ex + ey
            ax, ay, bx, by, u = self.twissBL(mode)
            nu = self.phase_diff()
            param_vec = np.array([ax, ay, bx, by, u, nu])
            pad = 1e-5
            bounds = (
                (-np.inf, -np.inf, pad, pad, pad, pad),
                (np.inf, np.inf, np.inf, np.inf, 1-pad, np.pi-pad)
            )
        
            # Run optimizer
            result = opt.least_squares(
                cost,
                param_vec,
                args=(lattice, mode, eps, nturns),
                bounds=bounds,
                verbose=verbose,
                xtol=1e-12,
            )

            # Fit parameters to result
            ax, ay, bx, by, u, nu = result.x
            matched_Sigma = SigmaBL(ax, ay, bx, by, u, nu, eps, mode)
            self.fit_to_cov_mat(matched_Sigma)
        
            print '    C = {:.2e}, attempts = {}'.format(result.cost, i)
            if result.cost < tol:
                print '    SUCCESS'
                break
            
        return result
