"""
This module contains functions related to the envelope parameterization of
the Danilov distribution.
"""

# 3rd party
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
    return np.array([[beta, 0], [-alpha, 1]]) / np.sqrt(beta)


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
                           
    
def bounds(pad=1e-4):
    """Return bounds for optimizer."""
    nu_min, nu_max = pad, np.pi - pad
    r_min, r_max = pad, 1 - pad
    alpha_min, alpha_max = -np.inf, np.inf
    beta_min, beta_max = pad, np.inf
    bounds = (
        (alpha_min, alpha_min, beta_min, beta_min, r_min, nu_min),
        (alpha_max, alpha_max, beta_max, beta_max, r_max, nu_max)
    )
    return bounds
        

class Envelope:

    def __init__(self, params=None, mass=1., energy=1.):
        self.params = params
        if params is None:
            self.params = np.array([1, 0, 0, 1, 0, -1, 1, 0])
        self.mass = mass
        self.energy = energy
                
    def matrix(self):
        """Convert the envelope parameters to the envelope matrix."""
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        """Unpack the envelope matrix."""
        return P.flatten()
        
    def cov(self):
        """Return the transverse covariance matrix."""
        P = self.matrix()
        return 0.25 * np.matmul(P, P.T)
        
    def twiss(self):
        """Return the horizontal/vertical Twiss parameters and emittances."""
        S = self.cov()
        ex = np.sqrt(la.det(S[:2, :2]))
        ey = np.sqrt(la.det(S[2:, 2:]))
        bx = S[0, 0] / ex
        by = S[2, 2] / ey
        ax = -S[0, 1] / ex
        ay = -S[2, 3] / ey
        return ax, ay, bx, by, ex, ey
        
    def twissBL(self, mode):
        """Get the mode Twiss parameters, as defined by Bogacz & Lebedev."""
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
        
    def tilt_angle(self, x1='x', x2='y'):
        """Return the ccw tilt angle of ellipse in the x1-x2 plane."""
        a, b, ap, bp, e, f, ep, fp = self.params
        var_to_params = {
            'x': (a, b),
            'y': (e, f),
            'xp': (ap, bp),
            'yp': (ep, fp)
        }
        a, b = var_to_params[x1]
        e, f = var_to_params[x2]
        return 0.5 * np.arctan2(2*(a*e + b*f), a**2 + b**2 - e**2 - f**2)

    def radii(self, x1='x', x2='y'):
        """Return the semi-major and semi-minor axes in the x1-x2 plane."""
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
        a, b = var_to_params[x1]
        e, f = var_to_params[x2]
        a2b2, e2f2 = a**2 + b**2, e**2 + f**2
        area = a*f - b*e
        cx = np.sqrt(area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
        cy = np.sqrt(area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
        return cx, cy
        
    def normed2D(self):
        """Return the normalized envelope parameters.
        
        Here 'normalized' means the x-x' and y-y' ellipse will be upright. It
        is not normalized in the 4D sense (diagonal covariance matrix).
        """
        P = self.matrix()
        ax, ay, bx, by, _, _ = self.twiss()
        V = norm_mat(ax, bx, ay, by)
        return self.to_vec(np.matmul(la.inv(V), P))
        
    def phases(self):
        """Return the horizontal/vertical phases (in range [0, 2pi] of a
         particle with x=a, x'=a', y=e, y'=e'."""
        a, b, ap, bp, e, f, ep, fp = self.normed2D()
        mux = -np.arctan2(ap, a)
        muy = -np.arctan2(ep, e)
        if mux < 0:
            mux += 2*np.pi
        if muy < 0:
            muy += 2*np.pi
        return mux, muy
        
    def phase_diff(self):
        """Return the x-y phase difference (nu) of all particles in the beam.
        The value returned is in the range [0, pi].
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        return nu if nu < np.pi else 2*np.pi - nu
        
    def track(self, lattice, nturns=1):
        """Track the envelope through the lattice."""
        bunch, params_dict = self.to_bunch()
        for _ in range(nturns):
            lattice.trackBunch(bunch, params_dict)
        self.from_bunch(bunch)
        
    def tunes(self, lattice):
        """Get the fractional horizontal and vertical tunes."""
        mux0, muy0 = self.phases()
        self.track(lattice)
        mux1, muy1 = self.phases()
        tune_x = (mux1 - mux0) / (2*np.pi)
        tune_y = (muy1 - muy0) / (2*np.pi)
        tune_x %= 1
        tune_y %= 1
        return tune_x, tune_y
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * covmat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12)
        return result.x
        
    def fit_twiss(self, ax, ay, bx, by, ex, ey, nu, mode):
        """Fit the envelope to the Twiss parameters.
        
        Parameters
        ----------
        ax{y} : float
            The horizontal{vertical} alpha function.
        bx{y} : float
            The horizontal{vertical} beta function.
        ex{y} : float
            The horizontal{vertical} emittance.
        nu : float
            The x-y phase difference.
        mode : int
            The mode (1 or 2) of the beam. Mode 1 means e1 = 0 and mode 2 means
            e2 = 0, where e1 and e2 are the mode emittances.
        """
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
        
    def param_vec(self):
        """Construct the parameter vector for use in optimizer."""
        ax, ay, bx, by, ex, ey = self.twiss()
        return np.array([ax, ay, bx, by, ex/(ex+ey), self.phase_diff()])
        
    def fit_param_vec(self, param_vec, eps, mode):
        """Get envelope parameters from the parameter vector."""
        ax, ay, bx, by, r, nu = param_vec
        self.fit_twiss(ax, ay, bx, by, r*eps, (1-r)*eps, nu, mode)
        return self.params
        
    def get_part_coords(self, psi=0):
        """Return the coordinates of a single particle on the envelope.
        
        x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
        y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi).
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        cos, sin = np.cos(psi), np.sin(psi)
        x = a*cos + b*sin
        y = e*cos + f*sin
        xp = ap*cos + bp*sin
        yp = ep*cos + fp*sin
        return np.array([x, xp, y, yp])
        
    def generate_dist(self, nparts):
        """Generate a distribution of particles from the envelope.

        Parameters
        ----------
        nparts : int
            The number of particles in the bunch.

        Returns
        -------
        X : NumPy array, shape (nparts, 4)
            The coordinate array for the distribution.
        """
        psis = np.linspace(0, 2*np.pi, nparts)
        X = np.array([self.get_part_coords(psi) for psi in psis])
        radii = np.sqrt(np.random.random(size=(nparts, 1)))
        return X * np.sqrt(np.random.random((nparts, 1)))
        
    def from_bunch(self, bunch):
        """Extract the envelope parameters from a Bunch object."""
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params
         
    def to_bunch(self):
        """Create bunch with the first two particles storing the envelope
        parameters."""
        bunch, params_dict = initialize_bunch(self.mass, self.energy)
        a, b, ap, bp, e, f, ep, fp = self.params
        bunch.addParticle(a, ap, e, ep, 0, 0)
        bunch.addParticle(b, bp, f, fp, 0, 0)
        return bunch, params_dict
        
    def dist_to_bunch(self, nparts, bunch_length):
        """Generate a distribution of particles from the envelope and store
        in Bunch object.
        
        Parameters
        ----------
        nparts : int
            The number of particles in the bunch.
        bunch_length : float
            The length of the bunch (meters).
        
        Returns
        -------
        bunch: Bunch object
            The bunch representing the distribution of size 2 + nparts. The
            first two particles store the envelope parameters.
        params_dict : dict
            The dictionary of parameters for the bunch.
        """
        X = self.generate_dist(nparts)
        z = bunch_length * np.random.random(nparts)
        bunch, params_dict = self.to_bunch()
        for (x, xp, y, yp), _z in zip(X, z):
            bunch.addParticle(x, xp, y, yp, _z, 0.)
        return bunch, params_dict
        
    def _match(self, lattice, mode, nturns=1, verbose=0):
        """Run least squares optimization to find matched envelope.
        
        Uses the current envelope as a seed, keeping the mode emittance fixed.
        The envelope is then fit to the solution returned by the optimizer. This
        method does not always work; sometimes the optimizer gets stuck on
        a solution which is not a perfect match.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into.
        mode : int
            The mode (1 or 2) of the distribution corresponding to the which
            mode emittance is chosen to be zero.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        verbose : int
            Whether to diplay the optimizer progress (0, 1, or 2). 0 is no
            output, 1 shows the end result, and 2 shows each iteration.
        """
        def cost(param_vec, lattice, mode, eps, nturns=1):
            self.fit_param_vec(param_vec, eps, mode)
            initial_cov_mat = self.cov()
            self.track(lattice)
            return 1e12 * covmat2vec(self.cov() - initial_cov_mat)
            
        ax, ay, bx, by, ex, ey = self.twiss()
        eps = ex + ey
        result = opt.least_squares(
            cost,
            self.param_vec(),
            args=(lattice, mode, eps, nturns),
            bounds=bounds(1e-4),
            verbose=verbose,
            xtol=1e-12
        )
        self.fit_param_vec(result.x, eps, mode)
        return result.cost
            
    def perturb(self, radius, mode, param_vec=None):
        """Randomly perturb the envelope about param_vec.
        
        Parameters
        ----------
        radius : float
            For each parameter p, the search interval will be
            [(1 - radius)*p, (1 + radius)*p].
        mode : float
            The mode (1 or 2) of the beam.
        param_vec : list or array, length 6
            The vector of beam parameters [ax, ay, bx, by, r, nu]. If not
            provided, the current beam parameters will be used.
        """
        lo, hi = 1 - radius, 1 + radius
        
        _, _, _, _, ex, ey = self.twiss()
        eps = ex + ey
        if param_vec is None:
            param_vec = np.array([ax, ay, bx, by, ex/(ex+ey), self.phase_diff()])
        ax, ay, bx, by, r, nu = param_vec
   
        ax = np.random.uniform(lo*ax, hi*ax)
        ay = np.random.uniform(lo*ay, hi*ay)

        bx_lo = lo * by
        bx_hi = hi * by
        by_lo = lo * by
        by_hi = hi * by
        if bx_lo < 0.01:
         bx_lo = 0.01
        if by_lo < 0.01:
         by_lo = 0.01
        bx = np.random.uniform(bx_lo, bx_hi)
        by = np.random.uniform(by_lo, by_hi)

        nu_lo = lo * nu
        nu_hi = hi * nu
        if nu_lo < 0.01 * np.pi:
            nu_lo = 0.01 * np.pi
        if nu_hi > 0.99 * np.pi:
            nu_hi = 0.99 * np.pi
        nu = np.random.uniform(nu_lo, nu_hi)

        r_lo = lo * r
        r_hi = hi * r
        if r_lo < 0.01:
            r_lo = 0.01
        if r_hi > 0.99:
            r_hi = 0.99
        r = np.random.uniform(r_lo, r_hi)
        
        self.fit_param_vec(np.array([ax, ay, bx, by, r, nu]), eps, mode)
        
    def match(self, lattice, nturns, mode, tol, max_attempts=100, radius=0.75):
        """Modify the envelope so that is matched to the lattice.
        
        For certain combinations of lattice coupling, mode emittances, and beam
        intensity, the least squares optimizer converges to a solution which is
        not an exact match. This method runs the optimizer a number of times,
        each time using a different seed, until an exact match is found.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        mode : int
            The mode (1 or 2) of the distribution corresponding to the which
            mode emittance is chosen to be zero.
        tol : float
            The maximum acceptable value of the cost function from the optimizer.
        max_attempts : int
            The maximum number of attempts to match. Each attempt runs the
            optimizer with a different seed.
        radius : float
            Each parameter p in the initial parameter vector will be modified in
            the interval [(1 - radius)*p, (1 + radius)*p]. The best value I have
            found is 0.75.
        """
        print 'Matching.'
        init_param_vec = self.param_vec()
        for i in range(max_attempts):
            cost = self._match(lattice, mode, nturns, verbose=0)
            print '    cost = {:.2e},  attempt {}'.format(cost, i + 1)
            if cost < tol:
                print '    SUCCESS'
                break
            self.perturb(radius, mode, init_param_vec)
                
    def get_transfer_matrix(self, lattice):
        """Compute the linear transfer matrix with space charge included.
        
        The method is taken from /src/teapot/MatrixGenerator.cc. That method
        computes the 7x7 transfer matrix, but we just need the 4x4 matrix.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice may have envelope solver nodes. These nodes should
            track the beam envelope using the first two particles in the bunch,
            then use these to apply the appropriate linear space charge kicks to
            the rest of the particles.
        
        Returns
        -------
        M : NumPy array, shape (4, 4)
            The 4x4 linear transfer matrix of the combinded lattice + space
            charge focusing system.
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
