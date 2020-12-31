"""
This module contains functions related to the envelope parameterization of
the {2, 2} Danilov distribution.

Reference: https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.6.094202
"""

# Imports
#------------------------------------------------------------------------------
# Standard
import time
import copy
# Third party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange
# PyORBIT
from bunch import Bunch
from orbit.analysis.analysis import covmat2vec
from orbit.coupling import bogacz_lebedev as BL
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.utils import helper_funcs as hf
from orbit.utils.helper_funcs import (
    initialize_bunch,
    tprint,
    is_stable,
    unequal_eigtunes,
    get_perveance,
    params_from_transfer_matrix,
    toggle_spacecharge_nodes
)
    
# Helper functions
#------------------------------------------------------------------------------
def rotation_matrix(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])
    
def rotation_matrix_4D(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, 0, S, 0], [0, C, 0, S], [-S, 0, C, 0], [0, -S, 0, C]])
    
def phase_adv_matrix(phi1, phi2):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(phi1)
    R[2:, 2:] = rotation_matrix(phi2)
    return R

def Vmat_2D(alpha_x, beta_x, alpha_y, beta_y):
    """Normalization matrix (uncoupled)"""
    def V_uu(alpha, beta):
        return np.array([[beta, 0], [-alpha, 1]]) / np.sqrt(beta)
    V = np.zeros((4, 4))
    V[:2, :2] = V_uu(alpha_x, beta_x)
    V[2:, 2:] = V_uu(alpha_y, beta_y)
    return V

# Define bounds on the 4D Twiss parameters
pad = 1e-4
alpha_min, alpha_max = -np.inf, np.inf
beta_min, beta_max = pad, np.inf
nu_min, nu_max = pad, np.pi - pad
u_min, u_max = pad, 1 - pad
lb = (alpha_min, alpha_min, beta_min, beta_min, u_min, nu_min)
ub = (alpha_max, alpha_max, beta_max, beta_max, u_max, nu_max)
twiss_bounds = (lb, ub)


# Class definitions
#------------------------------------------------------------------------------

class Envelope:
    """Class for the transverse beam envelope.
    
    This class includes methods to compute the beam statistics and interact
    with PyORBIT using the envelope parameterization.
    
    Attributes
    ----------
    eps : float
        The rms intrinsic emittance of the beam [m*rad].
    mode : int
        Whether to choose eps2=0 (mode 1) or eps1=0 (mode 2).
    ex_frac : float
        The x emittance ratio, such that ex = ex_frac * eps
    mass : float
        The particle mass [GeV/c^2].
    energy : float
        The kinetic energy per particle [GeV].
    intensity : float
        The number of particles contained in the envelope. If 0, the particles
        are non-interacting.
    length : float
        The bunch length [m]. This is used to calculate the axial charge
        density.
    perveance : float
        The dimensionless beam perveance.
    params : list, optional
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    """
    def __init__(self, eps=1., mode=1, ex_frac=0.5, mass=0.93827231,
                 energy=1.0, length=1e-5, intensity=0.0, params=None):
        self.eps = eps
        self.mode = mode
        self.ex_frac, self.ey_frac = ex_frac, 1 - ex_frac
        self.mass = mass
        self.energy = energy
        self.length = length
        self.set_spacecharge(intensity)
        if params is not None:
            self.params = np.array(params)
            ex, ey = self.emittances()
            self.eps = ex + ey
            self.ex_frac = ex / self.eps
        else:
            ex, ey = ex_frac * eps, (1 - ex_frac) * eps
            rx, ry = np.sqrt(4 * ex), np.sqrt(4 * ey)
            if mode == 1:
                self.params = np.array([rx, 0, 0, rx, 0, -ry, +ry, 0])
            elif mode == 2:
                self.params = np.array([rx, 0, 0, rx, 0, +ry, -ry, 0])
        self.twiss_bounds = twiss_bounds
        
    def copy(self):
        return copy.deepcopy(self)
                
    def set_params(self, a, b, ap, bp, e, f, ep, fp):
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        
    def get_params_for_dim(self, dim='x'):
        """Return envelope parameters associated with the given dimension."""
        a, b, ap, bp, e, f, ep, fp = self.params
        return {'x':(a, b), 'y':(e, f), 'xp':(ap, bp), 'yp': (ep, fp)}[dim]
        
    def set_spacecharge(self, intensity):
        """Set the beam perveance."""
        self.intensity = intensity
        self.charge_density = intensity / self.length
        self.perveance = get_perveance(self.mass, self.energy,
                                       self.charge_density)
                                          
    def set_length(self, length):
        """Set the axial beam length. This will change the beam perveance."""
        self.length = length
        self.set_spacecharge(self.intensity)
        
    def matrix(self):
        """Create the envelope matrix P from the envelope parameters.
        
        The matrix is defined by x = P.c, where x = [x, x', y, y']^T,
        c = [cos(psi), sin(psi)], and '.' means matrix multiplication, with
        0 <= psi <= 2pi. This is useful because any transformation to the
        particle coordinate vector x also done to P. For example, if x -> M.x,
        then P -> M.P.
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        """Convert the envelope matrix to vector form."""
        return P.ravel()
        
    def get_norm_mat_2D(self, inv=False):
        """Return the normalization matrix V (2D sense)."""
        ax, ay, bx, by = self.twiss2D()
        V = Vmat_2D(ax, bx, ay, by)
        return la.inv(V) if inv else V
        
    def norm4D(self):
        """Normalize the envelope parameters in the 4D sense.
        
        In the transformed coordates the covariance matrix is diagonal, and the
        x-x' and y-y' emittances are the intrinsic emittances.
        """
        r_n = np.sqrt(4 * self.eps)
        if self.mode == 1:
            self.params = np.array([r_n, 0, 0, r_n, 0, 0, 0, 0])
        elif self.mode == 2:
            self.params = np.array([0, 0, 0, 0, 0, r_n, r_n, 0])
                                
    def norm2D(self, scale=False):
        """Normalize the envelope parameters in the 2D sense and return the
        parameters.
        
        Here 'normalized' means the x-x' and y-y' ellipses will be circles of
        radius sqrt(ex) and sqrt(ey), where ex and ey are the apparent
        emittances. The cross-plane elements of the covariance matrix will not
        all be zero. If `scale` is True, the x-x' and y-y' ellipses will be
        scaled to unit radius.
        """
        self.transform(self.get_norm_mat_2D(inv=True))
        if scale:
            ex, ey = 4 * self.emittances()
            self.params[:4] /= np.sqrt(ex)
            self.params[4:] /= np.sqrt(ey)
        return self.params
            
    def normed_params_2D(self):
        """Return the normalized envelope parameters in the 2D sense without
        actually changing the envelope."""
        true_params = np.copy(self.params)
        normed_params = self.norm2D()
        self.params = true_params
        return normed_params
                
    def transform(self, M):
        """Apply matrix M to the coordinates."""
        P_new = np.matmul(M, self.matrix())
        self.params = self.to_vec(P_new)
        
    def norm_transform(self, M):
        """Normalize, then apply M to the coordinates."""
        self.norm4D()
        self.transform(M)
        
    def advance_phase(self, mux=0., muy=0.):
        """Advance the x{y} phase by mux{muy} degrees.

        It is equivalent to tracking through an uncoupled lattice which the
        envelope is matched to.
        """
        mux, muy = np.radians([mux, muy])
        V = self.get_norm_mat_2D()
        M = la.multi_dot([V, phase_adv_matrix(mux, muy), la.inv(V)])
        self.transform(M)
        
    def rotate(self, phi):
        """Apply clockwise rotation by phi degrees in x-y space."""
        self.transform(rotation_matrix_4D(np.radians(phi)))

    def swap_xy(self):
        """Exchange (x, x') <-> (y, y')."""
        self.params[:4], self.params[4:] = self.params[4:], self.params[:4]
        
    def cov(self):
        """Return the transverse covariance matrix."""
        P = self.matrix()
        return 0.25 * np.matmul(P, P.T)
        
    def emittances(self, mm_mrad=False):
        """Return the horizontal/vertical rms emittance."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        emittances = np.array([ex, ey])
        if mm_mrad:
            emittances *= 1e6
        return  emittances
        
    def twiss2D(self):
        """Return the horizontal/vertical Twiss parameters and emittances."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        bx = Sigma[0, 0] / ex
        by = Sigma[2, 2] / ey
        ax = -Sigma[0, 1] / ex
        ay = -Sigma[2, 3] / ey
        return np.array([ax, ay, bx, by])
        
    def twiss4D(self):
        """Return the 4D Twiss parameters, as defined by Bogacz & Lebedev."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        bx = Sigma[0, 0] / self.eps
        by = Sigma[2, 2] / self.eps
        ax = -Sigma[0, 1] / self.eps
        ay = -Sigma[2, 3] / self.eps
        nu = self.phase_diff()
        if self.mode == 1:
            u = ey / self.eps
        elif self.mode == 2:
            u = ex / self.eps
        return np.array([ax, ay, bx, by, u, nu])
        
    def tilt_angle(self, x1='x', x2='y'):
        """Return the ccw tilt angle in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        return 0.5 * np.arctan2(2*(a*e + b*f), a**2 + b**2 - e**2 - f**2)

    def radii(self, x1='x', x2='y'):
        """Return the semi-major and semi-minor axes in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        phi = self.tilt_angle(x1, x2)
        cos, sin = np.cos(phi), np.sin(phi)
        cos2, sin2, sincos = cos**2, sin**2, sin*cos
        x2, y2 = a**2 + b**2, e**2 + f**2
        A = abs(a*f - b*e)
        cx = np.sqrt(A**2 / (y2*cos2 + x2*sin2 + 2*(a*e + b*f)*sincos))
        cy = np.sqrt(A**2 / (x2*cos2 + y2*sin2 - 2*(a*e + b*f)*sincos))
        return np.array([cx, cy])
        
    def area(self, x1='x', x2='y'):
        """Return the area in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        return np.pi * np.abs(a*f - b*e)
        
    def phases(self):
        """Return the horizontal/vertical phases in range [0, 2*pi] of a
         particle with x=a, x'=a', y=e, y'=e'."""
        a, b, ap, bp, e, f, ep, fp = self.normed_params_2D()
        mux, muy = -np.arctan2(ap, a), -np.arctan2(ep, e)
        if mux < 0:
            mux += 2*np.pi
        if muy < 0:
            muy += 2*np.pi
        return mux, muy
        
    def phase_diff(self):
        """Return the x-y phase difference (nu) of all particles in the beam.
        
        The value returned is in the range [0, pi]. This can also be found from
        the equation cos(nu) = r, where r is the x-y correlation coefficient.
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        return nu if nu < np.pi else 2*np.pi - nu
        
    def fit_twiss2D(self, ax, ay, bx, by, ex_frac):
        """Fit the envelope to the 2D Twiss parameters."""
        V = Vmat_2D(ax, bx, ay, by)
        ex, ey = ex_frac * self.eps, (1 - ex_frac) * self.eps
        A = np.sqrt(4 * np.diag([ex, ex, ey, ey]))
        self.norm2D(scale=True)
        self.transform(np.matmul(V, A))
        
    def fit_twiss4D(self, twiss_params):
        """Fit the envelope to the 4D Twiss parameters.
        
        `twiss_params` is an array containing the 4D Twiss params for a single
        mode: [ax, ay, bx, by, u, nu], where
        * ax{y} : The horizontal{vertical} alpha function -<xx'>/e1 {-<xx'>/e2}.
        * bx{y} : The horizontal{vertical} beta function <xx>/e1 {<xx>/e2}.
        * u : The coupling parameter in range [0, 1]. This is equal to ey/e1
              when mode=1 or ex/e2 when mode=2.
        * nu : The x-y phase difference in range [0, pi].
        """
        ax, ay, bx, by, u, nu = twiss_params
        V = BL.Vmat(ax, ay, bx, by, u, nu, self.mode)
        self.norm_transform(V)
        
    def set_twiss_param_4D(self, name, value):
        """Change a single Twiss parameter while keeping the others fixed."""
        ax, ay, bx, by, u, nu = self.twiss4D()
        twiss_dict = {'ax':ax, 'ay':ay, 'bx':bx, 'by':by, 'u':u, 'nu':nu}
        twiss_dict[name] = value
        self.fit_twiss4D([twiss_dict[key]
                          for key in ('ax', 'ay', 'bx', 'by', 'u', 'nu')])
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * covmat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12, verbose=verbose)
        return result.x
        
    def get_part_coords(self, psi=0):
        """Return the coordinates of a single particle on the envelope."""
        a, b, ap, bp, e, f, ep, fp = self.params
        cos, sin = np.cos(psi), np.sin(psi)
        x = a*cos + b*sin
        y = e*cos + f*sin
        xp = ap*cos + bp*sin
        yp = ep*cos + fp*sin
        return np.array([x, xp, y, yp])
        
    def generate_dist(self, nparts, density='uniform'):
        """Generate a distribution of particles from the envelope.

        Returns: NumPy array, shape (nparts, 4)
            The coordinate array for the distribution.
        """
        nparts = int(nparts)
        psis = np.linspace(0, 2*np.pi, nparts)
        X = np.array([self.get_part_coords(psi) for psi in psis])
        if density == 'uniform':
            radii = np.sqrt(np.random.random(nparts))
        elif density == 'on_ellipse':
            radii = np.ones(nparts)
        elif density == 'gaussian':
            radii = np.random.normal(size=nparts)
        return radii[:, np.newaxis] * X
            
    def from_bunch(self, bunch):
        """Extract the envelope parameters from a Bunch object."""
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params
        
    def to_bunch(self, nparts=0, no_env=False):
        """Add the envelope parameters to a Bunch object. The first two
        particles represent the envelope parameters.
        
        Parameters
        ----------
        nparts : int
            The number of particles in the bunch. The bunch will just hold the
            envelope parameters if nparts == 0.
        no_env : bool
            If True, do not include the envelope parameters in the first
            two bunch particles.
        
        Returns
        -------
        bunch: Bunch object
            The bunch representing the distribution of size 2 + nparts
            (unless `no_env` is True).
        params_dict : dict
            The dictionary of parameters for the bunch.
        """
        bunch, params_dict = initialize_bunch(self.mass, self.energy)
        if not no_env:
            a, b, ap, bp, e, f, ep, fp = self.params
            bunch.addParticle(a, ap, e, ep, 0., 0.)
            bunch.addParticle(b, bp, f, fp, 0., 0.)
        for (x, xp, y, yp) in self.generate_dist(nparts):
            z = np.random.uniform(0, self.length)
            bunch.addParticle(x, xp, y, yp, z, 0.)
        if nparts > 0:
            bunch.macroSize(self.intensity/nparts if self.intensity > 0 else 1)
        return bunch, params_dict
        
    def track(self, lattice, nturns=1, ntestparts=0, progbar=False):
        """Track the envelope through the lattice.
        
        The envelope parameters are updated after it is tracked. If
        `ntestparts` is nonzero, test particles will be tracked which receive
        linear space charge kicks based on the envelope parameters.
        """
        bunch, params_dict = self.to_bunch(ntestparts)
        turns = trange(nturns) if progbar else range(nturns)
        for _ in turns:
            lattice.trackBunch(bunch, params_dict)
        self.from_bunch(bunch)
        
    def track_store_params(self, lattice, nturns):
        tbt_params = [self.params]
        for _ in range(nturns):
            self.track(lattice)
            tbt_params.append(self.params)
        return tbt_params
        
    def tunes(self, lattice):
        """Get the fractional horizontal and vertical tunes."""
        mux0, muy0 = self.phases()
        self.track(lattice)
        mux1, muy1 = self.phases()
        tune_x = (mux1 - mux0) / (2*np.pi)
        tune_y = (muy1 - muy0) / (2*np.pi)
        tune_x %= 1
        tune_y %= 1
        return np.array([tune_x, tune_y])
            
    def transfer_matrix(self, lattice):
        """Compute the linear transfer matrix with space charge included.
        
        The method is taken from /src/teapot/MatrixGenerator.cc. That method
        computes the 7x7 transfer matrix, but we just need the 4x4 matrix.
                
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice may have envelope solver nodes. These nodes should
            track the beam envelope using the first two particles in the bunch,
            then use these to apply the appropriate linear space charge kicks
            to the rest of the particles.
        
        Returns
        -------
        M : NumPy array, shape (4, 4)
            The 4x4 linear transfer matrix of the combined lattice + space
            charge focusing system.
        """
        if self.perveance == 0:
            return hf.transfer_matrix(lattice, self.mass, self.energy)
            
        # The envelope parameters will change if the beam is not matched to the
        # lattice, so make a copy.
        env = self.copy()
        
        step_arr_init = np.full(6, 1e-6)
        step_arr = np.copy(step_arr_init)
        step_reduce = 20.
        bunch, params_dict = env.to_bunch()
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
        
    def match_bare(self, lattice, method='auto', sc_nodes=None):
        """Match to the lattice without space charge.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice in which to match. If envelope solver nodes nodes are
            in the lattice, a list of these nodes needs to be passed as the
            `sc_nodes` parameter so they can be turned off/on.
        method : str
            If '4D', match to the lattice using the eigenvectors of the
            transfer matrix. This may result in the beam being completely
            flat, for example when the lattice is uncoupled. The '2D' method
            will only match the x-x' and y-y' ellipses of the beam.
        sc_nodes : list, optional
            List of nodes which are sublasses of SC_Base_AccNode. If provided,
            call `node.switcher = False` to prevent the node from tracking.
            
        Returns
        -------
        NumPy array, shape (8,)
            The matched envelope parameters.
        """
        if sc_nodes is not None:
            hf.toggle_spacecharge_nodes(sc_nodes, 'off')
            
        # Get linear transfer matrix
        M = self.transfer_matrix(lattice)
        if not is_stable(M):
            print 'WARNING: transfer matrix is not stable.'
        # Match to the lattice
        if method == 'auto':
            method = '4D' if unequal_eigtunes(M) else '2D'
        if method == '2D':
            lattice_params = params_from_transfer_matrix(M)
            ax, ay = [lattice_params[key] for key in ('alpha_x', 'alpha_y')]
            bx, by = [lattice_params[key] for key in ('beta_x', 'beta_y')]
            self.fit_twiss2D(ax, ay, bx, by, self.ex_frac)
        elif method == '4D':
            eigvals, eigvecs = la.eig(M)
            V = BL.construct_V(eigvecs)
            self.norm_transform(V)
        # If rms beam size and divergence are zero in either plane, make
        # them slightly nonzero. This will occur when the lattice is uncoupled
        # and has unequal x/y tunes.
        a, b, ap, bp, e, f, ep, fp = self.params
        pad = 1e-8
        if np.all(np.abs(self.params[:4]) < pad):
            a = bp = pad
        if np.all(np.abs(self.params[4:]) < pad):
            f = ep = pad
        self.set_params(a, b, ap, bp, e, f, ep, fp)
        # Avoid diagonal line in x-y space. This may occur for coupled lattice.
        self.advance_phase(1e-8 * np.pi)
        
        if sc_nodes is not None:
            hf.toggle_spacecharge_nodes(sc_nodes, 'on')
        return self.params
        
    def match(self, lattice, solver_nodes, tol=1e-4, verbose=0, method=None):
        """Match the envelope to the lattice."""
        if self.perveance == 0:
            return self.match_bare(lattice, solver_nodes)
            
        def initialize():
            self.set_twiss_param_4D('u', 0.5)
            self.set_twiss_param_4D('nu', np.pi/2)
            self.match_bare(lattice, '2D', solver_nodes)
        initialize()
        
        if method == 'lsq':
            return self._match_lsq(lattice, verbose=verbose)
        elif method == 'replace_by_avg':
            return self._match_replace_by_avg(lattice, verbose=verbose)
        else:
            result = self._match_lsq(lattice, verbose=verbose)
            if result.cost > tol:
                print "Cost = {:.2e} > tol.".format(result.cost)
                print "Trying 'replace by average' method."
                initialize()
                result = self._match_replace_by_avg(lattice, verbose=verbose)
            return result
            
    def _mismatch_error(self, lattice, factor=1e6, ssq=False):
        """Return the difference between the initial/final beam moments after
        tracking once through the lattice. The method does not change the
        envelope."""
        env = self.copy()
        Sigma0 = env.cov()
        env.track(lattice)
        Sigma1 = env.cov()
        residuals = factor * covmat2vec(Sigma1 - Sigma0)
        if ssq:
            return 0.5 * np.sum(residuals**2)
        else:
            return residuals
        
    def _match_lsq(self, lattice, **kwargs):
        """Run least squares optimizer to find the matched envelope.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be in
            place.
        **kwargs
            Keyword arguments to be passed to `scipy.optimize.least_squares`
            method.

        Returns
        -------
        result : scipy.optimize.OptimizeResult object
            See the documentation for the description. The two important
            fields are `x`: the final parameter vector, and `cost`: the final
            cost function.
        """
        def cost_func(twiss_params):
            self.fit_twiss4D(twiss_params)
            return self._mismatch_error(lattice)

        result = opt.least_squares(cost_func, self.twiss4D(),
                                   bounds=twiss_bounds, **kwargs)
        self.fit_twiss4D(result.x)
        return result

    def _match_replace_by_avg(self, lattice, nturns_avg=15, max_iters=20,
                              tol=1e-6, ftol=1e-8, xtol=1e-8, verbose=0):
        """Simple 4D matching algorithm.
        
        The method works be tracking the beam for a number of turns, then
        computing the average of the mismatch oscillations. The average is used
        to generate the beam for the next iteration, and this is repeated
        until convergence.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be in
            place.
        nturns : int
            Number of turns after which to enforce the matching condition.
        nturns_avg : int
            Number of turns to average over when updating the parameter vector.
        max_iters : int
            Maximum number of iterations to perform.
        tol : float
            Tolerance for the value of the cost function C = 1e12 *
            |sigma1 - sigma0|**2, where sigma0 and sigma1 are the initial and
            final moment vectors, respectively.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the parameter vector
            norm.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations
        """
        def get_avg_p():
            p_tracked = []
            for _ in range(nturns_avg + 1):
                p_tracked.append(self.twiss4D())
                self.track(lattice)
            return np.mean(p_tracked, axis=0)
            
        def is_converged(cost, cost_reduction, step_norm, p):
            converged, message = False, 'Did not converge.'
            if cost < tol:
                converged == True
                msg = '`tol` termination condition is satisfied.'
            if abs(cost_reduction) < ftol * cost:
                converged = True
                msg = '`ftol` termination condition is satisfied.'
            if abs(step_norm) < xtol * (xtol + la.norm(p)):
                converged = True
                message = '`xtol` termination condition is satisfied.'
            return converged, message
                
        if self.perveance == 0:
            return self.match_bare(lattice, '2D')
                    
        iteration = 0
        old_p, old_cost = self.twiss4D(), +np.inf
        history = [old_p]
        converged, message = False, 'Did not converge.'
        
        t_start = time.time()
        if verbose == 2:
            print_header()
        while not converged and iteration < max_iters:
            iteration += 1
            p = get_avg_p()
            self.fit_twiss4D(p)
            cost = self._mismatch_error(lattice, ssq=True)
            cost_reduction = cost - old_cost
            step_norm = la.norm(p - old_p)
            converged, message = is_converged(cost, cost_reduction,
                                              step_norm, p)
            old_p, old_cost = p, cost
            history.append(p)
            if verbose == 2:
                print_iteration(iteration, cost, cost_reduction, step_norm)
        t_end = time.time()
        
        if verbose > 0:
            tprint(message, 3)
            tprint('cost = {:.4e}'.format(cost), 3)
            tprint('iters = {}'.format(iteration), 3)
        return MatchingResult(p, cost, iteration, t_end-t_start,
                              message, history)

    def _match_free(self, lattice):
        """Match by varying the envelope parameters without constraints. The
        intrinsic emittance of the beam will not be conserved.
        """
        def cost(params):
            self.params = params
            return self._mismatch_error(lattice)
            
        return opt.least_squares(cost, self.params, verbose=2, xtol=1e-12)

    def perturb(self, radius=0.1):
        """Randomly perturb the 4D Twiss parameters."""
        if radius == 0:
            return
        lo, hi = 1 - radius, 1 + radius
        ax, ay, bx, by, u, nu = self.twiss4D()
        ax_min, ax_max = lo*ax, hi*ax
        ay_min, ay_max = lo*ay, hi*ay
        bx_min, bx_max = lo*bx, hi*bx
        by_min, by_max = lo*by, hi*by
        u_min, u_max = lo*u, hi*u
        nu_min, nu_max = lo*nu, hi*nu
        if bx_min < 0.1:
            bx_min = 0.1
        if by_min < 0.1:
            by_min = 0.1
        if u_min < 0.05:
            u_min = 0.05
        if u_max > 0.95:
            u_max = 0.95
        if nu_min < 0.05 * np.pi:
            nu_min = 0.05 * np.pi
        if nu_max > 0.95 * np.pi:
            nu_max = 0.95 * np.pi
        ax = np.random.uniform(ax_min, ax_max)
        ay = np.random.uniform(ay_min, ay_max)
        bx = np.random.uniform(bx_min, bx_max)
        by = np.random.uniform(by_min, by_max)
        u = np.random.uniform(u_min, u_max)
        nu = np.random.uniform(nu_min, nu_max)
        twiss_params = (ax, ay, bx, by, u, nu)
        self.fit_twiss4D(twiss_params)
        
    def print_twiss2D(self, indent=4):
        (ax, ay, bx, by), (ex, ey) = self.twiss2D(), self.emittances()
        print '2D Twiss parameters:'
        tprint('ax, ay = {:.3f}, {:.3f} [rad]'.format(ax, ay))
        tprint('bx, by = {:.3f}, {:.3f} [m]'.format(bx, by))
        tprint('ex, ey = {:.3e}, {:.3e} [m*rad]'.format(ex, ey))
    
    def print_twiss4D(self):
        ax, ay, bx, by, u, nu = self.twiss4D()
        print '4D Twiss parameters:'
        tprint('mode = {}'.format(self.mode))
        tprint('e{} = {:.3e} [m*rad]'.format(self.mode, self.eps))
        tprint('ax, ay = {:.3f}, {:.3f} [rad]'.format(ax, ay))
        tprint('bx, by = {:.3f}, {:.3f} [m]'.format(bx, by))
        tprint('u = {:.3f}'.format(u))
        tprint('nu = {:.3f} [deg]'.format(np.degrees(nu)))
            
            
            
class MatchingResult:
    """Class to store the results of the matching algorithm"""
    def __init__(self, p, cost, iters, runtime, message, history):
        self.p, self.cost, self.iters, self.time = p, cost, iters, runtime
        self.message = message
        self.history = np.array(history)

def print_header():
    print '{0:^15}{1:^15}{2:^15}{3:^15}'.format(
        'Iteration', 'Cost', 'Cost reduction', 'Step norm')

def print_iteration(iteration, cost, cost_reduction, step_norm):
    if cost_reduction is None:
        cost_reduction = ' ' * 15
    else:
        cost_reduction = '{0:^15.3e}'.format(cost_reduction)

    if step_norm is None:
        step_norm = ' ' * 15
    else:
        step_norm = '{0:^15.2e}'.format(step_norm)

    print '{0:^15}{1:^15.4e}{2}{3}'.format(
        iteration, cost, cost_reduction, step_norm)
