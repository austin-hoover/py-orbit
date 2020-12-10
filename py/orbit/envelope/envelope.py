"""
This module contains functions related to the envelope parameterization of
the Danilov distribution.

To do
-----
* For `match_barelattice` method, automatically figure out if the lattice is
  coupled or has unequal phase advances from its transfer matrix. If it does
  not, then just match in the 2D sense. Otherwise, match in the 4D sense.
"""

# 3rd party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange, tqdm

# PyORBIT modules
from bunch import Bunch
from orbit.analysis.analysis import covmat2vec
from orbit.coupling import bogacz_lebedev as BL
from orbit.space_charge.envelope import (
    set_env_solver_nodes,
    set_perveance
)
from orbit.utils import helper_funcs as hf
from orbit.utils.helper_funcs import (
    initialize_bunch,
    tprint,
    transfer_matrix,
    is_stable,
    unequal_eigtunes,
    twiss_at_injection,
    fodo_lattice
)

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

# Define bounds on 4D twiss parameters
pad = 1e-5
nu_min, nu_max = pad, np.pi - pad
u_min, u_max = pad, 1 - pad
alpha_min, alpha_max = -np.inf, np.inf
beta_min, beta_max = pad, np.inf
lb = (alpha_min, alpha_min, beta_min, beta_min, u_min, nu_min)
ub = (alpha_max, alpha_max, beta_max, beta_max, u_max, nu_max)
bounds = (lb, ub)


class Envelope:
    """Class for the Danilov distribution envelope.
    
    This class includes methods to compute the beam statistics and interact
    with PyORBIT using the envelope parameterization.
    
    Attributes
    ----------
    eps : float
        The rms mode emittance of the beam [m*rad].
    mode : int
        Whether to choose eps2=0 (mode 1) or eps1=0 (mode 2).
    ex_ratio : float
        The x emittance ratio, such that ex = epsx_ratio * eps
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
    params : list (optional)
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    """
    def __init__(self, eps=1., mode=1, ex_ratio=0.5, mass=0.93827231,
                 energy=1., intensity=0., length=0, params=None):
        self.eps = eps
        self.mode = mode
        self.ex_ratio, self.ey_ratio = ex_ratio, 1 - ex_ratio
        self.mass = mass
        self.energy = energy
        self.length = length
        self.set_spacecharge(intensity)
        if params is not None:
            self.params = np.array(params)
            ex, ey = self.emittances()
            self.eps = ex + ey
            self.ex_ratio = ex / self.eps
        else:
            ex, ey = ex_ratio * eps, (1 - ex_ratio) * eps
            rx, ry = np.sqrt(4 * ex), np.sqrt(4 * ey)
            if mode == 1:
                self.params = np.array([rx, 0, 0, rx, 0, -ry, +ry, 0])
            elif mode == 2:
                self.params = np.array([rx, 0, 0, rx, 0, +ry, -ry, 0])
        
    def set_params(self, a, b, ap, bp, e, f, ep, fp):
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        
    def set_spacecharge(self, intensity):
        self.intensity = intensity
        self.charge_density = intensity / self.length
        self.perveance = hf.get_perveance(self.mass, self.energy,
                                          self.charge_density)
                                          
    def set_length(self, length):
        self.length = length
        self.set_spacecharge(self.intensity)
        
    def get_norm_mat_2D(self, inv=False):
        """Return the normalization matrix (2D sense)."""
        ax, ay, bx, by, _, _ = self.twiss2D()
        V = norm_mat(ax, bx, ay, by)
        if inv:
            return la.inv(V)
        else:
            return V
        
    def norm(self):
        """Transform the envelope to normalized coordinates.
        
        In the transformed coordates the covariance matrix is diagonal, and the
        x-x' and y-y' emittances are the mode emittances.
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
        radius sqrt(ex) and sqrt(ey). The cross-plane elements of the
        covariance matrix will not all be zero. If `scale` is True, the x-x'
        and y-y' ellipses will be scaled to unit radius.
        """
        self.transform(self.get_norm_mat_2D(inv=True))
        if scale:
            ex, ey = 4 * self.emittances()
            self.params[:4] /= np.sqrt(ex)
            self.params[4:] /= np.sqrt(ey)
        return self.params
            
    def normed_params_2D(self):
        """Return the normalized envelope parameters in the 2D sense."""
        true_params = np.copy(self.params)
        normed_params = self.norm2D()
        self.params = true_params
        return normed_params
            
    def matrix(self):
        """Create the envelope matrix P from the envelope parameters.
        
        The matrix is defined by x = Pc, where x = [x, x', y, y']^T and
        c = [cos(psi), sin(psi)], with 0 <= psi <= 2pi. This is useful because
        any transformation to the particle coordinate vector x also done to P.
        For example, if x -> M.x, then P -> M.P.
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        """Convert the envelope matrix to vector form."""
        return P.flatten()
                
    def transform(self, M):
        """Apply matrix M to the coordinates."""
        P_new = np.matmul(M, self.matrix())
        self.params = self.to_vec(P_new)
        
    def norm_transform(self, M):
        """Normalize, then apply M to the coordinates."""
        self.norm()
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
        S = self.cov()
        ex = np.sqrt(la.det(S[:2, :2]))
        ey = np.sqrt(la.det(S[2:, 2:]))
        emittances = np.array([ex, ey])
        if mm_mrad:
            emittances *= 1e6
        return emittances
        
    def twiss2D(self):
        """Return the horizontal/vertical Twiss parameters and emittances."""
        S = self.cov()
        ex = np.sqrt(la.det(S[:2, :2]))
        ey = np.sqrt(la.det(S[2:, 2:]))
        bx = S[0, 0] / ex
        by = S[2, 2] / ey
        ax = -S[0, 1] / ex
        ay = -S[2, 3] / ey
        return np.array([ax, ay, bx, by, ex, ey])
        
    def twiss4D(self):
        """Return the mode Twiss parameters, as defined by Bogacz & Lebedev."""
        ax, ay, bx, by, ex, ey = self.twiss2D()
        if self.mode == 1:
            u = ey / self.eps
            bx *= (1 - u)
            ax *= (1 - u)
            by *= u
            ay *= u
        elif self.mode == 2:
            u = ex / self.eps
            bx *= u
            ax *= u
            by *= (1 - u)
            ay *= (1 - u)
        nu = self.phase_diff()
        return np.array([ax, ay, bx, by, u, nu])
        
    def tilt_angle(self, x1='x', x2='y'):
        """Return the ccw tilt angle in the x1-x2 plane."""
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
        phi = self.tilt_angle(x1, x2)
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
        cx = np.sqrt(
            area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
        cy = np.sqrt(
            area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
        return np.array([cx, cy])
        
    def phases(self):
        """Return the horizontal/vertical phases (in range [0, 2pi] of a
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
        The value returned is in the range [0, pi].
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        return nu if nu < np.pi else 2*np.pi - nu
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * covmat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12)
        return result.x
        
    def fit_twiss2D(self, ax, ay, bx, by, ex_ratio):
        """Fit the envelope to the 2D Twiss parameters."""
        self.norm2D(scale=True)
        V = norm_mat(ax, bx, ay, by)
        ex, ey = ex_ratio * self.eps, (1 - ex_ratio) * self.eps
        A = np.sqrt(4 * np.diag([ex, ex, ey, ey]))
        self.transform(np.matmul(V, A))
        
    def fit_twiss4D(self, twiss_params):
        """Fit the envelope to the BL Twiss parameters.
        
        `twiss_params` is an array containing the Bogacz-Lebedev Twiss
        parameters for a single mode: [ax, ay, bx, by, u, nu], where
            
        ax{y} : float
            The horizontal{vertical} alpha function.
        bx{y} : float
            The horizontal{vertical} beta function.
        u : float
            The coupling parameter in range [0, 1]. This is equal to ey/e1
            when mode=1 or ex/e2 when mode=2.
        nu : float
            The x-y phase difference in range [0, pi].
        """
        ax, ay, bx, by, u, nu = twiss_params
        V = BL.Vmat(ax, ay, bx, by, u, nu, self.mode)
        self.norm_transform(V)
        
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

        Parameters
        ----------
        nparts : int
            The number of particles in the bunch.

        Returns
        -------
        X : NumPy array, shape (nparts, 4)
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
        
    def avoid_zero_emittance(self, padding=1e-6):
        """If the x or y emittance is truly zero, make it slightly nonzero.
        
        This will occur if the bare lattice has unequal tunes and we call
        `match_barelattice(lattice, method='4D').
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        ex, ey = self.emittances()
        if ex == 0:
            a, bp = padding, padding
        if ey == 0:
            if self.mode == 1:
                f, ep = -padding, +padding
            elif self.mode == 2:
                f, ep = +padding, -padding
        self.set_params(a, b, ap, bp, e, f, ep, fp)
        
    def from_bunch(self, bunch):
        """Extract the envelope parameters from a Bunch object."""
        a, ap, e, ep = bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)
        b, bp, f, fp = bunch.x(1), bunch.xp(1), bunch.y(1), bunch.yp(1)
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        return self.params
        
    def to_bunch(self, nparts=0, length=0, no_env=False):
        """Add the envelope parameters to a Bunch object. The first two
        particles represent the envelope parameters.
        
        Parameters
        ----------
        nparts : int
            The number of particles in the bunch. The bunch will just hold the
            envelope parameters if nparts == 0.
        length : float
            The length of the bunch [meters].
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
            z = np.random.random() * length
            bunch.addParticle(x, xp, y, yp, z, 0.)
        return bunch, params_dict
        
    def track(self, lattice, ntestparts=0, nturns=1):
        """Track the envelope through the lattice.
        
        The envelope parameters are updated after it is tracked. If
        `ntestparts` is nonzero, test particles will be tracked which receive
        linear space charge kicks based on the envelope parmaeters.
        """
        bunch, params_dict = self.to_bunch(ntestparts, lattice.getLength())
        turns = tqdm(range(nturns)) if nturns > 1 else range(nturns)
        for _ in turns:
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
        return np.array([tune_x, tune_y])
        
    def match_barelattice(self, lattice, method='auto', avoid_zero_eps=True):
        """Match to the lattice without space charge.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into (no solver nodes).
        method : str
            If '4D', match to the lattice using the eigenvectors of the
            transfer matrix. This may result in the beam being completely
            flat, for example when the lattice is uncoupled. TO DO: add method
            to make sure the beam never has exactly zero area in x-y space.
            The '2D' method will only match the x-x' and y-y' ellipses of the
            beam.
            
        Returns
        -------
        NumPy array, shape (8,)
            The matched envelope parameters.
        """
        M = self.transfer_matrix(lattice)
        if not is_stable(M):
            print 'WARNING: transfer matrix is not stable.'
        lat_params = hf.params_from_transfer_matrix(M)
        ax, ay = [lat_params[key] for key in ('alpha_x','alpha_y')]
        bx, by = [lat_params[key] for key in ('beta_x','beta_y')]
        if method == 'auto':
            method = '4D' if unequal_eigtunes(M) else '2D'
        if method == '2D':
            self.fit_twiss2D(ax, ay, bx, by, self.ex_ratio)
        elif method == '4D':
            eigvals, eigvecs = la.eig(M)
            self.norm_transform(BL.construct_V(eigvecs))
            if avoid_zero_eps:
                self.avoid_zero_emittance()
        return self.params
            
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
        
    def _match(self, lattice, nturns=1, verbose=0):
        """Run least squares optimization to find matched envelope.
        
        The envelope is fit to the solution returned by the optimizer. This
        method does not always converge to the matched solution. In
        particular, it will fail when the lattice skew strength is large. For
        this reason it is recommended that the initial seed be as close as
        possible to the matched solution.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be
            in place.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        verbose : int
            Whether to diplay the optimizer progress (0, 1, or 2). 0 is no
            output, 1 shows the end result, and 2 shows each iteration.
            
        Returns
        -------
        result : scipy.optimize.OptimizeResult object
            See the documentation for the description. The two important
            fields are `x`: the final parameter vector, and `cost`: the final
            cost function.
        """
        def cost(twiss_params, nturns=1):
            self.fit_twiss4D(twiss_params)
            Sigma0 = self.cov()
            self.track(lattice)
            Sigma1 = self.cov()
            return 1e12 * covmat2vec(Sigma1 - Sigma0)
    
        result = opt.least_squares(
            cost,
            self.twiss4D(),
            args=(nturns,),
            bounds=bounds,
            verbose=verbose,
            xtol=1e-12
        )
        self.fit_twiss4D(result.x)
        return result.cost
    
    def match(self, lattice, solver_nodes=None, nturns=1, Qstep=None,
              tol=1e-2, max_fails=1000, Qstep_max=1e-3, Qstep_min=1e-8,
              win_thresh=100, Qstep_incr=1.5, Qstep_decr=2, display=False):
        """Match the envelope to the lattice by slowly ramping the intensity.
        
        This method starts by matching the beam (in the 2D sense) to the
        bare lattice. It then increases the intensity in steps, matching at
        each step. If it fails at any step, it will restart from the last
        known match with a smaller step size. The step size is increased after
        a number of successful matches.
        
        This method currently fails in some cases, such as when the matched
        beam has near zero area.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be
            in place.
        solver_nodes : list[EnvSolverNode]
            A list of the envelope solver nodes in the lattice. The method
            will adjust the strengths of these nodes.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        Qstep : float
            The perveance step size. If None, the initial step size is set
            to the final perveance.
        tol : float
            The beam is matched if the cost function is less than `tol`.
        max_fails : int
            The maximum number of failed matches.
        Qstep_max{Qstep_min} : float
            The maximum{minimum} perveance step size.
        win_thresh : int
            After `win_thresh` successful matches are performed, the step size
            will be increased.
        Qstep_incr : float
            The factor by which `Qstep` is increased if `win_thresh` is
            satisfied.
        display : bool
            Whether to print the result at each step.
            
        Returns
        -------
        NumPy array, shape (8,)
            The matched envelope parameters.
        """
        if self.perveance == 0 or solver_nodes is None:
            return self.match_barelattice(lattice)
            
        if Qstep is None:
            Qstep = self.perveance
            
        def attempt_to_match(Q):
            self.perveance = Q
            set_perveance(solver_nodes, Q)
            if Q == 0:
                # Match in 2D sense so that the beam never becomes a line
                # with zero area. This can cause problems.
                self.match_barelattice(lattice, '2D')
                cost = 0.
            else:
                cost = self._match(lattice, nturns)
            return cost
                
        winstreak = fails = 0
        Qfinal = self.perveance
        Q, last_matched_Q, last_matched_params = 0., 0., self.params
        stop = False
        
        while not stop:
            cost = attempt_to_match(Q)
            converged = cost < tol
            if display:
                tprint('Q = {:.2e}, cost = {:.2e}'.format(Q, cost), 4)
            if converged:
                # Store the matched params
                last_matched_params = np.copy(self.params)
                last_matched_Q = Q
                winstreak += 1
                # Increase the step size if the method has worked recently
                if winstreak > win_thresh and Qstep < Qstep_max:
                    Qstep *= Qstep_incr
                    winstreak = 0
                    if display:
                        tprint('Doing well. New Qstep = {:.2e}'.format(Qstep), 8)
            else:
                # Restart from last known match with a smaller step size.
                fails, winstreak = fails + 1, 0
                self.params, Q = last_matched_params, last_matched_Q
                if Qstep > Qstep_min:
                    Qstep /= Qstep_decr
                if display:
                    tprint('Failed. New Qstep = {:.2e}.'.format(Qstep), 8)
            # Check stop condition
            stop = (converged and Q == Qfinal) or fails > max_fails
            Q += Qstep
            if Q > Qfinal:
                Q = Qfinal
            if fails > max_fails:
                tprint('Maximum # of failures exceeded ({})'.format(fails), 8)
        return self.params
                                        
    def match_free(self, lattice):
        """Match by varying the envelope parameters without constraints."""
        def cost(params):
            self.params = params
            Sigma0 = self.cov()
            self.track(lattice)
            Sigma1 = self.cov()
            return 1e12 * covmat2vec(Sigma1 - Sigma0)
    
        result = opt.least_squares(cost, self.params, verbose=2, xtol=1e-12)
        return result.cost
        
    def perturb(self, radius=0.1):
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
                        
    def print_twiss(self, short=False, indent=4):
        """Print the horizontal and vertical Twiss parameters."""
        ax, ay, bx, by, ex, ey = self.twiss2D()
        ex, ey = 1e6 * np.array([ex, ey])
        nu = np.degrees(self.phase_diff())
        if short:
            tprint('twiss_params = {}'.format(
                np.round([ax, ay, bx, by, ex, ey], 2)), indent)
        else:
            print 'Envelope Twiss parameters:'
            tprint('ax, ay = {:.2f}, {:.2f} rad'.format(ax, ay), indent)
            tprint('bx, by = {:.2f}, {:.2f} m'.format(bx, by), indent)
            tprint('ex, ey = {:.2e}, {:.2e} mm*mrad'.format(ex, ey), indent)
            tprint('nu = {:.2f} deg'.format(nu), indent)
