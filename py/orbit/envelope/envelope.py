"""
This module contains functions related to the envelope parameterization of
the Danilov distribution.
"""

# 3rd party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from tqdm import trange, tqdm

# PyORBIT modules
from bunch import Bunch
from orbit.analysis import covmat2vec
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
    twiss_at_injection,
    is_stable,
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

# Define bounds on BL twiss parameters
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
    params : NumPy array, shape (8,)
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    mass : float
        The particle mass [GeV/c^2].
    energy : float
        The kinetic energy per particle [GeV].
    eps : float
        The rms mode emittance of the beam [m*rad].
    mode : int
        Whether to choose eps2=0 (mode 1) or eps1=0 (mode 2).
    """
    
    def __init__(self, mass=1., energy=1., eps=1., mode=1, params=None):
        self.mass = mass
        self.energy = energy
        self.eps = eps
        self.mode = mode
        if params is not None:
            self.params = np.array(params)
        else:
            r = np.sqrt(4 * eps/2)
            self.params = np.array([r, 0, 0, r, 0, -r, r, 0])

    def norm(self):
        """Return envelope to normalized frame.
        
        In this frame the covariance matrix is diagonal, and the x-x' and y-y' 
        emittances are the mode emittances.
        """
        r_n = np.sqrt(4 * self.eps)
        if self.mode == 1:
            self.params = np.array([r_n, 0, 0, r_n, 0, 0, 0, 0])
        elif self.mode == 2:
            self.params = np.array([0, 0, 0, 0, 0, r_n, r_n, 0])
            
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
        
    def swap_xy(self):
        """Exchange (x, x') <-> (y, y')."""
        self.params[:4], self.params[4:] = self.params[4:], self.params[:4]
        
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
        
    def twissBL(self):
        """Return the mode Twiss parameters, as defined by Bogacz & Lebedev."""
        ax, ay, bx, by, ex, ey = self.twiss()
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
        cx = np.sqrt(
            area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
        cy = np.sqrt(
            area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
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
        """Track the envelope through the lattice.
        
        The envelope parameters are updated after it is tracked.
        """
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
        return np.array([tune_x, tune_y])
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * covmat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12)
        return result.x
        
    def fit_twissBL(self, twiss_params):
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
        self.norm()
        ax, ay, bx, by, u, nu = twiss_params
        V = BL.Vmat(ax, ay, bx, by, u, nu, self.mode)
        self.transform(V)
        
    def get_part_coords(self, psi=0):
        """Return the coordinates of a single particle on the envelope."""
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
        nparts = int(nparts)
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
        
    def to_bunch(self, nparts=0, bunch_length=0):
        """Add the envelope parameters to a Bunch object.
        
        Parameters
        ----------
        nparts : int
            The number of particles in the bunch. The bunch will just hold the
            envelope parameters if nparts == 0.
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
        bunch, params_dict = initialize_bunch(self.mass, self.energy)
        a, b, ap, bp, e, f, ep, fp = self.params
        bunch.addParticle(a, ap, e, ep, 0., 0.)
        bunch.addParticle(b, bp, f, fp, 0., 0.)
        if nparts > 0:
            for (x, xp, y, yp) in self.generate_dist(nparts):
                z = np.random.random() * bunch_length
                bunch.addParticle(x, xp, y, yp, z, 0.)
        return bunch, params_dict
        
    def match_barelattice(self, lattice):
        """Match to the lattice without space charge."""
        M = self.transfer_matrix(lattice)
        if not is_stable(M):
            print 'WARNING: bare lattice transfer matrix is not stable.'
        Sigma = BL.matched_Sigma_onemode(M, self.eps, self.mode)
        self.fit_cov(Sigma)
        
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
        possible to the matched solution. This can be done using the 
        `match_ramp_sc` or `match_ramp_tilt` method below.
        
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
        def cost(twiss_params, lattice, nturns=1):
            self.fit_twissBL(twiss_params)
            Sigma0 = self.cov()
            self.track(lattice)
            Sigma1 = self.cov()
            return 1e12 * covmat2vec(Sigma1 - Sigma0)
    
        result = opt.least_squares(
            cost,
            self.twissBL(),
            args=(lattice, nturns),
            bounds=bounds,
            verbose=verbose,
            xtol=1e-12
        )
        self.fit_twissBL(result.x)
        return result.cost
    
    
    def match_ramp_sc(self, lattice, solver_nodes, intensity, nturns=1,
                     stepsize=1e-12, tol=1e-2, display=False, progbar=False):
        """Match by slowly ramping the intensity.
        
        This method exists because the least squares optimizer fails to
        converge in lattices with skew quads when the skew strength is large
        for certain values of the envelope mode number. The approach here is
        to start with a beam which is matched to the bare lattice, then slowly
        increase the intensity, matching at each step. In this way the matched
        beam should remain close to the seed value.
        
        This works, but the step size must be quite small (~1e-12), otherwise
        it will randomly fail. If it fails, the method will cut the step size
        in half and restart from the last known match.
        
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be
            in place.
        solver_nodes : list[EnvSolverNode]
            The list of envelope solver nodes in the lattice.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        intensity : float
            The beam intensity.
        stepsize : float
            The intensity step size. A safe value is 1e12.
        tol : float
            The maximum value of the cost function.
        display : bool
            Whether to print the result at each step.
        progbar : bool
            Whether to use a tqdm progress bar.
        """
        lattice_length = lattice.getLength()

        def match(I):
            Q = hf.get_perveance(self.energy, self.mass, I/lattice_length)
            set_perveance(solver_nodes, Q)
            if Q == 0:
                self.match_barelattice(lattice)
                cost = 0.
            else:
                cost = self._match(lattice, nturns)
            return cost
       
        def init_intensities(stepsize, start=0.):
            if intensity == 0:
                return [0]
            if abs(stepsize) > intensity:
                stepsize = intensity
            nsteps = int(intensity / stepsize) + 1
            return np.linspace(start, intensity, nsteps)
        
        converged = False
        last_matched_I, last_matched_params = 0, np.copy(self.params)
        
        while not converged:
            intensities = init_intensities(stepsize, start=last_matched_I)
            if progbar:
                intensities = tqdm(intensities)
            for i, I in enumerate(intensities):
                cost = match(I)
                if display:
                    tprint('I = {:.2e}, cost = {:.2e}'.format(I, cost), 4)
                converged = cost < tol
                if converged:
                    last_matched_params = np.copy(self.params)
                    last_matched_I = I
                else:
                    # Stop. Restart from last known matched intensity with
                    # a smaller step size.
                    stepsize *= 0.5
                    self.params = last_matched_params
                    tprint(
                        'FAILED. Trying stepsize={:.2e}.'.format(stepsize), 8)
                    break
                    
    def match_ramp_tilt(self, lattice, angle, nturns, stepsize=0.1,
                        tol=1e-2, display=False, progbar=False):
        """Match in skew quad lattice by slowly ramping the quad tilt angle.
        
        Tight now this method only handles a single FODO cell. For some reason
        ramping the skew strength works much better than ramping the intensity.
                
        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be
            in place. Right now it can only handle a lattice with one focusing
            quad and one defocusing quad.
        angle : float
            The focusing quad will be tilted by `angle` degrees, while the 
            defocusing quad will be tilted by -`angle` degrees.
        nturns : int
            The number of passes throught the lattice before the matching
            condition is enforced.
        stepsize : float
            The step size for the quad tilt angle.
        tol : float
            The maximum value of the cost function.
        display : bool
            Whether to print the result at each step.
        progbar : bool
            Whether to use a tqdm progress bar.
        """
        qf = lattice.getNodeForName('qf')
        qd = lattice.getNodeForName('qd')
        
        def tilt_quads(angle):
            qf.setTiltAngle(+angle)
            qd.setTiltAngle(-angle)
            
        def init_angles(stepsize, start=0.):
            if angle == 0:
                return [0]
            if abs(stepsize) > abs(angle):
                stepsize = abs(angle)
            nsteps = int(abs(angle) / stepsize) + 1
            return np.linspace(start, angle, nsteps)
            
        converged = False
        last_matched_angle, last_matched_params = 0., np.copy(self.params)
        
        while not converged:
            angles = init_angles(stepsize, start=last_matched_angle)
            if progbar:
                angles = tqdm(angles)
            for phi in angles:
                tilt_quads(np.radians(phi))
                cost = self._match(lattice, nturns)
                if display:
                    tprint('angle = {:.2f}, cost = {:.2e}'.format(phi, cost), 4)
                converged = cost < tol
                if converged:
                    last_matched_params = np.copy(self.params)
                    last_matched_angle = phi
                else:
                    # Stop. Restart from last known matched angle with a
                    # smaller step size.
                    stepsize *= 0.5
                    self.params = last_matched_params
                    tprint(
                        'FAILED. Trying stepsize={:.2e}.'.format(stepsize), 8)
                    break
                        
    def print_twiss(self):
        """Print the horizontal and vertical Twiss parameters."""
        ax, ay, bx, by, ex, ey = self.twiss()
        nu = self.phase_diff()
        print 'ax, ay = {:.2f}, {:.2f} rad'.format(ax, ay)
        print 'bx, by = {:.2f}, {:.2f} m'.format(bx, by)
        print 'ex, ey = {:.2e}, {:.2e} mm*mrad'.format(1e6*ex, 1e6*ey)
        print 'nu = {:.2f} deg'.format(np.degrees(nu))
