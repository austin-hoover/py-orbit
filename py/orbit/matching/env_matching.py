import numpy as np
from scipy import optimize as opt
from tqdm import trange, tqdm

from bunch import Bunch
from spacecharge import EnvSolverDanilov
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.space_charge.envelope import setEnvAccNodes
from orbit.bunch_generators import TwissContainer, DanilovDist2D
from orbit.utils import helper_funcs as hf
    

def get_env_params(param_vec, e1, rot_dir):
    """Get envelope parameters from beam parameters."""
    ax, ay, bx, by, r, nu = param_vec
    ex, ey = r * e1, (1 - r) * e1
    dist = DanilovDist2D(
        TwissContainer(ax, bx, ex), 
        TwissContainer(ay, by, ey), 
        nu, rot_dir
    )
    return dist.params    
    
    
def get_moments(env_params):
    """Return 10 element moment vector from envelope parameters."""
    a, b, ap, bp, e, f, ep, fp = env_params
    M = np.array([[a, b, 0, 0], [ap, bp, 0, 0], [e, f, 0, 0], [ep, fp, 0, 0]])
    cov_mat = 0.25 * np.matmul(M, M.T)
    return hf.mat2vec(cov_mat)

    
class DanilovMatcher:
    """Class to find the matched envelope of the Danilov distribution."""

    def __init__(self, param_vec, latfile, latseq, mode_emittance, mass, 
                 energy, rot_dir='ccw', pad=1e-6):
        """Constructor.
        
        Parameters
        ----------
        param_vec : array-like
            The seed value for the beam parameters: [ax, ay, bx, by, r, nu]. 
            ax{y} : The alpha Twiss parameter -<xx'>/ex {-<yy'>/ey}.
            bx{y} : The beta Twiss parameter <x^2>/ex {<y^2>/ey}.
            r : The emittance ratio ex / (ex + ey).
            nu : The x-y phase difference (between 0 and pi).
        latfile : str
            The name of the MADX lattice file.
        latseq : str
            The name of the 'sequence' keyword in the MADX lattice file.
        mode_emittance : float
            The mode emittance ex + ey.
        mass : float
            The particle mass [GeV/c^2].
        energy : float
            The kinetic energy per particle [GeV].
        rot_dir : str
            The direction of rotation of the beam in normalized space ('cw' 
            or 'ccw').
        """
        self.param_vec = param_vec
        self.latfile, self.latseq = latfile, latseq
        self.e1 = mode_emittance
        self.mass = mass
        self.energy = energy
        self.rot_dir = rot_dir
        pad=1e-6
        self.bounds = (
            [-np.inf, -np.inf, pad, pad, pad, pad],
            [np.inf, np.inf, np.inf, np.inf, 1.0 - pad, np.pi - pad]
        )
        self.matches = []
        self.output_file = None

    def cost(self, param_vec, lattice, nturns=1, scale=1e12):
        """Cost function for optimizer.
        
        Parameters
        ----------
        param_vec : NumPy array, shape (6,) 
            The parameters used to generate the beam (see __init__).
        lattice : TEAPOT_Lattice object
            The lattice which tracks the bunch.
        nturns : int
            The number of turns to track before enforcing the matching 
            condition.
        scale : float
            A scaling factor for the output. 
        
        Returns
        -------
        NumPy array, shape (10,)
            The difference between final and intial moment vectors.
        """
        p_old = get_env_params(param_vec, self.e1, self.rot_dir)
        bunch, params_dict = hf.initialize_envelope(
            p_old, self.mass, self.energy
        )
        for _ in range(nturns):
            lattice.trackBunch(bunch, params_dict)
        p_new = hf.env_from_bunch(bunch)
        return scale * (get_moments(p_new) - get_moments(p_old))

    def match(self, 
        intensity, 
        nsteps, 
        nturns=1, 
        max_solver_spacing=0.01, 
        min_solver_spacing=1e-6, 
        verbose=0, 
        xtol=1e-12, 
        ftol=1e-12, 
        gtol=1e-12,
    ):
        """Compute the matched envelope.
        
        Parameters
        ----------
        intensity : float
            The beam intensity.
        nsteps : int
            If nsteps > 1, matching is performed at increasingly large
            intensities until the final intensity is reached, using the 
            matched beam at each step as the seed for the next iteration.
            This is to keep the initial guess close to the matched solution. 
        nturns : int
            The number of turns to track before enforcing the matching 
            condition.
        max_solver_spacing : float
            The maximum distance between solver nodes in the lattice.
        min_solver_spacing : float
            The minimum distance between solver nodes in the lattice.
        verbose : int
            Option for displaying the optimizer progress to the terminal. 
            Options are {0: no, 1: show summary, 2: show each iteration}.
        xtol, ftol, gtol : float
            Convergence tolerances. See scipy.optimize.least_squares
        
        Returns
        -------
        NumPy array, shape (6,) 
            The parameters used to generate the matched beam (see __init__).
        """
        intensities = np.linspace(0, intensity, nsteps + 1)[1:]
        self.matches = []

        for I in tqdm(intensities):

            # Create lattice with envelope solvers for given intensity 
            lattice = hf.lattice_from_file(self.latfile, self.latseq)
            lattice.split(max_solver_spacing)
            line_density = I/lattice.getLength()
            Q = hf.get_perveance(self.energy, self.mass, line_density)
            setEnvAccNodes(lattice, min_solver_spacing, EnvSolverDanilov(Q))
            
            # Search for the matched solution
            result = opt.least_squares(
                self.cost, self.param_vec, args=(lattice, nturns),
                bounds=self.bounds, xtol=xtol, ftol=ftol, gtol=gtol,
                verbose=verbose, max_nfev=int(1e4)
            )
            self.param_vec = result.x
            self.matches.append(self.param_vec)
            
        # Save matched envelope parameters at each intensity to a file. 
        # Columns are [I, a, b, a', b', e, f, e', f'].
        if self.output_file and len(self.matches) > 0:
            matches = [get_env_params(pvec, self.e1, self.rot_dir) 
                       for pvec in self.matches]
            matches = np.vstack([intensities, np.array(matches).T]).T
            np.savetxt(self.output_file, matches)
                
        return self.param_vec
    
    
class KVMatcher:
    """Class to find the matched envelope of the KV distribution."""
    
    def __init__(self):
        return