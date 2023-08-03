"""Envelope model for the KV distribution ({2, 0} Danilov distribution)."""
from __future__ import print_function
import copy
import time

import numpy as np
import scipy.optimize
from tqdm import tqdm

from bunch import Bunch
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import TwissContainer
from orbit.envelope import utils
from orbit.matrix_lattice.parameterizations import courant_snyder as CS
from orbit.matrix_lattice.transfer_matrix import TransferMatrix
from orbit.matrix_lattice.transfer_matrix import TransferMatrixCourantSnyder
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils import consts

import utils


class DanilovEnvelope20:
    """Class for the beam envelope of the Danilov distribution.

    Attributes
    ----------
    params : ndarray, shape(4,)
        The envelope parameters [cx, cx', cy, cy'].
    eps_x : float
        The rms emittance of the x-x' distribution: sqrt(<xx><x'x'> - <xx'><xx'>).
    eps_y : float
        The rms emittance of the y-y' distribution: sqrt(<yy><y'y'> - <yy'><yy'>).
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    intensity : float
        Number of particles in the bunch represented by the envelope.
    length : float
        Bunch length [m].
    perveance : float
        Dimensionless beam perveance.
    """
    def __init__(
        self,
        eps_x=1.0,
        eps_y=1.0,
        mass=consts.mass_proton,
        kin_energy=1.0,
        length=1.0,
        intensity=0.0,
        params=None,
    ):
        self.eps_x_rms = eps_x
        self.eps_y_rms = eps_y
        self.eps_x = 4.0 * eps_x
        self.eps_y = 4.0 * eps_y
        self.mass = mass
        self.kin_energy = kin_energy
        self.length = length
        self.set_intensity(intensity)
        if params is None:
            cx = 2.0 * np.sqrt(self.eps_x)
            cy = 2.0 * np.sqrt(self.eps_y)
            self.params = [cx, 0.0, cy, 0.0]
        else:
            self.params = params
        self.params = np.array(self.params)
        
    def set_params(self, params):
        self.params[:] = params
        
    def copy(self):
        """Produced a deep copy of the envelope."""
        return copy.deepcopy(self)

    def set_intensity(self, intensity):
        """Set beam intensity and re-calculate perveance."""
        self.intensity = intensity
        self.line_density = intensity / self.length
        self.perveance = utils.get_perveance(self.mass, self.kin_energy, self.line_density)

    def set_length(self, length):
        """Set beam length and re-calculate perveance."""
        self.length = length
        self.set_intensity(self.intensity)
        
    def cov(self):
        """Return 4 x 4 transverse covariance matrix.
        
        See Table II here: https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.7.024801.
        Note the typo for <xx'> and <yy>, the first time should be rx'^2 not rx^2.
        """
        (cx, cxp, cy, cyp) = self.params
        Sigma = np.zeros((4, 4))
        Sigma[0, 0] = cx ** 2
        Sigma[2, 2] = cy ** 2
        Sigma[0, 1] = cx * cxp
        Sigma[2, 3] = cy * cyp
        Sigma[1, 1] = cxp**2 + (self.eps_x / cx)**2
        Sigma[3, 3] = cyp**2 + (self.eps_y / cy)**2
        Sigma = Sigma * 0.25
        return Sigma
    
    def twiss(self):
        """Return (alpha_x, beta_x, alpha_y, beta_y)."""
        Sigma = self.cov()
        alpha_x, beta_x = utils.twiss_2x2(Sigma[:2, :2])
        alpha_y, beta_y = utils.twiss_2x2(Sigma[2:, 2:])
        return (alpha_x, beta_x, alpha_y, beta_y)        

    def from_bunch(self, bunch):
        """Extract the envelope parameters from a Bunch object."""
        self.params[0] = bunch.x(0)
        self.params[1] = bunch.xp(0)
        self.params[2] = bunch.y(0)
        self.params[3] = bunch.yp(0)
        return self.params

    def to_bunch(self, n_parts=0, no_env=False):
        """Add the envelope parameters to a Bunch object. The first
        particle represents the envelope parameters.

        Parameters
        ----------
        n_parts : int
            The number of particles in the bunch. The bunch will just hold the
            envelope parameters if n_parts == 0.
        no_env : bool
            If True, do not store envelope parameters in first particle.

        Returns
        -------
        bunch: Bunch object
            The bunch representing the distribution of size 2 + n_parts
            (unless `no_env` is True).
        params_dict : dict
            The dictionary of parameters for the bunch.
        """
        bunch = Bunch()
        bunch.mass(self.mass)
        bunch.getSyncParticle().kinEnergy(self.kin_energy)
        params_dict = {"bunch": bunch}
        if not no_env:
            cx, cxp, cy, cyp = self.params
            bunch.addParticle(cx, cxp, cy, cyp, 0.0, 0.0)
        alpha_x, beta_x, alpha_y, beta_y = self.twiss()
        dist = KVDist2D(
            TwissContainer(alpha_x, beta_x, self.eps_x_rms),
            TwissContainer(alpha_y, beta_y, self.eps_y_rms),
        )
        for _ in range(n_parts):
            (x, xp, y, yp) = dist.getCoordinates()
            z = np.random.uniform(0.0, self.length)
            bunch.addParticle(x, xp, y, yp, z, 0.0)
        if n_parts > 0:
            bunch.macroSize(self.intensity / n_parts if self.intensity > 0.0 else 1.0)
        return bunch, params_dict

    def track(self, lattice, n_turns=1, n_test_parts=0, progbar=False):
        """Track the envelope through the lattice.
        
        If `ntestparts` is nonzero, test particles will be tracked which receive
        linear space charge kicks based on the envelope parameters.
        """
        bunch, params_dict = self.to_bunch(n_test_parts)
        turns = range(n_turns)
        if progbar:
            turns = tqdm(turns)
        for turn in turns:
            lattice.trackBunch(bunch, params_dict)
        self.from_bunch(bunch)

    def track_store_params(self, lattice, n_turns=1):
        """Track and return the turn-by-turn envelope parameters."""
        params_tbt = [self.params]
        for _ in range(n_turns):
            self.track(lattice)
            params_tbt.append(self.params)
        return params_tbt

    def get_transfer_matrix(self, lattice):
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
        M : ndarray, shape (4, 4)
            The 4x4 linear transfer matrix of the combined lattice + space
            charge focusing system.
        """
        # If space charge is zero, we can just use the TEAPOT_MATRIX_Lattice
        # class to calculate the transfer matrix.
        if self.perveance == 0:
            M = utils.get_transfer_matrix(lattice, self.mass, self.kin_energy)
            return M[:4, :4]

        # The envelope parameters will change if the beam is not matched to
        # the lattice, so make a copy of the intial state.
        env = self.copy()

        step_arr_init = np.full(6, 1.0e-6)
        step_arr = np.copy(step_arr_init)
        step_reduce = 20.0
        bunch, params_dict = env.to_bunch()
        bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(step_arr[0] / step_reduce, 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, step_arr[1] / step_reduce, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, step_arr[2] / step_reduce, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, 0.0, step_arr[3] / step_reduce, 0.0, 0.0)
        bunch.addParticle(step_arr[0], 0.0, 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, step_arr[1], 0.0, 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, step_arr[2], 0.0, 0.0, 0.0)
        bunch.addParticle(0.0, 0.0, 0.0, step_arr[3], 0.0, 0.0)

        lattice.trackBunch(bunch, params_dict)
        X = [
            [bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)]
            for i in range(1, bunch.getSize())
        ]
        X = np.array(X)            
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                x1 = step_arr[i] / step_reduce
                x2 = step_arr[i]
                y0 = X[0, j]
                y1 = X[i + 1, j]
                y2 = X[i + 1 + 4, j]
                M[j, i] = ((y1 - y0) * x2 * x2 - (y2 - y0) * x1 * x1) / (x1 * x2 * (x2 - x1))
        return M

    def match_bare(self, lattice, solver_nodes=None):
        """Match to the lattice without space charge.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice in which to match. If envelope solver nodes nodes are
            in the lattice, a list of these nodes needs to be passed as the
            `solver_nodes` parameter so they can be turned off/on.
        solver_nodes : list, optional
            List of nodes which are sublasses of SC_Base_AccNode. If provided,
            all space charge nodes are turned off, then the envelope is matched
            to the bare lattice, then all space charge nodes are turned on.

        Returns
        -------
        ndarray, shape (4,)
            The matched envelope parameters.
        """
        if solver_nodes is not None:
            for node in solver_nodes:
                node.active = False

        M = self.get_transfer_matrix(lattice)
        tmat = TransferMatrix(M)
        if not tmat.stable:
            print("WARNING: transfer matrix is not stable.")

        tmat = TransferMatrixCourantSnyder(M)
        tmat.analyze()
        alpha_x = tmat.params["alpha_x"]
        alpha_y = tmat.params["alpha_y"]
        beta_x = tmat.params["beta_x"]
        beta_y = tmat.params["beta_y"]
        sig_xx = self.eps_x_rms * beta_x 
        sig_yy = self.eps_y_rms * beta_y
        sig_xxp = -self.eps_x_rms * alpha_x
        sig_yyp = -self.eps_y_rms * alpha_y
        cx = 2.0 * np.sqrt(sig_xx)
        cy = 2.0 * np.sqrt(sig_yy)
        cxp = 4.0 * sig_xxp / cx
        cyp = 4.0 * sig_yyp / cy
        self.set_params([cx, cxp, cy, cyp])

        if solver_nodes is not None:
            for node in solver_nodes:
                node.active = True
                
        return self.params
        
    def match_lsq(self, lattice, **kws):
        """Compute matched envelope using scipy.least_squares optimizer.

        Parameters
        ----------
        lattice : TEAPOT_Lattice object
            The lattice to match into. The solver nodes should already be in place.
        **kws
            Key word arguments passed to `scipy.optimize.least_squares`.
            
        Returns
        -------
        result : scipy.optimize.OptimizeResult
            See scipy documentation.
        """        
        def cost_function(x):   
            x0 = x.copy()
            self.params = x
            self.track(lattice)
            residuals = self.params - x0
            residuals = 1000.0 * residuals
            return residuals
        
        kws.setdefault("xtol", 1.00e-12)
        kws.setdefault("ftol", 1.00e-12)
        kws.setdefault("gtol", 1.00e-12)

        lb = [1.00e-12, -np.inf, 1.00e-12, -np.inf]
        result = scipy.optimize.least_squares(
            cost_function,
            self.params.copy(),
            bounds=(lb, np.inf),
            **kws
        )
        self.params = result.x
        return result
    
    def match_lsq_ramp_intensity(self, lattice, solver_nodes=None, n_steps=10, **kws):
        self.match_bare(lattice, solver_nodes=solver_nodes)
        if self.perveance == 0.0:
            return
        verbose = kws.get("verbose", 0)
        max_intensity = self.intensity
        for intensity in np.linspace(0.0, max_intensity, n_steps):
            self.set_intensity(intensity)
            for solver_node in solver_nodes:
                solver_node.set_perveance(self.perveance)
            result = self.match_lsq(lattice, **kws)
            self.set_params(result.x)
            if verbose > 0:
                print("intensity = {:.2e}".format(intensity))
        return result