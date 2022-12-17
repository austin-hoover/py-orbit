"""Matrix lattice for coupled systems."""
import math
import os

import numpy as np

from bunch import Bunch
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccNodeBunchTracker
from orbit.matrix_lattice.BaseMATRIX import BaseMATRIX
from orbit.matrix_lattice.MATRIX_Lattice import MATRIX_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.twiss import twiss
from orbit.utils import orbitFinalize
from orbit.utils.consts import speed_of_light
from orbit.utils.utils import orbit_matrix_to_numpy
from orbit_utils import Matrix


class MATRIX_Lattice_Coupled(MATRIX_Lattice):
    """MATRIX_Lattice for linearly coupled lattices."""

    def __init__(self, name=None, parameterization='LB'):
        MATRIX_Lattice.__init__(self, name)
        self.parameterization = parameterization

    def getRingParametersDict(self, momentum=None, mass=None):
        """Analyze the one-turn transfer matrix.

        Parameters
        ----------
        momentum : float
            Syncronous particle momentum [GeV / c].
        mass : float
            Particle mass [GeV / c^2].
        parameterization : str
            The parameterization to use.

        Returns
        -------
        dict
            * 'kin_energy': syncronous particle kinetic energy [GeV]
            * 'mass': particle mass [GeV / c]
            * 'momentum': syncronous particle momentum [GeV / c]
            * 'eigvals' : Eigenvalues of transverse matrix.
            * 'eigvecs' : Eigenvectors of transverse matrix.
            * 'eigtunes' : Transverse tunes computed from the eigenvalues.
            * 'stable' : Whether the transverse motion is bounded.
            * 'coupled`: Whether the transverse motino is coupled.
            * Transverse parameters depend on the parameterization.
        """
        params = dict()

        # Synchonous particle
        energy = math.sqrt(momentum**2 + mass**2)
        beta = momentum / energy
        gamma = energy / mass
        kin_energy = energy - mass
        params["momentum"] = momentum
        params["mass"] = mass
        params["kin_energy"] = kin_energy

        # Longitudinal parameters
        ring_length = self.getLength()
        period = ring_length / (beta * speed_of_light)
        params["period"] = period
        params["frequency"] = 1.0 / period
        ## I am not yet sure how to handle dispersion/chromaticity when the transverse
        ## motion is x-y coupled. I assume this is done elsewhere, such as in
        ## BMAD or MADX using the Edwards-Teng parameterization.

        # Transverse parameters
        M = orbit_matrix_to_numpy(self.oneTurnMatrix)
        tmat = None
        if self.parameterization == 'LB':
            tmat = twiss.LebedevBogacz(M)
        else:
            raise ValueError('Invalid parameterization.')
        params["eigvals"] = tmat.eigvals
        params["eigvecs"] = tmat.eigvecs
        params["eigtunes"] = tmat.eigtunes
        params["stable"] = tmat.stable
        params["coupled"] = tmat.coupled
        params.update(**tmat.params)

        return params

    def trackTwiss(self):
        raise NotImplementedError
    
    def trackDispersion(self):
        raise NotImplementedError