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
from orbit.matrix_lattice import transfer_matrix_analysis
from orbit.teapot_base import MatrixGenerator
from orbit.utils import orbitFinalize
from orbit.utils.consts import speed_of_light
from orbit_utils import Matrix


def orbit_matrix_to_numpy(matrix):
    """Return ndarray from two-dimensional orbit matrix."""
    array = np.zeros(matrix.size())
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.get(i, j)
    return array


class MATRIX_Lattice_Coupled(MATRIX_Lattice):
    """MATRIX_Lattice for linearly coupled lattices."""

    def __init__(self, name=None, parameterization='LB'):
        MATRIX_Lattice.__init__(self, name)
        self.parameterization = parameterization
        self.one_turn_matrices = dict()
        
    def getOneTurnMatrices(self):
        """Return one-turn transfer matrix at each node."""
        self.one_turn_matrices = dict()
        M = orbit_matrix_to_numpy(self.getOneTurnMatrix())
        for node in self.getNodes():
            if isinstance(node, BaseMATRIX):
                M_node = orbit_matrix_to_numpy(node.getMatrix())
                M_node_inv = np.linalg.inv(M_node)
                M = np.linalg.multi_dot([M_node, M, M_node_inv])
                self.one_turn_matrices[node] = M
        return self.one_turn_matrices      
        
    def getRingParametersDict(self, momentum=None, mass=None, node=None):
        """Analyze the one-turn transfer matrix.

        Parameters
        ----------
        momentum : float
            Syncronous particle momentum [GeV / c].
        mass : float
            Particle mass [GeV / c^2].
        node : orbit.lattice.AccNode or str (optional)
            The node to use as the start of the lattice.

        Returns
        -------
        dict
            * 'kin_energy': syncronous particle kinetic energy [GeV]
            * 'mass': particle mass [GeV / c]
            * 'momentum': syncronous particle momentum [GeV / c]
            * 'eigvals' : Eigenvalues of transverse matrix.
            * 'eigvecs' : Eigenvectors of transverse matrix.
            * 'eigtunes' : Transverse tunes computed from the eigenvalues.
            * 'stable' : Whether the transverse motion is stable.
            * 'coupled`: Whether the transverse motion is coupled.
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

        # Transverse parameters
        if node is None:
            M = orbit_matrix_to_numpy(self.oneTurnMatrix)
        else:
            if type(node) is str:
                node = self.getNodeForName(node)
            if not self.one_turn_matrices:
                self.one_turn_matrices = self.getOneTurnMatrices()
            M = self.one_turn_matrices[node]
            
        if self.parameterization == 'CS':
            tmat = transfer_matrix_analysis.CourantSnyder(M)
        if self.parameterization == 'LB':
            tmat = transfer_matrix_analysis.LebedevBogacz(M)
        else:
            raise ValueError('Invalid parameterization.')
            
        params.update(**tmat.params)
        params["M"] = M
        params["eigvals"] = tmat.eigvals
        params["eigvecs"] = tmat.eigvecs
        params["eigtunes"] = tmat.eigtunes
        params["stable"] = tmat.stable
        params["coupled"] = tmat.coupled
    
        # Longitudinal parameters
        ring_length = self.getLength()
        period = ring_length / (beta * speed_of_light)
        params["period"] = period
        params["frequency"] = 1.0 / period     
        # [...]
        
        return params