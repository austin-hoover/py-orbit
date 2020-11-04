import numpy as np

from spacecharge import EnvSolver
from orbit.space_charge.envelope import EnvSolverNode
from orbit.lattice import (
    AccLattice,
    AccNode,
    AccActionsContainer,
    AccNodeBunchTracker
)
from orbit.space_charge.scLatticeModifications import setSC_General_AccNodes
from orbit.utils.consts import classical_proton_radius

def setEnvSolverNodes(lattice, intensity, mass, energy, max_sep, min_sep):
    """Place set of envelope solver nodes into the lattice.
    
    The method will place the set into the lattice as child nodes of
    the first level accelerator nodes. The nodes will be inserted at
    the beginning of a particular part of the first level AccNode
    element.
    
    Parameters
    ----------
    lattice : AccLattice object
        The lattice in which to insert the nodes.
    max_sep : float
        The maximum separation between the nodes
    min_sep : float
        The minimum distance between the nodes.
        
    Returns
    -------
    list[EnvSolverNode]
        The list of inserted envelope solver nodes.
    """
    # Compute perveance
    gamma = 1 + (energy / mass) # Lorentz factor
    beta = np.sqrt(1 - (1 / (gamma**2))) # v/c
    density = intensity / lattice.getLength()
    perveance = (2 * classical_proton_radius * density) / (beta**2 * gamma**3)
    
    lattice.split(max_sep)
    sc_nodes = setSC_General_AccNodes(
        lattice, min_sep, EnvSolver(perveance), EnvSolverNode)
    for sc_node in sc_nodes:
        sc_node.setName(''.join([sc_node.getName(), 'envsolver']))
    lattice.initialize()
    return sc_nodes
