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


def set_env_solver_nodes(lattice, perveance, max_sep=0.01, min_sep=1e-6):
    """Place a set of envelope solver nodes into the lattice.
    
    The method will place the set into the lattice as child nodes of
    the first level accelerator nodes. The nodes will be inserted at
    the beginning of a particular part of the first level AccNode
    element. 
    
    Parameters
    ----------
    lattice : AccLattice object
        The lattice in which to insert the nodes.
    perveance : float
        The dimensionless beam perveance.
    max_sep : float
        The maximum separation between the nodes
    min_sep : float
        The minimum distance between the nodes.
        
    Returns
    -------
    list[EnvSolverNode]
        The list of inserted envelope solver nodes.
    """
    lattice.split(max_sep)
    env_solver_nodes = setSC_General_AccNodes(
        lattice, min_sep, EnvSolver(perveance), EnvSolverNode)
    for env_solver_node in env_solver_nodes:
        name = ''.join([env_solver_node.getName(), 'envsolver'])
        env_solver_node.setName(name)
    lattice.initialize()
    return env_solver_nodes
        
        
def set_perveance(env_solver_nodes, perveance):
    """Change the perveance of the solver nodes."""
    sc_calculator = EnvSolver(perveance)
    for env_solver_node in env_solver_nodes:
        env_solver_node.sc_calculator = sc_calculator
