from spacecharge import DanilovEnvSolver
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNodeBunchTracker
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
        Lattice in which to insert the nodes.
    perveance : float
        Dimensionless beam perveance.
    max_sep : float
        Maximum separation between the nodes
    min_sep : float
        Minimum separation between the nodes.
        
    Returns
    -------
    list[DanilovEnvSolverNode]
        List of the inserted envelope solver nodes.
    """
    lattice.split(max_sep)
    solver = DanilovEnvSolver(perveance)
    solver_nodes = setSC_General_AccNodes(lattice, min_sep, solver, DanilovEnvSolverNode)
    for solver_node in solver_nodes:
        name = ''.join([solver_node.getName(), ':', 'envsolver'])
        solver_node.setName(name)
    lattice.initialize()
    return solver_nodes
        
        
def set_perveance(solver_nodes, perveance):
    """Change the perveance of the solver nodes."""
    calculator = DanilovEnvSolver(perveance)
    for solver_node in solver_nodes:
        solver_node.sc_calculator = calculator